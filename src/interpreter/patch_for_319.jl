# The purpose of the code in this file is to provide a work-around for the Julia compiler
# bug discussed in https://github.com/chalk-lab/Mooncake.jl/issues/319 . You do not need to
# understand it in order to understand Mooncake. I (Will) would recommend against spending
# any time at all reading / understanding this file unless you are actively working on this
# issue, and find it useful.
#
# As soon as patch versions of Julia (both LTS and Release) are made which contain a fix for
# the bug underlying the above issue, this code can and should be removed, and the minimum
# versions of Julia which Mooncake supports bumped.
#
# The only place in which this code seeps into Mooncake.jl code is in Mooncake.optimise_ir!,
# located in src/interpreter/ir_utils.jl . In particular, we replace the `local_interp`
# variable with the `AbstractInterpreter`.
#
# The work around:
# We define a new `AbstractInterpreter` which wraps around the `Compiler.NativeInterpreter`.
# This makes it possible to add methods to various functions in `Compiler`, thereby enabling
# us to insert the bug fixes.

struct BugPatchInterpreter <: CC.AbstractInterpreter
    interp::CC.NativeInterpreter
    BugPatchInterpreter() = new(CC.NativeInterpreter())
end

CC.InferenceParams(ip::BugPatchInterpreter) = CC.InferenceParams(ip.interp)
CC.OptimizationParams(ip::BugPatchInterpreter) = CC.OptimizationParams(ip.interp)
CC.get_inference_cache(ip::BugPatchInterpreter) = CC.get_inference_cache(ip.interp)
CC.code_cache(ip::BugPatchInterpreter) = CC.code_cache(ip.interp)
function CC.get(wvc::CC.WorldView{BugPatchInterpreter}, mi::Core.MethodInstance, default)
    return get(wvc.cache.dict, mi, default)
end
function CC.getindex(wvc::CC.WorldView{BugPatchInterpreter}, mi::Core.MethodInstance)
    return getindex(wvc.cache.dict, mi)
end
function CC.haskey(wvc::CC.WorldView{BugPatchInterpreter}, mi::Core.MethodInstance)
    return haskey(wvc.cache.dict, mi)
end
function CC.setindex!(
    wvc::CC.WorldView{BugPatchInterpreter}, ci::Core.CodeInstance, mi::Core.MethodInstance
)
    return setindex!(wvc.cache.dict, ci, mi)
end
CC.method_table(ip::BugPatchInterpreter) = CC.method_table(ip.interp)

@static if VERSION < v"1.11.0"
    CC.get_world_counter(ip::BugPatchInterpreter) = CC.get_world_counter(ip.interp)
else
    CC.get_inference_world(ip::BugPatchInterpreter) = CC.get_inference_world(ip.interp)
    CC.cache_owner(ip::BugPatchInterpreter) = CC.cache_owner(ip.interp)
end

# You can't write for n in thing_from_compiler unless `Base.iterate(thing_from_compiler)`
# is implemented. Sadly, it's usually the case that `Compiler.iterate(thing_from_compiler)`
# is implemented, but not the function from `Base.` This is convenience functionality to
# ensure that we don't have to write everything out manually each time iteration over
# something from the Compiler is encountered.
function core_iterate(f, iterator)
    it = CC.iterate(iterator)
    while it !== nothing
        val, state = it
        f(val)
        it = CC.iterate(iterator, state)
    end
end

@static if VERSION >= v"1.11"

    # Original contains bugs. Apply patch from Jules Merck.
    function patched_populate_def_use_map!(
        tpdum::CC.TwoPhaseDefUseMap, scanner::CC.BBScanner
    )
        CC.scan!(scanner, false) do inst::CC.Instruction, lstmt::Int, bb::Int
            core_iterate(CC.userefs(inst[:stmt])) do ur # replace inst with inst[:stmt]
                val = CC.getindex(ur)
                if isa(val, SSAValue)
                    CC.push!(CC.getindex(tpdum, val.id), inst.idx)
                end
            end
            return true
        end
    end

    # Calls populate_def_use_map! -- see above.
    function ir_abstract_constant_propagation(
        interp::BugPatchInterpreter,
        irsv::CC.IRInterpretationState;
        externally_refined::Union{Nothing,BitSet}=nothing,
    )
        (; ir, tpdum, ssa_refined) = irsv

        @assert CC.isempty(ir.new_nodes) "IRCode should be compacted before irinterp"

        all_rets = Int[]
        scanner = CC.BBScanner(ir)

        function check_ret!(@nospecialize(stmt), idx::Int)
            return isa(stmt, ReturnNode) && isdefined(stmt, :val) && push!(all_rets, idx)
        end

        # Fast path: Scan both use counts and refinement in one single pass of
        #            of the instructions. In the absence of backedges, this will
        #            converge.
        completed_scan =
            CC.scan!(scanner, true) do inst::CC.Instruction, lstmt::Int, bb::Int
                idx = inst.idx
                irsv.curridx = idx
                stmt = inst[:stmt]
                typ = inst[:type]
                flag = inst[:flag]
                any_refined = false
                if CC.has_flag(flag, CC.IR_FLAG_REFINED)
                    any_refined = true
                    CC.sub_flag!(inst, CC.IR_FLAG_REFINED)
                elseif CC.is_all_const_call(stmt, interp, irsv)
                    # force reinference on calls with all constant arguments
                    any_refined = true
                end
                core_iterate(CC.userefs(stmt)) do ur
                    val = CC.getindex(ur)
                    if isa(val, Argument)
                        any_refined |= irsv.argtypes_refined[val.n]
                    elseif isa(val, SSAValue)
                        any_refined |= CC.in(val.id, ssa_refined)
                        CC.count!(tpdum, val)
                    end
                end
                if isa(stmt, CC.PhiNode) && CC.in(idx, ssa_refined)
                    any_refined = true
                    CC.delete!(ssa_refined, idx)
                end
                check_ret!(stmt, idx)
                @static if VERSION > v"1.12-"
                    is_terminator_or_phi = (
                        isa(stmt, PhiNode) || stmt === nothing || CC.isterminator(stmt)
                    )
                else
                    is_terminator_or_phi = (isa(stmt, PhiNode) || CC.isterminator(stmt))
                end
                if typ === CC.Bottom && !(idx == lstmt && is_terminator_or_phi)
                    return true
                end
                if (
                    any_refined && CC.reprocess_instruction!(interp, inst, idx, bb, irsv)
                ) || (externally_refined !== nothing && idx in externally_refined)
                    CC.push!(ssa_refined, idx)
                    stmt = inst[:stmt]
                    typ = inst[:type]
                end
                if typ === CC.Bottom && !is_terminator_or_phi
                    CC.kill_terminator_edges!(irsv, lstmt, bb)
                    if idx != lstmt
                        for idx2 in ((idx + 1):(lstmt - 1))
                            CC.setindex!(ir, nothing, SSAValue(idx2))
                        end
                        CC.setindex!(ir[SSAValue(lstmt)], ReturnNode(), :stmt)
                    end
                    return false
                end
                return true
            end

        if !completed_scan
            # Slow path
            stmt_ip = CC.BitSetBoundedMinPrioritySet(length(ir.stmts))

            # Slow Path Phase 1.A: Complete use scanning
            CC.scan!(scanner, false) do inst::CC.Instruction, lstmt::Int, bb::Int
                idx = inst.idx
                irsv.curridx = idx
                stmt = inst[:stmt]
                flag = inst[:flag]
                if CC.has_flag(flag, CC.IR_FLAG_REFINED)
                    CC.sub_flag!(inst, CC.IR_FLAG_REFINED)
                    CC.push!(stmt_ip, idx)
                end
                check_ret!(stmt, idx)
                core_iterate(CC.userefs(stmt)) do ur
                    val = CC.getindex(ur)
                    if isa(val, Argument)
                        if irsv.argtypes_refined[val.n]
                            CC.push!(stmt_ip, idx)
                        end
                    elseif isa(val, SSAValue)
                        CC.count!(tpdum, val)
                    end
                end
                return true
            end

            # Slow Path Phase 1.B: Assemble def-use map
            CC.complete!(tpdum)
            CC.push!(scanner.bb_ip, 1)
            patched_populate_def_use_map!(tpdum, scanner)

            # Slow Path Phase 2: Use def-use map to converge cycles.
            # TODO: It would be possible to return to the fast path after converging
            #       each cycle, but that's somewhat complicated.
            core_iterate(ssa_refined) do val
                # for use in CC.getindex(tpdum, val)
                core_iterate(CC.getindex(tpdum, val)) do use
                    if !CC.in(use, ssa_refined)
                        CC.push!(stmt_ip, use)
                    end
                end
            end
            while !CC.isempty(stmt_ip)
                idx = CC.popfirst!(stmt_ip)
                irsv.curridx = idx
                inst = ir[SSAValue(idx)]
                if CC.reprocess_instruction!(interp, inst, idx, nothing, irsv)
                    CC.append!(stmt_ip, CC.getindex(tpdum, idx))
                end
            end
        end

        ultimate_rt = CC.Bottom
        for idx in all_rets
            bb = CC.block_for_inst(ir.cfg, idx)
            if bb != 1 && length(ir.cfg.blocks[bb].preds) == 0
                # Could have discovered this block is dead after the initial scan
                continue
            end
            inst = ir[SSAValue(idx)][:stmt]::ReturnNode
            rt = CC.argextype(inst.val, ir)
            ultimate_rt = CC.tmerge(CC.typeinf_lattice(interp), ultimate_rt, rt)
        end

        nothrow = noub = true
        for idx in 1:length(ir.stmts)
            if ir[SSAValue(idx)][:stmt] === nothing
                # skip `nothing` statement, which might be inserted as a dummy node,
                # e.g. by `finish_current_bb!` without explicitly marking it as `:nothrow`
                continue
            end
            flag = ir[SSAValue(idx)][:flag]
            nothrow &= CC.has_flag(flag, CC.IR_FLAG_NOTHROW)
            noub &= CC.has_flag(flag, CC.IR_FLAG_NOUB)
            (nothrow | noub) || break
        end

        @static if VERSION ≥ v"1.12-"
            if irsv.frameid != 0
                callstack = irsv.callstack::Vector{CC.AbsIntState}
                @assert callstack[end] === irsv && length(callstack) == irsv.frameid
                pop!(callstack)
            end
        else
            if CC.last(irsv.valid_worlds) >= CC.get_world_counter()
                # if we aren't cached, we don't need this edge
                # but our caller might, so let's just make it anyways
                CC.store_backedges(CC.frame_instance(irsv), irsv.edges)
            end
        end

        return Pair{Any,Tuple{Bool,Bool}}(
            CC.maybe_singleton_const(ultimate_rt), (nothrow, noub)
        )
    end

    @static if VERSION ≥ v"1.12-"
        CC.ir_abstract_constant_propagation(interp::BugPatchInterpreter, irsv::CC.IRInterpretationState; externally_refined::Union{Nothing,BitSet}=nothing) = ir_abstract_constant_propagation(
            interp, irsv; externally_refined
        )
    else
        CC._ir_abstract_constant_propagation(interp::BugPatchInterpreter, irsv::CC.IRInterpretationState; externally_refined::Union{Nothing,BitSet}=nothing) = ir_abstract_constant_propagation(
            interp, irsv; externally_refined
        )
    end

    struct ScanStmtPatch
        sv::CC.PostOptAnalysisState
    end

    # Original contains bugs.
    function ((; sv)::ScanStmtPatch)(inst::CC.Instruction, lstmt::Int, bb::Int)
        stmt = inst[:stmt]

        if isa(stmt, CC.EnterNode)
            # try/catch not yet modeled
            CC.give_up_refinements!(sv)
            return true # don't bail out early -- replaces `nothing` with `true` 
        end

        CC.scan_non_dataflow_flags!(inst, sv)

        stmt_inconsistent = patched_scan_inconsistency!(inst, sv)

        if stmt_inconsistent
            if !CC.has_flag(inst[:flag], CC.IR_FLAG_NOTHROW)
                # Taint :consistent if this statement may raise since :consistent requires
                # consistent termination. TODO: Separate :consistent_return and :consistent_termination from :consistent.
                sv.all_retpaths_consistent = false
            end
            if inst.idx == lstmt
                if isa(stmt, ReturnNode) && isdefined(stmt, :val)
                    sv.all_retpaths_consistent = false
                elseif isa(stmt, GotoIfNot)
                    # Conditional Branch with inconsistent condition.
                    # If we do not know this function terminates, taint consistency, now,
                    # :consistent requires consistent termination. TODO: Just look at the
                    # inconsistent region.
                    if !sv.result.ipo_effects.terminates
                        sv.all_retpaths_consistent = false
                    elseif CC.visit_conditional_successors(
                        sv.lazypostdomtree, sv.ir, bb
                    ) do succ::Int
                        return CC.any_stmt_may_throw(sv.ir, succ)
                    end
                        # check if this `GotoIfNot` leads to conditional throws, which taints consistency
                        sv.all_retpaths_consistent = false
                    else
                        (; cfg, domtree) = CC.get!(sv.lazyagdomtree)
                        for succ in CC.iterated_dominance_frontier(
                            cfg,
                            CC.BlockLiveness(sv.ir.cfg.blocks[bb].succs, nothing),
                            domtree,
                        )
                            if succ == CC.length(cfg.blocks)
                                # Phi node in the virtual exit -> We have a conditional
                                # return. TODO: Check if all the retvals are egal.
                                sv.all_retpaths_consistent = false
                            else
                                CC.visit_bb_phis!(sv.ir, succ) do phiidx::Int
                                    CC.push!(sv.inconsistent, phiidx)
                                end
                            end
                        end
                    end
                end
            end
        end

        # Do not bail out early, as this can cause tpdum counts to be off.
        # # bail out early if there are no possibilities to refine the effects
        # if !any_refinable(sv)
        #     return nothing
        # end

        return true
    end

    # Original contains bug.
    function patched_scan_inconsistency!(inst::CC.Instruction, sv::CC.PostOptAnalysisState)
        flag = inst[:flag]
        stmt_inconsistent = !CC.has_flag(flag, CC.IR_FLAG_CONSISTENT)
        stmt = inst[:stmt]
        # Special case: For `getfield` and memory operations, we allow inconsistency of the :boundscheck argument
        (; inconsistent, tpdum) = sv
        if CC.iscall_with_boundscheck(stmt, sv)
            for i in 1:length(stmt.args) # explore all args -- don't assume boundscheck is not an SSA
                val = stmt.args[i]
                if isa(val, SSAValue)
                    stmt_inconsistent |= CC.in(val.id, inconsistent)
                    CC.count!(tpdum, val)
                end
            end
        else
            core_iterate(CC.userefs(stmt)) do ur
                val = CC.getindex(ur)
                if isa(val, SSAValue)
                    stmt_inconsistent |= CC.in(val.id, inconsistent)
                    CC.count!(tpdum, val)
                end
            end
        end
        stmt_inconsistent && CC.push!(inconsistent, inst.idx)
        return stmt_inconsistent
    end

    # Calls check_inconsistentcy! -- see below.
    function CC.ipo_dataflow_analysis!(
        interp::BugPatchInterpreter, ir::CC.IRCode, result::CC.InferenceResult
    )
        if !CC.is_ipo_dataflow_analysis_profitable(result.ipo_effects)
            return false
        end

        @assert CC.isempty(ir.new_nodes) "IRCode should be compacted before post-opt analysis"

        sv = CC.PostOptAnalysisState(result, ir)
        scanner = CC.BBScanner(ir)

        completed_scan = CC.scan!(ScanStmtPatch(sv), scanner, true)

        if !completed_scan
            if sv.all_retpaths_consistent
                patched_check_inconsistentcy!(sv, scanner)
            else
                # No longer any dataflow concerns, just scan the flags
                CC.scan!(scanner, false) do inst::CC.Instruction, lstmt::Int, bb::Int
                    CC.scan_non_dataflow_flags!(inst, sv)
                    # bail out early if there are no possibilities to refine the effects
                    if !CC.any_refinable(sv)
                        return nothing
                    end
                    return true
                end
            end
        end

        return CC.refine_effects!(interp, sv)
    end

    # Calls populate_def_use_map! -- see above.
    function patched_check_inconsistentcy!(
        sv::CC.PostOptAnalysisState, scanner::CC.BBScanner
    )
        (; ir, inconsistent, tpdum) = sv

        CC.scan!(ScanStmtPatch(sv), scanner, false)
        CC.complete!(tpdum)
        CC.push!(scanner.bb_ip, 1)
        patched_populate_def_use_map!(tpdum, scanner)

        stmt_ip = CC.BitSetBoundedMinPrioritySet(length(ir.stmts))
        core_iterate(inconsistent) do def
            core_iterate(CC.getindex(tpdum, def)) do use
                if !CC.in(use, inconsistent)
                    CC.push!(inconsistent, use)
                    CC.append!(stmt_ip, CC.getindex(tpdum, use))
                end
            end
        end
        lazydomtree = CC.LazyDomtree(ir)
        while !CC.isempty(stmt_ip)
            idx = CC.popfirst!(stmt_ip)
            inst = ir[SSAValue(idx)]
            stmt = inst[:stmt]
            if CC.iscall_with_boundscheck(stmt, sv)
                any_non_boundscheck_inconsistent = false
                for i in 1:(length(stmt.args) - 1)
                    val = stmt.args[i]
                    if isa(val, SSAValue)
                        any_non_boundscheck_inconsistent |= val.id in inconsistent
                        any_non_boundscheck_inconsistent && break
                    end
                end
                any_non_boundscheck_inconsistent || continue
            elseif isa(stmt, ReturnNode)
                sv.all_retpaths_consistent = false
            elseif isa(stmt, GotoIfNot)
                bb = CC.block_for_inst(ir, idx)
                cfg = ir.cfg
                blockliveness = CC.BlockLiveness(cfg.blocks[bb].succs, nothing)
                for succ in
                    CC.iterated_dominance_frontier(cfg, blockliveness, get!(lazydomtree))
                    CC.visit_bb_phis!(ir, succ) do phiidx::Int
                        push!(inconsistent, phiidx)
                        push!(stmt_ip, phiidx)
                    end
                end
            end
            sv.all_retpaths_consistent || break
            CC.append!(inconsistent, tpdum[idx])
            CC.append!(stmt_ip, tpdum[idx])
        end
    end
end
