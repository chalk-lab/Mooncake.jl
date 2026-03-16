module MooncakecuDNNExt

using CUDA
using cuDNN
using Mooncake
import Mooncake: tangent_type, NoTangent

# All cuDNN types — @cenum integers, opaque C handle structs, plain C structs, and their
# Ptr{T} aliases — are non-differentiable C library internals.
# This extension is separate from MooncakeCUDAExt so that it only loads when the cuDNN
# library is available, avoiding import errors on systems where CUDA is present but cuDNN
# is not installed.
#
# Three categories handled programmatically:
#   1. @cenum types     — isprimitivetype(T), parentmodule(T) == cuDNN
#   2. Opaque C structs — isstructtype(T),    parentmodule(T) == cuDNN (e.g. cudnnContext)
#   3. Ptr{T} handles   — T <: Ptr{S}        where parentmodule(S) == cuDNN
#                         (e.g. cudnnHandle_t = Ptr{cudnnContext})
let _seen = Set{DataType}()
    function _register(T::DataType)
        T in _seen && return nothing
        push!(_seen, T)
        already = try
            tangent_type(T) === NoTangent
        catch
            false
        end
        already || @eval tangent_type(::Type{$T}) = NoTangent
    end

    for _nm in names(cuDNN; all=true)
        _val = try
            getfield(cuDNN, _nm)
        catch
            nothing
        end
        _val isa DataType || continue

        # Category 1 & 2: primitive (@cenum) or struct types defined in cuDNN
        if parentmodule(_val) == cuDNN
            _register(_val)
        end

        # Category 3: Ptr{S} where S is defined in cuDNN (the _t handle aliases)
        if _val <: Ptr
            _S = _val.parameters[1]
            if _S isa DataType && parentmodule(_S) == cuDNN
                _register(_val)
            end
        end
    end
end

end # module
