module MooncakeSpecialFunctionsExt

using SpecialFunctions, Mooncake
using Base: IEEEFloat

import Mooncake: DefaultCtx, @from_chain_rule, @zero_derivative

@from_chain_rule DefaultCtx Tuple{typeof(airyai),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(airyaix),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(airyaiprime),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(airybi),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(airybiprime),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(besselj0),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(besselj1),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(bessely0),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(bessely1),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(dawson),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(digamma),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(erf),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(erf),IEEEFloat,IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(erfc),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(logerfc),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(erfcinv),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(erfcx),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(logerfcx),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(erfi),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(erfinv),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(gamma),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(invdigamma),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(trigamma),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(polygamma),Integer,IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(beta),IEEEFloat,IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(logbeta),IEEEFloat,IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(logabsgamma),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(loggamma),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(expint),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(expintx),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(expinti),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(sinint),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(cosint),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(ellipk),IEEEFloat}
@from_chain_rule DefaultCtx Tuple{typeof(ellipe),IEEEFloat}

@zero_derivative DefaultCtx Tuple{typeof(logfactorial),Integer}

end
