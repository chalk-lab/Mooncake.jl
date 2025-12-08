module MooncakeLogExpFunctionsExt

using LogExpFunctions, Mooncake
using Base: IEEEFloat

import Mooncake: DefaultCtx, @from_chainrules

@from_chainrules DefaultCtx Tuple{typeof(xlogx),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(xlogy),IEEEFloat,IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(xlog1py),IEEEFloat,IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(xexpx),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(xexpy),IEEEFloat,IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logistic),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logit),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logcosh),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logabssinh),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(log1psq),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(log1pexp),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(log1mexp),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(log2mexp),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logexpm1),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(log1pmx),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logmxp1),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logaddexp),IEEEFloat,IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logsubexp),IEEEFloat,IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(cloglog),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(cexpexp),IEEEFloat}

end
