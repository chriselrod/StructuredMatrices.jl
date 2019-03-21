module StructuredMatrices

using SIMDPirates, SLEEFPirates, VectorizationBase, LinearAlgebra, PaddedMatrices
import VectorizationBase: REGISTER_SIZE, REGISTER_COUNT

@noinline ThrowBoundsError(args...) = throw(BoundsError(args...))

include("static_ranges.jl")
include("triangular_representation_utilities.jl")
include("symmetric_matrix.jl")
include("triangular_matrix.jl")
include("triangle_inverse.jl")
include("autoregressive_matrix.jl")

end # module
