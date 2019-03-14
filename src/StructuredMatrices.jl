module StructuredMatrices

using SIMDPirates, SLEEFPirates, VectorizationBase, LinearAlgebra, PaddedMatrices
import VectorizationBase: REGISTER_SIZE, REGISTER_COUNT

@noinline ThrowBoundsError(args...) = throw(BoundsError(args...))

include("triangular_representation_utilities.jl")
include("symmetric_matrix.jl")
include("triangular_matrix.jl")
include("triangle_inverse.jl")

end # module
