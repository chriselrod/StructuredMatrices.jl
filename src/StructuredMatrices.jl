module StructuredMatrices

using SIMDPirates, SLEEFPirates, VectorizationBase, LinearAlgebra, PaddedMatrices, LoopVectorization
using PaddedMatrices: AbstractFixedSizePaddedVector, AbstractFixedSizePaddedMatrix
using VectorizationBase: REGISTER_SIZE, REGISTER_COUNT

export addmul!, submul!, inv′, ∂inv′,
        UpperTriangularMatrix, LowerTriangularMatrix, SymmetricMatrixL,
        AutoregressiveMatrixLowerCholeskyInverse

@noinline ThrowBoundsError(args...) = throw(BoundsError(args...))
@inline binomial2(n) = (n*(n-1)) >> 1

include("static_ranges.jl")
include("triangular_representation_utilities.jl")
include("symmetric_matrix.jl")
include("triangular_matrix.jl")
include("triangle_inverse.jl")
include("vector_of_triangular_matrix_operations.jl")
include("autoregressive_matrix.jl")
include("block_diagonal.jl")
include("adjoint.jl")

end # module
