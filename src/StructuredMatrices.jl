module StructuredMatrices

using SIMDPirates, SLEEFPirates, VectorizationBase, LinearAlgebra, PaddedMatrices, LoopVectorization, StackPointers
using PaddedMatrices: AbstractFixedSizeVector, AbstractFixedSizeMatrix
using VectorizationBase: REGISTER_SIZE, REGISTER_COUNT
using ReverseDiffExpressionsBase

import PaddedMatrices: param_type_length, type_length

export addmul!, submul!, inv′, ∂inv′,
        UpperTriangularMatrix, LowerTriangularMatrix, SymmetricMatrixL,
        AutoregressiveMatrixLowerCholeskyInverse, AutoregressiveMatrix,
        MutableLowerTriangularMatrix, MutableUpperTriangularMatrix

@noinline ThrowBoundsError(args...) = throw(BoundsError(args...))
@inline binomial2(n::Integer) = (n*(n-1)) >>> 1

# include("static_ranges.jl")
include("triangular_representation_utilities.jl")
include("symmetric_matrix.jl")
include("triangular_matrix.jl")
include("triangle_inverse.jl")
include("vector_of_triangular_matrix_operations.jl")
include("autoregressive_matrix.jl")
include("block_diagonal.jl")
include("adjoint.jl")
include("decompositions.jl")
include("triangular_equations.jl")
include("ragged_matrix.jl")

@def_stackpointer_fallback rank_update rank_update! reverse_cholesky_grad ∂rank_update
function __init__()
    @add_stackpointer_method rank_update rank_update! reverse_cholesky_grad ∂rank_update
end

end # module
