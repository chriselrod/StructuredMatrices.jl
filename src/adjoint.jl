import ReverseDiffExpressionsBase: RESERVED_INCREMENT_SEED_RESERVED!

using PaddedMatrices: UninitializedVector

struct ∂DiagLowerTri∂Diag{M,T,L <: AbstractLowerTriangularMatrix{M,T}}
    data::L
end
struct ∂DiagLowerTri∂LowerTri{M,T,V <: AbstractFixedSizeVector{M,T}}
    data::LinearAlgebra.Diagonal{T,V}
end

@inline function RESERVED_INCREMENT_SEED_RESERVED!(
    seedout::AbstractFixedSizeVector,
    jac::∂DiagLowerTri∂Diag,
    seedin::AbstractLowerTriangularMatrix
)
    row_sum_prod_add!(seedout, jac.data, seedin); nothing
end
@inline function RESERVED_INCREMENT_SEED_RESERVED!(
    seedout::UninitializedVector,
    jac::∂DiagLowerTri∂Diag,
    seedin::AbstractLowerTriangularMatrix
)
    row_sum_prod!(seedout, jac.data, seedin); nothing
end
@inline function RESERVED_INCREMENT_SEED_RESERVED!(
    # seedout::Union{<:AbstractMutableFixedSizeVector,<:AbstractLowerTriangularMatrix},
    seedout::AbstractLowerTriangularMatrix,
    jac::∂DiagLowerTri∂LowerTri,
    seedin::AbstractLowerTriangularMatrix
)
    muladd!(seedout, jac.data, seedin); nothing
end
@inline function RESERVED_INCREMENT_SEED_RESERVED!(
    # seedout::Union{<:UnitializedVector,<:UninitializedLowerTriangularMatrix},
    seedout::UninitializedLowerTriangularMatrix,
    jac::∂DiagLowerTri∂LowerTri,
    seedin::AbstractLowerTriangularMatrix
)
    mul!(seedout, jac.data, seedin); nothing
end
# @inline function RESERVED_MULTIPLY_SEED_RESERVED(
    # sp::StackPointer,
    # seedin::AbstractLowerTriangularMatrix,
    # jac::∂DiagLowerTri∂LowerTri
# )
    # *(sp, jac.data, seedin)
# end
@generated function RESERVED_INCREMENT_SEED_RESERVED(
    A::AbstractMutableDiagMatrix{M,T},
    D::LinearAlgebra.Diagonal{T,<:AbstractFixedSizeVector{M,T}}
) where {M,T}
    quote
        $(Expr(:meta,:inline))
        LoopVectorization.@vvectorize $T for m ∈ 1:$M
            A[m] = D[m] + A[m]
        end
    end
end
@generated function RESERVED_INCREMENT_SEED_RESERVED!(
    B::AbstractLowerTriangularMatrix{P,T,L},
    A::AbstractLowerTriangularMatrix{P,T,L}
) where {P,T,L}
    quote
        $(Expr(:meta,:inline))
        @vvectorize $T for l in 1:$L
            B[l] += A[l]
        end
    end
end

