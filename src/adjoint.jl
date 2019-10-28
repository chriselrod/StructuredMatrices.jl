import ReverseDiffExpressionsBase: RESERVED_INCREMENT_SEED_RESERVED!


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
    sp, a = row_sum_prod_add(sp, seedin, jac.data, seedout)
    sp, a'
end
@inline function RESERVED_MULTIPLY_SEED_RESERVED(
    sp::StackPointer,
    seedin::AbstractLowerTriangularMatrix,
    jac::∂DiagLowerTri∂Diag
)
    sp, a = row_sum_prod(sp, seedin, jac.data)
    sp, a'
end
@inline function RESERVED_INCREMENT_SEED_RESERVED(
    sp::StackPointer,
    seedin::AbstractLowerTriangularMatrix,
    jac::∂DiagLowerTri∂LowerTri,
    seedout::Union{<:AbstractFixedSizeVector,<:AbstractLowerTriangularMatrix}
)
    muladd(sp, jac.data, seedin, seedout)
end
@inline function RESERVED_MULTIPLY_SEED_RESERVED(
    sp::StackPointer,
    seedin::AbstractLowerTriangularMatrix,
    jac::∂DiagLowerTri∂LowerTri
)
    *(sp, jac.data, seedin)
end
@generated function RESERVED_INCREMENT_SEED_RESERVED(
    sp::StackPointer,
    D::LinearAlgebra.Diagonal{T,<:AbstractFixedSizeVector{M,T}},
    A::AbstractMutableDiagMatrix{M,T}
) where {M,T}
    quote
        $(Expr(:meta,:inline))
        LoopVectorization.@vvectorize for m ∈ 1:$M
            A[m] = D[m] + A[m]
        end
        (sp, A)
    end
end
function RESERVED_INCREMENT_SEED_RESERVED(
    sp::StackPointer,
    A::AbstractLowerTriangularMatrix{P,T,L},
    B::AbstractLowerTriangularMatrix{P,T,L}
) where {P,T,L}
    @inbounds @simd ivdep for l in 1:L
        B[l] += A[l]
    end
    sp, B
end
