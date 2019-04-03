import PaddedMatrices: RESERVED_INCREMENT_SEED_RESERVED,
                    RESERVED_DECREMENT_SEED_RESERVED,
                    RESERVED_MULTIPLY_SEED_RESERVED,
                    RESERVED_NMULTIPLY_SEED_RESERVED


struct ∂DiagLowerTri∂Diag{M,T,L <: AbstractLowerTriangularMatrix{M,T}}
    data::L
end
struct ∂DiagLowerTri∂LowerTri{M,T,V <: AbstractFixedSizePaddedVector{M,T}}
    data::LinearAlgebra.Diagonal{T,V}
end

@inline function RESERVED_INCREMENT_SEED_RESERVED(
        seedin::AbstractLowerTriangularMatrix,
        jac::∂DiagLowerTri∂Diag,
        seedout::AbstractFixedSizePaddedVector
    )
    row_sum_prod_add(seedin, jac.data, seedout)'
end
@inline function RESERVED_MULTIPLY_SEED_RESERVED(
        seedin::AbstractLowerTriangularMatrix,
        jac::∂DiagLowerTri∂Diag
    )
    row_sum_prod(seedin, jac.data)'
end


@inline function RESERVED_INCREMENT_SEED_RESERVED(
        seedin::AbstractLowerTriangularMatrix,
        jac::∂DiagLowerTri∂LowerTri,
        seedout::AbstractFixedSizePaddedVector
    )
    muladd(jac.data, seedin, seedout)
end
@inline function RESERVED_MULTIPLY_SEED_RESERVED(
        seedin::AbstractLowerTriangularMatrix,
        jac::∂DiagLowerTri∂LowerTri
    )
    jac.data * seedin
end


@generated function RESERVED_INCREMENT_SEED_RESERVED(D::LinearAlgebra.Diagonal{T,<:AbstractFixedSizePaddedVector{M,T}}, A::AbstractMutableDiagMatrix{M,T}) where {M,T}
    quote
        $(Expr(:meta,:inline))
        @vectorize for m ∈ 1:$M
            A[m] = D[m] + A[m]
        end
        A
    end
end
