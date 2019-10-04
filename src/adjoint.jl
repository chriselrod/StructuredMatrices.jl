import PaddedMatrices: RESERVED_INCREMENT_SEED_RESERVED,
                    RESERVED_DECREMENT_SEED_RESERVED,
                    RESERVED_MULTIPLY_SEED_RESERVED,
                    RESERVED_NMULTIPLY_SEED_RESERVED


struct ∂DiagLowerTri∂Diag{M,T,L <: AbstractLowerTriangularMatrix{M,T}}
    data::L
end
struct ∂DiagLowerTri∂LowerTri{M,T,V <: AbstractFixedSizeVector{M,T}}
    data::LinearAlgebra.Diagonal{T,V}
end

@inline function RESERVED_INCREMENT_SEED_RESERVED(
        seedin::AbstractLowerTriangularMatrix,
        jac::∂DiagLowerTri∂Diag,
        seedout::AbstractFixedSizeVector
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
        seedout::Union{<:AbstractFixedSizeVector,<:AbstractLowerTriangularMatrix}
    )
    muladd(jac.data, seedin, seedout)
end
@inline function RESERVED_MULTIPLY_SEED_RESERVED(
        seedin::AbstractLowerTriangularMatrix,
        jac::∂DiagLowerTri∂LowerTri
    )
    jac.data * seedin
end


@generated function RESERVED_INCREMENT_SEED_RESERVED(
    D::LinearAlgebra.Diagonal{T,<:AbstractFixedSizeVector{M,T}},
    A::AbstractMutableDiagMatrix{M,T}
) where {M,T}
    quote
        $(Expr(:meta,:inline))
        # Should this be made to copy!?!?!?
        @vvectorize for m ∈ 1:$M
            A[m] = D[m] + A[m]
        end
        A
    end
end

@inline function RESERVED_INCREMENT_SEED_RESERVED(
    sp::StackPointer,
    seedin::AbstractLowerTriangularMatrix,
    jac::∂DiagLowerTri∂Diag,
    seedout::AbstractFixedSizeVector
)
#    @show seedin
#    @show jac.data
#    @show seedout
    sp, a = row_sum_prod_add(sp, seedin, jac.data, seedout)
#    @show a'
    sp, a'
end
@inline function RESERVED_MULTIPLY_SEED_RESERVED(
    sp::StackPointer,
    seedin::AbstractLowerTriangularMatrix,
    jac::∂DiagLowerTri∂Diag
)
#    @show seedin
#    @show jac.data
    sp, a = row_sum_prod(sp, seedin, jac.data)
#    @show a'
    sp, a'
end
@inline function RESERVED_INCREMENT_SEED_RESERVED(
    sp::StackPointer,
    seedin::AbstractLowerTriangularMatrix,
    jac::∂DiagLowerTri∂LowerTri,
    seedout::Union{<:AbstractFixedSizeVector,<:AbstractLowerTriangularMatrix}
)
#    @show seedin
#    @show jac.data
    muladd(sp, jac.data, seedin, seedout)
#    @show a
#    sp, a
end
@inline function RESERVED_MULTIPLY_SEED_RESERVED(
    sp::StackPointer,
    seedin::AbstractLowerTriangularMatrix,
    jac::∂DiagLowerTri∂LowerTri
)
    *(sp, jac.data, seedin)
end


#=@generated function RESERVED_INCREMENT_SEED_RESERVED(
    D::LinearAlgebra.Diagonal{T,<:AbstractFixedSizeVector{M,T}},
    A::AbstractMutableDiagMatrix{M,T}
) where {M,T}
    quote
        $(Expr(:meta,:inline))
        @vectorize for m ∈ 1:$M
            A[m] = D[m] + A[m]
        end
        A
    end
end=#
@generated function RESERVED_INCREMENT_SEED_RESERVED(
    sp::StackPointer,
    D::LinearAlgebra.Diagonal{T,<:AbstractFixedSizeVector{M,T}},
    A::AbstractMutableDiagMatrix{M,T}
) where {M,T}
    quote
        $(Expr(:meta,:inline))
        # Should this be made to copy!?!?!?
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
    @inbounds @simd for l in 1:L
        B[l] += A[l]
    end
    sp, B
end
