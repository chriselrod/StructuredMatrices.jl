
abstract type AbstractAutoregressiveMatrix{T,V} <: AbstractMatrix{T} end
abstract type AbstractAutoregressiveMatrixAdjoint{T,V} <: AbstractMatrix{T} end

abstract type AbstractInvervalSpacing end
abstract type AbstractEvenSpacing <: AbstractInvervalSpacing end
abstract type AbstractUnevenSpacing <: AbstractInvervalSpacing end

struct EvenSpacing{T1,T2} <: AbstractEvenSpacing
    ρᵗ::T1
    invOmρ²ᵗ::T2
    rinvOmρ²ᵗ::T2
end
# We don't precompute, because storage cost is high.. 
struct UnevenSpacing <: AbstractUnevenSpacing end
struct CachedUnevenSpacing{T} <: AbstractUnevenSpacing
    ρᵗ::T
    invOmρ²ᵗ::T
    rinvOmρ²ᵗ::T
end



"""
ρ is the autoregressive parameter.
τ is either:
1. A range of times, eg 1:10, -5:16, or 0.12:0.23:15. Use this for regularly spaced time intervals.
2. A vector of time deltas, eg [0.2, 0.15, 0.35, 0.3] would correspond to times [0.0, 0.2, 0.35, 0.7, 1.0].
    Use this for irregularly spaced time intervals. Note that the size of the matrix is 1 greater than the
    length of these deltas (eg, length(time deltas) == 4 in this example, corresponding to 5 times, or a 5x5 matrix).

"""
struct AutoregressiveMatrixLowerCholeskyInverse{T,V <: AbstractVector, S <: AbstractInvervalSpacing} <: AbstractAutoregressiveMatrix{T,V}
    ρ::T
    τ::V
    spacing::S
end
# struct AutoregressiveMatrixLowerCholeskyInverseAdjoint{T,V <: AbstractVector} <: AbstractAutoregressiveMatrixAdjoint{T,V}
#     ρ::T
#     ρᵗ::TV
#     invOmρ²ᵗ::TV
#     rinvOmρ²ᵗ::TV
#     τ::V
# end
struct AutoregressiveMatrix{T,V <: AbstractVector, S <: AbstractInvervalSpacing} <: AbstractAutoregressiveMatrix{T,V}
    ρ::T
    τ::V
    spacing::S
end
# struct AutoregressiveMatrixAdjoint{T,V <: AbstractVector} <: AbstractAutoregressiveMatrixAdjoint{T,V}
#     ρ::T
#     ρᵗ::TV
#     invOmρ²ᵗ::TV
#     rinvOmρ²ᵗ::TV
#     τ::V
# end

function AutoregressiveMatrixLowerCholeskyInverse(ρ::T, τ::AbstractUnitRange) where {T}
    invOmρ²ᵗ = 1 / ( 1 - ρ*ρ )
    AutoregressiveMatrixLowerCholeskyInverse(
        ρ, τ, EvenSpacing(nothing, invOmρ²ᵗ, sqrt(invOmρ²ᵗ))
    )
end
function AutoregressiveMatrixLowerCholeskyInverse(ρ::T, τ::AbstractRange) where {T}
    ρᵗ = copysign(abs(ρ)^(step(τ)), ρ)
    invOmρ²ᵗ = 1 / ( 1 - ρᵗ*ρᵗ )
    AutoregressiveMatrixLowerCholeskyInverse(
        ρ, τ, EvenSpacing(ρᵗ, invOmρ²ᵗ, sqrt(invOmρ²ᵗ))
    )
end
# @generated function AutoregressiveMatrixLowerCholeskyInverse(ρ::T, τ::AbstractFixedSizePaddedVector{M,T,L}) where {M,L,T}
#     quote
#         ρᵗ = MutableFixedSizePaddedVector{L,T}(undef)
#         invOmρ²ᵗ = = MutableFixedSizePaddedVector{L,T}(undef)
#         rinvOmρ²ᵗ = = MutableFixedSizePaddedVector{L,T}(undef)
#         @vectorize $T for i ∈ 1:$L
#             ρᵗ[i] = SIMDPirates.vcopysign(SIMDPirates.vpow(SIMDPirates.vabs(ρ),  τ[i] ), ρ[i])
#             vinvOmρ²ᵗ = 1 / (1 - ρᵗ[i]*ρᵗ[i])
#             invOmρ²ᵗ[i] = vinvOmρ²ᵗ
#             rinvOmρ²ᵗ[i] = sqrt(vinvOmρ²ᵗ)
#         end
#         AutoregressiveMatrixLowerCholeskyInverse(
#             ρ, ConstantFixedSizePaddedVector(ρᵗ), ConstantFixedSizePaddedVector(invOmρ²ᵗ), ConstantFixedSizePaddedVector(rinvOmρ²ᵗ), τ
#         )
#     end
# end
function AutoregressiveMatrixLowerCholeskyInverse(ρ::T, τ::AbstractVector) where {T}
    AutoregressiveMatrixLowerCholeskyInverse(
        ρ, τ, UnevenSpacing()
    )
end
cache(A::AbstractAutoregressiveMatrix) = A
@generated function cache(A::AbstractAutoregressiveMatrix{T,V,UnevenSpacing}) where {M,T,L,V <: AbstractFixedSizePaddedVector{M,T,L}}
    quote
        ρᵗ = MutableFixedSizePaddedVector{$M,$T}(undef)
        invOmρ²ᵗ = = MutableFixedSizePaddedVector{$M,$T}(undef)
        rinvOmρ²ᵗ = = MutableFixedSizePaddedVector{$M,$T}(undef)
        @vectorize $T for i ∈ 1:$M
            ρᵗ[i] = SIMDPirates.vcopysign(SIMDPirates.vpow(SIMDPirates.vabs(ρ),  τ[i] ), ρ[i])
            vinvOmρ²ᵗ = 1 / (1 - ρᵗ[i]*ρᵗ[i])
            invOmρ²ᵗ[i] = vinvOmρ²ᵗ
            rinvOmρ²ᵗ[i] = sqrt(vinvOmρ²ᵗ)
        end
        AutoregressiveMatrixLowerCholeskyInverse(
            A.ρ, A.τ, CachedUnevenSpacing(ConstantFixedSizePaddedVector(ρᵗ), ConstantFixedSizePaddedVector(invOmρ²ᵗ), ConstantFixedSizePaddedVector(rinvOmρ²ᵗ))
        )
    end
end



# function AutoregressiveMatrix(ρ::AutoregressiveParameter{T}, τ::AbstractRange) where {T}
#     absρ = abs(ρ)

# end
# function AutoregressiveMatrix(ρ::AutoregressiveParameter{T}, τ::AbstractUnitRange) where {T}
#     absρ = abs(ρ)
#     ptm1 = copysign(absρ, ρ)

# end
# function AutoregressiveMatrix(ρ::AutoregressiveParameter{T}, τ::AbstractVector) where {T}
#     AutoregressiveMatrix(ρ, τ)
# end


@inline Base.size(AR::AbstractAutoregressiveMatrix{T,V}) where {T,V<:AbstractRange} = (t = length(AR.τ); (t,t))
@inline Base.size(AR::AbstractAutoregressiveMatrix) = (t = length(AR.τ)+1; (t,t))
# @inline Base.size(AR::AbstractAutoregressiveMatrixAdjoint{T,V}) where {T,V<:AbstractRange} = (t = length(AR.τ); (t,t))
# @inline Base.size(AR::AbstractAutoregressiveMatrixAdjoint) = (t = length(AR.τ)+1; (t,t))
function Base.getindex(AR::AutoregressiveMatrixLowerCholeskyInverse{T,V,S}, i, j) where {T,V,S <: AbsractEvenSpacing}
    @boundscheck max(i,j) > length(ar.τ) || PaddedMatrices.ThrowBoundsError("max(i,j) = $(max(i,j)) > length(AR.τ) = $(length(AR.τ)).")
    if i == j
        if i == 1
            return one(T)
        else
            return AR.spacing.rinvOmρ²ᵗ
        end
    end
    δij = i - j
    δij != 1 && return zero(T)
    @fastmath - AR.ρ * AR.spacing.rinvOmρ²ᵗ
end

function Base.getindex(AR::AutoregressiveMatrixLowerCholeskyInverse{T,V,S}, i, j) where {T,V,S <: AbsractEvenSpacing}
    @boundscheck max(i,j) > length(ar.τ) || PaddedMatrices.ThrowBoundsError("max(i,j) = $(max(i,j)) > length(AR.τ) = $(length(AR.τ)).")
    if i == j
        if i == 1
            return one(T)
        else
            return AR.spacing.rinvOmρ²ᵗ
        end
    end
    δij = i - j
    δij == 1 ? - AR.spacing.ρᵗ * AR.spacing.rinvOmρ²ᵗ : zero(T)
end

function Base.getindex(AR::AutoregressiveMatrixLowerCholeskyInverse{T,V,S}, i, j) where {T,V,S <: UnevenSpacing}
    @boundscheck max(i,j) > length(ar.τ) || PaddedMatrices.ThrowBoundsError("max(i,j) = $(max(i,j)) > length(AR.τ) = $(length(AR.τ)).")
    if i == j
        if i == 1
            return one(T)
        else
            rhot = AR.ρ ^ (ar.τ[i-1])
            return 1 / sqrt(1 - rhot * rhot)
        end
    end
    δij = i - j
    δij == 1 || return zero(T)
    rhot = AR.ρ ^ (ar.τ[j])
    - rhot / sqrt(1 - rhot*rhot)
end
function Base.getindex(AR::AutoregressiveMatrixLowerCholeskyInverse{T,V,S}, i, j) where {T,V,S <: CachedUnevenSpacing}
    @boundscheck max(i,j) > length(ar.τ) || PaddedMatrices.ThrowBoundsError("max(i,j) = $(max(i,j)) > length(AR.τ) = $(length(AR.τ)).")
    if i == j
        if i == 1
            return one(T)
        else
            return AR.spacing.rinvOmρ²ᵗ[i-1]
        end
    end
    δij = i - j
    δij == 1 ? - AR.spacing.ρᵗ[j] * AR.spacing.rinvOmρ²ᵗ[j] : zero(T)
end


@inline function Base.getindex(AR::AutoregressiveMatrix{T,V,S}, i, j) where {T,V,S <: AbsractEvenSpacing}
    @boundscheck max(i,j) > length(ar.τ) || PaddedMatrices.ThrowBoundsError("max(i,j) = $(max(i,j)) > length(AR.τ) = $(length(AR.τ)).")
    SIMDPirates.vcopysign(SLEEFPirates.power(AR.ρ, SIMDPirates.abs(i - j), AR.ρ)
end

@inline function Base.getindex(AR::AutoregressiveMatrix{T,V,S}, i, j) where {T,V,S <: AbsractEvenSpacing}
    @boundscheck max(i,j) > length(ar.τ) || PaddedMatrices.ThrowBoundsError("max(i,j) = $(max(i,j)) > length(AR.τ) = $(length(AR.τ)).")
    SLEEFPirates.power(AR.ρ, step(AR.τ) * SIMDPirates.abs(i - j))
end

# function Base.getindex(AR::AutoregressiveMatrix{T,V}, i, j) where {T,V <: AbstractVector}
#     @boundscheck max(i,j) > length(ar.τ) || PaddedMatrices.ThrowBoundsError("max(i,j) = $(max(i,j)) > length(AR.τ) = $(length(AR.τ)).")
#     i == j && return one(T)
#     δt = zero(T)
#     i, j = minmax(i, j)
#     for k ∈ i:j-1
#         δt += AR.τ[k]
#     end
#     ρᵗ = abs(AR.ρ)^δt
#     iseven(j - i) ? ρᵗ : copysign(ρᵗ, AR.ρ)
# end


@generated function Base.:*(
            A::AbstractAutoregressiveMatrix{T,R,S},
            B::AbstractFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}}
        ) where {M,N,W,T,R,S}

    # register_count = VectorizationBase.REGISTER_COUNT

    V = NTuple{W,Core.VecElement{T}}
    # if R <: AbstractUnitRange
    #     loop_body = :( $vinvOmρ² * ( B[ 1+m + M*n ] - A.ρ * B[ m + M*n ] ) )
    if S <: AbstractEvenSpacing
        loop_body = :( AB[ 1+m + $M*n ] = SIMDPirates.evmul( vinvOmρ², SIMDPirates.vfnmadd( vρ, B[ m + $M*n ], B[ 1+m + $M*n ] ) ) )
    elseif S == UnevenSpacing
        loop_body = quote
            ρᵗ = A.ρ ^ (A.τ[m])
            rinvOmρ²ᵗ = 1 / sqrt(1 - ρᵗ*ρᵗ)
            AB[ 1+m + $M*n ] = SIMDPirates.evmul( SIMDPirates.vbroadcast($V, rinvOmρ²ᵗ), SIMDPirates.vfnmadd( SIMDPirates.vbroadcast($V, ρᵗ), B[ m + M*n ], B[ 1+m + M*n ] ) )
        end
    elseif S == CachedUnevenSpacing
        loop_body = quote
            vinvOmρ² = SIMDPirates.vbroadcast($V, A.spacing.rinvOmρ²ᵗ[m])
            vρᵗ = SIMDPirates.vbroadcast($V, A.ρᵗ[m])
            AB[ 1+m + $M*n ] = SIMDPirates.evmul( vinvOmρ², SIMDPirates.vfnmadd( vρᵗ, B[ m + M*n ], B[ 1+m + M*n ] ) )
        end
    end

    q = quote
        AB = MutableFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}}(undef)
        triind = N
        @inbounds for n ∈ 0:N-1
            AB[ 1 + M*n ] = B[ 1 + M*n ]
            for m ∈ 1:M-1
                $loop_body
            end
        end
        ConstantFixedSizePaddedMatrix(AB)
    end
    if R <: AbstractUnitRange
        pushfirst!(q.args, quote
            vinvOmρ² = SIMDPirates.vbroadcast($V, A.spacing.rinvOmρ²ᵗ)
            vρ = SIMDPirates.vbroadcast($V, A.ρ)
        end)
    elseif R <: AbstractRange
        pushfirst!(q.args, quote
            vinvOmρ² = SIMDPirates.vbroadcast($V, A.spacing.rinvOmρ²ᵗ)
            vρ = SIMDPirates.vbroadcast($V, A.spacing.ρᵗ)
        end)
    end
    q
end

function LinearAlgebra.det(A::AbstractAutoregressiveMatrix{T,R,S}) where {T,R,S<:EvenSpacing}
    A.spacing.rinvOmρ²ᵗ ^ (length(A.τ) - 1)
end
function LinearAlgebra.logdet(A::AbstractAutoregressiveMatrix{T,R,S}) where {T,R,S<:EvenSpacing}
    (length(A.τ) - 1) * log(A.spacing.rinvOmρ²ᵗ)
end

# @generated function LinearAlgebra.det(A::AbstractAutoregressiveMatrix{T,R,S}) where {T,R,S<:UnevenSpacing}
#     prod(A.rinvOmρ²ᵗ)
# end
# @generated function LinearAlgebra.logdet(A::AbstractAutoregressiveMatrix{T,R,S}) where {T,R,S<:UnevenSpacing}
#     quote
#         out = zero(T)
#         rinvOmρ²ᵗ = A.rinvOmρ²ᵗ
#         ρ = AR.ρ
#         @vectorize $T for i ∈ eachindex(A.τ)
#             ρᵗ = ρ ^ A.τ[i]
#             out += -0.5 * log(1 - ρᵗ * ρᵗ)
#         end
#         out
#     end
# end

function ∂det(A::AbstractAutoregressiveMatrix{T,R,S}) where {T,R <: AbstractUnitRange,S <: EvenSpacing}
    N = length(A.τ) - 1
    d = A.spacing.rinvOmρ²ᵗ^N
    d, A.ρ * N * d * A.spacing.invOmρ²ᵗ
end
function ∂logdet(A::AbstractAutoregressiveMatrix{T,R,S}) where {T,R <: AbstractUnitRange,S <: EvenSpacing}
    N = (length(A.τ) - 1)
    N * log(A.spacing.rinvOmρ²ᵗ), N * A.ρ * A.spacing.invOmρ²ᵗ
end
function ∂det(A::AbstractAutoregressiveMatrix{T,R,S}) where {T,R <: AbstractRange,S <: EvenSpacing}
    N = length(A.τ) - 1
    d = A.spacing.rinvOmρ²ᵗ^N
    d, step(A.τ) * A.ρ^(2step(A.τ) - 1) * N * d * A.spacing.invOmρ²ᵗ
end
function ∂logdet(A::AbstractAutoregressiveMatrix{T,R,S}) where {T,R <: AbstractRange,S <: EvenSpacing}
    N = (length(A.τ) - 1)
    N * log(A.spacing.rinvOmρ²ᵗ),  - 2N*step(A.τ) * A.ρ^(2step(A.τ) - 1) * A.spacing.invOmρ²ᵗ
end

# function ∂det(A::AbstractAutoregressiveMatrix{T,R}) where {T,R<:AbstractVector}
#     prod(A.rinvOmρ²ᵗ)
# end
# @generated function ∂logdet(A::AbstractAutoregressiveMatrix{T,R}) where {T,R<:AbstractVector}
#     quote
#         out = zero(T)
#         partialout = zero(T)
#         rinvOmρ²ᵗ = A.rinvOmρ²ᵗ
#         @vectorize $T for i ∈ eachindex(A.τ)
#             out += log(rinvOmρ²ᵗ[i])
#             partialout += 1 / rinvOmρ²ᵗ[i]
#         end
#         out, partialout
#     end
# end


@generated function quadform(
            A::AbstractAutoregressiveMatrix{T,R},
            B::AbstractFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}}
        ) where {T,R,M,N,W}
    V = NTuple{W,Core.VecElement{T}}
    q = quote
        $(Expr(:meta, :inline))
        qf = SIMDPirates.vbroadcast($V, zero($T))
    end
    if R <: AbstractUnitRange
        push!(q.args, quote
            vrinvOmρ² = SIMDPirates.vbroadcast($V, A.spacing.rinvOmρ²ᵗ)
            vρ = SIMDPirates.vbroadcast($V, A.ρ)
        end)
    elseif R <: AbstractRange
        push!(q.args, quote
            vrinvOmρ² = SIMDPirates.vbroadcast($V, A.spacing.rinvOmρ²ᵗ)
            vρ = SIMDPirates.vbroadcast($V, A.spacing.ρᵗ)
        end)
    end
    if R <: AbstractRange
        push!(q.args, quote
            for n ∈ 0:$(N-1)
                A_1 = B[1 + $M*n]
                qf = SIMDPirates.vmuladd(A_1, A_1, qf)
                $([quote
                        $(Symbol(:A_,m)) = SIMDPirates.evmul( $vrinvOmρ², SIMDPirates.vfnmadd( $vρ, B[ $(m-1) + $M*n ], B[ $m + $M*n ] ) )
                        qf = SIMDPirates.vmuladd($(Symbol(:A_,m)), $(Symbol(:A_,m)), qf)
                    end for m ∈ 2:M]...)
            end
        end)
    else
    # loop_body = :( SIMDPirates.evmul( , SIMDPirates.vfnmadd( , B[ m + M*n ], B[ 1+m + M*n ] ) ) )
        push!(q.args, quote
            $([ :($(Symbol(:ρᵗ_,    m)) = SIMDPirates.vbroadcast($V, A.ρᵗ[$m]))        for m ∈ 1:M-1]...)
            $([ :($(Symbol(:riOmρ²_,m)) = SIMDPirates.vbroadcast($V, A.rinvOmρ²ᵗ[$m])) for m ∈ 1:M-1]...)
            for n ∈ 0:$(N-1)
                A_1 = B[1 + $M*n]
                qf = SIMDPirates.vmuladd(A_1, A_1, qf)
                $([quote
                        $(Symbol(:A_,m)) = SIMDPirates.evmul( $(Symbol(:riOmρ²_,m)), SIMDPirates.vfnmadd( $(Symbol(:ρᵗ,m)), B[ $(m-1) + $M*n ], B[ $m + $M*n ] ) )
                        qf = SIMDPirates.vmuladd($(Symbol(:A_,m)), $(Symbol(:A_,m)), qf)
                    end for m ∈ 2:M]...)
            end
        end)
    end

    push!(q.args, :(qf, ConstantFixedSizePaddedMatrix(product)))

    q

end

@generated function mul_and_quadform(
            A::AbstractAutoregressiveMatrix{T,R},
            B::AbstractFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}}
        ) where {T,R,M,N,W}
    V = NTuple{W,Core.VecElement{T}}
    q = quote
        $(Expr(:meta, :inline))
        qf = SIMDPirates.vbroadcast($V, zero($T))
        product = MutableFixedSizePaddedMatrix{$M,$N,$V}(undef)
    end
    if R <: AbstractUnitRange
        push!(q.args, quote
            vrinvOmρ² = SIMDPirates.vbroadcast($V, A.spacing.rinvOmρ²ᵗ)
            vρ = SIMDPirates.vbroadcast($V, A.ρ)
        end)
    elseif R <: AbstractRange
        push!(q.args, quote
            vrinvOmρ² = SIMDPirates.vbroadcast($V, A.spacing.rinvOmρ²ᵗ)
            vρ = SIMDPirates.vbroadcast($V, A.spacing.ρᵗ)
        end)
    # else
    end
    if R <: AbstractRange
        push!(q.args, quote
            for n ∈ 0:$(N-1)
                A_1 = B[1 + $M*n]
                product[1 + $M*n] = A_1
                qf = SIMDPirates.vmuladd(A_1, A_1, qf)
                $([quote
                        $(Symbol(:A_,m)) = SIMDPirates.evmul( $vrinvOmρ², SIMDPirates.vfnmadd( $vρ, B[ $(m-1) + $M*n ], B[ $m + $M*n ] ) )
                        product[$m + $M*n] = $(Symbol(:A_,m))
                        qf = SIMDPirates.vmuladd($(Symbol(:A_,m)), $(Symbol(:A_,m)), qf)
                    end for m ∈ 2:M]...)
            end
        end)
    else
        # loop_body = :( SIMDPirates.evmul( , SIMDPirates.vfnmadd( , B[ m + M*n ], B[ 1+m + M*n ] ) ) )
        push!(q.args, quote
            $([ :($(Symbol(:ρᵗ_,    m)) = SIMDPirates.vbroadcast($V, A.ρᵗ[$m]))        for m ∈ 1:M-1]...)
            $([ :($(Symbol(:riOmρ²_,m)) = SIMDPirates.vbroadcast($V, A.rinvOmρ²ᵗ[$m])) for m ∈ 1:M-1]...)
            for n ∈ 0:$(N-1)
                A_1 = B[1 + $M*n]
                product[1 + $M*n] = A_1
                qf = SIMDPirates.vmuladd(A_1, A_1, qf)
                $([quote
                        $(Symbol(:A_,m)) = SIMDPirates.evmul( $(Symbol(:riOmρ²_,m)), SIMDPirates.vfnmadd( $(Symbol(:ρᵗ,m)), B[ $(m-1) + $M*n ], B[ $m + $M*n ] ) )
                        product[$m + $M*n] = $(Symbol(:A_,m))
                        qf = SIMDPirates.vmuladd($(Symbol(:A_,m)), $(Symbol(:A_,m)), qf)
                    end for m ∈ 2:M]...)
            end
        end)
    end

    push!(q.args, :(qf, ConstantFixedSizePaddedMatrix(product)))

    q

end
function ∂mul_and_quadform(
            A::AbstractAutoregressiveMatrix{T,R,S},
            B::AbstractFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}}
        ) where {T,R,M,N,W,S}
    V = NTuple{W,Core.VecElement{T}}
    q = quote
        $(Expr(:meta, :inline))
        qf = SIMDPirates.vbroadcast($V, zero($T))
        product = MutableFixedSizePaddedMatrix{$M,$N,$V}(undef)
    end
    if R <: AbstractUnitRange
        push!(q.args, quote
            vrinvOmρ² = SIMDPirates.vbroadcast($V, A.spacing.rinvOmρ²ᵗ)
            vρ = SIMDPirates.vbroadcast($V, A.ρ)
            ρ⁻¹2 = SIMDPirates.vbroadcast($V, 2 / A.ρ)
        end)
    elseif R <: AbstractRange
        push!(q.args, quote
            vrinvOmρ² = SIMDPirates.vbroadcast($V, A.spacing.rinvOmρ²ᵗ)
            vρ = SIMDPirates.vbroadcast($V, A.spacing.ρᵗ)
            ρ⁻¹2 = SIMDPirates.vbroadcast($V, T(2*step(AR.τ)) / A.ρ)
        end)
    # else
    end
    if S <: AbstractEvenSpacing
        push!(q.args, quote
            ρ²ᵗ = SIMDPirates.vbroadcast($V, $( R <: AbstractUnitRange ? :(A.ρ*A.ρ) : :(A.spacing.ρᵗ*A.spacing.ρᵗ)  ) )
            for n ∈ 0:$(N-1)
                A_1 = B[1 + $M*n]
                product[1 + $M*n] = A_1
                qf = SIMDPirates.vmuladd(A_1, A_1, qf)
                $([quote
                        ρᵗB = SIMDPirates.evmul($vρ, B[$(m-1)+$M*n ])
                        $(Symbol(:A_,m)) = SIMDPirates.evmul( $vrinvOmρ², SIMDPirates.vsub(B[$m+$M*n], ρᵗB ) )
                        product[$m + $M*n] = $(Symbol(:A_,m))

                        qf = SIMDPirates.vmuladd($(Symbol(:A_,m)), $(Symbol(:A_,m)), qf)

                        δ = SIMDPirates.evmul( $(Symbol(:A_,m)), $vrinvOmρ² )
                        ∂out = SIMDPirates.vmuladd(SIMDPirates.evmul(δ, ρ⁻¹2), SIMDPirates.vfmsub(δ, ρ²ᵗ, ρᵗB), ∂out)

                    end for m ∈ 2:M]...)
            end
        end)
    elseif S == CachedUnevenSpacing
        push!(q.args, quote
            $([ :($(Symbol(:riOmρ²_,m)) = SIMDPirates.vbroadcast($V, A.spacing.rinvOmρ²ᵗ[$m])) for m ∈ 1:M-1]...)
            $([ :($(Symbol(:ρᵗ_,    m)) = SIMDPirates.vbroadcast($V, A.spacing.ρᵗ[$m]))        for m ∈ 1:M-1]...)
            $([ :($(Symbol(:ρ²ᵗ_,   m)) = SIMDPirates.vbroadcast($V, A.spacing.ρᵗ[$m]*A.spacing.ρᵗ[$m])) for m ∈ 1:M-1]...)
            $([ :($(Symbol(:ρ⁻¹2_, m)) = SIMDPirates.vbroadcast($V, T(2*AR.τ[$m]) / A.ρ)) for m ∈ 1:M-1]...)
            for n ∈ 0:$(N-1)
                A_1 = B[1 + $M*n]
                product[1 + $M*n] = A_1
                qf = SIMDPirates.vmuladd(A_1, A_1, qf)
                $([quote
                        ρᵗB = SIMDPirates.evmul($(Symbol(:ρᵗ,     m)), B[$(m-1)+$M*n ])
                        $(Symbol(:A_,m)) = SIMDPirates.evmul( $(Symbol(:riOmρ²_,m)), SIMDPirates.vsub(B[$m+$M*n], ρᵗB ) )
                        product[$m + $M*n] = $(Symbol(:A_,m))

                        qf = SIMDPirates.vmuladd($(Symbol(:A_,m)), $(Symbol(:A_,m)), qf)

                        δ = SIMDPirates.evmul( $(Symbol(:A_,m)), $(Symbol(:riOmρ²_,m)) )
                        ∂out = SIMDPirates.vmuladd(SIMDPirates.evmul(δ, $(Symbol(:ρ⁻¹2_, m))), SIMDPirates.vfmsub(δ, $(Symbol(:ρ²ᵗ_,   m)), ρᵗB), ∂out)

                    end for m ∈ 2:M]...)
            end
        end)
    else
        throw("Spacing type $S not yet supported. Perhaps try caching intermediate results: `cache(AR_matrix)`")
    end

    push!(q.args, :(qf, ConstantFixedSizePaddedMatrix(product)))

    q

end



@generated function mul_and_quadform(
            A::AbstractAutoregressiveMatrix{T,R,S},
            B::AbstractFixedSizePaddedMatrix{M,N,T}
        ) where {T <: Number,R,M,N,S}

    W, Wshift = VectorizationBase.pick_vector_widthshift(M-1, T)
    V = NTuple{W,Core.VecElement{T}}
    q = quote
        $(Expr(:meta, :inline))
        Base.Cartesian.@nexprs $N n -> qf_n = SIMDPirates.vbroadcast($V, zero($T))
        product = MutableFixedSizePaddedMatrix{$M,$N,$T}(undef)
    end
    if R <: AbstractUnitRange
        push!(q.args, quote
            vrinvOmρ² = SIMDPirates.vbroadcast($V, A.spacing.rinvOmρ²ᵗ)
            vρ = SIMDPirates.vbroadcast($V, A.ρ)
        end)
    elseif R <: AbstractRange
        push!(q.args, quote
            vrinvOmρ² = SIMDPirates.vbroadcast($V, A.spacing.rinvOmρ²ᵗ)
            vρ = SIMDPirates.vbroadcast($V, A.spacing.ρᵗ)
        end)
    # else
    end
    if S <: AbstractEvenSpacing
        push!(q.args, quote
            A_1 = B[1]
            product[1] = A_1
            qf = A_1 * A_1
            Base.Cartesian.@nexprs $(N-1) n -> begin
                A_1 = B[1 + $M*n]
                product[1 + $M*n] = A_1
                qf = SIMDPirates.vmuladd(A_1, A_1, qf)
            end
            # Base.Cartesian.@nexprs $(N-1) n -> qf_n = SIMDPirates.evmul(A_1, A_1)
            @vectorize $T for m in 1:$(M-1)
                $([quote
                        A_m = SIMDPirates.evmul( vrinvOmρ², SIMDPirates.vfnmadd( vρ, B[ m + $(M*n-1) ], B[ m + $(M*n) ] ) )
                        product[m + $(M*n)] = A_m
                        $(Symbol(:qf_,n)) = SIMDPirates.vmuladd(A_m, A_m, $(Symbol(:qf_,n)))
                    end for n in 1:N]...)
            end
            # for n ∈ 0:$(N-1)
            #     A_1 = B[1 + $M*n]
            #     product[1 + $M*n] = A_1
            #     qf = SIMDPirates.vmuladd(A_1, A_1, qf)
            #     @vectorize $T for m in 1:$(M-1)
            #         A_m = SIMDPirates.evmul( vrinvOmρ², SIMDPirates.vfnmadd( vρ, B[ m - 1 + $M*n ], B[ m + $M*n ] ) )
            #         product[m + $M*n] = A_m
            #         qf = SIMDPirates.vmuladd(A_m, A_m, qf)
            #     end
            # end
        end)
    elseif S == CachedUnevenSpacing
        push!(q.args, quote
            A_1 = B[1]
            product[1] = A_1
            qf = A_1 * A_1
            Base.Cartesian.@nexprs $(N-1) n -> begin
                A_1 = B[1 + $M*n]
                product[1 + $M*n] = A_1
                qf = SIMDPirates.vmuladd(A_1, A_1, qf)
            end
            rinvOmρ²ᵗ = A.spacing.rinvOmρ²ᵗ
            ρᵗ = A.spacing.ρᵗ
            @vectorize $T for m in 1:$(M-1)
                vρ = ρᵗ[m]
                vrinvOmρ² = rinvOmρ²ᵗ[m]
                $([quote
                    A_m = SIMDPirates.evmul( vrinvOmρ², SIMDPirates.vfnmadd( vρ, B[ m + $(M*n-1) ], B[ m + $(M*n) ] ) )
                    product[m + $(M*n)] = A_m
                    $(Symbol(:qf_,n)) = SIMDPirates.vmuladd(A_m, A_m, $(Symbol(:qf_,n)))
                end for n in 1:N]...)
            end
        end)
    end

    push!(q.args, :(qf, ConstantFixedSizePaddedMatrix(product)))

    q

end
function ∂mul_and_quadform(
            A::AbstractAutoregressiveMatrix{T,R,S},
            B::AbstractFixedSizePaddedMatrix{M,N,T}
        ) where {T <: Number,R,M,N,W,S}
    V = NTuple{W,Core.VecElement{T}}
    q = quote
        $(Expr(:meta, :inline))
        qf = SIMDPirates.vbroadcast($V, zero($T))
        product = MutableFixedSizePaddedMatrix{$M,$N,$V}(undef)
    end
    if R <: AbstractUnitRange
        push!(q.args, quote
            vrinvOmρ² = SIMDPirates.vbroadcast($V, A.spacing.rinvOmρ²ᵗ)
            vρ = SIMDPirates.vbroadcast($V, A.ρ)
            ρ⁻¹2 = SIMDPirates.vbroadcast($V, 2 / A.ρ)
        end)
    elseif R <: AbstractRange
        push!(q.args, quote
            vrinvOmρ² = SIMDPirates.vbroadcast($V, A.spacing.rinvOmρ²ᵗ)
            vρ = SIMDPirates.vbroadcast($V, A.spacing.ρᵗ)
            ρ⁻¹2 = SIMDPirates.vbroadcast($V, T(2*step(AR.τ)) / A.ρ)
        end)
    # else
    end
    if S <: AbstractEvenSpacing
        push!(q.args, quote
            ρ²ᵗ = SIMDPirates.vbroadcast($V, $( R <: AbstractUnitRange ? :(A.ρ*A.ρ) : :(A.spacing.ρᵗ*A.spacing.ρᵗ)  ) )
            for n ∈ 0:$(N-1)
                A_1 = B[1 + $M*n]
                product[1 + $M*n] = A_1
                qf = SIMDPirates.vmuladd(A_1, A_1, qf)
                $([quote
                        ρᵗB = SIMDPirates.evmul($vρ, B[$(m-1)+$M*n ])
                        $(Symbol(:A_,m)) = SIMDPirates.evmul( $vrinvOmρ², SIMDPirates.vsub(B[$m+$M*n], ρᵗB ) )
                        product[$m + $M*n] = $(Symbol(:A_,m))

                        qf = SIMDPirates.vmuladd($(Symbol(:A_,m)), $(Symbol(:A_,m)), qf)

                        δ = SIMDPirates.evmul( $(Symbol(:A_,m)), $vrinvOmρ² )
                        ∂out = SIMDPirates.vmuladd(SIMDPirates.evmul(δ, ρ⁻¹2), SIMDPirates.vfmsub(δ, ρ²ᵗ, ρᵗB), ∂out)

                    end for m ∈ 2:M]...)
            end
        end)
    elseif S == CachedUnevenSpacing
        push!(q.args, quote
            $([ :($(Symbol(:riOmρ²_,m)) = SIMDPirates.vbroadcast($V, A.rinvOmρ²ᵗ[$m])) for m ∈ 1:M-1]...)
            $([ :($(Symbol(:ρᵗ_,    m)) = SIMDPirates.vbroadcast($V, A.spacing.ρᵗ[$m]))        for m ∈ 1:M-1]...)
            $([ :($(Symbol(:ρ²ᵗ_,   m)) = SIMDPirates.vbroadcast($V, A.spacing.ρᵗ[$m]*A.spacing.ρᵗ[$m])) for m ∈ 1:M-1]...)
            $([ :($(Symbol(:ρ⁻¹2_, m)) = SIMDPirates.vbroadcast($V, T(2*AR.τ[$m]) / A.ρ)) for m ∈ 1:M-1]...)
            for n ∈ 0:$(N-1)
                A_1 = B[1 + $M*n]
                product[1 + $M*n] = A_1
                qf = SIMDPirates.vmuladd(A_1, A_1, qf)
                $([quote
                        ρᵗB = SIMDPirates.evmul($(Symbol(:ρᵗ, m)), B[$(m-1)+$M*n ])
                        $(Symbol(:A_,m)) = SIMDPirates.evmul( $(Symbol(:riOmρ²_,m)), SIMDPirates.vsub(B[$m+$M*n], ρᵗB ) )
                        product[$m + $M*n] = $(Symbol(:A_,m))

                        qf = SIMDPirates.vmuladd($(Symbol(:A_,m)), $(Symbol(:A_,m)), qf)

                        δ = SIMDPirates.evmul( $(Symbol(:A_,m)), $(Symbol(:riOmρ²_,m)) )
                        ∂out = SIMDPirates.vmuladd(SIMDPirates.evmul(δ, $(Symbol(:ρ⁻¹2_, m))), SIMDPirates.vfmsub(δ, $(Symbol(:ρ²ᵗ_,   m)), ρᵗB), ∂out)

                    end for m ∈ 2:M]...)
            end
        end)
    else
        throw("Spacing type $S not yet supported. Perhaps cache(AR_matrix) will work.")
    end

    push!(q.args, :(qf, ConstantFixedSizePaddedMatrix(product)))

    q

end

