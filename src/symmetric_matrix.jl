
"""
The data layout is:
diagonal elements
sub diagonal lower triangular part

The matrix is also padded.
Given avx2, a 7x7 matrix of Float64 would be stored as:
7 diagonal elements + 1 element padding

6x6 lower triangle...
column 1: 6 elements + 2 padding
column 2: 5 elements + 3 padding
column 3: 4 elements + 0 padding
column 4: 3 elements + 1 padding
column 4: 2 elements + 2 padding
column 4: 1 elements + 3 padding
"""
@inline function Base.getindex(S::AbstractSymmetricMatrixL{P,T,L}, i, j) where {P,T,L}
    j, i = minmax(j, i)
    @boundscheck i > P && ThrowBoundsError("max(i, j) = $j > $P.")
    @inbounds S.data[lt_sub2ind(P, i, j)]
end
@inline function Base.getindex(S::AbstractSymmetricMatrixU{P,T,L}, i, j) where {P,T,L}
    i, j = minmax(i, j)
    @boundscheck j > P && ThrowBoundsError("max(i, j) = $j > $P.")
    @inbounds S.data[ut_sub2ind(P, i, j)]
end

@inline function Base.setindex!(S::AbstractMutableSymmetricMatrixL{P,T,L}, v::T, i, j) where {P,T,L}
    j, i = minmax(j, i)
    @boundscheck i > P && ThrowBoundsError("max(i, j) = $j > $P.")
    VectorizationBase.store!(pointer(S) + sizeof(T) * (lt_sub2ind(P, i, j)-1), v)
    @inbounds S.data[lt_sub2ind(P, i, j)]
end
@inline function Base.setindex!(S::AbstractMutableSymmetricMatrixU{P,T,L}, v::T, i, j) where {P,T,L}
    i, j = minmax(i, j)
    @boundscheck j > P && ThrowBoundsError("max(i, j) = $j > $P.")
    VectorizationBase.store!(pointer(S) + sizeof(T) * (ut_sub2ind(P, i, j)-1), v)
end

#@inline function Base.setindex!(S::AbstractMutableSymmetricMatrixL{P,T,L}, v::T, i) where {P,T,L}
#    j, i = minmax(j, i)
#    @boundscheck i > P && ThrowBoundsError("i = $i > $L = L")
#    VectorizationBase.store!(pointer(S) + sizeof(T) * i, v)
#end
#@inline function Base.setindex!(S::AbstractMutableSymmetricMatrixU{P,T,L}, v::T, i) where {P,T,L}
#    @boundscheck i > L && ThrowBoundsError("i = $i > $L = L")
#    VectorizationBase.store!(pointer(S) + sizeof(T) * i, v)
#end

@generated function SymmetricMatrixL(S::PaddedMatrices.AbstractFixedSizePaddedMatrix{M,M,T,P}) where {M,T,P}
    Lbase = binomial2(M+1)
    Wm1 = VectorizationBase.pick_vector_width(Lbase, T) - 1
    Lfull = (Lbase + Wm1) & ~Wm1

    outtup = Expr(:tuple)
    for m ∈ 1:M
        push!(outtup.args, :(S[ $(P*(m-1) + m) ]))
    end
    for mc ∈ 1:M-1
        for mr ∈ mc+1:M
            push!(outtup.args, :( S[$( P*(mc-1) + mr )] ))
        end
    end
    for l ∈ Lbase+1:Lfull
        push!(outtup.args, :($(zero(T))))
    end
    :(@inbounds SymmetricMatrixL{$M,$T,$Lfull}($outtup))
end
function MutableSymmetricMatrixL(S::PaddedMatrices.AbstractFixedSizePaddedMatrix{M,M,T,P}) where {M,T,P}
    Lbase = binomial2(M+1)
#    Wm1 = VectorizationBase.pick_vector_width(Lbase, T) - 1
    #    Lfull = (Lbase + Wm1) & ~Wm1
    Sm = MutableSymmetricMatrixL{M,T,Lbase}(undef)
    @inbounds for m ∈ 1:M
        Sm[m] = S[m,m]
    end
    @inbounds for mc ∈ 1:M-1, mr ∈ mc+1:M
        Sm[mr,mc] = S[mr,mc]
    end
    Sm
end


# @generated function lower_cholesky(Σ::SymmetricMatrixL{P,T,L}) where {P,T,L}
#
# end

### Be more clever than this.
# @generated function quadform(x::AbstractFixedSizePaddedVector{P,T,R}, Σ::SymmetricMatrixL{P,T,L}) where {P,T,R,L}
#     W = pick_vector_width(P, T)
#     q = quote
#         # vout = vbroadcast(Vec{$P,$T}, zero($T))
#         out_1 = zero($T)
#         @vectorize $T for i ∈ 1:$P
#             out_1 += x[i] * Σ[i]
#         end
#     end
#     for p ∈ 2:P
#         out_temp = Symbol(:out_, p)
#         push!(q.args, quote
#             @vectorize $T for i ∈ 1:$P
#                 $out_temp += x[i] * Σ[i]
#             end
#         end)
#     end
#     q
# end

# @generated function Base.:*(A::AbstractFixedSizePaddedMatrix{M,P,T}, Σ::SymmetricMatrix{P,T,L}) where {M,P,T,L}
#     W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
#
#     quote
#
#     end
# end
