struct TriangleInverseAdjoint{P,T,L}  <: AbstractArray{T,4}
    data::NTuple{L,T}
end


#
num_nonzero_in_∂inv(N) = sum(i -> binomial2(i+1) * (N+1-i), 1:N)



function create_inv_tri_adj_tuple(P, Ubase = :U, Lbase = :L)
    outtup = Expr(:tuple,)
    for p ∈ 1:P
        push!(outtup.args, ∂sym(Ubase, p, p, Lbase, p, p))
    end
    ind = P
    for lcol ∈ 1:P
        for lrow ∈ lcol:P
            for ucol ∈ max(2,lrow):P
                for urow ∈ 1:min(P,ucol-1,lcol)
                    push!(outtup.args, ∂sym(Ubase, urow, ucol, Lbase, lrow, lcol))
                end
            end
        end
    end
    outtup
end

# function ∂inv(Lt::LowerTriangularMatrix{P,T,L}) where {P,L,T}
@generated function ∂inv(Lt::AbstractLowerTriangularMatrix{P,T,L}) where {P,L,T}
# @generated function ∂inv(Lt::LowerTriangularMatrix{P,T,L}) where {P,T,L}
    q = quote end
    qa = q.args
    load_packed_L_quote!(qa, P, :Lt, :Lt)
    ∂inv_L_core_quote!(qa, P, :U, :Lt, T)
    uq = store_packed_U_quote!(qa, P, :U, T, L)
    Ladj = num_nonzero_in_∂inv(P)
    quote
        $(Expr(:meta,:inline))
        @fastmath @inbounds begin
            # begin
            $q
            $uq, TriangleInverseAdjoint{$P,$T,$Ladj}($(create_inv_tri_adj_tuple(P, :U, :Lt)))
        end
    end
end
struct ∂Inverse{T}
    x⁻¹::T
end
@inline function ∂inv(x)
    x⁻¹ = Base.FastMath.inv_fast(x)
    x⁻¹, ∂Inverse(x⁻¹)
end
@inline function ReverseDiffExpressionsBase.RESERVED_INCREMENT_SEED_RESERVED!(
    c::ReverseDiffExpressionsBase.AbstractUninitializedReference, ∂::∂Inverse, a
)
    x⁻¹′ = ∂.x⁻¹'
    c[] = @fastmath - x⁻¹′ * a * x⁻¹′
end
@inline function ReverseDiffExpressionsBase.RESERVED_INCREMENT_SEED_RESERVED!(
    c, ∂::∂Inverse, a
)
    x⁻¹′ = ∂.x⁻¹'
    @fastmath c[] -= x⁻¹′ * a * x⁻¹′
end

@generated function ∂inv′(Lt::AbstractLowerTriangularMatrix{P,T,L}) where {P,L,T}
# @generated function ∂inv(Lt::LowerTriangularMatrix{P,T,L}) where {P,T,L}
    q = quote end
    qa = q.args
    load_packed_L_quote!(qa, P, :Lt, :Lt)
    ∂inv_L_core_quote!(qa, P, :U, :Lt, T)
    uq = store_packed_U_quote!(qa, P, :U, T, L)
    Ladj = num_nonzero_in_∂inv(P)
    quote
        # $(Expr(:meta,:inline))
        @fastmath @inbounds begin
            # begin
            $q
            $uq, TriangleInverseAdjoint{$P,$T,$Ladj}($(create_inv_tri_adj_tuple(P, :U, :Lt)))
        end
    end
end
@inline function ∂inv′(x)
    x⁻¹ = Base.FastMath.inv_fast(x)'
    x⁻¹, ∂Inverse(x⁻¹)
end

function partial_inv_quote(P,T,L = binomial2(P+1))
    q = quote end
    qa = q.args
    load_packed_L_quote!(qa, P, :Lt, :Lt)
    ∂inv_L_core_quote!(qa, P, :U, :Lt, T)
    uq = store_packed_U_quote!(qa, P, :U, T, L)
    Ladj = num_nonzero_in_∂inv(P)
    quote
        # $(Expr(:meta,:inline))
        # @fastmath @inbounds begin
            # begin
            $q
            $uq, TriangleInverseAdjoint{$P,$T,$Ladj}($(create_inv_tri_adj_tuple(P, :U, :Lt)))
        # end
    end
end


# extract triangle_transpose

function triangle_adjoint_quote(P,T,L,PL)

    # How to order elements: first P ∂diags
    q = quote end
    loadq = quote end
    storeq = Expr(:tuple,)

    for p ∈ 1:P
        push!(loadq.args, :( $(Symbol(:∂U_,p,:_,p)) = t[$p]))
        push!(q.args, :( $(Symbol(:∂L_,p,:_,p)) = $(Symbol(:∂U_,p,:_,p)) * adj[$p] ))
        push!(storeq.args, Symbol(:∂L_,p,:_,p))
    end
    ind = P
    for ucol ∈ 2:P, urow ∈ 1:ucol-1
        ind += 1
        push!(loadq.args, :( $(Symbol(:∂U_,urow,:_,ucol)) = t[$ind]))
    end

    ind = P
    for lcol ∈ 1:P
        for lrow ∈ lcol:P
            first = lrow == lcol ? false : true
            for ucol ∈ max(2,lrow):P
                for urow ∈ 1:min(P,ucol-1,lcol)
                    ind += 1
                    if first
                        push!(q.args, :( $(Symbol(:∂L_,lrow,:_,lcol)) = $(Symbol(:∂U_,urow,:_,ucol)) * adj[$ind] ))
                        first = false
                    else
                        push!(q.args, :( $(Symbol(:∂L_,lrow,:_,lcol)) += $(Symbol(:∂U_,urow,:_,ucol)) * adj[$ind] ))
                    end
                    # push!(qa, :($(Symbol(:∂U_,urow,:_,ucol,:_∂L_,lrow,:_,lcol)) = $adjsym[$ind] ))
                end
            end
        end
    end

    for lcol ∈ 1:P-1, lrow ∈ lcol+1:P
        push!(storeq.args, Symbol(:∂L_,lrow,:_,lcol))
    end
    outsize = binomial2(P+1)
    for i ∈ outsize+1:PL
        push!(storeq.args, zero(T))
    end

    quote
        $loadq
        $q
    end, storeq
end


@generated function Base.:*(
    t::LinearAlgebra.Adjoint{T,<: PaddedMatrices.AbstractFixedSizeVector{M,T,PL}},
    adj::TriangleInverseAdjoint{P,T,L}
) where {P,T,L,M,PL}
    outsize = binomial2(P+1)
    q, storeq = triangle_adjoint_quote(P,T,L,PL)
    quote
        @fastmath @inbounds begin
            $q
            ConstantFixedSizeVector{$outsize,$T}($storeq)'
        end
    end
end
@generated function Base.:*(
            t::AbstractUpperTriangularMatrix{P,T},
            adj::TriangleInverseAdjoint{P,T,L}
        ) where {P,T,L}
    M = binomial2(P+1)
    Wm1 = VectorizationBase.pick_vector_width(M,T) - 1
    PL = (M + Wm1) & ~Wm1
    q, storeq = triangle_adjoint_quote(P,T,L,PL)
    quote
        @fastmath @inbounds begin
            $q
            LowerTriangularMatrix{$P,$T,$PL}($storeq)
        end
    end
end

function load_inv_tri_adj!(qa, P, adjsym = :triangle_adjoint)
    for p ∈ 1:P
        push!(qa, :($(Symbol(:∂U_,p,:_,p,:_∂L_,p,:_,p)) = $adjsym[$p] ))
    end
    ind = P
    for lcol ∈ 1:P
        for lrow ∈ lcol:P
            for ucol ∈ max(2,lrow):P
                for urow ∈ 1:min(P,ucol-1,lcol)
                    ind += 1
                    push!(qa, :($(Symbol(:∂U_,urow,:_,ucol,:_∂L_,lrow,:_,lcol)) = $adjsym[$ind] ))
                end
            end

        end
    end

    qa

end


@inline function Base.getindex(adj::TriangleInverseAdjoint{P,T,L}, i) where {P,T,L}
    @boundscheck i > L && ThrowBoundsError("i = $i > $L;\nnote TriangleInverseAdjoint matrices are sparse.")
    @inbounds adj.data[i]
end
function Base.getindex(adj::TriangleInverseAdjoint{P,T}, i, j, k, l) where {P,T}
    # for p ∈ 1:P
    #     push!(qa, :($(Symbol(:∂U_,p,:_,p,:_∂L_,p,:_,p)) = $adjsym[$p] ))
    # end
    i == j == k == l && return adj[i]
    ind = P
    for lcol ∈ 1:P
        for lrow ∈ lcol:P
            for ucol ∈ max(2,lrow):P
                for urow ∈ 1:min(P,ucol-1,lcol)
                    ind += 1
                    if urow == i && ucol == j && lrow == k && lcol == l
                        return adj[ind]
                    end
                    # push!(qa, :($(Symbol(:∂U_,urow,:_,ucol,:_∂L_,lrow,:_,lcol)) = $adjsym[$ind] ))
                end
            end

        end
    end
    zero(T)
end
Base.size(::TriangleInverseAdjoint{P}) where {P} = (P,P,P,P)
