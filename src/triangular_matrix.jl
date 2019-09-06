

@inline function Base.getindex(S::AbstractLowerTriangularMatrix{P,T,L}, i, j) where {P,T,L}
    @boundscheck i > P && ThrowBoundsError("i == $i > $P")
    j > i && return zero(T)
    @inbounds S.data[lt_sub2ind(P, i, j)]
end

@inline function Base.getindex(S::AbstractMutableLowerTriangularMatrix{P,T,L}, i, j) where {P,T,L}
    @boundscheck i > P && ThrowBoundsError("i == $i > $P")
    j > i && return zero(T)
    VectorizationBase.load(pointer(S) + (lt_sub2ind(P, i, j) - 1)*sizeof(T))
end

@inline function Base.getindex(S::AbstractUpperTriangularMatrix{P,T,L}, i, j) where {P,T,L}
    i > j && return zero(T)
    @boundscheck j > P && ThrowBoundsError("j == $j > $P.")
    @inbounds S.data[ut_sub2ind(P, i, j)]
end
@inline function Base.getindex(S::AbstractMutableUpperTriangularMatrix{P,T,L}, i, j) where {P,T,L}
    @boundscheck i > P && ThrowBoundsError("i == $i > $P")
    i > j && return zero(T)
    VectorizationBase.load(pointer(S) + (ut_sub2ind(P, i, j) - 1)*sizeof(T))
end
@inline function Base.getindex(S::LinearAlgebra.Adjoint{Union{},<:AbstractUpperTriangularMatrix{P,Vec{W,T},L}}, i::Int, j::Int) where {P,T,L,W}
    j > i && return zero(T)
    ind = ut_sub2ind(P, j, i)
    @boundscheck i > L && ThrowBoundsError("ind == $ind > $L.")
    @inbounds S.parent.data[ind]
end
@inline function Base.getindex(S::AbstractLowerTriangularMatrix{P,NTuple{W,Core.VecElement{T}},L}, i, j) where {P,W,T,L}
    @boundscheck i > P && ThrowBoundsError("i == $i > $P")
    j > i && return SIMDPirates.vbroadcast(Vec{W,T}, zero(T))
    @inbounds S.data[lt_sub2ind(P, i, j)]
end

@inline function Base.getindex(S::AbstractUpperTriangularMatrix{P,NTuple{W,Core.VecElement{T}},L}, i, j) where {P,W,T,L}
    @boundscheck j > P && ThrowBoundsError("j == $j > $P.")
    i > j && return SIMDPirates.vbroadcast(Vec{W,T}, zero(T))
    @inbounds S.data[ut_sub2ind(P, i, j)]
end

@generated function LinearAlgebra.det(A::AbstractTriangularMatrix{P,T,L}) where {P,T,L}
    quote
        out = one(T)
        @vectorize for i ∈ 1:$P
            out *= A[i]
        end
        out
    end
end
"""
logdet(A) will be slower than log(det(A)),
but should be more numerically accurate.
det(A) is at high risk of over/underflow
for large matrices.
"""
@generated function LinearAlgebra.logdet(A::AbstractTriangularMatrix{P,T,L}) where {P,L,T}
    quote
        $(Expr(:meta,:inline))
        out = zero(T)
        @vectorize $T for i ∈ 1:$P
            out += log(A[i])
        end
        out
    end
end

@generated function ∂logdet(A::AbstractTriangularMatrix{P,T,L}) where {P,T,L}
    quote
        $(Expr(:meta,:inline))
        out = zero(T)
        ∂out = PaddedMatrices.MutableFixedSizePaddedVector{$P,$T}(undef)
        @vectorize $T for i ∈ 1:$P
            out += log(A[i])
            ∂out[i] = one($T) / A[i]
        end
        out, ∂out
    end
end

#
# function lower_chol_small(P,T,L = calculate_L(P, T))
#     W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
#     V = Vec{W,T}
#     MT = VectorizationBase.mask_type(W)
#     q = quote
#         vΣ = VectorizationBase.vectorizable(Σ)
#         @inbounds @fastmath diag_1 = sqrt( vΣ[1] )
#         c_1 = SIMDPirates.vfdiv(vload($V, vΣ + $W), vbroadcast($V, diag_1))
#         diag² = vload($V, vΣ)
#         diag² = vifelse( one($MT), diag², SIMDPirates.vfnmadd(c_1, c_1, diag²) )
#         @inbounds @fastmath diag_2 = sqrt(diag²[2].value)
#     end
#     mask64 = 1
#     for p ∈ 2:P-1
#         c_p = Symbol(:c_, p)
#         push!(q.args, :(
#             $c_p = vload($V, vΣ + $W*$p)
#         ))
#         for j ∈ 1:p-1
#             i = p - j
#             c_prev = Symbol(:c_, j)
#             push!(q.args, :(
#                 $c_p = SIMDPirates.vfnmadd($c_prev, vbroadcast($V, $c_prev[$p]), $c_p)
#             ))
#         end
#         mask64 += 2^(p-1)
#         push!(q.args, quote
#             $c_p = SIMDPirates.vfdiv($c_p, vbroadcast($V, $(Symbol(:diag_, p))))
#             diag² = vifelse( $(MT(mask64)), diag², SIMDPirates.vfnmadd($c_p, $c_p, diag²) )
#             @inbounds @fastmath $(Symbol(:diag_, p+1)) = sqrt(diag²[$(p+1)].value)
#         end)
#     end
#
#     push!(q.args, quote
#         LowerTriangularMatrix{$P,$T,$L}(
#             $(diagonal_lowertri_output_tuple_expr(P, W, :diag_, :c_))
#         )
#     end)
#     q
# end
#
# @generated function lower_cholesky(Σ::SymmetricMatrixL{P,T,L}) where {P,T,L}
#     W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
#     Wm1 = W - 1
#     PpWm1 = P + Wm1
#     num_diag_loads = PpWm1 >> Wshift
#     num_diag_loads == 1 && return lower_chol_small(P, T, L)
#     rem = P & Wm1
#     # num_diag_loads == 1 && return lower_chol_small(P, T, L)
#     num_full_diag_loads = P >> Wshift
#     # Do we start with the full number of loads, or not?
#     # initial_diag_ind = lower_triangle_sub2ind(Val(P), T, 1, 1)
#     V = Vec{W,T}
#     MT = mask_type(W)
#     q = quote
#         vΣ = VectorizationBase.vectorizable(Σ)
#         @inbounds @fastmath diag_1 = sqrt( vΣ[1] )
#         @fastmath diag⁻¹_1 = 1 / diag_1
#         vdiag⁻¹_1 = vbroadcast($V, diag⁻¹_1)
#         $([
#             :($(Symbol(:diag²_, n)) = vload($V, vΣ + $(n*W)))
#             for n ∈ 0:num_full_diag_loads-1
#         ]...)
#     end
#     if num_full_diag_loads < num_diag_loads
#         push!(q.args, :($(Symbol(:diag²_, num_diag_loads)) = vload($V, vΣ + $(num_full_diag_loads*W), $(MT(2^(P & Wm1)-1)))))
#     end
#     num_col_loads = num_diag_loads
#     col_rem = W - (P & Wm1)
#     for col ∈ 1:P-1
#         col_rem += 1
#         if col_rem == W
#             col_rem = 0
#             num_col_loads -= 1
#         end
#         for load ∈ 1:num_col_loads
#             push!(q.args, :(
#                 $(Symbol(:c_, load, :_, 1)) = SIMDPirates.evmul(vdiag⁻¹_1, vload(vΣ + $(num_diag_loads*W + (load-1)*W)))
#             ))
#         end
#     end
#
# end

function load_packed_L_quote!(qa, P, symbol_name, extract_from)
    for p ∈ 1:P
        push!(qa, :($(PaddedMatrices.sym(symbol_name, p, p)) = $extract_from[$p] ))
    end
    ind = P
    for pc ∈ 1:P
        for pr ∈ pc+1:P
            ind += 1
            push!(qa, :($(PaddedMatrices.sym(symbol_name, pr, pc)) = $extract_from[$ind] ))
            # push!(qa, :($(PaddedMatrices.sym(symbol_name, pr, pc)) = $extract_from[$(lt_sub2ind(P,pc,pr))] ))
        end
    end
    qa
end
function store_packed_L_quote!(qa, P, symbol_name, T, L)
    outtup = Expr(:tuple,)
    for p ∈ 1:P
        push!(outtup.args, PaddedMatrices.sym(symbol_name, p, p) )
    end
    for pc ∈ 1:P
        for pr ∈ pc+1:P
            push!(outtup.args, PaddedMatrices.sym(symbol_name, pr, pc))
        end
    end
    for p ∈ binomial2(P+1)+1:L
        push!(outtup.args, zero(T))
    end
    # push!(qa, :(LowerTriangularMatrix{$P,$T,$L}($(
    #     Expr(:tuple, outtup...)
    # ))))
    :(LowerTriangularMatrix{$P,$T,$L}($outtup))
end
function store_packed_Ut_quote!(qa, P, symbol_name, T, L)
    outtup = Expr(:tuple,)
    for p ∈ 1:P
        push!(outtup.args, PaddedMatrices.sym(symbol_name, p, p) )
    end
    for pc ∈ 1:P
        for pr ∈ 1:pc-1
            push!(outtup.args, PaddedMatrices.sym(symbol_name, pc, pr))
        end
    end
    for p ∈ binomial2(P+1)+1:L
        push!(outtup.args, zero(T))
    end
    # push!(qa, :(UpperTriangularMatrix{$P,$T,$L}($(
    #     Expr(:tuple, outtup...)
    # ))))
    :(UpperTriangularMatrix{$P,$T,$L}($outtup))
end
function store_packed_U_quote!(qa, P, symbol_name, T, L)
    outtup = Expr(:tuple,)
    for p ∈ 1:P
        push!(outtup.args, PaddedMatrices.sym(symbol_name, p, p) )
    end
    for pc ∈ 1:P
        for pr ∈ 1:pc-1
            push!(outtup.args, PaddedMatrices.sym(symbol_name, pr, pc))
        end
    end
    for p ∈ binomial2(P+1)+1:L
        push!(outtup.args, zero(T))
    end
    # push!(qa, :(UpperTriangularMatrix{$P,$T,$L}($(
    #     Expr(:tuple, outtup...)
    # ))))
    :(UpperTriangularMatrix{$P,$T,$L}($outtup))
end

import PaddedMatrices: sym
∂sym(A, i, j, B, k, l) = Symbol(:∂, sym(A, i, j), :_∂, sym(B, k, l))
∂sym(A, B) = Symbol(:∂, A, :_∂, B)

function ∂inv_L_core_quote!(qa, P, output = :L, input = :S, ::Type{T} = Float64) where {T}
    # ∂tup = Expr(:tuple,)
    # The lazy way to write this function is to just use a dict
    # to find out what I have to do.
    ∂relations = Dict{Symbol,Set{Tuple{Symbol,Symbol}}}()
    defined_syms = Set{Symbol}()
    for c ∈ 1:P
        inputsym = sym(input, c, c)
        outputsym = sym(output, c, c)
        ∂outinsym = ∂sym(outputsym, inputsym)
        push!(qa, :( $outputsym = $(one(T)) / $inputsym ))
        push!(qa, :( $∂outinsym = - $outputsym*$outputsym ))
        # push!(∂tup.args, ∂outinsym)
        push!(defined_syms, ∂outinsym)
        push!(get!(() -> Set{Tuple{Symbol,Symbol}}(), ∂relations, outputsym), (∂outinsym, inputsym) )
    end
    for c ∈ 1:P
        for r ∈ c+1:P
            outputsym_r_c = sym(output, c, r)
            inputsym_r_c = sym(input, r, c)
            outputsym_c_c = sym(output, c, c)
            outputsym_r_r = sym(output, r, r)

            push!(qa, :( $outputsym_r_c = $inputsym_r_c * $outputsym_c_c ))
            push!(qa, :( $(∂sym(outputsym_r_c, inputsym_r_c)) = -$outputsym_r_r * $outputsym_c_c ))
            # expand partial to outputs with respect to inputs
            push!(qa, :( $(∂sym(outputsym_r_c, sym(input, c, c))) = -$outputsym_r_r * $inputsym_r_c * $(∂sym(outputsym_c_c, sym(input, c, c)) )))

            # if ∂sym(output, r, c, input, r, c) ∉ defined_syms
                # push!(∂tup.args, ∂sym(output, r, c, input, r, c))
                push!(defined_syms, ∂sym(output, c, r, input, r, c))
            # else
            #     throw("oops, $(∂sym(output, r, c, input, r, c)) ∈ $defined_syms")
            # end
            outrc_vec = get!(() -> Set{Tuple{Symbol,Symbol}}(), ∂relations, outputsym_r_c)
            push!(outrc_vec, (∂sym(output, c, r, input, r, c), sym(input, r, c)) )
            # if ∂sym(output, r, c, input, c, c) ∉ defined_syms
                # push!(∂tup.args, ∂sym(output, r, c, input, c, c))
                push!(defined_syms, ∂sym(output, c, r, input, c, c))
            # else
            #     throw("oops, $(∂sym(output, r, c, input, c, c)) ∈ $defined_syms")
            # end

            push!(qa, :( $(∂sym(output, c, r, input, r, r)) =  - $(sym(output, c, c)) * $(sym(input, r, c)) * $(∂sym(output, r, r, input, r, r)) ) )
            push!(outrc_vec, (∂sym(output, c, r, input, r, r), sym(input, r, r)) )
            push!(outrc_vec, (∂sym(output, c, r, input, c, c), sym(input, c, c)) )
            for cr ∈ c+1:r-1
                push!(qa, :( $outputsym_r_c += $(sym(input, r, cr)) * $(sym(output, c, cr)) ))
                push!(qa, :( $(∂sym(output, c, r, input, r, cr)) = -$(sym(output, r, r)) * $(sym(output, c, cr)) ))
                # push!(∂tup.args, ∂sym(output, r, c, input, r, cr))
                push!(outrc_vec, (∂sym(output, c, r, input, r, cr), sym(input, r, cr)) )
                # expand partial to outputs with respect to inputs
                ∂outoutsym = ∂sym(output, c, r, output, c, cr)
                push!(qa, :( $∂outoutsym = -$(sym(output, r, r)) * $(sym(input, r, cr)) ))
                outsym_cr_c = sym(output, c, cr)


                # push!(qa, :( $(∂sym(output, r, c, output, cr, c)) = -$(sym(output, r, r)) * $(sym(input, r, cr)) ))

                for (∂outinsym, inputsym) ∈ ∂relations[outsym_cr_c]
                    defsym = ∂sym(outputsym_r_c, inputsym)
                    push!(outrc_vec, (defsym, inputsym) )
                    if defsym ∈ defined_syms
                        push!(qa, :( $defsym += $∂outoutsym * $∂outinsym ) )
                    else
                        push!(qa, :( $defsym = $∂outoutsym * $∂outinsym ) )
                        # push!(∂tup.args, defsym)
                        push!(defined_syms, defsym)
                    end
                end
            end
            # expand partial to outputs with respect to inputs
            ∂outoutsym_r_r = ∂sym(output, c, r, output, r, r)
            ∂outoutsym_r_c = ∂sym(output, c, r, output, c, r)
            push!(qa, :( $∂outoutsym_r_r =  - $(sym(output, c, r)) ) )
            push!(qa, :( $∂outoutsym_r_c =  - $(sym(output, c, r)) ) )
            push!(qa, :( $outputsym_r_c *=  -$(sym(output, r, r)) ) )

            for (∂outinsym, inputsym) ∈ ∂relations[sym(output, r, r)]
                defsym = ∂sym(outputsym_r_c, inputsym)
                if defsym ∈ defined_syms
                    push!(qa, :( $defsym += $∂outoutsym_r_r * $∂outinsym ) )
                else
                    push!(qa, :( $defsym = $∂outoutsym_r_r * $∂outinsym ) )
                    # push!(∂tup.args, defsym)
                    push!(defined_syms, defsym)
                end
            end
        end
    end
    # ∂tup
end

@generated function lower_chol(Σ::SymmetricMatrixL{P,T,L}) where {P,T,L}
    q = quote end
    qa = q.args
    load_packed_L_quote!(qa, P, :Σ, :Σ)
    PaddedMatrices.chol_L_core_quote!(qa, P, :Σ, T)
    lq = store_packed_L_quote!(qa, P, :Σ, T, L)
    quote
        $(Expr(:meta,:inline))
        @fastmath @inbounds begin
            # begin
            $q
            $lq
        end
    end
end
@generated function lower_chol(Σ::AbstractFixedSizePaddedMatrix{P,P,T,R}) where {P,T,R}
    q = quote end
    qa = q.args
    PaddedMatrices.load_L_quote!(qa, P, R, :Σ, :Σ)
    PaddedMatrices.chol_L_core_quote!(qa, P, :Σ, T)
    L = binomial2(P+1)
    Wm1 = VectorizationBase.pick_vector_width(L, T)-1
    L = (L + Wm1) & ~Wm1
    lq = store_packed_L_quote!(qa, P, :Σ, T, L)
    quote
        $(Expr(:meta,:inline))
        @fastmath @inbounds begin
            # begin
            $q
            $lq
        end
    end
end

@generated function PaddedMatrices.invchol(Σ::SymmetricMatrixL{P,T,L}) where {P,T,L}
    q = quote end
    qa = q.args
    load_packed_L_quote!(qa, P, :Σ, :Σ)
    PaddedMatrices.invchol_L_core_quote!(qa, P, :Lt, :Σ, T)
    uq = store_packed_Ut_quote!(qa, P, :Lt, T, L)
    quote
        $(Expr(:meta,:inline))
        @fastmath @inbounds begin
            # begin
            $q
            $uq
        end
    end
end
@generated function Base.inv(Lt::AbstractLowerTriangularMatrix{P,T,L}) where {P,T,L}
    q = quote end
    qa = q.args
    load_packed_L_quote!(qa, P, :Lt, :Lt)
    PaddedMatrices.inv_L_core_quote!(qa, P, :U, :Lt, T)
    uq = store_packed_Ut_quote!(qa, P, :U, T, L)
    quote
        $(Expr(:meta,:inline))
        @fastmath @inbounds begin
            # begin
            $q
            $uq
        end
    end
end
@generated function inv′(Lt::AbstractLowerTriangularMatrix{P,T,L}) where {P,T,L}
    q = quote end
    qa = q.args
    load_packed_L_quote!(qa, P, :Lt, :Lt)
    PaddedMatrices.inv_L_core_quote!(qa, P, :U, :Lt, T)
    uq = store_packed_Ut_quote!(qa, P, :U, T, L)
    quote
        $(Expr(:meta,:inline))
        @fastmath @inbounds begin
            # begin
            $q
            $uq
        end
    end
end



@generated function Base.:*(
            D::LinearAlgebra.Diagonal{T,<:AbstractFixedSizePaddedVector{M,T,P}},
            L::AbstractLowerTriangularMatrix{M,T,N}
        ) where {M,T,P,N}

    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    Wm1 = W - 1
    V = Vec{W,T}
    q = quote end
    # Now, what remains of L is a (M-1) x (M-1) lower triangle
    # we will do W of these at a time.
    reps = M >> Wshift
    rem = M & Wm1

    # handle rem first
    if rem > 0
        # If rem is smaller than half of W, we may use a smaller vector size here
        Wrem = VectorizationBase.pick_vector_width(rem, T)
        full_mask = UInt(2)^Wrem - one(UInt)
        rem_mask_type = VectorizationBase.mask_type(rem)
        # could be typemax(UInt) ???
        # but "unsafe_trunc" says "arbitrary value" is returned if this is greater
        # so seems like this is safer.

        miss = Wrem - rem
        base_ind = - miss
        triangle_ind = base_ind
        increment = M - 1

        initial_mask = Base.unsafe_trunc(rem_mask_type, ((UInt(2))^(miss) - one(UInt)) ⊻ full_mask )
        push!(q.args, quote
            vd = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vD + $base_ind, $initial_mask )
            vl = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL + $base_ind, $initial_mask )
            SIMDPirates.vstore!(vout + $base_ind, SIMDPirates.vmul(vd, vl), $initial_mask)
        end)
        triangle_ind += increment
        increment -= 1
        for r ∈ 1:rem-1
            vl = Symbol(:vl_,r)
            mask = Base.unsafe_trunc(rem_mask_type, ((UInt(2))^(miss+r) - one(UInt)) ⊻ full_mask )
            push!(q.args, quote
                $vl = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL + $triangle_ind, $mask )
                SIMDPirates.vstore!(
                    vout + $triangle_ind,
                    SIMDPirates.vmul(vd, $vl), $mask
                )
            end)
            triangle_ind += increment
            increment -= 1
        end
        base_ind = rem
    else
        base_ind = 0
    end

    # then do reps of W
    full_mask = UInt(2)^W - one(UInt)
    if reps > 0
        rem_quote = quote end
        mask_type = VectorizationBase.mask_type(W)
        for w ∈ 1:Wm1
            vl = Symbol(:vld1_,w)
            mask = Base.unsafe_trunc(mask_type, ((UInt(2))^(w) - one(UInt)) ⊻ full_mask )
            push!(rem_quote.args, quote
                $vl = SIMDPirates.vload(SIMDPirates.Vec{$W,$T}, vL + triangle_ind, $mask)
                SIMDPirates.vstore!(vout + triangle_ind, SIMDPirates.vmul(vd, $vl), $mask)
                triangle_ind += increment
                increment -= 1
            end)
        end

        push!(q.args, quote

            for rep ∈ 0:$(reps-1)
                col1ind = $base_ind + $W * rep
                # load diagonals
                vd = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vD + col1ind )
                vl = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL + col1ind )
                SIMDPirates.vstore!(vout + col1ind, SIMDPirates.vmul(vd, vl))

                triangle_ind = col1ind + $(M-1)
                increment = $(M - 2)

                for r ∈ 0:($(rem-1) + $W*rep)
                    vli = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL + triangle_ind )
                    SIMDPirates.vstore!(vout + triangle_ind, SIMDPirates.vmul(vd, vli))
                    triangle_ind += increment
                    increment -= 1
                end
                $rem_quote
            end
        end)
    end
    quote
        $(Expr(:meta,:inline))
        d = D.diag
        out = MutableLowerTriangularMatrix{$M,$T,$N}(undef)
        vD = VectorizationBase.vectorizable(d)
        vL = VectorizationBase.vectorizable(L)
        vout = VectorizationBase.vectorizable(out)

        GC.@preserve d L out begin
            $q
        end
        out
    end
end

@generated function Base.muladd(
            D::LinearAlgebra.Diagonal{T,<:AbstractFixedSizePaddedVector{M,T,P}},
            L::AbstractLowerTriangularMatrix{M,T,N},
            A::AbstractLowerTriangularMatrix{M,T,N}
        ) where {M,T,P,N}

    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    Wm1 = W - 1
    V = Vec{W,T}
    q = quote end
    # Now, what remains of L is a (M-1) x (M-1) lower triangle
    # we will do W of these at a time.
    reps = M >> Wshift
    rem = M & Wm1

    # handle rem first
    if rem > 0
        # If rem is smaller than half of W, we may use a smaller vector size here
        Wrem = VectorizationBase.pick_vector_width(rem, T)
        full_mask = UInt(2)^Wrem - one(UInt)
        rem_mask_type = VectorizationBase.mask_type(rem)
        # could be typemax(UInt) ???
        # but "unsafe_trunc" says "arbitrary value" is returned if this is greater
        # so seems like this is safer.

        miss = Wrem - rem
        base_ind = - miss
        triangle_ind = base_ind
        increment = M - 1

        initial_mask = Base.unsafe_trunc(rem_mask_type, ((UInt(2))^(miss) - one(UInt)) ⊻ full_mask )
        push!(q.args, quote
            vd = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vD + $base_ind, $initial_mask )
            vl = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL + $base_ind, $initial_mask )
            va = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vA + $base_ind, $initial_mask )
            SIMDPirates.vstore!(vout + $base_ind, SIMDPirates.vmuladd(vd, vl, va), $initial_mask)
        end)
        triangle_ind += increment
        increment -= 1
        for r ∈ 1:rem-1
            vl = Symbol(:vl_,r)
            va = Symbol(:va_,r)
            mask = Base.unsafe_trunc(rem_mask_type, ((UInt(2))^(miss+r) - one(UInt)) ⊻ full_mask )
            push!(q.args, quote
                $vl = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL + $triangle_ind, $mask )
                $va = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vA + $triangle_ind, $mask )
                SIMDPirates.vstore!(
                    vout + $triangle_ind,
                    SIMDPirates.vmuladd(vd, $vl, $va), $mask
                )
            end)
            triangle_ind += increment
            increment -= 1
        end
        base_ind = rem
    else
        base_ind = 0
    end

    # then do reps of W
    full_mask = UInt(2)^W - one(UInt)
    if reps > 0
        rem_quote = quote end
        mask_type = VectorizationBase.mask_type(W)
        for w ∈ 1:Wm1
            vl = Symbol(:vl_,w)
            va = Symbol(:va_,w)
            mask = Base.unsafe_trunc(mask_type, ((UInt(2))^(w) - one(UInt)) ⊻ full_mask )
            push!(rem_quote.args, quote
                $vl = SIMDPirates.vload(SIMDPirates.Vec{$W,$T}, vL + triangle_ind, $mask)
                $va = SIMDPirates.vload(SIMDPirates.Vec{$W,$T}, vA + triangle_ind, $mask)
                SIMDPirates.vstore!(vout + triangle_ind, SIMDPirates.vmuladd(vd, $vl, $va), $mask)
                triangle_ind += increment
                increment -= 1
            end)
        end

        push!(q.args, quote

            for rep ∈ 0:$(reps-1)
                col1ind = $base_ind + $W * rep
                # load diagonals
                vd = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vD + col1ind )
                vl = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL + col1ind )
                va = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vA + col1ind )
                SIMDPirates.vstore!(vout + col1ind, SIMDPirates.vmuladd(vd, vl, va))

                triangle_ind = col1ind + $(M-1)
                increment = $(M - 2)

                for r ∈ 0:($(rem-1) + $W*rep)
                    vli = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL + triangle_ind )
                    vai = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vA + triangle_ind )
                    SIMDPirates.vstore!(vout + triangle_ind, SIMDPirates.vmuladd(vd, vli, vai))
                    triangle_ind += increment
                    increment -= 1
                end
                $rem_quote
            end
        end)
    end
    quote
        $(Expr(:meta,:inline))
        d = D.diag
        out = MutableLowerTriangularMatrix{$M,$T,$N}(undef)
        vD = VectorizationBase.vectorizable(d)
        vL = VectorizationBase.vectorizable(L)
        vA = VectorizationBase.vectorizable(A)
        vout = VectorizationBase.vectorizable(out)

        GC.@preserve d L vA out begin
            $q
        end
        out
    end
end


@generated function row_sum_prod(L1::AbstractLowerTriangularMatrix{M,T,N},L2::AbstractLowerTriangularMatrix{M,T,N}) where {M,T,N}
    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    Wm1 = W - 1
    V = Vec{W,T}
    q = quote end
    # Now, what remains of L is a (M-1) x (M-1) lower triangle
    # we will do W of these at a time.
    reps = M >> Wshift
    rem = M & Wm1

    # handle rem first
    if rem > 0
        # If rem is smaller than half of W, we may use a smaller vector size here
        Wrem = VectorizationBase.pick_vector_width(rem, T)
        full_mask = UInt(2)^Wrem - one(UInt)
        rem_mask_type = VectorizationBase.mask_type(rem)
        # could be typemax(UInt) ???
        # but "unsafe_trunc" says "arbitrary value" is returned if this is greater
        # so seems like this is safer.

        miss = Wrem - rem
        base_ind = - miss
        triangle_ind = base_ind
        increment = M - 1

        initial_mask = Base.unsafe_trunc(rem_mask_type, ((UInt(2))^(miss) - one(UInt)) ⊻ full_mask )
        push!(q.args, quote
            vld1 = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL1 + $base_ind, $initial_mask )
            vld2 = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL2 + $base_ind, $initial_mask )
            vcumulative = SIMDPirates.vmul(vld1, vld2)
        end)
        triangle_ind += increment
        increment -= 1
        for r ∈ 1:rem-1
            vld1 = Symbol(:vld1_,r)
            vld2 = Symbol(:vld2_,r)
            mask = Base.unsafe_trunc(rem_mask_type, ((UInt(2))^(miss+r) - one(UInt)) ⊻ full_mask )
            push!(q.args, quote
                $vld1 = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL1 + $triangle_ind, $mask )
                $vld2 = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL2 + $triangle_ind, $mask )
                vcumulative = SIMDPirates.vmuladd($vld1, $vld2, vcumulative)
            end)
            triangle_ind += increment
            increment -= 1
        end
        push!(q.args, :(SIMDPirates.vstore!(vout + $base_ind, vcumulative, $initial_mask)))
        base_ind = rem
    else
        base_ind = 0
    end

    # then do reps of W
    full_mask = UInt(2)^W - one(UInt)
    if reps > 0
        rem_quote = quote end
        mask_type = VectorizationBase.mask_type(W)
        for w ∈ 1:Wm1
            vld1 = Symbol(:vld1_,w)
            vld2 = Symbol(:vld2_,w)
            mask = Base.unsafe_trunc(mask_type, ((UInt(2))^(w) - one(UInt)) ⊻ full_mask )
            push!(rem_quote.args, quote
                $vld1 = SIMDPirates.vload(SIMDPirates.Vec{$W,$T}, vL1 + triangle_ind, $mask)
                $vld2 = SIMDPirates.vload(SIMDPirates.Vec{$W,$T}, vL2 + triangle_ind, $mask)
                vcumulative = SIMDPirates.vmuladd($vld1, $vld2, vcumulative)
                triangle_ind += increment
                increment -= 1
            end)
        end

        push!(q.args, quote

            for rep ∈ 0:$(reps-1)
                col1ind = $base_ind + $W * rep
                # load diagonals
                vld1 = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL1 + col1ind )
                vld2 = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL2 + col1ind )
                vcumulative = SIMDPirates.vmul(vld1, vld2)

                triangle_ind = col1ind + $(M-1)
                increment = $(M - 2)

                for r ∈ 0:($(rem-1) + $W*rep)
                    vld1 = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL1 + triangle_ind )
                    vld2 = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL2 + triangle_ind )
                    vcumulative = SIMDPirates.vmuladd(vld1, vld2, vcumulative)
                    triangle_ind += increment
                    increment -= 1
                end
                $rem_quote
                SIMDPirates.vstore!(vout + col1ind, vcumulative)
            end
        end)
    end
    quote
        $(Expr(:meta,:inline))
        out = MutableFixedSizePaddedVector{$M,$T}(undef)
        vL1 = VectorizationBase.vectorizable(L1)
        vL2 = VectorizationBase.vectorizable(L2)
        vout = VectorizationBase.vectorizable(out)
        GC.@preserve L1 L2 out begin
            $q
        end
        out
    end
end

@generated function row_sum_prod_add(
        L1::AbstractLowerTriangularMatrix{M,T,N},
        L2::AbstractLowerTriangularMatrix{M,T,N},
        v::AbstractFixedSizePaddedVector{M,T}) where {M,T,N}
    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    Wm1 = W - 1
    V = Vec{W,T}
    q = quote end
    # Now, what remains of L is a (M-1) x (M-1) lower triangle
    # we will do W of these at a time.
    reps = M >> Wshift
    rem = M & Wm1

    # handle rem first
    if rem > 0
        # If rem is smaller than half of W, we may use a smaller vector size here
        Wrem = VectorizationBase.pick_vector_width(rem, T)
        full_mask = UInt(2)^Wrem - one(UInt)
        rem_mask_type = VectorizationBase.mask_type(rem)
        # could be typemax(UInt) ???
        # but "unsafe_trunc" says "arbitrary value" is returned if this is greater
        # so seems like this is safer.

        miss = Wrem - rem
        base_ind = - miss
        triangle_ind = base_ind
        increment = M - 1

        initial_mask = Base.unsafe_trunc(rem_mask_type, ((UInt(2))^(miss) - one(UInt)) ⊻ full_mask )
        push!(q.args, quote
            vld1 = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL1 + $base_ind, $initial_mask )
            vld2 = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL2 + $base_ind, $initial_mask )
            vcumulative = SIMDPirates.vmuladd(vld1, vld2, SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vv + $base_ind, $initial_mask ))
        end)
        triangle_ind += increment
        increment -= 1
        for r ∈ 1:rem-1
            vld1 = Symbol(:vld1_,r)
            vld2 = Symbol(:vld2_,r)
            mask = Base.unsafe_trunc(rem_mask_type, ((UInt(2))^(miss+r) - one(UInt)) ⊻ full_mask )
            push!(q.args, quote
                $vld1 = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL1 + $triangle_ind, $mask )
                $vld2 = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL2 + $triangle_ind, $mask )
                vcumulative = SIMDPirates.vmuladd($vld1, $vld2, vcumulative)
            end)
            triangle_ind += increment
            increment -= 1
        end
        push!(q.args, :(SIMDPirates.vstore!(vout + $base_ind, vcumulative, $initial_mask)))
        base_ind = rem
    else
        base_ind = 0
    end

    # then do reps of W
    full_mask = UInt(2)^W - one(UInt)
    if reps > 0
        rem_quote = quote end
        mask_type = VectorizationBase.mask_type(W)
        for w ∈ 1:Wm1
            vld1 = Symbol(:vld1_,w)
            vld2 = Symbol(:vld2_,w)
            mask = Base.unsafe_trunc(mask_type, ((UInt(2))^(w) - one(UInt)) ⊻ full_mask )
            push!(rem_quote.args, quote
                $vld1 = SIMDPirates.vload(SIMDPirates.Vec{$W,$T}, vL1 + triangle_ind, $mask)
                $vld2 = SIMDPirates.vload(SIMDPirates.Vec{$W,$T}, vL2 + triangle_ind, $mask)
                vcumulative = SIMDPirates.vmuladd($vld1, $vld2, vcumulative)
                triangle_ind += increment
                increment -= 1
            end)
        end

        push!(q.args, quote

            for rep ∈ 0:$(reps-1)
                col1ind = $base_ind + $W * rep
                # load diagonals
                vld1 = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL1 + col1ind )
                vld2 = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL2 + col1ind )
                vcumulative = SIMDPirates.vmuladd(vld1, vld2, SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vv + col1ind))

                triangle_ind = col1ind + $(M-1)
                increment = $(M - 2)

                for r ∈ 0:($(rem-1) + $W*rep)
                    vld1 = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL1 + triangle_ind )
                    vld2 = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL2 + triangle_ind )
                    vcumulative = SIMDPirates.vmuladd(vld1, vld2, vcumulative)
                    triangle_ind += increment
                    increment -= 1
                end
                $rem_quote
                SIMDPirates.vstore!(vout + col1ind, vcumulative)
            end
        end)
    end
    quote
        $(Expr(:meta,:inline))
        out = MutableFixedSizePaddedVector{$M,$T}(undef)
        vv = VectorizationBase.vectorizable(v)
        vL1 = VectorizationBase.vectorizable(L1)
        vL2 = VectorizationBase.vectorizable(L2)
        vout = VectorizationBase.vectorizable(out)
        GC.@preserve v L1 L2 out begin
            $q
        end
        out
    end
end




@generated function Base.:*(
    sp::StackPointer,
    D::LinearAlgebra.Diagonal{T,<:AbstractFixedSizePaddedVector{M,T,P}},
    L::AbstractLowerTriangularMatrix{M,T,N}
) where {M,T,P,N}

    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    V = Vec{W,T}
    Wm1 = W - 1
#    q = quote @show D;  @show L  end
    q = quote end
    # Now, what remains of L is a (M-1) x (M-1) lower triangle
    # we will do W of these at a time.
    reps = M >> Wshift
    rem = M & Wm1

    # handle rem first
    if rem > 0
        # If rem is smaller than half of W, we may use a smaller vector size here
        Wrem = VectorizationBase.pick_vector_width(rem, T)
        full_mask = UInt(2)^Wrem - one(UInt)
        rem_mask_type = VectorizationBase.mask_type(rem)
        # could be typemax(UInt) ???
        # but "unsafe_trunc" says "arbitrary value" is returned if this is greater
        # so seems like this is safer.

        miss = Wrem - rem
        base_ind = - miss
        triangle_ind = base_ind
        increment = M - 1

        initial_mask = Base.unsafe_trunc(rem_mask_type, ((UInt(2))^(miss) - one(UInt)) ⊻ full_mask )
        push!(q.args, quote
            vd = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vD + $base_ind, $initial_mask )
            vl = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL + $base_ind, $initial_mask )
            SIMDPirates.vstore!(vout + $base_ind, SIMDPirates.vmul(vd, vl), $initial_mask)
        end)
        triangle_ind += increment
        increment -= 1
        for r ∈ 1:rem-1
            vl = Symbol(:vl_,r)
            mask = Base.unsafe_trunc(rem_mask_type, ((UInt(2))^(miss+r) - one(UInt)) ⊻ full_mask )
            push!(q.args, quote
                $vl = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL + $triangle_ind, $mask )
                SIMDPirates.vstore!(
                    vout + $triangle_ind,
                    SIMDPirates.vmul(vd, $vl), $mask
                )
            end)
            triangle_ind += increment
            increment -= 1
        end
        base_ind = rem
    else
        base_ind = 0
    end

    # then do reps of W
    full_mask = UInt(2)^W - one(UInt)
    if reps > 0
        rem_quote = quote end
        mask_type = VectorizationBase.mask_type(W)
        for w ∈ 1:Wm1
            vl = Symbol(:vld1_,w)
            mask = Base.unsafe_trunc(mask_type, ((UInt(2))^(w) - one(UInt)) ⊻ full_mask )
            push!(rem_quote.args, quote
                $vl = SIMDPirates.vload(SIMDPirates.Vec{$W,$T}, vL + triangle_ind, $mask)
                SIMDPirates.vstore!(vout + triangle_ind, SIMDPirates.vmul(vd, $vl), $mask)
                triangle_ind += increment
                increment -= 1
            end)
        end

        push!(q.args, quote

            for rep ∈ 0:$(reps-1)
                col1ind = $base_ind + $W * rep
                # load diagonals
                vd = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vD + col1ind )
                vl = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL + col1ind )
                SIMDPirates.vstore!(vout + col1ind, SIMDPirates.vmul(vd, vl))

                triangle_ind = col1ind + $(M-1)
                increment = $(M - 2)

                for r ∈ 0:($(rem-1) + $W*rep)
                    vli = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL + triangle_ind )
                    SIMDPirates.vstore!(vout + triangle_ind, SIMDPirates.vmul(vd, vli))
                    triangle_ind += increment
                    increment -= 1
                end
                $rem_quote
            end
        end)
    end
    quote
#        $(Expr(:meta,:inline))
        d = D.diag
        (sp,out) = PtrLowerTriangularMatrix{$M,$T,$N}(sp)
        vD = VectorizationBase.vectorizable(d)
        vL = VectorizationBase.vectorizable(L)
        vout = VectorizationBase.vectorizable(out)

        GC.@preserve d L begin
            $q
        end
        (sp,out)
    end
end

@generated function Base.muladd(
    sp::StackPointer,
    D::LinearAlgebra.Diagonal{T,<:AbstractFixedSizePaddedVector{M,T,P}},
    L::AbstractLowerTriangularMatrix{M,T,N},
    A::AbstractLowerTriangularMatrix{M,T,N}
) where {M,T,P,N}

    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    Wm1 = W - 1
    V = Vec{W,T}
    q = quote end
    # Now, what remains of L is a (M-1) x (M-1) lower triangle
    # we will do W of these at a time.
    reps = M >> Wshift
    rem = M & Wm1

    # handle rem first
    if rem > 0
        # If rem is smaller than half of W, we may use a smaller vector size here
        Wrem = VectorizationBase.pick_vector_width(rem, T)
        full_mask = UInt(2)^Wrem - one(UInt)
        rem_mask_type = VectorizationBase.mask_type(rem)
        # could be typemax(UInt) ???
        # but "unsafe_trunc" says "arbitrary value" is returned if this is greater
        # so seems like this is safer.

        miss = Wrem - rem
        base_ind = - miss
        triangle_ind = base_ind
        increment = M - 1

        initial_mask = Base.unsafe_trunc(rem_mask_type, ((UInt(2))^(miss) - one(UInt)) ⊻ full_mask )
        push!(q.args, quote
            vd = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vD + $base_ind, $initial_mask )
            vl = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL + $base_ind, $initial_mask )
            va = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vA + $base_ind, $initial_mask )
            SIMDPirates.vstore!(vout + $base_ind, SIMDPirates.vmuladd(vd, vl, va), $initial_mask)
        end)
        triangle_ind += increment
        increment -= 1
        for r ∈ 1:rem-1
            vl = Symbol(:vl_,r)
            va = Symbol(:va_,r)
            mask = Base.unsafe_trunc(rem_mask_type, ((UInt(2))^(miss+r) - one(UInt)) ⊻ full_mask )
            push!(q.args, quote
                $vl = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL + $triangle_ind, $mask )
                $va = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vA + $triangle_ind, $mask )
                SIMDPirates.vstore!(
                    vout + $triangle_ind,
                    SIMDPirates.vmuladd(vd, $vl, $va), $mask
                )
            end)
            triangle_ind += increment
            increment -= 1
        end
        base_ind = rem
    else
        base_ind = 0
    end

    # then do reps of W
    full_mask = UInt(2)^W - one(UInt)
    if reps > 0
        rem_quote = quote end
        mask_type = VectorizationBase.mask_type(W)
        for w ∈ 1:Wm1
            vl = Symbol(:vl_,w)
            va = Symbol(:va_,w)
            mask = Base.unsafe_trunc(mask_type, ((UInt(2))^(w) - one(UInt)) ⊻ full_mask )
            push!(rem_quote.args, quote
                $vl = SIMDPirates.vload(SIMDPirates.Vec{$W,$T}, vL + triangle_ind, $mask)
                $va = SIMDPirates.vload(SIMDPirates.Vec{$W,$T}, vA + triangle_ind, $mask)
                SIMDPirates.vstore!(vout + triangle_ind, SIMDPirates.vmuladd(vd, $vl, $va), $mask)
                triangle_ind += increment
                increment -= 1
            end)
        end

        push!(q.args, quote

            for rep ∈ 0:$(reps-1)
                col1ind = $base_ind + $W * rep
                # load diagonals
                vd = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vD + col1ind )
                vl = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL + col1ind )
                va = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vA + col1ind )
                SIMDPirates.vstore!(vout + col1ind, SIMDPirates.vmuladd(vd, vl, va))

                triangle_ind = col1ind + $(M-1)
                increment = $(M - 2)

                for r ∈ 0:($(rem-1) + $W*rep)
                    vli = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL + triangle_ind )
                    vai = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vA + triangle_ind )
                    SIMDPirates.vstore!(vout + triangle_ind, SIMDPirates.vmuladd(vd, vli, vai))
                    triangle_ind += increment
                    increment -= 1
                end
                $rem_quote
            end
        end)
    end
    quote
#        $(Expr(:meta,:inline))
        d = D.diag
        (sp,out) = PtrLowerTriangularMatrix{$M,$T,$N}(sp)
        vD = VectorizationBase.vectorizable(d)
        vL = VectorizationBase.vectorizable(L)
        vA = VectorizationBase.vectorizable(A)
        vout = VectorizationBase.vectorizable(out)

        GC.@preserve d L vA begin
            $q
        end
        (sp,out)
    end
end


@generated function row_sum_prod(
    sp::StackPointer,
    L1::AbstractLowerTriangularMatrix{M,T,N},
    L2::AbstractLowerTriangularMatrix{M,T,N}
) where {M,T,N}
    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    Wm1 = W - 1
    V = Vec{W,T}
    q = quote end
    # Now, what remains of L is a (M-1) x (M-1) lower triangle
    # we will do W of these at a time.
    reps = M >> Wshift
    rem = M & Wm1

    # handle rem first
    if rem > 0
        # If rem is smaller than half of W, we may use a smaller vector size here
        Wrem = VectorizationBase.pick_vector_width(rem, T)
        full_mask = UInt(2)^Wrem - one(UInt)
        rem_mask_type = VectorizationBase.mask_type(rem)
        # could be typemax(UInt) ???
        # but "unsafe_trunc" says "arbitrary value" is returned if this is greater
        # so seems like this is safer.

        miss = Wrem - rem
        base_ind = - miss
        triangle_ind = base_ind
        increment = M - 1

        initial_mask = Base.unsafe_trunc(rem_mask_type, ((UInt(2))^(miss) - one(UInt)) ⊻ full_mask )
        push!(q.args, quote
            vld1 = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL1 + $base_ind, $initial_mask )
            vld2 = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL2 + $base_ind, $initial_mask )
            vcumulative = SIMDPirates.vmul(vld1, vld2)
        end)
        triangle_ind += increment
        increment -= 1
        for r ∈ 1:rem-1
            vld1 = Symbol(:vld1_,r)
            vld2 = Symbol(:vld2_,r)
            mask = Base.unsafe_trunc(rem_mask_type, ((UInt(2))^(miss+r) - one(UInt)) ⊻ full_mask )
            push!(q.args, quote
                $vld1 = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL1 + $triangle_ind, $mask )
                $vld2 = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL2 + $triangle_ind, $mask )
                vcumulative = SIMDPirates.vmuladd($vld1, $vld2, vcumulative)
            end)
            triangle_ind += increment
            increment -= 1
        end
        push!(q.args, :(SIMDPirates.vstore!(vout + $base_ind, vcumulative, $initial_mask)))
        base_ind = rem
    else
        base_ind = 0
    end

    # then do reps of W
    full_mask = UInt(2)^W - one(UInt)
    if reps > 0
        rem_quote = quote end
        mask_type = VectorizationBase.mask_type(W)
        for w ∈ 1:Wm1
            vld1 = Symbol(:vld1_,w)
            vld2 = Symbol(:vld2_,w)
            mask = Base.unsafe_trunc(mask_type, ((UInt(2))^(w) - one(UInt)) ⊻ full_mask )
            push!(rem_quote.args, quote
                $vld1 = SIMDPirates.vload(SIMDPirates.Vec{$W,$T}, vL1 + triangle_ind, $mask)
                $vld2 = SIMDPirates.vload(SIMDPirates.Vec{$W,$T}, vL2 + triangle_ind, $mask)
                vcumulative = SIMDPirates.vmuladd($vld1, $vld2, vcumulative)
                triangle_ind += increment
                increment -= 1
            end)
        end

        push!(q.args, quote

            for rep ∈ 0:$(reps-1)
                col1ind = $base_ind + $W * rep
                # load diagonals
                vld1 = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL1 + col1ind )
                vld2 = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL2 + col1ind )
                vcumulative = SIMDPirates.vmul(vld1, vld2)

                triangle_ind = col1ind + $(M-1)
                increment = $(M - 2)

                for r ∈ 0:($(rem-1) + $W*rep)
                    vld1 = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL1 + triangle_ind )
                    vld2 = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL2 + triangle_ind )
                    vcumulative = SIMDPirates.vmuladd(vld1, vld2, vcumulative)
                    triangle_ind += increment
                    increment -= 1
                end
                $rem_quote
                SIMDPirates.vstore!(vout + col1ind, vcumulative)
            end
        end)
    end
    quote
#        $(Expr(:meta,:inline))
        (sp,out) = PtrVector{$M,$T}(sp)
        vL1 = VectorizationBase.vectorizable(L1)
        vL2 = VectorizationBase.vectorizable(L2)
        vout = VectorizationBase.vectorizable(out)
        GC.@preserve L1 L2 begin
            $q
        end
        sp,out
    end
end

@generated function row_sum_prod_add(
    sp::StackPointer,
    L1::AbstractLowerTriangularMatrix{M,T,N},
    L2::AbstractLowerTriangularMatrix{M,T,N},
    v::AbstractFixedSizePaddedVector{M,T}
) where {M,T,N}
    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    Wm1 = W - 1
    V = Vec{W,T}
    q = quote end
    # Now, what remains of L is a (M-1) x (M-1) lower triangle
    # we will do W of these at a time.
    reps = M >> Wshift
    rem = M & Wm1

    # handle rem first
    if rem > 0
        # If rem is smaller than half of W, we may use a smaller vector size here
        Wrem = VectorizationBase.pick_vector_width(rem, T)
        full_mask = UInt(2)^Wrem - one(UInt)
        rem_mask_type = VectorizationBase.mask_type(rem)
        # could be typemax(UInt) ???
        # but "unsafe_trunc" says "arbitrary value" is returned if this is greater
        # so seems like this is safer.

        miss = Wrem - rem
        base_ind = - miss
        triangle_ind = base_ind
        increment = M - 1

        initial_mask = Base.unsafe_trunc(rem_mask_type, ((UInt(2))^(miss) - one(UInt)) ⊻ full_mask )
        push!(q.args, quote
            vld1 = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL1 + $base_ind, $initial_mask )
            vld2 = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL2 + $base_ind, $initial_mask )
            vcumulative = SIMDPirates.vmuladd(vld1, vld2, SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vv + $base_ind, $initial_mask ))
        end)
        triangle_ind += increment
        increment -= 1
        for r ∈ 1:rem-1
            vld1 = Symbol(:vld1_,r)
            vld2 = Symbol(:vld2_,r)
            mask = Base.unsafe_trunc(rem_mask_type, ((UInt(2))^(miss+r) - one(UInt)) ⊻ full_mask )
            push!(q.args, quote
                $vld1 = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL1 + $triangle_ind, $mask )
                $vld2 = SIMDPirates.vload(SIMDPirates.Vec{$Wrem, $T}, vL2 + $triangle_ind, $mask )
                vcumulative = SIMDPirates.vmuladd($vld1, $vld2, vcumulative)
            end)
            triangle_ind += increment
            increment -= 1
        end
        push!(q.args, :(SIMDPirates.vstore!(vout + $base_ind, vcumulative, $initial_mask)))
        base_ind = rem
    else
        base_ind = 0
    end

    # then do reps of W
    full_mask = UInt(2)^W - one(UInt)
    if reps > 0
        rem_quote = quote end
        mask_type = VectorizationBase.mask_type(W)
        for w ∈ 1:Wm1
            vld1 = Symbol(:vld1_,w)
            vld2 = Symbol(:vld2_,w)
            mask = Base.unsafe_trunc(mask_type, ((UInt(2))^(w) - one(UInt)) ⊻ full_mask )
            push!(rem_quote.args, quote
                $vld1 = SIMDPirates.vload(SIMDPirates.Vec{$W,$T}, vL1 + triangle_ind, $mask)
                $vld2 = SIMDPirates.vload(SIMDPirates.Vec{$W,$T}, vL2 + triangle_ind, $mask)
                vcumulative = SIMDPirates.vmuladd($vld1, $vld2, vcumulative)
                triangle_ind += increment
                increment -= 1
            end)
        end

        push!(q.args, quote

            for rep ∈ 0:$(reps-1)
                col1ind = $base_ind + $W * rep
                # load diagonals
                vld1 = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL1 + col1ind )
                vld2 = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL2 + col1ind )
                vcumulative = SIMDPirates.vmuladd(vld1, vld2, SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vv + col1ind))

                triangle_ind = col1ind + $(M-1)
                increment = $(M - 2)

                for r ∈ 0:($(rem-1) + $W*rep)
                    vld1 = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL1 + triangle_ind )
                    vld2 = SIMDPirates.vload(SIMDPirates.Vec{$W, $T}, vL2 + triangle_ind )
                    vcumulative = SIMDPirates.vmuladd(vld1, vld2, vcumulative)
                    triangle_ind += increment
                    increment -= 1
                end
                $rem_quote
                SIMDPirates.vstore!(vout + col1ind, vcumulative)
            end
        end)
    end
    quote
        $(Expr(:meta,:inline))
        (sp,out) = PtrVector{$M,$T}(sp)
        vv = VectorizationBase.vectorizable(v)
        vL1 = VectorizationBase.vectorizable(L1)
        vL2 = VectorizationBase.vectorizable(L2)
        vout = VectorizationBase.vectorizable(out)
        GC.@preserve v L1 L2 out begin
            $q
        end
        sp,out
    end
end


function unrolled_update(
    P::Int, W::Int, Wshift::Int, T::DataType;
    xsym::Symbol = :ptrx, xusym::Symbol = :ptrx, Lsym::Symbol = :ptrLdiag, Lusym::Symbol = :ptrLdiag, Ltrisym::Symbol = :ptrLtri, Lutrisym::Symbol = :ptrLtri,
    rem_loop::Bool = true#, xscalar::Bool = false, track_L::Bool = false, track_x::Bool = false
)
    size_T = sizeof(T)
    V = Vec{W,T}
    q = quote end
    for c ∈ (W-P):W-1
        p = W - c
        d_c = Symbol(:d_,c)
        x_c = Symbol(:x_,c)
        c_c = Symbol(:c_,c)
        s_c = Symbol(:s_,c)
        l_c_c = Symbol(:L_,c,:_,c)
        invl_c_c = Symbol(:invL_,c,:_,c)
        push!(q.args, Expr(:(=), x_c, :(VectorizationBase.load($xsym + $c*$size_T))))
        push!(q.args, Expr(:(=), l_c_c, :(VectorizationBase.load($Lsym + $c*$size_T))))
        push!(q.args, Expr(:(=), invl_c_c, :(Base.FastMath.div_fast(one($T), $l_c_c))))
        push!(q.args, Expr(:(=), d_c, macroexpand(Base, :(@fastmath sqrt($l_c_c*$l_c_c + $x_c*$x_c)))))
        push!(q.args, Expr(:(=), c_c, :(SIMDPirates.vbroadcast($V,Base.FastMath.mul_fast($d_c, $invl_c_c)))))
        push!(q.args, Expr(:(=), s_c, :(SIMDPirates.vbroadcast($V,Base.FastMath.mul_fast($x_c, $invl_c_c)))))
        push!(q.args, Expr(:call, :(VectorizationBase.store!), :($Lusym + $c*$size_T), d_c))
        # if track_x
        #     if xscalar
        #         ∂d∂x = Symbol(:∂d_,c,:∂x)
        #         push!(q.args, Expr(:(=), ∂d∂x, :(Base.FastMath.div_fast($x_c, $d_c))))
        #         ∂s∂x = Symbol(:∂s_,c,:∂x)
        #         ∂c∂x = Symbol(:∂c_,c,:∂x)
        #         push!(q.args, Expr(:(=), ∂c∂x, :(Base.FastMath.mul_fast($∂d∂x, $invl_c_c))))
        #         push!(q.args, Expr(:(=), ∂s∂x, invl_c_c))
        #         push!(q.args, :(VectorizationBase.store!($(Symbol(:∂,xsym)) + $c*$size_T), VectorizationBase.load($(Symbol(:∂,xsym,:∂a)) + $c*$size_T) * ∂d∂x))
        #     else
        #         throw("xscalar == $xscalar while track_x == $track_x is not yet supported.")
        #     end
        # end
        # if track_L
        #     ∂d∂Lcc = Symbol(:∂d∂L_,p,:_,p)
        #     push!(q.args, Expr(:(=), ∂d∂Lcc, :(Base.FastMath.div_fast($l_c_c, $d_c))))
        #     ∂invL∂L = Symbol(:∂invL_,p,:_,p,:∂L_,p,:_,p)
        #     push!(q.args, Expr(:(=), ∂invL∂L, :(Base.FastMath.mul_fast(-one($T), $invl_c_c, $invl_c_c))))
        #     ∂s∂Lcc = Symbol(:∂s_,c,:∂L_,c,:_,c)
        #     ∂c∂Lcc = Symbol(:∂c_,c,:∂L_,c,:_,c)
        #     push!(q.args, Expr(:(=), ∂s∂Lcc, :(Base.FastMath.mul_fast($x_c, $∂invL∂L))))
        #     push!(q.args, Expr(:(=), ∂c∂Lcc, :(Base.FastMath.add_fast(Base.FastMath.mul_fast($d_c, $∂invL∂L), Base.FastMath.mul_fast($∂d∂Lcc,$invl_c_c)))))
        # end
        if p > 1
            rem = p - 1
            mask = ~ VectorizationBase.mask(T, c+1)
            vL = Symbol(:vL_,c+1,:_,c)
            vLu = Symbol(:vLu_,c+1,:_,c)
            vx = Symbol(:vL_,c+1)
            push!(q.args, Expr(:(=), vL, :(SIMDPirates.vload($V,$Ltrisym + trioffset*$size_T, $mask))))
            push!(q.args, Expr(:(=), vx, :(SIMDPirates.vload($V,$xsym, $mask))))
            push!(q.args, Expr(:(=), vLu, :(SIMDPirates.vfdiv(SIMDPirates.vadd($vL, SIMDPirates.vmul($s_c, $vx)), $c_c))))
            push!(q.args, :(SIMDPirates.vstore!($Lutrisym + trioffset*$size_T, $vLu,$mask)))
            push!(q.args, :(SIMDPirates.vstore!($xusym, SIMDPirates.vsub(SIMDPirates.vmul($c_c,$vx), SIMDPirates.vmul($s_c, $vLu)), $mask)))
        end
        if rem_loop
            rem_quote = quote
                for rep ∈ 1:repetitions
                    vL_j_p = SIMDPirates.vload($V, $Ltrisym + $size_T*trioffset + $size_T*$W*rep)
                    vx_j = SIMDPirates.vload($V, $xsym + $size_T*$W*rep)
                    vLu_j_p = SIMDPirates.vfdiv(SIMDPirates.vadd(vL_j_p, SIMDPirates.vmul($s_c,vx_j)), $c_c)
                    SIMDPirates.vstore!($Lutrisym + $size_T*trioffset + $size_T*$W*rep, vLu_j_p)
                    SIMDPirates.vstore!($xsym + $size_T*$W*rep, SIMDPirates.vsub(SIMDPirates.vmul($c_c,vx_j), SIMDPirates.vmul($s_c,vLu_j_p)))
                end
            end
            push!(q.args, rem_quote)
        end
        if p > 1
            push!(q.args, :(trioffset += triincrement))
        else
            push!(q.args, :(trioffset += triincrement+$W))
        end
        push!(q.args, :(triincrement -= 1))
    end
    push!(q.args, :(repetitions -= 1; $xsym += $W * $size_T; $Lsym += $W*$size_T;))
    Lusym == Lsym || push!(q.args, :($Lusym += $W*$size_T))
    xusym == xsym || push!(q.args, :($xusym += $W*$size_T))
    q
end

function rank_one_updated_lower_triangle_quote(P, T; xsym = :x, xusym = :x, Lsym = :L, Lusym = :L)
    W, Wshift = VectorizationBase.pick_vector_width_shift(P-1,T) # Even with avx512, we'd want W=4 for P=5, or W=2 for P=3,etc.
    Wm1 = W - 1
    V = Vec{W,T}
    reps = P >> Wshift
    rem = P & Wm1
    size_T = sizeof(T)
    xsymp = Symbol(:ptr, xsym)
    xusymp = Symbol(:ptr, xusym)
    Lsymp = Symbol(:ptr, Lsym,:diag)
    Lusymp = Symbol(:ptr, Lusym,:diag)
    Ltrisymp = Symbol(:ptr, Lsym,:tri)
    Lutrisymp = Symbol(:ptr, Lusym,:tri)
    # We loop over batches of 8
    initial_offset = rem == 0 ? 0 : W - rem
    q = quote
        triincrement = $(P - 2)
        trioffset = 0
        repetitions = $(reps + (rem > 0) - 1)
        $xsymp = pointer($xsym) - $size_T * $initial_offset
        $Lsymp = pointer($Lsym) - $size_T * $initial_offset
        $Ltrisymp = $Lsymp + $size_T * $(P-1)
    end
    if xsym != xusym
        push!(q.args, :($xusymp = pointer($xusym) - $size_T * $initial_offset))
    end
    if Lsym != Lusym
        push!(q.args, :($Lusymp = pointer($Lusym) - $size_T * $initial_offset; $Lutrisymp = $Lusymp + $size_T * $(P-1)))
    end
    if rem > 0
        push!(q.args, unrolled_update(rem, W, Wshift, T, xsym = xsymp, xusym = xusymp, Lsym = Lsymp, Lusym = Lusymp, Ltrisym = Ltrisymp, Lutrisym = Lutrisymp, rem_loop = reps > 0))
    end
    if reps == 1
        push!(q.args, unrolled_update(W, W, Wshift, T, xsym = xsymp, xusym = xusymp, Lsym = Lsymp, Lusym = Lusymp, Ltrisym = Ltrisymp, Lutrisym = Lutrisymp, rem_loop = false))
    elseif reps > 1
        loop_quote = quote
            for _ ∈ 1:$reps
                $(unrolled_update(W, W, Wshift, T, xsym = xsymp, xusym = xusymp, Lsym = Lsymp, Lusym = Lusymp, Ltrisym = Ltrisymp, Lutrisym = Lutrisymp, rem_loop = true))
            end
        end
        push!(q.args, loop_quote)
    end
    push!(q.args, nothing)
    q
end

@generated function rank_update!(
    Lu::AbstractMutableLowerTriangularMatrix{P,T},
    L::AbstractMutableLowerTriangularMatrix{P,T},
    a::Union{<:PaddedMatrices.AbstractMutableFixedSizePaddedVector{P,T},T}
) where {T,P}
    quote
        x = MutableFixedSizePaddedVector{$P,$T}(undef)
        x .= a
        $(rank_one_updated_lower_triangle_quote(P,T,Lsym = :L, Lusym = :Lu))
        Lu
    end
end
@generated function rank_update!(
    sptr::StackPointer,
    Lu::AbstractMutableLowerTriangularMatrix{P,T},
    L::AbstractMutableLowerTriangularMatrix{P,T},
    a::Union{<:PaddedMatrices.AbstractMutableFixedSizePaddedVector{P,T},T}
) where {T,P}
    quote
        x = PtrVector{$P,$T}(pointer(sptr,$T))
        x .= a
        $(rank_one_updated_lower_triangle_quote(P,T,Lsym = :L, Lusym = :Lu))
        Lu
    end
end
function rank_update(L::AbstractLowerTriangularMatrix{P,T}, a) where {P,T}
    Lu = MutableLowerTriangularMatrix{P,T}(undef)
    rank_update!(Lu, L, a)
end
@generated function rank_update(sptr::StackPointer, L::AbstractLowerTriangularMatrix{P,T,N}, a) where {P,T,N}
    ptr_offset = VectorizationBase.align(N*sizeof(T))
    N2 = ptr_offset ÷ sizeof(T)
    quote
        Lu = PtrLowerTriangularMatrix{$P,$T,$N2}(pointer(sptr,$T))
#        rank_update!(sptr + $(VectorizationBase.align(N*sizeof(T))), Lu, L, a)
        sptr + $ptr_offset, rank_update!(Lu, L, a)
    end
end

function phi_at_b_quote(P, T, load_A = true, store_C = true, halve_diagonal = true)
    W = VectorizationBase.pick_vector_width(P,T)
    if P > 4W && store_C && load_A
        return quote
            increment = $P
            @inbounds @fastmath for c in 1:$P
                for r in 1:c-1
                    C[r,c] = zero($T)
                end
                C_c_c = A[c] * B[c]
                for d in 1:$P-c
                    C_c_c += A[increment + d] * B[increment + d]
                end
                C[c,c] = $(halve_diagonal ? :($(T(0.5))*C_c_c) : :C_c_c)
                incr_r = increment + $P - c
                for r in 1+c:$P
                    C_r_c = A[r] * B[increment+r-c]
                    # @show C_r_c
                    incr_r -= r
                    for i in r+1:$P
                        # @show r,c, (incr_r+i), (increment+i-c)
                        C_r_c += A[incr_r + i] * B[increment+i-c]
                    end
                    incr_r += $P
                    C[r,c] = C_r_c
                end
                increment += $P - c
            end
            C
        end
    end
    q = quote end
    increment = P
    for c in 1:P
        if store_C
            for r in 1:c-1
                push!(q.args, :($(Symbol(:C_,r,:_,c)) = zero($T)))
            end
        end
        C_c_c = Symbol(:C_,c,:_,c)
        load_A ? push!(q.args, :($C_c_c = A[$c] * B[$c])) : push!(q.args, :($C_c_c = $(Symbol(:A_,c,:_,c)) * B[$c]))
        if load_A
            for d in 1:P-c
                push!(q.args, :($C_c_c += A[$(increment + d)] * B[$(increment + d)]))
            end
        else
            for d in 1:P-c
                push!(q.args, :($C_c_c += $(Symbol(:A_,c+d,:_,c)) * B[$(increment + d)]))
            end
        end
        halve_diagonal && push!(q.args, :($C_c_c *= $(T(0.5))))
        incr_r = increment + P - c
        for r in 1+c:P
            if load_A
                push!(q.args, :($(Symbol(:C_,r,:_,c)) = A[$r] * B[$(increment+r-c)]))
                incr_r -= r
                for i in r+1:P
                    push!(q.args, :($(Symbol(:C_,r,:_,c)) += A[$(incr_r + i)] * B[$(increment+i-c)]))
                end
                incr_r += P
            else
                push!(q.args, :($(Symbol(:C_,r,:_,c)) = $(Symbol(:A_,r,:_,r)) * B[$(increment+r-c)]))
                for i in r+1:P
                    push!(q.args, :($(Symbol(:C_,r,:_,c)) += $(Symbol(:A_,i,:_,r)) * B[$(increment+i-c)]))
                end
            end
        end
        increment += P - c
        if store_C
            for r in 1:P
                push!(q.args, :(C[$r,$c] = $(Symbol(:C_,r,:_,c))))
            end
        end
    end
    q
end

@generated function PhiAtB!(
    C::PaddedMatrices.AbstractMutableFixedSizePaddedMatrix{P,P,T},
    A::AbstractMutableLowerTriangularMatrix{P,T}, # original chol
    B::AbstractMutableLowerTriangularMatrix{P,T} # Adjoint
) where {P,T}
    q = phi_at_b_quote(P, T)
    quote
        @fastmath @inbounds begin
            $q
        end
        C
    end
end

function reverse_chol_grad_expr(P, T; store_S::Bool = true, halve_diagonal = true)
    q = quote end
    inc = P
    for c in 1:P
        push!(q.args, Expr(:(=), Symbol(:A_,c,:_,c), :(A[$c])))
        push!(q.args, Expr(:(=), Symbol(:invA_,c,:_,c), :(Base.FastMath.div_fast(one($T),$(Symbol(:A_,c,:_,c))))))
        for r in c+1:P
            inc += 1
            push!(q.args, Expr(:(=), Symbol(:A_,r,:_,c), :(A[$inc])))
        end
    end
    push!(q.args, phi_at_b_quote(P, T, false, false, false)) # We treat as a reflected Symmetric matrix.
    # Now, we must calc A' \ C / A = (C / A)' / A
    for _c_ in 0:P-1
        c = P - _c_
        for r in 1:P
            X_r_c = Symbol(:X_,c,:_,r) # we transpose X; hence Symbol(:X_,c,:_,r) instead of _r_c
            clt, rlt = minmax(r, c)
            C_r_c = Symbol(:C_,rlt,:_,clt)
            push!(q.args, :($X_r_c = $C_r_c))
            for j in c+1:P
                clt, rlt = minmax(r, j)
                X_r_j = Symbol(:X_,j,:_,r)
                A_j_c = Symbol(:A_,j,:_,c)
                push!(q.args, :($X_r_c -= $X_r_j * $A_j_c ))
            end
            push!(q.args, :($X_r_c *= $(Symbol(:invA_,c,:_,c))))
        end
    end
    # Second division.
    for _c_ in 0:P-1
        c = P - _c_
        for r in 1:c
            X_r_c = Symbol(:X_,r,:_,c)
            S_r_c = Symbol(:S_,c,:_,r)
            push!(q.args, :($S_r_c = $X_r_c))
            for j in c+1:P
                S_r_j = Symbol(:S_,j,:_,r)
                A_j_c = Symbol(:A_,j,:_,c)
                push!(q.args, :($S_r_c -= $S_r_j * $A_j_c ))
            end
            push!(q.args, :($S_r_c *= $(Symbol(:invA_,c,:_,c))))
        end
    end
    if store_S
        inc = P
        for c in 1:P
            if halve_diagonal
                push!(q.args, Expr(:(=), :(S[$c]), Expr(:call, :(*), T(0.5), Symbol(:S_,c,:_,c))))
            else
                push!(q.args, Expr(:(=), :(S[$c]), Symbol(:S_,c,:_,c)))
            end
            for r in c+1:P
                inc += 1
                push!(q.args, Expr(:(=), :(S[$inc]), Symbol(:S_,r,:_,c)))
            end
        end
    end
    q
end

@generated function reverse_cholesky_grad!(
    S::AbstractMutableLowerTriangularMatrix{P,T},
    A::AbstractMutableLowerTriangularMatrix{P,T}, # original chol
    B::AbstractMutableLowerTriangularMatrix{P,T} # Adjoint
) where {P,T}
# ) where {T,P}
    q = reverse_chol_grad_expr(P, T, halve_diagonal = false)
    quote
        @fastmath @inbounds begin
            $q
        end
        S
    end
end
function reverse_cholesky_grad(
    A::AbstractMutableLowerTriangularMatrix{P,T}, # original chol
    B::AbstractMutableLowerTriangularMatrix{P,T} # Adjoint
) where {P,T}
    S = MutableLowerTriangularMatrix{P,T}(undef)
    reverse_cholesky_grad!(S, A, B)
end
@generated function reverse_cholesky_grad(
    sptr::StackPointer,
    A::AbstractMutableLowerTriangularMatrix{P,T}, # original chol
    B::AbstractMutableLowerTriangularMatrix{P,T} # Adjoint
) where {P,T}
    Wm1 = VectorizationBase.pick_vector_width(T) - 1
    N = (binomial2(P+1) + Wm1) & ~Wm1
    quote
        S = PtrLowerTriangularMatrix{$P,$T,$N}(pointer(sptr,$T))
        sptr + $(sizeof(T)) * $N, spreverse_cholesky_grad!(S, A, B)
    end
end


struct RankUpdateAdjoint{P,T,L,A<:AbstractMutableLowerTriangularMatrix{P,T,L}}
    data::A
end

function ∂rank_update_quote(P, T; track_L::Bool, track_x::Bool, xscalar::Bool = false, store_S::Bool = false)
    (track_L || track_x) || return quote end
    q = reverse_chol_grad_expr(P, T, store_S = store_S)
    if track_L
        ### for L
        # if s isa Number
        #     sv = fill!(MutableFixedSizePaddedVector{8,Float64}(undef), s);
        # else
        #     sv = s
        # end
        # Lu = StructuredMatrices.rank_update(L1, sv)
        # StructuredMatrices.reverse_cholesky_grad!(S, Lu, Ladjoint)
        # LowerTriangular(Symmetric(Array(S),:L)*L1)
        for k ∈ 1:P
            for c ∈ 1:k
                assignment = k == c ? :(=) : :(+=)
                L_k_c = Symbol(:L_,k,:_,c)
                if k == c
                    push!(q.args, Expr(:(=), L_k_c, :(Linput[$k])))
                else
                    push!(q.args, Expr(:(=), L_k_c, :(Linput[$(lt_sub2ind_fast(P,k,c))])))
                end
                for r ∈ c:P
                    pL_r_c = Symbol(:∂L_,r,:_,c)
                    cs, rs = minmax(r, k)
                    S_r_k = Symbol(:S_,rs,:_,cs)
                    push!(q.args, Expr(assignment, pL_r_c, :($S_r_k * $L_k_c)))
                end
            end
        end
        inc = P
        for c ∈ 1:P
            push!(q.args, :(∂L[$c] = $(Symbol(:∂L_,c,:_,c))))
            for r ∈ c+1:P
                pL_r_c = Symbol(:∂L_,r,:_,c)
                inc += 1
                push!(q.args, :(∂L[$inc] = $pL_r_c))
            end
        end
    end
    if track_x
        ### for xscalar
        # x2p3 = fill!(MutableFixedSizePaddedVector{8,Float64}(undef), s);
        # Lu = StructuredMatrices.rank_update(L1, x2p3)
        # StructuredMatrices.reverse_cholesky_grad!(S, Lu, Ladjoint)
        # sumSsub = zero(eltype(S))
        # sumSdiag = zero(eltype(S))
        # N = size(S,2)
        # for n ∈ 1:N
        #     sumSdiag += S[n,n]
        # end
        # for c ∈ 1:N
        #     for r ∈ c+1:N
        #         sumSsub += S[r,c]
        #     end
        # end
        # s*(sumSdiag + 2sumSsub)
        W, Wshift = VectorizationBase.pick_vector_width_shift(P,T)
        Wm1 = W - 1
        indsumsub = 0
        if xscalar
            for c ∈ 1:P
                sumSdiag = Symbol(:sumdiag_, (c-1) & Wm1)
                assigment = ( ( (c-1) >> Wshift ) == 0 ) ? :(=) : :(+=)
                push!(q.args, Expr(assigment, sumSdiag, Symbol(:S_,c,:_,c)))
                for r ∈ c+1:P
                    sumSsub = Symbol(:sumsub_, indsumsub & Wm1)
                    assigment = ( ( indsumsub >> Wshift ) == 0 ) ? :(=) : :(+=)
                    push!(q.args, Expr(assigment, sumSsub, Symbol(:S_,r,:_,c)))
#                    push!(q.args, :(@show $(string(assigment)), $sumSsub, $(Symbol(:S_,r,:_,c))))
                    indsumsub += 1
                end
            end
            WLd = min(Wm1, P-1)
            WLs = min(Wm1, indsumsub-1)
            push!(q.args, Expr(:(=), :sumSdiag, Expr(:call, :(+), [Symbol(:sumdiag_,i) for i ∈ 0:WLd]...)))
            push!(q.args, Expr(:(=), :sumSsub, Expr(:call, :(+), [Symbol(:sumsub_,i) for i ∈ 0:WLs]...)))
#            push!(q.args, :(@show sumSdiag, sumSsub))
            push!(q.args, Expr(:(=), :∂x, :(xinput * (sumSdiag + $(T(2))*sumSsub))))
        ### for xvector
        # Lu = StructuredMatrices.rank_update(L1, s)
        # StructuredMatrices.reverse_cholesky_grad!(S, Lu, Ladjoint)
        # grad = similar(s)
        # for i ∈ eachindex(grad)
        #     g = zero(eltype(grad))
        #     for j ∈ 1:i-1
        #         g += S[i,j] * s[j]
        #     end
        #     g += S[i,i] * s[i]
        #     for j ∈ i+1:length(grad)
        #         g += S[j,i] * s[j]
        #     end
        #     grad[i] = g
        # end
        # grad
        else
            for k in 1:P
                assignment = k == 1 ? :(=) : :(+=)
                push!(q.args, :(x_k = xinput[$k]))
                for i in 1:P
                    cs, rs = minmax(i,k)
                    S_i_k = Symbol(:S_,rs,:_,cs)
                    push!(q.args, Expr(assignment, Symbol(:x_,i), :($S_i_k * x_k)))
                end
            end
            for i in 1:P
                push!(q.args, :(∂x[$i] = $(Symbol(:x_,i))))
            end
        end
    end
    q
end

@generated function ∂rank_update(
    sptr::StackPointer,
    A::AbstractMutableLowerTriangularMatrix{P,T}, # original output argument
    B::AbstractMutableLowerTriangularMatrix{P,T}, # adjoint
    Linput::AbstractMutableLowerTriangularMatrix{P,T}, # original input argument
    xinput::T # original input argument
#) where {P,T}
) where {T,P}
    q = ∂rank_update_quote(P, T, track_L = true, track_x = true, xscalar = true)
    PL = VectorizationBase.align(binomial2(P+1), T)
    sptroffset = sizeof(T) * PL
    quote
        ∂L = PtrLowerTriangularMatrix{$P,$T,$PL}(pointer(sptr,$T))
        #∂x = MutableFixedSizePaddedVector{$P,$T}(undef)
        @inbounds @fastmath begin
            $q
        end
        sptr + $sptroffset, (∂L, ∂x)
    end
end
@generated function ∂rank_update(
    A::AbstractMutableLowerTriangularMatrix{P,T}, # original output argument
    B::AbstractMutableLowerTriangularMatrix{P,T}, # adjoint
    Linput::AbstractMutableLowerTriangularMatrix{P,T}, # original input argument
    xinput::T # original input argument
#) where {P,T}
) where {T,P}
    q = ∂rank_update_quote(P, T, track_L = true, track_x = true, xscalar = true)
    quote
        ∂L = MutableLowerTriangularMatrix{$P,$T}(undef)
        #∂x = MutableFixedSizePaddedVector{$P,$T}(undef)
        @inbounds @fastmath begin
            $q
        end
        ∂L, ∂x
    end
end

@generated function ∂rank_update(
    sptr::StackPointer,
    A::AbstractMutableLowerTriangularMatrix{P,T}, # original output argument
    B::AbstractMutableLowerTriangularMatrix{P,T}, # adjoint
    Linput::AbstractMutableLowerTriangularMatrix{P,T}, # original input argument
    xinput::PaddedMatrices.AbstractMutableFixedSizePaddedVector{P,T} # original input argument
) where {P,T}
    q = ∂rank_update_quote(P, T, track_L = true, track_x = true, xscalar = false)
    PL = VectorizationBase.align(binomial2(P+1), T)
    sptroffset1 = sizeof(T) * PL
    sptroffset2 = sptroffset1 + VectorizationBase.align(P*sizeof(T))
    quote
        _sptr = pointer(sptr,$T)
        ∂L = PtrLowerTriangularMatrix{$P,$T,$PL}(_sptr)
        ∂x = PtrVector{$P,$T}(_sptr + $sptroffset1)
        @inbounds @fastmath begin
            $q
        end
        sptr + $sptroffset2, (∂L, ∂x)
    end
end
@generated function ∂rank_update(
    A::AbstractMutableLowerTriangularMatrix{P,T}, # original output argument
    B::AbstractMutableLowerTriangularMatrix{P,T}, # adjoint
    Linput::AbstractMutableLowerTriangularMatrix{P,T}, # original input argument
    xinput::PaddedMatrices.AbstractMutableFixedSizePaddedVector{P,T} # original input argument
) where {P,T}
# ) where {T,P}
    q = ∂rank_update_quote(P, T, track_L = true, track_x = true, xscalar = false)#, store_S = true)
    quote
        ∂L = MutableLowerTriangularMatrix{$P,$T}(undef)
        ∂x = MutableFixedSizePaddedVector{$P,$T}(undef)
        # S = MutableLowerTriangularMatrix{$P,$T}(undef)
        @inbounds @fastmath begin
            $q
        end
        # S, ∂L, ∂x
        ∂L, ∂x
    end
end

@generated function ∂rank_update(
    sptr::StackPointer,
    A::AbstractMutableLowerTriangularMatrix{P,T}, # original output argument
    B::AbstractMutableLowerTriangularMatrix{P,T}, # adjoint
    xinput::T  # original input argument
) where {P,T}
    q = ∂rank_update_quote(P, T, track_L = false, track_x = true, xscalar = true)
    quote
        @inbounds @fastmath begin
            $q
        end
        sptr, ∂x
    end
end
@generated function ∂rank_update(
    A::AbstractMutableLowerTriangularMatrix{P,T}, # original output argument
    B::AbstractMutableLowerTriangularMatrix{P,T}, # adjoint
    xinput::T # original input argument
) where {P,T}
    q = ∂rank_update_quote(P, T, track_L = false, track_x = true, xscalar = true)
    quote
        @inbounds @fastmath begin
            $q
        end
        ∂x
    end
end

@generated function ∂rank_update(
    sptr::StackPointer,
    A::AbstractMutableLowerTriangularMatrix{P,T}, # original output argument
    B::AbstractMutableLowerTriangularMatrix{P,T}, # adjoint
    xinput::PaddedMatrices.AbstractMutableFixedSizePaddedVector{P,T} # original input argument
) where {P,T}
    q = ∂rank_update_quote(P, T, track_L = false, track_x = true, xscalar = false)
    sptroffset1 = VectorizationBase.align(P*sizeof(T))
    quote
        ∂x = PtrVector{$P,$T}(pointer(sptr,$T))
        @inbounds @fastmath begin
            $q
        end
        sptr + $sptroffset1, ∂x
    end
end
@generated function ∂rank_update(
    A::AbstractMutableLowerTriangularMatrix{P,T}, # original output argument
    B::AbstractMutableLowerTriangularMatrix{P,T}, # adjoint
    xinput::PaddedMatrices.AbstractMutableFixedSizePaddedVector{P,T} # original input argument
) where {P,T}
    q = ∂rank_update_quote(P, T, track_L = false, track_x = true, xscalar = false)
    quote
        ∂x = MutableFixedSizePaddedVector{$P,$T}(undef)
        @inbounds @fastmath begin
            $q
        end
        ∂x
    end
end
@generated function ∂rank_update(
    sptr::StackPointer,
    A::AbstractMutableLowerTriangularMatrix{P,T}, # original output argument
    B::AbstractMutableLowerTriangularMatrix{P,T}, # adjoint
    Linput::AbstractMutableLowerTriangularMatrix{P,T} # original input argument
) where {P,T}
    q = ∂rank_update_quote(P, T, track_L = true, track_x = false)
    PL = VectorizationBase.align(binomial2(P+1), T)
    quote
        ∂L = PtrLowerTriangularMatrix{$P,$T,$PL}(pointer(sptr,$T))
        @inbounds @fastmath begin
            $q
        end
        sptr + $(PL*sizeof(T)), ∂L
    end
end
@generated function ∂rank_update(
    A::AbstractMutableLowerTriangularMatrix{P,T}, # original output argument
    B::AbstractMutableLowerTriangularMatrix{P,T}, # adjoint
    Linput::AbstractMutableLowerTriangularMatrix{P,T} # original input argument
) where {P,T}
    q = ∂rank_update_quote(P, T, track_L = true, track_x = false)
    quote
        ∂L = MutableLowerTriangularMatrix{$P,$T}(undef)
        @inbounds @fastmath begin
            $q
        end
        ∂L
    end
end


# PaddedMatrices.@support_stack_pointer ∂rank_update!


#@generated function rank_update()
#
#end
