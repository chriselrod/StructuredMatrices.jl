
@inline function Base.getindex(S::AbstractLowerTriangularMatrix{P,T,L}, i, j) where {P,T,L}
    @boundscheck i > P && ThrowBoundsError("i == $i > $P")
    j > i && return zero(T)
    @inbounds S.data[lt_sub2ind(P, i, j)]
end

@inline function Base.getindex(S::AbstractUpperTriangularMatrix{P,T,L}, i, j) where {P,T,L}
    i > j && return zero(T)
    @boundscheck j > P && ThrowBoundsError("j == $j > $P.")
    @inbounds S.data[ut_sub2ind(P, i, j)]
end
@inline function Base.getindex(S::LinearAlgebra.Adjoint{Union{},<:AbstractUpperTriangularMatrix{P,Vec{W,T},L}}, i::Int, j::Int) where {P,T,L,W}
    i < j && return zero(T)
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
@generated function LinearAlgebra.logdet(A::AbstractTriangularMatrix{P,T,L}) where {P,T,L}
    quote
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
