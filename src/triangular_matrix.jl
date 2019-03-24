
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
        @vectorize for i ∈ 1:$P
            out += log(A[i])
        end
        out
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

