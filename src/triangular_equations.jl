
###
### We are solving for A in
### B = A * L
### or
### B = A * U
### where L is lower triangular and U is upper triangular.
### That is, we're solving the system of triangular equations
### A = B / L
### or
### A = B / U
### Our strategy for doing so involves solving for blocks
### from A at a time, iterating through earlier/later
### columns to update when solving  U/K
### A[:,k] = ( B[:,k] - ∑ⱼ₌ₖ₊₁ᴷ A[:,j] * L[j,k] ) / L[k,k]
### or
### A[:,k] = ( B[:,k] - ∑ⱼ₌₁ᵏ⁻¹ A[:,j] * U[j,k] ) / U[k,k]
### eg a 2vec x 3 block from each will allow
### 


# Ustride
# if ilL′

# K is number of earlier iterations
function A_rdiv_U_kernel_quote(
    R, C, K::Union{Symbol,Integer}, ::Type{T},
    Astride, Bstride, Ustride, isL′, invdiagptr,
    Bsym = :ptrB, Asym = :ptrA, Utrisym = :ptrUtri, Udiagsym = :ptrUdiag,
    maskload = true, loadB = true, storeA = true
) where {T}
    # K is either a symbol or integer, indicating number of preceding columns
    size_T = sizeof(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(R, T)
    Wm1 = W - 1
    Riter = R >> Wshift
    Rrem = R & Wm1
    mask = VectorizationBase.mask_from_remainder(T, Rrem)
    if loadB
        if K isa Symbol
            q = quote
                BsymK = $Bsym + $(size_T*Bstride)*$K
                AsymK = $Asym + $(size_T*Astride)*$K
            end
        else
            q = quote
                BsymK = $Bsym + $(size_T*Bstride*K)
                AsymK = $Asym + $(size_T*Astride*K)
            end
        end
        for c ∈ 0:C-1
            for r ∈ 0:Riter-1
                push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vload(Vec{$W,$T}, BsymK + $(size_T * (r*W + c*Bstride)) ) ))
            end
            if Rrem > 0
                # Only need to mask if we're on last column
                if maskload && c == C-1
                    push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vload(Vec{$W,$T}, BsymK + $(size_T * (Riter*W + c*Bstride)), $mask ) ))
                else
                    push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vload(Vec{$W,$T}, BsymK + $(size_T * (Riter*W + c*Bstride)) ) ))
                end
            end
        end
    end
    if !isL′
        coff = Ustride
        for c ∈ 0:C-1
            push!(q.args, Expr(:(=), Symbol(:Uoffset_,c), coff*size_T))
            coff += c
        end
    end
    # We don't need to mask these loads.
    Riterl = Rrem > 0 ? Riter : Riter-1
    # Updating based on all earlier columns
    if K isa Symbol || K > 0
        loopbody = quote end
        for r ∈ 0:Riterl
            push!(loopbody.args, :($(Symbol(:A_,r,:_j)) = SIMDPirates.vload(Vec{$W,$T}, $Asym + j*$(Astride*size_T) + $(r*W*size_T))))
        end
        if isL′
            push!(loopbody.args, :($Utrisym += $Ustride - j))
        end
        for c ∈ 0:C-1
            if isL′ # U is actually a transposed lower triangular matrix
                # c increase corresponds row increase
                push!(loopbody.args, :($(Symbol(:U_j_,c)) = SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Utrisym + $(c*size_T)))))
            else
                push!(loopbody.args, :($(Symbol(:U_j_,c)) = SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Utrisym + $(Symbol(:Uoffset_,c)) + (j*$size_T)))))
            end
            for r ∈ 0:Riterl
                push!(loopbody.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vfnmadd($(Symbol(:A_,r,:_j)), $(Symbol(:U_j_,c)), $(Symbol(:A_,r,:_,c)))))
            end
        end
        mainloop = quote
            @inbounds for j ∈ 0:$(K isa Symbol ? Expr(:call, :(-), K, 1) : K-1)
                $loopbody
            end
        end
        push!(q.args, mainloop)
    end

    # Update using just-calculated block
    scaleop = invdiagptr ? :vmul : :vfdiv
    if isL′
        trioff = 0
    end
    for j ∈ 0:C-1
        vUjj = Symbol(:vU_,j,:_,j)
        push!(q.args, Expr(:(=), vUjj, :(SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Udiagsym + $(size_T*j))))))
        for r ∈ 0:Riterl
            push!(q.args, :($(Symbol(:A_,r,:_,j)) = SIMDPirates.$scaleop($(Symbol(:A_,r,:_,j)), $vUjj)))
        end
        if isL′ && j < C-1 && j > 0
            if K isa Symbol
                push!(q.args, :($Utrisym += $size_T*($(Ustride - j) - $K)))
            else
                trioff += size_T*(Ustride - K - j)
            end
        end
        for c ∈ j+1:C-1
            if isL′ # U is actually a transposed lower triangular matrix
                # c increase corresponds row increase
                push!(q.args, :($(Symbol(:U_,j,:_,c)) = SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Utrisym + $((c-1-j)*size_T + trioff)))))
            else
                push!(q.args, :($(Symbol(:U_,j,:_,c)) = SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Utrisym + $(Symbol(:Uoffset_,c)) + $(K isa Symbol ? :($size_T*($K+$j)) : size_T*(K+j) )))))
            end
            for r ∈ 0:Riterl
                push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vfnmadd($(Symbol(:A_,r,:_,j)), $(Symbol(:U_,j,:_,c)), $(Symbol(:A_,r,:_,c)))))
            end
        end
    end
    # Store in A
    if storeA
        for c ∈ 0:C-1
            for r ∈ 0:Riter-1
                push!(q.args, :( SIMDPirates.vstore!( AsymK + $(size_T * (r*W + c*Astride)), $(Symbol(:A_,r,:_,c)) ) ))
            end
            if Rrem > 0
                push!(q.args, :( SIMDPirates.vstore!( AsymK + $(size_T * (Riter*W + c*Astride)), $(Symbol(:A_,Riter,:_,c)), $mask ) ))
            end
        end
    end
    q
end
function A_rdiv_U_quote(K::Union{Symbol,Integer}, ::Type{T}) where {T}


end

function A_rdiv_L_quote()


end



# ptr and Lstride is so that the first of the reverse iterations has offset 0
# K is the amount of preceding columns.
function A_rdiv_L_kernel_quote(
    R, C, K::Union{Symbol,Integer}, Kmax::Union{Symbol,Integer}, ::Type{T},
    Astride, Bstride, Lstride, isU′, invdiagptr,
    Bsym = :ptrB, Asym = :ptrA, Ltrisym = :ptrLtri, Ldiagsym = :ptrLdiag,
    maskload = true, loadB = true, storeA = true
) where {T}
    # K is either a symbol or integer, indicating number of preceding columns
    size_T = sizeof(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(R, T)
    Wm1 = W - 1
    Riter = R >> Wshift
    Rrem = R & Wm1
    mask = VectorizationBase.mask_from_remainder(T, Rrem)
    if loadB
        if K isa Symbol
            q = quote
                BsymK = $Bsym + $(size_T*Bstride)*$K
                AsymK = $Asym + $(size_T*Astride)*$K
            end
        else
            q = quote
                BsymK = $Bsym + $(size_T*Bstride*K)
                AsymK = $Asym + $(size_T*Astride*K)
            end
        end
        for c ∈ 0:C-1
            for r ∈ 0:Riter-1
                push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vload(Vec{$W,$T}, BsymK + $(size_T * (r*W + c*Bstride)) ) ))
            end
            if Rrem > 0
                # Only need to mask if we're on last column
                if maskload && c == C-1
                    push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vload(Vec{$W,$T}, BsymK + $(size_T * (Riter*W + c*Bstride)), $mask ) ))
                else
                    push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vload(Vec{$W,$T}, BsymK + $(size_T * (Riter*W + c*Bstride)) ) ))
                end
            end
        end
    end
    # We don't need to mask these loads.
    Riterl = Rrem > 0 ? Riter : Riter-1
    if !isU′
        Ltrisym2 = Ltrisym
        coff = Lstride
        for c ∈ 0:C-1
            push!(q.args, Expr(:(=), Symbol(:Loffset_,c), coff*size_T))
            coff += Lstride - c - 2
        end
    end
    # Updating based on all earlier columns
    if K isa Symbol || Kmax isa Symbol || (K + C < Kmax)
        if isU′
            Ltrisym2 = Symbol(:Ltrisym, 2)
            CtriL = (C*(C+1)) >> 1
            if K isa Symbol
#                push!(q.args, :(KpC = $K+$C))
                #                push!(q.args, Expr(:(=), Ltrisym2, :($Ltrisym + ((KpC*(KpC+1))>>1) - (($K*($K+1))>>1))))
                push!(q.args, Expr(:(=), Ltrisym2, :($CtriL + $K*$C)))
            else
#                KpC = K + C
                #                push!(q.args, Expr(:(=), Ltrisym2, :($Ltrisym + $(((KpC*(KpC+1))>>1)-((K*(K+1))>>1)))))
                push!(q.args, Expr(:(=), Ltrisym2, :($(CtriL + K*C))))
            end
        end
        loopbody = quote end
        for r ∈ 0:Riterl
            push!(loopbody.args, :($(Symbol(:A_,r,:_j)) = SIMDPirates.vload(Vec{$W,$T}, $Asym + j*$(Astride*size_T) + $(r*W))))
        end
        if isU′
            push!(loopbody.args, :($Ltrisym += j))
        end
        for c ∈ 0:C-1
            if isU′ # U is actually a transposed lower triangular matrix
                # c increase corresponds row increase
                push!(loopbody.args, :($(Symbol(:L_j_,c)) = SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Ltrisym2 + $(c*size_T)))))
            else
                push!(loopbody.args, :($(Symbol(:L_j_,c)) = SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Ltrisym2 + $(Symbol(:Loffset_,c))+$(C*size_T) + j*$size_T))))
            end
            for r ∈ 0:Riterl
                push!(loopbody.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vfnmadd($(Symbol(:A_,r,:_j)), $(Symbol(:L_j_,c)), $(Symbol(:A_,r,:_,c)))))
            end
        end
        jloopstart = K isa Symbol ? :($K - C) : K - C
        jloopend = Kmax isa Symbol ? :($Kmax - 1) : Kmax - 1
        mainloop = quote
            @inbounds for j ∈ $jloopstart:$jloopend
                $loopbody
            end
        end
#        jloopend = K isa Symbol ?
#            ( Kmax isa Symbol ? :($Kmax - $K - $C) : :($(Kmax-C) - $K)) :
#            ( Kmax isa Symbol ? :($Kmax - $(K + C)) : Kmax - C - K)
#        mainloop = quote
#            @inbounds for nj ∈ 1:$jloopend
#                j = $Kmax - nj
#                $loopbody
#            end
#        end
        push!(q.args, mainloop)
    end
    # Update using just-calculated block
    finish_block = quote end
    scaleop = invdiagptr ? :vmul : :vfdiv
    if isU′
        Uoffset = ((K*(K-1))>>>1) + (C*K) + (((C-2)*(C-1))>>>1)
    end
    for nj ∈ 1:C
        j = C - nj
        vLjj = Symbol(:vL_,j,:_,j)
        push!(q.args, Expr(:(=), vLjj, :(SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Ldiagsym + $(size_T*j))))))
        for r ∈ 0:Riterl
            push!(q.args, :($(Symbol(:A_,r,:_,j)) = SIMDPirates.$scaleop($(Symbol(:A_,r,:_,j)), $vLjj)))
        end
        for c ∈ 0:j-1
            if isU′ # U is actually a transposed lower triangular matrix
                # c increase corresponds row increase
                push!(q.args, :($(Symbol(:L_,j,:_,c)) = SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Ltrisym + $((c+Uoffset)*size_T)))))
            else
                push!(q.args, :($(Symbol(:L_,j,:_,c)) = SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Ltrisym + $(Symbol(:Loffset_,c)) + $((j-1)*size_T)))))
            end
            for r ∈ 0:Riterl
                push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vfnmadd($(Symbol(:A_,r,:_,j)), $(Symbol(:L_,j,:_,c)), $(Symbol(:A_,r,:_,c)))))
            end
        end
        if isU′
            Uoffset -= K + j - 1
#            push!(q.args, :($Ltrisym += $(size_T*(j+1))))
        end
    end
    push!(q.args, finish_block)
    # Store in A
    if storeA
        for c ∈ 0:C-1
            for r ∈ 0:Riter-1
                push!(q.args, :( SIMDPirates.vstore!( AsymK + $(size_T * (r*W + c*Astride)), $(Symbol(:A_,r,:_,c)) ) ))
            end
            if Rrem > 0
                push!(q.args, :( SIMDPirates.vstore!( AsymK + $(size_T * (Riter*W + c*Astride)), $(Symbol(:A_,Riter,:_,c)), $mask ) ))
            end
        end
    end
    q
end

