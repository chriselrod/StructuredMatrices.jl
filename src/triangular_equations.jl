using PaddedMatrices: AbstractMutableFixedSizeMatrix
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
# divides B by U, stores in A
function A_rdiv_U_kernel_quote(
    R, C, K::Union{Symbol,Integer}, ::Type{T},
    Astride, Bstride, Ustride, isL′, invdiagptr;
    Bsym = :ptrB, Asym = :ptrA, Utrisym = :ptrUtri, Udiagsym = :ptrUdiag,
    maskload = true, loadB = true, storeA = true, reduce_sym::Union{Symbol,Nothing} = nothing,
    maskexpr = :__mask__
) where {T}
    # K is either a symbol or integer, indicating number of preceding columns
    size_T = sizeof(T)
    if R isa Integer
        W, Wshift = VectorizationBase.pick_vector_width_shift(R, T)
        Wm1 = W - 1
        Riter = R >>> Wshift
        Rrem = R & Wm1
        mask = VectorizationBase.mask(T, Rrem)
    else # We assume this is meant to handle a single vector remainder
        W, Wshift = VectorizationBase.pick_vector_width_shift(T)
        Wm1 = W - 1
        Riter = 0
        Rrem = 1
        mask = maskexpr# :(VectorizationBase.mask($T, $R & $Wm1))
    end
    q = quote end
    if loadB
        if K isa Symbol && Bstride isa Symbol
            push!(q.args, :( BsymK = $Bsym + $size_T * $Bstride * $K ))
        elseif K isa Symbol
            push!(q.args, :( BsymK = $Bsym + $(size_T*Bstride)*$K ))
        elseif Bstride isa Symbol
            push!(q.args, :( BsymK = $Bsym + $(size_T*K)*$Bstride ))
        else
            push!(q.args, :( BsymK = $Bsym + $(size_T*Bstride*K) ))
        end
        for c ∈ 0:C-1
            if Bstride isa Symbol
                for r ∈ 0:Riter-1
                    push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vload(Vec{$W,$T}, BsymK + $(size_T * (r*W)) + $(c*size_T)*$Bstride)) )
                end
                if Rrem > 0
                    # Only need to mask if we're on last column
                    if maskload && c == C-1
                        push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vload(Vec{$W,$T}, BsymK + $(size_T * (Riter*W)) + $(c*size_T)*$Bstride, $mask )))
                    else
                        push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vload(Vec{$W,$T}, BsymK + $(size_T * (Riter*W)) + $(c*size_T)*$Bstride )))
                    end
                end
            else
                for r ∈ 0:Riter-1
                    push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vload(Vec{$W,$T}, BsymK + $(size_T * (r*W + c*Bstride)) )))
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
    end
    if !isL′
        if Ustride isa Integer
            coff = Ustride
            for c ∈ 0:C-1
                push!(q.args, Expr(:(=), Symbol(:Uoffset_,c), coff*size_T))
                coff += c
            end
        else
            push!(q.args, Expr(:(=), :Uoffset_0, 0))
            for c ∈ 0:C-2
                push!(q.args, Expr(:(=), Symbol(:Uoffset_,c+1), :($(Symbol(:Uoffset_,c)) + $size_T*($c+K))))
            end
        end
    end
    # We don't need to mask these loads.
    Riterl = Rrem > 0 ? Riter : Riter-1
    # Updating based on all earlier columns
    if K isa Symbol || K > 0
        loopbody = quote end
        for r ∈ 0:Riterl
            push!(loopbody.args, :($(Symbol(:A_,r,:_j)) = SIMDPirates.vload(Vec{$W,$T}, $Asym + (j*$Astride + $(r*W))*$size_T)))
        end
        for c ∈ 0:C-1
            if isL′ # U is actually a transposed lower triangular matrix
                # c increase corresponds row increase
                push!(loopbody.args, :($(Symbol(:U_j_,c)) = SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Utrisym + $((c-1)*size_T)))))
            else
                push!(loopbody.args, :($(Symbol(:U_j_,c)) = SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Utrisym + $(Symbol(:Uoffset_,c)) + (j*$size_T)))))
            end
            for r ∈ 0:Riterl
                push!(loopbody.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vfnmadd($(Symbol(:A_,r,:_j)), $(Symbol(:U_j_,c)), $(Symbol(:A_,r,:_,c)))))
            end
        end
        if isL′
            if Ustride isa Symbol
                push!(loopbody.args, :($Utrisym += $size_T * ($Ustride - j - 2)))
            else
                push!(loopbody.args, :($Utrisym += $size_T * ($(Ustride-2) - j)))
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
        trioff = 0#size_T
        #push!(q.args, :(ptrUtr -= $size_T))
    end
    for j ∈ 0:max(C-2,0)
        if j == 0
            vUjj = Symbol(:vU_,j,:_,j)
            push!(q.args, Expr(:(=), vUjj, :(SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Udiagsym + $(size_T*j))))))
            for r ∈ 0:Riterl
                Arj = Symbol(:A_,r,:_0)
                push!(q.args, Expr(:(=), Arj, :(SIMDPirates.$scaleop($Arj, $vUjj))))
            end
            # if storeA
            #     for r ∈ 0:Riter-1
            #         push!(q.args, :( SIMDPirates.vstore!( AsymK + $(size_T * (r*W)), $(Symbol(:A_,r,:_0)) ) ))
            #     end
            #     if Rrem > 0
            #         push!(q.args, :( SIMDPirates.vstore!( AsymK + $(size_T * (Riter*W)), $(Symbol(:A_,Riter,:_0)), $mask ) ))
            #     end
            # end
            if reduce_sym isa Symbol
                for r ∈ 0:Riter-1
                    Arj = Symbol(:A_,r,:_0)
                    reduce_r = Symbol(reduce_sym, :_, r % 4)
                    push!(q.args, Expr(:(=), reduce_r, :(SIMDPirates.vmuladd($Arj, $Arj, $reduce_r))))
                end
                if Rrem > 0
                    Arj = Symbol(:A_,Riter,:_0)
                    reduce_r = Symbol(reduce_sym, :_, Riter % 4)
                    push!(q.args, Expr(:(=), reduce_r, :(SIMDPirates.vifelse($mask, SIMDPirates.vadd(SIMDPirates.vmul($Arj,$Arj),$reduce_r),$reduce_r))))
                end
            end
        end
        if isL′ && j > 0
            if K isa Symbol
                if Ustride isa Symbol
                    push!(q.args, :($Utrisym += $size_T*($Ustride - $K - $j)))
                else
                    push!(q.args, :($Utrisym += $size_T*($(Ustride-j) - $K)))
                end
            else
                trioff += size_T*(Ustride - K - j)
            end
        end
        for c ∈ j+1:C-1
            if isL′ # U is actually a transposed lower triangular matrix
                # c increase corresponds row increase
                #push!(q.args, :($(Symbol(:U_,j,:_,c)) = SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Utrisym + $((c-1-j)*size_T + trioff)))))
                push!(q.args, :($(Symbol(:U_,j,:_,c)) = SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Utrisym + $((c-1-j)*size_T + trioff)))))
            else
                push!(q.args, :($(Symbol(:U_,j,:_,c)) = SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Utrisym + $(Symbol(:Uoffset_,c)) + $(K isa Symbol ? :($size_T*($K+$j)) : size_T*(K+j) )))))
            end
            for r ∈ 0:Riterl
                push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vfnmadd($(Symbol(:A_,r,:_,j)), $(Symbol(:U_,j,:_,c)), $(Symbol(:A_,r,:_,c)))))
            end
            if c == j+1
                vUjj = Symbol(:vU_,c,:_,c)
                push!(q.args, Expr(:(=), vUjj, :(SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Udiagsym + $(size_T*c))))))
                for r ∈ 0:Riterl
                    push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.$scaleop($(Symbol(:A_,r,:_,c)), $vUjj)))
                end
                # if storeA
                #     for r ∈ 0:Riter-1
                #         push!(q.args, :( SIMDPirates.vstore!( AsymK + $(size_T * (r*W + c*Astride)), $(Symbol(:A_,r,:_,c)) ) ))
                #     end
                #     if Rrem > 0
                #         push!(q.args, :( SIMDPirates.vstore!( AsymK + $(size_T * (Riter*W + c*Astride)), $(Symbol(:A_,Riter,:_,c)), $mask ) ))
                #     end
                # end
                if reduce_sym isa Symbol
                    for r ∈ 0:Riter-1
                        Arj = Symbol(:A_,r,:_,c)
                        reduce_r = Symbol(reduce_sym, :_, r % 4)
                        push!(q.args, Expr(:(=), reduce_r, :(SIMDPirates.vmuladd($Arj, $Arj, $reduce_r))))
                    end
                    if Rrem > 0
                        Arj = Symbol(:A_,Riter,:_,c)
                        reduce_r = Symbol(reduce_sym, :_, Riter % 4)
                        push!(q.args, Expr(:(=), reduce_r, :(SIMDPirates.vifelse($mask, SIMDPirates.vmuladd($Arj,$Arj,$reduce_r), $reduce_r))))
                    end
                end
            end
        end
        #if j == 0 && isL′
        #    trioff -= size_T
        #end
    end
    ### Store in A
    ### Seems faster when I place all the stores together at the end???
    if storeA
        # if K isa Symbol
        push!(q.args, :(AsymK = $Asym + $size_T*$Astride*$K))
        # else
            # push!(q.args, :(AsymK = $Asym + $(size_T*Astride*K)))
        # end
    # end
    # if storeA
        for c ∈ 0:C-1
            for r ∈ 0:Riter-1
                push!(q.args, :( SIMDPirates.vstore!( AsymK + $size_T * ($(r*W) + $c*$Astride), $(Symbol(:A_,r,:_,c)) ) ))
            end
            if Rrem > 0
                push!(q.args, :( SIMDPirates.vstore!( AsymK + $size_T * ($(Riter*W) + $c*$Astride), $(Symbol(:A_,Riter,:_,c)), $mask ) ))
            end
        end
    end
    q
end

# ptr and Lstride is so that the first of the reverse iterations has offset 0
# K is the amount of preceding columns.
function A_rdiv_L_kernel_quote(
    R, C, Kmax::Union{Symbol,Integer}, ::Type{T},
    Astride, Bstride, isU′, invdiagptr;
    Bsym = :ptrB, Asym = :ptrA, Ltrisym = :ptrLtri, Ldiagsym = :ptrLdiag,
    maskload = true, loadB = true, storeA = true, calc_product::Int = 0,
    maskexpr = :__mask__
) where {T}
    # K is either a symbol or integer, indicating number of preceding columns
    # if isU′
    #     throw("Should be false!!!!")
    # end
    size_T = sizeof(T)
    if isU′
        K = Kmax
    end
    if R isa Integer
        W, Wshift = VectorizationBase.pick_vector_width_shift(R, T)
        Wm1 = W - 1
        Riter = R >>> Wshift
        Rrem = R & Wm1
        mask = VectorizationBase.mask(T, Rrem)
    else # We assume this is meant to handle a single vector remainder
        W, Wshift = VectorizationBase.pick_vector_width_shift(T)
        Wm1 = W - 1
        Riter = 0
        Rrem = 1
        mask = maskexpr #:(VectorizationBase.mask($T, $R & $Wm1))
    end
    V = Vec{W,T}
    q = quote end
    if loadB
        for c ∈ 0:C-1
            for r ∈ 0:Riter-1
                set_asym = Symbol(:A_,r,:_,c)
                push!(q.args, :($set_asym = SIMDPirates.vload(Vec{$W,$T}, $Bsym + $size_T * ($(r*W) + $c*$Bstride)) ) )
            end
            if Rrem > 0
                # Only need to mask if we're on last column
                set_asym = Symbol(:A_,Riter,:_,c)
                if maskload && c == C-1
                    push!(q.args, :($set_asym = SIMDPirates.vload(Vec{$W,$T}, $Bsym + $size_T * ($(Riter*W) + $c*$Bstride), $mask) ))
                else
                    push!(q.args, :($set_asym = SIMDPirates.vload(Vec{$W,$T}, $Bsym + $size_T * ($(Riter*W) + $c*$Bstride)) ))
                end
            end
        end
    end
    # We don't need to mask these loads.
    Riterl = Rrem > 0 ? Riter : Riter-1
    if !isU′
        Ltrisym2 = Ltrisym
        coff = 0
        for c ∈ 0:C-1
            if Kmax isa Symbol
                #push!(q.args, Expr(:(=), Symbol(:Loffset_,c), :($(2c*size_T)*$Kmax + $((coff-2c*C)*size_T))))
                #coff -=  c + 2
                if c == 0
                    push!(q.args, Expr(:(=), Symbol(:Loffset_,c), 0  ))
                else
                    # push!(q.args, Expr(:(=), Symbol(:Loffset_,c), :($Kmax*$(c*size_T) - $(coff*size_T))  ))
                    push!(q.args, Expr(:(=), Symbol(:Loffset_,c), :( ($Kmax*$(c*size_T) +  $((coff-C*c)*size_T)))))
                end
            else
                push!(q.args, Expr(:(=), Symbol(:Loffset_,c), size_T * ((Kmax-C)*c + coff)))
            end
            coff += C - c - 2
        end
    end
    # Updating based on all earlier columns
    if Kmax isa Symbol || (C < Kmax)
        if isU′
            Ltrisym2 = Symbol(:Ltrisym, 2)
            CtriL = (C*(C+1)) >>> 1
            if K isa Symbol
#                push!(q.args, :(KpC = $K+$C))
                #                push!(q.args, Expr(:(=), Ltrisym2, :($Ltrisym + ((KpC*(KpC+1))>>>1) - (($K*($K+1))>>>1))))
                push!(q.args, Expr(:(=), Ltrisym2, :($CtriL + $K*$C)))
            else
#                KpC = K + C
                #                push!(q.args, Expr(:(=), Ltrisym2, :($Ltrisym + $(((KpC*(KpC+1))>>>1)-((K*(K+1))>>>1)))))
                push!(q.args, Expr(:(=), Ltrisym2, :($(CtriL + K*C))))
            end
        else
            push!(q.args, Expr(:(=), :ptrAloop, :($Asym + $(C*size_T)*$Astride)))
            push!(q.args, Expr(:(=), :Ltripointer2, :($Ltrisym2 + $((C-1)*size_T))))
        end
        loopbody = quote end
        for r ∈ 0:Riterl
            push!(loopbody.args, :($(Symbol(:A_,r,:_j)) = SIMDPirates.vload(Vec{$W,$T}, ptrAloop + j*$Astride*$size_T + $(r*W*size_T))))
        end
        if isU′
            push!(loopbody.args, :($Ltrisym += j))
        end
        for c ∈ 0:C-1
            if isU′ # U is actually a transposed lower triangular matrix
                # c increase corresponds row increase
                push!(loopbody.args, :($(Symbol(:L_j_,c)) = SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Ltrisym2 + $(c*size_T)))))
            else
                push!(loopbody.args, :($(Symbol(:L_j_,c)) = SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load(Ltripointer2 + $(Symbol(:Loffset_,c)) + j*$size_T))))
            end
            for r ∈ 0:Riterl
                push!(loopbody.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vfnmadd($(Symbol(:A_,r,:_j)), $(Symbol(:L_j_,c)), $(Symbol(:A_,r,:_,c)))))
            end
        end
        if isU′
            jloopstart = K isa Symbol ? :($K - $C) : K - C
            jloopend = Kmax isa Symbol ? :($Kmax - 1) : Kmax - 1
            # @show jloopend, isU′
        else
            jloopstart = 0#C-1
            jloopend = Kmax isa Symbol ? :($Kmax - $(C+1)) : Kmax - C - 1
            # @show jloopend, isU′
        end
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
    if calc_product > 0 # calculate lower tirangle of A' * B
        # calc_product is an indicator
        # 0 indicates don't calc product
        # otherwise, it indicates number of columns to left of kernel
        # outer loop iterates over column of v∂L; we update these columns, iterating over rows of A and B inside to maximize out of order execution
        for c ∈ 0:C-1
            b2c = binomial2(c+1)
            # We load elements from v∂L of this column
            # diagonal
            diagv∂L = Symbol(:v∂L_, c, :_, c)
            push!(q.args, Expr(:(=), diagv∂L, :(SIMDPirates.vload($V, ptrv∂Ldiag + $(c*size_T*W)))))
            # now, the off diagonal
            for cc ∈ c:C-2
                subdiagv∂L = Symbol(:v∂L_, cc+1, :_, c)
                push!(q.args, Expr(:(=), subdiagv∂L, :(SIMDPirates.vload($V, ptrv∂Ltri + $(size_T*W)*($Kmax*$c + $(cc-c - b2c))))))
            end
            # Now, we load vectors from B one at a time, and multiply
            for r ∈ 0:Riterl
                mask_this_iter = maskload && r == Riterl && Rrem > 0
                # load
                set_bsym = Symbol(:B_,r,:_,c)
                # if mask_this_iter
                #     push!(q.args, :($set_bsym = SIMDPirates.vload($V, $Bsym + $size_T * ($(Riterl*W) + $c*$Bstride), $mask) ))
                # else
                    push!(q.args, :($set_bsym = SIMDPirates.vload($V, $Bsym + $size_T * ($(r*W) + $c*$Bstride)) ) )
                # end
                # multuplication; diag first
                if mask_this_iter
                    push!(q.args, Expr(:(=), diagv∂L, :(SIMDPirates.vifelse($mask,SIMDPirates.vmuladd($(Symbol(:A_,r,:_,c)),$set_bsym,$diagv∂L),$diagv∂L))))
                    for cc ∈ c+1:C-1
                        subdiagv∂L = Symbol(:v∂L_, cc, :_, c)
                        push!(q.args, Expr(:(=), subdiagv∂L, :(SIMDPirates.vifelse($mask,SIMDPirates.vmuladd($(Symbol(:A_,r,:_,cc)),$set_bsym,$subdiagv∂L),$subdiagv∂L))))
                    end
                else
                    push!(q.args, Expr(:(=), diagv∂L, :(SIMDPirates.vmuladd($(Symbol(:A_,r,:_,c)),$set_bsym,$diagv∂L))))
                    for cc ∈ c+1:C-1
                        subdiagv∂L = Symbol(:v∂L_, cc, :_, c)
                        push!(q.args, Expr(:(=), subdiagv∂L, :(SIMDPirates.vmuladd($(Symbol(:A_,r,:_,cc)),$set_bsym,$subdiagv∂L))))
                    end
                end
            end
            # Now, store the results
            push!(q.args, :(SIMDPirates.vstore!(ptrv∂Ldiag + $(c*size_T*W), $diagv∂L)))
            # now, store the off diagonal
            for cc ∈ c:C-2
                subdiagv∂L = Symbol(:v∂L_, cc+1, :_, c)
                push!(q.args, :(SIMDPirates.vstore!(ptrv∂Ltri + $(size_T*W)*($Kmax*$c + $(cc-c - b2c)), $subdiagv∂L)))
            end
        end
    end
    # take a break from calc_product while that part of A is still close in memory, in case A === B
    # Store in A
    if storeA
        for c ∈ 0:C-1
            for r ∈ 0:Riter-1
                push!(q.args, :( SIMDPirates.vstore!( $Asym + $size_T * ($(r*W) + $c*$Astride), $(Symbol(:A_,r,:_,c)) ) ))
            end
            if Rrem > 0
                push!(q.args, :( SIMDPirates.vstore!( $Asym + $size_T * ($(Riter*W) + $c*$Astride), $(Symbol(:A_,Riter,:_,c)), $mask ) ))
            end
        end
    end
    if calc_product > 0 && (Kmax isa Symbol || (Kmax < calc_product))
        loopbody = quote end
        # load elements from v∂L
        for c ∈ 0:C-1
            subdiagv∂L = Symbol(:v∂L_, c, :_x)
            push!(loopbody.args, Expr(:(=), subdiagv∂L, :(SIMDPirates.vload($V, ptrv∂Ltri + $(W*size_T)*($c - decrement)))))
        end
        # Now, we load vectors from B one at a time, and multiply
        for r ∈ 0:Riterl
            # load
            set_bsym = Symbol(:B_,r,:_x)
            mask_this_iter = maskload && r == Riterl && Rrem > 0
            if mask_this_iter
                push!(loopbody.args, :($set_bsym = SIMDPirates.vload($V, $Bsym + $size_T * ($(Riter*W) - colleft*$Bstride), $mask) ))
            else
                push!(loopbody.args, :($set_bsym = SIMDPirates.vload($V, $Bsym + $size_T * ($(r*W) - colleft*$Bstride)) ) )
            end
            for c ∈ 0:C-1
                subdiagv∂L = Symbol(:v∂L_, c, :_x)
                if mask_this_iter
                    push!(loopbody.args, Expr(:(=), subdiagv∂L, :(SIMDPirates.vifelse($mask,SIMDPirates.vmuladd($(Symbol(:A_,r,:_,c)),$set_bsym,$subdiagv∂L),$subdiagv∂L) )))
                else
                    push!(loopbody.args, Expr(:(=), subdiagv∂L, :(SIMDPirates.vmuladd($(Symbol(:A_,r,:_,c)),$set_bsym,$subdiagv∂L))))
                end
            end
        end
        # now, store the sub-diagonal
        for c ∈ 0:C-1
            subdiagv∂L = Symbol(:v∂L_, c, :_x)
            push!(loopbody.args, :(SIMDPirates.vstore!(ptrv∂Ltri + $(W*size_T)*($c - decrement),$subdiagv∂L)))
        end

        calc_prod_quote = quote
            decrement = $Kmax
            numtrianglerows = $Kmax
            colleft = 0
            # do one column at a time
            while numtrianglerows < $calc_product
                colleft += 1
                $loopbody
                decrement += numtrianglerows
                numtrianglerows += 1
            end
        end
        push!(q.args, calc_prod_quote)
    end
    q
end


function div_triangle_loads_crfirst(ncol, aloads, colblock, ncolblock, colrem)
    necessary_loads = aloads*ncol + binomial2(ncol+1)
    # if we load the remainder first, it has to be reloaded on all future ncolblocks
    repeat_aloads = (colrem*ncolblock + binomial2(ncolblock)*colblock) * aloads
    necessary_loads + repeat_aloads
end
function div_triangle_loads_crfirst(ncol, aloads, colblock)#, ncolblock = div(ncol, colblock), ncolrem = rem(ncol, colblock))
#    if ncol == typemax(typeof(ncol))
#        ncol = colblock
#        ncolblock, colrem = 1, 0
#    else
        ncolblock, colrem = divrem(ncol, colblock)
#    end
    ncolblock, colrem = divrem(ncol, colblock)
    div_triangle_loads_crfirst(ncol, aloads, colblock, ncolblock, colrem)
end
function div_triangle_loads_crlast(ncol, aloads, colblock, ncolblock, colrem)
    necessary_loads = aloads*ncol + binomial2(ncol+1)
    repeat_aloads = colblock*binomial2(ncolblock)*aloads
    # if we load the remainder last, it has to load all (colblock*ncolblock) completed columns
    if colrem > 0
        repeat_aloads += aloads*colblock*ncolblock
    end
    necessary_loads + repeat_aloads
end
function div_triangle_loads_crlast(ncol, aloads, colblock)
    ncolblock, colrem = divrem(ncol, colblock)
    div_triangle_loads_crlast(ncol, aloads, colblock, ncolblock, colrem)
end


# returns:
# Number of loads, whether there is a remainder
function div_ul_loads(ncol, aloads, colblock, ncolblock, colrem)
    tfirst = div_triangle_loads_crfirst(ncol, aloads, colblock, ncolblock, colrem)
    twotfirst = tfirst + tfirst
    if colrem == 0
        twotfirst - (aloads * colblock), true, false
    else
        tlast = div_triangle_loads_crlast(ncol, aloads, colblock, ncolblock, colrem)
        alt_strat = tfirst + tlast - (aloads * colblock)
        primary_strat = twotfirst - (aloads * colrem)
        min(primary_strat, alt_strat), false, alt_strat < primary_strat
    end
end
function div_ul_loads(ncol, aloads, colblock)
    ncolblock, colrem = divrem(ncol, colblock)
    div_ul_loads(ncol, aloads, colblock, ncolblock, colrem)
end

not_max(x::T) where {T} = x != typemax(T)
function div_triangle_blocking_structure(rows = typemax(UInt), cols = typemax(UInt), ::Type{T} = Float64; reg_count = VectorizationBase.REGISTER_COUNT, reg_size = VectorizationBase.REGISTER_SIZE, verbose = false) where {T}
    cols == typemax(typeof(cols)) && return PaddedMatrices.pick_kernel_size(T, rows, cols,  W = reg_size ÷ sizeof(T), NREG = reg_count)
    W = div(reg_size, sizeof(T))
    if not_max(rows)
        while 2rows < W
            W >>>= 1
        end
    end
    max_aloads = (reg_count >>> 1) - 1
    ratios = Vector{Float64}(undef, max_aloads)
    row_counts = Vector{Int}(undef, max_aloads)
    col_counts = Vector{Int}(undef, max_aloads)
    for aloads in 1:max_aloads
        ncol = min(div(reg_count - aloads - 1, aloads), cols)
        loads = div_triangle_loads_crfirst(cols, aloads, ncol)
        nrow = aloads * W
        ratio = aloads / loads
        if not_max(rows)
            complete_rows, row_rem = divrem(rows, nrow)
            if row_rem > 0
                rem_aloads = cld(row_rem, W)
                ratio = ( ratio * complete_rows * nrow + rem_aloads * row_rem / div_triangle_loads_crfirst(cols, rem_aloads, ncol) ) / rows
            end
        end
        verbose && @show nrow, ncol, ratio
        ratios[aloads] = ratio
        row_counts[aloads] = nrow
        col_counts[aloads] = ncol
        if nrow >= rows
            ratios[aloads+1:end] .= 0
            break
        end
    end
    bestratio, bestind = findmax(ratios)
    W, row_counts[bestind], col_counts[bestind]
end

function div_ul_blocking_structure(rows = typemax(UInt), cols = typemax(UInt), ::Type{T} = Float64; reg_count = VectorizationBase.REGISTER_COUNT, reg_size = VectorizationBase.REGISTER_SIZE) where {T}
    W = div(reg_size, sizeof(T))
    if not_max(rows)
        while 2rows < W
            W >>>= 1
        end
    end
    max_aloads = (reg_count >>> 1) - 1
    ratios = Vector{Float64}(undef, max_aloads)
    row_counts = Vector{Int}(undef, max_aloads)
    col_counts = Vector{Int}(undef, max_aloads)
    # is the remainder 0, are we using the alt strat (doing the remainder first then last) instead of remainder first for both divisions
    strategy = Vector{Tuple{Bool,Bool}}(undef, max_aloads)
    if not_max(cols)
        for aloads in 1:max_aloads
            ncol = min(div(reg_count - aloads - 1, aloads), cols)
            loads, b1, b2 = div_ul_loads(cols, aloads, ncol)
            strategy[aloads] = (b1,b2)
            ratio = aloads / loads
            nrow = aloads * W
            if not_max(rows)
                complete_rows, row_rem = divrem(rows, nrow)
                if row_rem > 0
                    rem_aloads = cld(row_rem, W)
                    rem_loads, b1, b2 = div_ul_loads(cols, rem_aloads, ncol)
                    ratio = ( ratio * complete_rows * nrow + row_rem * rem_aloads / rem_loads ) / rows
                end
            end
#            @show aloads, ncol, ratio
            ratios[aloads] = ratio
            row_counts[aloads] = nrow
            col_counts[aloads] = ncol
            if nrow >= rows
                ratios[aloads+1:end] .= 0
                break
            end
        end
    else
        for aloads in 1:max_aloads
            ncol = div(reg_count - aloads - 1, aloads)
            ncol = min(cols, ncol)
            nrow = aloads * W
            row_counts[aloads] = nrow
            col_counts[aloads] = ncol
            loads = aloads + ncol
            ratio = aloads * ncol / loads
            if not_max(rows)
                complete_rows, row_rem = divrem(rows, nrow)
                if row_rem > 0
                    rem_aloads = cld(row_rem, W)
                    ratio = ( ratio * complete_rows * nrow + row_rem * (rem_aloads * ncol) / (rem_aloads + ncol) ) / rows
                end
            end
            ratios[aloads] = ratio
            if nrow >= rows
                ratios[aloads+1:end] .= 0
                break
            end
        end
    end
    bestratio, bestind = findmax(ratios)
    W, row_counts[bestind], col_counts[bestind], strategy[bestind]
end

function A_div_U_rowiter(
    Mk, Nk, col_rem, T, CP, AP, n_col_reps
)
    size_T = sizeof(T)
    if col_rem > 0
        row_iter = A_rdiv_U_kernel_quote(
            Mk, col_rem, 0, T, CP, AP, 0, false, true
        )
        push!(row_iter.args, :(ptrUtri += $(binomial2(col_rem)*size_T)))
        push!(row_iter.args, :(ptrUdiag += $(col_rem*size_T)))
        base_K = col_rem
    else
        row_iter = quote end
        base_K = 0
    end
    if n_col_reps > 1
        iterquote = A_rdiv_U_kernel_quote(
            Mk, Nk, :K, T, CP, AP, :K, false, true
        )
        row_iter_loop = quote
            K = $base_K
            for crep ∈ 0:$(n_col_reps-1)
                $iterquote
                ptrUtri += $(size_T*binomial2(Nk)) + $(size_T*Nk)*K
                ptrUdiag += $(size_T*Nk)
                K += $Nk
            end
        end
        push!(row_iter.args, row_iter_loop)
    elseif n_col_reps == 1
        row_iter_single = A_rdiv_U_kernel_quote(
            Mk, Nk, 0, T, CP, AP, 0, false, true
        )
        push!(row_iter.args, row_iter_single)
    end
    row_iter
end
function A_rdiv_U_quote(
    M, N, T, CP, AP, NBL
)
    # W = vector width
    # Mk = kernel rows
    # Nk = kernel colums
    W, Mk, Nk = div_triangle_blocking_structure(M, N, T)
    Wm1 = W - 1
    n_row_reps, row_rem = divrem(M, Mk)
    total_row_iterations = n_row_reps + (row_rem > 0)
    n_col_reps, col_rem = divrem(N, Nk)
    total_col_iterations = n_col_reps + (col_rem > 0)
    Nl = ( N + Wm1 ) & ~Wm1
    Nl = Nl > NBL ? N : Nl # Don't segfault
    size_T = sizeof(T)
    q = quote
        invdiag = FixedSizeVector{$N,$T,$Nl,$Nl}(undef)
        LoopVectorization.@vvectorize $T for n ∈ 1:$Nl
            invdiag[n] = one($T) / B[n]
        end
        ptrB = pointer(A)
        ptrA = pointer(C)
    end
    Mk1 = n_row_reps == 0 ? row_rem : Mk
    row_iter = A_div_U_rowiter(
        Mk1, Nk, col_rem, T, CP, AP, n_col_reps
    )
    if n_row_reps > 1
        row_loops = quote
            for rrep ∈ 1:$n_row_reps
                ptrUdiag = pointer(invdiag)
                ptrUtri = pointer(B) + $(size_T * N)
                $row_iter
                ptrB += $(size_T*Mk)
                ptrA += $(size_T*Mk)
            end
        end
        push!(q.args, row_loops)
    else
        push!(q.args, :(ptrUdiag = pointer(invdiag)))
        push!(q.args, :(ptrUtri = pointer(B) + $(size_T * N)))
        push!(q.args, row_iter)
        row_rem > 0 && push!(q.args, :(ptrB += $(size_T*Mk); ptrA += $(Size_T*Mk)))
    end
    if row_rem > 0 && n_row_reps > 0
        push!(q.args, :(ptrUdiag = pointer(invdiag)))
        push!(q.args, :(ptrUtri = pointer(B) + $(size_T * N)))
        push!(q.args, A_div_U_rowiter( row_rem, Nk, col_rem, T, CP, AP, n_col_reps ))
    end
    push!(q.args, :C)
    q    
end
@generated function A_rdiv_B!(
    C::AbstractMutableFixedSizeMatrix{M,N,T,CP},
    A::AbstractMutableFixedSizeMatrix{M,N,T,AP},
    B::AbstractUpperTriangularMatrix{N,T,NBL}
#) where {M,N,T,NBL,CP,AP}
) where {M,N,T,CP,AP,NBL}
    A_rdiv_U_quote(M, N, T, CP, AP, NBL)
end



function A_div_L′_rowiter(
    Mk, Nk, col_rem, T, CP, AP, n_col_reps
)
    N = Nk * n_col_reps + col_rem
    size_T = sizeof(T)
    if col_rem > 0
        row_iter = A_rdiv_U_kernel_quote(
            Mk, col_rem, 0, T, CP, AP, N, true, true
        )
        #pushfirst!(row_iter.args, :(ptrUtri = ptrUtribase))
        push!(row_iter.args, :(ptrUdiag += $(col_rem*size_T)))
        base_K = col_rem
        KmZ = false
    else
        row_iter = quote end
        base_K = 0
        KmZ = true
    end
    if n_col_reps > 1
        iterquote = A_rdiv_U_kernel_quote(
            Mk, Nk, :K, T, CP, AP, N, true, true
        )
        row_iter_loop = quote
            K = $base_K
            for crep ∈ 0:$(n_col_reps-1)
                ptrUtri = ptrUtribase + K*$size_T
                $iterquote
                ptrUdiag += $(size_T*Nk)
                K += $Nk
            end
        end
        push!(row_iter.args, row_iter_loop)
    elseif n_col_reps == 1
        row_iter_single = A_rdiv_U_kernel_quote(
            Mk, Nk, col_rem, T, CP, AP, N, true, true
        )
        push!(row_iter.args, row_iter_single)
    end
    row_iter
end
function A_rdiv_L′_quote(
    M, N, T, CP, AP, NBL
)
    # W = vector width
    # Mk = kernel rows
    # Nk = kernel colums
    W, Mk, Nk = div_triangle_blocking_structure(M, N, T)
    Wm1 = W - 1
    n_row_reps, row_rem = divrem(M, Mk)
    total_row_iterations = n_row_reps + (row_rem > 0)
    n_col_reps, col_rem = divrem(N, Nk)
    total_col_iterations = n_col_reps + (col_rem > 0)
    Nl = ( N + W - 1 ) & ~Wm1
    Nl = Nl > NBL ? N : Nl # Don't segfault
    size_T = sizeof(T)
    q = quote
        B = Badj.parent
        invdiag = FixedSizeVector{$N,$T,$Nl,$Nl}(undef)
        LoopVectorization.@vvectorize $T for n ∈ 1:$Nl
            invdiag[n] = one($T) / B[n]
        end
        ptrB = pointer(A)
        ptrA = pointer(C)
        ptrUtribase = pointer(B) + $(N*size_T)
    end
    Mk1 = n_row_reps == 0 ? row_rem : Mk
    row_iter = A_div_L′_rowiter(
        Mk1, Nk, col_rem, T, CP, AP, n_col_reps
    )
    if n_row_reps > 1
        row_loops = quote
            for rrep ∈ 1:$n_row_reps
                ptrUdiag = pointer(invdiag)
                ptrUtri = ptrUtribase#pointer(B) + $(size_T * N)
                $row_iter
                ptrB += $(size_T*Mk)
                ptrA += $(size_T*Mk)
            end
        end
        push!(q.args, row_loops)
    else
        push!(q.args, :(ptrUdiag = pointer(invdiag)))
        push!(q.args, :(ptrUtri = ptrUtribase))
        push!(q.args, row_iter)
        push!(q.args, :(ptrB += $(size_T*Mk); ptrA += $(size_T*Mk) ))
    end
    if row_rem > 0 && n_row_reps > 0 # then n_row_reps == 1 and row_rem > 0
        push!(q.args, :(ptrUdiag = pointer(invdiag)))
        push!(q.args, :(ptrUtri = ptrUtribase))
        push!(q.args, A_div_L′_rowiter( row_rem, Nk, col_rem, T, CP, AP, n_col_reps ))
    end
    push!(q.args, :C)
    q
end
@generated function A_rdiv_B!(
    C::AbstractMutableFixedSizeMatrix{M,N,T,CP},
    A::AbstractMutableFixedSizeMatrix{M,N,T,AP},
    Badj::Adjoint{T,<:AbstractLowerTriangularMatrix{N,T,NBL}}
# ) where {M,N,T,CP,AP,NBL}
) where {M,N,T,NBL,CP,AP}
    A_rdiv_L′_quote(
        M, N, T, CP, AP, NBL
    )
end




function A_div_L_rowiter(
    Mk, Nk, col_rem, T, CP, AP, n_col_reps
)
    N = Nk * n_col_reps + col_rem
    size_T = sizeof(T)
    if col_rem > 0
        
        row_iter = A_rdiv_L_kernel_quote(
            Mk, col_rem, col_rem, T, CP, AP, false, true
        )
        #pushfirst!(row_iter.args, :(ptrUtri = ptrUtribase))
        fullcols = Nk * n_col_reps
        # handle following in A_rdiv_L_quote
        # pushfirst!(row_iter.args, :(ptrA -= $(CP*col_rem*size_T)))
        # pushfirst!(row_iter.args, :(ptrB -= $(AP*col_rem*size_T)))
        push!(row_iter.args, :(ptrLdiag -= $(col_rem*size_T)))
        push!(row_iter.args, :(ptrLtri -= $((binomial2(Nk) + Nk*col_rem)*size_T)))
        base_K = col_rem
        KmZ = false
    else
        row_iter = quote end
        base_K = 0
        KmZ = true
    end
    if n_col_reps > 1
        iterquote = A_rdiv_L_kernel_quote(
            Mk, Nk, :K, T, CP, AP, false, true
        )
        row_iter_loop = quote
            K = $col_rem
            for crep ∈ 0:$(n_col_reps-1)
                K += $Nk
                $iterquote
                ptrLdiag -= $(size_T*Nk)
                #K += $Nk
                ptrA -= $(Nk*CP*size_T)
                ptrB -= $(Nk*AP*size_T)
                ptrLtri -= $size_T*($Nk*K + $(binomial2(Nk)))  # = ptrLtribase + K*$size_T
            end
        end
        push!(row_iter.args, row_iter_loop)
    elseif n_col_reps == 1
        row_iter_single = A_rdiv_L_kernel_quote(
            Mk, Nk, N, T, CP, AP, false, true
        )
        push!(row_iter.args, row_iter_single)
    end
    row_iter
end
function A_rdiv_L_quote(
    M, N, T, CP, AP, NBL
)
    # W = vector width
    # Mk = kernel rows
    # Nk = kernel colums
    W, Mk, Nk = div_triangle_blocking_structure(M, N, T)
    Wm1 = W - 1
    n_row_reps, row_rem = divrem(M, Mk)
    total_row_iterations = n_row_reps + (row_rem > 0)
    n_col_reps, col_rem = divrem(N, Nk)
    total_col_iterations = n_col_reps + (col_rem > 0)
    Nl = ( N + W - 1 ) & ~Wm1
    Nl = Nl > NBL ? N : Nl # Don't segfault
    size_T = sizeof(T)
    startoffset = (total_col_iterations-1) * Nk
    q = quote
        invdiag = FixedSizeVector{$N,$T,$Nl,$Nl}(undef)
        LoopVectorization.@vvectorize $T for n ∈ 1:$Nl
            invdiag[n] = one($T) / B[n]
        end
        ptrB_base = pointer(A) + $(size_T*AP*startoffset)
        ptrA_base = pointer(C) + $(size_T*CP*startoffset)
        ptrLtribase = pointer(B) + $(size_T * (N + binomial2(startoffset) + startoffset * (N - startoffset))) # diag + triangle + subtriangle
        ptrLdiagbase = pointer(invdiag) + $(size_T * startoffset)
    end
    Mk1 = n_row_reps == 0 ? row_rem : Mk
    row_iter = A_div_L_rowiter(
        Mk1, Nk, col_rem, T, CP, AP, n_col_reps
    )
    if n_row_reps > 1
        row_loops = quote
            for rrep ∈ 1:$n_row_reps
                ptrLdiag = ptrLdiagbase
                ptrLtri = ptrLtribase
                ptrA = ptrA_base
                ptrB = ptrB_base
                $row_iter
                ptrB_base += $(size_T*Mk)
                ptrA_base += $(size_T*Mk)
            end
        end
        push!(q.args, row_loops)
    else
        push!(q.args, :(ptrLdiag = ptrLdiagbase))
        push!(q.args, :(ptrLtri = ptrLtribase))
        push!(q.args, :(ptrA = ptrA_base))
        push!(q.args, :(ptrB = ptrB_base))
        push!(q.args, row_iter)
    end
    if row_rem > 0 && n_row_reps > 0
        push!(q.args, :(ptrLdiag = ptrLdiagbase))
        push!(q.args, :(ptrLtri = ptrLtribase))
        push!(q.args, :(ptrA = ptrA_base + $(size_T*Mk)))
        push!(q.args, :(ptrB = ptrB_base + $(size_T*Mk)))
        push!(q.args, A_div_L_rowiter( row_rem, Nk, col_rem, T, CP, AP, n_col_reps ))
    end
    push!(q.args, :C)
    q    
end

@generated function A_rdiv_B!(
    C::AbstractMutableFixedSizeMatrix{M,N,T,CP},
    A::AbstractMutableFixedSizeMatrix{M,N,T,AP},
    B::AbstractLowerTriangularMatrix{N,T,NBL}
# ) where {M,N,T,CP,AP,NBL}
) where {M,N,T,NBL,CP,AP}
    A_rdiv_L_quote(
        M, N, T, CP, AP, NBL
    )
end
function A_rdiv_B_expr(
    C::AbstractMutableFixedSizeMatrix{M,N,T,CP},
    A::AbstractMutableFixedSizeMatrix{M,N,T,AP},
    B::AbstractLowerTriangularMatrix{N,T,NBL}
) where {M,N,T,CP,AP,NBL}
#) where {M,N,T,NBL,CP,AP}
    PaddedMatrices.prettify(A_rdiv_L_quote(
        M, N, T, CP, AP, NBL
    ))
end


function A_rdiv_B!(
    C::AbstractMutableFixedSizeMatrix{M,N,T},
    A::AbstractMutableFixedSizeMatrix{M,N,T},
    B::Adjoint{T,<:AbstractUpperTriangularMatrix{N,T}}
) where {M,N,T,CP,AP,NBL}
#) where {M,N,T,NBL,CP,AP}

end


# function A_rdiv_B!(
#     C::AbstractMutableFixedSizeMatrix{M,N,T},
#     A::AbstractMutableFixedSizeMatrix{M,N,T},
#     B::AbstractSymmetricMatrixU{N,T}
# ) where {M,N,T,CP,AP,NBL}
# #) where {M,N,T,NBL,CP,AP}
# end
# function A_rdiv_B!(
#     C::AbstractMutableFixedSizeMatrix{M,N,T},
#     A::AbstractMutableFixedSizeMatrix{M,N,T},
#     B::AbstractSymmetricMatrixL{N,T}
# ) where {M,N,T,CP,AP,NBL}
# #) where {M,N,T,NBL,CP,AP}
# end




