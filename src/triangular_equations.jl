
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



function A_rdiv_U_kernel_quote(
    R, C, K::Union{Symbol,Integer}, ::Type{T},
    Astride, Bstride, Ustride, isL′, invdiagptr,
    Bsym = :ptrB, Asym = :ptrA, Utrisym = :ptrUtri, Udiagsym = :ptrUdiag,
    maskload = true
) where {T}
    # K is either a symbol or integer, indicating number of preceding columns
    size_T = sizeof(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(R, T)
    Wm1 = W - 1
    Riter = R >> Wshift
    Rrem = R & Wm1
    mask = VectorizationBase.mask_from_remainder(T, Rrem)
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
    
    if !isL′
        coff = Uoffset
        for c ∈ 0:C-1
            coff += c
            push!(q.args, Expr(:(=), Symbol(:Uoffset_,c), coff))
        end
    end
    
    # Updating based on all earlier columns
    if K isa Symbol || K > 0
        loopbody = quote end
        # We don't need to mask these loads.
        Riterl = Rrem > 0 ? Riter : Riter-1
        for r ∈ 0:Riterl
            push!(loopbody.args, :($(Symbol(:A_,r,:_j)) = SIMDPirates.vload(Vec{$W,$T}, $Asym + j*$(Astride*size_T) + $(r*W))))
        end
        if isL′
            push!(loopbody.args, :($Utrisym += Ustride - k))
        end
        for c ∈ 0:C-1
            if isL′ # U is actually a transposed lower triangular matrix
                # c increase corresponds row increase
                push!(loopbody.args, :($(Symbol(:U_j_,c)) = SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Utrisym + $c))))
            else
                push!(loopbody.args, :($(Symbol(:U_j_,c)) = SIMDPirates.vbroadcast(Vec{$W,$T}, VectorizationBase.load($Utrisym + $(Symbol(:Uoffset_,c)) + j))))
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
    

    # Store in A
    for c ∈ 0:C-1
        for r ∈ 0:Riter-1
            push!(q.args, :( SIMDPirates.vstore!( AsymK + $(size_T * (r*W + c*Astride)), $(Symbol(:A_,r,:_,c)) ) ))
        end
        if Rrem > 0
            push!(q.args, :( SIMDPirates.vstore!( AsymK + $(size_T * (Riter*W + c*Astride)), $(Symbol(:A_,Riter,:_,c)), $mask ) ))
        end
    end
    

end
function A_rdiv_U_quote(K::Union{Symbol,Integer}, ::Type{T}) where {T}


end

function A_rdiv_L_quote()


end




