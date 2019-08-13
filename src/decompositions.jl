

using VectorizationBase, SIMDPirates, StructuredMatrices, PaddedMatrices



"""
Creates a quite that already assumes L (and S if S != L) are VectorizationBase.vpointers

"""
function cholesky_quote(
    M::Int, ::Type{T} = Float64,
    base_offset = M, base_stride = M - 1,
    Lsym = :L, Ssym = :S,
    first_diag_expr =  :( $(Symbol(Lsym,:scalar_,1,:_,1)) = Base.FastMath.sqrt_fast(VectorizationBase.load($Ssym)) ),
    mask_first_load = true
) where {T}
    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    Wm1 = W - 1
    Mm1 = M - 1
    Mrep = Mm1 >> Wshift
    Mrem = Mm1 & Wm1
    V = Vec{W,T}
    Maxrowrep = (Mm1 + Wm1) >>> Wshift
    MT = VectorizationBase.mask_type(W)
    mask_bits = 8sizeof(MT)
    q = quote
        $first_diag_expr
        VectorizationBase.store!($Lsym, $(Symbol(Lsym,:scalar_, 1, :_, 1)))
    end
    # Load S
    # Load diagonal
    dind = 0
    if Mrem > 0
        # Offset for first diaognal vector
        initial_offset = 1 + Mrem - W
        if mask_first_load && initial_offset < 0
            mask = typemax(MT) << (-initial_offset)
            push!(q.args, Expr(:(=), Symbol(Lsym, :d_, dind), :(SIMDPirates.vload($V, $Ssym + $initial_offset, $mask) ) ) )
        else
            push!(q.args, Expr(:(=), Symbol(Lsym, :d_, dind), :(SIMDPirates.vload($V, $Ssym + $initial_offset) ) ) )
        end
        offset = initial_offset + W
        dind += 1
    else
        initial_offset = 1
        offset = 1
    end
    for m ∈ 1:Mrep # Should really be 0 or 1 (or 2 with Mrem == 0); greater leads to register spills
        push!(q.args, Expr(:(=), Symbol(Lsym, :d_, dind), :(SIMDPirates.vload($V, $Ssym + $offset) ) ) )
        offset += W
        dind += 1
    end
    # load off diagonal
    col_offset = Mrem == 0 ? base_offset : base_offset - 1
    rroffset = 0
    for c ∈ 1:Mm1
        rows = M - c
        rowreps = (rows + Wm1) >> Wshift
#        rowrem = rows & Wm1
        for rr ∈ 0:rowreps-1
            push!(q.args, Expr(:(=), Symbol(Lsym, :_, rr + rroffset, :_, c), :(SIMDPirates.vload($V, $Ssym + $(col_offset + rr * W)))))
        end
        col_offset += base_stride - c
        if ((rows - 1) & Wm1) == 0
            col_offset += W
            rroffset += 1
        end
    end
    # S has been loaded
    # Now, start the factorization
    rroffsetcoffset = Mrem == 0 ? -1 : Wm1 - Mrem
    col_offset = Mrem == 0 ? base_offset : base_offset - 1
    for couter ∈ 1:Mm1
        rroffset_outer = (couter + rroffsetcoffset) >> Wshift
        rows = M - couter
        rowreps = (rows + Wm1) >> Wshift
        rowrem = rows & Wm1
        # Optimizer will handle choice between Ldinv = 1 / Ld; Lcol * Ldinv vs Lcol / Ld
        vdiag = Symbol(:v, Lsym, :_, couter, :_, couter)
        push!(q.args, :( $vdiag = SIMDPirates.vbroadcast($V, $(Symbol(Lsym, :scalar_, couter, :_, couter)))))
        for rr ∈ rroffset_outer:rowreps-1 + rroffset_outer
            L_rr_couter = Symbol(Lsym, :_, rr, :_, couter)
            push!(q.args, Expr(:(=), L_rr_couter, :(SIMDPirates.vfdiv($L_rr_couter, $vdiag))))
            diagsym = Symbol(Lsym, :d_, rr)# + Maxrowrep - rowreps)
            push!(q.args, Expr(:(=), diagsym, :(SIMDPirates.vfnmadd($L_rr_couter,$L_rr_couter,$diagsym))))
            if rr == rroffset_outer # we sqrt and store diagonal element
                Lds = Symbol(Lsym, :scalar_, couter + 1, :_, couter + 1)
                doff = rowrem == 0 ? 1 : W + 1 - rowrem
                push!(q.args, Expr(:(=), Lds, :(@inbounds Base.FastMath.sqrt_fast($diagsym[ $doff ].value))))
                push!(q.args, :(VectorizationBase.store!($Lsym + $couter, $Lds)))
            end
            if rr == rroffset_outer && rowrem > 0 # mask store
                push!(q.args, :(SIMDPirates.vstore!($Lsym + $(col_offset), $L_rr_couter, $(typemax(MT) << (W - rowrem)))))
            else
                push!(q.args, :(SIMDPirates.vstore!($Lsym + $(col_offset + (rr-rroffset_outer)*W), $L_rr_couter)))
            end
        end
        scalar_base = col_offset - couter - 1 + ((W - rowrem) & Wm1)
        for cinner ∈ couter+1:Mm1
            colsadjusted = cinner + rroffsetcoffset
            rroffset_inner = (colsadjusted) >> Wshift
            rows = M - cinner
            rowreps = (rows + Wm1) >> Wshift
            Ls = Symbol(:Ls_,cinner,:_,couter)
            push!(q.args, Expr(:(=), Ls, :( SIMDPirates.vbroadcast($V,VectorizationBase.load($Lsym + $(cinner + scalar_base))  ))))
            for rr ∈ rroffset_inner:rowreps - 1 + rroffset_inner
                L_rr_cinner = Symbol(Lsym, :_, rr, :_, cinner)
                push!(q.args, :($L_rr_cinner = SIMDPirates.vfnmadd($(Symbol(Lsym, :_, rr, :_, couter)), $Ls, $L_rr_cinner)))
            end
        end
        col_offset += base_stride - couter
        if ((M - couter - 1) & Wm1) == 0
            col_offset += W
        end
    end
    q   
end

using StructuredMatrices: AbstractMutableLowerTriangularMatrix, AbstractSymmetricMatrixL, MutableSymmetricMatrixL
@generated function choltest!(
    Lm::AbstractMutableLowerTriangularMatrix{M,T},
    Sm::AbstractSymmetricMatrixL{M,T}
#) where {T,M}
) where {M,T}
    chol_quote = cholesky_quote(M, T)
    quote
        L = VectorizationBase.vectorizable(Lm)
        S = VectorizationBase.vectorizable(Sm)
        $chol_quote
        Lm
    end
end

