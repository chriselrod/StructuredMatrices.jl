
function triangle_vbroadcast_quote(V::DataType, Alength::Int)
    q = quote end
    outtup = Expr(:tuple,)
    for m ∈ 1:Alength
        Asym = Symbol(:A_, m)
        push!(q.args, :($Asym = vbroadcast($V, L[$m])))
        push!(outtup.args, Asym)
    end
    q, outtup
end

# @generated
@generated function SIMDPirates.vbroadcast(::Type{Vec{W,T}}, L::AbstractUpperTriangularMatrix{M,T}) where {W,T,M}
    Alength = (M*(M+1)) >> 1
    V = Vec{W,T}
    q, outtup = triangle_vbroadcast_quote(V, Alength)
    # push!(q.args, :($(A.name){$M,$V,$Alength}($outtup) ) )
    quote
        @inbounds begin
            $q
        end
        $(Expr(:call, Expr(:curly, UpperTriangularMatrix, M, V, Alength), outtup))
        # $(A.name){$M,$V,$Alength}($outtup)
    end
end
@generated function SIMDPirates.vbroadcast(::Type{Vec{W,T}}, L::AbstractLowerTriangularMatrix{M,T}) where {W,T,M}
    Alength = (M*(M+1)) >> 1
    V = Vec{W,T}
    q, outtup = triangle_vbroadcast_quote(V, Alength)
    # push!(q.args, :($(A.name){$M,$V,$Alength}($outtup) ) )
    quote
        @inbounds begin
            $q
        end
        $(Expr(:call, Expr(:curly, LowerTriangularMatrix, M, V, Alength), outtup))
        # $(A.name){$M,$V,$Alength}($outtup)
    end
end

@generated function Base.zero(::Type{LT}) where {N,W,T, LT <: AbstractMutableDiagMatrix{N,Vec{W,T}}}
    quote
        $(Expr(:meta,:inline))
        out = $LT(undef)
        z = vbroadcast(Vec{$W,$T}, zero($T))
        @inbounds for i ∈ 1:$(binomial2(N+1))
            out[i] = z
        end
        out
    end
end

function triangle_vsum_quote(T::DataType,M::Int,L::Int,Lfull::Int)
    quote
        @inbounds for l ∈ 1:$L
            out[l] = SIMDPirates.vsum(triangle[l])
        end
        @inbounds for l ∈ $(L+1):$Lfull
            out[l] = zero($T)
        end
    end
end
@generated function SIMDPirates.vsum(triangle::AbstractLowerTriangularMatrix{M,Vec{W,T},L}) where {T,W,M,L}
    Wm1 = VectorizationBase.pick_vector_width(L, T) - 1
    Lfull = (L + Wm1) & ~Wm1
    q = triangle_vsum_quote(T,M,L,Lfull)
    quote
        $(Expr(:meta,:inline))
        out = MutableLowerTriangularMatrix{$M,$T,$Lfull}(undef)
        $q
        out
    end
end
@generated function SIMDPirates.vsum(triangle::AbstractUpperTriangularMatrix{M,Vec{W,T},L}) where {T,W,M,L}
    Wm1 = VectorizationBase.pick_vector_width(L, T) - 1
    Lfull = (L + Wm1) & ~Wm1
    q = triangle_vsum_quote(T,M,L,Lfull)
    quote
        $(Expr(:meta,:inline))
        out = MutableUpperTriangularMatrix{$M,$T,$Lfull}(undef)
        $q
        out
    end
end
@inline function vsumvec(triangle::AbstractDiagTriangularMatrix{M,Vec{W,T},L}) where {W,T,M,L}
    out = MutableFixedSizePaddedVector{L,T}(undef)
    @inbounds for l ∈ 1:L
        out[l] = SIMDPirates.vsum(triangle[l])
    end
    out
end

# function triangle_vsum_quote(T::DataType,M::Int,L::Int,Lfull::Int)
#     q = quote end
#     outtup = Expr(:tuple,)
#     for l ∈ 1:L
#         sym = Symbol(:triangle_, l)
#         push!(q.args, :($sym = SIMDPirates.vsum(triangle[$l])))
#         push!(outtup.args, sym)
#     end
#     for l ∈ L+1:Lfull
#         push!(outtup.args, zero(T))
#     end
#     q, outtup
# end
#
# @generated function SIMDPirates.vsum(triangle::AbstractLowerTriangularMatrix{M,Vec{W,T},L}) where {W,T,M,L}
#     Lfull = VectorizationBase.pick_vector_width(L, T)
#     q, outtup = triangle_vsum_quote(T,M,L,Lfull)
#     quote
#         @inbounds begin
#             $q
#         end
#         $(Expr(:call, Expr(:curly, LowerTriangularMatrix, M, T, Lfull), outtup))
#     end
# end
# @generated function SIMDPirates.vsum(triangle::AbstractUpperTriangularMatrix{M,Vec{W,T},L}) where {W,T,M,L}
#     Lfull = VectorizationBase.pick_vector_width(L, T)
#     q, outtup = triangle_vsum_quote(T,M,L,Lfull)
#     quote
#         @inbounds begin
#             $q
#         end
#         $(Expr(:call, Expr(:curly, UpperTriangularMatrix, M, T, Lfull), outtup))
#     end
# end

function mkernel_triangle_AU_quote(kernel_size_m::Int, kernel_size_n::Int, mindexpr, nindexpr, add::Bool = true, W::Int = 0, T::DataType = Float64)
    muladdfunc = add ? :vmuladd : :vfnmadd

    # kernel_size = kernel_size_n #min(kernel_size_m, kernel_size_n)
    # @assert kernel_size == kernel_size_m
    # Assumes A_m_n block is already loaded

    q = quote end
    for nouter in 1:kernel_size_n
        for m in 1:kernel_size_m
            push!(q.args, :( $(Symbol(:AU_,m)) = AU[$m + $mindexpr, $nindexpr + $nouter ] ) )
        end
        for ninner in 1:nouter
            if ninner == nouter
                Uind = :(U[ $nindexpr + $ninner ])
            else
                Uind = :(U[ triangle_ind_tuple[$nouter + $nindexpr] + $ninner + $nindexpr ])
            end
            W == 0 ? push!(q.args, :(vU = $Uind )) : push!(q.args, :(vU = vbroadcast(Vec{$W,$T}, $Uind) ))
            for m in 1:kernel_size_m
                AUsym = Symbol(:AU_,m)
                push!(q.args, :( $AUsym = SIMDPirates.$muladdfunc($(Symbol(:A_,m,:_,ninner)), vU, $AUsym) ) )
            end
        end
        for m in 1:kernel_size_m
            push!(q.args, :( AU[$m + $mindexpr, $nindexpr + $nouter ] =  $(Symbol(:AU_,m)) ) )
        end
    end
    q
end
function square_iterations_AU_quote(kernel_size_m::Int, kernel_size_n::Int, mindexpr, nindexpr, N::Int, add::Bool, W::Int, T::DataType)
    muladdfunc = add ? :vmuladd : :vfnmadd
    vUind = :(U[ $nindexpr + n + triangle_ind_tuple[ncol_iter] ])
    vUexpr = W == 0 ? vUind : :(vbroadcast(Vec{$W,$T}, $vUind))
    quote
        # Base.Cartesian.@nexprs $kernel_size_n n -> Base.Cartesian.@nexprs $kernel_size_m m -> A_m_n = A[$mindexpr+m,$nindexpr+n]
        for ncol_iter in $(nindexpr)+$(kernel_size_n+1):$N
            Base.Cartesian.@nexprs $kernel_size_m m -> AU_m = AU[m + $mindexpr, ncol_iter]
            Base.Cartesian.@nexprs $kernel_size_n n -> begin
                vU = $vUexpr
                Base.Cartesian.@nexprs $kernel_size_m m -> begin
                    AU_m = SIMDPirates.$muladdfunc(A_m_n, vU, AU_m)
                end
            end
            Base.Cartesian.@nexprs $kernel_size_m m -> AU[m + $mindexpr, ncol_iter] = AU_m
        end
    end
end
function mkern_iteration_AU_quote(kernel_size_m::Int, kernel_size_n::Int, nk::Int, nr::Int, mindexpr, N::Int, add::Bool, W::Int, T::DataType)
    nindexpr = :(nkern * $kernel_size_n)
    if nr == 0
        q = quote
            for nkern in 0:$(nk-1)
                Base.Cartesian.@nexprs $kernel_size_n n -> begin
                    Base.Cartesian.@nexprs $kernel_size_m m -> A_m_n = A[m+$mindexpr,n+$nindexpr]
                end
                $(mkernel_triangle_AU_quote(kernel_size_m, kernel_size_n, mindexpr, nindexpr, add, W, T))
                $(square_iterations_AU_quote(kernel_size_m, kernel_size_n, mindexpr, nindexpr, N, add, W, T))
            end
        end
    else
        # nindexpr = :($nr + nkern * $kernel_size_n)
        q = quote
            for nkern in 0:$(nk-1)
                Base.Cartesian.@nexprs $kernel_size_n n -> begin
                    Base.Cartesian.@nexprs $kernel_size_m m -> A_m_n = A[m+$mindexpr,n+$nindexpr]
                end
                $(mkernel_triangle_AU_quote(kernel_size_m, kernel_size_n, mindexpr, nindexpr, add, W, T))
                $(square_iterations_AU_quote(kernel_size_m, kernel_size_n, mindexpr, nindexpr, N, add, W, T))
            end
            Base.Cartesian.@nexprs $nr n -> begin
                Base.Cartesian.@nexprs $kernel_size_m m -> A_m_n = A[m+$mindexpr,n+$(nk*kernel_size_n)]
            end
            $(mkernel_triangle_AU_quote(kernel_size_m, nr, mindexpr, nk * kernel_size_n, add, W, T))
        end
    end
    q
end

function matrix_upper_triangle_mul_quote(M::Int, N::Int, T::DataType, add::Bool = true, W::Int = 0)
    mk, mr, nk, nr, kernel_size_m, kernel_size_n = PaddedMatrices.determine_pattern(M,N)

    if nr > 0
        # @assert nk > 0
        # nk -= 1
        # nr = kernel_size_n
        if (N % (nk+1)) == 0
            nr = 0
            nk += 1
            kernel_size_n = N ÷ nk
        end
    end
    if mr > 0
        if (M % (mk+1)) == 0
            mr = 0
            mk += 1
            kernel_size_m = M ÷ mk
        end
    end
    q_triangle_ind_defs = Expr(:tuple, Int32(N))
    triangle_ind = N
    triangle_increment = 0
    for n in 2:N
        push!(q_triangle_ind_defs.args, Int32(triangle_ind))
        triangle_increment += 1
        triangle_ind += triangle_increment
    end


    q = quote
        triangle_ind_tuple = $q_triangle_ind_defs
        for mkern in 0:$(mk-1)
            $(mkern_iteration_AU_quote(kernel_size_m, kernel_size_n, nk, nr, :(mkern * $kernel_size_m), N, add, W, T))
        end
    end
    if mr > 0
        push!(q.args, mkern_iteration_AU_quote(mr, kernel_size_n, nk, nr, mk * kernel_size_m, N, add, W, T))
    end
    q
end

@generated function PaddedMatrices.zero!(U::AbstractDiagTriangularMatrix{M,NTuple{W,Core.VecElement{T}}}) where {M,W,T}
    L = binomial2(M+1)
    quote
        $(Expr(:meta,:inline))
        @inbounds for l ∈ 1:$L
            U[l] = SIMDPirates.vbroadcast(Vec{$W,$T}, zero($T))
        end
        nothing
    end
end

@generated function Base.:*(
            A::AbstractFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            U::AbstractUpperTriangularMatrix{N,NTuple{W,Core.VecElement{T}},L}
        ) where {M,N,T,W,L}

    quote
        $(Expr(:meta,:inline))
        AU = MutableFixedSizePaddedMatrix{$M,$N,NTuple{$W,Core.VecElement{$T}}}(undef)
        @inbounds for i in 1:$(M*N)
            AU[i] = vbroadcast(Vec{$W,$T}, zero($T))
        end
        @inbounds begin
            $(matrix_upper_triangle_mul_quote(M, N, T, true, 0))
        end
        AU
    end
end
@generated function Base.:*(
            A::AbstractFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            U::AbstractUpperTriangularMatrix{N,T,L}
        ) where {M,N,T,W,L}

    quote
        $(Expr(:meta,:inline))
        AU = MutableFixedSizePaddedMatrix{$M,$N,NTuple{$W,Core.VecElement{$T}}}(undef)
        @inbounds for i in 1:$(M*N)
            AU[i] = vbroadcast(Vec{$W,$T}, zero($T))
        end
        @inbounds begin
            $(matrix_upper_triangle_mul_quote(M, N, T, true, W))
        end
        AU
    end
end

@generated function LinearAlgebra.mul!(
            AU::MutableFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            A::AbstractFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            U::AbstractUpperTriangularMatrix{N,NTuple{W,Core.VecElement{T}},L}
        ) where {M,N,W,T,L}
    quote
        $(Expr(:meta,:inline))
        PaddedMatrices.zero!(AU)
        @inbounds begin
            $(matrix_upper_triangle_mul_quote(M, N, T, true, 0))
        end
        AU
    end
end
@generated function LinearAlgebra.mul!(
            AU::MutableFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            A::AbstractFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            U::AbstractUpperTriangularMatrix{N,T,L}
        ) where {M,N,W,T,L}
    quote
        $(Expr(:meta,:inline))
        PaddedMatrices.zero!(AU)
        @inbounds begin
            $(matrix_upper_triangle_mul_quote(M, N, T, true, W))
        end
        AU
    end
end
@generated function addmul!(
            AU::MutableFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            A::AbstractFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            U::AbstractUpperTriangularMatrix{N,NTuple{W,Core.VecElement{T}},L}
        ) where {M,N,W,T,L}
    quote
        $(Expr(:meta,:inline))
        @inbounds begin
            $(matrix_upper_triangle_mul_quote(M, N, T, true, 0))
        end
        AU
    end
end
@generated function addmul!(
            AU::MutableFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            A::AbstractFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            U::AbstractUpperTriangularMatrix{N,T,L}
        ) where {M,N,W,T,L}
    quote
        $(Expr(:meta,:inline))
        @inbounds begin
            $(matrix_upper_triangle_mul_quote(M, N, T, true, W))
        end
        AU
    end
end

@generated function submul!(
            AU::MutableFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            A::AbstractFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            U::AbstractUpperTriangularMatrix{N,NTuple{W,Core.VecElement{T}},L}
        ) where {M,N,W,T,L}

    quote
        $(Expr(:meta,:inline))
        @inbounds begin
            $(matrix_upper_triangle_mul_quote(M, N, T, false, 0))
        end
        AU
    end
end
@generated function submul!(
            AU::MutableFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            A::AbstractFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            U::AbstractUpperTriangularMatrix{N,T,L}
        ) where {M,N,W,T,L}

    quote
        $(Expr(:meta,:inline))
        @inbounds begin
            $(matrix_upper_triangle_mul_quote(M, N, T, false, W))
        end
        AU
    end
end


function mkernel_end_AUt_quote(kernel_size_m::Int, kernel_size_n::Int, mindexpr, nindexpr, add::Bool = true, W::Int = 0, T::DataType = Float64)
    muladdfunc = add ? :vmuladd : :vfnmadd

    # kernel_size = min(kernel_size_m, kernel_size_n)
    # @assert kernel_size == kernel_size_m
    # Assumes A_m_n block is already loaded
    q = quote end
    for nouter in 1:kernel_size_n
        for m in 1:kernel_size_m
            push!(q.args, :( $(Symbol(:AUt_,m)) = AUt[$m + $mindexpr, $nindexpr + $nouter ] ) )
        end
        for ninner in nouter:kernel_size_n
            if ninner == nouter
                Utind = :(Ut[ $nindexpr + $nouter ])
            else
                Utind = :(Ut[ $(Symbol(:triangle_ind_,  ninner)) + $nindexpr + $nouter ] )
            end
            W == 0 ? push!(q.args, :(vUt = $Utind)) : push!(q.args, :(vUt = vbroadcast(Vec{$W,$T}, $Utind)))
            for m in 1:kernel_size_m
                AUtsym = Symbol(:AUt_,m)
                push!(q.args, :( $AUtsym = SIMDPirates.$muladdfunc($(Symbol(:A_,m,:_,ninner)), vUt, $AUtsym) ) )
            end
        end
        for m in 1:kernel_size_m
            push!(q.args, :( AUt[$m + $mindexpr, $nindexpr + $nouter ] =  $(Symbol(:AUt_,m)) ) )
        end
    end
    q
end
function square_iterations_AUt_quote(kernel_size_m::Int, kernel_size_n::Int, mindexpr, nindexpr, add::Bool, W::Int, T::DataType)
    muladdfunc = add ? :vmuladd : :vfnmadd
    vUtexpr = W == 0 ? :(Ut[ triangle_ind_n + ncol_iter ]) : :(vbroadcast(Vec{$W,$T}, Ut[ triangle_ind_n + ncol_iter ]))
    quote
        # Base.Cartesian.@nexprs $kernel_size_n n -> Base.Cartesian.@nexprs $kernel_size_m m -> A_m_n = A[$mindexpr+m,$nindexpr+n]
        for ncol_iter in 1:$(nindexpr)
            Base.Cartesian.@nexprs $kernel_size_m m -> AUt_m = AUt[m + $mindexpr, ncol_iter]
            Base.Cartesian.@nexprs $kernel_size_n n -> begin
                vUt = $vUtexpr
                Base.Cartesian.@nexprs $kernel_size_m m -> begin
                    AUt_m = SIMDPirates.$muladdfunc(A_m_n, vUt, AUt_m)
                end
            end
            Base.Cartesian.@nexprs $kernel_size_m m -> AUt[m + $mindexpr, ncol_iter] = AUt_m
        end
    end
end
function mkern_iteration_AUt_quote(kernel_size_m::Int, kernel_size_n::Int, nk::Int, nr::Int, mindexpr, add::Bool, W::Int, T::DataType)
    if nr == 0
        nindexpr = :(nkern * $kernel_size_n)
        q = quote
            for nkern in 0:$(nk-1)
                Base.Cartesian.@nexprs $kernel_size_n n -> begin
                    triangle_ind_n = triangle_ind_tuple[n + $nindexpr]
                    Base.Cartesian.@nexprs $kernel_size_m m -> A_m_n = A[m+$mindexpr,n+$nindexpr]
                end
                $(square_iterations_AUt_quote(kernel_size_m, kernel_size_n, mindexpr, nindexpr, add, W, T))
                $(mkernel_end_AUt_quote(kernel_size_m, kernel_size_n, mindexpr, nindexpr, add, W, T))
            end
        end
    else
        nindexpr = :($nr + nkern * $kernel_size_n)
        q = quote
            Base.Cartesian.@nexprs $nr n -> begin
                triangle_ind_n = triangle_ind_tuple[n]
                Base.Cartesian.@nexprs $kernel_size_m m -> A_m_n = A[m+$mindexpr,n]
            end
            $(mkernel_end_AUt_quote(kernel_size_m, nr, mindexpr, 0, add, W, T))
            for nkern in 0:$(nk-1)
                Base.Cartesian.@nexprs $kernel_size_n n -> begin
                    triangle_ind_n = triangle_ind_tuple[n + $nindexpr]
                    Base.Cartesian.@nexprs $kernel_size_m m -> A_m_n = A[m+$mindexpr,n+$nindexpr]
                end
                $(square_iterations_AUt_quote(kernel_size_m, kernel_size_n, mindexpr, nindexpr, add, W, T))
                $(mkernel_end_AUt_quote(kernel_size_m, kernel_size_n, mindexpr, nindexpr, add, W, T))
            end
        end
    end
    q
end

function matrix_adjoint_upper_triangle_mul_quote(M::Int, N::Int, T::DataType, add::Bool = true, W::Int = 0)
    mk, mr, nk, nr, kernel_size_m, kernel_size_n = PaddedMatrices.determine_pattern(M,N)

    if nr > 0
        # @assert nk > 0
        # nk -= 1
        # nr = kernel_size_n
        if (N % (nk+1)) == 0
            nr = 0
            nk += 1
            kernel_size_n = N ÷ nk
        end
    end
    if mr > 0
        if (M % (mk+1)) == 0
            mr = 0
            mk += 1
            kernel_size_m = M ÷ mk
        end
    end
    q_triangle_ind_defs = Expr(:tuple, Int32(N))
    triangle_ind = N
    triangle_increment = 0
    for n in 2:N
        push!(q_triangle_ind_defs.args, Int32(triangle_ind))
        triangle_increment += 1
        triangle_ind += triangle_increment
    end


    q = quote
        triangle_ind_tuple = $q_triangle_ind_defs
        for mkern in 0:$(mk-1)
            $(mkern_iteration_AUt_quote(kernel_size_m, kernel_size_n, nk, nr, :(mkern * $kernel_size_m), add, W, T))
        end
    end
    if mr > 0
        push!(q.args, mkern_iteration_AUt_quote(mr, kernel_size_n, nk, nr, mk * kernel_size_m, add, W, T))
    end
    q
end

@generated function Base.:*(
            A::AbstractFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            Ut::LinearAlgebra.Adjoint{Union{},<: AbstractUpperTriangularMatrix{N,NTuple{W,Core.VecElement{T}},L}}
        ) where {M,N,W,T,L}
    quote
        $(Expr(:meta,:inline))
        AUt = MutableFixedSizePaddedMatrix{$M,$N,NTuple{$W,Core.VecElement{$T}}}(undef)
        # AU = MutableFixedSizePaddedMatrix{$M,$N,NTuple{$W,Core.VecElement{$T}}}(undef)
        @inbounds for i in 1:$(M*N)
            AUt[i] = vbroadcast(Vec{$W,$T}, zero($T))
        end
        @inbounds begin
            $(matrix_adjoint_upper_triangle_mul_quote(M, N, T, true, 0))
        end
        AUt
    end
end
@generated function Base.:*(
            A::AbstractFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            Ut::LinearAlgebra.Adjoint{T,<: AbstractUpperTriangularMatrix{N,T,L}}
        ) where {M,N,W,T,L}
    quote
        $(Expr(:meta,:inline))
        AUt = MutableFixedSizePaddedMatrix{$M,$N,NTuple{$W,Core.VecElement{$T}}}(undef)
        # AU = MutableFixedSizePaddedMatrix{$M,$N,NTuple{$W,Core.VecElement{$T}}}(undef)
        @inbounds for i in 1:$(M*N)
            AUt[i] = vbroadcast(Vec{$W,$T}, zero($T))
        end
        @inbounds begin
            $(matrix_adjoint_upper_triangle_mul_quote(M, N, T, true, W))
        end
        AUt
    end
end

@generated function addmul!(
            AUt::MutableFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            A::AbstractFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            Ut::LinearAlgebra.Adjoint{Union{},<: AbstractUpperTriangularMatrix{N,NTuple{W,Core.VecElement{T}},L}}
        ) where {M,N,W,T,L}
    quote
        $(Expr(:meta,:inline))
        @inbounds begin
            $(matrix_adjoint_upper_triangle_mul_quote(M, N, T, true, 0))
        end
        AUt
    end
end
@generated function addmul!(
            AUt::MutableFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            A::AbstractFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            Ut::LinearAlgebra.Adjoint{T,<: AbstractUpperTriangularMatrix{N,T,L}}
        ) where {M,N,W,T,L}
    quote
        $(Expr(:meta,:inline))
        @inbounds begin
            $(matrix_adjoint_upper_triangle_mul_quote(M, N, T, true, W))
        end
        AUt
    end
end

@generated function submul!(
            AUt::MutableFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            A::AbstractFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            Ut::LinearAlgebra.Adjoint{Union{},<: AbstractUpperTriangularMatrix{N,NTuple{W,Core.VecElement{T}},L}}
        ) where {M,N,W,T,L}
    quote
        $(Expr(:meta,:inline))
        @inbounds begin
            $(matrix_adjoint_upper_triangle_mul_quote(M, N, T, false, 0))
        end
        AUt
    end
end
@generated function submul!(
            AUt::MutableFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            A::AbstractFixedSizePaddedMatrix{M,N,NTuple{W,Core.VecElement{T}}},
            Ut::LinearAlgebra.Adjoint{T,<: AbstractUpperTriangularMatrix{N,T,L}}
        ) where {M,N,W,L,T}
    quote
        $(Expr(:meta,:inline))
        @inbounds begin
            $(matrix_adjoint_upper_triangle_mul_quote(M, N, T, false, W))
        end
        AUt
    end
end


function mkernal_start_triangle_view_quote(kernel_size_m::Int, kernel_size_n::Int, mindexpr, nindexpr, add::Bool)#, W::Int, T::DataType)
    muladdfunc = add ? :vmuladd : :vfnmadd

    load_As = quote
        $([:($(Symbol(:A_,m,:_,n)) = A[$mindexpr + $m, $nindexpr + $n]) for m in 1:kernel_size_m, n in 1:kernel_size_n]...)
    end

    if kernel_size_m == 1
        mkern_start_nloop = quote
            U_1_1 = SIMDPirates.$muladdfunc(A_1_n, B[n+$nindexpr,m_col], U_1_1)
        end

        return quote
            $load_As
            tri_ind = tri_ind_base
            tri_increment = tri_increment_base
            m_col = 1+$mindexpr
            U_1_1 = U[m_col]
            Base.Cartesian.@nexprs $kernel_size_n n -> begin
                $mkern_start_nloop
            end
            U[m_col]   = U_1_1
            tri_ind += tri_increment
            tri_ind_col = tri_ind - $kernel_size_m
            tri_increment_col = tri_increment
            tri_increment += 1
        end

    end


    mkern_start_nloop = quote
        vB = B[n+$nindexpr,m_col+1]
        U_1_1 = SIMDPirates.$muladdfunc(A_1_n, B[n+$nindexpr,m_col], U_1_1)
        U_1_2 = SIMDPirates.$muladdfunc(A_1_n, vB, U_1_2)
        U_2_2 = SIMDPirates.$muladdfunc(A_2_n, vB, U_2_2)
    end

    mkern_start = quote
        $load_As
        tri_ind = tri_ind_base
        tri_increment = tri_increment_base
        m_col = 1+$mindexpr
        U_1_1 = U[m_col]
        U_1_2 = U[tri_ind]
        U_2_2 = U[m_col+1]
        Base.Cartesian.@nexprs $kernel_size_n n -> begin
            $mkern_start_nloop
        end
        U[m_col]   = U_1_1
        U[tri_ind] = U_1_2
        tri_ind += tri_increment
        U[m_col+1] = U_2_2
    end
    if kernel_size_m == 2
        push!(mkern_start.args, quote
            tri_ind_col = tri_ind - $kernel_size_m
            tri_increment_col = tri_increment
            tri_increment += 1
        end)
        return mkern_start
    else
        push!(mkern_start.args, :(tri_increment += 1))
    end


    push!(mkern_start.args, quote
    # println("tri_ind = $tri_ind; column: $(m_col + 2) ")
        U_1_3 = U[tri_ind-1]
        U_2_3 = U[tri_ind]
        U_3_3 = U[m_col+2]
    end)
    for n in 1:kernel_size_n
        push!(mkern_start.args, quote
            vB = B[$n+$nindexpr,m_col+2]
            U_1_3 = SIMDPirates.$muladdfunc($(Symbol(:A_1_,n)), vB, U_1_3)
            U_2_3 = SIMDPirates.$muladdfunc($(Symbol(:A_2_,n)), vB, U_2_3)
            U_3_3 = SIMDPirates.$muladdfunc($(Symbol(:A_3_,n)), vB, U_3_3)
        end)
    end
    push!(mkern_start.args, quote
        U[tri_ind-1] = U_1_3
        U[tri_ind] = U_2_3
        tri_ind += tri_increment
        U[m_col+2] = U_3_3
    end)
    if kernel_size_m == 4
        push!(mkern_start.args, quote
            tri_increment += 1
            # println("tri_ind = $tri_ind; column: $(m_col +3) ")
            U_1_4 = U[tri_ind-2]
            U_2_4 = U[tri_ind-1]
            U_3_4 = U[tri_ind]
            U_4_4 = U[m_col+3]
        end)
        for n in 1:kernel_size_n
            push!(mkern_start.args, quote
                vB = B[$n+$nindexpr,m_col+3]
                U_1_4 = SIMDPirates.$muladdfunc($(Symbol(:A_1_,n)), vB, U_1_4)
                U_2_4 = SIMDPirates.$muladdfunc($(Symbol(:A_2_,n)), vB, U_2_4)
                U_3_4 = SIMDPirates.$muladdfunc($(Symbol(:A_3_,n)), vB, U_3_4)
                U_4_4 = SIMDPirates.$muladdfunc($(Symbol(:A_4_,n)), vB, U_4_4)
            end)
        end
        push!(mkern_start.args, quote
            U[tri_ind-2] = U_1_4
            U[tri_ind-1] = U_2_4
            U[tri_ind] = U_3_4
            tri_ind += tri_increment
            # tri_increment += 1
            tri_increment_col = tri_increment
            tri_increment += 1
            U[m_col+3] = U_4_4
            tri_ind_col = tri_ind - $kernel_size_m
        end)
    else
        push!(mkern_start.args, quote
            tri_ind_col = tri_ind - $kernel_size_m
            # tri_increment += 1
            tri_increment_col = tri_increment
            tri_increment += 1
        end)
    end
    mkern_start
end

function square_iterations_triangle_view_quote(kernel_size_m::Int, kernel_size_n::Int, nindexpr, M::Int, add::Bool)
    muladdfunc = add ? :vmuladd : :vfnmadd
    ## The operation is UpperTriangle(MxN * NxM) matrices A and B
    ## this function iterates across the columns of the upper triangular product
    ## The idea of this function is to iterate from
    quote
        for m_coliter in m_col+$kernel_size_m:$M
            Base.Cartesian.@nexprs $kernel_size_m m -> begin
                # println("tri_ind_col + m = $tri_ind_col + $m; column: $m_coliter")
                U_m = U[tri_ind_col + m]
            end
            Base.Cartesian.@nexprs $kernel_size_n n -> begin
            # for n ∈ 1:$kernel_size_n
                vB = B[n+$nindexpr,m_coliter]
                Base.Cartesian.@nexprs $kernel_size_m m -> begin
                    U_m = SIMDPirates.$muladdfunc(A_m_n, vB, U_m)
                end
            end
            Base.Cartesian.@nexprs $kernel_size_m m -> U[tri_ind_col + m] = U_m
            tri_ind_col += tri_increment_col
            tri_increment_col += 1
        end
    end
end

function mul_mat_of_vecs_upper_triangle_view_quote(M::Int,N::Int,W::Int,T::DataType,add::Bool = true)

    mk, mr, nk, nr, kernel_size_m, kernel_size_n = PaddedMatrices.determine_pattern(M,N)
    P = M


    if mr > 0
        if (M % (mk+1)) == 0
            mr = 0
            mk += 1
            kernel_size_m = div(M, mk)
        end
    end
    if nk == 1 && nr > 0
        # More evenly split the kernel.
        kernel_size_n += nr
        nr = (kernel_size_n >> 1)
        kernel_size_n -= nr
    end
    mindexpr = :(mkern * $kernel_size_m)
    nindexpr = :(nkern * $kernel_size_n)

    mindexpr_complete = mk * kernel_size_m
    nindexpr_complete = nk * kernel_size_n
    # At start, A block is already loaded. Now calc U upper.

    mkern_loop_body = quote
        for nkern in 0:$(nk-1)
            $(mkernal_start_triangle_view_quote(kernel_size_m, kernel_size_n, mindexpr, nindexpr,add))
            $(square_iterations_triangle_view_quote(kernel_size_m, kernel_size_n, nindexpr,M, add))
        end
    end
    if nr > 0
        push!(mkern_loop_body.args, mkernal_start_triangle_view_quote(kernel_size_m, nr, mindexpr, nindexpr_complete,add))
        push!(mkern_loop_body.args, square_iterations_triangle_view_quote(kernel_size_m, nr, nindexpr_complete,M,add))
    end

    q = quote
        tri_ind = 0
        tri_increment = 0
        tri_ind_base = $(M + 1)
        tri_increment_base = 2
        @inbounds for mkern in 0:$(mk-1)
            $mkern_loop_body

            tri_ind_base = tri_ind + tri_increment
            tri_increment_base = tri_increment + 1
        end
    end
    if mr > 0
        push!(q.args,
            quote
                @inbounds begin
                    for nkern in 0:$(nk-1)
                        $(mkernal_start_triangle_view_quote(mr, kernel_size_n, mindexpr_complete, nindexpr,add))
                    end
                    $(nr == 0 ? nothing : mkernal_start_triangle_view_quote(mr, kernel_size_n, mindexpr_complete, nindexpr_complete,add))
                end
            end
        )
    end
    push!(q.args, :U)
    q
end

# @generated function mul_upper_triangle_view!(
@generated function addmul!(
            U::MutableUpperTriangularMatrix{M,NTuple{W,Core.VecElement{T}}},
            A::LinearAlgebra.Adjoint{Union{},<: AbstractFixedSizePaddedMatrix{N,M,NTuple{W,Core.VecElement{T}}}},
            B::AbstractFixedSizePaddedMatrix{N,M,NTuple{W,Core.VecElement{T}}}
        ) where {M,N,W,T}

    quote
        $(Expr(:meta,:inline))
        $(mul_mat_of_vecs_upper_triangle_view_quote(M,N,W,T,true))
    end
end
@generated function submul!(
            U::MutableUpperTriangularMatrix{M,NTuple{W,Core.VecElement{T}}},
            A::LinearAlgebra.Adjoint{Union{},<: AbstractFixedSizePaddedMatrix{N,M,NTuple{W,Core.VecElement{T}}}},
            B::AbstractFixedSizePaddedMatrix{N,M,NTuple{W,Core.VecElement{T}}}
        ) where {M,N,T,W}
        # ) where {M,N,W,T}

    quote
        $(Expr(:meta,:inline))
        $(mul_mat_of_vecs_upper_triangle_view_quote(M,N,W,T,false))
    end
end

@generated function mul_upper_triangle_view(
            A::LinearAlgebra.Adjoint{Union{},<: AbstractFixedSizePaddedMatrix{N,M,NTuple{W,Core.VecElement{T}}}},
            B::AbstractFixedSizePaddedMatrix{N,M,NTuple{W,Core.VecElement{T}}}
        ) where {M,N,T,W}
        # ) where {M,N,W,T}

    L = binomial2(M+1)
    quote
        $(Expr(:meta,:inline))
        U = MutableUpperTriangularMatrix{$M,NTuple{$W,Core.VecElement{$T}},$L}(undef)
        vzero = vbroadcast(NTuple{$W,Core.VecElement{$T}}, zero($T))
        @inbounds for l in 1:$L
            U[l] = vzero
        end
        $(mul_mat_of_vecs_upper_triangle_view_quote(M,N,W,T,true))
    end
end

function all_finite(U::AbstractDiagTriangularMatrix{N,NTuple{W,Core.VecElement{T}},L}) where {N,W,T,L}
    @inbounds begin
        reduction = SIMDPirates.visfinite(U[1])
        for l ∈ 2:L
            reduction = SIMDPirates.vand(reduction, SIMDPirates.visfinite(U[l]))
        end
    end
    SIMDPirates.vall(reduction)
end

function PaddedMatrices.zero!(A::AbstractDiagTriangularMatrix{N,T,L}) where {N,T,L}
    @inbounds for i ∈ 1:L
        A[i] = zero(T)
    end
end
# function PaddedMatrices.zero!(A::AbstractDiagTriangularMatrix{N,NTuple{W,Core.VecElement{T}},L}) where {N,W,T,L}
#     @inbounds for i ∈ 1:L
#         A[i] = SIMDPirates.vbroadcast(NTuple{W,Core.VecElement{T}}, zero(T))
#     end
# end
