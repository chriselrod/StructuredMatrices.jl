
abstract type AbstractDiagonalSymmetricMatrix{M,T,FS,L} <: DiagonalStructure{M,T,FS,L} end

mutable struct DiagonalSymmetricMatrix{M,T,FS,L} <: AbstractDiagonalSymmetricMatrix{M,T,FS,L}
    data::NTuple{L,T}
    function DiagonalSymmetricMatrix{M,T,FS,L}(::UndefInitializer) where {M,T,FS,L}
        new()
    end
    @generated function DiagonalSymmetricMatrix{M,T}(::UndefInitializer) where {M,T}
        :(DiagonalSymmetricMatrix{$M,$T,$M,$(abs2(M))}(undef))
    end
end

struct PointerDiagonalSymmetricMatrix{M,T,FS,L}; data::Ptr{T}; end

@generated Base.length(::AbstractDiagonalSymmetricMatrix{M}) where M = abs2(M)
@inline Base.size(::AbstractDiagonalSymmetricMatrix{M}) where M = (M,M)

@inline Base.pointer(A::PointerDiagonalSymmetricMatrix) = A.data
@inline function Base.pointer(A::DiagonalSymmetricMatrix{M,T}) where {M,T}
    Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
end

@inline function Base.getindex(A::AbstractDiagonalSymmetricMatrix{M,T,FS,L}, i::Integer) where {M,T,FS,L}
    @boundscheck i > L && throw(BoundsError())
    unsafe_load(pointer(A), i)
end


@inline function Base.getindex(A::AbstractDiagonalSymmetricMatrix{M,T,FS,L}, i::Integer, j::Integer) where {M,T,FS,L}
    i == j && return A[i]
    i, j = minmax(i, j)
    diagonal_structure_getindex(A, i, j)
end

@inline function Base.setindex!(A::AbstractDiagonalSymmetricMatrix{M,T,FS,L}, v, i::Integer) where {M,T,FS,L}
    @boundscheck i > L && throw(BoundsError())
    unsafe_store!(pointer(A), convert(T,v), i)
    v
end
@inline function Base.setindex!(A::AbstractDiagonalSymmetricMatrix{M,T,FS,L}, v, i::Integer, j::Integer) where {M,T,FS,L}
    i == j && return A[i] = v
    i, j = minmax(i, j)
    diagonal_structure_setindex!(A, v, i, j)
end

@inline Base.firstindex(A::AbstractDiagonalSymmetricMatrix) = 1
@inline Base.lastindex(A::AbstractDiagonalSymmetricMatrix{M,T,FS,L}) where {M,T,FS,L}= L

@inline function Base.similar(A::AbstractDiagonalSymmetricMatrix{M,T,FS,L}) where {M,T,FS,L}
    DiagonalSymmetricMatrix{M,T,FS,L}(undef)
end

id_symbol(A, i) = Symbol(A, :_, i)
id_symbol(A, i, j) = Symbol(A, :_, i, :_, j)
# using LinearAlgebra
function gen_ip_chol_quote!(qa,::Type{T},N::Int,L,U = :U, S = :S) where T

    chunksize = 32 ÷ sizeof(T)
    # @show qa

    for i ∈ 1:N
        Ui_i = id_symbol(U, i, i)
        U_i = id_symbol(S, i, i)
        if i == 1
            push!(qa, :($Ui_i = sqrt($U_i)))
        else
            nchunks, r = divrem(i-1, chunksize)
            if r == 0
                push!(qa, :($Ui_i = $U_i - +$([:( $(id_symbol(U, j, i)) * $(id_symbol(U, j, i))) for j ∈ 1:chunksize ]...)  ))
                for chunk ∈ 2:nchunks
                    push!(qa, :($Ui_i = $Ui_i - +$([:( $(id_symbol(U, j, i)) * $(id_symbol(U, j, i))) for j ∈ 1+(chunk-1)*chunksize:chunk*chunksize ]...)  ))
                end

            else
                push!(qa, :($Ui_i = $U_i - +$([:( $(id_symbol(U, j, i)) * $(id_symbol(U, j, i))) for j ∈ 1:r ]...)  ))
                for chunk ∈ 1:nchunks
                    push!(qa, :($Ui_i = $Ui_i - +$([:( $(id_symbol(U, j, i)) * $(id_symbol(U, j, i))) for j ∈ 1+r+(chunk-1)*chunksize:chunk*chunksize+r ]...)  ))
                end
            end
            # for j ∈ 1:i - 1
            #     Uj_i = id_symbol(U, lti+j)
            #     push!(qa, :($Ui_i -= $Uj_i*$Uj_i))
            # end
            push!(qa, :($Ui_i = sqrt($Ui_i)))
        end


        for j ∈ i+1:N
            Uj_i = id_symbol(U, i, j)
            U_i = id_symbol(S, i, j)
            if i == 1
                push!(qa, :($Uj_i = $U_i / $Ui_i) )
            else


                nchunks, r = divrem(i-1, chunksize)
                if r == 0
                    push!(qa, :($Uj_i = $U_i - +$([:( $(id_symbol(U, k, j)) * $(id_symbol(U, k, i))) for k ∈ 1:chunksize ]...)  ))
                    for chunk ∈ 2:nchunks
                        push!(qa, :($Uj_i= $Uj_i - +$([:( $(id_symbol(U, k, j)) * $(id_symbol(U, k, i))) for k ∈ 1+(chunk-1)*chunksize:chunk*chunksize ]...)  ))
                    end

                else
                    push!(qa, :($Uj_i = $U_i - +$([:( $(id_symbol(U, k, j)) * $(id_symbol(U, k, i))) for k ∈ 1:r ]...)  ))
                    for chunk ∈ 1:nchunks
                        push!(qa, :($Uj_i = $Uj_i - +$([:( $(id_symbol(U, k, j)) * $(id_symbol(U, k, i))) for k ∈ 1+r+(chunk-1)*chunksize:chunk*chunksize+r ]...)  ))
                    end
                end

                # push!(qa, :($Uj_i = $U_i))
                # for k ∈ 1:i - 1
                #     Ujk, Uik = id_symbol(U, ltj+k), id_symbol(U, lti+k)
                #     push!(qa, :($Uj_i -= $Ujk * $Uik))
                # end
                push!(qa, :($Uj_i /= $Ui_i) )
            end
        end
    end

    # push!(q.args, U)
    qa
end

@generated function chol!(U::DiagonalUpperTriangular{M,T,FS,L},
                            S::AbstractDiagonalSymmetricMatrix{M,T,FS,L}) where {M,T,FS,L}
    q = quote end
    ind = 0
    for k ∈ 0:M-1, m ∈ 1:M-k
        ind += 1
        push!(q.args, :( $(Symbol(:S_, m, :_, m+k)) = S[$ind] ) )
    end

    gen_ip_chol_quote!(q.args, T, M, L, :U, :S )

    ind = 0
    for k ∈ 0:M-1, m ∈ 1:M-k
        ind += 1
        push!(q.args, :( U[$ind] = $(Symbol(:U_, m, :_, m+k)) ) )
    end
    quote
        @fastmath @inbounds begin
            $q
            nothing
        end
    end
end


S = DiagonalSymmetricMatrix{8,Float64}(undef);
for i ∈ 1:8
   for j ∈ 1:i-1
       global S[i, j] = 1.0
   end
   S[i,i] = 5.0
end
U = DiagonalUpperTriangularMatrix{8,Float64}(undef);
chol!(U, S); U
