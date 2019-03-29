
abstract type AbstractBlockDiagonal{M,N,T} <: AbstractMatrix{T} end

struct BlockDiagonalColumnView{M,N,T,P,L} <: AbstractBlockDiagonal{M,N,T}
    data::ConstantFixedSizePaddedMatrix{M,N,T,P,L}
end
function Base.getindex(bd::BlockDiagonalColumnView{M,N,T,P,L}) where {M,N,T,P,L}

end
Base.size(::BlockDiagonalColumnView{M,N}) where {M,N} = (M*N,N)

@generated function Base.:*(BD::BlockDiagonalColumnView{M,N,T,P,L}, A::AbstractFixedSizePaddedMatrix{M,N,T,P,L}) where {M,N,T,P,L}
    q = quote
        c = MutableFixedSizePaddedVector{N,T}(undef)
        for n ∈ 0:$(N>>2-1)
            Base.Cartesian.@nexprs 4 i -> s_i
            @vectorize $T for m ∈ 1:$P # Can we assume the padding is uncontaminated?
                s_1 += A[m + $(4P)*n]       * BD[m + $(4P)*n]
                s_2 += A[m + $(4P)*n+$P]    * BD[m + $(4P)*n+$P]
                s_3 += A[m + $(4P)*n+$(2P)] * BD[m + $(4P)*n+$(2P)]
                s_4 += A[m + $(4P)*n+$(3P)] * BD[m + $(4P)*n+$(3P)]
            end
            @inbounds begin
                Base.Cartesian.@nexprs 4 i -> c[i + 4n] = s_i
            end
        end
    end
    if (N & 3) > 0
        loop_body = quote end
        assignments = quote end
        for n ∈ (N & ~3)+1:N
            ssym = Symbol(:s_,n)
            push!(q.args, :($ssym = zero($T)))
            push!(loop_body.args, :($ssym += A[m + $((n-1)*P) ] * BD[m + $((n-1)*P)]))
            push!(assignments.args, :( c[$n] = $ssym ) )
        end
        push!(q.args, quote
            @vectorize $T for m ∈ 1:$P # Can we assume the padding is uncontaminated?
                $loop_body
            end
            @inbounds begin
                $assignments
            end
        end)
    end
    push!(q.args, :(ConstantFixedSizePaddedVector(c)'))
    q
end
