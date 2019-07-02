
abstract type AbstractBlockDiagonal{M,N,T} <: AbstractMatrix{T} end

struct BlockDiagonalColumnView{M,N,T,P,L,A<:AbstractFixedSizePaddedMatrix{M,N,T,P,L}} <: AbstractBlockDiagonal{M,N,T}
    data::A#ConstantFixedSizePaddedMatrix{M,N,T,P,L}
end
# function Base.getindex(bd::BlockDiagonalColumnView{M,N,T,P,L}) where {M,N,T,P,L}
#
# end
Base.size(::BlockDiagonalColumnView{M,N}) where {M,N} = (M*N,N)
@inline VectorizationBase.vectorizable(A::BlockDiagonalColumnView) = VectorizationBase.vectorizable(A.data)


function block_diagonal_column_view_quote(M,N,T,PA,PB,increment::Bool = false)
    unroll = N > 7 ? 4 : N
    if N == unroll
        loop_body = quote end
        for n ∈ 0:N-1
            push!(loop_body.args, :($(Symbol(:s_,n+1)) += A[m + $(PA*n) ] * BD[m + $(PB*n)]))
        end
        #        return quote
        q = quote
            Base.Cartesian.@nexprs $unroll i -> s_i = zero($T)
            @vectorize $T for m ∈ 1:$M #$P # Can we assume the padding is uncontaminated?
                $loop_body
            end
        end
        if increment
            push!(q.args, quote
                  @inbounds begin
                  Base.Cartesian.@nexprs $unroll i -> c[i] = d[i] + s_i
                  end
                  end)
        else
            push!(q.args, quote
                  @inbounds begin
                  Base.Cartesian.@nexprs $unroll i -> c[i] = s_i
                  end
                  end)
        end
        push!(q.args, :c)
#        println(q)
        return q
    end
    reps, rem = divrem(N, unroll)
    if rem > 0
        # catch cases such as N = 9,
        # where we can evenly split the iterations 3 x 3
        # or N = 10, where iterations are 5 x 2
        reps_candidate, rem_candidate = divrem(N, unroll - 1)
        if rem_candidate == 0 
            unroll -= 1
            reps, rem = reps_candidate, rem_candidate
        else
            reps_candidate, rem_candidate = divrem(N, unroll - 2)
            if rem_candidate == 0 
                unroll -= 2
                reps, rem = reps_candidate, rem_candidate
            end
        end
    end
    loop_body = quote end
    for n ∈ 0:unroll-1
        push!(loop_body.args, :($(Symbol(:s_,n+1)) += A[m + $(unroll*PA)*n + $(PA*n) ] * BD[m + $(unroll*PB)*n + $(PB*n)]))
    end
    q = quote
        for n ∈ 0:$((N÷unroll)-1)
            Base.Cartesian.@nexprs $unroll i -> s_i = zero($T)
            @vectorize $T for m ∈ 1:$M #$P # Can we assume the padding is uncontaminated?
                $loop_body
            end
        end
        if increment
            push!(q.args, quote
                  @inbounds begin
                  Base.Cartesian.@nexprs $unroll i -> c[i + $unroll*n] = d[i + $unroll*n] + s_i
                  end
                  end)
        else
            push!(q.args, quote
                  @inbounds begin
                  Base.Cartesian.@nexprs $unroll i -> c[i + $unroll*n] = s_i
                  end
                  end)
        end
    end
    
    if rem > 0
        loop_body = quote end
        assignments = quote end
        for n ∈ N-rem+1:N
            ssym = Symbol(:s_,n)
            push!(q.args, :($ssym = zero($T)))
            push!(loop_body.args, :($ssym += A[m + $((n-1)*PA) ] * BD[m + $((n-1)*PB)]))
            if increment
                push!(assignments.args, :( c[$n] = d[$n] + $ssym ) )
            else
                push!(assignments.args, :( c[$n] = $ssym ) )
            end
        end
        push!(q.args, quote
            @vectorize $T for m ∈ 1:$M
                $loop_body
            end
            @inbounds begin
                $assignments
            end
        end)
    end
 #   push!(q.args, :(c))
    q

end

@generated function LinearAlgebra.mul!(
    c::PaddedMatrices.AbstractMutableFixedSizePaddedVector{N,T},
    A::AbstractFixedSizePaddedMatrix{M,N,T,PA},
    BD::BlockDiagonalColumnView{M,N,T,PB}
) where {M,N,T,PA,PB}
    quote
        $(block_diagonal_column_view_quote(M,N,T,PA,PB,false))
        c
    end
end
function Base.:*(
    A::AbstractFixedSizePaddedMatrix{M,N,T,PA},
    BD::BlockDiagonalColumnView{M,N,T,PB}
) where {M,N,T,PA,PB}
    c  = MutableFixedSizePaddedVector{N,T}(undef)
    mul!(c, A, BD)'
end
@generated function Base.:*(
    sp::PaddedMatrices.StackPointer,
    A::AbstractFixedSizePaddedMatrix{M,N,T,PA},
    BD::BlockDiagonalColumnView{M,N,T,PB}
) where {M,N,T,PA,PB}
    P = min(PA,PB)
    quote
        c  = PtrVector{$N,$T,$N,$N}(pointer(sp,$T))
        sp + $(sizeof(T)*N), mul!(c, A, BD)'
    end
end
@generated function PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(
    sp::PaddedMatrices.StackPointer,
    A::AbstractFixedSizePaddedMatrix{M,N,T,PA},
    BD::BlockDiagonalColumnView{M,N,T,PB},
    d′::LinearAlgebra.Adjoint{T,<:PaddedMatrices.AbstractMutableFixedSizePaddedVector{N,T,PC,PC}}
) where {M,N,T,PA,PB,PC}
    quote
        d = d′'
        c = PtrVector{$N,$T,$N,$N}(pointer(sp,$T))
        $(block_diagonal_column_view_quote(M,N,T,PA,PB,true))
        sp + $(N*sizeof(T)), c'
    end
end
