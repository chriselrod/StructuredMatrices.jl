
abstract type AbstractRaggedMatrix{T,I} <: AbstractMatrix{T} end

# struct RaggedMatrix{T,I} <: AbstractRaggedMatrix{T,I}
#     data::Vector{T} # length(data) == sum(column_lengths) == column_offsets[end]
#     column_offsets::Vector{I} # length(column_offsets) == size(A,2)
#     column_lengths::Vector{I} # length(column_lengths) == size(A,2)
#     nrow::Int
# end

# Column offsets go from 0 (col 1 offset), col 2 offset,...,2nd last col offset
struct RaggedMatrix{T,I,VI<:AbstractVector{I},VT<:AbstractVector{T}} <: AbstractRaggedMatrix{T,I}
    data::VT # length(data) == sum(column_lengths) == column_offsets[end]
    column_offsets::VI # length(column_offsets) == size(A,2)
    column_lengths::VI # length(column_lengths) == size(A,2)
    nrow::Int
end
const FixedSizeRaggedMatrix{T,I,NC,ND} = RaggedMatrix{T,I,MutableFixedSizePaddedVector{NC,I,NC,NC},MutableFixedSizePaddedVector{ND,T,ND,ND}}
nrow(A::AbstractRaggedMatrix) = A.nrow
ncol(A::AbstractRaggedMatrix) = length(A.column_lengths)


Base.size(A::AbstractRaggedMatrix) = (nrow(A),ncol(A))
number_not_structurally_zero(A::AbstractRaggedMatrix) = length(A.data)


@Base.propagate_inbounds Base.getindex(A::AbstractRaggedMatrix, i::Integer) = A.data[i]
@inline function Base.getindex(A::AbstractRaggedMatrix{T,I}, i::Integer, j::Integer) where {T,I}
    @boundscheck begin
        ( (i > nrow(A)) || ( (j > ncol(A)) || (min(i,j) < 1) ) ) && PaddedMatrices.ThrowBoundsError("Tried to index array of size $(size(A)) with index ($i,$j).")
    end
    @inbounds begin
        col_j_offset = A.column_offsets[j] #j == one(j) ? zero(I) : A.column_offsets[j-1]#; col_j_length = A.column_descriptions[j,2]
        col_j_nextoffset = j == length(A.column_offsets) ? length(A.data) : A.column_offsets[j+1]
        ind = col_j_offset + Base.unsafe_trunc(I, i)
        col_j_nextoffset >= ind ? A.data[ind] : zero(T)
    end
end

### Iteration returns (row, column, value)
@inline function Base.iterate(A::AbstractRaggedMatrix)
    (1, 1, 1), (2, 2, 1)
end
@inline function Base.iterate(A::AbstractRaggedMatrix, (k,i,j)::NTuple{3,Int})
    if i > A.column_lengths[j]
        j == ncol(A) && return nothing
        i, j = 1, j+1
    end
    (k, i, j), (k+1, i+1, j)
end

function expand_by_x_quote!(q, N, ncol, nrl, T, sp::Bool)
    push!(q.args, ind = 1)
    loop_body = quote end
    a_getindex = quote end
    return_expression = Expr(:tuple)
    for n in 1:N
        asym = Symbol(:a_,n)
        asymi = Symbol(:a_i_,n)
        bsym = Symbol(:b_,n)
        push!(q.args, pointer_vector_expr( bsym, nrl, T, sp, :sptr) )
        push!(a_getindex.args, :($asymi = $asym[i]))
        push!(loop_body.args, :($bsym[ind] = $asymi))
        push!(return_expression.args, bsym)
    end
    return_expression = sp ? :((sptr,$return_expression)) : return_expression
    loop_quote = quote
        @inbounds for i in 1:$ncol
            $a_getindex
            nr = column_lengths[i]
            for r in one(nr):nr
                $loop_body
                ind += 1
            end
        end
        $return_expression
    end
    push!(q.args, loop_body)
    q
end
function contract_by_x_quote!(q, N, ncol, nrl, T, sp::Bool)
    push!(q.args, ind = 1)
    loop_body = quote end
    bset_quote = quote end
    loop_suffix = quote end
    return_expression = Expr(:tuple)
    for n in 1:N
        asym = Symbol(:a_,n)
        asymi = Symbol(:a_i_,n)
        bsym = Symbol(:b_,n)
        bsymi = Symbol(:b_i_,n)
        push!(q.args, pointer_vector_expr( bsym, ncol, T, sp, :sptr) )
        push!(bset_quote.args, :($bsymi = zero(T)))
        push!(loop_body.args, :($bsymi += $asym[i]))
        push!(loop_suffix.args, :($bsym[ind] = $bymi))
        push!(return_expression.args, bsym)
    end
    return_expression = sp ? :((sptr,$return_expression)) : return_expression
    loop_quote = quote
        @inbounds for i in 1:$ncol
            $bset_quote
            nr = column_lengths[i]
            for r in one(nr):nr
                $loop_body
                ind += 1
            end
            $loop_suffix
        end
        $return_expression
    end
    push!(q.args, loop_body)
    q
end
function expand_contract_prequote(VI, VT)
    q = quote column_lengths = x.column_lengths end
    if VI <: AbstractFixedSizeVector
        ncol = full_length(VI)
    else
        ncol = :ncol
        push!(q.args, :($ncol = :(ncol(a))))
    end
    if VT <: AbstractFixedSizeVector
        nrl = full_length(VI)
    else
        nrl = :nrl
        push!(q.args, :($nrl = :(number_not_structurally_zero(a))))
    end
    ncol, nrl, q
end

@generated function expand_by_x(x::RaggedMatrix{T,I,VI,VT}, a::Vararg{<:AbstractVector,N}) where {N,T,VI,VT}
    ncol, nrl, q = expand_contract_prequote(VI, VT)
    expand_by_x_quote!(q, N, ncol, nrl, T, false)
end
@generated function contract_by_x(x::RaggedMatrix{T,I,VI,VT}, a::Vararg{<:AbstractVector,N}) where {N,T,VI,VT}
    ncol, nrl, q = expand_contract_prequote(VI, VT)
    contract_by_x_quote!(q, N, ncol, nrl, T, false)
end
@generated function expand_by_x(sptr::StackPointer, x::RaggedMatrix{T,I,VI,VT}, a::Vararg{<:AbstractVector,N}) where {N,T,VI,VT}
    ncol, nrl, q = expand_contract_prequote(VI, VT)
    expand_by_x_quote!(q, N, ncol, nrl, T, true)
end
@generated function contract_by_x(sptr::StackPointer, x::RaggedMatrix{T,I,VI,VT}, a::Vararg{<:AbstractVector,N}) where {N,T,VI,VT}
    ncol, nrl, q = expand_contract_prequote(VI, VT)
    contract_by_x_quote!(q, N, ncol, nrl, T, true)
end


