struct StaticUnitRange{R,T} <: AbstractUnitRange{T} end
Base.@pure StaticUnitRange(R::UnitRange{T}) where {T} = StaticUnitRange{R,T}()
@generated function Base.getindex(::StaticUnitRange{R}, i::Integer) where {R}
    quote
        $(Expr(:meta,:inline))
        @boundscheck ((i < $(first(R))) || (i > $(last(R)))) && ThrowBoundsError()
        i + $(first(R)-1)
    end
end
@generated Base.first(::StaticUnitRange{R}) where {R} = first(R)
@generated Base.last(::StaticUnitRange{R}) where {R} = last(R)
@generated Base.length(::StaticUnitRange{R}) where {R} = length(R)
@generated Base.size(::StaticUnitRange{R}) where {R} = size(R)
