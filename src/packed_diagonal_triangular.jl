abstract type DiagonalStructure{M,T,FS,L} <: AbstractMatrix{T} end
abstract type AbstractDiagonalTriangularMatrix{M,T,FS,L} <: DiagonalStructure{M,T,FS,L} end

mutable struct DiagonalUpperTriangularMatrix{M,T,FS,L} <: AbstractDiagonalTriangularMatrix{M,T,FS,L}
    data::NTuple{L,T}
    function DiagonalUpperTriangularMatrix{M,T,FS,L}(::UndefInitializer) where {M,T,FS,L}
        new()
    end
    @generated function DiagonalUpperTriangularMatrix{M,T}(::UndefInitializer) where {M,T}
        :(DiagonalUpperTriangularMatrix{$M,$T,$M,$(abs2(M))}(undef))
    end
end

mutable struct DiagonalLowerTriangularMatrix{M,T,FS,L} <: AbstractDiagonalTriangularMatrix{M,T,FS,L}
    data::NTuple{L,T}
    function DiagonalLowerTriangularMatrix{M,T,FS,L}(::UndefInitializer) where {M,T,FS,L}
        new()
    end
    @generated function DiagonalLowerTriangularMatrix{M,T}(::UndefInitializer) where {M,T}
        :(DiagonalLowerTriangularMatrixTriangular{$M,$T,$M,$(abs2(M))}(undef))
    end
end

struct PointerDiagonalUpperTriangularMatrix{M,T,FS,L}; data::Ptr{T}; end
struct PointerDiagonalLowerTriangularMatrix{M,T,FS,L}; data::Ptr{T}; end

const DiagonalUpperTriangular{M,T,FS,L} = Union{
    DiagonalUpperTriangularMatrix{M,T,FS,L},
    PointerDiagonalUpperTriangularMatrix{M,T,FS,L}
}
const DiagonalLowerTriangular{M,T,FS,L} = Union{
    DiagonalLowerTriangularMatrix{M,T,FS,L},
    PointerDiagonalLowerTriangularMatrix{M,T,FS,L}
}
const ConcreteDiagonalTriangularMatrix{M,T,FS,L} = Union{
    DiagonalUpperTriangularMatrix{M,T,FS,L},
    DiagonalLowerTriangularMatrix{M,T,FS,L}
}
const PointerDiagonalTriangularMatrix{M,T,FS,L} = Union{
    PointerDiagonalUpperTriangularMatrix{M,T,FS,L},
    PointerDiagonalLowerTriangularMatrix{M,T,FS,L}
}

@generated Base.length(::AbstractDiagonalTriangularMatrix{M}) where M = abs2(M)
@inline Base.size(::AbstractDiagonalTriangularMatrix{M}) where M = (M,M)

@inline Base.pointer(A::PointerDiagonalTriangularMatrix) = A.data
@inline function Base.pointer(A::ConcreteDiagonalTriangularMatrix{M,T}) where {M,T}
    Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
end


@inline function Base.getindex(A::AbstractDiagonalTriangularMatrix{M,T,FS,L}, i::Integer) where {M,T,FS,L}
    @boundscheck i > L && throw(BoundsError())
    unsafe_load(pointer(A), i)
end

@generated function diagonal_structure_sub2ind(::Val{FS}, i, k) where FS
    quote
        $(Expr(:meta, :inline))
        i - ((($(FS+1) - k)*($(FS+2) - k))>>1) + $((FS+1)*(FS+2)>>1) - k
    end
end
@generated function diagonal_structure_getindex(A::DiagonalStructure{M,T,FS,L}, i, j) where {M,T,FS,L}
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            if i > j
                i > $M && throw(BoundsError())
                return zero(T)
            end
        end
        k = j - i
        ind = diagonal_structure_sub2ind(Val{$FS}(), i, k)
        @boundscheck ind > $L && ThrowBoundsError("")
        unsafe_load(pointer(A), ind)
    end
end
@inline function Base.getindex(A::DiagonalUpperTriangular{M,T,FS,L}, i::Integer, j::Integer) where {M,T,FS,L}
    diagonal_structure_getindex(A, i, j)
end
@inline function Base.getindex(A::DiagonalLowerTriangular{M,T,FS,L}, i::Integer, j::Integer) where {M,T,FS,L}
    diagonal_structure_getindex(A, j, i)
end

@inline function Base.setindex!(A::AbstractDiagonalTriangularMatrix{M,T,FS,L}, v, i::Integer) where {M,T,FS,L}
    @boundscheck i > L && throw(BoundsError())
    unsafe_store!(pointer(A), convert(T,v), i)
    v
end
@generated function diagonal_structure_setindex!(A::DiagonalStructure{M,T,FS,L}, v, i, j) where {M,T,FS,L}
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            if i > j
                throw(BoundsError())
            end
        end
        k = j - i
        ind = diagonal_structure_sub2ind(Val{$FS}(), i, k)
        @boundscheck ind > $L && throw(BoundsError())
        unsafe_store!(pointer(A), convert(T,v), ind)
        v
    end
end
@inline function Base.setindex!(A::DiagonalUpperTriangular{M,T,FS,L}, v, i::Integer, j::Integer) where {M,T,FS,L}
    diagonal_structure_setindex!(A, v, i, j)
end
@inline function Base.setindex!(A::DiagonalLowerTriangular{M,T,FS,L}, v, i::Integer, j::Integer) where {M,T,FS,L}
    diagonal_structure_setindex!(A, v, j, i)
end

@inline Base.firstindex(A::AbstractDiagonalTriangularMatrix) = 1
@inline Base.lastindex(A::AbstractDiagonalTriangularMatrix{M,T,FS,L}) where {M,T,FS,L} = L

@inline function Base.similar(A::DiagonalUpperTriangular{M,T,FS,L}) where {M,T,FS,L}
    DiagonalUpperTriangularMatrix{M,T,FS,L}(undef)
end
@inline function Base.similar(A::DiagonalLowerTriangular{M,T,FS,L}) where {M,T,FS,L}
    DiagonalLowerTriangularMatrix{M,T,FS,L}(undef)
end
