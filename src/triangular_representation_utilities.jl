abstract type AbstractDiagTriangularMatrix{P,T,L} <: AbstractMatrix{T} end



abstract type AbstractTriangularMatrix{P,T,L} <: AbstractDiagTriangularMatrix{P,T,L} end
abstract type AbstractLowerTriangularMatrix{P,T,L} <: AbstractTriangularMatrix{P,T,L} end
abstract type AbstractUpperTriangularMatrix{P,T,L} <: AbstractTriangularMatrix{P,T,L} end
abstract type AbstractMutableLowerTriangularMatrix{P,T,L} <: AbstractLowerTriangularMatrix{P,T,L} end
abstract type AbstractMutableUpperTriangularMatrix{P,T,L} <: AbstractUpperTriangularMatrix{P,T,L} end

struct LowerTriangularMatrix{P,T,L} <: AbstractLowerTriangularMatrix{P,T,L}
    data::NTuple{L,T}
end
mutable struct MutableLowerTriangularMatrix{P,T,L} <: AbstractMutableLowerTriangularMatrix{P,T,L}
    data::NTuple{L,T}
    MutableLowerTriangularMatrix{P,T,L}(::UndefInitializer) where {P,T,L} = new{P,T,L}()
end
@inline LowerTriangularMatrix(M::MutableLowerTriangularMatrix{P,T,L}) where {P,T,L} = LowerTriangularMatrix{P,T,L}(M.data)
struct UpperTriangularMatrix{P,T,L} <: AbstractUpperTriangularMatrix{P,T,L}
    data::NTuple{L,T}
end
mutable struct MutableUpperTriangularMatrix{P,T,L} <: AbstractMutableUpperTriangularMatrix{P,T,L}
    data::NTuple{L,T}
    MutableUpperTriangularMatrix{P,T,L}(::UndefInitializer) where {P,T,L} = new{P,T,L}()
end
struct PtrLowerTriangularMatrix{P,T,L} <: AbstractMutableLowerTriangularMatrix{P,T,L}
    ptr::Ptr{T}
end
struct PtrUpperTriangularMatrix{P,T,L} <: AbstractMutableUpperTriangularMatrix{P,T,L}
    ptr::Ptr{T}
end

function PtrLowerTriangularMatrix{P,T,L}(sp::StackPointer) where {P,T,L}
    sp + sizeof(T) * L, PtrLowerTriangularMatrix{P,T,L}(pointer(sp, T))
end

@generated function PtrLowerTriangularMatrix{P,T}(sp::PaddedMatrices.StackPointer) where {P,T}
    L = binomial2(P+1)
    quote
        A = PtrLowerTriangularMatrix{$P,$T,$L}(pointer(sp, $T))
        sp + $(L*sizeof(T)), A
    end
end
@generated function PtrUpperTriangularMatrix{P,T}(sp::PaddedMatrices.StackPointer) where {P,T}
    L = binomial2(P+1)
    quote
        A = PtrUpperTriangularMatrix{$P,$T,$L}(pointer(sp, $T))
        sp + $(L*sizeof(T)), A
    end
end


@inline UpperTriangularMatrix(M::MutableUpperTriangularMatrix{P,T,L}) where {P,T,L} = UpperTriangularMatrix{P,T,L}(M.data)
@generated function MutableLowerTriangularMatrix{P,T}(undef) where {P,T}
    Lbase = binomial2(P+1)
    Wm1 = VectorizationBase.pick_vector_width(Lbase,T) - 1
    L = (Lbase + Wm1) & ~Wm1
    quote
        $(Expr(:meta,:inline))
        MutableLowerTriangularMatrix{$P,$T,$L}(undef)
    end
end
@generated function MutableUpperTriangularMatrix{P,T}(undef) where {P,T}
    Lbase = binomial2(P+1)
    Wm1 = VectorizationBase.pick_vector_width(Lbase,T) - 1
    L = (Lbase + Wm1) & ~Wm1
    quote
        $(Expr(:meta,:inline))
        MutableUpperTriangularMatrix{$P,$T,$L}(undef)
    end
end
function MutableLowerTriangularMatrix(
    B::Union{A,LinearAlgebra.Adjoint{T,A}}
) where {T,M,A <: AbstractFixedSizePaddedMatrix{M,M,T}}
    L = MutableLowerTriangularMatrix{M,T}(undef)
    @inbounds for mc ∈ 1:M, mr ∈ mc:M
        L[mr,mc] = B[mr,mc]
    end
    L
end
function MutableUpperTriangularMatrix(
    B::Union{A,LinearAlgebra.Adjoint{T,A}}
) where {T,M,A <: AbstractFixedSizePaddedMatrix{M,M,T}}
    U = MutableLowerTriangularMatrix{M,T}(undef)
    @inbounds for mc ∈ 1:M, mr ∈ 1:mc
        U[mr,mc] = B[mr,mc]
    end
    U
end

function copyto!(L::MutableLowerTriangularMatrix{M,T}, A::Matrix) where {M,T}
    for m ∈ 1:M
        L[m] = A[m,m]
    end
    for mc ∈ 1:M-1
        for mr ∈ mc+1:M
            L[mr,mc] = A[mr,mc]
        end
    end
end
function MutableLowerTriangularMatrix(A::Matrix{T}) where {T}
    M, N = size(A)
    @assert M == N
    L = MutableLowerTriangularMatrix{M,T}(undef)
    copyto!(L, A)
    L
end

function copyto!(L::MutableUpperTriangularMatrix{M,T}, A::Matrix) where {M,T}
    for m ∈ 1:M
        L[m] = A[m,m]
    end
    for mc ∈ 2:M
        for mr ∈ 1:mc-1
            L[mr,mc] = A[mr,mc]
        end
    end
end
function MutableUpperTriangularMatrix(A::Matrix{T}) where {T}
    M, N = size(A)
    @assert M == N
    L = MutableUpperTriangularMatrix{M,T}(undef)
    copyto!(L, A)
    L
end

abstract type AbstractSymmetricMatrix{P,T,L} <: AbstractDiagTriangularMatrix{P,T,L} end
abstract type AbstractSymmetricMatrixL{P,T,L} <: AbstractSymmetricMatrix{P,T,L} end
abstract type AbstractSymmetricMatrixU{P,T,L} <: AbstractSymmetricMatrix{P,T,L} end
abstract type AbstractMutableSymmetricMatrixL{P,T,L} <: AbstractSymmetricMatrix{P,T,L} end
abstract type AbstractMutableSymmetricMatrixU{P,T,L} <: AbstractSymmetricMatrix{P,T,L} end

struct SymmetricMatrixL{P,T,L} <: AbstractSymmetricMatrixL{P,T,L}
    data::NTuple{L,T}
end
mutable struct MutableSymmetricMatrixL{P,T,L} <: AbstractMutableSymmetricMatrixL{P,T,L}
    data::NTuple{L,T}
    MutableSymmetricMatrixL{P,T,L}(::UndefInitializer) where {P,T,L} = new{P,T,L}()
end
struct PtrSymmetricMatrixL{P,T,L} <: AbstractMutableSymmetricMatrixL{P,T,L}
    ptr::Ptr{T}
end
# @generated function SymmetricMatrixL(S::PaddedMatrices.AbstractFixedSizePaddedMatrix{P,P,T,R}) where {P,T,R}
#     q = quote end
#     qa = q.args
#     PaddedMatrices.load_L_quote!(qa, P, R, :Σ, :Σ)
#     L = binomial2(P+1)
#     Wm1 = VectorizationBase.pick_vector_width(L, T)-1
#     L = (L + Wm1) & ~Wm1
#     lq = store_packed_L_quote!(qa, P, :Σ, T, L)
#     quote
#         # $(Expr(:meta,:inline))
#         @inbounds begin
#             # begin
#             $q
#             $lq
#         end
#     end
# end

struct SymmetricMatrixU{P,T,L} <: AbstractSymmetricMatrix{P,T,L}
    data::NTuple{L,T}
end
struct PtrSymmetricMatrixU{P,T,L} <: AbstractMutableSymmetricMatrixU{P,T,L}
    ptr::Ptr{T}
end
mutable struct MutableSymmetricMatrixU{P,T,L} <: AbstractMutableSymmetricMatrixU{P,T,L}
    data::NTuple{L,T}
    MutableSymmetricMatrixU{P,T,L}(::UndefInitializer) where {P,T,L} = new{P,T,L}()
end
@generated function MutableSymmetricMatrixL{P,T}(undef) where {P,T}
    Lbase = binomial2(P+1)
    W = VectorizationBase.pick_vector_width(Lbase,T)
    Wm1 = W - 1
    L = (Lbase + Wm1) & ~Wm1
    quote
        $(Expr(:meta,:inline))
        MutableSymmetricMatrixL{$P,$T,$L}(undef)
    end
end
@generated function MutableSymmetricMatrixU{P,T}(undef) where {P,T}
    Lbase = binomial2(P+1)
    W = VectorizationBase.pick_vector_width(Lbase,T)
    Wm1 = W - 1
    L = (Lbase + Wm1) & ~Wm1
    quote
        $(Expr(:meta,:inline))
        MutableSymmetricMatrixU{$P,$T,$L}(undef)
    end
end

const AbstractMutableDiagMatrix{P,T,L} = Union{
    MutableLowerTriangularMatrix{P,T,L},
    MutableUpperTriangularMatrix{P,T,L},
    MutableSymmetricMatrixL{P,T,L},
    MutableSymmetricMatrixU{P,T,L},
    PtrLowerTriangularMatrix{P,T,L},
    PtrUpperTriangularMatrix{P,T,L},
    PtrSymmetricMatrixL{P,T,L},
    PtrSymmetricMatrixU{P,T,L}
}

@inline function Base.getindex(A::AbstractDiagTriangularMatrix{P,T,L}, i::Integer) where {P,T,L}
    @boundscheck i > L && ThrowBoundsError("i = $i > L = $L")
    @inbounds A.data[i]
end
@inline function Base.getindex(A::LinearAlgebra.Adjoint{T,<:AbstractDiagTriangularMatrix{P,T,L}}, i::Integer) where {P,T,L}
    @boundscheck i > L && ThrowBoundsError("i = $i > L = $L")
    @inbounds A.parent.data[i]
end
@inline function Base.getindex(A::LinearAlgebra.Adjoint{Union{},<:AbstractDiagTriangularMatrix{P,Vec{W,T},L}}, i::Integer) where {P,T,L,W}
    @boundscheck i > L && ThrowBoundsError("i = $i > L = $L")
    @inbounds A.parent.data[i]
end

const MutableDiagTriangle{P,T,L} = Union{
    MutableLowerTriangularMatrix{P,T,L},
    MutableUpperTriangularMatrix{P,T,L},
    MutableSymmetricMatrixL{P,T,L},
    MutableSymmetricMatrixU{P,T,L}
}

@inline Base.pointer(A::MutableDiagTriangle{P,T,L}) where {P,T,L} = Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
@inline function Base.pointer(A::MutableDiagTriangle{P,NTuple{W,Core.VecElement{T}},L}) where {P,T,L,W}
    Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
end
@inline Base.pointer(A::PtrLowerTriangularMatrix) = A.ptr
@inline Base.pointer(A::PtrUpperTriangularMatrix) = A.ptr
@inline Base.pointer(A::PtrSymmetricMatrixL) = A.ptr
@inline Base.pointer(A::PtrSymmetricMatrixU) = A.ptr

@inline function Base.getindex(A::AbstractMutableDiagMatrix{P,T,L}, i::Integer) where {P,T,L}
    @boundscheck i > L && ThrowBoundsError("i = $i > L = $L")
    VectorizationBase.load(pointer(A) + sizeof(T) * (i-1))
end
@inline function Base.getindex(A::LinearAlgebra.Adjoint{T,<:AbstractMutableDiagMatrix{P,T,L}}, i::Integer) where {P,T,L}
    @boundscheck i > L && ThrowBoundsError("i = $i > L = $L")
    VectorizationBase.load(pointer(A.parent) + sizeof(T) * (i-1))
end
@inline function Base.getindex(A::AbstractMutableDiagMatrix{P,Vec{W,T},L}, i::Integer) where {P,W,T,L}
    @boundscheck i > L && ThrowBoundsError("i = $i > L = $L")
    SIMDPirates.vload(Vec{W,T}, pointer(A) + W*sizeof(T)*(i-1))
end
@inline function Base.getindex(A::LinearAlgebra.Adjoint{Union{},<:AbstractMutableDiagMatrix{P,Vec{W,T},L}}, i::Integer) where {P,W,T,L}
    @boundscheck i > L && ThrowBoundsError("i = $i > L = $L")
    SIMDPirates.vload(Vec{W,T}, pointer(A.parent) + W*sizeof(T)*(i-1))
end


@inline function Base.setindex!(A::AbstractMutableDiagMatrix{P,T,L}, v, i::Integer) where {P,T,L}
    @boundscheck i > L && ThrowBoundsError("i > $L.")
    VectorizationBase.store!(pointer(A) + (i-1) * sizeof(T), convert(T,v))
    v
end
@inline function Base.setindex!(A::AbstractMutableDiagMatrix{P,NTuple{W,Core.VecElement{T}},L}, v::NTuple{W,Core.VecElement{T}}, i::Integer) where {P,T,L,W}
    @boundscheck i > L && ThrowBoundsError("i > $L.")
    SIMDPirates.vstore!(pointer(A) + (i-1) * sizeof(NTuple{W,Core.VecElement{T}}), v)
    v
end

@inline function Base.setindex!(A::AbstractMutableLowerTriangularMatrix{P,T,L}, v, i::Integer, j::Integer) where {P,T,L}
    ind = lt_sub2ind(P, i, j)
    @boundscheck i > L && ThrowBoundsError("i > $L.")
    VectorizationBase.store!(pointer(A) + (ind-1) * sizeof(T), convert(T,v))
    v
end
@inline function Base.setindex!(A::AbstractMutableLowerTriangularMatrix{P,NTuple{W,Core.VecElement{T}},L}, v::NTuple{W,Core.VecElement{T}}, i::Integer, j::Integer) where {P,T,L,W}
    ind = lt_sub2ind(P, i, j)
    @boundscheck i > L && ThrowBoundsError("i > $L.")
    SIMDPirates.vstore!(pointer(A) + (ind-1) * sizeof(NTuple{W,Core.VecElement{T}}), v)
    v
end
@inline function Base.setindex!(A::AbstractMutableUpperTriangularMatrix{P,T,L}, v, i::Integer, j::Integer) where {P,T,L}
    ind = lt_sub2ind(P, i, j)
    @boundscheck i > L && ThrowBoundsError("i > $L.")
    VectorizationBase.store!(pointer(A) + (ind-1) * sizeof(T), convert(T,v))
    v
end
@inline function Base.setindex!(A::AbstractMutableUpperTriangularMatrix{P,NTuple{W,Core.VecElement{T}},L}, v::NTuple{W,Core.VecElement{T}}, i::Integer, j::Integer) where {P,T,L,W}
    ind = lt_sub2ind(P, i, j)
    @boundscheck i > L && ThrowBoundsError("i > $L.")
    SIMDPirates.vstore!(pointer(A) + (ind-1) * sizeof(NTuple{W,Core.VecElement{T}}), v)
    v
end

#
# struct LowerTriangleColumnIterator{MT}
#     num_col::Int
#     W::Int
#     first_mask::MT
#     last_mask::MT
#     max_mask::MT
# end
# function LowerTriangleColumnIterator(::Val{P}, ::Type{T}) where {P,T}
#     W, Wshift = pick_vector_width_shift(P, T)
#     Wm1 = W - 1
#     MT = VectorizationBase.mask_type(Val(P), T)
#
#     max_mask = MT((1 << Wshift) - 1)
#     LowerTriangleColumnIterator(P, W, first_mask, last_mask, max_mask)
# end
# # Base.eltype(::Type{LowerTriangleColumnIterator}) =
# Base.length(i::LowerTriangleColumnIterator) = i.num_col - 1
#
# function Base.iterate(t::LowerTriangleColumnIterator)
#
# end
# function Base.iterate(t::LowerTriangleColumnIterator, state)
#
# end

@generated type_length(::AbstractDiagTriangularMatrix{P}) where {P} = binomial2(P+1)
@generated param_type_length(::AbstractDiagTriangularMatrix{P}) where {P} = binomial2(P+1)

Base.size(::AbstractDiagTriangularMatrix{P}) where {P} = (P,P)

@inline VectorizationBase.vectorizable(A::AbstractMutableDiagMatrix) = VectorizationBase.vpointer(pointer(A))
@inline VectorizationBase.vectorizable(A::AbstractDiagTriangularMatrix) = PaddedMatrices.vStaticPaddedArray(A, 0)


# @inline binom2(n::Int) = (nu = reinterpret(UInt, n); reinterpret(Int, (nu*(nu-one(UInt))) >> one(UInt)))

# function padded_diagonal_length(P, T)
#     Wm1 = VectorizationBase.pick_vector_width(P, T) - 1
#     (P + Wm1) & ~Wm1
# end
# function num_complete_blocks(P, T)
#     W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
#     (P - 1) >> Wshift
# end
# function calculate_L(P, T)
#     W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
#     W² = W*W
#     Wm1 = W - 1
#     rem = P & Wm1
#     padded_diag_length = (P + Wm1) & ~Wm1
#     L = padded_diag_length
#
#     Pm1 = P - 1
#     num_complete_blocks = Pm1 >> Wshift
#     L += W² * binomial(1+num_complete_blocks, 2)
#     rem_block = Pm1 & Wm1
#     L += (1+num_complete_blocks)*W*rem_block
#     L
# end
#
# function lower_triangle_sub2ind_quote(P, T)
#     # assume inputs are i, j
#     # for row, column
#     W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
#     W² = W * W
#     Wm1 = W - 1
#     rem = P & Wm1
#     if rem == 0
#         pad = 0
#         diag_length = P
#         num_rows = P
#         rem_row = Wm1
#     else
#         pad = W - rem
#         diag_length = P + pad
#         if rem == 1
#             num_rows = P - rem
#             rem_row = W
#         else
#             num_rows = diag_length
#             rem_row = rem - 1
#         end
#         # num_rows = rem == 1 ? P - rem : diag_length
#         # rem_row = rem - 1
#     end
#     # @show diag_length, num_rows, rem_row
#     quote
#         # We want i >= j, for a lower triangular matrix.
#         j, i = minmax(j, i)
#         # @boundscheck i > $P && ThrowBoundsError(str)
#         i == j && return i
#         ind = $(P - num_rows + Wm1 - rem_row) + j*$num_rows + i
#         if j > $rem_row
#             j2 = j - $rem_row
#             jrem = j2 & $Wm1
#             number_blocks = 1+(j2 >> $Wshift)
#             ind -= jrem * $W * number_blocks + binomial2(number_blocks) << $(2Wshift)
#         end
#         ind
#     end
# end
# @generated function lower_triangle_sub2ind(::Val{P}, ::Type{T}, i, j) where {T,P}
#     quote
#         # $(Expr(:meta, :inline))
#         $(lower_triangle_sub2ind_quote(P, T))
#     end
# end
#
# function number_lower_triangular_vector_loads_for_column(P, T, j)
#     W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
# end
#
# function upper_triangle_sub2ind_quote(P, T)
#     # assume inputs are i, j
#     # for row, column
#     W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
#     Wm1 = W - 1
#     rem = P & Wm1
#     diag_length = (P + Wm1) & ~Wm1
#     quote
#         # We want i <= j, for a upper triangular matrix.
#         i, j = minmax(i, j)
#         i == j && return i
#         num_blocks = (j-2) >> $Wshift
#         ind = $diag_length + ((num_blocks*(num_blocks+1)) << $(2Wshift - 1))
#         jrem = (j-2) & $(W-1)
#         ind += ((num_blocks+1) * jrem) << $Wshift
#         ind + i
#     end
# end
# @generated function upper_triangle_sub2ind(::Val{P}, ::Type{T}, i, j) where {T,P}
#     quote
#         # $(Expr(:meta, :inline))
#         $(upper_triangle_sub2ind_quote(P, T))
#     end
# end

function lt_sub2ind(P, i, j)
    i == j && return i
    j, i = minmax(i, j)
    j * P - binomial2(j) + i - j
end
function ut_sub2ind(P, i, j)
    i == j && return i
    i, j = minmax(i, j)
    1 + P + binomial2(j) + i - j
end


@generated function Base.:+(A::AbstractLowerTriangularMatrix{M,T,L}, B::AbstractLowerTriangularMatrix{M,T,L}) where {M,T,L}
    quote
        $(Expr(:meta,:inline))
        C = MutableLowerTriangularMatrix{$M,$T,$L}(undef)
        @vectorize $T for l ∈ 1:$L
            C[l] = A[l] + B[l]
        end
        C
    end
end
@generated function Base.:+(A::AbstractUpperTriangularMatrix{M,T,L}, B::AbstractUpperTriangularMatrix{M,T,L}) where {M,T,L}
    quote
        $(Expr(:meta,:inline))
        C = MutableUpperTriangularMatrix{$M,$T,$L}(undef)
        @vectorize $T for l ∈ 1:$L
            C[l] = A[l] + B[l]
        end
        C
    end
end
@generated function Base.:+(A::UpperTriangularMatrix{M,T,L}, B::UpperTriangularMatrix{M,T,L}) where {M,T,L}
    quote
        $(Expr(:meta,:inline))
        C = MutableUpperTriangularMatrix{$M,$T,$L}(undef)
        @vectorize $T for l ∈ 1:$L
            C[l] = A[l] + B[l]
        end
        UpperTriangularMatrix(C)
    end
end



# function diagonal_lowertri_output_tuple_expr(P, W, diagname = :diag_, colname = :c_)
#     outtup = Expr(:tuple, )
#     for p ∈ 1:P
#         push!(outtup.args, :($(Symbol(diagname, p))))
#     end
#     for w ∈ P+1:W
#         push!(outtup.args, :(zero($T)))
#     end
#     for p ∈ 1:P-1
#         c_p = Symbol(colname, p)
#         for w ∈ 1:W
#             push!(outtup.args, :($c_p[$w].value))
#         end
#     end
#     outtup
# end

# @generated function Base.inv(L::LowerTriangularMatrix{P,T,L}) where {P,T,L}
#
# end
#
# @generated function invchol(S::SymmetricMatrixL{P,T,L}) where {P,T,L}
#
# end


# K = 12
# [upper_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [upper_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 13
# [upper_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [upper_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 14
# [upper_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [upper_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 15
# [upper_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [upper_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 16
# [upper_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [upper_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 17
# [upper_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [upper_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 18
# [upper_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [upper_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
#
#
#
# K = 12
# [lower_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [lower_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 13
# [lower_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [lower_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 14
# [lower_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [lower_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 15
# [lower_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [lower_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 16
# [lower_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [lower_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 17
# [lower_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [lower_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
# K = 18
# [lower_triangle_sub2ind(Val(K), Float64, i, j) for i ∈ 1:K, j ∈ 1:K]
# [lower_triangle_sub2ind(Val(K), UInt128, i, j) for i ∈ 1:K, j ∈ 1:K]
