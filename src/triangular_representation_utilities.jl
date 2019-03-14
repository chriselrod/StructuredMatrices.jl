abstract type AbstractDiagTriangularMatrix{P,T,L} <: AbstractMatrix{T} end



abstract type AbstractTriangularMatrix{P,T,L} <: AbstractDiagTriangularMatrix{P,T,L} end
abstract type AbstractLowerTriangularMatrix{P,T,L} <: AbstractTriangularMatrix{P,T,L} end
abstract type AbstractUpperTriangularMatrix{P,T,L} <: AbstractTriangularMatrix{P,T,L} end

struct LowerTriangularMatrix{P,T,L} <: AbstractLowerTriangularMatrix{P,T,L}
    data::NTuple{L,T}
end
mutable struct MutableLowerTriangularMatrix{P,T,L} <: AbstractLowerTriangularMatrix{P,T,L}
    data::NTuple{L,T}
end
struct UpperTriangularMatrix{P,T,L} <: AbstractUpperTriangularMatrix{P,T,L}
    data::NTuple{L,T}
end
mutable struct MutableUpperTriangularMatrix{P,T,L} <: AbstractUpperTriangularMatrix{P,T,L}
    data::NTuple{L,T}
end


abstract type AbstractSymmetricMatrix{P,T,L} <: AbstractDiagTriangularMatrix{P,T,L} end
abstract type AbstractSymmetricMatrixL{P,T,L} <: AbstractSymmetricMatrix{P,T,L} end
abstract type AbstractSymmetricMatrixU{P,T,L} <: AbstractSymmetricMatrix{P,T,L} end

struct SymmetricMatrixL{P,T,L} <: AbstractSymmetricMatrixL{P,T,L}
    data::NTuple{L,T}
end
mutable struct MutableSymmetricMatrixL{P,T,L} <: AbstractSymmetricMatrixL{P,T,L}
    data::NTuple{L,T}
end

struct SymmetricMatrixU{P,T,L} <: AbstractSymmetricMatrix{P,T,L}
    data::NTuple{L,T}
end
mutable struct MutableSymmetricMatrixU{P,T,L} <: AbstractSymmetricMatrix{P,T,L}
    data::NTuple{L,T}
end

@inline function Base.getindex(A::AbstractDiagTriangularMatrix{P,T,L}, i::Integer) where {P,T,L}
    @boundscheck i > L && ThrowBoundsError("i = $i > L = $L")
    @inbounds A.data[i]
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


Base.size(::AbstractDiagTriangularMatrix{P}) where {P} = (P,P)


@generated function VectorizationBase.vectorizable(A::M) where {T, P, M <: AbstractDiagTriangularMatrix{P,T}}
    if M.mutable
        return quote
            :(Expr(:meta, :inline))
            VectorizationBase.vpointer(Base.unsafe_convert(Ptr{$T}, pointer_from_objref(A)))
        end
    else
        return quote
            :(Expr(:meta, :inline))
            PaddedMatrices.vStaticPaddedArray(A, 0)
        end
    end
end


@inline binomial2(n) = (n*(n-1)) >> 1

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
