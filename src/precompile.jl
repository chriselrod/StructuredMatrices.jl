function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(StructuredMatrices.binomial2),Int64})
    precompile(Tuple{typeof(StructuredMatrices.muladd_quote),Int64,Type{T} where T,Int64,Int64,Symbol})
    precompile(Tuple{typeof(StructuredMatrices.row_sum_add_quote),Int64,Type{T} where T,Symbol})
end
