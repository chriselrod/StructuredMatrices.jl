function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(StructuredMatrices.muladd_quote),Int64,Type,Int64,Int64,Symbol})
    # precompile(Tuple{StructuredMatrices.var"##s54#33",Any,Any,Any,Any,Any,Any,Any})
    precompile(Tuple{typeof(StructuredMatrices.row_sum_add_quote),Int64,Type,Symbol})
    # precompile(Tuple{StructuredMatrices.var"##s54#41",Any,Any,Any,Any,Any,Any})
end
