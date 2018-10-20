
###
### The advantage of the diagonal versions is improved performance for operations
### like sqrt / exp / log of diagonal elements, which is O(1) -- asymptotically irrelevant.
### Therefore, I am not planning on bothering with implementing loops over kernels for
### diagonal matrices within the forseeable future.
### For now, for medium to large sizes, packed matrices are the way to go.
### The switch here simply decides which sort to return.
###
