/* Round trips a scalar through a C API tensor map, from C. */

#include <stdio.h>
#include <stdlib.h>

#include <viam/trajex/capi/capi.h>

int main(void) {
    viam_trajex_tensor_map_t* tensor_map = viam_trajex_tensor_map_create();
    if (!tensor_map) {
        puts("capi: could not create a tensor map");
        return EXIT_FAILURE;
    }

    viam_trajex_dtype_t dtype;
    size_t rank;
    const size_t* dims;
    const void* data;
    const int inserted = viam_trajex_tensor_map_insert_scalar_f64(tensor_map, "value", 2.5);
    const int viewed = viam_trajex_tensor_map_view(tensor_map, "value", &dtype, &rank, &dims, &data);
    const int ok =
        inserted == 0 && viewed == 0 && dtype == VIAM_TRAJEX_DTYPE_F64 && rank == 1 && dims[0] == 1 && *(const double*)data == 2.5;
    viam_trajex_tensor_map_destroy(tensor_map);

    puts(ok ? "capi: round trip ok" : "capi: round trip failed");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
