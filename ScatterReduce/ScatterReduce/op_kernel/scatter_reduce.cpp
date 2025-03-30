#include "kernel_operator.h"

extern "C" __global__ __aicore__ void scatter_reduce(GM_ADDR self, GM_ADDR index, GM_ADDR src, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}