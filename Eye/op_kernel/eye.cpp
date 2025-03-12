#include "kernel_operator.h"

extern "C" __global__ __aicore__ void eye(GM_ADDR y, GM_ADDR y_ref, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}