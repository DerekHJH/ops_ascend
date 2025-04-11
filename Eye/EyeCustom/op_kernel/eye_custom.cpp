#include "kernel_operator.h"
using namespace AscendC;

__global__ __aicore__ void eye_custom(GM_ADDR self, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    GlobalTensor<DTYPE_SELF> yGm;
    const int64_t block_id = GetBlockIdx();
    const uint32_t inner_step = tiling_data.numColumns + 1,
            inner_iter = tiling_data.innerIter, outter_step = tiling_data.outerStep;
    uint32_t p = block_id * tiling_data.blockLength % tiling_data.matrixSize, inner_id = 0, len = 0;
    // 如果p位置所属于的矩阵已完成 Eye 矩阵的填充，就跳到下一个矩阵的开头
    if (p > (inner_iter-1) * inner_step) p = tiling_data.matrixSize - p;
    else {
        // 跳到下一个应该填1的位置
        inner_id = (p + tiling_data.numColumns) / inner_step;
        p = inner_id * inner_step - p;
    }
    if (block_id < tiling_data.formerNum)
        len = tiling_data.blockLength;
    else if (block_id == tiling_data.formerNum)
        len = tiling_data.tailLength;
    else return;
    yGm.SetGlobalBuffer((__gm__ DTYPE_SELF *)y + tiling_data.blockLength * block_id, len);
    while (p < len) {
        yGm.SetValue(p, 1);
        inner_id++;
        if (inner_id == inner_iter) {
            inner_id = 0;
            p += outter_step;
        } else p += inner_step;
    }
}