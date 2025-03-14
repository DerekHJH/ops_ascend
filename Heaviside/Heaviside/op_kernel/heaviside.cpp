#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 2;
class Heaviside {
public:
    __aicore__ inline Heaviside() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tiling)
    {
        GET_TILING_DATA(tiling_data, tiling);
        this->num_elements_total = tiling_data.num_elements_total;
        this->num_elements_per_core = tiling_data.num_elements_per_core;
        this->num_tiles = tiling_data.num_tiles;
        this->num_elements_per_tile = tiling_data.num_elements_per_tile;
        this->num_elements_per_buffer = tiling_data.num_elements_per_buffer;
        uint32_t start_address = this->num_elements_per_core * AscendC::GetBlockIdx();
        uint32_t num_real_elements = min(this->num_elements_per_core, this->num_elements_total - start_address);
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + start_address, num_real_elements);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + start_address, num_real_elements);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + start_address, num_real_elements);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->num_elements_per_buffer * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->num_elements_per_buffer * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->num_elements_per_buffer * sizeof(DTYPE_Z));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->num_tiles * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        uint32_t
        AscendC::DataCopy(xLocal, xGm[progress * this->num_elements_per_buffer], this->num_elements_per_buffer);
        AscendC::DataCopy(yLocal, yGm[progress * this->num_elements_per_buffer], this->num_elements_per_buffer);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::Add(zLocal, xLocal, yLocal, this->num_elements_per_buffer);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->num_elements_per_buffer], zLocal, this->num_elements_per_buffer);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t num_elements_total;
    uint32_t num_elements_per_core;
    uint32_t num_tiles;
    uint32_t num_elements_per_tile;
    uint32_t num_elements_per_buffer;
};

extern "C" __global__ __aicore__ void heaviside(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    Heaviside op;
    op.Init(x, y, z, tiling);
    op.Process();
}

