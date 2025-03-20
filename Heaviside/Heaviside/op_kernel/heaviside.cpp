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
        this->ub_num_elements_per_tile = tiling_data.ub_num_elements_per_tile;
        this->ub_num_elements_per_repeat = tiling_data.ub_num_elements_per_repeat;
        this->ub_num_repeats_per_tile = tiling_data.ub_num_repeats_per_tile;

        int32_t start_idx = this->num_elements_per_core * AscendC::GetBlockIdx();
        this->num_real_elements_per_core = this->num_elements_total - start_idx;
        AscendC::printf("num_elements_total: %d\n", this->num_elements_total);
        AscendC::printf("num_elements_per_core: %d\n", this->num_elements_per_core);
        AscendC::printf("num_tiles: %d\n", this->num_tiles);
        AscendC::printf("ub_num_elements_per_tile: %d\n", this->ub_num_elements_per_tile);
        AscendC::printf("ub_num_elements_per_repeat: %d\n", this->ub_num_elements_per_repeat);
        AscendC::printf("ub_num_repeats_per_tile: %d\n", this->ub_num_repeats_per_tile);

        AscendC::printf("start_idx: %d\n", start_idx);
        if (this->num_real_elements_per_core > this->num_elements_per_core)
            this->num_real_elements_per_core = this->num_elements_per_core;
        AscendC::printf("num_real_elements_per_core: %d\n", this->num_real_elements_per_core);

        if (this->num_real_elements_per_core <= 0)
            return;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + start_idx, this->num_real_elements_per_core);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + start_idx, this->num_real_elements_per_core);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + start_idx, this->num_real_elements_per_core);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->ub_num_elements_per_tile * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->ub_num_elements_per_tile * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->ub_num_elements_per_tile * sizeof(DTYPE_Z));

        pipe.InitBuffer(buf0, this->ub_num_elements_per_tile * sizeof(DTYPE_X));
        this->buf0Local = buf0.Get<DTYPE_X>();
        AscendC::Duplicate(this->buf0Local, 0.0, this->ub_num_elements_per_tile);
        pipe.InitBuffer(buf1, this->ub_num_elements_per_tile * sizeof(DTYPE_X));
        this->buf1Local = buf1.Get<DTYPE_X>();
        AscendC::Duplicate(this->buf1Local, 1.0, this->ub_num_elements_per_tile);
        pipe.InitBuffer(mask, this->ub_num_elements_per_tile / sizeof(uint8_t));
        this->maskLocal = mask.Get<uint8_t>();
    }
    __aicore__ inline void Process()
    {
        int32_t start_idx;
        int32_t num_real_elements_per_tile;
        for (int32_t i = 0; i < this->num_tiles; i++) {
            start_idx = i * this->ub_num_elements_per_tile;
            num_real_elements_per_tile = this->num_real_elements_per_core - start_idx;
            if(num_real_elements_per_tile > this->ub_num_elements_per_tile) {
                num_real_elements_per_tile = this->ub_num_elements_per_tile;
            }
            AscendC::printf("start_idx: %d, num_real_elements_per_tile: %d\n", start_idx, num_real_elements_per_tile);

            if(num_real_elements_per_tile <= 0) {
                break; // All that is left are extra elements.
            }

            CopyIn(i, start_idx, num_real_elements_per_tile);
            Compute(i, start_idx, num_real_elements_per_tile);
            CopyOut(i, start_idx, num_real_elements_per_tile);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, int32_t start_idx, int32_t num_real_elements_per_tile)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        AscendC::DataCopy(xLocal, xGm[start_idx], num_real_elements_per_tile);
        AscendC::DataCopy(yLocal, yGm[start_idx], num_real_elements_per_tile);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress, int32_t start_idx, int32_t num_real_elements_per_tile)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        // We use this->ub_num_elements_per_tile to calculate the number of elements to process.
        // Because this->ub_num_elements_per_tile % this->num_elements_per_repeat == 0
        // num_real_elements_per_tile <= this->ub_num_elements_per_tile.
        this->maskLocal = xLocal < (this->buf0Local);
        AscendC::Select(zLocal, this->maskLocal, this->buf0Local, zLocal, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->ub_num_elements_per_tile);
        this->maskLocal = xLocal > (this->buf0Local);
        AscendC::Select(zLocal, this->maskLocal, this->buf1Local, zLocal, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->ub_num_elements_per_tile);
        this->maskLocal = xLocal == (this->buf0Local);
        AscendC::Select(zLocal, this->maskLocal, yLocal, zLocal, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->ub_num_elements_per_tile);

        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, int32_t start_idx, int32_t num_real_elements_per_tile)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[start_idx], zLocal, num_real_elements_per_tile);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> buf0;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> buf1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> mask;
    AscendC::LocalTensor<DTYPE_X> buf0Local;
    AscendC::LocalTensor<DTYPE_X> buf1Local;
    AscendC::LocalTensor<DTYPE_X> maskLocal;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    int32_t num_elements_total;
    int32_t num_elements_per_core;
    int32_t num_tiles;
    int32_t ub_num_elements_per_tile;
    int32_t ub_num_elements_per_repeat;
    int32_t ub_num_repeats_per_tile;

    int32_t num_real_elements_per_core;
};

extern "C" __global__ __aicore__ void heaviside(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    Heaviside op;
    op.Init(x, y, z, tiling);
    op.Process();
}

