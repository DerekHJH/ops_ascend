#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelSinh {
public:
    __aicore__ inline KernelSinh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        ASSERT(tileNum != 0 && "tile num can not be zero!");
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * GetBlockIdx(), 
        this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * GetBlockIdx(), 
        this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuffer1, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuffer3, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuffer4, this->tileLength * sizeof(DTYPE_X));
    }
    __aicore__ inline void Process()
    {
        //考生补充对“loopCount”的定义，注意对Tiling的处理
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        LocalTensor<DTYPE_X> tmpTensor1 = tmpBuffer1.Get<DTYPE_X>();
        LocalTensor<DTYPE_X> tmpTensor2 = tmpBuffer2.Get<DTYPE_X>();
        LocalTensor<DTYPE_X> tmpTensor3 = tmpBuffer3.Get<DTYPE_X>();
        LocalTensor<DTYPE_X> tmpTensor4 = tmpBuffer4.Get<DTYPE_X>();
        DTYPE_X inputVal1 = -1;
        DTYPE_X inputVal2 = 0.5;
        //sinh(x) = (exp(x) - exp(-x)) / 2.0
        Muls(tmpTensor1, xLocal, inputVal1, this->tileLength);
        Exp(tmpTensor2, tmpTensor1, this->tileLength);
        Exp(tmpTensor3, xLocal, this->tileLength);
        Sub(tmpTensor4, tmpTensor3, tmpTensor2, this->tileLength);
        Muls(yLocal, tmpTensor4, inputVal2, this->tileLength);
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    //create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;

    TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2, tmpBuffer3, tmpBuffer4;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;

};

extern "C" __global__ __aicore__ void sinh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSinh op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
