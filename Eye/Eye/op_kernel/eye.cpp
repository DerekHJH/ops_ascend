#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class Eye {
public:
    __aicore__ inline Eye() {}
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR y_ref, GM_ADDR tiling) {
        GET_TILING_DATA(tiling_data, tiling);
        this->num_elements_total = tiling_data.num_elements_total;
        this->num_rows = tiling_data.num_rows;
        this->num_columns = tiling_data.num_columns;
        this->num_batches = tiling_data.num_batches;
        this->num_elements_per_batch = tiling_data.num_elements_per_batch;
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y_ref +
            this->num_elements_total * AscendC::GetBlockIdx(), this->num_elements_total);
    }
    __aicore__ inline void Process() {
        int32_t index, t;
        for (int32_t i = 0; i < this->num_batches; i++) {
            for (int32_t j = 0; j < this->num_rows; j++) {
                if (j < this->num_columns) {
                    index = i * this->num_elements_per_batch + j * this->num_columns + j;
                    yGm.SetValue(index, 1);
                }
            }
        }
    }

private:
    uint32_t num_elements_total;
    int32_t num_rows;
    int32_t num_columns;
    int32_t num_batches;
    int32_t num_elements_per_batch;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
};

extern "C" __global__ __aicore__ void eye(GM_ADDR y, GM_ADDR y_ref, GM_ADDR workspace, GM_ADDR tiling) {
    Eye op;
    op.Init(y, y_ref, tiling);
    op.Process();
}