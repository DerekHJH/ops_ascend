#include "register/tilingdata_base.h"


#ifndef EYE_TILING_H
#define EYE_TILING_H
namespace optiling {
BEGIN_TILING_DATA_DEF(EyeTilingData)
    TILING_DATA_FIELD_DEF(int32_t, num_elements_total);
    TILING_DATA_FIELD_DEF(int32_t, num_rows);
    TILING_DATA_FIELD_DEF(int32_t, num_columns);
    TILING_DATA_FIELD_DEF(int32_t, num_batches);
    TILING_DATA_FIELD_DEF(int32_t, num_elements_per_batch);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Eye, EyeTilingData)
}

#endif // EYE_TILING_H
