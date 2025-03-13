
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HeavisideTilingData)
TILING_DATA_FIELD_DEF(uint32_t, num_elements_per_core);
TILING_DATA_FIELD_DEF(uint32_t, num_tiles);
TILING_DATA_FIELD_DEF(uint32_t, num_elements_per_tile);
TILING_DATA_FIELD_DEF(uint32_t, num_elements_last_tile);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Heaviside, HeavisideTilingData)
}
