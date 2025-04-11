
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EyeCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, blockLength);
  TILING_DATA_FIELD_DEF(uint32_t, formerNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailLength);
  TILING_DATA_FIELD_DEF(uint32_t, numColumns);
  TILING_DATA_FIELD_DEF(uint32_t, matrixSize);
  TILING_DATA_FIELD_DEF(uint32_t, innerIter);
  TILING_DATA_FIELD_DEF(uint32_t, outerStep);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(EyeCustom, EyeCustomTilingData)
}
