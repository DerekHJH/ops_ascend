
#include "heaviside_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"

namespace optiling {
constexpr uint64_t BUFFER_NUM = 2;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t INPUT_NUM = 2;
constexpr uint64_t OUTPUT_NUM = 1;
constexpr uint64_t TEMP_NUM = 1;
constexpr uint64_t VECTOR_NUM = (INPUT_NUM + OUTPUT_NUM) * BUFFER_NUM + TEMP_NUM;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    HeavisideTilingData tiling;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t size_of_dtype;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), size_of_dtype);
    printf("size_of_dtype: %u\n", size_of_dtype);
    auto num_cores = ascendcPlatform.GetCoreNumAiv();
    printf("num_cores: %llu\n", num_cores);
    context->SetBlockDim(num_cores);
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    printf("ub_size: %llu\n", ub_size);
    uint64_t ub_num_blocks = ub_size / BLOCK_SIZE; 
    printf("ub_num_blocks: %llu\n", ub_num_blocks);
    uint64_t ub_num_elements_per_block = BLOCK_SIZE / size_of_dtype;
    printf("ub_num_elements_per_block: %llu\n", ub_num_elements_per_block);
    uint64_t ub_num_blocks_per_tile = ub_num_blocks / VECTOR_NUM;
    printf("ub_num_blocks_per_tile: %llu\n", ub_num_blocks_per_tile);
    uint64_t num_elements_per_tile = ub_num_blocks_per_tile * ub_num_elements_per_block;
    tiling.set_num_elements_per_tile(num_elements_per_tile);
    printf("num_elements_per_tile: %llu\n", num_elements_per_tile);

    uint64_t num_elements_total = context->GetInputTensor(0)->GetShapeSize();
    tiling.set_num_elements_total(num_elements_total);
    printf("num_elements_total: %llu\n", num_elements_total);
    uint64_t align_ = num_cores * ub_num_elements_per_block;
    printf("align_: %u\n", align_);
    uint64_t num_elements_per_core = (num_elements_total + align_ - 1) / align_ * align_ / num_cores; // We might calc extra elements
    tiling.set_num_elements_per_core(num_elements_per_core);
    printf("num_elements_per_core: %llu\n", num_elements_per_core);
    uint64_t num_tiles = (num_elements_per_core + num_elements_per_tile - 1) / num_elements_per_tile;
    tiling.set_num_tiles(num_tiles);
    printf("num_tiles: %llu\n", num_tiles);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class Heaviside : public OpDef {
public:
    explicit Heaviside(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Heaviside);
}
