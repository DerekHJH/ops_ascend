
#include "heaviside_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <numeric>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    HeavisideTilingData tiling;
    /*
        Start preparing information for tiling
    */
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto dtype = context->GetInputTensor(0)->GetDataType();
    switch (dtype) {
        case ge::DT_FLOAT16: size_of_dtype = 2; break;
        case ge::DT_FLOAT:   size_of_dtype = 4; break;
        default: size_of_dtype = 4; // Default to float
    }

    auto num_cores = ascendcPlatform.GetCoreNumAiv();
    context->SetBlockDim(num_cores);

    // Set the largest memory unit: UB and the smallest memory unit: block
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    uint64_t block_size = 32;
    uint64_t ub_size_aligned = ub_size / block_size * block_size;
    uint64_t ub_block_num = ub_size / block_size;
    uint64_t num_elements_per_block = block_size / size_of_dtype;
    /*
        End preparing information for tiling
    */

    uint64_t num_elements_total = context->GetInputTensor(0)->GetShapeSize();
    uint64_t align_ = num_elements_per_block * num_cores;
    uint64_t num_elements_total_aligned = ((num_elements_total + align_ - 1) / align_) * align_;

    uint64_t num_elements_per_core = num_elements_total_aligned / num_cores; // Evenly distributed thanks to align_
    tiling.set_num_elements_per_core(num_elements_per_core);

    // All tiles are properly aligned thanks to align_
    uint64_t num_tiles = (num_elements_per_core + ub_size_aligned - 1) / ub_size_aligned;
    uint64_t num_elements_per_tile = ub_size_aligned;
    uint64_t num_elements_last_tile = num_elements_per_core % ub_size_aligned;
    tiling.set_num_tiles(num_tiles);
    tiling.set_num_elements_per_tile(num_elements_per_tile);
    tiling.set_num_elements_last_tile(num_elements_last_tile);


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
        this->AICore().AddConfig("ascend910");

    }
};

OP_ADD(Heaviside);
}
