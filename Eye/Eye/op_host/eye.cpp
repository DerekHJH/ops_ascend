
#include "eye_tiling.h"
#include "register/op_def_registry.h"

#define BLOCK_DIM 1
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // For eye, we finish all work on one ai core.
    context->SetBlockDim(BLOCK_DIM);

    EyeTilingData tiling;
    uint32_t num_elements_total = context->GetInputTensor(0)->GetShapeSize();

    const int64_t* p_num_rows = context->GetAttrs()->GetInt(0);
    const int64_t* p_num_columns = context->GetAttrs()->GetInt(1);

    int32_t num_rows = *p_num_rows;
    int32_t num_columns = *p_num_columns;
    if (num_columns == 0) {
        num_columns = num_rows;
    }

    int32_t num_batches = 1;
    int32_t num_elements_per_batch = num_rows * num_columns;
    if (context->GetInputTensor(0)->GetOriginShape().GetDimNum() > 2) {
        num_batches = num_elements_total / num_elements_per_batch;
    }

    tiling.set_num_rows(num_rows);
    tiling.set_num_columns(num_columns);
    tiling.set_num_batches(num_batches);
    tiling.set_num_elements_per_batch(num_elements_per_batch);

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
class Eye : public OpDef {
public:
    explicit Eye(const char* name) : OpDef(name)
    {
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_DOUBLE})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_DOUBLE})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("num_rows").Int();
        this->Attr("num_columns").AttrType(OPTIONAL).Int(0);
        this->Attr("batch_shape").AttrType(OPTIONAL).ListInt({});
        this->Attr("dtype").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Eye);
}
