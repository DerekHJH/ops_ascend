
#include "eye_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  EyeCustomTilingData tiling;
  const gert::RuntimeAttrs * eyeattrs = context->GetAttrs();
  const uint32_t num_rows = *(eyeattrs->GetAttrPointer<uint32_t>(0)),
                num_columns = *(eyeattrs->GetAttrPointer<uint32_t>(1)),
                matrix_size = num_rows * num_columns,
                inner_iter = std::min(num_rows, num_columns),
                core_num = platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetCoreNum();
  context->SetBlockDim(core_num);
  uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize(),
            blockLength = totalLength / core_num, size;
  ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
  if (dtype == ge::DataType::DT_DOUBLE) size = 8;
  else if (dtype == ge::DataType::DT_FLOAT16) size = 2;
  else size = 4;
  // 64字节对齐
  if (blockLength * size % 64)
      blockLength = (blockLength * size / 64 + 1) * 64 / size;
    
  tiling.set_blockLength(blockLength);
  tiling.set_formerNum(totalLength / blockLength);
  tiling.set_tailLength(totalLength % blockLength);
  tiling.set_numColumns(num_columns);
  tiling.set_matrixSize(matrix_size);
  tiling.set_innerIter(inner_iter);
  tiling.set_outerStep(matrix_size - (inner_iter-1)*(num_columns+1));
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
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
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class EyeCustom : public OpDef {
public:
    explicit EyeCustom(const char* name) : OpDef(name)
    {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_DOUBLE})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_DOUBLE})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("num_rows").Int();
        this->Attr("num_columns").AttrType(OPTIONAL).Int(0);
        this->Attr("batch_shape").ListInt();
        this->Attr("dtype").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(EyeCustom);
}
