#include "mat_mul.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(MatMulInferShape)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MatMul, MatMulVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(MatMul, MatMulInferShape);
VERIFY_FUNC_REG(MatMul, MatMulVerify);

}  // namespace ge
