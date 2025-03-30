import tbe.dsl as tbe
from tbe import tvm
from tbe.common.register import register_op_compute
from tbe.common.utils import para_check


@register_op_compute("mat_mul")
def mat_mul_compute(x, y, bias, z, kernel_name="mat_mul"):
    """
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    """

    res = tbe.XXX(x, y, bias)
    return res

@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def mat_mul(x, y, bias, z, kernel_name="mat_mul"):
    """
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    """
    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")
    data_y = tvm.placeholder(y.get("shape"), dtype=y.get("dtype"), name="data_y")
    data_bias = tvm.placeholder(bias.get("shape"), dtype=bias.get("dtype"), name="data_bias")

    res = mat_mul_compute(data_x, data_y, data_bias, z, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, data_bias, res]}
    tbe.build(schedule, config)
    