import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  
import threading
from typing import Optional, Tuple
case_data = {
    'case1': {
        'input_shape': [32],
        'data_type': np.float32,
        'values_shape': [32]
    }
}
def run_with_timeout(func, args=(), kwargs={}, timeout=30):
    result = []
    def target():
        try:
            result.append(func(*args, **kwargs))
        except Exception as e:
            result.append(e)
            print("函数执行异常:",e)
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return None
    if isinstance(result[0], Exception):
        raise result[0]
    return result[0]

def verify_result(real_result, golden):
      # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
    if golden.dtype == np.float16:
        loss = 1e-3
    else:
        loss = 1e-4
    
    minimum = 10e-10
    result = np.abs(real_result - golden)  # 计算运算结果和预期结果偏差

    deno = np.maximum(np.abs(real_result), np.abs(golden))  # 获取最大值并组成新数组
    result_atol = np.less_equal(result, loss)  # 计算绝对误差
    result_rtol = np.less_equal(result / np.add(deno, minimum), loss)  # 计算相对误差
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == False) > real_result.size * loss and np.sum(result_atol == False) > real_result.size * loss:  # 误差超出预期时返回打印错误，返回对比失败
            print("[ERROR] result error")
            return False
    print("test pass")
    return True

class TestCustomOP(TestCase):
    def test_custom_op_case(self,num):
        print(num)
        caseNmae='case'+num
        tensor_input = np.random.uniform(1, 100,case_data[caseNmae]['input_shape']).astype(case_data[caseNmae]['data_type'])
        tensor_values = np.random.uniform(1, 100,case_data[caseNmae]['values_shape']).astype(case_data[caseNmae]['data_type'])

        golden = torch.heaviside(torch.from_numpy(tensor_input), torch.from_numpy(tensor_values)).numpy()
        
        
        tensor_input_npu = torch.from_numpy(tensor_input).npu()
        tensor_values_npu = torch.from_numpy(tensor_values).npu()
        

        # 修改输入
        

        output = run_with_timeout(custom_ops_lib.custom_op, args=(tensor_input_npu, tensor_values_npu), timeout=30)
        if output is None:
            print(f" {caseNmae} execution timed out!")
        else:
            output = output.cpu().numpy()
            if verify_result(output, golden):
                print(f"{caseNmae} verify result pass!")
            else:
                print(f"{caseNmae} verify result failed!")

if __name__ == "__main__":
    print(sys.argv)
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
