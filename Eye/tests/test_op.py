import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  
import threading
from typing import Optional, Tuple
test_data = {
     'case1': {
        'shape': [32, 32, 32, 32],
        'data_type': np.float16,
        'numRows': 32,
        'numColumns': 32,
        'batchShape': [32, 32],
        'dtype': 1,
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
    loss = 10e-6
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
        
        test_name = "case" + num
        tensor_self = np.zeros(test_data[test_name]['shape']).astype(test_data[test_name]['data_type'])
        numRows=np.array(test_data[test_name]['numRows']).astype(np.int32)
        dtype=np.array(test_data[test_name]['dtype']).astype(np.int32)
        numColumns=np.array(test_data[test_name]['numColumns']).astype(np.int32)
        batchShape=np.array(test_data[test_name]['batchShape']).astype(np.int32)
        
        res = torch.eye(n=test_data[test_name]['numRows'],m=test_data[test_name]['numColumns'])
        res = torch.broadcast_to(res, test_data[test_name]['shape'])
        golden=res.numpy().astype(test_data[test_name]['data_type'])

        tensor_self_npu = torch.from_numpy(tensor_self).npu()
        numColumns_npu = torch.from_numpy(numColumns).npu()
        numRows_npu = torch.from_numpy(numRows).npu()
        batchShape_npu=torch.from_numpy(batchShape).npu()
        batchShape_npu = tuple(batchShape_npu.tolist())
        dtype_npu=torch.from_numpy(dtype).npu()
        
        output = run_with_timeout(custom_ops_lib.custom_op, args=(tensor_self_npu, numRows_npu, numColumns_npu, batchShape_npu, dtype_npu), timeout=30)
        if output is None:
            print(f"{test_name} execution timed out!")
        else:
            print(type(output))
            output = output.cpu().numpy()
            if verify_result(output, golden):
                print(f"{test_name} verify result pass!")
            else:
                print(f"{test_name} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
