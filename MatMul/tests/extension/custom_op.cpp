/**
*
* Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include "../common/pytorch_npu_helper.hpp"
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;
using namespace at;


at::Tensor my_op_impl_npu(const at::Tensor& self, const at::Tensor& gamma,const c10::optional<at::Tensor> &bais) {
    int index;
    if(self.sizes().size()==2){
    index=0;
    }
    else{
    index=1;
    }
    int64_t viewDims[] = {self.sizes()[0], self.sizes()[index],gamma.sizes()[gamma.sizes().size()-1]}; 
		at::Tensor result = at::empty(viewDims, at::TensorOptions().dtype(self.dtype()).device(at::kCPU));
    
		EXEC_NPU_CMD(aclnnMatMul, self, gamma,bais,result);
		return result;
}



// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
		m.def("my_op(Tensor self, Tensor gamma,Tensor? gamma) -> Tensor");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
		m.impl("my_op", &my_op_impl_npu);
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("custom_op", &my_op_impl_npu, "tf.where");
}
