from asyncio.log import logger
import os
import tensorrt as trt
from collections import namedtuple, OrderedDict
from typing import Union, List
import numpy as np
import torch

class TensorRTModel:
    def __init__(self, config) -> None:
        self.config = config
        model_path = config.model_path
        assert not os.path.exists(model_path), "[ERROR] Model path is not exist."
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.model = runtime.deserialize_cuda_engine(f.read())
        bindings = OrderedDict()
        fp16 = False  # default updated below
        device = config.device
        for index in range(self.model.num_bindings):
            name = self.model.get_binding_name(index)
            dtype = trt.nptype(self.model.get_binding_dtype(index))
            shape = tuple(self.model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            if self.model.binding_is_input(index) and dtype == np.float16:
                fp16 = True
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        self.context = self.model.create_execution_context()
        self.batch_size = bindings['images'].shape[0]

    def forward(self, inputs: Union[List[np.ndarray], np.ndarray]) -> List[np.ndarray]:
        if type(inputs) != list:
            inputs = [inputs]
        inputs = [torch.from_numpy(input) for input in inputs]
        input_names = self.config.input_names
        output_names = self.config.output_names

        assert len(inputs) == len(input_names), f"len(input names) = {len(input_names)}, len(inputs) = {len(inputs)}"
        for input, input_name in zip(inputs, input_names):
            self.binding_addrs[input_name] = int(input.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[output_name].data for output_name in output_names]
        out = [ym.cpu().numpy() for ym in y]
        return out