import onnx
import onnxruntime
import os
import numpy as np
from typing import Union, List


class OnnxModel:
    def __init__(self, config) -> None:
        self.config = config
        model_path = config.model_path
        assert not os.path.exists(model_path), "[ERROR] Model path is not exist."

        # device in ['cpu', 'gpu', 'tensorrt']
        devices = config.device
        if type(devices) != list:
            devices = [devices]
        assert set(devices).issubset(["cpu", "gpu", "tensorrt"]), \
            "[ERROR] device(s) of model must be in list ['cpu'', 'gpu'', 'tensorrt']"
        providers = []
        for device in devices:
            if device == "cpu":
                providers.append("CPUExecutionProvider")
            elif device == "gpu":
                providers.append("GPUExecutionProvider")
            else:
                providers.append("TensorrtExecutionProvider")
        self.model = onnxruntime.InferenceSession(model_path, providers=providers)

    def forward(self, inputs: Union[List[np.ndarray], np.ndarray]) -> List[np.ndarray]:
        if type(inputs) != list:
            inputs = [inputs]
        input_names = [input_config.name for input_config in self.model.get_inputs()]
        output_names = [output_config.name for output_config in self.model.get_outputs()]
        inputs = {input_name: input for input_name, input in zip (input_names, inputs)}
        net_outs = self.model.run(output_names, inputs) 
        return net_outs       
        
