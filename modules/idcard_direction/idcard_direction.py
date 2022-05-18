import numpy as np
from modules.configs import IDCardDirectionConfig
from modules.utils.face_detect import custom_resize
import time
from modules.model_server import TritonClient


class IDCardDirection:
	def __init__(self, config: IDCardDirectionConfig, triton_client: TritonClient = None) -> None:
		self.config = config
		input_sizes = [[1,3, config.input_width, config.input_height]]
		self.triton_client = triton_client
		self.triton_client.initialize(
			model_name=config.model_name,
			input_names=config.input_names,
			input_sizes=input_sizes,
			output_names=config.output_names
		)
		print(f"[{self.__class__.__name__}][{self.config.model_name}]: Using model on triton server.")
	
	def forward(self, input: np.ndarray):
		preds = self.triton_client.infer(
			model_name=self.config.model_name, 
			inputs=[input]
		)
		preds = [preds.as_numpy(output_name) for output_name in self.config.output_names]
		return preds

	def __call__(self, image: np.ndarray):
		start_time = time.time()
		input_size = (self.config.input_height, self.config.input_width)
		resized_img, _ = custom_resize(image, input_size)
		resized_img = np.transpose(resized_img, (2, 0, 1))
		exp_resized_img = np.expand_dims(resized_img, axis=0).astype(np.float32)
		results = self.forward(exp_resized_img)
		print(f"[{self.__class__.__name__}][{self.config.model_name}] results: {results}")
		print(f"[{self.__class__.__name__}][{self.config.model_name}]: Direction classify finished, inference time: {time.time()- start_time}")
		return results
		