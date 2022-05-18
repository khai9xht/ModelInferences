from __future__ import division
import numpy as np
from ..configs import FASNetConfig
from ..utils.face_anti import CropImage, softmax
from ..base_model.model_server import TritonClient
class MiniFASNet:
	def __init__(self, config: FASNetConfig, triton_client: TritonClient) -> None:
		self.__class__.__name__ = config.model_name
		self.config = config
		input_sizes = [tuple([1, 3, config.input_width, config.input_height])]
		self.triton_client = triton_client
		self.triton_client.initialize(
			model_name=config.model_name, 
			input_names=config.input_names, 
			input_sizes=input_sizes,
			output_names=config.output_names	
		)
		print(f"[{self.__class__.__name__}][{self.config.model_name}]: Using model on triton server.")
		self.cropper = CropImage()

	def forward(self, image: np.ndarray):
		image_cp = image.copy()
		image_cp = np.transpose(image_cp, (2, 0, 1))
		image_cp = np.expand_dims(image_cp, 0)
		results = self.triton_client.infer(self.config.model_name, [image_cp])
		net_outs = [results.as_numpy(name) for name in self.config.output_names]
		net_outs = softmax(net_outs[0][0])
		print(f"[{self.__class__.__name__}][{self.config.model_name}]: Checking finished.")
		return net_outs

	def align_bbox(self, image: np.ndarray, bbox):
		image_cp = image.copy()
		param = {
			"org_img": image_cp,
			"bbox": bbox,
			"scale": self.config.scale,
			"out_w": self.config.input_width,
			"out_h": self.config.input_height,
			"crop": True,
		}
		crop_img = self.cropper.crop(**param)
		return np.array(crop_img, dtype=np.float32)
	
	def check(self, image: np.ndarray, bbox: np.ndarray):
		crop_img = self.align_bbox(image, bbox)
		net_outs = self.forward(crop_img)
		label = np.argmax(net_outs)
		score = net_outs[label]
		if label == 1 and score >= self.config.threshold:
			return True
		return False