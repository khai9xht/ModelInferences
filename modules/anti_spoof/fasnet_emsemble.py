from .fasnet import MiniFASNet
from modules.configs import FASNetConfig
from typing import List
import numpy as np
from typing import Union
from modules.model_server import TritonClient
import enum

class FaceLive(enum.Enum):
	REAL = 0
	FAKE = 1

class MiniFASNetEmsemble:
	def __init__(self, configs: List[FASNetConfig], \
			triton_client: TritonClient = None) -> None:
		self.mini_fasnet_v1se = MiniFASNet(configs[0], triton_client)
		self.mini_fasnet_v2 = MiniFASNet(configs[1], triton_client)
		self.threshold = (configs[0].threshold + configs[1].threshold) / 2
	
	def check(self, image: np.ndarray, bbox: np.ndarray) -> Union[FaceLive, float]:
		img_crop1 = self.mini_fasnet_v1se.align_bbox(image, bbox)
		img_crop2 = self.mini_fasnet_v2.align_bbox(image, bbox)
		result1 = self.mini_fasnet_v1se.forward(img_crop1)
		result2 = self.mini_fasnet_v2.forward(img_crop2)
		results = (result1 + result2) / 2
		label = np.argmax(results)
		score = results[label]

		if label == 1 and score >= self.threshold:
			label_name = FaceLive.REAL
		else:
			label_name = FaceLive.FAKE

		return label_name, score