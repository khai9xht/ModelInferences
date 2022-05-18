from __future__ import division
import numpy as np
import cv2, time
from modules.utils import face_align, face_detect
from modules.configs import ArcFaceConfig
from modules.model_server import TritonClient
class ArcFace:
	def __init__(self, config: ArcFaceConfig, triton_client: TritonClient) -> None:
		self.config = config
		self.input_size = tuple([config.input_width, config.input_height])
		self.output_shape = tuple([1, config.output_shape])
		input_sizes = [tuple([1, 3, config.input_width, config.input_height])]
		self.triton_client = triton_client
		self.triton_client.initialize(
			model_name=config.model_name,
			input_names=config.input_names,
			input_sizes=input_sizes,
			output_names=config.output_names
		)
		print(f"[{self.__class__.__name__}][{self.config.model_name}]: Using model on triton server.")
	
	def get(self, img: np.ndarray, face: face_detect.Face):
		aimg = face_align.norm_crop(img, landmark=face.kps)
		face.embedding = self.get_feat(aimg).flatten()
		return face.normed_embedding
	
	def get_feat(self, imgs: np.ndarray):
		if not isinstance(imgs, list):
			imgs = [imgs]
		input_size = self.input_size
		blob = cv2.dnn.blobFromImages(
			imgs, 1.0 / self.config.input_std, input_size,
			(self.config.input_mean, self.config.input_mean, self.config.input_mean), swapRB=True
		)
		results = self.triton_client.infer(self.config.model_name, [blob])
		net_out = [results.as_numpy(name) for name in self.config.output_names][0]
		print(f"[{self.__class__.__name__}][{self.config.model_name}]: Encode finished.")
		return net_out
	
	def compute_sim(self, feat1, feat2):
		feat1 = feat1.ravel()
		feat2 = feat2.ravel()
		sim = np.dot(feat1, feat2) 
		return sim

	def is_same(self, feat1, feat2):
		sim = self.compute_sim(feat1, feat2)
		if sim > self.config.threshold:
			return True
		else:
			return False