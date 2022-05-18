from .base_config import Config
from typing import List, Dict


class IDCardDirectionConfig(Config):
	def __init__(self, config_fp=""):
		super(IDCardDirectionConfig, self).__init__(config_fp)
	
	@property
	def model_name(self) -> str:
			return self.config["model_name"]

	@property
	def input_width(self) -> int:
			return self.config["input_width"]

	@property
	def input_height(self) -> int:
			return self.config["input_height"]

	@property
	def input_names(self) -> List[str]:
			return [str(input_name) for input_name in self.config["input_names"]]

	@property
	def output_names(self) -> List[str]:
			return [str(output_name) for output_name in self.config["output_names"]]

	@property
	def label_map(self) -> Dict[int, str]:
			return self.config["label_map"]