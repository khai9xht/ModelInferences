from .base_config import Config

class ModelConfig(Config):
	def __init__(self, config_path) -> None:
		super(ModelConfig, self).__init__(config_path)
	
	@property
	def model_name(self):
		return self.config["model_name"]
	
	@property
	def input_width(self):
		return self.config["input_width"]
	
	@property
	def input_height(self):
		return self.config["input_height"]
	
	@property
	def input_names(self):
		return self.config["input_names"]
	
	@property
	def output_names(self):
		return self.config["output_names"]
