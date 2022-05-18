from .base_config import Config

class CraftConfig(Config):
	def __init__(self, config_fp="") -> None:
			super(CraftConfig, self).__init__(config_fp)
	
	@property
	def model_name(self):
		return self.config["model_name"]
	
	@property
	def canvas_size(self):
		return self.config["canvas_size"]
	
	@property
	def input_names(self):
		return self.config["input_names"]
	
	@property
	def output_names(self):
		return self.config["output_names"]
	
	@property
	def mag_ratio(self):
		return self.config["magnification_ratio"]

class RefinerConfig(Config):
	def __init__(self, config_fp="") -> None:
			super(RefinerConfig, self).__init__(config_fp)
	
	@property
	def model_name(self):
		return self.config["model_name"]
	
	@property
	def remote(self):
		return self.config["remote"]
	
	@property
	def model_path(self):
		return self.config["model_path"]
	
	@property
	def device(self):
		return self.config["device"]
	
	@property
	def canvas_size(self):
		return self.config["canvas_size"]
	
	@property
	def input_names(self):
		return self.config["input_names"]
	
	@property
	def output_names(self):
		return self.config["output_names"]
	
	@property
	def text_threshold(self):
		return self.config["text_threshold"]
	
	@property
	def link_threshold(self):
		return self.config["link_threshold"]
	
	@property
	def text_low(self):
		return self.config["text_lower_bound"]

class CraftRefinerConfig:
  def __init__(self, config: dict) -> None:
    self.config = config

  @property
  def craft(self):
    return CraftConfig(self.config["craft"])
  
  @property
  def refiner(self):
    return RefinerConfig(self.config["refiner"])