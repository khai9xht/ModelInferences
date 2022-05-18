from .base_config import Config


class PaddleRecognizerConfig(Config):
	@property
	def rec_algorithm(self):
			return self.config["rec_algorithm"]

	@property
	def rec_image_shape(self):
			return self.config["rec_image_shape"]

	@property
	def rec_char_type(self):
			return self.config["rec_char_type"]

	@property
	def rec_batch_num(self):
			return self.config["rec_batch_num"]

	@property
	def max_text_length(self):
			return self.config["max_text_length"]

	@property
	def rec_char_dict_path(self):
			return self.config["rec_char_dict_path"]

	@property
	def use_space_char(self):
			return self.config["use_space_char"]
	
	@property
	def input_names(self):
		return self.config["input_names"]
	
	@property
	def output_names(self):
		return self.config["output_names"]
	
	@property
	def model_name(self):
		return self.config["model_name"]
	
	@property
	def fields_domain(self):
		return self.config["fields_domain"]