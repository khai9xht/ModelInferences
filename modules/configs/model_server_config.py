class ModelServerConfig:
	def __init__(self, config: dict) -> None: 
		self.config = config
	
	@property
	def host(self):
		return self.config["host"]
	
	@property
	def port(self):
		return self.config["port"]