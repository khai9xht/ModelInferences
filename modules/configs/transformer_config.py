from typing import List
from .base_config import Config


class VggCnn:
    def __init__(self, config: dict) -> None:
        self.config = config
    
    @property
    def model_name(self) -> str:
        return self.config["model_name"]

    @property
    def input_names(self) -> List[str]:
        return self.config["input_names"]
    
    @property
    def output_names(self) -> List[str]:
        return self.config["output_names"]

class Encoder:
    def __init__(self, config: dict) -> None:
        self.config = config   

    @property
    def model_name(self) -> str:
        return self.config["model_name"]

    @property
    def input_names(self) -> List[str]:
        return self.config["input_names"]
    
    @property
    def output_names(self) -> List[str]:
        return self.config["output_names"]
    
    @property
    def input_shape(self) -> List[int]:
        return self.config["input_shape"]    

class Decoder:
    def __init__(self, config: dict) -> None:
        self.config = config   

    @property
    def model_name(self) -> str:
        return self.config["model_name"]
        
    @property
    def input_names(self) -> List[str]:
        return self.config["input_names"]
    
    @property
    def output_names(self) -> List[str]:
        return self.config["output_names"]
    
    @property
    def input_shape(self) -> List[List[int]]:
        return self.config["input_shape"]  

class Seq2SeqConfig(Config): 
    def __init__(self, config_path: str) -> None:
        super(Seq2SeqConfig, self).__init__(config_path)
    @property
    def chars(self) -> str:
        return self.config["chars"]
    
    @property
    def max_seq_length(self) -> int:
        return self.config["max_seq_length"]
    
    @property
    def sos_token(self) -> int:
        return self.config["sos_token"]
    
    @property
    def eos_token(self) -> int:
        return self.config["eos_token"]
    
    @property
    def image_height(self) -> int:
        return self.config["image_height"]
    
    @property
    def image_min_width(self) -> int:
        return self.config["image_min_width"]
    
    @property
    def image_max_width(self) -> int:
        return self.config["image_max_width"]
    
    @property
    def cnn(self) -> VggCnn:
        return VggCnn(self.config["cnn"])
    
    @property
    def encoder(self) -> Encoder:
        return Encoder(self.config["encoder"])
    
    @property
    def decoder(self) -> Decoder:
        return Decoder(self.config["decoder"])
    
    @property
    def fields_domain(self) -> dict:
        return self.config["fields_domain"]
    
