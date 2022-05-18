import yaml
import json
from typing import List

class Config:
    def __init__(self, config_fp: str) -> None:
        self.config = {}
        if config_fp.endswith(".yaml"):
            with open(config_fp) as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
        elif config_fp.endswith(".json"):
            with open(config_fp) as f:
                self.config = json.load(f)


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class OnnxConfig(Config):
    def __init__(self, config: dict) -> None:
        self.config = config
    
    @property
    def model_path(self) -> str:
        return self.config["model_path"]


class TensorRTConfig:
    def __init__(self, config: dict) -> None:
        self.config = config
    
    @property
    def model_path(self) -> str:
        return self.config["model_path"]
    
    @property
    def param_mode(self) -> str:
        return self.config["param_mode"]

    @property
    def input_names(self) -> List[str]:
        return self.config["input_names"]
    
    @property
    def output_names(self) -> List[str]:
        return self.config["output_names"]


class ModelRemoteConfig:
    def __init__(self, config: dict) -> None:
        self.config = config
    
    @property
    def input_names(self) -> List[str]:
        return self.config["input_names"]
    
    @property
    def output_names(self) -> List[str]:
        return self.config["output_names"]
    @property
    def param_modes(self) -> List[str]:
        return self.config["param_modes"]


class CommonConfig(Config):
    def __init__(self, config_fp: str) -> None:
        super(CommonConfig, self).__init__(config_fp)
        assert self.config["mode"] in ['onnx', 'tensorrt', 'model_server'], \
            f"[ERROR] Don't support model with platform {self.config['mode']}, Only support model in ['onnx', 'tensorrt', 'model_server']"
        if self.config["mode"] == "onnx":
            self.common = OnnxConfig(self.config)
        elif self.config["mode"] == "tensorrt":
            self.common = TensorRTConfig(self.config)
        else:
            self.common = ModelRemoteConfig(self.config)
