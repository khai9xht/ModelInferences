from typing import Dict, List
from .base_config import Config


class Yolov5sConfig(Config):
    def __init__(self, config_path) -> None:
        super(Yolov5sConfig, self).__init__(config_path)
    
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
    def nms_threshold(self) -> float:
        return self.config["nms_threshold"]
    
    @property
    def conf_th(self) -> float:
        return self.config["conf_th"]
    
    @property
    def label_map(self) -> Dict[int, str]:
        return self.config["label_map"]
    
    @property
    def expands_top(self) -> float:
        return self.config["expand_bbox"]["top"]

    @property
    def expands_left(self) -> float:
        return self.config["expand_bbox"]["left"]

    @property
    def expands_bottom(self) -> float:
        return self.config["expand_bbox"]["bottom"]

    @property
    def expands_right(self) -> float:
        return self.config["expand_bbox"]["right"]
    

 