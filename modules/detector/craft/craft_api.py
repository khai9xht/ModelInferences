from .modules import get_bboxes, adjustResultCoordinates, \
  resize_aspect_ratio, normalizeMeanVariance
import cv2, time
import numpy as np
from modules.configs import CraftRefinerConfig
from typing import Dict, List, Union
from modules.model_server import TritonClient

class CraftRefiner:
  def __init__(self, config: CraftRefinerConfig, triton_client: TritonClient) -> None:
    self.craft_config = config.craft
    self.refiner_config = config.refiner
    self.triton_client = triton_client
    self.craft_initialize()
    self.refiner_initialize()
    
  def craft_initialize(self) -> None:
    input_sizes = [
      [1, 3, self.craft_config.canvas_size, self.craft_config.canvas_size]
    ]
    self.triton_client.initialize(
      model_name=self.craft_config.model_name,
      input_names=self.craft_config.input_names,
      input_sizes=input_sizes,
      output_names=self.craft_config.output_names
    )
    print(f"[{self.__class__.__name__}][{self.craft_config.model_name}]: Using model on triton server.")

  def refiner_initialize(self) -> None:
    input_sizes = [
      [1, self.craft_config.canvas_size // 2, self.craft_config.canvas_size // 2, 2],
      [1, 32, self.craft_config.canvas_size // 2, self.craft_config.canvas_size // 2]
    ]
    self.triton_client.initialize(
      model_name=self.refiner_config.model_name,
      input_names=self.refiner_config.input_names,
      input_sizes=input_sizes,
      output_names=self.refiner_config.output_names
    )
    print(f"[{self.__class__.__name__}][{self.refiner_config.model_name}]: Using model on triton server.")


  def craft_forward(self, img: np.ndarray) -> Union[List[np.ndarray], float]:
    img_resized, target_ratio, _ = resize_aspect_ratio(
      img,
      self.craft_config.canvas_size,
      interpolation=cv2.INTER_LINEAR,
      mag_ratio=self.craft_config.mag_ratio,
    )
    ratio = 1.0 / target_ratio
    x = normalizeMeanVariance(img_resized)
    x = np.transpose(x, axes=[2, 0, 1])
    x = np.expand_dims(x, axis=0).astype(np.float32)
    results = self.triton_client.infer(model_name=self.craft_config.model_name, inputs=[x])
    output = [results.as_numpy(name) for name in self.craft_config.output_names]
    return output, ratio

  def refiner_forward(self, craft_output: List[np.ndarray]) -> np.ndarray:
    results = self.triton_client.infer(self.refiner_config.model_name, inputs=craft_output)
    output = [results.as_numpy(name) for name in self.refiner_config.output_names]
    return output[0]

  def __call__(self, img: np.ndarray) -> Dict[str, List[np.ndarray]]:
    start_time = time.time()
    craft_output, ratio = self.craft_forward(img)
    score_text = craft_output[0][0, :, :, 0]
    refiner_output = self.refiner_forward(craft_output)
    refined_link = refiner_output[0, :, :, 0]
    field_bboxes = get_bboxes(
      score_text,
      refined_link,
      self.refiner_config.text_threshold,
      self.refiner_config.link_threshold,
      self.refiner_config.text_low,
    )
    field_bboxes = adjustResultCoordinates(
      field_bboxes, ratio_h=ratio, ratio_w=ratio
    )
    
    print(f"[{self.__class__.__name__}]: Detection finished, time inference: {time.time() - start_time}")
    return field_bboxes