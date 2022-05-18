import numpy as np
from modules.configs import Yolov5sConfig
from typing import List
from .processing import non_max_suppression, BoundingBox
import cv2, time
from modules.model_server import TritonClient

class Yolov5s: 
    def __init__(self, config: Yolov5sConfig, triton_client: TritonClient) -> None:
        self.config = config
        self.input_size = tuple([config.input_width, config.input_height])
        input_sizes = [tuple([1, 3, config.input_width, config.input_height])]

        self.triton_client = triton_client
        self.triton_client.initialize(
            model_name=config.model_name,
            input_names=config.input_names,
            input_sizes=input_sizes,
            output_names=config.output_names
        )
        print(f"[{self.__class__.__name__}][{self.config.model_name}]: Using model on triton server.")
       
    def preprocess(self, raw_rgb_image: np.ndarray) -> np.ndarray:
        """
        description: Preprocess an image before TRT YOLO inferencing.
                    Convert BGR image to RGB,
                    resize and pad it to target size, normalize to [0,1],
                    transform to NCHW format.          
        param:
            raw_rgb_image: int8 numpy array of shape (img_h, img_w, 3)
        return:
            image:  the processed image float32 numpy array of shape (3, H, W)
        """
        input_w, input_h = self.input_size
        image_raw = raw_rgb_image
        h, w, c = image_raw.shape
        image = image_raw.copy()
        # image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = input_w / w
        r_h = input_h / h
        if r_h > r_w:
            tw = input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((input_h - th) / 2)
            ty2 = input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = input_h
            tx1 = int((input_w - tw) / 2)
            tx2 = input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        return image

    def postprocess(self, output: List[np.ndarray], origin_w: int, origin_h: int) -> List[BoundingBox]:
        """Postprocess outputs.
        # Args
            output: list of detections with schema 
            [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
        # Returns
            list of bounding boxes with all detections above threshold and after nms, see class BoundingBox
        """
        
        # Get the num of boxes detected
        # Here we use the first row of output in that batch_size = 1
        pred = output[0][0].copy()
        # print(output[0].shape)
        # num = int(output[0])
        # # Reshape to a two dimentional ndarray
        # pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # Do nms
        pred[:, 5] = np.argmax(pred[:, 5:], axis=-1)
        boxes = non_max_suppression(
            pred[:, :6], origin_h, 
            origin_w, self.input_size[0], self.input_size[1], 
            conf_thres=self.config.conf_th, nms_thres=self.config.nms_threshold
        )
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, -1] if len(boxes) else np.array([])
            
        detected_objects = []
        expand_bbox = [
            self.config.expands_top, self.config.expands_left, \
            self.config.expands_bottom, self.config.expands_right
        ]
        for box, score, label in zip(result_boxes, result_scores, result_classid):
            x1, y1, x2, y2 = self.expand_bboxes(box, expand_bbox)
            x1 = x1 if x1 > 0 else 0
            y1 = y1 if y1 > 0 else 0
            x2 = x2 if x2 < origin_w else origin_w - 1
            y2 = y2 if y2 < origin_h else origin_h - 1
            detected_objects.append(BoundingBox(
                int(label), score, \
                x1, x2, y1, y2,  \
                origin_w, origin_h
            ))
        return detected_objects

    def forward(self, input: np.ndarray) -> List[np.ndarray]:
        results = self.triton_client.infer(model_name= self.config.model_name, inputs=[input])
        net_outs = [results.as_numpy(name) for name in self.config.output_names]
        net_outs = [results.as_numpy(self.config.output_names[0])]
        return net_outs

    def expand_bboxes(self, box: np.ndarray, expand_ratio: list) -> list:
        x1, y1, x2, y2 = box
        rt, rl, rb, rr = expand_ratio
        x1 = x1 - (x2 - x1) * rl
        y1 = y1 - (y2 - y1) * rt
        x2 = x2 + (x2 - x1) * rr
        y2 = y2 + (y2 - y1) * rb
        return x1, y1, x2, y2


    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        start_time = time.time()
        h, w, _ = image.shape
        img_pre = self.preprocess(image)
        inputs = np.expand_dims(img_pre, axis=0)
        net_outs = self.forward(inputs)
        print(np.max(net_outs[0][0][:, 4]))
        net_outs = self.postprocess(net_outs, origin_h=h, origin_w=w)
        score = [net.classID for net in net_outs]
        print(f"score: {score}")
        print(f"[{self.__class__.__name__}][{self.config.model_name}]: Detection finished, inference time: {time.time()- start_time}")
        return net_outs


    