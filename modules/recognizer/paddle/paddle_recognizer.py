from .ppocr.predict_rec import TextRecognizer
from .rec_object import RecObject
import typing
import time
from modules.model_server import TritonClient
from modules.configs import PaddleRecognizerConfig 

class PaddleRecognizer(object):
    def __init__(self, config: PaddleRecognizerConfig, triton_client: TritonClient):
        self.config = config
        self.text_recognizer = TextRecognizer(self.config, triton_client)

    def recognize(self, img_list: list) -> typing.List[RecObject]:
        start_time = time.time()
        if not isinstance(img_list, list):
            img_list = [img_list]

        ret = self.text_recognizer(img_list)
        rec_objs = [RecObject(text, score) for text, score in ret]
        print(f"[{self.__class__.__name__}]: Recognize all lines finished, inference time: {time.time()- start_time}")
        return rec_objs
