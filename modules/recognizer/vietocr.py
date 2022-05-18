import vietocr
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from .paddle.rec_object import RecObject

class VietOCR:
    def __init__(self) -> None:
        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
        config['cnn']['pretrained']=True
        config['device'] = 'cuda:0'
        config['predictor']['beamsearch']=False
        self.detector = Predictor(config)
    def recognize(self, img_list: list):
        results = []
        for img in img_list:
            image = Image.fromarray(img)
            result = self.detector.predict(image)
            result = RecObject(result, 1.0)
            results.append(result)
        return results
