from .base_config import Config, Singleton
from .model_server_config import ModelServerConfig
from .craft_refiner_config import CraftRefinerConfig, CraftConfig, RefinerConfig
from .face_anti_spoofing_config import FASNetConfig
from .face_detection_config import SCRFDConfig
from .face_encode_config import ArcFaceConfig
from .id_card_direction_config import IDCardDirectionConfig
from .paddle_recognizer_config import PaddleRecognizerConfig
from .yolov5_config import Yolov5sConfig
from .transformer_config import Seq2SeqConfig, VggCnn, Encoder, Decoder