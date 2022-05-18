from typing import List
import numpy as np
from .vocab import Vocab
import math
from modules.model_server import TritonClient
from modules.configs import Seq2SeqConfig
from PIL import Image
import cv2

class Seq2Seq:
    def __init__(self, config: Seq2SeqConfig, triton_client: TritonClient) -> None:
        self.config = config
        self.triton_client = triton_client
        self.cnn_config = config.cnn
        self.decoder_config = config.decoder
        self.encoder_config = config.encoder

        self.vocab = Vocab(config.chars)
    
    def resize(self, w: int, h: int) -> int:
        new_w = int(self.config.image_height * float(w) / float(h))
        round_to = 10
        new_w = math.ceil(new_w/round_to)*round_to
        new_w = max(new_w, self.config.image_min_width)
        new_w = min(new_w, self.config.image_max_width)

        return new_w

    def process_image(self, image: Image) -> np.ndarray:
        img = image.convert('RGB')

        w, h = img.size
        new_w = self.resize(w, h)

        img = img.resize((new_w, self.config.image_height), Image.ANTIALIAS)

        img = np.asarray(img).transpose(2,0, 1)
        img = img/255
        return img


    def process_input(self, image: Image) -> np.ndarray:
        img = self.process_image(image)
        img = img[np.newaxis, ...]
        return np.array(img, dtype=np.float32)

    def translate_onnx(self, img: np.ndarray):
        """data: BxCxHxW"""        
        # create cnn input
        input_sizes = [img.shape]
        self.triton_client.initialize(
            model_name=self.cnn_config.model_name,
            input_names=self.cnn_config.input_names,
            input_sizes=input_sizes,
            output_names=self.cnn_config.output_names
        )
        cnn_output = self.triton_client.infer(
            model_name=self.cnn_config.model_name,
            inputs=[img]
        )
        src = [cnn_output.as_numpy(name) for name in self.cnn_config.output_names]

        
        # create encoder input
        self.triton_client.initialize(
            model_name=self.encoder_config.model_name,
            input_names=self.encoder_config.input_names,
            input_sizes=[src[0].shape],
            output_names=self.encoder_config.output_names,
        )
        encoder_output = self.triton_client.infer(
            model_name=self.encoder_config.model_name,
            inputs=src
        )
        encoder_outputs, hidden = [
            encoder_output.as_numpy(name) for name in \
                self.encoder_config.output_names
        ]

        self.triton_client.initialize(
            model_name=self.decoder_config.model_name,
            input_names=self.decoder_config.input_names,
            input_sizes=[[1], hidden.shape, encoder_outputs.shape],
            output_names=self.decoder_config.output_names,
            float_modes=["INT64", "FP32", "FP32"]
        )

        translated_sentence = [[self.config.sos_token] * len(img)]
        max_length = 0

        while max_length <= self.config.max_seq_length and not all(
            np.any(np.asarray(translated_sentence).T == self.config.eos_token, axis=1)
        ):
            tgt_inp = translated_sentence

            decoder_output = self.triton_client.infer(
                model_name=self.decoder_config.model_name,
                inputs=[np.array(tgt_inp[-1], dtype=np.int64), hidden, encoder_outputs]
            )
            output, hidden, _ = [
                decoder_output.as_numpy(name) for name in \
                self.decoder_config.output_names
            ]
            
            indices = np.argmax(output, axis=1)
            translated_sentence.append(indices)
            max_length += 1
            del output
        translated_sentence = np.asarray(translated_sentence).T


        return translated_sentence

    def recognize(self, image: Image) -> str:
        image = self.process_input(image=image)
        s = self.translate_onnx(np.array(image))[0].tolist()
        s = self.vocab.decode(s)
        return s
    
    def recognize_batch(self, input_list: List[np.ndarray]) -> List[str]:
        results = []
        for np_img in input_list:
            img = Image.fromarray(np_img)
            s = self.recognize(img)
            results.append(s)
        return results
    
    def recog_batch(self, input_list: List[np.ndarray]) -> List[List[str]]:
        img_num = len(input_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in input_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        soft_indices = np.argsort(np.array(width_list))

        # rec_res = []
        rec_res = [['', 0.0]] * img_num
        # batch_num = self.rec_batch_num
        batch_num = 3
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                # h, w = input_list[ino].shape[0:2]
                h, w = input_list[soft_indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(input_list[soft_indices[ino]],
                                                max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
    
            norm_img_batch = np.concatenate(norm_img_batch)
            # norm_img_batch = norm_img_batch.copy()
            # start inference here
            input_sizes = [norm_img_batch.shape]
            self.triton_client.initialize(
                model_name=self.cnn_config.model_name,
                input_names=self.cnn_config.input_names,
                input_sizes=input_sizes,
                output_names=self.cnn_config.output_names
            )
            cnn_output = self.triton_client.infer(
                model_name=self.cnn_config.model_name,
                inputs=[norm_img_batch]
            )
            src = [cnn_output.as_numpy(name) for name in self.cnn_config.output_names]
            self.triton_client.initialize(
                model_name=self.encoder_config.model_name,
                input_names=self.encoder_config.input_names,
                input_sizes=[src[0].shape],
                output_names=self.encoder_config.output_names,
            )
            encoder_output = self.triton_client.infer(
                model_name=self.encoder_config.model_name,
                inputs=src
            )
            encoder_outputs, hidden = [
                encoder_output.as_numpy(name) for name in \
                    self.encoder_config.output_names
            ]
            
            translated_sentence = [[self.config.sos_token] * len(norm_img_batch)]
            max_length = 0
            while max_length <= self.config.max_seq_length and not all(
                np.any(np.asarray(translated_sentence).T == self.config.eos_token, axis=1)
            ):
                tgt_inp = translated_sentence
                self.triton_client.initialize(
                    model_name=self.decoder_config.model_name,
                    input_names=self.decoder_config.input_names,
                    input_sizes=[[norm_img_batch.shape[0]], hidden.shape, encoder_outputs.shape],
                    output_names=self.decoder_config.output_names,
                    float_modes=["INT64", "FP32", "FP32"]
                )
                decoder_output = self.triton_client.infer(
                    model_name=self.decoder_config.model_name,
                    inputs=[np.array(tgt_inp[-1], dtype=np.int64), hidden, encoder_outputs]
                )
                output, hidden, _ = [
                    decoder_output.as_numpy(name) for name in \
                    self.decoder_config.output_names
                ]
                
                indices = np.argmax(output, axis=1)
                translated_sentence.append(indices)
                max_length += 1
                del output
            translated_sentence = np.asarray(translated_sentence).T
            # end inference
            for rno in range(len(translated_sentence)):
                rec_res[soft_indices[beg_img_no + rno]] = translated_sentence[rno]
        rec_res = [self.vocab.decode(rec_r.tolist()) for rec_r in rec_res]
        return rec_res

    def resize_norm_img(self, img, max_wh_ratio: float, round_to: int = 10):
        imgC, imgH, imgW = 3, self.config.image_height, self.config.image_min_width
        assert imgC == img.shape[2]
        imgW = int((self.config.image_height * max_wh_ratio))
        imgW = math.ceil(imgW / round_to) * round_to
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_w = math.ceil(resized_w / round_to) * round_to
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im
    
    def recog_batch_2(self, input_list: List[np.ndarray]) -> List[List[str]]:
        img_num = len(input_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in input_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        soft_indices = np.argsort(np.array(width_list))

        # rec_res = []
        rec_res = [['', 0.0]] * img_num
        # batch_num = self.rec_batch_num
        batch_num = 9
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                # h, w = input_list[ino].shape[0:2]
                h, w = input_list[soft_indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(input_list[soft_indices[ino]],
                                                max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
    
            norm_img_batch = np.concatenate(norm_img_batch)
            # norm_img_batch = norm_img_batch.copy()
            # start inference here
            input_sizes = [norm_img_batch.shape]
            self.triton_client.initialize(
                model_name=self.cnn_config.model_name,
                input_names=self.cnn_config.input_names,
                input_sizes=input_sizes,
                output_names=self.cnn_config.output_names
            )
            cnn_output = self.triton_client.infer(
                model_name=self.cnn_config.model_name,
                inputs=[norm_img_batch]
            )
            src = [cnn_output.as_numpy(name) for name in self.cnn_config.output_names]
            self.triton_client.initialize(
                model_name=self.encoder_config.model_name,
                input_names=self.encoder_config.input_names,
                input_sizes=[src[0].shape],
                output_names=self.encoder_config.output_names,
            )
            encoder_output = self.triton_client.infer(
                model_name=self.encoder_config.model_name,
                inputs=src
            )
            encoder_outputs, hidden = [
                encoder_output.as_numpy(name) for name in \
                    self.encoder_config.output_names
            ]
            translated_sentences = []
            self.triton_client.initialize(
                model_name=self.decoder_config.model_name,
                input_names=self.decoder_config.input_names,
                input_sizes=[
                    [1], 
                    [1, hidden.shape[1]], 
                    [encoder_outputs.shape[0], 1, encoder_outputs.shape[2]]
                ],
                output_names=self.decoder_config.output_names,
                float_modes=["INT64", "FP32", "FP32"]
            )
            for i in range(hidden.shape[0]):
                translated_sentence = [[self.config.sos_token]]
                max_length = 0
                hid = hidden[i, :][np.newaxis, ...]
                enc_out = np.expand_dims(encoder_outputs[:, i, :], axis=1)
                while max_length <= self.config.max_seq_length and not all(
                    np.any(np.asarray(translated_sentence).T == self.config.eos_token, axis=1)
                ):
                    tgt_inp = translated_sentence
                    
                    decoder_output = self.triton_client.infer(
                        model_name=self.decoder_config.model_name,
                        inputs=[np.array(tgt_inp[-1], dtype=np.int64), hid, enc_out]
                    )
                    output, hid, _ = [
                        decoder_output.as_numpy(name) for name in \
                        self.decoder_config.output_names
                    ]
                    
                    indices = np.argmax(output, axis=1)
                    translated_sentence.append(indices)
                    max_length += 1
                    del output
                translated_sentence = np.asarray(translated_sentence).T
                translated_sentences.append(translated_sentence[0])
                # end inference
            for rno in range(len(translated_sentences)):
                rec_res[soft_indices[beg_img_no + rno]] = translated_sentences[rno]
        rec_res = [self.vocab.decode(rec_r.tolist()) for rec_r in rec_res]
        return rec_res

    
            