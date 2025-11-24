import logging
from dataclasses import dataclass
from transformers import ProcessorMixin
from src.arguments import DataArguments, ModelArguments
import torch
from PIL import ImageFile
from src.utils import Truncation
from src.processing_message import process_to_train_qwen2_5_vl_with_default_chat_template
from copy import deepcopy
from transformers.feature_extraction_utils import BatchFeature
from torch.nn.utils.rnn import pad_sequence
from transformers.image_utils import ChannelDimension
import PIL

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
PADDING_TO_MAX_LENGTH = False

def pad_position_ids(sequences, batch_first=False, padding_value=0, **kwargs):
    # [3, seq_len] --> [seq_len, 3]
    transposed = [seq.permute(1, 0) for seq in sequences]  # [seq_len, 3]
    padded = pad_sequence(transposed, batch_first=False, padding_value=padding_value, **kwargs)  # [seq_len, bs, 3]
    if batch_first:
        padded = padded.permute(1, 2, 0)  # [seq_len, bs, 3] --> [bs, 3, seq_len]
    else:
        padded = padded.permute(2, 1, 0)  # [seq_len, bs, 3] --> [3, bs, seq_len]
    return padded


@dataclass
class QWEN25TrainCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        qry_inputs = self._get_batch_inputs(examples, 0, 1)
        pos_inputs = self._get_batch_inputs(examples, 2, 3)
        neg_inputs = self._get_batch_inputs(examples, 4, 5)

        return qry_inputs, pos_inputs, neg_inputs

    def _get_batch_inputs(self, examples, text_idx, image_idx):
        if text_idx == 4:
            texts = [t for exp in examples for t in exp[text_idx] if t is not None]
            images = [i for exp in examples for i in exp[image_idx] if i is not None]
        else:
            texts = [exp[text_idx] for exp in examples if exp[text_idx] is not None]
            images = [exp[image_idx] for exp in examples if exp[image_idx] is not None]
        print(f"texts: {len(texts)}, images: {len(images)}")
        print(f"texts: {texts}, images: {images}")
        if len(texts) == 0:
            return None

        if len(images) == 0:
            inputs = self.processor(text=texts, images=None, padding=True, return_tensors="pt", pad_to_multiple_of=32)
        else:
            inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt", pad_to_multiple_of=32)
        
        trucation = Truncation(train=True)
        inputs = trucation.truncate(inputs, self.data_args.max_len)
        
        return inputs

@dataclass
class EvalCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        logger.debug(f"Collating batch of size: {len(examples)}")
        inputs = self._get_batch_inputs(examples)
        logger.debug(f"Collated input keys: {inputs.keys()}")
        logger.debug(f"Collated input shapes: {[(k, v.shape) for k, v in inputs.items() if isinstance(v, torch.Tensor)]}")
        return inputs

    def _get_batch_inputs(self, examples):
        texts = [exp[0] for exp in examples if exp[0] is not None]
        images = [exp[1] for exp in examples if exp[1] is not None]
        
        
        if len(images) == 0:
            inputs = self.processor(text=texts, images=None, padding=True, return_tensors="pt", pad_to_multiple_of=32)
        else:
            inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt", pad_to_multiple_of=32)
        trucation = Truncation(train=False)
        inputs = trucation.truncate(inputs, self.data_args.max_len)

        return inputs

class ARCollator(QWEN25TrainCollator):
    def __init__(self, *args, **kwargs):
        self.model_for_position_ids = kwargs.pop("model_for_position_ids", None)
        super().__init__(*args, **kwargs)
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        self.max_length = None # no limit
        self.process_exceed = 'truncate'

    def construct_messages_rationale(self, text=None, image=None, rationale=None):
        if type(image) == list:
            assert len(image) == 1, "only support one image"
            image = image[0]

        if image is None:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{text.replace('<|image_pad|>', '')}"}  # for buggy datasets like WebQA
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"{rationale}"}
                    ]
                },
            ]
        elif text is None:
            assert 0, "should not have text None"
        else:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text.replace("<|image_pad|>", "")},
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"{rationale}"}
                    ]
                },
            ]
        return message

    def construct_messages_wo_rationale(self, text=None, image=None, rationale=None):
        if type(image) == list:
            assert len(image) == 1, "only support one image"
            image = image[0]

        if image is None:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{text.replace('<|image_pad|>', '')}"}  # for buggy training datasets like WebQA
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f""}
                    ]
                },
            ]
        elif text is None:
            assert 0, "should not have text None"
        else:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text.replace("<|image_pad|>", "")},
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f""}
                    ]
                },
            ]
        return message


    def __call__(self, examples):
        if len(examples[0]) == 8 and examples[0][6] is not None:
            rationales = [e[6] for e in examples if e[6] is not None]
        else:
            rationales = None
        
        qry_inputs = self._get_batch_inputs(examples, 0, 1, rationale=rationales, emb_type="question")
        pos_inputs = self._get_batch_inputs(examples, 2, 3, rationale=None, emb_type="emb-only")

        return qry_inputs, pos_inputs


    def get_messages(self, texts, images, rationales=None, emb_token_type="question"):
        # turn the batch inputs to messages
        messages = [] # for debug
        examples = []
        if rationales is not None:
            for text, image, rationale in zip(texts, images, rationales):
                message = self.construct_messages_rationale(text, image, rationale)
                messages.append(message) # for debug
                data = process_to_train_qwen2_5_vl_with_default_chat_template(
                    self.processor, message,
                    min_pixels=getattr(self.data_args, 'min_pixels', self.processor.image_processor.min_pixels), 
                    max_pixels=getattr(self.data_args, 'max_pixels', self.processor.image_processor.max_pixels),
                    video_min_pixels=getattr(self.data_args, 'video_min_pixels', self.processor.image_processor.min_pixels),
                    video_max_pixels=getattr(self.data_args, 'video_max_pixels', self.processor.image_processor.max_pixels),
                    model_for_position_ids=self.model_for_position_ids,
                    emb_token_type=emb_token_type,
                )
                data['messages'] = deepcopy(message)
                data['messages'].pop(1)
                examples.append(BatchFeature(data=data))
        else:
            for text, image in zip(texts, images):
                message = self.construct_messages_wo_rationale(text, image)
                messages.append(message) # for debug
                data = process_to_train_qwen2_5_vl_with_default_chat_template(
                    self.processor, message,
                    min_pixels=getattr(self.data_args, 'min_pixels', self.processor.image_processor.min_pixels), 
                    max_pixels=getattr(self.data_args, 'max_pixels', self.processor.image_processor.max_pixels),
                    video_min_pixels=getattr(self.data_args, 'video_min_pixels', self.processor.image_processor.min_pixels),
                    video_max_pixels=getattr(self.data_args, 'video_max_pixels', self.processor.image_processor.max_pixels),
                    model_for_position_ids=self.model_for_position_ids,
                    emb_token_type="emb-only"
                )
                examples.append(BatchFeature(data=data))
        return examples
    
    def _get_batch_inputs(self, examples, text_idx, image_idx, rationale=None, emb_type="question"):
        texts = [exp[text_idx] for exp in examples]
        images = [exp[image_idx] for exp in examples]
        
        if len(texts) == 0:
            return None

        # BEGIN: AR-Emebeds
        messages = self.get_messages(texts, images, rationale, emb_type)
        # END: AR-Emebeds
        processed_inputs = self._construct_batch_samples(messages)
        return processed_inputs

    
    def _construct_batch_samples(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_values_videos = []
        batch_image_thw = []
        batch_video_thw = []
        batch_second_per_grid_ts = []
        batch_position_ids = []
        batch_attention_mask = []
        batch_messages = []

        for example in examples:
            keys = example.keys()
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            if "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            if "messages" in keys:
                batch_messages.append(example["messages"])
            if "pixel_values_videos" in keys:
                batch_pixel_values_videos.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])
            if "position_ids" in keys:
                batch_position_ids.append(example["position_ids"])
            if "attention_mask" in keys:
                batch_attention_mask.append(example["attention_mask"])
        
        input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_side='right', padding_value=self.pad_token_id)
        labels = pad_sequence(batch_label_ids, batch_first=True, padding_side='right', padding_value=IGNORE_INDEX)
        if len(batch_attention_mask) > 0:
            attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_side='right', padding_value=False)  # TODO: maybe check padding_value=False
        else:
            attention_mask = input_ids != self.pad_token_id

        if self.max_length is not None:
            input_ids, labels, attention_mask, non_exceed_idx, idx_to_replace, is_exceed, seq_len = self._process_exceed(input_ids, labels, attention_mask)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
        
        if len(batch_pixel_values) > 0:
            data_dict["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
            data_dict["image_grid_thw"] = torch.cat(batch_image_thw, dim=0)

        if len(batch_pixel_values_videos) > 0:
            data_dict["pixel_values_videos"] = torch.cat(batch_pixel_values_videos, dim=0)
            data_dict["video_grid_thw"] = torch.cat(batch_video_thw, dim=0)

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        if len(batch_position_ids) > 0:
            position_ids = pad_position_ids(batch_position_ids, batch_first=False, padding_side='right', padding_value=1)

            if self.max_length is not None:
                position_ids = self._process_position_ids(position_ids, non_exceed_idx, idx_to_replace, is_exceed, seq_len)
            data_dict["position_ids"] = position_ids

        if len(batch_messages) > 0:
            data_dict["messages"] = batch_messages
        return data_dict


class AREvalCollator(EvalCollator):
    def __init__(self, *args, **kwargs):
        self.encode_side = kwargs.pop("encode_side", None)
        super().__init__(*args, **kwargs)
    
    def construct_prompt(self, text):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text}
                ]
            }
        ]

    def _get_batch_inputs(self, examples):
        images = [exp[1] for exp in examples]
        if images[0] is None:
            images = []


        if self.encode_side == "qry":
            texts = [self.processor.apply_chat_template(self.construct_prompt(exp[0]), tokenize=False, add_generation_prompt=True) for exp in examples]
            # texts = [self.processor.apply_chat_template(self.construct_prompt(exp[0]), tokenize=False, add_generation_prompt=True) + "<emb>" for exp in examples]
        elif self.encode_side == "tgt":
            texts = [self.processor.apply_chat_template(self.construct_prompt(exp[0]), tokenize=False, add_generation_prompt=True) + "<emb>" for exp in examples]
            # texts = [self.processor.apply_chat_template(self.construct_prompt(exp[0]), tokenize=False, add_generation_prompt=True) for exp in examples]
        else:
            raise ValueError(f"Invalid encode_side: {self.encode_side}")        

        text_copy = deepcopy(texts)
        image_copy = deepcopy(images) if len(images) > 0 else [None] * len(texts)

        if len(images) == 0:
            inputs = self.processor(text=texts, images=None, padding=True, return_tensors="pt", pad_to_multiple_of=32)
        else:
            inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt", pad_to_multiple_of=32)
        
        trucation = Truncation(train=False)
        processed_inputs = trucation.truncate(inputs)
        processed_inputs['text'] = text_copy
        processed_inputs['image'] = image_copy

        return processed_inputs
