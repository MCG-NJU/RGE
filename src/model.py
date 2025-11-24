import sys
sys.path.extend(['.', '..'])
from typing import Dict, Optional
import torch.nn.functional as F
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig, MllamaForConditionalGeneration, LlavaNextForConditionalGeneration 
from peft import LoraConfig, get_peft_model, PeftModel
from src.arguments import ModelArguments, TrainingArguments
from src.model_utils import QWEN2_VL, QWEN2_5_VL
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
import logging

logger = logging.getLogger(__name__)

class MMEBModel(nn.Module):
    TRANSFORMER_CLS = {
        "meta-llama/Llama-3.2-11B-Vision": MllamaForConditionalGeneration,
        "intfloat/mmE5-mllama-11b-instruct": MllamaForConditionalGeneration
    }

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 training_args: TrainingArguments = None,
                 model_args: ModelArguments = None,
                 processor = None,
                 ):
        super().__init__()
        self.config = encoder.config
        if hasattr(self.config, 'hidden_size'):
            self.hidden_size = self.config.hidden_size
        else:
            self.hidden_size = self.config.text_config.hidden_size
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        self.training_args = training_args
        self.model_args = model_args
        self.processor = processor
        print(f"DDP: {self.is_ddp}")
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            print(f"Process rank: {self.process_rank}, World size: {self.world_size}")

    def encode_input(self, input):
        logger.debug(f"Input keys: {input.keys()}")
        logger.debug(f"Input shapes: {[(k, v.shape) for k, v in input.items() if isinstance(v, torch.Tensor)]}")
        hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
        hidden_states = hidden_states.hidden_states[-1]
        pooled_output = self._pooling(hidden_states, input['attention_mask'])
        logger.debug(f"Output shape: {pooled_output.shape}")
        return pooled_output

    def _ar_pooling(self, input_ids, hidden_states):
        emb_id = self.processor.tokenizer.convert_tokens_to_ids("<emb>")
        embed_indices = torch.argmax((input_ids == emb_id).int(), dim=1)
        embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices]
        if self.normalize:
            embed_features = torch.nn.functional.normalize(embed_features, p=2, dim=-1)
        return embed_features
        
    def ar_encode(self, inputs: dict):
        if 'pixel_values' in inputs:
            if inputs['pixel_values'] is None or inputs['pixel_values'][0] is None:
                inputs.pop('pixel_values', None)
                inputs.pop('image_grid_thw', None)
                inputs.pop('pixel_values_videos', None)
                inputs.pop('video_grid_thw', None)
        inputs.pop('text', None)
        inputs.pop('image', None)
        outputs = self.encoder(**inputs, output_hidden_states=True)
        pooled_output = self._ar_pooling(inputs['input_ids'], outputs.hidden_states[-1])
        return pooled_output
    
    @torch.no_grad()
    def encode_and_generate(self, inputs: dict):
        # baseline -> caption 
        generation_output = self.encoder.generate(
            **inputs,
            return_dict_in_generate=True,
            output_hidden_states=True,
            max_new_tokens=128,
            eos_token_id=self.processor.tokenizer.convert_tokens_to_ids("<emb>"),
            use_cache=True,
        )

        prompt_len = inputs['input_ids'].shape[1]
        generated_ids = generation_output.sequences[0][prompt_len:]
        generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=False)
        print(f"generated_text: {generated_text}")

        past_seen = generation_output.past_key_values.get_seq_length()

        new_inputs = {
            'input_ids': generation_output.sequences[:, -1:],
            'past_key_values': generation_output.past_key_values,
            'cache_position': torch.arange(past_seen, past_seen + 1, device=generation_output.sequences.device),
        }

        outputs = self.encoder(**new_inputs, output_hidden_states=True, use_cache=True)

        pooled_output = outputs.hidden_states[-1][:, -1, :]
        if self.normalize:
            pooled_output = torch.nn.functional.normalize(pooled_output, p=2, dim=-1)
        return pooled_output, generated_text


    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == 'last' or self.pooling == 'eos':
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[Dict] = None):
        gradient_checkpointing_kwargs={'use_reentrant': False}
        if self.model_args.lora:
            model = self.encoder.base_model.model
        else:
            model = self.encoder
        if self.training_args.bf16:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        else:
            model.gradient_checkpointing_enable()

    @classmethod
    def build(cls, model_args: ModelArguments, training_args: TrainingArguments, **hf_kwargs):
        force_download = True
        if dist.get_world_size() == 1:
            force_download = False
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True, force_download=force_download)
        if hasattr(config, 'use_cache'):
            config.use_cache = False
        elif hasattr(config, 'text_config'):
            config.text_config.use_cache = False

        config.padding_side = "right"
        

        if model_args.model_backbone in [QWEN2_VL, QWEN2_5_VL]:
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_name,
                attn_implementation='flash_attention_2',
                torch_dtype=torch.bfloat16,
            )
        else:
            if model_args.model_name in cls.TRANSFORMER_CLS:
                model_type = cls.TRANSFORMER_CLS[model_args.model_name]
            else:
                model_type = AutoModelForCausalLM
            
            if 'Llama' in model_args.model_name or model_args.model_backbone == "mllama":
                base_model = MllamaForConditionalGeneration.from_pretrained(
                model_args.model_name, **hf_kwargs, config=config, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True)
            else:
                base_model = model_type.from_pretrained(
                    model_args.model_name, **hf_kwargs, config=config, 
                    attn_implementation="flash_attention_2", 
                    torch_dtype=torch.bfloat16, 
                    trust_remote_code=True,
                    force_download=force_download
                    )
            base_model.padding_side = "right"

        if hasattr(base_model.config, 'text_config'):
            base_model.config.hidden_size = base_model.config.text_config.hidden_size
            base_model.config.text_config.use_cache = False

        if model_args.lora:
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(','),
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False
            )
            lora_model = get_peft_model(base_model, lora_config)

            trainable_modules = ['linear']
            for name, param in lora_model.named_parameters():
                if any(module_name in name for module_name in trainable_modules):
                    param.requires_grad = True

            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                training_args=training_args,
                model_args=model_args,
                processor=hf_kwargs.get('processor', None)
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                training_args=training_args,
                model_args=model_args,
                processor=hf_kwargs.get('processor', None)
            )
        
        if training_args.gradient_checkpointing:
            base_model.enable_input_require_grads()
        
        return model

    @classmethod
    def load(cls, model_args: ModelArguments, **hf_kwargs):
        # checkpoint_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        checkpoint_path = model_args.model_name
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        logger.info(f"Model arguments: {model_args}")
        
        if model_args.model_name:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            if hasattr(config, 'use_cache'):
                config.use_cache = False
            config.padding_side = "right"
        if model_args.model_backbone == QWEN2_5_VL:
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                attn_implementation='flash_attention_2',
                torch_dtype=torch.bfloat16,
            )
        elif model_args.model_backbone == QWEN2_VL:
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                attn_implementation='flash_attention_2',
                torch_dtype=torch.bfloat16,
            )
            
        elif model_args.model_backbone == "llava_next":
            config.use_cache = False
            config.padding_side = "right"
            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            base_model.padding_side = "right"
        else: 
            if model_args.model_name in cls.TRANSFORMER_CLS:
                model_type = cls.TRANSFORMER_CLS[model_args.model_name]
            else:
                model_type = AutoModelForCausalLM
            if 'Llama' in model_args.model_name or model_args.model_backbone == "mllama":
                base_model = model_type.from_pretrained(
                checkpoint_path, **hf_kwargs, config=config, 
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True)
            else:
                base_model = model_type.from_pretrained(
                    checkpoint_path, **hf_kwargs, config=config,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True)
            base_model.padding_side = "right"

        # Building the model on top of the base
        if model_args.lora:
            checkpoint_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
            lora_config = LoraConfig.from_pretrained(checkpoint_path)
            lora_model = PeftModel.from_pretrained(base_model, checkpoint_path, config=lora_config)
            
            merged_model = lora_model.merge_and_unload()
            model = cls(
                encoder=merged_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                model_args=model_args,
                processor=hf_kwargs.get('processor', None)
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                model_args=model_args,
                processor=hf_kwargs.get('processor', None)
            )
        
        logger.info(f"Model loaded successfully")
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, neg: Dict[str, Tensor] = None):
        qry_reps = self.ar_encode(qry) if qry else None  # (bsz_per_device, dim)
        tgt_reps = self.ar_encode(tgt) if tgt else None # (bsz_per_device, dim)
        neg_reps = self.ar_encode(neg) if neg else None # (bsz_per_device * negative_ratio, dim)
        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
            all_neg_reps = self._dist_gather_tensor(neg_reps) if neg else None 
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps
            all_neg_reps = neg_reps

        pos_scores = self.compute_similarity(all_qry_reps, all_tgt_reps)
        pos_scores = pos_scores.view(all_qry_reps.size(0), -1)
        scores = pos_scores.clone()

        neg_ratio = 0
        batch_size = len(all_qry_reps)
        if neg is not None:
            neg_ratio = int(all_neg_reps.shape[0] / all_qry_reps.shape[0])
            neg_scores = torch.sum(all_qry_reps.unsqueeze(1) * all_neg_reps.view(batch_size, neg_ratio, -1), dim = -1) # B * neg_ratio
            scores = torch.cat([pos_scores, neg_scores], dim = 1)
        
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))

        loss = self.cross_entropy(scores / self.temperature, target)

        return loss

    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors
    

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
