from transformers import LLaMAForCausalLM, LLaMAModel
from typing import List, Optional, Tuple, Union
from peft import PeftModelForCausalLM, PromptTuningConfig, PromptLearningConfig, PeftType
import torch 
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
import torch.nn as nn
from dataclasses import dataclass, field
from torch.nn import CrossEntropyLoss
from transformers.utils import logging
logger = logging.get_logger(__name__)

def prepare_prompt_learning_config(peft_config, model):
    model_config = model.config.to_dict() if hasattr(model.config, "to_dict") else model.config
    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
    
    if peft_config.num_layers is None:
        if "num_hidden_layers" in model_config:
            num_layers = model_config["num_hidden_layers"]
        elif "num_layers" in model_config:
            num_layers = model_config["num_layers"]
        elif "n_layer" in model_config:
            num_layers = model_config["n_layer"]
        else:
            raise ValueError("Please specify `num_layers` in `peft_config`")
        peft_config.num_layers = num_layers

    if peft_config.token_dim is None:
        if "hidden_size" in model_config:
            token_dim = model_config["hidden_size"]
        elif "n_embd" in model_config:
            token_dim = model_config["n_embd"]
        elif "d_model" in model_config:
            token_dim = model_config["d_model"]
        else:
            raise ValueError("Please specify `token_dim` in `peft_config`")
        peft_config.token_dim = token_dim

    if peft_config.num_attention_heads is None:
        if "num_attention_heads" in model_config:
            num_attention_heads = model_config["num_attention_heads"]
        elif "n_head" in model_config:
            num_attention_heads = model_config["n_head"]
        elif "num_heads" in model_config:
            num_attention_heads = model_config["num_heads"]
        elif "encoder_attention_heads" in model_config:
            num_attention_heads = model_config["encoder_attention_heads"]
        else:
            raise ValueError("Please specify `num_attention_heads` in `peft_config`")
        peft_config.num_attention_heads = num_attention_heads

    if getattr(peft_config, "encoder_hidden_size", None) is None:
        setattr(peft_config, "encoder_hidden_size", token_dim)

    return peft_config


@dataclass
class MyPromptTuningConfig(PromptTuningConfig):
    use_special_attention_mechanism: Optional[bool] = field(
        default=False,
        metadata={
            "help": "prompt tokens only can be attened by other tokens"
        },
    )

class MyLLaMAModel(LLaMAModel):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_virtual_tokens:Optional[int] = None, #! for prompt tuning
        use_special_attention_mechanism:Optional[bool] = None, #! for prompt tuning
        instruction_lens:Optional[torch.Tensor] = None, #! for prompt tuning
        padding_lengths:Optional[torch.Tensor] = None, #! for prompt tuning 
        
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        #! for prompt tuning attention mask
        if use_special_attention_mechanism:
            # print(instruction_lens, padding_lengths)
            MASK_WEIGHT = torch.finfo(inputs_embeds.dtype).min
            
            
            for idx, (instruction_len, pad_len) in enumerate(zip(instruction_lens, padding_lengths)):
                #! virtual_tokens, padding token, instruction, response
                instruction_start_idx = num_virtual_tokens + pad_len
                instruction_end_idx = num_virtual_tokens + pad_len + instruction_len
                # print(num_virtual_tokens)
                # print(instruction_start_idx, instruction_end_idx)
                
                
                #! each prompt token can attend themselves
                attention_mask[idx,:,:num_virtual_tokens,:num_virtual_tokens] = 0
                #! each prompt token can attend instruction
                attention_mask[idx,:,:num_virtual_tokens,instruction_start_idx:instruction_end_idx] = 0
                #! instruction tokens can attend themselves and prompt tokens
                attention_mask[idx,:,instruction_start_idx:instruction_end_idx,instruction_start_idx:instruction_end_idx] = 0
                attention_mask[idx,:,instruction_start_idx:instruction_end_idx,:num_virtual_tokens] = 0
                
                
                #! each prompt cannot attend response token
                #! 这个其实默认就是 attend 不到
                attention_mask[idx,:,:num_virtual_tokens,instruction_end_idx:] = MASK_WEIGHT

                #! response token can only attend prompt tokens and its previous token
                #! 这个也是默认的
                attention_mask[idx,:,instruction_end_idx:,:num_virtual_tokens] = 0
            
                #! it cannot see instruction tokens
                attention_mask[idx,:,instruction_end_idx:,instruction_start_idx:instruction_end_idx] = MASK_WEIGHT
            #! test
            # attention_mask[:,:,:,:] = MASK_WEIGHT
            # print(attention_mask.shape)
            # print(attention_mask)
            # print('#'*100)
            logger.warning_once(
                    "\nmodify the original attention mask for personal prompt tuning\n"
            )
            # print('modify attention')
            
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    


class MyLLaMAForCausalLM(LLaMAForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = MyLLaMAModel(config)
     

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_virtual_tokens: Optional[int] = None, #! for prompt tuning
        use_special_attention_mechanism: Optional[bool] = None, #! for prompt tuning 
        instruction_lens:Optional[torch.Tensor] = None, #! for prompt tuning
        padding_lengths:Optional[torch.Tensor] = None, #! for prompt tuning 
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            num_virtual_tokens=num_virtual_tokens, #! for prompt tuning 
            use_special_attention_mechanism=use_special_attention_mechanism, #! for prompt tuning 
            padding_lengths=padding_lengths, #! for prompt tuning 
            instruction_lens=instruction_lens, #! for prompt tuning 
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MyPeftModelForCausalLM(PeftModelForCausalLM):
    
    def __init__(self, model, peft_config, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        
    #! model.generate 的时候不会用到这个函数
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        instruction_lens:Optional[torch.Tensor] = None, #! for prompt tuning
        padding_lengths:Optional[torch.Tensor] = None, #! for prompt tuning 
        **kwargs,
    ):
        #! 这边 kwargs 会传过来 instruction length 的
        peft_config = self.active_peft_config
        if not isinstance(peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)


        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        
        #! pass num_virtual_tokens for prompt tuning
        #! use_special_attention_mechanism
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "num_virtual_tokens": peft_config.num_virtual_tokens,
                "use_special_attention_mechanism":peft_config.use_special_attention_mechanism,
                "instruction_lens":instruction_lens,
                "padding_lengths":padding_lengths,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        else:
            
            if inputs_embeds is None:
                #! 先得到 inputs_embeds
                inputs_embeds = self.word_embeddings(input_ids)

            # concat prompt labels
            if labels is not None:
                #! label 也要 concat
                prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            
          
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
          
            
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
    
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        peft_config = self.active_peft_config
        

        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        #! test
        #! 
        # kwargs['instruction_lens'] = torch.tensor([1,1,2,1]).to(model_kwargs["input_ids"].device)
        # kwargs['padding_lengths'] = torch.tensor([1,1,2,1]).to(model_kwargs["input_ids"].device)

        if isinstance(peft_config, PromptLearningConfig):
            if peft_config.peft_type == PeftType.PREFIX_TUNING:
                prefix_attention_mask = torch.ones(
                    model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
                ).to(model_kwargs["input_ids"].device)
                model_kwargs["attention_mask"] = torch.cat(
                    (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
                )

            if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
                past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])

                if self.base_model_torch_dtype is not None:
                    # handle the case for Bloom where it outputs tuple of tuples
                    if isinstance(past_key_values[0], tuple):
                        past_key_values = tuple(
                            tuple(
                                past_key_value.to(self.base_model_torch_dtype)
                                for past_key_value in past_key_value_tuple
                            )
                            for past_key_value_tuple in past_key_values
                        )
                    else:
                        past_key_values = tuple(
                            past_key_value.to(self.base_model_torch_dtype) for past_key_value in past_key_values
                        )

                model_kwargs["past_key_values"] = past_key_values
            else:
                if model_kwargs["past_key_values"] is None:
                    inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                    prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                    prompts = prompts.to(inputs_embeds.dtype)
                    model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                    model_kwargs["input_ids"] = None
                    #! add for special attention mechanism
                    #! for model.forward function param
                    model_kwargs["num_virtual_tokens"] = peft_config.num_virtual_tokens
                    model_kwargs["use_special_attention_mechanism"] = peft_config.use_special_attention_mechanism
                    model_kwargs["instruction_lens"] = kwargs['instruction_lens']
                    model_kwargs["padding_lengths"] = kwargs['padding_lengths']
                    #! False 的话就是严格每次生成一个 token 都重新构建 attention_mask, 和 training 保持一致
                    #! True 的话相当于就用一开始的 attention_mask 得到的 key values, 然后后面的 response 都直接用好像
                    #! 可以对比看看, 看拿个结果更好
                    #! 先用 use_cache = True 吧
                    # model_kwargs["use_cache"] = False

        return model_kwargs
    
    @classmethod
    def from_pretrained(cls, model, model_id, adapter_name="default", is_trainable=False, **kwargs):
        r"""
        Instantiate a [`LoraModel`] from a pretrained Lora configuration and weights.

        Args:
            model ([`~transformers.PreTrainedModel`]):
                The model to be adapted. The model should be initialized with the
                [`~transformers.PreTrainedModel.from_pretrained`] method from the 🤗 Transformers library.
            model_id (`str` or `os.PathLike`):
                The name of the Lora configuration to use. Can be either:
                    - A string, the `model id` of a Lora configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a Lora configuration file saved using the `save_pretrained`
                      method (`./my_lora_config_directory/`).
        """
        # MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING
        # from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING

        # load the config
        #! 自己的 PromptTuningConfig
        config = MyPromptTuningConfig.from_pretrained(model_id, subfolder=kwargs.get("subfolder", None))

        if (getattr(model, "hf_device_map", None) is not None) and len(
            set(model.hf_device_map.values()).intersection({"cpu", "disk"})
        ) > 0:
            remove_hook_from_submodules(model)

        if isinstance(config, PromptLearningConfig) and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable

        # if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
        model = cls(model, config, adapter_name)
        # else:
        #     model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config, adapter_name)
        model.load_adapter(model_id, adapter_name, **kwargs)
        return model