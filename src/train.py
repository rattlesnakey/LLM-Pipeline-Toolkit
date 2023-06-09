
# -*- coding: utf-8 -*-
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import torch
import transformers
from torch.utils.data import Dataset, IterableDataset, DataLoader
from transformers import Trainer
import sys
import utils
from utils import *
from typing import List
from transformers.deepspeed import is_deepspeed_zero3_enabled
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    PromptTuningInit,
    PromptTuningConfig,
    get_peft_model_state_dict,
)
from accelerate import Accelerator

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"




@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    is_extra_large_model: bool = field(default=False, metadata={"help": "model size bigger that 13b"})
    lora_weights_path: Optional[str] = field(default=None)

    

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    prompt_type: str = field(default='stanford', metadata={"help": "prompt type to use"})
    stream_loading: bool = field(default=False, metadata={"help": "use stream mode to load data and do preprocessing, the input_file need to be a jsonline file"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    #! post-pretraining for embedding matric
    only_tune_embedding: bool = field(default=False, metadata={"help": "tuning embedding or not"})
    
    
    #! lora
    lora_train: bool = field(default=False, metadata={"help": "use lora training or not"})
    #! lora hyperparams
    lora_r: int = field(
        default=8,
        metadata={"help": " r dim for lora param"},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": " alpha / r = scaling, to scale lora param"},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": " lora dropout rate "},
    )

    #! prompt tuning
    prompt_train: bool = field(default=False, metadata={"help": "use prompt tuning or not"})
    prompt_tokens: int = field(
        default=200,
        metadata={"help": "number of virtual trainable tokens"},
    )
   

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

#! only response part has loss 
def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

#! Extra large dataset
class IterSupervisedDataset(IterableDataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, prompt_type: str):
        super(IterSupervisedDataset, self).__init__()
        self.data_path = data_path
        self.prompt_type = prompt_type
        self.tokenizer = tokenizer
    
    def add_prompt(self, line):
        try:
            line = json.loads(line)
        except Exception:
            raise ValueError("stream mode lead to load a jsonline file, please change file mode to jsonline.")
            sys.exit()
        prompt_input, prompt_no_input = PROMPT_DICT[self.prompt_type]["prompt_input"], PROMPT_DICT[self.prompt_type]["prompt_no_input"]
        cur_input = prompt_input.format_map(line) if line.get("input", "") != "" else prompt_no_input.format_map(line)
        cur_output = f"{line['output']}{self.tokenizer.eos_token}"
        return dict(sources=cur_input, targets=cur_output)
    
    def __iter__(self):
        f = open(self.data_path, 'r') 
        add_prompt_iter = map(self.add_prompt, f)
        #! return add_prompt iterator
        #! each line is a dict, {'instruction':xxx, 'input':xx, 'output':xxx}
        return add_prompt_iter
    
    def __len__(self):
        datasize = 0
        f = open(self.data_path, 'r')
        for line in f:
            datasize += 1
        return datasize

        
            
            
        

    
#! Extra large dataset
@dataclass
class IterDataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources, targets = tuple([instance[key] for instance in instances] for key in ("sources", "targets"))
        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids = data_dict['input_ids']
        labels = data_dict['labels']
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, prompt_type: str):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT[prompt_type]["prompt_input"], PROMPT_DICT[prompt_type]["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
       
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    #! extra large data use stream loading method
    if data_args.stream_loading:
        logging.warning("Stream Loading ...")
        train_dataset = IterSupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, prompt_type=data_args.prompt_type)
        data_collator = IterDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, prompt_type=data_args.prompt_type)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)



def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    #! when using stream loading, can't remove_unused_colums, it will remove collate_fn's output
    if data_args.stream_loading:
        training_args.remove_unused_columns = False
    
    logging.warning("Loading model ...")
    
    #! extra large model
    if model_args.is_extra_large_model:
        model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
        
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        low_cpu_mem_usage=True,
    )
    
    assert training_args.lora_train != training_args.prompt_train, "Choose lora_train or prompt_train"
    
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left", #! same padding side for batch inference
        use_fast=False,
    )
    #! add pad token
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    #! only if LLaMA should add bos, eos, bos
    #! if other model doesn't have bos, eos, unk, also need to add 
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
    #! only tune embedding matrix
    if training_args.only_tune_embedding:
        logging.warning("Only tuning embedding param, fixing other param ..")
        for param in model.parameters():
            param.requires_grad_(False)

        logging.warning("Activating input require_grad hook ..")
        model.enable_input_require_grads()
        
    #! only use lora training
    if training_args.lora_train:
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=["q_proj", "v_proj",] if 'llama' in model_args.model_name_or_path else ["c_proj",], 
            lora_dropout=training_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        logging.warning("Adapt to Lora Model ...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    #! use prompt tuning after merging lora weight
    if training_args.prompt_train:
        assert model_args.lora_weights_path != None, "Please use --lora_weights_path to load the lora weights for merging!"
        
        model = PeftModel.from_pretrained(
            model,
            model_args.lora_weights_path,
            is_trainable=False, #! need to change, 让 adapter 的部分还是可以 train 的
        )
    
        logging.warning("Merging Lora Weight ...")
        model.print_trainable_parameters()

        #! merge lora
        model = model.merge_and_unload()
        
        for param in model.parameters():
            param.requires_grad = True

        
        prompt_config = PromptTuningConfig(
            task_type="CAUSAL_LM",
            prompt_tuning_init=PromptTuningInit.RANDOM,
            num_virtual_tokens=training_args.prompt_tokens,
            tokenizer_name_or_path=model_args.model_name_or_path,
        )
        logging.warning("Adding Virtual tokens to Model ...")

        model = get_peft_model(model, prompt_config)
        model.print_trainable_parameters()
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()

    
    #! only save peft param
    if training_args.lora_train or training_args.prompt_train:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))
        

   
    accelerator = Accelerator()
    
    #! save_models
    accelerator.wait_for_everyone()
    
    if is_deepspeed_zero3_enabled():
        #! if use zero3, need to use accelerator to get state dict
        accelerator.print('using zero3 mode saving ...')
        state_dict = accelerator.get_state_dict(model)
    else:
        state_dict = model.state_dict()

    model.save_pretrained(training_args.output_dir)
    WEIGHT_NAME = 'adapter_model.bin' if training_args.lora_train or training_args.prompt_train else 'pytorch_model.bin'
    torch.save(state_dict, f'{training_args.output_dir}/{WEIGHT_NAME}')

    trainer.save_state()
    accelerator.wait_for_everyone()
   
if __name__ == "__main__":
    train()