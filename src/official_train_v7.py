
#! adding instruction length for special attention mechanism
#! solving left padding problem
import copy
import logging
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, LLaMAForCausalLM
from typing import Optional, Dict, Sequence
import jsonlines
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
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModelForCausalLM,
    PromptTuningConfig,
)
from accelerate import Accelerator
from my_prompt_tuning_utils import MyPeftModelForCausalLM, MyLLaMAForCausalLM, MyPromptTuningConfig, prepare_prompt_learning_config

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

    # lora_target_modules: List[str] = field(
    #     default=["q_proj", "v_proj",],
    #     metadata={"help": " target layer to append lora param "},
    # )
    #! prompt tuning
    prompt_train: bool = field(default=False, metadata={"help": "use prompt tuning or not"})
    prompt_tokens: int = field(
        default=200,
        metadata={"help": "number of virtual trainable tokens"},
    )
    use_special_attention_mechanism: bool = field(default=False, metadata={"help": "make prompt tokens cannot attended by other tokens"})
    
    # merge_lora: bool = field(default=False, metadata={"help": "merge lora weight or not"})
    

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

    #! use avarage embedding to initialize
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    #! each text padding to max_length
    #! 这边是一条一条 tokenize 的，所以其实是没有 padding 的
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
    #! labels should also put padding token to IGNORE_INDEX
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    #! adding instruction length here
    return dict(input_ids=input_ids, labels=labels, instruction_lens=sources_tokenized["input_ids_lens"])

#! Change to IterableDataset 
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
        #! here is right 
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
        # except Exception:
        #     raise ValueError("stream mode lead to load a jsonline file, please change file mode to jsonline.")
        #     sys.exit()
        
            
            
        

    
#! Put processing to here
@dataclass
class IterDataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        #! 这边数据是空的，不知道为啥
        sources, targets = tuple([instance[key] for instance in instances] for key in ("sources", "targets"))
        #! put preprocess here
        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids = data_dict['input_ids']
        labels = data_dict['labels']
        #TODO: check whether there is padding or not 
        #! why here padding again ?
        #! default batch longest strategy
        #! 这边其实是 right padding 的, 之前 tokenize_fn 其实只是 tokenize 了而已
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            #TODO: 专门针对 pad token 的 mask
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
        #! add eos_token
        #! seems like it doesn't use bos token, 
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        #! should be added eos token
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        #! add instruction length
        self.instruction_lens = data_dict["instruction_lens"]
        

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], instruction_lens=self.instruction_lens[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        #! add instruction lens, 这个应该不需要是 torch tensor 吧
        if training_args.use_special_attention_mechanism:
            input_ids, labels, instruction_lens = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "instruction_lens"))
            
            #! 修改成用 tokenizer padding
            input_ids = self.tokenizer.pad({"input_ids":input_ids}, padding=True, return_tensors='pt')["input_ids"]
            # input_ids = torch.nn.utils.rnn.pad_sequence(
            #     input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            # )
            #! for label padding
            labels = self.tokenizer.pad({"input_ids":labels}, padding=True, return_tensors='pt')["input_ids"]
            labels = labels.masked_fill(labels==self.tokenizer.pad_token_id, IGNORE_INDEX)
            
            padding_lengths = []
            for input_id in input_ids:
                padding_indice = torch.where(input_id == self.tokenizer.pad_token_id)[0].tolist()
                if padding_indice:
                    padding_length = padding_indice[-1] + 1
                else:
                    padding_length = 0
                padding_lengths.append(padding_length)
                
            
            return dict(
                input_ids=input_ids,
                labels=labels,
                instruction_lens=instruction_lens,
                padding_lengths=padding_lengths,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )
        else:
            input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
            
            input_ids = self.tokenizer.pad({"input_ids":input_ids}, padding=True, return_tensors='pt')["input_ids"]
            
            #! for label padding
            labels = self.tokenizer.pad({"input_ids":labels}, padding=True, return_tensors='pt')["input_ids"]
            labels = labels.masked_fill(labels==self.tokenizer.pad_token_id, IGNORE_INDEX)
            
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    #! change here
    if data_args.stream_loading:
        logging.warning("Stream Loading ...")
        train_dataset = IterSupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, prompt_type=data_args.prompt_type)
        data_collator = IterDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, prompt_type=data_args.prompt_type)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)



def train():
    
    #! when using stream loading, can't remove_unused_colums, it will remove collate_fn's output
    # if data_args.stream_loading:
    #! instruction_lens 也会被它抹掉
    if data_args.stream_loading or training_args.use_special_attention_mechanism:
        training_args.remove_unused_columns = False
    
    if training_args.prompt_train and training_args.use_special_attention_mechanism:
        Pretrained_Model_CLS = MyLLaMAForCausalLM
        Peft_Model_CLS = MyPeftModelForCausalLM

    else:
        Pretrained_Model_CLS = AutoModelForCausalLM
        Peft_Model_CLS = PeftModelForCausalLM

        
    
    #! extra large model
    if model_args.is_extra_large_model:
            
        #! change for MyLLaMAForCausalLM
        logging.warning("Loading model in fp16...")
        model = Pretrained_Model_CLS.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True #! zero3 的时候不能用
    )   
    else:
        #! 用 MyLLaMA 的时候会爆内存，不知道为啥
        #! 所以就都用 float16 吧
        logging.warning("Loading model in fp32 ...")
        model = Pretrained_Model_CLS.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            low_cpu_mem_usage=True
        )
    # import pdb; pdb.set_trace()
    
    
    assert training_args.lora_train != training_args.prompt_train, "Choose lora_train or prompt_train"
    
    
    #! tokenizer 要先处理
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
    #! only use lora training
    if training_args.lora_train:
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=["q_proj", "v_proj",] if 'llama' in model_args.model_name_or_path else ["c_proj",], #! 记得更换如果不是 LLaMA
            lora_dropout=training_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        logging.warning("Adapt to Lora Model ...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    #! use prompt tuning after merging lora weight
    if training_args.prompt_train:
        #! 在进行 prompt tuning 的时候，要先 load lora weights 进来 
        #! 要么就是 load 进来原来的 lora weight 并且 merge 起来，然后再单独 train prompt 
        assert model_args.lora_weights_path != None, "Please use --lora_weights_path to load the lora weights for merging!"
        logging.warning("Loading Lora Weight ...")
        # assert training_args.merge_lora != training_args.lora_train, "when merge lora weights, don't use lora train"
        # ! PeftModel Load 进来的时候，全部参数的 grad 都是 false
        model = PeftModel.from_pretrained(
            model,
            model_args.lora_weights_path,
            is_trainable=False, #! need to change, 让 adapter 的部分还是可以 train 的
        )
        
        #! make it all trainable again
        logging.warning("Merging Lora Weight ...")
        model.print_trainable_parameters()

        #! 变回去原来的 Model Class 而不是 PeftModel Class 了
        #! 这里变回去后，所有的 param 都是不可 train
        #! gpt2 和 int8 的模型没法 merge
        #! 检查过了，merge 后没有问题
        model = model.merge_and_unload()
        
        #! 重新变成都是可以 train 的
        for param in model.parameters():
            param.requires_grad = True

        # import pdb; pdb.set_trace()
        #! prompt embedding 里面有复制原来的 word embedding matric
        #! 如果是后面才加的 pad token 的话就会有问题，所以要在最前面就加 pad token
        #! 加了 prompt tokens 之后占用的显存会一下子增加很多，可能单卡推理都没办法了
        prompt_config = MyPromptTuningConfig(
            task_type="CAUSAL_LM",
            prompt_tuning_init=PromptTuningInit.RANDOM,
            num_virtual_tokens=training_args.prompt_tokens,
            tokenizer_name_or_path=model_args.model_name_or_path,
            use_special_attention_mechanism=training_args.use_special_attention_mechanism,
        )
        logging.warning("Adding Virtual tokens to Model ...")
        #! 这边 save_pretrained 下面的 model_config 应该都是 PromptConfig 吧，Lora 虽然一起 train, 但不知道是不是会被覆盖掉
        #TODO: 这边直接用 PeftCausalLM 这个类就好了，不用 get peft model 了
        
        #! 完善 prompt config 
        prompt_config = prepare_prompt_learning_config(prompt_config, model)
        
        #! specify customized PeftForCaualLM
        model = Peft_Model_CLS(model, prompt_config)
        model.print_trainable_parameters()
        #TODO: Prompt 的 attention 部分也还没修改, 参考 https://github.com/huggingface/peft/blob/main/src/peft/peft_model.py#673
    

    
    accelerator.print(model)
    
    # with training_args.main_process_first(desc="dataset tokenization"):
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    
    callbacks = []
    #! 开始占用显存，trainer 里面会把 param 放到 cuda 上
    # if training_args.lora_train or training_args.prompt_train:
    #     callbacks=[SavePeftModelCallback]
    
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, callbacks=callbacks, **data_module)
    # import pdb; pdb.set_trace()
    trainer.train()
    # if trainer.is_local_process_zero():
    #     import pdb; pdb.set_trace()
    
    #! only save peft param
    # import pdb; pdb.set_trace()
    if training_args.lora_train or training_args.prompt_train:
        old_state_dict = model.state_dict
        #! replace only with peft param
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))
        

   
    
    
    #! save_models
    accelerator.wait_for_everyone()
    
    if is_deepspeed_zero3_enabled():
        #! if use zero3, need to use accelerator to get state dict
        accelerator.print('using zero3 mode saving ...')
        state_dict = accelerator.get_state_dict(model)
    else:
        state_dict = model.state_dict()
    #! 不知道为什么有时候 save_pretrained 下来的 adapter_model.bin 是空的，所以只能自己重新 save 了
    #! 现在保存的也都没问题了
    model.save_pretrained(training_args.output_dir)
    WEIGHT_NAME = 'adapter_model.bin' if training_args.lora_train or training_args.prompt_train else 'pytorch_model.bin'
    torch.save(state_dict, f'{training_args.output_dir}/{WEIGHT_NAME}')
    # if trainer.deepspeed:
    #     logging.warning('using deepspeed mode saving ...')
        
    # else:
    #     model.save_pretrained(training_args.output_dir)

    trainer.save_state()
    accelerator.wait_for_everyone()
    #! this save strategy when using deepspeed training will throw error
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    accelerator = Accelerator()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    train()