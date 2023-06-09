# LLM-Pipeline-Toolkit
🚀 This repo includes the code of  instruction tuning  (**full fine-tuning**, **lora** and **prompt tuning** PEFT **with Deepspeed**) and inferencing (**interact** and **ddp batch inference**) current prevalent LLM (e.g. LLaMA, BELLE). Also, it support different prompt types (e.g. stanford_alpaca, BELLE) and different eval tasks with [llm_evaluation_harness](https://github.com/EleutherAI/lm-evaluation-harness).



## 📄 Get Started

## 📝 Setup

**Manually install**

```shell
conda env create -f environment.yaml
or
pip install -r requirements.txt
```



**Docker**

I Have pack up an docker image, which get pull from docker hub 

```shell
docker pull hengyuan/llm:v4
```



## ⛳️ Setting

You should download the instruction tuning dataset and put it into `datas` folder, also, you should download the pre-trained model weights and put it into `pretrained_models` folder.



### Deepspeed Config

Change different size model with corresponding deepspeed config, for example, fully fine-tune a 7B LM, you probably should use zero_3_offload_config, if you want change the optimizer and scheduler in training stage, you should cutomize the corresponding config value in deepspeed config.

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": "auto"
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```



### Variable

**Change the variable** in `pipeline_train_eval.sh`, the important variables are as follows: 

```shell
#! MODEL_ARGS
declare -a MODEL_NAMES
MODEL_NAMES=('pre-trained model name, e.g. LLAMA')
declare -a PRETRAINED_MODELS
PRETRAINED_MODELS=('pre-trained model path, e.g. pretrained_models/llama/7b')
declare -a TRAIN_DATASETS

#! change dataset file path here!!
TRAIN_DATASETS=('train dataset path, e.g. alpaca_data.json')

TRAIN_DATASET_TYPES=('train dataset type, e.g. alpaca_data')

LRS=('1e-4')
USE_LORAS=('True')
USE_PROMPT=('False')
PROMPT_TOKENS=('100')
ONLY_TUNE_EMBED=('False')
EPOCHS=('4')

```

You should customize the `MODEL_NAMES`, `PRETRAINED_MODELS`, `TRAIN_DATASETS`, `TRAIN_DATASET_TYPES`, `LRS`,`USE_LORAS`,`USE_PROMPT`,`PROMPT_TOKENS`,`ONLY_TUNE_EMBED`,`EPOCHS` .



### Prompt

In `utils.py`, You can change your task prompts here, the repo now supports XCSQA, XNLI tasks' prompts

```json
TASK_PROMPT = {
    'no_example':{
        'X_CSQA_text':(
            "{question}"
        ),
        'X_NLI_text':(
            "Given the premise: {premise} and hypothesis: {hypothesis} \nYour task is to determine the relationship between the premise and hypothesis by selecting one of the following options: 'entailment', 'neutral', or 'contradiction'.\n"
            "entailment means the hypothesis follows logically from the information contained in the premise.\n"
            "neutral means it is not possible to determine whether the hypothesis is true or false without further information.\n"
            "contradiction means the hypothesis is logically false from the information contained in the premise.\n"
        ),
        'X_CSQA_choice':(
            "Given the question: {question} \nChoose a more reasonable answer from the {num_option} options {options}. The options are as follows:\n"
            "{option_content}"
            "Please only output the option letter you selected and do not output any other content. \n"
        ),
        'X_NLI_choice':(
            "Given the premise: {premise} and hypothesis: {hypothesis} \nYou need to decide whether the hypothesis is entailed by the premise by choosing from the 3 options A, B, C. The options are as follows:\n"
            "A: entailment\n"
            "B: neutral\n"
            "C: contradiction\n\n"
            "entailment means the hypothesis follows logically from the information contained in the premise.\n"
            "neutral means it is not possible to determine whether the hypothesis is true or false without further information.\n"
            "contradiction means the hypothesis is logically false from the information contained in the premise.\n"
            "Please only output the option letter you selected and do not output any other content.\n"
        ),
    },
    'use_example':{
        'X_CSQA_text':(
            "Given a question, You need to provide a concise and accurate answer to the question.\n\n"
            "### Example:\n{example}\n\n"
            "Question: {question}\n" 
        ),
        'X_NLI_text':(
            "Given a premise and its corresponding hypothesis, your task is to determine the relationship between the premise and hypothesis by selecting one of the following options: 'entailment', 'neutral', or 'contradiction'.\n"
            "entailment means the hypothesis follows logically from the information contained in the premise.\n"
            "neutral means it is not possible to determine whether the hypothesis is true or false without further information.\n"
            "contradiction means the hypothesis is logically false from the information contained in the premise.\n\n\n"
            "### Example:\n{example}\n\n"
            "Premise: {premise}\n"
            "Hypothesis: {hypothesis}\n"
        ),
        'X_CSQA_choice':(
            "Given a question, you need to choose a reasonable answer from the {num_option} options {options}.\n"
            "Only output the option letter you selected and do not output any other content.\n\n"
            "### Example:\n{example}\n\n"
            "Question: {question}\n" 
            "Options:\n"
            "{option_content}\n"
        ),
        'X_NLI_choice':(
            "Given a premise and its corresponding hypothesis, you need to decide whether the hypothesis is entailed by the premise by choosing from the 3 options A, B, C.\n"
            "The options are as follows:\n"
            "A: entailment\n"
            "B: neutral\n"
            "C: contradiction\n\n"
            "entailment means the hypothesis follows logically from the information contained in the premise.\n"
            "neutral means it is not possible to determine whether the hypothesis is true or false without further information.\n"
            "contradiction means the hypothesis is logically false from the information contained in the premise.\n"
            "Please only output the option letter you selected and do not output any other content.\n\n"
            "### Example:\n{example}\n\n"
            "Premise: {premise}\n"
            "Hypothesis: {hypothesis}\n"
        ),
    } 
}
```



### 📥 Run

```shell
#! train and eval on task datasets
bash pipeline_train_eval.sh

#! interact
bash interact.sh
```





# 📝 Customized

### Ddp_batch_inference

The repo implements different decoding strategies (e.g. generation based, choices' logits softmax based, probing text log-likelyhood based ), you can add your customized decoding strategy in `utils.py` and implement in `ddp_batch_inference.py` .

```python
dataset = Mydataset(prompts=prompts)
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=32, pin_memory=True)

dataloader = accelerator.prepare_data_loader(data_loader=dataloader)
results = ddp_batch_generate(
  		tokenizer=tokenizer, 
  		model=model,
  		device=device,
  		dataloader=dataloader, 
  		temperature=args.temperature, 
  		top_p=args.top_p if args.do_sample else None, 
  		top_k=args.top_k if args.do_sample else None,
  		num_beams=args.num_beams,
  		repetition_penalty=args.repetition_penalty,
  		max_new_tokens=args.max_new_tokens,
  		do_sample=args.do_sample,
  		pad_token_id=tokenizer.pad_token_id,
  		eos_token_id=tokenizer.eos_token_id,
  		rank=accelerator.process_index,
  		accelerator=accelerator
)
.....
results = probing_method(
  		tokenizer=tokenizer, 
  		model=model,
  		device=device,
  		dataloader=dataloader,
  		rank=accelerator.process_index,
  		accelerator=accelerator,
)
```



### Extra Large Dataset

The repo include IterableDataset, which use streaming mode to loading large dataset to avoid cpu out of memory. You can customized your own dataset class in `train.py`.

```python
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
```



## 📚 Citation

If you find this repo useful, please cite us:

``` 
@misc{LLM-Pipeline-Toolkit,
  author = {Hengyuan Zhang},
  title = {LLM-Pipeline-Toolkit: An efficient pipeline toolkit for training and evaluating LLM},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rattlesnakey/LLM-Pipeline-Toolkit}},
}
```

