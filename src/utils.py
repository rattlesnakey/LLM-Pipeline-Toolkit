# -*- coding: utf-8 -*-
import json
from torch.utils.data import Dataset, DataLoader
import os
import io
from typing import Optional, Sequence, Union, List, Dict
import transformers
from tqdm import tqdm
import argparse
import torch
import numpy as np
import shutil
import json
import copy
from transformers import GenerationConfig, pipeline
import torch.nn.functional as F
from peft import get_peft_model_state_dict
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


LANGS = {
    'en': {
        'zh':' Please response in Chinese.',
        'en':' Please response in English.'
    },
    'zh':{
        'en':' 请用英文回答。',
        'zh':' 请用中文回答。'
    }
}

PROMPT_DICT = {
    'stanford':{
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    },
    'BELLE':{
        'prompt_input':(
            "<s>Human: {instruction}{input}\n\nAssistant: " 
        ),
        'prompt_no_input':(
           "<s>Human: {instruction}\n\nAssistant: " 
        )
    },
    'origin':{
        'prompt_input':(
            "### Instruction:{instruction}{input}\n\n### Response:" 
        ),
        'prompt_no_input':(
           "Question:{instruction}\n\n Answer:" 
        )
    }
}

EXAMPLE_PROMPT = {
    'X_CSQA_text':(
        "Question: {question}\n"
        "### Response:{answer}\n\n"
    ),
    'X_CSQA_choice':(
        "Question: {question}\n"
        "Options:\n"
        "{option_content}"
        "### Response:{answer}\n\n"
    ),
    'X_NLI':(
        "Premise: {premise}\n"
        "Hypothesis: {hypothesis}\n"
        "### Response:{answer}\n\n"
    )
}


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



def instruction_preprocess(dataset_type, line, demonstration=None, args=None):
    """
        demonstration:List[Dict]
    """
    options_list = None; target = None; input = None
    if 'self-instruct' in dataset_type:
        instruction, input, target, davinci_response = line['instruction'], line['input'], line['target'], line['response']
        del line['prompt']
        line['davinci_response'] = line.pop('response')
        
    elif 'seed_task' in dataset_type or 'translation' in dataset_type:
        instruction, input, target = line['instruction'], line['input'], line['output']
        
    elif 'X_CSQA' in dataset_type:
        instruction, options_list = X_CSQA_process(line, demonstration, args)

        
    elif 'X_NLI' in dataset_type:
        instruction, options_list = X_NLI_process(line, demonstration, args)

    return instruction, input, target, options_list



def X_NLI_process(line, demonstration, args):
    premise = line['premise']
    hypothesis = line['hypothesis']
    
    #! probing text or letter choice
    if args.probing_text:
        options_list = ["entailment", "neutral", "contradiction"]
        answer_key = 'label'
        PROB_TYPE = 'X_NLI_text'
    else:
        options_list = ["A", "B", "C"]
        answer_key = 'answerKey'
        PROB_TYPE = 'X_NLI_choice'
        
    if demonstration:
        example = ''
        for demo in demonstration:
            demo_premise, demo_hypothesis = demo['premise'], demo['hypothesis']
            
            #! demonstration's example type, if probing text, the answer should be the text
            demo_answer = demo[answer_key]
            
            #! exp is a dict {'premise':xx, 'hypothesis':xx, 'answer':xxx}
            exp = {
                'premise':demo_premise, 
                'hypothesis':demo_hypothesis,
                'answer':demo_answer
            }
            #! NLI's example prompt is the same for probing text or probing choice
            cur_example = EXAMPLE_PROMPT['X_NLI'].format_map(exp)
            example += cur_example
        instruction = TASK_PROMPT['use_example'][PROB_TYPE].format_map({'premise':premise, 'hypothesis':hypothesis, 'example':example})
    else:
        instruction = TASK_PROMPT['no_example'][PROB_TYPE].format_map({'premise':premise, 'hypothesis':hypothesis})
    return instruction, options_list


def parse_CSQA_line(line, args):
    question = line['question']['stem']
    choices = line['question']['choices']
    #! dynamic num option
    option_content = ""
    options_list = []
    
    #! cur sample's groundtruth choice
    label_choice = line['answerKey']
    
    for choice in choices:
        
        #! text or choice letter
        if args.probing_text:
            answer_key = 'text'
            
        else:
            cur_option_text = f"{choice['label']}: {choice['text']}\n"
            option_content += cur_option_text
            answer_key = 'label'
        
        options_list.append(choice[answer_key])

        #! get label text 
        if choice['label'] == label_choice:
            label_text = choice['text']
            
    
        
    return question, option_content, options_list, label_text
    

def X_CSQA_process(line, demonstration, args):
    
    question, option_content, options_list, label_text = parse_CSQA_line(line, args)
    
    #! letter choice or text
    if args.probing_text:
        PROB_TYPE = "X_CSQA_text"

    else:
        PROB_TYPE = "X_CSQA_choice"

        
    if demonstration:
        example = ''
        for demo in demonstration:
            
            #! don't need to use demo option_list
            demo_question, demo_option_content, _, demo_label_text = parse_CSQA_line(demo, args)
            
            #! probing text or choice
            if args.probing_text:
                demo_answer = demo_label_text
            else:
                demo_answer = demo['answerKey']
                
            #! exp is a dict {'question':xx, 'option_content':xx, 'answer':xxx}
            #! format map will ignore extra key, i.e., option_content for choice_text
            exp = {
                'question':demo_question, 
                'option_content':demo_option_content,
                'answer':demo_answer
            }
            cur_example = EXAMPLE_PROMPT[PROB_TYPE].format_map(exp)
            example += cur_example
        
        if args.probing_text:
            cur_input_dict = {
                'question':question,
                'example':example,
            }
        else:
            cur_input_dict = {
                'question':question, 
                'option_content':option_content, 
                "options":', '.join(options_list), 
                'num_option':len(options_list), 
                'example':example,
            }
            
        instruction = TASK_PROMPT['use_example'][PROB_TYPE].format_map(cur_input_dict)
    else:
        if args.probing_text:
            cur_input_dict = {
                'question':question,
            }
        else:
            cur_input_dict = {
                'question':question, 
                'option_content':option_content, 
                "options":', '.join(options_list), 
                'num_option':len(options_list), 
            }
        instruction = TASK_PROMPT['no_example'][PROB_TYPE].format_map(cur_input_dict)
    return instruction, options_list

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Mydataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
        

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index]
    
    

MyGenerationConfig = {
    'greedy':{
        'temperature':0.0,
        'top_p':100,
        'top_k':0, 
        'num_beams':1,
        'do_sample':False,
        'repetition_penalty':1.2
    },
    'sample':{
        'temperature':0.35,
        'top_p':0.85,
        'top_k':40, 
        'num_beams':1,
        'do_sample':True,
        'repetition_penalty':1.2
    }
}


def ddp_batch_probing_choice_text(
    tokenizer,
    model,
    device,
    dataloader, 
    accelerator,
    rank: int = 0,
    **kwargs
):
    results = dict()

    #! set padding side to right
    accelerator.print('set tokenizer padding side to right')
    tokenizer.padding_side = 'right'
    for data_tuple in tqdm(dataloader, desc='Inferencing ...', total=len(dataloader), disable=not rank):
        #! instructions: List[str], options_list: List[List[str]]
        idxes, instructions, options_lists = data_tuple
        idxes = idxes.tolist()

        #! concat instructions and corresponding candidate choice text
        #! one case to build a batch, batch size is choice size
        for idx, instruction, cand_choice_texts in zip(idxes, instructions, options_lists):
            #! json loads
            cand_choice_texts = json.loads(cand_choice_texts)
            
            #! each instruction concat with cand choice text 
            cur_text_batch = []
            for cand_choice_text in cand_choice_texts:
                whole_text = instruction + cand_choice_text
                #! delete last token for input
                cur_text_batch.append(whole_text)
            
            
            
            #! input encoding
            input_enc = tokenizer(cur_text_batch, return_tensors="pt", padding=True, return_length=True, add_special_tokens=True).to(device)
            #! get each sample valid subword length
            input_lens = input_enc.pop('length')
            input_lens = input_lens.tolist()
            
            #! only answer's encoding, add_special_token need to set False
            whole_cand_choice_enc = tokenizer(cand_choice_texts, return_tensors="pt", padding=True, return_length=True, add_special_tokens=False).to('cpu')
            #! get choice text subword length
            choice_text_lens = whole_cand_choice_enc.pop('length')
            choice_text_lens = choice_text_lens.tolist()
            whole_cand_choice_enc = whole_cand_choice_enc['input_ids']
            

            new_input_enc = {}; new_input_lens = []
            #! delete the last token
            for k, v in input_enc.items():
                new_vs = []
                for i in range(len(input_lens)):
                    new_v = torch.cat([v[i,:input_lens[i]-1], v[i, input_lens[i]:]])
                    new_vs.append(new_v)
                
                new_input_enc[k] = torch.stack(new_vs).to(v.device)
            
            #! update input_lens
            for input_len in input_lens:
                new_input_lens.append(input_len - 1)
            
            with torch.no_grad():
                #! (batch_size, sequence_length, config.vocab_size)
                logits = model(**new_input_enc).logits
            
            choice_text_logits_sums = []
            #! only extract choice part logits
            #! batch_size = len(input_lens)
            for i in range(len(new_input_lens)):
                choice_text_start_idx = new_input_lens[i] - choice_text_lens[i]
                choice_text_end_idx = new_input_lens[i]
                #! (choice_len, vocab_size)
                choice_logits = logits[i, choice_text_start_idx:choice_text_end_idx, :]
                choice_probs = F.log_softmax(choice_logits, dim=-1).cpu()
                
                #! (choice_len), get each groundtruth token's logit
                #! groundtruth choice_text_encoding, (choice_len)
                ground_truth_choice_encoding = whole_cand_choice_enc[i,:choice_text_lens[i]]
                
                #! ground_truth_choice_logits: (choice_len,)
                ground_truth_choice_logits = torch.gather(choice_probs, dim=1, index=ground_truth_choice_encoding.unsqueeze(-1)).squeeze(-1)
                
                
                choice_text_logits_sum = float(ground_truth_choice_logits.sum())
                choice_text_logits_sums.append(choice_text_logits_sum)
            
            pred_idx = np.argmax(choice_text_logits_sums)
            
            #! based on text probing, but pred with choice letter for further acc count
            #! A, B, C, D, or E
            pred = chr(ord('A') + pred_idx)
            
            results.update({f'{idx}': f"{pred}"})
        
            
            
    return results

def ddp_batch_probing_choice(
    tokenizer,
    model,
    device,
    dataloader, 
    accelerator,
    rank: int = 0,
    **kwargs
):
    results = dict()

    for data_tuple in tqdm(dataloader, desc='Inferencing ...', total=len(dataloader), disable=not rank):
        idxes, data, options_lists = data_tuple
        idxes = idxes.tolist()
        data = data
        options_lists = options_lists
        
        #! left padding
        inputs = tokenizer(data, return_tensors="pt", padding=True).to(device)
        
        
        with torch.no_grad():
            #! (batch_size, sequence_length, config.vocab_size)
            #! left padding, so last token is not pad token
            logits = model(**inputs).logits
            last_token_logits = logits[:,-1,:]
        
        for i in range(len(idxes)):
            idx = idxes[i]
            logit = last_token_logits[i]
            #! ["A", "B" ..]
            #! json load to list type
            options_list = json.loads(options_lists[i])
            
            option_probs = []
            for option in options_list:
                #! add_special_token= False, other wise it will prepend bos token
                option_id = tokenizer(option, add_special_tokens=False).input_ids[0]
                option_probs.append(logit[option_id])
                
            #! have to use .float() for using softmax
            option_probs = torch.nn.functional.softmax(torch.tensor(option_probs).float(), dim=0).detach().cpu().numpy()
            pred = options_list[np.argmax(option_probs)]
            results.update({f'{idx}': f"{pred}"})

            
            
    return results




def ddp_batch_generate(
    tokenizer,
    model,
    device,
    dataloader, 
    accelerator,
    temperature: float = 0.35,
    top_p: float = 0.85,
    top_k: int = 40,
    num_beams:int = 4,
    repetition_penalty: float = 1.2, 
    max_new_tokens: int = 256,
    do_sample: bool = False,
    rank: int = 0,
    **kwargs
):
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        **kwargs,
    )
    
    
    results = dict()
    
    for data_tuple in tqdm(dataloader, desc='Inferencing ...', total=len(dataloader), disable=not rank):
        idxes, data = data_tuple
        idxes = idxes.tolist()
        data = list(data)
        
        inputs = tokenizer(data, return_tensors="pt", padding=True).to(device)
        
        
        with torch.no_grad():
            generation_output = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        for idx, output in zip(idxes, outputs):
            #! index: result 
            results.update({f'{idx}': f"{output}"})
            
    return results


def generate(
    tokenizer, 
    model,
    device,
    prompt: str, 
    temperature: float = 0.35,
    top_p: float = 0.85,
    top_k: int = 40,
    num_beams:int = 4,
    repetition_penalty: float = 1.2, 
    max_new_tokens: int = 256,
    do_sample: bool = False,
    **kwargs
):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        **kwargs,
    )
    with torch.no_grad():
        generate_ids = model.generate(
            input_ids=inputs.input_ids, 
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
        )
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return result





def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def dump_json(path, item):
    with open(path, "w+") as f:
        json.dump(item, f, indent=4, sort_keys=True)
        
        
        

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


#TODO: Adding Peft save model callback
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")      
        
        old_state_dict = kwargs["model"].state_dict
        #! replace only with peft param
        cur_state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(kwargs["model"], type(kwargs["model"]))
        
        kwargs["model"].state_dict = cur_state_dict
        kwargs["model"].save_pretrained(checkpoint_folder)
        kwargs["model"].state_dict = old_state_dict
        WEIGHT_NAME = 'adapter_model.bin' 
        torch.save(cur_state_dict(), f'{checkpoint_folder}/{WEIGHT_NAME}')
        
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        
        global_ckpt_dir_path = os.path.join(checkpoint_folder, f'global_step{state.global_step}')
        if os.path.exists(global_ckpt_dir_path):
            shutil.rmtree(global_ckpt_dir_path)
        return control

