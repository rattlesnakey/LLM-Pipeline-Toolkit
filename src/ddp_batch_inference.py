
# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, GenerationConfig, AutoModel
try:
    from transformers import LLaMAForCausalLM
except ImportError:
    from transformers import LlamaForCausalLM as LLaMAForCausalLM
from loguru import logger
import argparse, os
import deepspeed
import torch
import jsonlines
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from peft import PeftModel
from utils import ddp_batch_generate, MyGenerationConfig, ddp_batch_probing_choice, ddp_batch_probing_choice_text
from utils import PROMPT_DICT, LANGS, Mydataset, str2bool, instruction_preprocess, smart_tokenizer_and_embedding_resize
import json

DEFAULT_PAD_TOKEN = "[PAD]"

def generate_prompt(prompt_type, instruction, input=''):
    if input:
        return PROMPT_DICT[prompt_type]['prompt_input'].format_map({'instruction':instruction, 'input':input})
    else:
        return PROMPT_DICT[prompt_type]['prompt_no_input'].format_map({'instruction':instruction})
   




#! adding language prompt for instruction, e.g  "请给我5个旅游计划，用英语回答我"
def add_language_for_instruction(instruction:str, src_lang='en', tgt_lang='zh'):
    return instruction + LANGS[src_lang][tgt_lang]


def ddp_batch_inference(test_file_path:str, output_file_path:str, dataset_type='self-instruct', demonstration=None):
    prompts = []; lines = []; instructions = []
    
    #! change output_file name
    output_file_name = output_file_path.split('.jsonl')[0]
    output_file_path = f'{output_file_name}+add_lang-{args.add_lang}+use_exp-{args.use_demonstration}.jsonl'
    
    with jsonlines.open(test_file_path, 'r') as f, jsonlines.open(output_file_path, 'w') as o:
        for idx, line in enumerate(f):
            #! process instruction
            instruction, input, target, options_list = instruction_preprocess(dataset_type, line, demonstration, args)
            
            #! add target language
            if args.add_lang:
                instruction = add_language_for_instruction(instruction, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
            prompt = generate_prompt(args.prompt_type, instruction, input)
           
            
            #! probing strategy need to pass options_list
            #! need to dumps, otherwise when batching will get wrong options_list 
            if args.use_probing_strategy:
                prompts.append((idx, prompt, json.dumps(options_list)))
            else:
                prompts.append((idx, prompt))
                
            lines.append(line)
            instructions.append(instruction)

        
        if 'GLM' in args.model_type:
            raise ValueError("GLM haven't support batch inference, please use inference_v2")
        else:

            dataset = Mydataset(prompts=prompts)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=32, pin_memory=True)
            dataloader = accelerator.prepare_data_loader(data_loader=dataloader)

            if args.use_probing_strategy:
                if args.probing_text:
                    probing_method = ddp_batch_probing_choice_text
                    accelerator.print(f'Using Probing Choice text ...')
                else:
                    probing_method = ddp_batch_probing_choice
                    accelerator.print(f'Using Probing Choice letter ...')
                    
                results = probing_method(
                    tokenizer=tokenizer, 
                    model=model,
                    device=device,
                    dataloader=dataloader,
                    rank=accelerator.process_index,
                    accelerator=accelerator,
                )
                
                    
                
            else:
                if args.decoding_mode:
                    accelerator.print(f'Using {args.decoding_mode} ...')
                    results = ddp_batch_generate(
                        tokenizer=tokenizer, 
                        model=model,
                        device=device,
                        dataloader=dataloader,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        rank=accelerator.process_index,
                        accelerator=accelerator,
                        **MyGenerationConfig[args.decoding_mode]
                )
                else:
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
        
        
        accelerator.wait_for_everyone()
        all_results = [None for _ in range(world_size)]
        dist.all_gather_object(all_results, results)
        if accelerator.is_main_process:
            final_all_results = dict()
            for results in all_results:
                final_all_results.update(results)
    
            for idx, line in enumerate(lines):
                result = final_all_results[str(idx)]
                instruction = instructions[idx]
                
                if args.use_probing_strategy:
                    pass
                else:
                    if 'T5' in args.model_type or args.prompt_type == 'origin':
                        pass
                    elif args.prompt_type == 'stanford':
                        result = result.split("### Response:")[1].strip()

                    elif args.prompt_type == 'BELLE':
                        result = result.replace(prompts[idx][1], "")
                line['instruction'] = instruction
                line[f'{args.model_type}_response'] = result
                o.write(line)
        accelerator.wait_for_everyone()
        




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file_path', default=None, type=str, required=True, help='test file path')
    parser.add_argument('--output_file_path', default=None, type=str, required=True, help='output file path')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='batch size')
    parser.add_argument('--num_beams', default=4, type=int, required=False, help='beam size')
    parser.add_argument('--temperature', default=0.35, type=float, required=False,)
    parser.add_argument('--top_p', default=0.85, type=float, required=False,)
    parser.add_argument('--top_k', default=40, type=int, required=False, help='Topk')
    parser.add_argument('--repetition_penalty', default=1.2, type=float, required=False,)
    parser.add_argument('--max_new_tokens', default=256, type=int, required=False, help='max length')
    parser.add_argument('--no_cuda', type=str2bool, default=False, help='use gpu or not')
    parser.add_argument('--use_lora', type=str2bool, default=False, help='use_lora')
    parser.add_argument('--use_prompt', type=str2bool, default=False, help='use_lora')
    parser.add_argument('--deepspeed_inference', type=str2bool, default=False, help='use deepspeed or not')
    parser.add_argument('--model_type', default='FLAN-T5', type=str, required=False, help='GPT2 or ALPACA or LLAMA or FLAN-T5')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='pre-trained model weight path')
    parser.add_argument('--lora_weights_path', type=str, default=None, help='lora weight path')
    parser.add_argument('--prompt_weights_path', type=str, default=None, help='lora weight path')
    parser.add_argument('--pretrained_tokenizer_path', type=str, default=None, help='tokenizer path')
    parser.add_argument('--local_rank', type=str, default=None, help='for deepspeed distributed inference')
    parser.add_argument('--dataset_type', type=str, default=None, help='inference dataset')
    parser.add_argument('--src_lang', type=str, default='en', help='the lang of instruction')
    parser.add_argument('--tgt_lang', type=str, default='en', help='the lang of response')
    parser.add_argument('--prompt_type', type=str, default='stanford', help='the prompt type')
    parser.add_argument('--add_lang', type=str2bool, default=False, help='add lang info in instruction')
    parser.add_argument('--is_extra_large_model', type=str2bool, default=False, help='larger than 13b')
    parser.add_argument('--use_demonstration', type=str2bool, default=False, help='few-shot or zero-shot')
    parser.add_argument('--decoding_mode', type=str, default=None, help='wheather use default decoding mode or not')
    parser.add_argument('--use_probing_strategy', type=str2bool, default=False, help='probing decoding strategy')
    parser.add_argument('--probing_text', type=str2bool, default=False, help='probing on choice text or only choice letter')

    
    args = parser.parse_args()
    accelerator = Accelerator()

    accelerator.print(args)
    args.cuda = torch.cuda.is_available() and not args.no_cuda 
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    
    if args.local_rank:
        device = f'cuda:{args.local_rank}' if args.cuda else 'cpu'
    else:
        device = 'cuda' if args.cuda else 'cpu'
        
    accelerator.print(f'Using {device} ...')
    accelerator.print('loading model ...')
    if args.model_type in 'FLAN-T5':
        model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    elif 'LLAMA' in args.model_type or 'ALPACA' in args.model_type:
        if args.is_extra_large_model:
            #! extra large model
            cur_memory_mapping = {i:( "32GiB" if i == int(args.local_rank) else "0GiB") for i in range(world_size)}
            model = LLaMAForCausalLM.from_pretrained(args.pretrained_model_path, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16, max_memory=cur_memory_mapping)
        else:
            model = LLaMAForCausalLM.from_pretrained(args.pretrained_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    elif 'BELLE' in args.model_type or 'GPT' in args.model_type or 'BLOOM' in args.model_type or 'OPT' in args.model_type:
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    elif 'GLM' in args.model_type:
        model = AutoModel.from_pretrained(args.pretrained_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True).to(device)

    if args.use_lora:
        accelerator.print('catch lora weight ...')
        model = PeftModel.from_pretrained(
            model,
            args.lora_weights_path,
            torch_dtype=torch.float16,
        )
    if args.use_prompt:
        accelerator.print('catch prompt weight ...')
        model = PeftModel.from_pretrained(
            model,
            args.prompt_weights_path,
            torch_dtype=torch.float16,
        )
    #! when using batch generate, need to config padding_side='left', because the eos token have to be the last token
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer_path, trust_remote_code=True, padding_side='left')
    

    if tokenizer.pad_token is None:
        accelerator.print('adding pad token ...')
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
      
            
    if args.deepspeed_inference and 'GLM' not in args.model_type:
        accelerator.print('Using Deepspeed accelerating ...')
        accelerator.print(f'Local rank:{args.local_rank} \t World_size:{world_size}')
        model = deepspeed.init_inference(model, mp_size=1, dtype=torch.float16, replace_with_kernel_inject=True)
    
    demonstration = None
    if args.use_demonstration:
        accelerator.print('Loading demonstrations ...')
        demonstration = []
        demonstration_dir = os.path.dirname(args.test_file_path)
        demonstration_file = os.path.join(demonstration_dir, 'demonstration.jsonl')
        with jsonlines.open(demonstration_file, 'r') as f:
            for line in f:
                demonstration.append(line)
            
            
        
    ddp_batch_inference(args.test_file_path, args.output_file_path, args.dataset_type, demonstration)
    accelerator.print('Done')
        

    
    
    

