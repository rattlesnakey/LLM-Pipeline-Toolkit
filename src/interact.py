# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, LLaMAForCausalLM, AutoModelForSeq2SeqLM, AutoModelForCausalLM, GenerationConfig
from loguru import logger
import argparse, os
import deepspeed
import torch
from peft import PeftModel
from utils import MyGenerationConfig
from utils import generate, PROMPT_DICT, str2bool



def generate_prompt(prompt_type, instruction, input=''):
    if input:
        return PROMPT_DICT[prompt_type]['prompt_input'].format_map({'instruction':instruction, 'input':input})
    else:
        return PROMPT_DICT[prompt_type]['prompt_no_input'].format_map({'instruction':instruction})
    
    
def DDP_Print(text):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_beams', default=4, type=int, required=False, help='beam size')
    parser.add_argument('--temperature', default=0.35, type=float, required=False, help='temperature')
    parser.add_argument('--top_p', default=0.85, type=float, required=False, help='topp')
    parser.add_argument('--top_k', default=40, type=int, required=False, help='Topk')
    parser.add_argument('--repetition_penalty', default=1.2, type=float, required=False, help='repetition penalty')
    parser.add_argument('--max_new_tokens', default=256, type=int, required=False, help='max new tokens')
    parser.add_argument('--no_cuda', action='store_true', help='use cuda or not')
    parser.add_argument('--deepspeed_inference', action='store_true', help='use deepspeed or not')
    parser.add_argument('--model_type', default='FLAN-T5', type=str, required=False, help='LLAMA or FLAN-T5')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='ckpt path')
    parser.add_argument('--pretrained_tokenizer_path', type=str, default=None, help='tokenizer path')
    parser.add_argument('--local_rank', type=str, default=None, help='for deepspeed distributed inference')
    parser.add_argument('--prompt_type', type=str, default='stanford', help='prompt type')
    parser.add_argument('--use_lora', type=str2bool, default=False, help='use_lora')
    parser.add_argument('--lora_weights_path', type=str, default=None, help='lora weight path')
    parser.add_argument('--decoding_mode', type=str, default=None, help='decoding mode')
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda 
    
    if args.local_rank:
        device = f'cuda:{args.local_rank}' if args.cuda else 'cpu'
    else:
        device = 'cuda' if args.cuda else 'cpu'
        
    DDP_Print(f'Using {device} ...')
    DDP_Print('loading model ...')
    
    if args.model_type == 'FLAN-T5':
        model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    elif args.model_type == 'LLAMA':
        model = LLaMAForCausalLM.from_pretrained(args.pretrained_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    if args.use_lora:
        DDP_Print('catch lora weight ...')
        model = PeftModel.from_pretrained(
            model,
            args.lora_weights_path,
            torch_dtype=torch.float16,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer_path)
    
    model.to(device)
    
    if args.deepspeed_inference:
        local_rank = args.local_rank
        world_size = int(os.getenv('WORLD_SIZE', '1'))
        DDP_Print('Using Deepspeed accelerating ...')
        DDP_Print(f'Local rank:{local_rank} \t World_size:{world_size}')
        
        model = deepspeed.init_inference(model, mp_size=world_size, dtype=torch.float16, replace_with_kernel_inject=True)

    Instruction = None
    while True:
        if Instruction:
            not_keep = input("Weather to replace the previous instruction ? press enter if keep, else input new instruction:\n")
            if not not_keep:
                pass
            else:
                Instruction = input("Please input the Instruction:\n")
                
        else:
            Instruction = input("Please input the Instruction:\n")
        DDP_Print(f'Instruction:{Instruction}\n\n')
        
        
        Input = input("Please input the Input: (if not need input, just press enter)\n")
        DDP_Print(f'Input:{Input}\n\n')
        
        prompt = generate_prompt(args.prompt_type, Instruction, Input)
        DDP_Print(f'Prompt:{prompt}\n\n')
    
        
        DDP_Print('generating response ...\n')
        if args.decoding_mode:
            DDP_Print(f'Using {args.decoding_mode} strategy')
            result = generate(
                tokenizer=tokenizer, 
                model=model,
                device=device,
                prompt=prompt, 
                max_new_tokens=args.max_new_tokens,
                **MyGenerationConfig[args.decoding_mode]
            )
        else:
            result = generate(tokenizer=tokenizer, 
                                model=model,
                                device=device,
                                prompt=prompt, 
                                temperature=args.temperature, 
                                top_p=args.top_p, 
                                top_k=args.top_k,
                                num_beams=args.num_beams,
                                repetition_penalty=args.repetition_penalty,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False if args.num_beams > 1 else True,
                    )
        result = result.split("### Response:")[1].strip()
    
        DDP_Print(f'Response:{result}')
        DDP_Print('#'*100+'\n\n\n\n')

    
    
    
    

