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
from utils import generate

def generate_prompt(model_type, instruction, input=''):
    # if lang == 'en':

    if 'BELLE' in model_type:
        if not tokenizer.bos_token:
            tokenizer.bos_token = ''
        if not input:
            return f"{tokenizer.bos_token}Human: {instruction} \n\nAssistant:"
        else:
            return f"{tokenizer.bos_token}Human: {instruction}{input} \n\nAssistant:"

    elif 'GLM' in model_type:
        return instruction
    else:
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    {instruction}
    ### Input:
    {input}
    ### Response:"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction:
    {instruction}
    ### Response:"""
   


    
def DDP_Print(text):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(text)

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
#TODO: 在 instruction 上面指定用某个语言, "用某个语言回答我"
def add_language_for_instruction(instruction:str, src_lang='en', tgt_lang='zh'):
    return instruction + LANGS[src_lang][tgt_lang]




def inference(test_file_path:str, output_file_path:str, dataset_type='self-instruct'):
    with jsonlines.open(test_file_path, 'r') as f, jsonlines.open(output_file_path, 'w') as o:
        for line in tqdm(f, desc='Inferencing'):
            if 'self-instruct' in dataset_type:
                instruction, input, target, davinci_response = line['instruction'], line['input'], line['target'], line['response']
                del line['prompt']
                line['davinci_response'] = line.pop('response')
                
            elif 'seed_task' in dataset_type:
                instruction, input, target = line['instruction'], line['input'], line['output']
            #! add target language
            if args.add_lang:
                instruction = add_language_for_instruction(instruction, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
            
            prompt = generate_prompt(args.model_type, instruction, input)
            if 'GLM' in args.model_type:
                result, history = model.chat(tokenizer, prompt, history=[])
            
            # response, history = model.chat(tokenizer, "你好", history=[])
            # import pdb; pdb.set_trace()
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
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id
                        )
            
            if 'T5' in args.model_type or 'GLM' in args.model_type:
                pass
                # line[f'_response'] = result
                
            
            elif 'LLAMA' in args.model_type or 'GPT2' in args.model_type or 'ALPACA' in args.model_type:
                result = result.split("### Response:")[1].strip()

            elif 'BELLE' in args.model_type:
                result = result.replace(prompt, "")
            line['instruction'] = instruction
            line[f'{args.model_type}_response'] = result
            
            o.write(line)
    





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--test_file_path', default=None, type=str, required=True, help='test file path')
    parser.add_argument('--output_file_path', default=None, type=str, required=True, help='output file path')
    
    parser.add_argument('--num_beams', default=4, type=int, required=False, help='beam size')
    parser.add_argument('--temperature', default=0.35, type=float, required=False,)
    parser.add_argument('--top_p', default=0.85, type=float, required=False,)
    parser.add_argument('--top_k', default=40, type=int, required=False, help='Topk')
    parser.add_argument('--repetition_penalty', default=1.2, type=float, required=False,)
    parser.add_argument('--max_new_tokens', default=256, type=int, required=False, help='max length')
    parser.add_argument('--no_cuda', action='store_true', help='use gpu or not')
    parser.add_argument('--deepspeed_inference', action='store_true', help='use deepspeed or not')
    parser.add_argument('--model_type', default='FLAN-T5', type=str, required=False, help='GPT2 or ALPACA or LLAMA or FLAN-T5')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='weight path')
    parser.add_argument('--pretrained_tokenizer_path', type=str, default=None, help='tokenizer path')
    parser.add_argument('--local_rank', type=str, default=None, help='for deepspeed distributed inference')
    parser.add_argument('--dataset_type', type=str, default=None, help='inference dataset')
    parser.add_argument('--src_lang', type=str, default='en', help='the lang of instruction')
    parser.add_argument('--tgt_lang', type=str, default='en', help='the lang of response')
    parser.add_argument('--add_lang', action='store_true', help='add lang info in instruction')
    # parser.add_argument('--prompt', type=str, default=None, help='模型存放位置')
    
    args = parser.parse_args()
    
    DDP_Print(args)
    try:
        args.cuda = torch.cuda.is_available() and not args.no_cuda  # 当用户使用GPU,并且GPU可用时
        
        if args.local_rank:
            device = f'cuda:{args.local_rank}' if args.cuda else 'cpu'
        else:
            device = 'cuda' if args.cuda else 'cpu'
            
        DDP_Print(f'Using {device} ...')
        DDP_Print('loading model ...')
        
        if args.model_type == 'FLAN-T5':
            model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
        elif 'LLAMA' in args.model_type or 'ALPACA' in args.model_type:
            model = LLaMAForCausalLM.from_pretrained(args.pretrained_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
        elif 'BELLE' in args.model_type or 'GPT2' in args.model_type or 'BLOOM' in args.model_type:
            model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
        elif 'GLM' in args.model_type:
            model = AutoModel.from_pretrained(args.pretrained_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True).to(device)
        # if 'GLM' in args.model_type:
        #     tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer_path, trust_remote_code=True)
        # else:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer_path, trust_remote_code=True)
        # import pdb; pdb.set_trace()
        # model.to(device)
        #! GLM deepspeed 封装以后没有 chat 方法
        if args.deepspeed_inference and 'GLM' not in args.model_type:
            # local_rank = int(os.getenv('LOCAL_RANK', '0'))
            local_rank = args.local_rank
            world_size = int(os.getenv('WORLD_SIZE', '1'))
            DDP_Print('Using Deepspeed accelerating ...')
            DDP_Print(f'Local rank:{local_rank} \t World_size:{world_size}')
            
            model = deepspeed.init_inference(model, mp_size=world_size, dtype=torch.float16, replace_with_kernel_inject=True)
        inference(args.test_file_path, args.output_file_path, args.dataset_type)
        logger.info('Done')
        
    except RuntimeError:
        import pdb; pdb.set_trace()
    
