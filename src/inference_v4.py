
#! version4: adding self-understand-instruction prompt method
#! fixing when tokenizer has no pad_token_id, because in official train, we manually add pad_token, but when lora training, it doesn't save tokenizer 
#! thus, need to manuall add same pad token here 
#! but later found it doesn't matter
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
from utils import generate, PROMPT_DICT, LANGS, SELF_UNDERSTAND, str2bool, smart_tokenizer_and_embedding_resize
from peft import PeftModel

DEFAULT_PAD_TOKEN = "[PAD]"

def generate_prompt(prompt_type, instruction, input=''):
    # if lang == 'en':
    if input:
        return PROMPT_DICT[prompt_type]['prompt_input'].format_map({'instruction':instruction, 'input':input})
    else:
        return PROMPT_DICT[prompt_type]['prompt_no_input'].format_map({'instruction':instruction})
   


    
def DDP_Print(text):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(text)


#! adding language prompt for instruction, e.g  "请给我5个旅游计划，用英语回答我"
def add_language_for_instruction(instruction:str, src_lang='en', tgt_lang='zh'):
    return instruction + LANGS[src_lang][tgt_lang]

#! adding self-understanding
def add_self_understand(instruction:str):
    #! question's lang
    self_understand_instruction = SELF_UNDERSTAND[args.src_lang]
    
    if args.add_lang:
        self_understand_instruction = add_language_for_instruction(self_understand_instruction, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    #! instruction as self-understand input, get result for add-lang-instruction
    #TODO: 直接拼接感觉有点问题
    prompt = generate_prompt(args.prompt_type, self_understand_instruction, instruction)
    self_understand_result = get_result(prompt=prompt)
    return self_understand_result

def get_result(prompt:str):
    """
        Args:instruction, input
        Return: templated instruction, result
    """
    
    if 'GLM' in args.model_type:
        result, history = model.chat(tokenizer, prompt, history=[])
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
    
    if 'T5' in args.model_type or args.prompt_type == 'origin':
        pass
    
    elif 'stanford' in args.prompt_type:
        result = result.split("### Response:")[1].strip()

    elif 'BELLE' in args.prompt_type:
        result = result.replace(prompt, "")
    return result

def inference(test_file_path:str, output_file_path:str, dataset_type='self-instruct'):
    #! change output_file name
    output_file_name = output_file_path.split('.jsonl')[0]
    output_file_path = f'{output_file_name}+add_lang-{args.add_lang}+add_self_understand-{args.add_self_understand}.jsonl'
    
    with jsonlines.open(test_file_path, 'r') as f, jsonlines.open(output_file_path, 'w') as o:
        for line in tqdm(f, desc='Inferencing'):
            if 'self-instruct' in dataset_type:
                instruction, input, target, davinci_response = line['instruction'], line['input'], line['target'], line['response']
                del line['prompt']
                line['davinci_response'] = line.pop('response')
                
            elif 'seed_task' in dataset_type:
                instruction, input, target = line['instruction'], line['input'], line['output']
            
            
            #! add lang
            if args.add_lang:
                instruction = add_language_for_instruction(instruction, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
            
            
            #! add self understand
            if args.add_self_understand:
                self_understand_result = add_self_understand(instruction)
                #! concat the self-understand result to original instruction
                instruction += self_understand_result

            prompt = generate_prompt(args.prompt_type, instruction, input)
            result = get_result(prompt)
            line['instruction'] = instruction #! add-self-understand-instruction
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
    parser.add_argument('--no_cuda', type=str2bool, default=False, help='use gpu or not')
    parser.add_argument('--use_lora', type=str2bool, default=False, help='use_lora')
    parser.add_argument('--deepspeed_inference', type=str2bool, default=False, help='use deepspeed or not')
    parser.add_argument('--model_type', default='FLAN-T5', type=str, required=False, help='GPT2 or ALPACA or LLAMA or FLAN-T5')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='weight path')
    parser.add_argument('--lora_weights_path', type=str, default=None, help='lora weight path')
    parser.add_argument('--pretrained_tokenizer_path', type=str, default=None, help='tokenizer path')
    parser.add_argument('--local_rank', type=str, default=None, help='for deepspeed distributed inference')
    parser.add_argument('--dataset_type', type=str, default=None, help='inference dataset')
    parser.add_argument('--src_lang', type=str, default='en', help='the lang of instruction')
    parser.add_argument('--tgt_lang', type=str, default='en', help='the lang of response')
    parser.add_argument('--prompt_type', type=str, default='stanford', help='the prompt type')
    parser.add_argument('--add_lang', type=str2bool, default=False, help='add lang info in instruction')
    parser.add_argument('--add_self_understand', type=str2bool, default=False, help='first generate understanding response')
    
    # parser.add_argument('--prompt', type=str, default=None, help='模型存放位置')
    
    args = parser.parse_args()
    
    DDP_Print(args)
    # try:
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
    if args.use_lora:
        DDP_Print('catch lora weight ...')
        model = PeftModel.from_pretrained(
            model,
            args.lora_weights_path,
            torch_dtype=torch.float16,
        )
        
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer_path, trust_remote_code=True)
    
    #! 有的模型的 tokenizer 的 pad token 是 None，重新添加一下
    if tokenizer.pad_token is None:
        DDP_Print('adding pad token ...')
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    model.to(device)
    # ! GLM deepspeed 封装以后没有 chat 方法
    if args.deepspeed_inference and 'GLM' not in args.model_type:
        # local_rank = int(os.getenv('LOCAL_RANK', '0'))
        local_rank = args.local_rank
        world_size = int(os.getenv('WORLD_SIZE', '1'))
        DDP_Print('Using Deepspeed accelerating ...')
        DDP_Print(f'Local rank:{local_rank} \t World_size:{world_size}')
        model = deepspeed.init_inference(model, mp_size=world_size, dtype=torch.float16, replace_with_kernel_inject=True)
    inference(args.test_file_path, args.output_file_path, args.dataset_type)
    logger.info('Done')
        
    # except RuntimeError:
    #     import pdb; pdb.set_trace()
    

