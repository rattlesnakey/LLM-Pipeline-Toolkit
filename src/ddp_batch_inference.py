
#! add ddp inference
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
from utils import ddp_batch_generate, PROMPT_DICT, LANGS, Mydataset, X_CSQA_process, str2bool, instruction_preprocess, smart_tokenizer_and_embedding_resize
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
# BatchSampler, SequentialSampler
# torch.utils.data.BatchSampler
import torch.distributed as dist
from peft import PeftModel

DEFAULT_PAD_TOKEN = "[PAD]"

def generate_prompt(prompt_type, instruction, input=''):
    # if lang == 'en':
    if input:
        return PROMPT_DICT[prompt_type]['prompt_input'].format_map({'instruction':instruction, 'input':input})
    else:
        return PROMPT_DICT[prompt_type]['prompt_no_input'].format_map({'instruction':instruction})
   


    
# def DDP_Print(text):
#     if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
#         logger.info(text)


#! adding language prompt for instruction, e.g  "请给我5个旅游计划，用英语回答我"
def add_language_for_instruction(instruction:str, src_lang='en', tgt_lang='zh'):
    return instruction + LANGS[src_lang][tgt_lang]


def ddp_batch_inference(test_file_path:str, output_file_path:str, dataset_type='self-instruct'):
    prompts = []; lines = []; instructions = []
    
    #! change output_file name
    output_file_name = output_file_path.split('.jsonl')[0]
    output_file_path = f'{output_file_name}+add_lang-{args.add_lang}+add_self_understand-{args.add_self_understand}.jsonl'
    
    with jsonlines.open(test_file_path, 'r') as f, jsonlines.open(output_file_path, 'w') as o:
        for idx, line in enumerate(f):
            #! process instruction
            instruction, input = instruction_preprocess(dataset_type, line)
            
            #! add target language
            if args.add_lang:
                instruction = add_language_for_instruction(instruction, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
            
            prompt = generate_prompt(args.prompt_type, instruction, input)
            #! 因为 gather 回来得到的 response 的顺序是混乱的，所以在构建 dataloader 的时候把它对应的原始的 idx 也放进去
            prompts.append((idx, prompt))
            lines.append(line)
            instructions.append(instruction)
        
        if 'GLM' in args.model_type:
            raise ValueError("GLM haven't support batch inference, please use inference_v2")
        else:

            #! dataset 给他全部弄到一起去，然后带 id, 最后按顺序重组
            dataset = Mydataset(prompts=prompts)
            #! 直接用 DistributedSampler 去得到 Dataloader 和 accelerator.prepare_data_loader 都是可以的
            # sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=accelerator.process_index)
            # dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
            dataloader = DataLoader(dataset, batch_size=args.batch_size)

            dataloader = accelerator.prepare_data_loader(data_loader=dataloader)

            results = ddp_batch_generate(
                tokenizer=tokenizer, 
                model=model,
                device=device,
                dataloader=dataloader, 
                temperature=args.temperature, 
                top_p=args.top_p, 
                top_k=args.top_k,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                do_sample=False if args.num_beams > 1 else True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                rank=accelerator.process_index,
                accelerator=accelerator
            )
        
        
        accelerator.wait_for_everyone()
        all_results = [None for _ in range(world_size)]
        dist.all_gather_object(all_results, results)

        
        if accelerator.is_main_process:
            # import pdb; pdb.set_trace()
           
            final_all_results = dict()
            for results in all_results:
                #! 整合到一个字典里面
                final_all_results.update(results)
    
            for idx, line in enumerate(lines):
                #! 取出对应 index 的答案回来
                result = final_all_results[str(idx)]
                instruction = instructions[idx]
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
    
    # parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    # parser.add_argument('--test_file_path', default=None, type=str, required=True, help='test file path')
    # parser.add_argument('--output_file_path', default=None, type=str, required=True, help='output file path')
    # parser.add_argument('--batch_size', default=4, type=int, required=False, help='batch size')
    # parser.add_argument('--num_beams', default=4, type=int, required=False, help='beam size')
    # parser.add_argument('--temperature', default=0.35, type=float, required=False,)
    # parser.add_argument('--top_p', default=0.85, type=float, required=False,)
    # parser.add_argument('--top_k', default=40, type=int, required=False, help='Topk')
    # parser.add_argument('--repetition_penalty', default=1.2, type=float, required=False,)
    # parser.add_argument('--max_new_tokens', default=256, type=int, required=False, help='max length')
    # parser.add_argument('--no_cuda', action='store_true', help='use gpu or not')
    # parser.add_argument('--deepspeed_inference', action='store_true', help='use deepspeed or not')
    # parser.add_argument('--model_type', default='FLAN-T5', type=str, required=False, help='GPT2 or ALPACA or LLAMA or FLAN-T5')
    # parser.add_argument('--pretrained_model_path', type=str, default=None, help='weight path')
    # parser.add_argument('--lora_weights_path', type=str, default=None, help='lora weight path')
    # parser.add_argument('--pretrained_tokenizer_path', type=str, default=None, help='tokenizer path')
    # parser.add_argument('--local_rank', type=str, default=None, help='for deepspeed distributed inference')
    # parser.add_argument('--dataset_type', type=str, default=None, help='inference dataset')
    # parser.add_argument('--src_lang', type=str, default='en', help='the lang of instruction')
    # parser.add_argument('--tgt_lang', type=str, default='en', help='the lang of response')
    # parser.add_argument('--prompt_type', type=str, default='stanford', help='the prompt type')
    # parser.add_argument('--add_lang', action='store_true', help='add lang info in instruction')
    # parser.add_argument('--add_self_understand', type=str2bool, default=False, help='first generate understanding response')
    
    # parser.add_argument('--prompt', type=str, default=None, help='模型存放位置')
    
    args = parser.parse_args()
    # even_batches=False, split_batches=True
    accelerator = Accelerator()

    accelerator.print(args)
    # try:
    args.cuda = torch.cuda.is_available() and not args.no_cuda  # 当用户使用GPU,并且GPU可用时
    
    if args.local_rank:
        device = f'cuda:{args.local_rank}' if args.cuda else 'cpu'
    else:
        device = 'cuda' if args.cuda else 'cpu'
        
    accelerator.print(f'Using {device} ...')
    accelerator.print('loading model ...')
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
        accelerator.print('catch lora weight ...')
        model = PeftModel.from_pretrained(
            model,
            args.lora_weights_path,
            torch_dtype=torch.float16,
        )
    #! when using batch generate, need to config padding_side='left', because the eos token have to be the last token
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer_path, trust_remote_code=True, padding_side='left')
    #! 有的模型的 tokenizer 的 pad token 是 None，重新添加一下
    if tokenizer.pad_token is None:
        accelerator.print('adding pad token ...')
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
        
    #! GLM deepspeed 封装以后没有 chat 方法
    if args.deepspeed_inference and 'GLM' not in args.model_type:
        # local_rank = int(os.getenv('LOCAL_RANK', '0'))
        local_rank = args.local_rank
        world_size = int(os.getenv('WORLD_SIZE', '1'))
        accelerator.print('Using Deepspeed accelerating ...')
        accelerator.print(f'Local rank:{local_rank} \t World_size:{world_size}')
        #! word size 都设置成 1，要不 model parallel 的话会报错
        model = deepspeed.init_inference(model, mp_size=1, dtype=torch.float16, replace_with_kernel_inject=True)
    ddp_batch_inference(args.test_file_path, args.output_file_path, args.dataset_type)
    accelerator.print('Done')
        
    # except RuntimeError:
    #     import pdb; pdb.set_trace()
    
    
    

