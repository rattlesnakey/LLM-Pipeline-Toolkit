from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse 
import os
import torch

DEFAULT_PAD_TOKEN = "[PAD]"


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer,
    model, 
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding_ckpt_dir",
        type=str,
        help="path to the desired checkpoint folder, e.g., path/checkpoint-12")
    
    parser.add_argument(
        "--backbone_dir",
        type=str,
        help='same as embedding_ckpt_dir'
    )
    args = parser.parse_args()

    embedding_state = torch.load(os.path.join(args.embedding_ckpt_dir, 'pytorch_model.bin'))
    backbone = AutoModelForCausalLM.from_pretrained(args.backbone_dir)
    backbone_tokenizer = AutoTokenizer.from_pretrained(args.backbone_dir)
    
    #! add pad token
    if backbone_tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=backbone_tokenizer,
            model=backbone,
        )
    
    
    backbone.load_state_dict(embedding_state, strict=False)
    # import pdb; pdb.set_trace()
    backbone.save_pretrained(args.embedding_ckpt_dir)
    os.remove(os.path.join(os.path.join(args.embedding_ckpt_dir, 'pytorch_model.bin')))