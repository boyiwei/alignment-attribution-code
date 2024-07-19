import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
from vllm import LLM

from lib.eval import eval_ppl, eval_zero_shot, eval_attack
from lib.prune import check_sparsity

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

SAVE_PATH = './ckpts' # path to save rewinded model temporarily (for VLLM)

modeltype2path = {
    # Fine-tuning related models
    'llama2-7b-chat': '', # the same as 'llama2-7b-chat-hf'
    'llama2-7b-chat-ft-pure-bad-10': '',
    'llama2-7b-chat-ft-pure-bad-50': '',
    'llama2-7b-chat-ft-pure-bad-100': '',
}

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        modeltype2path[model_name], 
        torch_dtype=torch.bfloat16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama2-7b-chat-ft-pure-bad-100', help='Model name to evaluate.')
    parser.add_argument('--model_no_ft', type=str, default='llama2-7b-chat', help='Path to original chat model (not finetuned). Used only when `mask` is specified.')
    parser.add_argument('--mask', type=str, default=None, help='Path to mask for rewinding weights.')
    parser.add_argument('--prompt_template_style', type=str, default="base", help='Prompt template style to use.')
    parser.add_argument('--seed', type=int, default=0, help='Seed.')
    parser.add_argument("--recover_from_base", action="store_true")
    parser.add_argument("--neg_mask", action="store_true")
    
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--save', type=str, default="out/ft_attack", help='Path to save results.')
    parser.add_argument('--alias', type=str, default=None, help='Alias.')
    # parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument("--eval_attack", action="store_true")
    parser.add_argument("--save_attack_res", action="store_true")
    
    args = parser.parse_args()


    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Load model
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(modeltype2path[args.model], use_fast=False)
    
    if args.mask is not None:
        print(f"loading original (not fine-tuned) llm model {args.model_no_ft}")
        model_no_ft = get_llm(args.model_no_ft, args.cache_dir)
        model_no_ft.eval()
        
        mask = torch.load(args.mask)
        print(f"Loaded weight mask from {args.mask}!")
        
        
        mask_num = 0
        total_num = 0
        for ((name, module), (name_no_ft, module_no_ft)) in zip(model.named_modules(), model_no_ft.named_modules()):
            if name in mask.keys():
                cur_mask = mask[name]
                if args.neg_mask:
                    module.weight.data[~cur_mask] = module_no_ft.weight.data[~cur_mask]
                else:
                    module.weight.data[cur_mask] = module_no_ft.weight.data[cur_mask] # rewind weights
                if args.neg_mask:
                    mask_num += cur_mask.eq(False).int().sum()
                else:
                    mask_num += cur_mask.eq(True).int().sum()
                total_num += cur_mask.numel()
        
        print(f"{(100 * mask_num / total_num):.2f}% weight entries are rewinded.\n")
    
    else:
        model_no_ft = None

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    # ################################################################
    # print("*"*30)
    # if not args.recover_from_base and args.sparsity_ratio > 0:
    #     sparsity_ratio = check_sparsity(model)
    # else:
    #     sparsity_ratio = args.sparsity_ratio
    # print(f"sparsity sanity check {sparsity_ratio:.4f}")
    # print("*"*30)
    # ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"test.out")
    
    
    if args.save_attack_res:
        save_attackpath = os.path.join(args.save, f"{args.alias}")
        print(save_attackpath)
        if not os.path.exists(save_attackpath):
            os.makedirs(save_attackpath)
    else:
        save_attackpath = ''
        
    if not os.path.exists(save_filepath):
        with open(save_filepath, "w") as f:
            print("alias\tmetric\tscore", file=f, flush=True)
            print(f"{args.alias}\tPPL\t{ppl_test:.4f}", file=f, flush=True)
    else:
        with open(save_filepath, "a") as f:
            print(f"{args.alias}\tPPL\t{ppl_test:.4f}", file=f, flush=True)

    if args.eval_attack:
        # note: since vLLM only supports loading from the path, we need to save the pruned model first for faster evaluation. We can reuse this temp folder to save disk spaces
        pruned_path = os.path.join(SAVE_PATH, f'tmp.ckpt')   
        model.save_pretrained(pruned_path)
        vllm_model = LLM(model=pruned_path, tokenizer=modeltype2path[args.model], dtype='bfloat16')
        
        # vllm_model = LLM(model=modeltype2path[args.model], tokenizer=modeltype2path[args.model], dtype='bfloat16')
        # for include_inst in [True, False]:
        for include_inst in [True]:
            suffix = "inst_" if include_inst else "no_inst_"
            print("********************************")
            
            score = eval_attack(vllm_model, tokenizer, num_sampled=1, add_sys_prompt=True, prompt_template_style=args.prompt_template_style, do_sample=False, save_attack_res=args.save_attack_res, include_inst=include_inst, filename=os.path.join(save_attackpath, f'{suffix}basic.jsonl'))
            print(f"attack evaluation results ({suffix}basic): {score:.4f}")
            with open(save_filepath, "a") as f:
                print(f"{args.alias}\t{suffix}ASR_basic\t{score:.4f}", file=f, flush=True)

            # print("********************************")
            # score = eval_attack(vllm_model, tokenizer, num_sampled=1, add_sys_prompt=False, prompt_template_style=args.prompt_template_style, do_sample=False, save_attack_res=args.save_attack_res, include_inst=include_inst, filename=os.path.join(save_attackpath, f'{suffix}basic_no_sys.jsonl'))
            # print(f"attack evaluation results ({suffix}basic, no sys prompt): {score:.4f}")
            # with open(save_filepath, "a") as f:
            #     print(f"{args.alias}\t{suffix}ASR_basic_nosys\t{score:.4f}", file=f, flush=True)
            
            
            # print("********************************")
            # score = eval_attack(vllm_model, tokenizer, num_sampled=5, add_sys_prompt=False, prompt_template_style=args.prompt_template_style, do_sample=True, save_attack_res=args.save_attack_res, include_inst=include_inst, filename=os.path.join(save_attackpath, f'{suffix}multiple_no_sys.jsonl'))
            # print(f"attack evaluation results ({suffix}multiple, no sys prompt): {score:.4f}")
            # with open(save_filepath, "a") as f:
            #     print(f"{args.alias}\t{suffix}ASR_multiple_nosys\t{score:.4f}", file=f, flush=True)

    

if __name__ == '__main__':
    main()
