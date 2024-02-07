import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
from vllm import LLM

from lib.prune import (
    prune_wanda,
    prune_random,
    prune_magnitude,
    prune_sparsegpt,
    prune_ablate,
    check_sparsity,
    find_layers,
    prune_wanda_decouple_activations,
    get_mask,
    prune_wandg_set_difference,
)
from lib.model_wrapper import prune_wanda_v2, prune_wandg
from lib.model_wrapper_low import make_low_rank
from lib.eval import eval_ppl, eval_zero_shot, eval_attack

print("torch", version("torch"))
print("transformers", version("transformers"))
print("accelerate", version("accelerate"))
print("# of gpus: ", torch.cuda.device_count())

SAVE_PATH = "temp"

modeltype2path = {
    "llama2-7b-chat-hf": "",
    "llama2-13b-chat-hf": "",
    "llama2-7b-hf": "",
    "llama2-13b-hf": "",
}


def get_llm(model_name, cache_dir="llm_weights"):
    if model_name in [
        "llama2-7b-chat-hf",
        "llama2-13b-chat-hf",
        "llama2-7b-hf",
        "llama2-13b-hf",
    ]:
        model = AutoModelForCausalLM.from_pretrained(
            modeltype2path[model_name],
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

    model.seqlen = model.config.max_position_embeddings
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama2-7b-chat-hf")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration samples."
    )
    parser.add_argument("--prune_method", type=str, choices=["low_rank"])
    parser.add_argument(
        "--prune_data",
        type=str,
        choices=[
            "wikitext",
            "alpaca",
            "alpaca_cleaned",
            "alpaca_cleaned_no_safety",
            "align",
            "align_short",
            "misalign",
            "align_misalign",
            "misalign_align",
            "align_short_misalign",
            "none",
        ],
        default="alpaca_cleaned_no_safety",
    )

    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument("--save", type=str, default=None, help="Path to save results.")
    parser.add_argument(
        "--save_model", type=str, default=None, help="Path to save the pruned model."
    )
    parser.add_argument(
        "--top_remove", action="store_true", help="Remove the top ranks."
    )
    parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument("--eval_attack", action="store_true")
    parser.add_argument("--save_attack_res", action="store_true")
    parser.add_argument(
        "--entangle_prompt_feat",
        dest="disentangle",
        action="store_false",
        help="entangle the prompt and response when computing the wanda score",
    )
    parser.add_argument(
        "--dump_U", action="store_true", help="dump the U matrix for analysis"
    )

    # low rank
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--niter", type=int, default=20)

    args = parser.parse_args()

    print("Disentangle:", args.disentangle)

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        modeltype2path[args.model], use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    device = torch.device("cuda:0")
    if (
        "30b" in args.model or "65b" in args.model
    ):  # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

    if args.prune_method == "low_rank":
        make_low_rank(args, model, tokenizer, device, prune_data=args.prune_data)

    ################################################################
    print("*" * 30)

    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    if args.save_attack_res:
        save_attackpath = os.path.join(args.save, f"attack_{args.rank}")
        print(save_attackpath)
        if not os.path.exists(save_attackpath):
            os.makedirs(save_attackpath)
    else:
        save_attackpath = ""
    if not os.path.exists(save_filepath):
        with open(save_filepath, "w") as f:
            print("method\trank_removed\tmetric\tscore", file=f, flush=True)
            print(
                f"{args.prune_method}\t{args.rank}\tPPL\t{ppl_test:.4f}",
                file=f,
                flush=True,
            )
    else:
        with open(save_filepath, "a") as f:
            print(
                f"{args.prune_method}\t{args.rank}\tPPL\t{ppl_test:.4f}",
                file=f,
                flush=True,
            )

    if args.eval_attack:
        # note: since vLLM only supports loading from the path, we need to save the pruned model first for faster evaluation. We can reuse this temp folder to save disk spaces
        pruned_path = os.path.join(SAVE_PATH, f"tmp_vllm_model")
        model.save_pretrained(pruned_path)
        vllm_model = LLM(
            model=pruned_path,
            tokenizer=modeltype2path[args.model],
            dtype="bfloat16",
            swap_space=64,
        )
        if True:
            vllm_model.llm_engine.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        for include_inst in [True, False]:
            suffix = "inst_" if include_inst else "no_inst_"
            print("********************************")

            score = eval_attack(
                vllm_model,
                tokenizer,
                num_sampled=1,
                add_sys_prompt=True,
                do_sample=False,
                save_attack_res=args.save_attack_res,
                include_inst=include_inst,
                filename=os.path.join(save_attackpath, f"{suffix}basic.jsonl"),
            )
            print(f"attack evaluation results ({suffix}basic): {score:.4f}")
            with open(save_filepath, "a") as f:
                print(
                    f"{args.prune_method}\t{args.rank}\t{suffix}ASR_basic\t{score:.4f}",
                    file=f,
                    flush=True,
                )

            print("********************************")
            score = eval_attack(
                vllm_model,
                tokenizer,
                num_sampled=1,
                add_sys_prompt=False,
                do_sample=False,
                save_attack_res=args.save_attack_res,
                include_inst=include_inst,
                filename=os.path.join(save_attackpath, f"{suffix}basic_no_sys.jsonl"),
            )
            print(
                f"attack evaluation results ({suffix}basic, no sys prompt): {score:.4f}"
            )
            with open(save_filepath, "a") as f:
                print(
                    f"{args.prune_method}\t{args.rank}\t{suffix}ASR_basic_nosys\t{score:.4f}",
                    file=f,
                    flush=True,
                )

            print("********************************")
            score = eval_attack(
                vllm_model,
                tokenizer,
                num_sampled=5,
                add_sys_prompt=False,
                do_sample=True,
                save_attack_res=args.save_attack_res,
                include_inst=include_inst,
                filename=os.path.join(
                    save_attackpath, f"{suffix}multiple_no_sys.jsonl"
                ),
            )
            print(
                f"attack evaluation results ({suffix}multiple, no sys prompt): {score:.4f}"
            )
            with open(save_filepath, "a") as f:
                print(
                    f"{args.prune_method}\t{args.rank}\t{suffix}ASR_multiple_nosys\t{score:.4f}",
                    file=f,
                    flush=True,
                )
        score = eval_attack(
            vllm_model,
            tokenizer,
            num_sampled=1,
            add_sys_prompt=False,
            gcg=True,
            do_sample=False,
            save_attack_res=args.save_attack_res,
            include_inst=True,
            filename=os.path.join(save_attackpath, f"gcg.jsonl"),
        )
        print(f"attack evaluation results (gcg): {score:.4f}")
        with open(save_filepath, "a") as f:
            print(
                f"{args.prune_method}\t{args.rank}\tASR_gcg\t{score:.4f}",
                file=f,
                flush=True,
            )
        del vllm_model

    if args.eval_zero_shot:
        accelerate = False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate = True

        task_list = [
            "boolq",
            "rte",
            "hellaswag",
            "winogrande",
            "arc_challenge",
            "openbookqa",
        ]
        num_shot = 0
        results = eval_zero_shot(
            modeltype2path[args.model],
            model,
            tokenizer,
            task_list,
            num_shot,
            accelerate,
            limit=200,
        )
        print("********************************")
        print("zero_shot evaluation results")
        sum_acc = 0
        with open(save_filepath, "a") as f:
            for k, v in results["results"].items():
                print(
                    f"{args.prune_method}\t{args.rank}\t{k}\t{v['acc']:.4f}",
                    file=f,
                    flush=True,
                )
                sum_acc += v["acc"]
            print(
                f"{args.prune_method}\t{args.rank}\taveraged\t{sum_acc/len(task_list):.4f}",
                file=f,
                flush=True,
            )

        print(results)


if __name__ == "__main__":
    main()
