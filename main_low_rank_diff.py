# difference from model_wrapper.py
# 1. store all input activations.

import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from typing import Optional
from importlib.metadata import version
from lib.data import get_loaders
from lib.eval import eval_ppl, eval_zero_shot, eval_attack
from functools import reduce
from vllm import LLM


class ActLinear(nn.Module):
    """
    drop in replacement of nn.Linear
    """

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.activation_norms = []  # offload to CPU
        self.record_activation = True

    def clear_act_buffer(self):
        self.activation_norms = []

    def forward(self, x):
        if self.record_activation:
            if hasattr(self, "mask") and self.mask is not None:
                x_ = x[self.mask]  # num * dim
            else:
                x_ = x  # bs * seq_len * dim
            self.activation_norms.append(
                x_.view(-1, x_.shape[-1]).cpu()
            )  # offload to CPU.

        out = self.base(x)
        return out


class no_act_recording:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.record_activation = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.record_activation = True


class set_mask:
    def __init__(self, model, mask):
        self.model = model
        self.mask = mask

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.mask = self.mask

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.mask = None


def make_Act(model, verbose=False):
    replace_map = dict()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            replace_map[name] = ActLinear(module)

    for name, module in model.named_modules():
        if verbose:
            print("current:", name)
        for k, v in replace_map.items():
            k_ = k.split(".")
            name_prefix, name_suffix = ".".join(k_[:-1]), k_[-1]
            if name_prefix == "":  # outer layer
                if name == name_suffix:
                    if verbose:
                        print(" not modifying ", name_suffix)
                    # setattr(model, name_suffix, v)
            elif name == name_prefix:
                if verbose:
                    print("    modifying ", name_suffix, "inside", name)
                setattr(module, name_suffix, v)
    return model


def revert_Act_to_Linear(model):
    """
    Reverts ActLinear modules back to their original nn.Linear layers.
    """
    for name, module in model.named_modules():
        if isinstance(module, ActLinear):
            # Extract the base nn.Linear module from ActLinear
            linear_module = module.base
            # Navigate to the parent module of the ActLinear module
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            print(f"Reverting {name}, parent: {parent_name}")
            parent_module = (
                model
                if parent_name == ""
                else reduce(getattr, parent_name.split("."), model)
            )
            # Replace the ActLinear module with the extracted nn.Linear module
            setattr(parent_module, name.split(".")[-1], linear_module)

    return model


def clear_act_buffer(act_model):
    for name, module in act_model.named_modules():
        if isinstance(module, ActLinear):
            module.clear_act_buffer()


def make_low_rank(
    args,
    model,
    tokenizer,
    device=torch.device("cuda:0"),
    prune_data_pos="wikitext",
    prune_data_neg="wikitext",
):
    """
    prune_data_pos: retain most useful (total_rank - rank_pos) ranks
    prune_data_neg: remove most useful (total_rank - rank_neg) ranks

    final_W = p_proj @ W + p_proj_ortho @ n_proj_ortho @ W = W - p_proj_ortho @ n_proj @ W,
        with rank <= (total_rank - rank_pos) + min(rank_pos, rank_neg)
                  <= total_rank - (rank_pos - rank_neg) , if rank_pos > rank_neg
    """
    model = make_Act(model, verbose=False)
    model.requires_grad_(False)
    clear_act_buffer(model)

    # globally disable recording.
    for name, module in model.named_modules():
        if isinstance(module, ActLinear):
            module.record_activation = False

    # load dataset
    print(f"loading calibdation data {prune_data_pos}, {prune_data_neg}")
    dataloader_pos, _ = get_loaders(
        prune_data_pos,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
    )
    dataloader_neg, _ = get_loaders(
        prune_data_neg,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
    )
    print("dataset loading complete")

    num_hidden_layers = model.config.num_hidden_layers

    for layer in range(num_hidden_layers):
        layer_filter_fn = (
            lambda x: f"layers.{layer}." in x
        )  ### TODO # hack for llama series

        # enable recording for the current layer.
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                # print("enabling recording for ", name)
                module.record_activation = True

        activation_norms_pos = {}
        activation_norms_neg = {}

        # forward pass and get activation records.
        with torch.no_grad():
            for batch in dataloader_pos:
                inp, tar = batch[0].to(device), batch[1].to(device)

                assert args.disentangle, "should run in disentangle mode"
                mask = tar.ne(-100)
                with set_mask(model, mask):
                    model(inp)
        # save to buffer & clear recorded values
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                activation_norms_pos[name] = module.activation_norms
                module.activation_norms = []

        ## repeat for neg data
        # forward pass and get activation records.
        with torch.no_grad():
            for batch in dataloader_neg:
                inp, tar = batch[0].to(device), batch[1].to(device)

                assert args.disentangle, "should run in disentangle mode"
                mask = tar.ne(-100)
                with set_mask(model, mask):
                    model(inp)
        # save to buffer & clear recorded values
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                activation_norms_neg[name] = module.activation_norms
                module.activation_norms = []
        ######

        # make low_rank
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                print("making low rank: ", name)
                d_out, d_in = module.base.weight.data.shape
                total_rank = min(d_out, d_in)

                activation_norms_p = torch.cat(activation_norms_pos[name], dim=0).to(
                    device
                )  # size * d_in
                score_p = (
                    activation_norms_p @ module.base.weight.data.T
                )  # (size * d_in) @ (d_out * d_in).T --> (size, d_out)
                _, _, Vp = torch.svd_lowrank(
                    score_p.float(), q=total_rank - args.rank_pos, niter=args.niter
                )  # (size, r) (r) (d_out, r)
                Vp_proj = (Vp @ Vp.T).type(
                    module.base.weight.data.dtype
                )  # (d_out, d_out)

                activation_norms_n = torch.cat(activation_norms_neg[name], dim=0).to(
                    device
                )  # size * d_in
                score_n = (
                    activation_norms_n @ module.base.weight.data.T
                )  # (size * d_in) @ (d_out * d_in).T --> (size, d_out)
                _, _, Vn = torch.svd_lowrank(
                    score_n.float(), q=total_rank - args.rank_neg, niter=args.niter
                )  # (size, r) (r) (d_out, r)
                Vn_proj = (Vn @ Vn.T).type(
                    module.base.weight.data.dtype
                )  # (d_out, d_out)

                Vp_proj_ortho = (torch.eye(d_out, device=device) - Vp_proj).type(
                    module.base.weight.data.dtype
                )  # (d_out, d_out)

                module.base.weight.data.sub_(
                    Vp_proj_ortho @ (Vn_proj @ module.base.weight.data)
                )

        # disable recording for the current layer.
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                # print("disabling recording for ", name)
                module.record_activation = False
                module.clear_act_buffer()

        print(torch.cuda.memory_allocated() / 1024 / 1024 / 1024)

    model = revert_Act_to_Linear(model)
    model.zero_grad()  # freeze gradient to save cuda memory


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama2-7b-chat-hf")

    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument("--rank_pos", type=int, default=1024)
    parser.add_argument("--rank_neg", type=int, default=1024)
    parser.add_argument("--niter", type=int, default=20)
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration samples."
    )

    data_choices = [
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
    ]
    parser.add_argument(
        "--prune_data_pos", type=str, choices=data_choices, default="misalign"
    )
    parser.add_argument(
        "--prune_data_neg", type=str, choices=data_choices, default="align"
    )

    parser.add_argument(
        "--disentangle",
        action="store_true",
        help="whether to disentangle the prompt and response when computing the wanda score",
    )
    parser.add_argument("--save", type=str, default="out", help="Path to save results.")

    parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument("--eval_attack", action="store_true")

    args = parser.parse_args()
    setattr(args, "disentangle", True)
    setattr(args, "save_attack_res", True)
    setattr(args, "alpha", 1)

    sanity_check = True
    modeltype2path = {
        "llama2-7b-chat-hf": "",
        "llama2-13b-chat-hf": "",
        "llama2-7b-hf": "",
        "llama2-13b-hf": "",
    }

    def get_llm(model_name, cache_dir="llm_weights"):
        model = AutoModelForCausalLM.from_pretrained(
            modeltype2path[model_name],
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="cuda",
        )

        model.seqlen = model.config.max_position_embeddings
        return model

    if args.model == "llama2-7b-chat-hf":
        tokenizer = AutoTokenizer.from_pretrained(modeltype2path["llama2-7b-chat-hf"])
        model = get_llm("llama2-7b-chat-hf")
    elif args.model == "llama2-13b-chat-hf":
        tokenizer = AutoTokenizer.from_pretrained(modeltype2path["llama2-13b-chat-hf"])
        model = get_llm("llama2-13b-chat-hf")
    else:
        raise NotImplementedError

    # model_act.forward(input_ids = inputs['input_ids'])

    if False:
        model_act = make_Act(model, verbose=False)
        clear_act_buffer(model_act)
        print(tokenizer.batch_decode(model_act.generate(**inputs, max_length=200)))
        for n, m in model_act.named_modules():
            if isinstance(m, ActLinear):
                print(n, m.activation_norms)

    if False:
        model_act = make_Act(model, verbose=False)
        with no_act_recording(model_act):
            print(tokenizer.batch_decode(model_act.generate(**inputs, max_length=20)))
            for n, m in model_act.named_modules():
                if isinstance(m, ActLinear):
                    print(n, m.activation_norms)

    if True:
        # model_base = get_llm('llama2-7b-hf')
        device = "cuda"
        make_low_rank(
            args,
            model,
            tokenizer,
            device,
            prune_data_pos=args.prune_data_pos,
            prune_data_neg=args.prune_data_neg,
        )

        # evaluation begin

        if True:
            ppl_test = eval_ppl(args, model, tokenizer, device)
            print(f"wikitext perplexity {ppl_test}")

            if not os.path.exists(args.save):
                os.makedirs(args.save)
            save_filepath = os.path.join(args.save, f"log.txt")
            save_attackpath = os.path.join(
                args.save, f"attack_{args.rank_pos}_{args.rank_neg}"
            )
            if not os.path.exists(save_attackpath):
                os.makedirs(save_attackpath)
            if not os.path.exists(save_filepath):
                with open(save_filepath, "w") as f:
                    print("rank\tINST\tmetric\tscore", file=f, flush=True)
                    print(
                        f"{args.rank_pos}_{args.rank_neg}\t{args.alpha}\tPPL\t{ppl_test:.4f}",
                        file=f,
                        flush=True,
                    )
            else:
                with open(save_filepath, "a") as f:
                    print(
                        f"{args.rank_pos}_{args.rank_neg}\t{args.alpha}\tPPL\t{ppl_test:.4f}",
                        file=f,
                        flush=True,
                    )

        if args.eval_attack:

            if True:
                # note: since vLLM only supports loading from the path, we need to save the pruned model first for faster evaluation. We can reuse this temp folder to save disk spaces
                pruned_path = os.path.join("temp", f"_vllm_tmp")
                model.save_pretrained(pruned_path)
                vllm_model = LLM(
                    model=pruned_path,
                    tokenizer=modeltype2path[args.model],
                    dtype="bfloat16",
                    swap_space=128,
                )
                if True:
                    vllm_model.llm_engine.tokenizer.add_special_tokens(
                        {"pad_token": "[PAD]"}
                    )
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
                            f"{args.rank_pos}_{args.rank_neg}\t{suffix}\tASR_basic\t{score:.4f}",
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
                        filename=os.path.join(
                            save_attackpath, f"{suffix}basic_no_sys.jsonl"
                        ),
                    )
                    print(
                        f"attack evaluation results ({suffix}basic, no sys prompt): {score:.4f}"
                    )
                    with open(save_filepath, "a") as f:
                        print(
                            f"{args.rank_pos}_{args.rank_neg}\t{suffix}\tASR_basic_nosys\t{score:.4f}",
                            file=f,
                            flush=True,
                        )
                    # seems that llama2-13b may run into error on this :(
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
                            f"{args.rank_pos}_{args.rank_neg}\t{suffix}\tASR_multiple_nosys\t{score:.4f}",
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
                        f"{args.rank_pos}_{args.rank_neg}\t{suffix}\tASR_gcg\t{score:.4f}",
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
                "arc_easy",
                "arc_challenge",
                "openbookqa",
            ]
            # task_list = ["rte","hellaswag","arc_easy","arc_challenge", "openbookqa"]
            num_shot = 0
            results = eval_zero_shot(
                modeltype2path[args.model],
                model,
                tokenizer,
                task_list,
                num_shot,
                accelerate,
                limit=1000,
            )
            print("********************************")
            print("zero_shot evaluation results")
            sum_acc = 0
            with open(save_filepath, "a") as f:
                for k, v in results["results"].items():
                    print(
                        f"{args.rank_pos}_{args.rank_neg}\t{args.alpha}\t{k}\t{v['acc']:.4f}",
                        file=f,
                        flush=True,
                    )
                    sum_acc += v["acc"]
                print(
                    f"{args.rank_pos}_{args.rank_neg}\t{args.alpha}\taveraged\t{sum_acc/len(task_list):.4f}",
                    file=f,
                    flush=True,
                )

            print(results)
