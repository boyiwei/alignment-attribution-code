import os
import time
import heapq
import torch
import torch.nn as nn
import pickle
from .sparsegpt import SparseGPT
from .layerwrapper import WrappedGPT
from .data import get_loaders
import json
import random
from .ablate import AblateGPT
import heapq
import re


def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params


def check_sparsity_layerwise(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()
            print(f"{float((W==0).sum().item())/W.numel():.6f},")

    model.config.use_cache = use_cache


def prepare_calibration_input(model, dataloader, device, nsamples):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    # inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps = []
    tars = []
    attention_mask = []
    position_ids = []
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            attention_mask.append(kwargs["attention_mask"])
            position_ids.append(kwargs["position_ids"])
            # inps[cache['i']] = inp
            # cache['i'] += 1
            # cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            tars.append(batch[1])
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = [None for _ in range(nsamples)]
    model.config.use_cache = use_cache

    return inps, outs, tars, attention_mask, position_ids


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(
        sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1
    )
    W_mask = W_metric <= thres
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_random(
    args,
    model,
    tokenizer,
    model_base=None,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
):
    if args.use_diff or args.recover_from_base:
        assert model_base is not None
        layers_base = model_base.model.layers
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if args.use_diff or args.recover_from_base:
            subset_base = find_layers(layers_base[i])

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.randn_like(W)
            if prune_n != 0:
                W_mask = torch.zeros_like(W) == 1
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][
                    int(W.numel() * args.sparsity_ratio)
                ].cpu()
                W_mask = W_metric <= thresh

            if args.recover_from_base:
                assert model_base is not None
                subset[name].weight.data[W_mask] = subset_base[name].weight.data[
                    W_mask
                ]  # patch with the base model's weights
            else:
                subset[name].weight.data[W_mask] = 0  ## set weights to zero


def prune_magnitude(
    args,
    model,
    tokenizer,
    model_base=None,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
):
    if args.use_diff or args.recover_from_base:
        assert model_base is not None
        layers_base = model_base.model.layers
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if args.use_diff or args.recover_from_base:
            subset_base = find_layers(layers_base[i])

        for name in subset:
            W = subset[name].weight.data
            if args.use_diff or args.recover_from_base:
                W_base = subset_base[name].weight.data
                W_metric = torch.abs(W - W_base)
            else:
                W_metric = torch.abs(W)
            if args.neg_prune:
                W_metric = -W_metric
            if prune_n != 0:
                W_mask = torch.zeros_like(W) == 1
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][
                    int(W.numel() * args.sparsity_ratio)
                ].cpu()
                print(f"Layer: {name}    Threshold: {thresh}")
                print(W_metric.flatten().cpu().mean())
                if thresh == 0:
                    frac_zero = (W_metric == 0).sum().item() / W_metric.numel()
                    W_mask = (W_metric == 0) * (
                        torch.rand_like(W_metric) < (args.sparsity_ratio / frac_zero)
                    )
                else:
                    W_mask = W_metric <= thresh

            W[W_mask] = 0


def prune_wanda(
    args,
    model,
    tokenizer,
    model_base=None,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    prune_data="wikitext",
):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    print(f"loading calibration data {prune_data}")
    assert prune_data in [
        "wikitext",
        "alpaca",
        "alpaca_cleaned",
        "alpaca_cleaned_no_safety",
        "align",
        "align_short",
        "misalign",
    ]
    dataloader, _ = get_loaders(
        prune_data,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
    )
    # dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, tars, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, args.nsamples
        )

    if not args.disentangle:
        tars = [torch.zeros_like(tar) for tar in tars]  # remove -100's

    inps = [inp.squeeze(0).to(device) for inp in inps]
    tars = [tar.squeeze(0).to(device) for tar in tars]
    attention_mask = [am.to(device) for am in attention_mask]
    position_ids = [pids.to(device) for pids in position_ids]

    layers = model.model.layers
    if args.use_diff or args.recover_from_base:
        assert model_base is not None
        layers_base = model_base.model.layers

    if args.prune_part:
        print("only prune the layer with low jaccard index")
    else:
        print("prune every linear layer")

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if args.use_diff or args.recover_from_base:
            subset_base = find_layers(layers_base[i])

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, tars, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                tars.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )  # TODO

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name, tar):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data, tar)

            return tmp

        for j in range(args.nsamples):
            handles = []
            for name in wrapped_layers:
                handles.append(
                    subset[name].register_forward_hook(add_batch(name, tars[j]))
                )

            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0]

            for h in handles:
                h.remove()

        if not args.prune_part:
            for name in subset:
                print(f"pruning layer {i} name {name}")
                if args.use_diff or args.recover_from_base:
                    magnitude = torch.abs(
                        subset[name].weight.data - subset_base[name].weight.data
                    )
                else:
                    magnitude = torch.abs(subset[name].weight.data)
                act = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                W_metric = magnitude * act
                if args.neg_prune:
                    W_metric = -W_metric

                if args.dump_wanda_score:
                    # Only save the score, no pruning
                    if args.use_diff:
                        if args.disentangle:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_weight_diff_disentangle",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_diff_disentangle.pkl",
                            )
                        else:
                            save_folder = os.path.join(
                                args.save, f"wanda_score/{prune_data}_weight_diff"
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_diff.pkl",
                            )
                    else:
                        if args.disentangle:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_weight_only_disentangle",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_only_disentangle.pkl",
                            )
                        else:
                            save_folder = os.path.join(
                                args.save, f"wanda_score/{prune_data}_weight_only"
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_only.pkl",
                            )
                    with open(target_file, "wb") as f:
                        print(
                            "Writing W_metric in layer {} and name {} with {} to the file".format(
                                i, name, prune_data
                            )
                        )
                        pickle.dump(W_metric, f)
                    continue

                W_mask = (
                    torch.zeros_like(W_metric) == 1
                )  ## initialize a mask to be all False
                if prune_n != 0:
                    # structured n:m sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii : (ii + prune_m)].float()
                            W_mask.scatter_(
                                1,
                                ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                                True,
                            )
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                    if args.use_variant:
                        # wanda variant
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)

                        alpha = 0.4
                        alpha_hist = [0.0, 0.8]
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                        while (
                            torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001
                        ) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > args.sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha

                            alpha = alpha_new
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before
                            )
                        print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                    else:
                        # unstructured pruning
                        indices = sort_res[1][
                            :, : int(W_metric.shape[1] * args.sparsity_ratio)
                        ]
                        W_mask.scatter_(1, indices, True)

                if args.recover_from_base:
                    assert model_base is not None
                    subset[name].weight.data[W_mask] = subset_base[name].weight.data[
                        W_mask
                    ]  # patch with the base model's weights
                else:
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero
        else:
            # args.prune_part == True. We only prune the layer with low jaccard index, which is:
            # layer 0 mlp_down_proj
            # layer 1 self_attn._proj and mlp_down_proj
            # rest of layers: self_attn.o_proj, mlp_gate_proj, mlp_down_proj, mlp_up_proj
            for name in subset:
                condition = (
                    ((i == 0) and (name == "mlp.down_proj"))
                    or (
                        (i == 1)
                        and ((name == "self_attn.o_proj") or (name == "mlp.down_proj"))
                    )
                    or (
                        (i > 1)
                        and (
                            (name == "self_attn.o_proj")
                            or (name == "mlp.gate_proj")
                            or (name == "mlp.down_proj")
                            or (name == "mlp.up_proj")
                        )
                    )
                )
                if condition:
                    print(f"pruning layer {i} name {name}")
                    if args.use_diff or args.recover_from_base:
                        magnitude = torch.abs(
                            subset[name].weight.data - subset_base[name].weight.data
                        )
                    else:
                        magnitude = torch.abs(subset[name].weight.data)
                    act = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                    W_metric = magnitude * act
                    if args.neg_prune:
                        W_metric = -W_metric

                    if args.dump_wanda_score:
                        # Only save the score, no pruning
                        if args.use_diff:
                            if args.disentangle:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_weight_diff_disentangle",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_diff_disentangle.pkl",
                                )
                            else:
                                save_folder = os.path.join(
                                    args.save, f"wanda_score/{prune_data}_weight_diff"
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_diff.pkl",
                                )
                        else:
                            if args.disentangle:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_weight_only_disentangle",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_only_disentangle.pkl",
                                )
                            else:
                                save_folder = os.path.join(
                                    args.save, f"wanda_score/{prune_data}_weight_only"
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{prune_data}_weight_only.pkl",
                                )
                        with open(target_file, "wb") as f:
                            print(
                                "Writing W_metric in layer {} and name {} with {} to the file".format(
                                    i, name, prune_data
                                )
                            )
                            pickle.dump(W_metric, f)
                        continue

                    W_mask = (
                        torch.zeros_like(W_metric) == 1
                    )  ## initialize a mask to be all False
                    if prune_n != 0:
                        # structured n:m sparsity
                        for ii in range(W_metric.shape[1]):
                            if ii % prune_m == 0:
                                tmp = W_metric[:, ii : (ii + prune_m)].float()
                                W_mask.scatter_(
                                    1,
                                    ii
                                    + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                                    True,
                                )
                    else:
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)

                        if args.use_variant:
                            # wanda variant
                            tmp_metric = torch.cumsum(sort_res[0], dim=1)
                            sum_before = W_metric.sum(dim=1)

                            alpha = 0.4
                            alpha_hist = [0.0, 0.8]
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before
                            )
                            while (
                                torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001
                            ) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                                if cur_sparsity > args.sparsity_ratio:
                                    alpha_new = (alpha + alpha_hist[0]) / 2.0
                                    alpha_hist[1] = alpha
                                else:
                                    alpha_new = (alpha + alpha_hist[1]) / 2.0
                                    alpha_hist[0] = alpha

                                alpha = alpha_new
                                W_mask, cur_sparsity = return_given_alpha(
                                    alpha, sort_res, W_metric, tmp_metric, sum_before
                                )
                            print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                        else:
                            # unstructured pruning
                            indices = sort_res[1][
                                :, : int(W_metric.shape[1] * args.sparsity_ratio)
                            ]
                            W_mask.scatter_(1, indices, True)

                    if args.recover_from_base:
                        assert model_base is not None
                        subset[name].weight.data[W_mask] = subset_base[
                            name
                        ].weight.data[
                            W_mask
                        ]  # patch with the base model's weights
                    else:
                        subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0].squeeze(0)
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_wanda_decouple_activations(
    args,
    model,
    tokenizer,
    model_base=None,
    model_extra=None,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    prune_data="align_misalign",
):
    """
    Compute wanda score based on the difference between the align activation and misalign activation (In an online way, do not need to load wanda score from file)

    Compute the subtraction between align activation and misalign activation before computing the norm. Currently only support align activation minus misalign activation.

    Args:
        model_extra (_type_, optional): An extra model to compute utility activation. For the consideration of the interference in KV cache, currently we are using to models to compute the safety and utility activation respectively. Defaults to None.
    """
    assert prune_data in ["align_misalign", "align_short_misalign", "misalign_align"]
    use_cache = model.config.use_cache
    model.config.use_cache = False
    assert (
        args.decouple_align_misalign == True
    )  # Only support align activation minus misalign activation
    # load prune_data
    print(f"loading calibration data {prune_data}")
    dataloader, dataloader_extra = get_loaders(
        prune_data,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
    )
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, tars, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, args.nsamples
        )
    with torch.no_grad():
        inps_extra, outs_extra, tars_extra, attention_mask_extra, position_ids_extra = (
            prepare_calibration_input(
                model_extra, dataloader_extra, device, args.nsamples
            )
        )

    if not args.disentangle:
        tars = [torch.zeros_like(tar) for tar in tars]  # remove -100's
        tars_extra = [torch.zeros_like(tar) for tar in tars_extra]  # remove -100's
    inps = [inp.squeeze(0).to(device) for inp in inps]
    inps_extra = [inp.squeeze(0).to(device) for inp in inps_extra]
    tars = [tar.squeeze(0).to(device) for tar in tars]
    tars_extra = [tar.squeeze(0).to(device) for tar in tars_extra]
    attention_mask = [am.to(device) for am in attention_mask]
    attention_mask_extra = [am.to(device) for am in attention_mask_extra]
    position_ids = [pids.to(device) for pids in position_ids]
    position_ids_extra = [pids.to(device) for pids in position_ids_extra]

    layers = model.model.layers
    layers_extra = model_extra.model.layers
    if args.use_diff or args.recover_from_base:
        assert model_base is not None
        layers_base = model_base.model.layers

    if args.prune_part:
        print("only prune the layer with low jaccard index")
    else:
        print("prune every linear layer")

    for i in range(len(layers)):
        layer = layers[i]
        layer_extra = layers_extra[i]
        subset = find_layers(layer)
        subset_extra = find_layers(layer_extra)
        if args.use_diff or args.recover_from_base:
            subset_base = find_layers(layers_base[i])

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, tars, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                tars.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )
            (
                inps_extra,
                outs_extra,
                tars_extra,
                attention_mask_extra,
                position_ids_extra,
            ) = (
                inps_extra.to(dev),
                outs_extra.to(dev),
                tars_extra.to(dev),
                attention_mask_extra.to(dev),
                position_ids_extra.to(dev),
            )

        wrapped_layers = {}
        wrapped_layers_extra = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
            wrapped_layers_extra[name] = WrappedGPT(subset_extra[name])

        # compute safety activation
        def add_batch(name, tar):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data, tar)

            return tmp

        for j in range(args.nsamples):
            handles = []
            for name in wrapped_layers:
                handles.append(
                    subset[name].register_forward_hook(add_batch(name, tars[j]))
                )

            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0]

            for h in handles:
                h.remove()

        # compute utility activation
        def add_batch_extra(name, tar):
            def tmp(_, inp, out):
                wrapped_layers_extra[name].add_batch(inp[0].data, out.data, tar)

            return tmp

        for j in range(args.nsamples):
            handles = []
            for name in wrapped_layers_extra:
                handles.append(
                    subset_extra[name].register_forward_hook(
                        add_batch_extra(name, tars_extra[j])
                    )
                )

            with torch.no_grad():
                outs_extra[j] = layer_extra(
                    inps_extra[j].unsqueeze(0),
                    attention_mask=attention_mask_extra[j],
                    position_ids=position_ids_extra[j],
                )[0]

            for h in handles:
                h.remove()

        if not args.prune_part:
            for name in subset:
                print(f"pruning layer {i} name {name}")
                if args.use_diff or args.recover_from_base:
                    magnitude = torch.abs(
                        subset[name].weight.data - subset_base[name].weight.data
                    )
                else:
                    magnitude = torch.abs(subset[name].weight.data)
                # act = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))) - torch.sqrt(wrapped_layers_extra[name].scaler_row.reshape((1,-1)))
                act1 = wrapped_layers[name].activations
                act2 = wrapped_layers_extra[name].activations
                if (
                    prune_data == "align_misalign"
                    or prune_data == "align_short_misalign"
                ):
                    act = [a1 - a2 for a1, a2 in zip(act1, act2)]
                elif prune_data == "misalign_align":
                    act = [a2 - a1 for a1, a2 in zip(act1, act2)]
                act_norms = [torch.norm(a, p=2, dim=1) ** 2 for a in act]
                act_norms_average = sum(act_norms) / len(act_norms)
                act_norms_average = torch.sqrt(act_norms_average.reshape(1, -1))
                W_metric = magnitude * act_norms_average
                if args.neg_prune:
                    W_metric = -W_metric

                if args.dump_wanda_score:
                    if args.use_diff:
                        if args.disentangle:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_utility_decouple_weight_diff_disentangle",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_diff_disentangle.pkl",
                            )
                        else:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_utility_decouple_weight_diff",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_diff.pkl",
                            )
                    else:
                        if args.disentangle:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_utility_decouple_weight_only_disentangle",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_only_disentangle.pkl",
                            )
                        else:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_utility_decouple_weight_only",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_only.pkl",
                            )
                    with open(target_file, "wb") as f:
                        print(
                            "Writing W_metric in layer {} and name {} with {} to the file".format(
                                i, name, prune_data
                            )
                        )
                        pickle.dump(W_metric, f)
                    continue

                W_mask = (
                    torch.zeros_like(W_metric) == 1
                )  ## initialize a mask to be all False
                if prune_n != 0:
                    # structured n:m sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii : (ii + prune_m)].float()
                            W_mask.scatter_(
                                1,
                                ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                                True,
                            )
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                    if args.use_variant:
                        # wanda variant
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)

                        alpha = 0.4
                        alpha_hist = [0.0, 0.8]
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                        while (
                            torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001
                        ) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > args.sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha

                            alpha = alpha_new
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before
                            )
                        print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                    else:
                        # unstructured pruning
                        indices = sort_res[1][
                            :, : int(W_metric.shape[1] * args.sparsity_ratio)
                        ]
                        W_mask.scatter_(1, indices, True)

                if args.recover_from_base:
                    assert model_base is not None
                    subset[name].weight.data[W_mask] = subset_base[name].weight.data[
                        W_mask
                    ]  # patch with the base model's weights
                else:
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero
        else:
            # args.prune_part == True. We only prune the layer with low jaccard index, which is:
            # layer 0 mlp_down_proj
            # layer 1 self_attn._proj and mlp_down_proj
            # rest of layers: self_attn.o_proj, mlp_gate_proj, mlp_down_proj, mlp_up_proj
            for name in subset:
                condition = (
                    ((i == 0) and (name == "mlp.down_proj"))
                    or (
                        (i == 1)
                        and ((name == "self_attn.o_proj") or (name == "mlp.down_proj"))
                    )
                    or (
                        (i > 1)
                        and (
                            (name == "self_attn.o_proj")
                            or (name == "mlp.gate_proj")
                            or (name == "mlp.down_proj")
                            or (name == "mlp.up_proj")
                        )
                    )
                )
                if condition:
                    print(f"pruning layer {i} name {name}")
                    if args.use_diff or args.recover_from_base:
                        magnitude = torch.abs(
                            subset[name].weight.data - subset_base[name].weight.data
                        )
                    else:
                        magnitude = torch.abs(subset[name].weight.data)
                    # act = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))) - torch.sqrt(wrapped_layers_extra[name].scaler_row.reshape((1,-1)))
                    act1 = wrapped_layers[name].activations
                    act2 = wrapped_layers_extra[name].activations
                    if (
                        prune_data == "align_misalign"
                        or prune_data == "align_short_misalign"
                    ):
                        act = [a1 - a2 for a1, a2 in zip(act1, act2)]
                    elif prune_data == "misalign_align":
                        act = [a2 - a1 for a1, a2 in zip(act1, act2)]
                    act_norms = [torch.norm(a, p=2, dim=1) ** 2 for a in act]
                    act_norms_average = sum(act_norms) / len(act_norms)
                    act_norms_average = torch.sqrt(act_norms_average.reshape(1, -1))
                    W_metric = magnitude * act_norms_average
                    if args.neg_prune:
                        W_metric = -W_metric

                    if args.dump_wanda_score:
                        # Only save the score, no pruning
                        save_folder = os.path.join(
                            args.save, f"wanda_score/{prune_data}__online"
                        )
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)
                        if args.use_diff:
                            if args.disentangle:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_utility_decouple_weight_diff_disentangle",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_diff_disentangle.pkl",
                                )
                            else:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_utility_decouple_weight_diff",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_diff.pkl",
                                )
                        else:
                            if args.disentangle:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_utility_decouple_weight_only_disentangle",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_only_disentangle.pkl",
                                )
                            else:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_utility_decouple_weight_only",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_only.pkl",
                                )
                        with open(target_file, "wb") as f:
                            print(
                                "Writing W_metric in layer {} and name {} with {} to the file".format(
                                    i, name, prune_data
                                )
                            )
                            pickle.dump(W_metric, f)
                        continue

                    W_mask = (
                        torch.zeros_like(W_metric) == 1
                    )  ## initialize a mask to be all False
                    if prune_n != 0:
                        # structured n:m sparsity
                        for ii in range(W_metric.shape[1]):
                            if ii % prune_m == 0:
                                tmp = W_metric[:, ii : (ii + prune_m)].float()
                                W_mask.scatter_(
                                    1,
                                    ii
                                    + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                                    True,
                                )
                    else:
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)

                        if args.use_variant:
                            # wanda variant
                            tmp_metric = torch.cumsum(sort_res[0], dim=1)
                            sum_before = W_metric.sum(dim=1)

                            alpha = 0.4
                            alpha_hist = [0.0, 0.8]
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before
                            )
                            while (
                                torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001
                            ) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                                if cur_sparsity > args.sparsity_ratio:
                                    alpha_new = (alpha + alpha_hist[0]) / 2.0
                                    alpha_hist[1] = alpha
                                else:
                                    alpha_new = (alpha + alpha_hist[1]) / 2.0
                                    alpha_hist[0] = alpha

                                alpha = alpha_new
                                W_mask, cur_sparsity = return_given_alpha(
                                    alpha, sort_res, W_metric, tmp_metric, sum_before
                                )
                            print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                        else:
                            # unstructured pruning
                            indices = sort_res[1][
                                :, : int(W_metric.shape[1] * args.sparsity_ratio)
                            ]
                            W_mask.scatter_(1, indices, True)

                    if args.recover_from_base:
                        assert model_base is not None
                        subset[name].weight.data[W_mask] = subset_base[
                            name
                        ].weight.data[
                            W_mask
                        ]  # patch with the base model's weights
                    else:
                        subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0].squeeze(0)
            with torch.no_grad():
                outs_extra[j] = layer_extra(
                    inps_extra[j].unsqueeze(0),
                    attention_mask=attention_mask_extra[j],
                    position_ids=position_ids_extra[j],
                )[0].squeeze(0)
        inps, outs = outs, inps
        inps_extra, outs_extra = outs_extra, inps_extra

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_wanda_decouple_activation_norms(
    args,
    model,
    tokenizer,
    model_base=None,
    model_extra=None,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    prune_data="align",
):
    """
    Compute wanda score based on the difference between tow activation norms (In an online way, do not need to load wanda score from file)

    Compute the norms first then compute the difference

    Args:
        model_extra (_type_, optional): An extra model to compute utility activation. For the consideration of the interference in KV cache, currently we are using to models to compute the safety and utility activation respectively. Defaults to None.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if args.decouple_align_utility:
        prune_data_extra = "alpaca_cleaned_no_safety"
    elif args.decouple_align_misalign:
        prune_data_extra = "misalign"
    else:
        raise NotImplementedError
    # load prune_data
    print(f"loading calibration data {prune_data}")
    dataloader, _ = get_loaders(
        prune_data,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
    )
    print("dataset loading complete")
    print(f"loading extra calibration data {prune_data_extra}")
    dataloader_extra, _ = get_loaders(
        prune_data_extra,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model_extra.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
    )
    print("extra dataset loading complete")
    with torch.no_grad():
        inps, outs, tars, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, args.nsamples
        )
    with torch.no_grad():
        inps_extra, outs_extra, tars_extra, attention_mask_extra, position_ids_extra = (
            prepare_calibration_input(
                model_extra, dataloader_extra, device, args.nsamples
            )
        )

    if not args.disentangle:
        tars = [torch.zeros_like(tar) for tar in tars]  # remove -100's
        tars_extra = [torch.zeros_like(tar) for tar in tars_extra]  # remove -100's
    inps = [inp.squeeze(0).to(device) for inp in inps]
    inps_extra = [inp.squeeze(0).to(device) for inp in inps_extra]
    tars = [tar.squeeze(0).to(device) for tar in tars]
    tars_extra = [tar.squeeze(0).to(device) for tar in tars_extra]
    attention_mask = [am.to(device) for am in attention_mask]
    attention_mask_extra = [am.to(device) for am in attention_mask_extra]
    position_ids = [pids.to(device) for pids in position_ids]
    position_ids_extra = [pids.to(device) for pids in position_ids_extra]

    layers = model.model.layers
    layers_extra = model_extra.model.layers
    if args.use_diff or args.recover_from_base:
        assert model_base is not None
        layers_base = model_base.model.layers

    if args.prune_part:
        print("only prune the layer with low jaccard index")
    else:
        print("prune every linear layer")

    for i in range(len(layers)):
        layer = layers[i]
        layer_extra = layers_extra[i]
        subset = find_layers(layer)
        subset_extra = find_layers(layer_extra)
        if args.use_diff or args.recover_from_base:
            subset_base = find_layers(layers_base[i])

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, tars, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                tars.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )
            (
                inps_extra,
                outs_extra,
                tars_extra,
                attention_mask_extra,
                position_ids_extra,
            ) = (
                inps_extra.to(dev),
                outs_extra.to(dev),
                tars_extra.to(dev),
                attention_mask_extra.to(dev),
                position_ids_extra.to(dev),
            )

        wrapped_layers = {}
        wrapped_layers_extra = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
            wrapped_layers_extra[name] = WrappedGPT(subset_extra[name])

        # compute safety activation
        def add_batch(name, tar):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data, tar)

            return tmp

        for j in range(args.nsamples):
            handles = []
            for name in wrapped_layers:
                handles.append(
                    subset[name].register_forward_hook(add_batch(name, tars[j]))
                )

            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0]

            for h in handles:
                h.remove()

        # compute utility activation
        def add_batch_extra(name, tar):
            def tmp(_, inp, out):
                wrapped_layers_extra[name].add_batch(inp[0].data, out.data, tar)

            return tmp

        for j in range(args.nsamples):
            handles = []
            for name in wrapped_layers_extra:
                handles.append(
                    subset_extra[name].register_forward_hook(
                        add_batch_extra(name, tars_extra[j])
                    )
                )

            with torch.no_grad():
                outs_extra[j] = layer_extra(
                    inps_extra[j].unsqueeze(0),
                    attention_mask=attention_mask_extra[j],
                    position_ids=position_ids_extra[j],
                )[0]

            for h in handles:
                h.remove()

        if not args.prune_part:
            for name in subset:
                print(f"pruning layer {i} name {name}")
                if args.use_diff or args.recover_from_base:
                    magnitude = torch.abs(
                        subset[name].weight.data - subset_base[name].weight.data
                    )
                else:
                    magnitude = torch.abs(subset[name].weight.data)
                # act = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))) - torch.sqrt(wrapped_layers_extra[name].scaler_row.reshape((1,-1)))
                act1 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                act2 = torch.sqrt(
                    wrapped_layers_extra[name].scaler_row.reshape((1, -1))
                )
                scale = torch.max(torch.sum(act1), torch.sum(act2))
                act1_norm = act1 / torch.sum(act1) * scale
                act2_norm = act2 / torch.sum(act2) * scale
                act = act1_norm - act2_norm
                W_metric = magnitude * act
                if args.neg_prune:
                    W_metric = -W_metric

                if args.dump_wanda_score:
                    if args.use_diff:
                        if args.disentangle:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_utility_decouple_weight_diff_disentangle",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_diff_disentangle.pkl",
                            )
                        else:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_utility_decouple_weight_diff",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_diff.pkl",
                            )
                    else:
                        if args.disentangle:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_utility_decouple_weight_only_disentangle",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_only_disentangle.pkl",
                            )
                        else:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_utility_decouple_weight_only",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_only.pkl",
                            )
                    with open(target_file, "wb") as f:
                        print(
                            "Writing W_metric in layer {} and name {} with {} to the file".format(
                                i, name, prune_data
                            )
                        )
                        pickle.dump(W_metric, f)
                    continue

                W_mask = (
                    torch.zeros_like(W_metric) == 1
                )  ## initialize a mask to be all False
                if prune_n != 0:
                    # structured n:m sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii : (ii + prune_m)].float()
                            W_mask.scatter_(
                                1,
                                ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                                True,
                            )
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                    if args.use_variant:
                        # wanda variant
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)

                        alpha = 0.4
                        alpha_hist = [0.0, 0.8]
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                        while (
                            torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001
                        ) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > args.sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha

                            alpha = alpha_new
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before
                            )
                        print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                    else:
                        # unstructured pruning
                        indices = sort_res[1][
                            :, : int(W_metric.shape[1] * args.sparsity_ratio)
                        ]
                        W_mask.scatter_(1, indices, True)

                if args.recover_from_base:
                    assert model_base is not None
                    subset[name].weight.data[W_mask] = subset_base[name].weight.data[
                        W_mask
                    ]  # patch with the base model's weights
                else:
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero
        else:
            # args.prune_part == True. We only prune the layer with low jaccard index, which is:
            # layer 0 mlp_down_proj
            # layer 1 self_attn._proj and mlp_down_proj
            # rest of layers: self_attn.o_proj, mlp_gate_proj, mlp_down_proj, mlp_up_proj
            for name in subset:
                condition = (
                    ((i == 0) and (name == "mlp.down_proj"))
                    or (
                        (i == 1)
                        and ((name == "self_attn.o_proj") or (name == "mlp.down_proj"))
                    )
                    or (
                        (i > 1)
                        and (
                            (name == "self_attn.o_proj")
                            or (name == "mlp.gate_proj")
                            or (name == "mlp.down_proj")
                            or (name == "mlp.up_proj")
                        )
                    )
                )
                if condition:
                    print(f"pruning layer {i} name {name}")
                    if args.use_diff or args.recover_from_base:
                        magnitude = torch.abs(
                            subset[name].weight.data - subset_base[name].weight.data
                        )
                    else:
                        magnitude = torch.abs(subset[name].weight.data)
                    # act = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))) - torch.sqrt(wrapped_layers_extra[name].scaler_row.reshape((1,-1)))
                    act1 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                    act2 = torch.sqrt(
                        wrapped_layers_extra[name].scaler_row.reshape((1, -1))
                    )
                    scale = torch.max(torch.sum(act1), torch.sum(act2))
                    act1_norm = act1 / torch.sum(act1) * scale
                    act2_norm = act2 / torch.sum(act2) * scale
                    W_metric = magnitude * act
                    if args.neg_prune:
                        W_metric = -W_metric

                    if args.dump_wanda_score:
                        # Only save the score, no pruning
                        save_folder = os.path.join(
                            args.save, f"wanda_score/{prune_data}__online"
                        )
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)
                        if args.use_diff:
                            if args.disentangle:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_utility_decouple_weight_diff_disentangle",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_diff_disentangle.pkl",
                                )
                            else:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_utility_decouple_weight_diff",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_diff.pkl",
                                )
                        else:
                            if args.disentangle:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_utility_decouple_weight_only_disentangle",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_only_disentangle.pkl",
                                )
                            else:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_utility_decouple_weight_only",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_only.pkl",
                                )
                        with open(target_file, "wb") as f:
                            print(
                                "Writing W_metric in layer {} and name {} with {} to the file".format(
                                    i, name, prune_data
                                )
                            )
                            pickle.dump(W_metric, f)
                        continue

                    W_mask = (
                        torch.zeros_like(W_metric) == 1
                    )  ## initialize a mask to be all False
                    if prune_n != 0:
                        # structured n:m sparsity
                        for ii in range(W_metric.shape[1]):
                            if ii % prune_m == 0:
                                tmp = W_metric[:, ii : (ii + prune_m)].float()
                                W_mask.scatter_(
                                    1,
                                    ii
                                    + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                                    True,
                                )
                    else:
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)

                        if args.use_variant:
                            # wanda variant
                            tmp_metric = torch.cumsum(sort_res[0], dim=1)
                            sum_before = W_metric.sum(dim=1)

                            alpha = 0.4
                            alpha_hist = [0.0, 0.8]
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before
                            )
                            while (
                                torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001
                            ) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                                if cur_sparsity > args.sparsity_ratio:
                                    alpha_new = (alpha + alpha_hist[0]) / 2.0
                                    alpha_hist[1] = alpha
                                else:
                                    alpha_new = (alpha + alpha_hist[1]) / 2.0
                                    alpha_hist[0] = alpha

                                alpha = alpha_new
                                W_mask, cur_sparsity = return_given_alpha(
                                    alpha, sort_res, W_metric, tmp_metric, sum_before
                                )
                            print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                        else:
                            # unstructured pruning
                            indices = sort_res[1][
                                :, : int(W_metric.shape[1] * args.sparsity_ratio)
                            ]
                            W_mask.scatter_(1, indices, True)

                    if args.recover_from_base:
                        assert model_base is not None
                        subset[name].weight.data[W_mask] = subset_base[
                            name
                        ].weight.data[
                            W_mask
                        ]  # patch with the base model's weights
                    else:
                        subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0].squeeze(0)
            with torch.no_grad():
                outs_extra[j] = layer_extra(
                    inps_extra[j].unsqueeze(0),
                    attention_mask=attention_mask_extra[j],
                    position_ids=position_ids_extra[j],
                )[0].squeeze(0)
        inps, outs = outs, inps
        inps_extra, outs_extra = outs_extra, inps_extra

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_wandg_set_difference(
    args,
    model,
    tokenizer,
    model_base=None,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    prune_data="align_short",
    p=0.5,
    q=0.5,
):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    if args.use_diff or args.recover_from_base:
        assert model_base is not None
        layers_base = model_base.model.layers
    metric1 = "alpaca_cleaned_no_safety"
    metric2 = prune_data

    print(
        "prune p = {}, q = {}, with metric1 = {}, metric2 = {}".format(
            p, q, metric1, metric2
        )
    )
    if args.prune_part:
        print("only prune the layer with low jaccard index")
    else:
        print("prune every linear layer")
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if args.use_diff or args.recover_from_base:
            subset_base = find_layers(layers_base[i])

        if not args.prune_part:
            for name in subset:
                print(f"pruning layer {i} name {name}")
                if args.model == "llama2-7b-chat-hf":
                    W_metric1 = pickle.load(
                        open(
                            f"out/llama2-7b-chat-hf/unstructured/wandg/{metric1}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                            "rb",
                        )
                    )
                    W_metric2 = pickle.load(
                        open(
                            f"out/llama2-7b-chat-hf/unstructured/wandg/{metric2}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                            "rb",
                        )
                    )
                elif args.model == "llama2-13b-chat-hf":
                    W_metric1 = pickle.load(
                        open(
                            f"out/llama2-13b-chat-hf/unstructured/wandg/{metric1}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                            "rb",
                        )
                    )
                    W_metric2 = pickle.load(
                        open(
                            f"out/llama2-13b-chat-hf/unstructured/wandg/{metric2}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                            "rb",
                        )
                    )
                else:
                    raise NotImplementedError

                top_p = int(
                    p * W_metric1.shape[1] * W_metric1.shape[0]
                )  # top_p utility
                top_q = int(q * W_metric2.shape[1] * W_metric2.shape[0])  # top_q safety

                top_p_indices = torch.topk(W_metric1.flatten(), top_p, largest=True)[1]
                top_q_indices = torch.topk(W_metric2.flatten(), top_q, largest=True)[1]
                unique_p = torch.unique(top_p_indices)
                unique_q = torch.unique(top_q_indices)

                # Create a boolean mask for elements in unique_q that are not in unique_p
                mask = ~torch.isin(unique_q, unique_p)

                # Apply the mask to unique_q to get filtered_indices
                filtered_indices = unique_q[mask]
                weight_dim = subset[name].weight.data.shape[1]
                filtered_indices_rows = filtered_indices // weight_dim
                filtered_indices_cols = filtered_indices % weight_dim

                assert (
                    args.dump_wanda_score == False
                )  # Only pruning from the saved score, won't save score again

                W_mask = torch.zeros_like(subset[name].weight.data) == 1
                W_mask[filtered_indices_rows, filtered_indices_cols] = (
                    True  # prune weights that has relatively high safety while not in top utility scores
                )

                if args.recover_from_base:
                    assert model_base is not None
                    subset[name].weight.data[W_mask] = subset_base[name].weight.data[
                        W_mask
                    ]  # patch with the base model's weights
                else:
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero
        else:
            # args.prune_part == True. We only prune the layer with low jaccard index, which is:
            # layer 0 mlp_down_proj
            # layer 1 self_attn._proj and mlp_down_proj
            # rest of layers: self_attn.o_proj, mlp_gate_proj, mlp_down_proj, mlp_up_proj
            for name in subset:
                condition = (
                    ((i == 0) and (name == "mlp.down_proj"))
                    or (
                        (i == 1)
                        and ((name == "self_attn.o_proj") or (name == "mlp.down_proj"))
                    )
                    or (
                        (i > 1)
                        and (
                            (name == "self_attn.o_proj")
                            or (name == "mlp.gate_proj")
                            or (name == "mlp.down_proj")
                            or (name == "mlp.up_proj")
                        )
                    )
                )
                if condition:
                    W_metric1 = pickle.load(
                        open(
                            f"out/llama2-7b-chat-hf/unstructured/wandg/{metric1}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                            "rb",
                        )
                    )
                    W_metric2 = pickle.load(
                        open(
                            f"out/llama2-7b-chat-hf/unstructured/wandg/{metric2}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                            "rb",
                        )
                    )
                    top_p = int(p * W_metric1.shape[1] * W_metric1.shape[0])
                    top_q = int(q * W_metric2.shape[1] * W_metric2.shape[0])

                    top_p_indices = torch.topk(
                        W_metric1.flatten(), top_p, largest=True
                    )[1]
                    top_q_indices = torch.topk(
                        W_metric2.flatten(), top_q, largest=True
                    )[1]
                    unique_p = torch.unique(top_p_indices)
                    unique_q = torch.unique(top_q_indices)

                    # Create a boolean mask for elements in unique_p that are not in unique_q
                    mask = ~torch.isin(unique_q, unique_p)

                    # Apply the mask to unique_p to get filtered_indices
                    filtered_indices = unique_q[mask]
                    weight_dim = subset[name].weight.data.shape[1]
                    filtered_indices_rows = filtered_indices // weight_dim
                    filtered_indices_cols = filtered_indices % weight_dim

                    assert (
                        args.dump_wanda_score == False
                    )  # Only pruning from the saved score, won't save score again

                    W_mask = torch.zeros_like(subset[name].weight.data) == 1
                    W_mask[filtered_indices_rows, filtered_indices_cols] = (
                        True  # prune weights that has relatively high safety while not in top utility scores
                    )

                    if args.recover_from_base:
                        assert model_base is not None
                        subset[name].weight.data[W_mask] = subset_base[
                            name
                        ].weight.data[
                            W_mask
                        ]  # patch with the base model's weights
                    else:
                        subset[name].weight.data[W_mask] = 0  ## set weights to zero


def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print("Starting ...")
    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    # dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print("Pruning ...")

            gpts[name].fasterprune(
                args.sparsity_ratio,
                prune_n=prune_n,
                prune_m=prune_m,
                percdamp=0.01,
                blocksize=128,
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print("Starting ...")
    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    # dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print("Pruning ...")

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(
                    args.sparsity_ratio, prune_n, prune_m
                )
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(
                    args.sparsity_ratio, prune_n, prune_m
                )
            elif "iter" in args.prune_method:
                prune_mask = None

            gpts[name].fasterprune(
                args,
                args.sparsity_ratio,
                mask=prune_mask,
                prune_n=prune_n,
                prune_m=prune_m,
                percdamp=0.01,
                blocksize=128,
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def get_mask(model, neg_prune=False):
    """
    Save mask for the unstructured pruned model (for ft-attack evaluation).
    `neg_prune`:
        - if `args.neg_prune` is False (bottom pruning), save the mask as True for the weights not pruned.
        - if `args.neg_prune` is True (top pruning), save the mask as True for the pruned weights.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    mask = {}

    mask_num = 0
    total_num = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            mask[name] = module.weight.data.abs().lt(1e-8).to("cpu").detach()
            if neg_prune is False:
                mask[name] = ~mask[name]

            mask_num += mask[name].eq(True).int().sum()
            total_num += mask[name].numel()

    print(f"{(100 * mask_num / total_num):.2f}% entries are True in mask.")
    return mask


def prune_attention_head(
    args, model, model_base=None, device=torch.device("cuda:0"), top_k_heads=10
):
    """Prune the attention_heads based on the probing results. Still not supporting reover from base. Only support Llama-2-7b-chat-hf

    Args:
        args (_type_): _description_
        model (_type_): _description_
        model_base (_type_, optional): _description_. Defaults to None.
        device (_type_, optional): _description_. Defaults to torch.device("cuda:0").

    Raises:
        ValueError: _description_
    """

    layers = model.model.layers
    k = top_k_heads
    print("Pruning top {} attention heads".format(k))

    # find the top-k attention heads in probing results based on the value in the probing_result
    if args.model == "llama2-7b-chat-hf":
        with open("data/probing_result_7b.json", "r") as f:
            # read json file to dict
            probing_result = json.load(f)
        count = sum(value == 1.0 for value in probing_result.values())
        if k <= count:
            top_k_heads_full = heapq.nlargest(
                132, probing_result, key=probing_result.get
            )
            top_k_heads = random.sample(top_k_heads_full, k)
        elif k <= len(probing_result):
            top_k_heads = heapq.nlargest(k, probing_result, key=probing_result.get)
        else:
            raise ValueError("k is larger than the number of attention heads")

        extracted_numbers = [
            list(map(int, re.findall(r"\d+", head))) for head in top_k_heads
        ]

        for head in extracted_numbers:
            block_id = head[0]
            head_id = head[1]
            layer = layers[block_id]
            subset = find_layers(layer)
            for name in ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"]:
                W = subset[name].weight.data
                W_metric = torch.zeros_like(W)
                W_metric[:, head_id * 128 : (head_id + 1) * 128] = 1
                W_mask = W_metric == 1
                subset[name].weight.data[W_mask] = 0
            name = "self_attn.o_proj"
            W = subset[name].weight.data
            W_metric = torch.zeros_like(W)
            W_metric[head_id * 128 : (head_id + 1) * 128, :] = 1
            W_mask = W_metric == 1
            subset[name].weight.data[W_mask] = 0
    elif args.model == "llama2-13b-chat-hf":
        with open("data/probing_result_13b.json", "r") as f:
            # read json file to dict
            probing_result = json.load(f)
        count = sum(value == 1.0 for value in probing_result.values())
        if k <= count:
            top_k_heads_full = heapq.nlargest(
                count, probing_result, key=probing_result.get
            )
            top_k_heads = random.sample(top_k_heads_full, k)
        elif k <= len(probing_result):
            top_k_heads = heapq.nlargest(k, probing_result, key=probing_result.get)
        else:
            raise ValueError("k is larger than the number of attention heads")

        extracted_numbers = [
            list(map(int, re.findall(r"\d+", head))) for head in top_k_heads
        ]

        for head in extracted_numbers:
            block_id = head[0]
            head_id = head[1]
            layer = layers[block_id]
            subset = find_layers(layer)
            for name in ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"]:
                W = subset[name].weight.data
                head_dim = W.shape[1] // 40
                W_metric = torch.zeros_like(W)
                W_metric[:, head_id * head_dim : (head_id + 1) * head_dim] = 1
                W_mask = W_metric == 1
                subset[name].weight.data[W_mask] = 0
            name = "self_attn.o_proj"
            W = subset[name].weight.data
            W_metric = torch.zeros_like(W)
            W_metric[head_id * head_dim : (head_id + 1) * head_dim, :] = 1
            W_mask = W_metric == 1
            subset[name].weight.data[W_mask] = 0
