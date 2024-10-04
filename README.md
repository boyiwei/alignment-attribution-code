# Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications

This repository provides an original implementation of [*Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications*](https://arxiv.org/abs/2402.05162) by Boyi Wei*, Kaixuan Huang*, Yangsibo Huang*, Tinghao Xie, Xiangyu Qi, Mengzhou Xia, Prateek Mittal, Mengdi Wang and Peter Henderson.

## 1. Setup

You can use the following instruction to create conda environment
```bash
conda env create -f environment.yml
```
Please notice that you need to specify your environment path inside ``environment.yml``

Besides, you need to manually install a hacked version of lm_eval to support evaluating the pruned model. See [wanda](https://github.com/locuslab/wanda?tab=readme-ov-file#zero-shot-evaluation).

There are known [issues](https://github.com/huggingface/transformers/issues/22222) with the transformers library on loading the LLaMA tokenizer correctly. Please follow the mentioned suggestions to resolve this issue.


Before running experiments, make sure you have specified the path pointing to the model stored in your locations.

## 2. Neuron Level Usage

### 2.1 Top-down Pruning

The main function is ``main.py``. When using Top-down pruning, we need to add ``--neg_prune`` in the command line.

Important parameters are:
1. ``--prune_method``: To specify the prune method. Available options are ``wanda``, ``wandg`` (SNIP in the paper), ``random``.
2. ``--prune_data``: To specify datasets used for pruning. When doing top-down pruning safety-critical neurons, we can use ``align``(safety-full in the paper) and ``align_short`` (safety-short in the paper) as our dataset.
3. ``--sparsity_ratio``: Specify the prune sparsity.
4. ``--eval_zero_shot``: Whether to evaluate the model's zero-shot-accuracy after pruning
5. ``--eval_attack`` : Whether to evaluate the model's ASR after pruning.
6. ``--save``: Specify the save location
7. ``--model``: Specify the model. Currently we only support ``llama2-7b-chat-hf`` and ``llama2-13b-chat-hf``


Example: Using ``llama2-7b-chat-hf`` to prune 0.5 part of weights, using safety-full dataset.
```bash
model="llama2-7b-chat-hf"
method="wanda"
type="unstructured"
suffix="weightonly"
save_dir="out/$model/$type/${method}_${suffix}/align/"

python main.py \
    --model $model \
    --prune_method $method \
    --prune_data align \
    --sparsity_ratio 0.5 \
    --sparsity_type $type \
    --neg_prune
    --save $save_dir \
    --eval_zero_shot \
    --eval_attack \
    --save_attack_res
```
### 2.2 Pruning the least safety-critical neurons.

Simply remove ``--neg_prune`` will reverse the order of pruning. We recommend using ``align_short`` (safety-short in our paper) when pruning the least safety-critical neurons to get more obvious results.


### 2.2 Pruning based on Set Difference


Select option ``--prune_method`` as ``wandg_set_difference`` (SNIP with set difference in our paper). Add option ``--p``, which corresponds to top-p scored entries in alpaca_no_safety-based wandg score; Add option ``--q``, which corresponds to top-q scored entries in aligned-based wandg score. Please notice that you have to specify a non-zero value of ``--sparsity_ratio``. For the dataset to compute the utility importance score, we use``alpaca_cleaned_no_safety `` by default, ``--prune_data`` here is used to specify the dataset to compute the safety importance score. Available options are ``align`` (safety-full in our paper) and ``align_short`` (safety-short in our paper)

Example: Pruning the set difference between top-10% utility-critical neurons (Use alpaca_cleaned_no_safety dataset to identify) and top-10% safety-critical (Use safety-full to identify) safety neurons. 

```bash
model="llama2-7b-chat-hf"
method="wandg_set_difference"
type="unstructured"
suffix="weightonly"
save_dir="out/$model/$type/wandg_set_difference_{$suffix}"

python main.py \
    --model $model \
    --prune_method $method \
    --sparsity_ratio 0.5 \
    --prune_data align
    --p 0.1\
    --q 0.1\
    --sparsity_type $type \
    --save $save_dir \
    --eval_zero_shot \
    --eval_attack \
    --save_attack_res
```


### 2.3 Wanda/SNIP Score dumping

Simply add option `--dump_wanda_score` into the command.

Example: Safety-first pruning with align_llama2-7b-chat dataset:

```bash
model="llama2-7b-chat-hf"
method="wanda"
type="unstructured"
suffix="weightonly"
save_dir="out/$model/$type/${method}_${suffix}/align/"

python main.py \
    --model $model \
    --prune_method $method \
    --prune_data align \
    --sparsity_ratio 0.5 \
    --sparsity_type $type \
    --save $save_dir \
    --dump_wanda_score
```

## 3. Rank Level Usage

### 3.1 Remove the most safety-critical rank

The main function of this pipeline is ``main_low_rank.py``. Most of the parameters are similar to the prune neurons situation. 

Important parameters are:

1. ``--prune_method``: To specify the pruning method, in this case we choose ``low_rank``, which corresponds to ActSVD in our paper.
2. ``--prune_data``: To specify the dataset used to identify the safety/utility projection matrix. Available options are ``align`` (safety-full), ``align_short`` (safety-short), ``alpaca_cleaned_no_safety`` (filtered alpaca_cleaned dataset)
3. ``--rank``: To determine how many ranks needed to be removed .
4. ``--top_remove`` : To determine whether to remove the top-critical ranks or the least-critical ranks. If true, remove the top critical ranks

Example: Prune the top-10 safety-critical rank based on the safety-full(``align`` in the code) dataset.

```bash
model="llama2-7b-chat-hf"
method="low_rank"
type="unstructured"
suffix="weightonly"
save_dir="out/$model/$type/${method}_${suffix}/align/"

python main_low_rank.py \
    --model $model \
    --prune_method $method \
    --prune_data align \
    --rank 10 \
    --top_remove \
    --save $save_dir \
    --eval_zero_shot \
    --eval_attack \
    --save_attack_res 
```

### 3.2 Remove the least safety-critical ranks
Similar to 3.1, but here we don't need to add ``--top_remove`` in the command line. 

Example: Remove the bottom-1000 safety-critical rank based on the safety-short(``align_short`` in the code) dataset.

```bash
model="llama2-7b-chat-hf"
method="low_rank"
type="unstructured"
save_dir="out/$model/$type/${method}/align_short/"

python main_low_rank.py \
    --model $model \
    --prune_method $method \
    --prune_data align_short \
    --rank 1000 \
    --top_remove \
    --save $save_dir \
    --eval_zero_shot \
    --eval_attack \
    --save_attack_res 
```

### 3.3 Remove rank with orthogonal projection
The main function of this program is ``main_low_rank_diff.py``.

Important parameters are:

1. ``--prune_method``: To specify the method of rank removal, here we use ``low_rank_diff``, which corresponds to the (ActSVD with orthogonal projection in the paper)
2. ``--rank_pos``: Specify the $r^u$ in the paper.
3. ``--rank_neg``: Specify the $r^s$ in the paper.
4. ``--prune_data_pos``: The data to determine the utility projection matrix, we use ``alpaca_cleaned_no_safety``.
5. ``--pruned_data_neg``: The data to determine the safety projection matrix, we recommend to use ``align``.

Example: Prune based on rank-3000 utility projection matrix and rank-4000 safety projection matrix on ``alpaca_cleaned_no_safety`` (filtered alpaca_cleaned dataset without safety-related prompt-response pairs) and safety-full on ``llama2-7b-chat-hf``.

```bash
model="llama2-7b-chat-hf"
type="unstructured"
ru=3000
rs=4000
method="low_rank_diff"
save_dir="out/$model/$type/${method}/align/"

python main_low_rank_diff.py \
    --model $model \
    --rank_pos $ru \
    --rank_neg $rs \
    --prune_data_pos "alpaca_cleaned_no_safety" \
    --prune_data_neg "align" \
    --save $save_dir \
    --eval_zero_shot \
    --eval_attack \
```

## 4. BibTeX
If you find our code and paper helpful, please consider citing our work:
```bibtex
@inproceedings{weiassessing,
  title={Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications},
  author={Wei, Boyi and Huang, Kaixuan and Huang, Yangsibo and Xie, Tinghao and Qi, Xiangyu and Xia, Mengzhou and Mittal, Prateek and Wang, Mengdi and Henderson, Peter},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
