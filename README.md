# The official code of FLASH

### Beyond Attention Imbalance: Mitigating Hallucinations via Spectral Surgery (ICML 2026)
**Siqi Lu, Wei Suo, Yongbin Zheng, Jianhang Yao, Wanying Xu, Peng Wang**

**FLASH (Frequency-Localized Attention SHaping)** is a training-free method for mitigating hallucinations in large vision-language models. It selects vision-related attention heads and modulates the frequency content of their visual attention scores and value representations. FLASH operates during inference without contrastive decoding or an additional model branch.


<img src="https://github.com/a6103121/FLASH/blob/main/a.png" width="512px">

## 1. Create and Activate the environment 

```bash
conda env create -f environment.yml -n flash
conda activate flash
```

## 2. Model and data preparation

### Model weights

Obtain the original-format [LLaVA-1.5 7B checkpoint](https://huggingface.co/liuhaotian/llava-v1.5-7b) and place it in a local directory such as:

```text
/path/to/checkpoints/llava-v1.5-7b/
```

### POPE-COCO data

Prepare the COCO 2014 validation images. `--data-path` must point directly to the directory containing image files, for example:

```text
/path/to/coco/val2014/
└── COCO_val2014_000000310196.jpg
```

The question annotations are already included:

```text
pope_coco/chat/coco_pope_chat_random.json
pope_coco/chat/coco_pope_chat_popular.json
pope_coco/chat/coco_pope_chat_adversarial.json
```

### Generate answers

After configuring a compatible environment, model weights, and image data, run:

```bash
mkdir -p pope/llava-1.5

python pope_eval.py \
  --model llava-1.5 \
  --model-path /path/to/checkpoints/llava-v1.5-7b \
  --data-path /path/to/coco/val2014 \
  --pope-type random 
  
```
Replace `random` with `popular` or `adversarial` to evaluate the other splits.


## Evaluation

### Score the included predictions

Scoring uses only the Python standard library; model weights and a GPU are not needed:

```bash
for split in random popular adversarial; do
  python pope_ans.py \
    --ans_file "pope/llava-1.5/pope_eval_${split}_lambdas_0.9_lambdav_1.2_tau_0.2_k_5.jsonl"
done
```




## Citation

If you use FLASH in your research, please cite **Beyond Attention Imbalance: Mitigating Hallucinations via Spectral Surgery**. The entry below uses manuscript metadata; replace it with the official proceedings or preprint entry when available.

```bibtex
@inproceedings{lubeyond,
  title={Beyond Attention Imbalance: Mitigating Hallucinations via Spectral Surgery},
  author={Lu, Siqi and Suo, Wei and Zheng, Yongbin and Yao, Jianhang and Xu, Wanying and Wang, Peng},
  booktitle={Forty-third International Conference on Machine Learning}
}
```

## Acknowledgments

Our code is adapted from [PAI](https://github.com/LALBJ/PAI). We thank the PAI authors for making their implementation publicly available.

This repository includes code derived from [LLaVA](https://github.com/haotian-liu/LLaVA) and additional third-party model implementations under `llava/`. We thank their authors and the creators of the evaluation datasets. Preserve the copyright and license notices included in those source files.
