# UniAlignment: Semantic Alignment for Unified Image Generation, Understanding, Manipulation and Perception

<h5 align="center"> 

[<img src="https://img.shields.io/badge/arXiv-Paper-red.svg?logo=arxiv">](https://arxiv.org/abs/2509.23760)
[![GitHub](https://img.shields.io/badge/GitHub-Code-green?logo=github)](https://github.com/glimmer16/UniAlignment)
[![HuggingFace](https://img.shields.io/badge/🤗%20Model-Huggingface-yellow)](https://huggingface.co/TencentARC/TokLIP)
[<img src="https://img.shields.io/badge/License-MIT-B762C1?logo=open-source-initiative&logoColor=white">](/LICENSE)


</div>

**This repository is the official PyTorch implementation of the paper "UniAlignment: Semantic Alignment for Unified Image Generation, Understanding, Manipulation and Perception".**

</div>

### Abstract
> The remarkable success of diffusion models in text-to-image generation has sparked growing interest in expanding their capabilities to a variety of multi-modal tasks, including image understanding, manipulation, and perception. These tasks require advanced semantic comprehension across both visual and textual modalities, especially in scenarios involving complex semantic instructions. However, existing approaches often rely heavily on vision-language models (VLMs) or modular designs for semantic guidance, leading to fragmented architectures and computational inefficiency. To address these challenges, we propose UniAlignment, a unified multimodal generation framework within a single diffusion transformer. UniAlignment introduces a dual-stream diffusion training strategy that incorporates both intrinsicmodal semantic alignment and cross-modal semantic alignment, thereby enhancing the model’s cross-modal consistency and instruction-following robustness. Additionally, we present SemGen-Bench, a new benchmark specifically designed to evaluate multimodal semantic consistency under complex textual instructions. Extensive experiments across multiple tasks and benchmarks demonstrate that UniAlignment outperforms existing baselines, underscoring the significant potential of diffusion models in unified multimodal generation.

## 👀 Introduction

<img src="./assets//UniAlignment.png" alt="UniAlignment" style="zoom:50%;" />

- We introduce UniAlignment, a unified multimodal generative model based on **a single Diffusion Transformer**, demonstrating outstanding performance while maintaining lightweight design and computational efficiency.

- UniAlignment introduces **two complementary semantic alignment mechanisms** that significantly enhances image-text semantic consistency and instruction-following robustness.

- A rigorous new benchmark **SemGen-Bench** is constructed for evaluating multimodal semantic alignment under complex, compositional instructions, establishing a highstandard baseline for future research.

## 🔜 TODOs
- [x] Technical report.
- [ ] Release training codes and pretrained weights.
- [ ] Release inference codes.
- [ ] Release SemGen-Bench.

## 🔧 Installation

```shell
git clone https://github.com/glimmer16/UniAlignment.git
cd UniAlignment
conda create -n unialignment python=3.10
conda activate unialignment
pip install --upgrade pip 
pip install -r requirements.txt
```

## ⚙️ Usage
### Training

1. Please refer to

### Data

111

### Inference

We provide the inference example in `src/inference.py`. 

```shell
cd src
python inference.py --model-config 'ViT-SO400M-16-SigLIP2-384-toklip' --pretrained 'YOUR_TOKLIP_PATH'
```

## 🙏 Acknowledgement
This repo is mainly based on [DualDiffusion](https://github.com/zijieli-Jlee/Dual-Diffusion) and [SD3](https://stability.ai/news/stable-diffusion-3-medium).

Thanks to the original authors for their excellent work!


## 📝 Citation
Please cite our work if you use our code or discuss our findings in your own research:
```bibtex
@article{song2025unialignment,
  title={UniAlignment: Semantic Alignment for Unified Image Generation, Understanding, Manipulation and Perception},
  author={Song, Xinyang and Wang, Libin and Wang, Weining and Liu, Shaozhen and Zheng, Dandan and Chen, Jingdong and Li, Qi and Sun, Zhenan},
  journal={arXiv preprint arXiv:2509.23760},
  year={2025}
}
```


## 📂 Contact
If you have further questions, please open an issue or contact <xinyang.song@cripac.ia.ac.cn>.

Discussions and potential collaborations are also welcome.
