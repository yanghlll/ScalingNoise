## ScalingNoise: Scaling Inference-Time Search for Generating Infinite Videos
<div align="center">

<p>
💾 <b> VRAM < 10GB </b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
🚀 <b> Infinitely Long Videos</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
⭐️ <b> Tuning-free</b>
</p>

<a href="https://arxiv.org/pdf/2503.16400"><img src='https://img.shields.io/badge/arXiv-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://yanghlll.github.io/ScalingNoise.github.io/"><img src='https://img.shields.io/badge/Project-Page-Green'></a>

</div>

## 📽️ See more video samples in our <a href="https://yanghlll.github.io/ScalingNoise.github.io"> project page</a>!
<div align="center">

<img src="assets/scalenoise.gif">

"Impressionist style, a yellow rubber duck floating on the wave on the sunset, 4k resolution.", 

VideoCrafter2, 100 frames, 320 X 512 resolution

</div>

## Clone our repository
```
git clone https://github.com/yanghlll/ScalingNoise.git
cd ScalingNoise
```

## ☀️ Start with <a href="https://github.com/AILab-CVC/VideoCrafter">VideoCrafter</a>

### 1. Environment Setup ⚙️ (python==3.9.21 recommended)
```
conda create -n ScalingNoise python=3.9.21 
pip install -r requirements.txt
```

### 2.1 Download the models from Hugging Face🤗
|Model|Resolution|Checkpoint
|:----|:---------|:---------
|VideoCrafter2 (Text2Video)|320x512|[Hugging Face](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)

### 2.2 Set file structure
Store them as following structure:
```
cd FIFO-Diffusion_public
    .
    └── videocrafter_models
        └── base_512_v2
            └── model.ckpt      # VideoCrafter2 checkpoint
```

### 3.1. Run with VideoCrafter2 (Single GPU)
```
bash scalenoise.sh
```



## 🤓 Acknowledgements
Our codebase builds on [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter) and [FIFO-Diffusion](https://github.com/jjihwan/FIFO-Diffusion_public). 
Thanks to the authors for sharing their awesome codebases!
