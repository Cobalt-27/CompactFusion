# 🐳[NeurIPS 2025] Accelerating Parallel Diffusion Model Serving with Residual Compression

> **TL;DR**: Diffusion models exhibit heavy temporal redundancy, yet we transmit full activations step after step.  
> **Why are we sending near-duplicated data across GPUs?**  
> **CompactFusion** transmits only the compressed residuals — the real information change — to drastically reduce bandwidth with minimal quality loss.

![Teaser Image](https://img.picgo.net/2025/05/15/teaser741bf3f5ec634b23.png)

## ☘️ Acknowledgements
We owe special thanks to the **xDiT** team—without their excellent open-source framework, this project would simply not exist.

Their work laid the foundation for everything we've built.

We thank the **DistriFusion** authors for sharing their code and system.

We also thank [common_metrics_on_video_quality](https://github.com/JunyaoHu/common_metrics_on_video_quality) for their excellent video quality evaluation tools.

## 🍀 Motivation

- Diffusion models generate data step-by-step, but their intermediate activations change **slowly and predictably**.
- In multi-GPU inference, these large activations are repeatedly transmitted between devices.
- The transmitted data are **highly redundant**, wasting precious bandwidth on near-duplicate content.
- We ask: **Why are we transmitting redundant stuff at all?**


## 🚀 Introducing CompactFusion

CompactFusion is a residual compression framework for parallel diffusion model serving.  
It compresses only the **change** (residual) between activations across steps — and adds optional **error feedback** to maintain reconstruction quality.

✅ CompactFusion only targets **communication**, making it:
- 📦 **Plug-and-play**: No model re-training or modification
- 🛠 **Framework-compatible**: Integrates into ring attention, patch parallelism and more
- ⚡ **Extremely efficient**: Up to **100× compression**, with **<1% data transmitted**, but still **outperforming DistriFusion in quality**.

![Intro Image](https://img.picgo.net/2025/05/15/intro40850a8451398a26.png)

## 🐚 Method Illustration

<table>
<tr>
<td><b>Residual Compression Principle</b><br><img src="https://img.picgo.net/2025/05/15/residualae697c4a98859629.png" alt="Residual Illustration"></td>
<td><b>System Architecture</b><br><img src="https://img.picgo.net/2025/05/15/archce65b198fc390fc9.png" alt="System Diagram"></td>
</tr>
</table>

## ✨ Supported Features

CompactFusion supports out-of-the-box compression for:

- ✅ Compressed Ring Attention
- ✅ Compressed Patch Parallel
- ✅ DistriFusion migrated to `xDiT`
- ✅ Patch Parallel migrated to `xDiT`
- ✅ FLUX, CogVideoX, SD, Pixart-alpha and other backbones


## 💾 Installation & Setup

We build CompactFusion on top of the excellent [`xDiT`](https://github.com/xdit-project/xDiT) framework.

## 🐳 Recommended Setup

You may simply use the pre-built Docker image from xDiT:

```bash
docker pull thufeifeibear/xdit-dev
```

## 🔧 Code Examples
Example usages are provided in:

`examples/cogvideox_example.py`

`examples/flux_example.py`

**We do not modify the setup of xDiT. You can refer directly to xDiT documentation for usage details.**

## 🎓 Citation

If you use this code for your research, please cite our paper.

```bibtex
@article{luo2025accelerating,
  title={Accelerating Parallel Diffusion Model Serving with Residual Compression},
  author={Luo, Jiajun and Xiao, Yicheng and Xu, Jianru and You, Yangxiu and Lu, Rongwei and Tang, Chen and Jiang, Jingyan and Wang, Zhi},
  journal={arXiv preprint arXiv:2507.17511},
  year={2025}
}
```