# Lightweight GigaGAN for Efficient Image Upscaling

This project explores model compression techniques for the unconditional GigaGAN model to reduce GPU memory usage and inference time while preserving image generation quality.

## 🧠 Key Techniques

- **Structured Pruning**: Iteratively removed 5% of convolutional channels using L1-norm scoring.
- **Post-Training Quantization**: Applied selective INT8 quantization on non-critical layers.
- **Knowledge Distillation**: Trained a compact student model using output supervision from the original GigaGAN teacher.

## 📁 Project Structure

- `notebooks/`: All Jupyter notebooks used in different compression stages.
- `gigagan_pytorch-0.4.0.whl`: Precompiled GigaGAN wheel for PyTorch.
- `HPML_Report-2.pdf`: Detailed project report.
- `requirements.txt`: Python dependencies.

## 🧪 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/GigaGAN-Lightweight.git
cd GigaGAN-Lightweight
```

### 2. Create Environment

```bash
conda create -n gigagan_env python=3.10 -y
conda activate gigagan_env
```

### 3. Install Dependencies

```bash
pip install gigagan_pytorch-0.4.0-py3-none-any.whl
pip install -r requirements.txt
```

### 4. Run the Notebooks

Navigate to the `notebooks/` directory and start with:

- `unconditional_gigagan.ipynb` for baseline
- `GigaGAN_Pruining.ipynb` for pruning
- `Final_quantized_gigagan.ipynb` for quantization
- `3_GigaGAN_Knowledge_Distillation_ipynb_(KD_Cell_Update).ipynb` for distillation

> ✅ All notebooks are compatible with A100 GPUs and optimized for AMP.

## 📊 Results

| Model Variant | Memory ↓ | Speed ↑ | Visual Fidelity |
|---------------|-----------|----------|------------------|
| Baseline      | —         | —        | ✅✅✅✅✅ |
| Pruned        | -30%      | +25%     | ✅✅✅✅ |
| Quantized     | -35%      | +30%     | ✅✅✅✅ |
| Distilled     | -40%      | +35%     | ✅✅✅✅✅ |

## 📜 Citation

If you use this code, please cite the original paper and our report:

> Bhaktharam, S., Peddi, M. (2024). *Lightweight GigaGAN for Efficient Image Upscaling*. New York University.
