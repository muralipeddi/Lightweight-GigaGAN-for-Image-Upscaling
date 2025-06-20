# Lightweight GigaGAN for Efficient Image Upscaling

This project explores model compression techniques for the unconditional GigaGAN model to reduce GPU memory usage and inference time while preserving image generation quality.

## Key Techniques

- **Structured Pruning**: Iteratively removed 5% of convolutional channels using L1-norm scoring.
- **Post-Training Quantization**: Applied selective INT8 quantization on non-critical layers.
- **Knowledge Distillation**: Trained a compact student model using output supervision from the original GigaGAN teacher.



- `unconditional_gigagan.ipynb` for baseline
- `GigaGAN_Pruining.ipynb` for pruning
- `Final_quantized_gigagan.ipynb` for quantization
- `3_GigaGAN_Knowledge_Distillation_ipynb_(KD_Cell_Update).ipynb` for distillation

>  All notebooks are compatible with A100 GPUs and optimized for AMP.

## Results

| Model Variant | Memory ↓ | Speed ↑ | Visual Fidelity |
|---------------|-----------|----------|------------------|
| Baseline      | —         | —        | 5 |
| Pruned        | -30%      | +25%     | 4 |
| Quantized     | -35%      | +30%     | 4 |
| Distilled     | -40%      | +35%     | 5 |


