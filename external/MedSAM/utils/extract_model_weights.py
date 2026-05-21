import torch

# Replace with your actual finetuned checkpoint path
finetuned_ckpt_path = 'external/MedSAM/work_dir/MedSAM/MedSAM-ViT-B-20260402-1553/medsam_model_best.pth'
# Output path for model-only weights
output_path = 'external/MedSAM/work_dir/MedSAM/MedSAM-ViT-B-20260402-1553/medsam_model_best_only_weights.pth'

ckpt = torch.load(finetuned_ckpt_path, map_location='cpu')
if 'model' in ckpt:
    torch.save(ckpt['model'], output_path)
    print(f"Saved model-only weights to {output_path}")
else:
    print("Error: 'model' key not found in checkpoint.")
