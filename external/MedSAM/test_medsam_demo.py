import numpy as np
from skimage import io

import torch
from segment_anything.build_sam import sam_model_registry
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from utils.lora_medsam import apply_lora_to_vit_encoder

# Load model

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')



def load_medsam_model(ckpt_path, device, is_finetuned=False):
    print(f"\nLoading MedSAM model from: {ckpt_path}")
    # Always instantiate from the original MedSAM weights
    medsam_model = sam_model_registry["vit_b"](checkpoint="external/MedSAM/work_dir/MedSAM/medsam_vit_b.pth")
    # Re-initialize prompt encoder for 1024x1024 images (patch size 16)
    from segment_anything.modeling.prompt_encoder import PromptEncoder
    prompt_embed_dim = 256
    medsam_model.prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(64, 64),  # 1024/16=64
        input_image_size=(1024, 1024),
        mask_in_chans=16,
    )
    # Always apply LoRA for finetuned checkpoints
    if is_finetuned:
        print("Applying LoRA to ViT encoder for finetuned checkpoint.")
        medsam_model.image_encoder = apply_lora_to_vit_encoder(medsam_model.image_encoder, r=8, alpha=16)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Handle both plain state_dict and dict with 'model' key
    if isinstance(ckpt, dict) and 'model' in ckpt:
        print("  Detected full training checkpoint. Extracting 'model' weights.")
        state_dict = ckpt['model']
        print(f"  [DEBUG] Extracted state_dict keys: {list(state_dict.keys())[:10]} ... (total: {len(state_dict)})")
    else:
        state_dict = ckpt
        print(f"  [DEBUG] Using checkpoint as state_dict, keys: {list(state_dict.keys())[:10]} ... (total: {len(state_dict)})")
    print(f"  [DEBUG] Passing state_dict with {len(state_dict)} keys to load_state_dict")
    try:
        medsam_model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"  Error loading checkpoint: {e}")
        print("  Attempting to load with strict=False...")
        try:
            medsam_model.load_state_dict(state_dict, strict=False)
            print("  Loaded with strict=False. Some keys may be missing or unexpected.")
        except Exception as e2:
            print(f"  Still failed: {e2}")
            return None
    medsam_model.eval()
    medsam_model.to(device)
    # Print model structure for debugging LoRA layers
    print("\n[DEBUG] MedSAM model structure after loading:")
    print(medsam_model)
    return medsam_model


# Load demo image and resize to 256x256
from skimage.transform import resize
img = io.imread("data/segmentation/train_data/images/cju0qkwl35piu0993l0dewei2.jpg")
if len(img.shape) == 2:
    img = np.repeat(img[:, :, None], 3, axis=-1)
img = resize(img, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.float32)
print(f"Demo image shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")

# Add a synthetic image for testing (already 1024x1024)
synthetic_img = np.zeros((1024, 1024, 3), dtype=np.float32)
synthetic_img[256:768, 256:768, :] = 255.0

ckpt_path = "external/MedSAM/work_dir/MedSAM/MedSAM-ViT-B-20260403-1052/medsam_model_best.pth"  # Path to your checkpoint
print(f"\n===== Testing MedSAM-ViT-B-20260403-0328 checkpoint =====")
try:
    medsam_model = load_medsam_model(ckpt_path, device, is_finetuned=True)
except Exception as e:
    print(f"  Error loading checkpoint: {e}")
    exit(1)
for img_name, img_input in [
    ("demo_raw", img),
    ("demo_normalized", img / 255.0),
    ("synthetic_raw", synthetic_img),
    ("synthetic_normalized", synthetic_img / 255.0)
]:
    print(f"\nTesting with {img_name} image input:")
    print(f"  [DEBUG] img_input shape: {img_input.shape}, dtype: {img_input.dtype}, min: {img_input.min()}, max: {img_input.max()}, mean: {img_input.mean()}")
    print(f"  [DEBUG] img_input sample (flattened, first 10 values): {img_input.flatten()[:10]}")
    # Prepare image tensor
    img_tensor = torch.from_numpy(img_input).float().permute(2, 0, 1).unsqueeze(0).to(device)
    # Use a center box prompt (full image)
    box = [0, 0, 1023, 1023]
    box_np = np.array([box], dtype=np.float32)
    box_torch = torch.from_numpy(box_np).to(device)
    with torch.no_grad():
        # Forward pass
        image_embedding = medsam_model.image_encoder(img_tensor)
        from segment_anything.modeling.prompt_encoder import PromptEncoder
        # Prepare box for prompt encoder
        box_torch = box_torch if len(box_torch.shape) == 3 else box_torch[:, None, :]
        sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
            points=None, boxes=box_torch, masks=None)
        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        import torch.nn.functional as F
        low_res_pred = torch.sigmoid(low_res_logits)
        low_res_pred = F.interpolate(
            low_res_pred, size=(1024, 1024), mode="bilinear", align_corners=False)
        low_res_pred = low_res_pred.squeeze().detach().cpu().numpy()
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    print(f"    Mask: min={medsam_seg.min()}, max={medsam_seg.max()}, mean={medsam_seg.mean()}")
    if medsam_seg.max() == 0:
        print("    [WARNING] Mask is all zeros. This may indicate a training issue or checkpoint mismatch.")
    out_path = f"external/MedSAM/assets/MedSAM-ViT-B-20260403-0328_{img_name}_custom_mask.png"
    io.imsave(out_path, (medsam_seg * 255).astype(np.uint8))
    print(f"    Saved mask as {out_path}")
