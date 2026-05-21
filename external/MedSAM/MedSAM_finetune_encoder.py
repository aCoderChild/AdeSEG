# Finetune MedSAM with LoRA using PEFT and YAML config

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datetime import datetime
import shutil
import wandb
import monai
from segment_anything import sam_model_registry
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from utils.lora_medsam import apply_lora_to_vit_encoder

# Image/mask dataset for standard segmentation
from PIL import Image
class ImageMaskDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, img_size=(1024, 1024)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
        self.image_files = [f for f in self.image_files if f in self.mask_files]
        assert len(self.image_files) > 0, 'No matching image/mask pairs found.'
        print(f"Found {len(self.image_files)} image/mask pairs.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        image = Image.open(img_path).convert('RGB').resize(self.img_size)
        mask = Image.open(mask_path).convert('L').resize(self.img_size)
        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32)
        mask = (mask > 127).astype(np.float32)  # binarize
        image = torch.from_numpy(image).permute(2,0,1)  # (3,H,W)
        mask = torch.from_numpy(mask).unsqueeze(0)      # (1,H,W)
        # full-image mask supervision - automatic mask generator
        bbox = torch.tensor([0,0,self.img_size[1],self.img_size[0]], dtype=torch.float32)
        return image, mask, bbox, img_name

# MedSAM wrapper (from train_one_gpu.py)
class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=box_torch, masks=None
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

def main():
    # 1. Load config
    with open("external/MedSAM/config/finetune.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Set device
    requested_device = config.get("device", "cuda:0")
    if requested_device.startswith("mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    elif requested_device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(requested_device)
        print(f"Using CUDA device: {requested_device}")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")
    torch.manual_seed(2023)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # output dir
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = os.path.join(config["work_dir"], f"MedSAM-ViT-B-{run_id}")
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(model_save_path, run_id + "_finetune_medsam_lora.py"))


    # Load model
    sam_model = sam_model_registry[config["model_type"]](checkpoint=config["checkpoint"])

    # prompt encoder for 1024x1024 images (patch size 16, 64x64 embedding)
    # TODO: read class PromptEncoder again
    from segment_anything.modeling.prompt_encoder import PromptEncoder
    prompt_embed_dim = 256
    sam_model.prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans=16,
    )

    # Apply LoRA for MedSAM ViT encoder
    if config.get("use_lora", False):
        r = config.get("lora_r", 8)
        lora_alpha = config.get("lora_alpha", 16)
        sam_model.image_encoder = apply_lora_to_vit_encoder(sam_model.image_encoder, r, lora_alpha) # finetune only the image encoder
        print("LoRA applied to MedSAM ViT encoder (qkv, proj)")

    # Freeze all parameters except LoRA (for computation)
    for name, param in sam_model.image_encoder.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    for param in sam_model.mask_decoder.parameters():
        param.requires_grad = False
    for param in sam_model.prompt_encoder.parameters():
        param.requires_grad = False

    sam_model.image_encoder = sam_model.image_encoder.to(device)
    sam_model.mask_decoder = sam_model.mask_decoder.to(device)
    sam_model.prompt_encoder = sam_model.prompt_encoder.to(device)

    # MedSAM wrapper
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.train()


    # device
    if config.get("device", "cuda:0").startswith("mps") and not torch.backends.mps.is_available():
        print("Warning: MPS device requested but not available. Using CPU instead.")

    # Optimizer: only LoRA params
    lora_params = [p for n, p in medsam_model.image_encoder.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        lora_params, lr=config["lr"], weight_decay=config["weight_decay"]
    )
    # Cosine Annealing LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"], eta_min=1e-6
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")


    # prepare dataset
    train_dataset = ImageMaskDataset(
        os.path.join(config["train_data_path"], "images"),
        os.path.join(config["train_data_path"], "masks"),
        img_size=(1024,1024)
    )
    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    # Training loop
    num_epochs = config["num_epochs"]
    iter_num = 0
    losses = []
    best_loss = 1e10
    from torch.amp import autocast, GradScaler
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    for epoch in range(num_epochs):
        epoch_loss = 0
        grad_norm = 0.0

        last_lora_grad = None
        last_lora_name = None
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            with autocast('cuda' if device.type == 'cuda' else device.type, enabled=(device.type == 'cuda')):
                medsam_pred = medsam_model(image, boxes_np)
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
            if device.type == 'cuda':
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Store the last LoRA parameter grad norm for this batch
            for name, param in medsam_model.image_encoder.named_parameters():
                if param.requires_grad and param.grad is not None and (('lora_A' in name) or ('lora_B' in name)):
                    last_lora_grad = param.grad.norm().item()
                    last_lora_name = name

            # Compute grad norm for all trainable params (LoRA only)
            total_norm = 0.0
            for name, param in medsam_model.image_encoder.named_parameters():
                if param.requires_grad and param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
            grad_norm = total_norm ** 0.5

            if device.type == 'cuda':
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            epoch_loss += loss.item()
            iter_num += 1
        epoch_loss /= step
        losses.append(epoch_loss)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Time: {datetime.now().strftime('%Y%m%d-%H%M')}, Epoch: {epoch}, Loss: {epoch_loss}, GradNorm: {grad_norm}, LR: {current_lr}")
        if last_lora_name is not None:
            print(f"Last LoRA grad norm after epoch: {last_lora_name}: grad norm = {last_lora_grad}")
        # wandb.log enabled 
        wandb.log({
            "epoch": epoch,
            "loss": epoch_loss,
            "learning_rate": current_lr,
            "grad_norm": grad_norm
        })

        # Save latest model
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, os.path.join(model_save_path, "medsam_model_latest.pth"))

        # Save best model (according to loss in training)
        # TODO: save best according to the val dataset (split train : val), not training
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, os.path.join(model_save_path, "medsam_model_best.pth"))


if __name__ == "__main__":
    main()
