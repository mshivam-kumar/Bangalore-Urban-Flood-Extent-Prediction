import torch

def masked_bce_loss(outputs, labels, x_static, pad_val, nodata_dict, channel_names, criterion):
    """
    outputs: (B,H,W) raw logits if using BCEWithLogitsLoss
    labels: (B,H,W) 0/1 mask
    """
    # 1. Build mask (valid pixels)
    valid_mask = (x_static[:, 0, :, :] != pad_val)   # start with VH

    for i, channel_name in enumerate(channel_names[:-1]):  # exclude Rainfall (last channel)
        if channel_name in nodata_dict:
            valid_mask &= (x_static[:, i, :, :] != nodata_dict[channel_name])
    
    valid_mask &= labels != -1.0

    # 2. Compute unreduced BCE
    loss_unreduced = criterion(outputs, labels.float())  # (B,H,W)

    # 3. Apply mask
    masked_loss = loss_unreduced * valid_mask.float()

    # 4. Normalize
    if valid_mask.sum() > 0:
        return masked_loss.sum() / valid_mask.sum()
    else:
        return torch.tensor(0.0, device=outputs.device)