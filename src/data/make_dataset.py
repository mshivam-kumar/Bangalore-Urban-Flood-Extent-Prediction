# flood_dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

def normalize_minmax(tensor, mask):
    """Scale tensor to [0,1], leave pad/nodata untouched."""
    valid = tensor[mask]
    mins = valid.min()
    maxs = valid.max()
    normed = tensor.clone()
    normed[mask] = (valid - mins) / (maxs - mins + 1e-6)
    return normed


def standardize(tensor, mask):
    """Mean-std standardization, leave pad/nodata untouched."""
    valid = tensor[mask]
    mean = valid.mean()
    std = valid.std()
    normed = tensor.clone()
    normed[mask] = (valid - mean) / (std + 1e-6)
    return normed


# --------------------------
# Dataset
# --------------------------
class FloodDataset(Dataset):
    """
    Dataset for flood feature stacks (.npy).

    Each .npy file is (11, H, W):
      [VH, VV, dVH, dVV,
       DEM_elevation, DEM_slope, DEM_sin_aspect, DEM_cos_aspect, DEM_curvature,
       HND, Rainfall]
    """

    CHANNELS = [
        "VH", "VV", "dVH", "dVV",
        "DEM_elevation", "DEM_slope", "DEM_sin_aspect", "DEM_cos_aspect", "DEM_curvature",
        "HND", "Rainfall"
    ]

    def __init__(self, index_list,
                 transform_normalize=None,
                 transform_standardize=None,
                 pad_val=-99999.0,
                 nodata_dict=None):

        self.files = index_list
        self.transform_normalize = transform_normalize or []
        self.transform_standardize = transform_standardize or []
        self.pad_val = pad_val

        # default nodata dict
        self.nodata_dict = nodata_dict or {
            "VH": None,
            "VV": None,
            "dVH": None,
            "dVV": None,
            "DEM_elevation": -32768.0,
            "DEM_slope": -32768.0,
            "DEM_sin_aspect": -32768.0,
            "DEM_cos_aspect": -32768.0,
            "DEM_curvature": -32768.0,
            "HND": None,
            "Rainfall": None
        }

    def __len__(self):
        return len(self.files)

# flood_dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

# --------------------------
# Masked normalization helpers (torch)
# --------------------------
def normalize_minmax_masked(tensor: torch.Tensor, mask: torch.BoolTensor, eps: float = 1e-6):
    """
    tensor: (H,W) torch.float32
    mask: boolean mask of same shape where True => valid pixel
    returns: tensor_clone where valid pixels are min-max scaled to [0,1],
             invalid pixels unchanged.
    """
    out = tensor.clone()
    vals = tensor[mask]
    if vals.numel() == 0:
        return out
    mn = vals.min()
    mx = vals.max()
    out[mask] = (vals - mn) / (mx - mn + eps)
    return out

def standardize_masked(tensor: torch.Tensor, mask: torch.BoolTensor, eps: float = 1e-6):
    """
    Standardize valid pixels (zero mean, unit std). Invalid pixels unchanged.
    """
    out = tensor.clone()
    vals = tensor[mask]
    if vals.numel() == 0:
        return out
    mu = vals.mean()
    sigma = vals.std()
    out[mask] = (vals - mu) / (sigma + eps)
    return out

# --------------------------
# Dataset
# --------------------------
class FloodDataset(Dataset):
    """
    Loads feature .npy files created by your features pipeline (11,H,W).
    Returns features separately (dict), label mask (H,W), and nodata/pad info.
    """

    CHANNELS = [
        "VH", "VV", "dVH", "dVV",
        "DEM_elevation", "DEM_slope", "DEM_sin_aspect", "DEM_cos_aspect", "DEM_curvature",
        "HND", "Rainfall"
    ]

    def __init__(self, index_list,
                 transform_normalize=None,
                 transform_standardize=None,
                 pad_val: float = -99999.0,
                 nodata_dict: dict = None,
                 vh_threshold: float = -15.5):
        """
        index_list: list of feature .npy file paths
        transform_normalize: list of channel indices to min-max normalize (e.g. [0,1,2,3])
        transform_standardize: list of channel indices to standardize (e.g. [4,5,6,7,8,9,10])
        pad_val: padded pixel marker (kept unchanged)
        nodata_dict: mapping channel_name -> nodata_value (or None)
        vh_threshold: threshold (dB) on raw VH to create binary flood label
        """
        self.files = list(index_list)
        self.transform_normalize = set(transform_normalize or [])
        self.transform_standardize = set(transform_standardize or [])
        self.pad_val = float(pad_val)
        self.vh_threshold = float(vh_threshold)

        # default nodata dict (user-provided overrides)
        default_nodata = {
            "VH": None,
            "VV": None,
            "dVH": None,
            "dVV": None,
            "DEM_elevation": -32768.0,
            "DEM_slope": -32768.0,
            "DEM_sin_aspect": -32768.0,
            "DEM_cos_aspect": -32768.0,
            "DEM_curvature": -32768.0,
            "HND": None,
            "Rainfall": None
        }
        self.nodata_dict = (nodata_dict.copy() if nodata_dict is not None else default_nodata)
        # ensure keys exist
        for k in self.CHANNELS:
            if k not in self.nodata_dict:
                self.nodata_dict[k] = default_nodata.get(k, None)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns:
            {
              "features": dict(channel_name -> torch.Tensor (H,W)),
              "label": torch.Tensor (H,W) float32 with {1.0 flood, 0.0 no-flood, -1.0 invalid},
              "pad_val": float,
              "nodata_dict": dict
            }
        """
        npy_path = self.files[idx]
        arr = np.load(npy_path)  # shape (11, H, W); dtype float32/float64
        arr = arr.astype(np.float32, copy=False)

        # Save raw VH (numpy) for label creation before any transform
        raw_vh = arr[0].copy()  # VH band as float32

        # Convert to torch
        x_all = torch.from_numpy(arr)  # shape (11, H, W), dtype=torch.float32

        H, W = x_all.shape[1], x_all.shape[2]

        features = {}

        # Process channels separately and store
        for ch_idx, ch_name in enumerate(self.CHANNELS):
            ch = x_all[ch_idx]  # (H,W) torch tensor

            nodata_val = self.nodata_dict.get(ch_name, None)

            # construct mask_valid (True -> valid pixel)
            mask_valid = torch.ones_like(ch, dtype=torch.bool)
            mask_valid &= (ch != self.pad_val)
            if nodata_val is not None:
                # nodata_val is scalar float
                mask_valid &= (ch != float(nodata_val))

            # If no valid pixels, leave channel as-is (preserve pad/nodata)
            if not mask_valid.any():
                features[ch_name] = ch.clone()
                continue

            # Apply desired transform (masked); prefer normalize over standardize if both listed
            if ch_idx in self.transform_normalize:
                ch_out = normalize_minmax_masked(ch, mask_valid)
            elif ch_idx in self.transform_standardize:
                ch_out = standardize_masked(ch, mask_valid)
            else:
                ch_out = ch.clone()

            features[ch_name] = ch_out

        # -----------------------
        # Build label from raw VH (use raw values, not transformed VH)
        # -----------------------
        vh_raw_t = torch.from_numpy(raw_vh).to(dtype=torch.float32)  # (H,W)
        # invalid pixels mask for label
        mask_invalid = (vh_raw_t == float(self.pad_val))
        vh_nodata_val = self.nodata_dict.get("VH", None)
        if vh_nodata_val is not None:
            mask_invalid |= (vh_raw_t == float(vh_nodata_val))

        flood_mask = (vh_raw_t < float(self.vh_threshold)).to(dtype=torch.float32)  # 1.0 flood, 0.0 non-flood
        # mark invalid pixels as -1.0
        flood_mask[mask_invalid] = -1.0

        return features, flood_mask 




# Inludes pw_mask in batch
# features_dict, pw_mask, labels = dataset[idx]

# --------------------------
# Masked normalization helpers (torch)
# --------------------------
def normalize_minmax_masked(tensor: torch.Tensor, mask: torch.BoolTensor, eps: float = 1e-6):
    """
    tensor: (H,W) torch.float32
    mask: boolean mask of same shape where True => valid pixel
    returns: tensor_clone where valid pixels are min-max scaled to [0,1],
             invalid pixels unchanged.
    """
    out = tensor.clone()
    vals = tensor[mask]
    if vals.numel() == 0:
        return out
    mn = vals.min()
    mx = vals.max()
    out[mask] = (vals - mn) / (mx - mn + eps)
    return out

def standardize_masked(tensor: torch.Tensor, mask: torch.BoolTensor, eps: float = 1e-6):
    """
    Standardize valid pixels (zero mean, unit std). Invalid pixels unchanged.
    """
    out = tensor.clone()
    vals = tensor[mask]
    if vals.numel() == 0:
        return out
    mu = vals.mean()
    sigma = vals.std()
    out[mask] = (vals - mu) / (sigma + eps)
    return out

# --------------------------
# Dataset
# --------------------------
class FloodDataset_with_pw_mask(Dataset):
    """
    Loads feature .npy files created by your features pipeline (11,H,W).
    Returns features separately (dict), label mask (H,W), and nodata/pad info.
    """

    CHANNELS = [
        "VH", "VV", "dVH", "dVV",
        "DEM_elevation", "DEM_slope", "DEM_sin_aspect", "DEM_cos_aspect", "DEM_curvature",
        "HND", "Rainfall"
    ]

    def __init__(self, index_list,
                 transform_normalize=None,
                 transform_standardize=None,
                 pad_val: float = -99999.0,
                 nodata_dict: dict = None,
                 vh_threshold: float = -15.5):
        """
        index_list: list of feature .npy file paths
        transform_normalize: list of channel indices to min-max normalize (e.g. [0,1,2,3])
        transform_standardize: list of channel indices to standardize (e.g. [4,5,6,7,8,9,10])
        pad_val: padded pixel marker (kept unchanged)
        nodata_dict: mapping channel_name -> nodata_value (or None)
        vh_threshold: threshold (dB) on raw VH to create binary flood label
        """
        self.files = list(index_list)
        self.transform_normalize = set(transform_normalize or [])
        self.transform_standardize = set(transform_standardize or [])
        self.pad_val = float(pad_val)
        self.vh_threshold = float(vh_threshold)

        # default nodata dict (user-provided overrides)
        default_nodata = {
            "VH": None,
            "VV": None,
            "dVH": None,
            "dVV": None,
            "DEM_elevation": -32768.0,
            "DEM_slope": -32768.0,
            "DEM_sin_aspect": -32768.0,
            "DEM_cos_aspect": -32768.0,
            "DEM_curvature": -32768.0,
            "HND": None,
            "Rainfall": None
        }
        self.nodata_dict = (nodata_dict.copy() if nodata_dict is not None else default_nodata)
        # ensure keys exist
        for k in self.CHANNELS:
            if k not in self.nodata_dict:
                self.nodata_dict[k] = default_nodata.get(k, None)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns:
            {
              "features": dict(channel_name -> torch.Tensor (H,W)),
              "label": torch.Tensor (H,W) float32 with {1.0 flood, 0.0 no-flood, -1.0 invalid},
              "pad_val": float,
              "nodata_dict": dict
            }
        """
        npy_path = self.files[idx]
        arr = np.load(npy_path)  # shape (11, H, W); dtype float32/float64
        arr = arr.astype(np.float32, copy=False)

        # Save raw VH (numpy) for label creation before any transform
        raw_vh = arr[0].copy()  # VH band as float32

        # Convert to torch
        x_all = torch.from_numpy(arr)  # shape (11, H, W), dtype=torch.float32

        H, W = x_all.shape[1], x_all.shape[2]

        features = {}

        # Process channels separately and store
        for ch_idx, ch_name in enumerate(self.CHANNELS):
            ch = x_all[ch_idx]  # (H,W) torch tensor

            nodata_val = self.nodata_dict.get(ch_name, None)

            # construct mask_valid (True -> valid pixel)
            mask_valid = torch.ones_like(ch, dtype=torch.bool)
            mask_valid &= (ch != self.pad_val)
            if nodata_val is not None:
                # nodata_val is scalar float
                mask_valid &= (ch != float(nodata_val))

            # If no valid pixels, leave channel as-is (preserve pad/nodata)
            if not mask_valid.any():
                features[ch_name] = ch.clone()
                continue

            # Apply desired transform (masked); prefer normalize over standardize if both listed
            if ch_idx in self.transform_normalize:
                ch_out = normalize_minmax_masked(ch, mask_valid)
            elif ch_idx in self.transform_standardize:
                ch_out = standardize_masked(ch, mask_valid)
            else:
                ch_out = ch.clone()

            features[ch_name] = ch_out

        # -----------------------
        # Build label from raw VH (use raw values, not transformed VH)
        # -----------------------
        vh_raw_t = torch.from_numpy(raw_vh).to(dtype=torch.float32)  # (H,W)
        # invalid pixels mask for label
        mask_invalid = (vh_raw_t == float(self.pad_val))
        vh_nodata_val = self.nodata_dict.get("VH", None)
        if vh_nodata_val is not None:
            mask_invalid |= (vh_raw_t == float(vh_nodata_val))

        flood_mask = (vh_raw_t < float(self.vh_threshold)).to(dtype=torch.float32)  # 1.0 flood, 0.0 non-flood
        # mark invalid pixels as -1.0
        flood_mask[mask_invalid] = -1.0

        pw_mask = arr[-1] # permanent water mask

        return features, pw_mask, flood_mask 
