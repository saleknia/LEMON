from torch.utils.tensorboard import SummaryWriter
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torchvision
import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, f1_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, valid_loss):
        if valid_loss < self.best_loss - self.min_delta:
            self.best_loss = valid_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

class SpaceToDepthFactor(nn.Module):
    """
    (B, C, H, W) → (B, C * f², H/f, W/f)
    Lossless: folds each f×f spatial block into the channel dim.
    """
    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        f = self.factor
        assert H % f == 0 and W % f == 0
        x = x.reshape(B, C, H//f, f, W//f, f)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.reshape(B, C * f * f, H//f, W//f)
        return x


class FPNLevelProcessor(nn.Module):
    """
    One FPN level (top-down pathway):

        1. Upsample top_down to lateral spatial size
        2. Compress top_down channels to lateral_channels  (1×1 conv)
        3. Add with lateral
        4. Reduce channels by factor f                     (1×1 conv)
        5. Space-to-depth with factor f

    Returns:
        fpn_out: (B, lateral_ch * f,    H//f, W//f)  → unified descriptor (1536×7×7)
        td_out:  (B, lateral_ch // f,   H,    W   )  → passed to next FPN level
                 (pre-s2d, keeps original spatial size for next level's upsample)

    Channel arithmetic (target descriptor = 1536):
        level3: lateral=768,  td_in=1536, f=2 → fpn:(B,1536,7,7)  td:(B,384,14,14)
        level2: lateral=384,  td_in=384,  f=4 → fpn:(B,1536,7,7)  td:(B,96, 28,28)
        level1: lateral=192,  td_in=96,   f=8 → fpn:(B,1536,7,7)  td:(B,24, 56,56) unused
    """
    def __init__(self, lateral_channels: int, top_down_channels: int, factor: int):
        super().__init__()
        self.s2d         = SpaceToDepthFactor(factor)
        self.compress_td = nn.Conv2d(top_down_channels, lateral_channels,          kernel_size=1, bias=False)
        self.reduce      = nn.Conv2d(lateral_channels,  lateral_channels // factor, kernel_size=1, bias=False)

    def forward(self, lateral: torch.Tensor, top_down: torch.Tensor):
        # 1+2: upsample then compress top-down to match lateral channels
        td      = F.interpolate(top_down, size=lateral.shape[-2:], mode='nearest')
        td      = self.compress_td(td)      # (B, lateral_ch, H, W)
        # 3: add
        x       = lateral + td              # (B, lateral_ch, H, W)
        # 4: channel reduction by f
        x       = self.reduce(x)            # (B, lateral_ch//f, H, W)
        # td_out passed to the next FPN level (pre-s2d, original spatial size)
        td_out  = x                         # (B, lateral_ch//f, H, W)
        # 5: space-to-depth — no compress, pure spatial folding
        fpn_out = self.s2d(x)               # (B, lateral_ch*f,  H//f, W//f)
        return fpn_out, td_out


class LearnableWeightFusion(nn.Module):
    """
    Fuses 4 same-shape descriptors with softmax-normalised learnable weights.
    Only 4 extra parameters — validates the pipeline before adaptive fusion.
    """
    def __init__(self, descriptor_dim: int = 1536, nclasses: int = 10):
        super().__init__()
        self.scale_weights = nn.Parameter(torch.ones(4) / 4)
        self.pool          = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier    = nn.Linear(descriptor_dim, nclasses)

    def forward(self, f1, f2, f3, f4):
        w     = torch.softmax(self.scale_weights, dim=0)        # (4,)
        stack = torch.stack([f1, f2, f3, f4], dim=1)           # (B, 4, C, H, W)
        fused = (w.view(1, 4, 1, 1, 1) * stack).sum(dim=1)    # (B, C, H, W)
        out   = self.pool(fused).flatten(1)                     # (B, C)
        return self.classifier(out)

    def get_scale_weights(self):
        return torch.softmax(self.scale_weights, dim=0).detach().cpu()


class MultiScaleConvNeXtLarge(nn.Module):
    """
    Full pipeline:

        ConvNeXt-Large encoder (bottom-up):
            s1: (B, 192,  56, 56)
            s2: (B, 384,  28, 28)
            s3: (B, 768,  14, 14)
            s4: (B, 1536,  7,  7)

        FPN neck (true cascaded top-down):
            f4        = s4
            f3, td3   = level3(s3, f4)    td3: (B, 384,  14, 14)
            f2, td2   = level2(s2, td3)   td2: (B, 96,   28, 28)
            f1, _     = level1(s1, td2)

        All f1..f4: (B, 1536, 7, 7) — unified descriptors

        LearnableWeightFusion → (B, nclasses)
    """
    def __init__(self, nclasses: int = 10):
        super().__init__()

        # encoder split into 4 stages for intermediate feature access
        backbone     = torchvision.models.convnext_large(weights='DEFAULT')
        self.stage1  = backbone.features[:2]    # → (B, 192,  56, 56)
        self.stage2  = backbone.features[2:4]   # → (B, 384,  28, 28)
        self.stage3  = backbone.features[4:6]   # → (B, 768,  14, 14)
        self.stage4  = backbone.features[6:8]   # → (B, 1536,  7,  7)

        # FPN levels — td_in matches td_out of level above
        self.level3 = FPNLevelProcessor(lateral_channels=768, top_down_channels=1536, factor=2)
        self.level2 = FPNLevelProcessor(lateral_channels=384, top_down_channels=384,  factor=4)
        self.level1 = FPNLevelProcessor(lateral_channels=192, top_down_channels=96,   factor=8)

        self.head = LearnableWeightFusion(descriptor_dim=1536, nclasses=nclasses)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bottom-up encoder
        s1 = self.stage1(x)     # (B, 192,  56, 56)
        s2 = self.stage2(s1)    # (B, 384,  28, 28)
        s3 = self.stage3(s2)    # (B, 768,  14, 14)
        s4 = self.stage4(s3)    # (B, 1536,  7,  7)

        # cascaded top-down FPN
        f4          = s4
        f3, td3     = self.level3(s3, f4)   # f3:(B,1536,7,7)  td3:(B,384,14,14)
        f2, td2     = self.level2(s2, td3)  # f2:(B,1536,7,7)  td2:(B,96, 28,28)
        f1, _       = self.level1(s1, td2)  # f1:(B,1536,7,7)

        return self.head(f1, f2, f3, f4)

    def get_scale_weights(self):
        return self.head.get_scale_weights()


class BaselineConvNeXtLarge(nn.Module):
    """Original LEMON setup — exact replica."""
    def __init__(self, nclasses: int = 10):
        super().__init__()
        backbone = torchvision.models.convnext_large(weights='DEFAULT')
        backbone.classifier[2] = nn.Linear(
            backbone.classifier[2].in_features, nclasses
        )
        self.net = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_lemon_weights(net: nn.Module, path: str, multiscale: bool) -> None:
    """Loads LemonFM teacher→backbone weights into net."""
    state_dict = torch.load(path, map_location="cpu")
    if 'teacher' in state_dict:
        state_dict = state_dict['teacher']
    state_dict = {
        k.replace("backbone.", ""): v
        for k, v in state_dict.items()
        if k.startswith('backbone.')
    }

    if not multiscale:
        net.net.load_state_dict(state_dict, strict=False)
    else:
        # keys are "0.*".."7.*" matching features[0]..features[7]
        stage_map = {
            'stage1': ('0.', '1.'),
            'stage2': ('2.', '3.'),
            'stage3': ('4.', '5.'),
            'stage4': ('6.', '7.'),
        }
        for stage_name, prefixes in stage_map.items():
            stage    = getattr(net, stage_name)
            stage_sd = {
                '.'.join(k.split('.')[1:]): v
                for k, v in state_dict.items()
                if any(k.startswith(p) for p in prefixes)
            }
            stage.load_state_dict(stage_sd, strict=False)

    print(f"[INFO] Pretrained weights loaded from {path}")


def build_model(
    nclasses:           int  = 10,
    pretrained_weights: str  = None,
    multiscale:         bool = False,
) -> nn.Module:
    """
    Args:
        nclasses:           Output classes (10 for CholecT50 verbs).
        pretrained_weights: Path to LemonFM .pth checkpoint (optional).
        multiscale:         False → original LEMON baseline.
                            True  → FPN + SpaceToDepth + LearnableWeightFusion.
    Returns:
        nn.Module on CUDA, ready for training.
    """
    if multiscale:
        print("[INFO] Model: MultiScale (FPN → SpaceToDepth → LearnableWeightFusion)")
        net = MultiScaleConvNeXtLarge(nclasses=nclasses)
    else:
        print("[INFO] Model: Baseline (ConvNeXt-Large + GlobalAvgPool)")
        net = BaselineConvNeXtLarge(nclasses=nclasses)

    if pretrained_weights:
        if os.path.isfile(pretrained_weights):
            _load_lemon_weights(net, pretrained_weights, multiscale)
        else:
            print(f"[WARN] Weights not found at {pretrained_weights}, skipping.")

    return net.cuda()

def setup_tensorboard(log_dir):
    return SummaryWriter(log_dir=log_dir)

def calculate_metrics(probabilities, targets):
    predictions = (probabilities > 0.5).float()
    targets_np = targets.detach().cpu().numpy()
    predictions_np = predictions.detach().cpu().numpy()
    probabilities_np = probabilities.detach().cpu().numpy()
    
    mAP = average_precision_score(targets_np, probabilities_np, average='macro')
    f1 = f1_score(targets_np, predictions_np, average='macro', zero_division=0)
    accuracy = accuracy_score(targets_np, predictions_np)
    recall = recall_score(targets_np, predictions_np, average='macro', zero_division=0)
    
    return mAP, f1, accuracy, recall

def valid(net: torch.nn.Module, valid_dl, loss_func, device='cuda'):
    valid_loss = 0
    all_probabilities = []
    all_targets = []

    net.eval()
    with torch.no_grad():
        pbar = tqdm.tqdm(enumerate(valid_dl), total=len(valid_dl))
        
        # Unpack the specific CholecT50 tuple structure to extract 'verbs' (index 2)
        for batch_idx, (inputs, (_, _, targets, _, _)) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = loss_func(outputs, targets)
            valid_loss += loss.item()
            display_loss = valid_loss / (batch_idx + 1)

            probabilities = torch.sigmoid(outputs)
            all_probabilities.append(probabilities)
            all_targets.append(targets)
            
            probabilities_current = torch.cat(all_probabilities)
            targets_current = torch.cat(all_targets)
            
            mAP, f1, accuracy, recall = calculate_metrics(probabilities_current, targets_current)

            pbar.set_description("Validation loss: %.3f | mAP: %.3f%%" % (display_loss, 100. * mAP))
    
    return display_loss, 100. * mAP, 100. * accuracy, 100. * f1, 100. * recall

if __name__ == '__main__':
    raise RuntimeError('[ERROR] This module is not supposed to be run as an executable.')
