import torch
import einops
import clip

from ..base import BaseModel
from .frozen import *



class CLIP4Clip(BaseModel):
    def __init__(self, clip_model, fp32=False, device=torch.device('cpu')):
        super().__init__()
        if fp32:
            self.model, _ = clip.load(clip_model, device=device, jit=False)
            self.model = self.model.float()
        else:
            self.model, _ = clip.load(clip_model, device=device)

    def forward(self, frames, captions):
        bz = frames.shape[0]
        frames = einops.rearrange(frames, 'b f c h w -> (b f) c h w')
        frame_features = self.model.encode_image(frames)
        text_features = self.model.encode_text(captions)
        
        frame_features = einops.rearrange(frame_features, '(b f) d -> b f d', b=bz)
        text_features = einops.rearrange(text_features, '(b n) d -> b n d', b=bz)
        
        frame_features = frame_features.mean(dim=1)
        return frame_features, text_features