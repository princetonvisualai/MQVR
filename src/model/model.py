import torch

import src.model.text_head_models as text_head_models
import src.model.base_models as base_models
from ..base import BaseModel



class VideoTextFeatureExtractor(BaseModel):
    def __init__(self, base_setting, text_head_setting, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.base = getattr(base_models, base_setting['type'])(device=device, **base_setting['args'])
        self.text_head = getattr(text_head_models, text_head_setting['type'])(device=device, **text_head_setting['args'])
        
    def forward(self, frames, captions):
        frame_features, text_features = self.base.forward(frames, captions)
        text_features = self.text_head(text_features)
        return frame_features, text_features


