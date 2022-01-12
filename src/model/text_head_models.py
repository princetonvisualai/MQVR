import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel



class FCNet(BaseModel):
    def __init__(self, dim_list, last_nonlinear=False, layer_norm=False):
        super().__init__()
        
        if len(dim_list) >= 2:
            layers = []
            for i in range(len(dim_list) - 1):
                layers.append(nn.Linear(dim_list[i], dim_list[i+1]))
                if layer_norm:
                    layers.append(nn.LayerNorm(dim_list[i+1]))
                layers.append(nn.ReLU())

            if not last_nonlinear:
                layers = layers[:-1]
                if layer_norm:
                    layers = layers[:-1]

            self.model = nn.Sequential(*layers)
        else:
            self.model = nn.Identity()

    def forward(self, x):
        return self.model(x)


class MeanHead(BaseModel):
    def __init__(self, device=torch.device('cpu')):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=1)


class TextSimWeightedHead(BaseModel):
    def __init__(self, temperature=1., device=torch.device('cpu')):
        super().__init__()
        self.temperature = temperature

    def forward(self, text_features):
        norm_text_features = F.normalize(text_features, dim=-1)
        if text_features.shape[1] > 1:
            text_sim_mat = torch.bmm(norm_text_features, norm_text_features.transpose(1,2))
            weight = (text_sim_mat.sum(dim=-1, keepdim=True) - 1) / (text_sim_mat.shape[-1] - 1)
            weight = F.softmax(weight / self.temperature, dim=1)
            weighted_text_features = text_features * weight
        else:
            weighted_text_features = text_features
        
        return weighted_text_features.mean(dim=1)

    def gen_weights(self, text_features):
        if text_features.shape[1] > 1:
            norm_text_features = F.normalize(text_features, dim=-1)
            text_sim_mat = torch.bmm(norm_text_features, norm_text_features.transpose(1,2))
            weight = (text_sim_mat.sum(dim=-1, keepdim=True) - 1) / (text_sim_mat.shape[-1] - 1)
            weight = F.softmax(weight / self.temperature, dim=1)
        
        return weight


class LocalizedWeightedHead(BaseModel):
    def __init__(self, fc_dim_list, temperature=1., device=torch.device('cpu'),
                    layer_norm=False):
        super().__init__()
        self.fc = FCNet(fc_dim_list, last_nonlinear=False, layer_norm=layer_norm)
        self.temperature = temperature

    def forward(self, text_features):
        weight = self.fc(text_features)
        weight = F.softmax(weight / self.temperature, dim=1)
        weighted_text_features = text_features * weight
        return weighted_text_features.mean(dim=1)

    def gen_weights(self, text_features):
        weight = self.fc(text_features)
        weight = F.softmax(weight / self.temperature, dim=1)
        return weight


class ContextualizedWeightedHead(BaseModel):
    def __init__(self, d_model, nhead, num_layers, fc_dim_list, temperature=1., 
                    device=torch.device('cpu'), layer_norm=False):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = FCNet(fc_dim_list, last_nonlinear=False, layer_norm=layer_norm)
        self.temperature = temperature
        
    def forward(self, text_features):
        attention_out = self.attention(text_features)
        weight = self.head(attention_out)
        weight = F.softmax(weight / self.temperature, dim=1)
        weighted_text_features = text_features * weight
        return weighted_text_features.mean(dim=1)

    def gen_weights(self, text_features):
        attention_out = self.attention(text_features)
        weight = self.head(attention_out)
        weight = F.softmax(weight / self.temperature, dim=1)
        return weight

