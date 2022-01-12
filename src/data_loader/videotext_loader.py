import os
import random
import numpy as np
import torch

from src.utils import read_json
from .utils import *



class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, data, video_setting, text_setting, transform_fn=None):
        self.data = data
        self.transform_fn = transform_fn
        self.video_setting = video_setting
        self.text_setting = text_setting
        
    def __getitem__(self, index):
        video_name, total_frames, captions = self.data[index]
        frames, frame_idxs = self._get_video(video_name)
        captions = self._get_text(captions)
        
        return frames, captions
    
    def _get_video(self, name):
        frames, frame_idxs = read_frames_decord(os.path.join(self.video_setting['path'], name),
                                                self.video_setting['num_frames'],
                                                self.video_setting['sample'],
                                                self.video_setting['fix_start'])
        if self.transform_fn is not None:
            frames = self.transform_fn(frames.float().div_(255).permute(0, 3, 1, 2))
            
        return frames, frame_idxs
    
    def _get_text(self, captions):
        if self.text_setting['sample'] == 'random':
            sampled_captions = [random.choice(captions)]
        elif self.text_setting['sample'] == 'all':
            sampled_captions = captions
        elif self.text_setting['sample'].startswith('mt-rand'):
            sampled_captions = random.sample(captions, 
                                    int(self.text_setting['sample'].split('_')[1]))

        return sampled_captions
    
    def __len__(self):
        return len(self.data)


def videotext_collate_fn(batch_data):
    frames = torch.stack([item[0] for item in batch_data])
    captions = [cap for item in batch_data for cap in item[1]]
    return frames, captions


def VideoTextLoader(data_path, video_setting, text_setting,
                    batch_size, transform_setting=1, num_workers=1, shuffle=True,
                    drop_last=True):

    data = read_json(data_path)
    transform_fn = get_transform_fn(transform_setting)
    dataset = VideoTextDataset(data, video_setting, text_setting, transform_fn)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                         shuffle=shuffle, collate_fn=videotext_collate_fn, 
                                         drop_last=drop_last)
    return loader



