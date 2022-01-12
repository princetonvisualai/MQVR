import os
import random
from PIL import Image
import glob
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import decord
decord.bridge.set_bridge("torch")



def get_transform_fn(transform_setting):
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                     std=(0.48145466, 0.4578275, 0.40821073))
    
    if transform_setting == 1:
        transform_fn = transforms.Compose([
                                Resize(224, interpolation=BICUBIC),
                                CenterCrop(224),
                                normalize,
                            ])
    elif transform_setting == 2:
        transform_fn = transforms.Compose([
                                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(0.1, 0.1, 0.1),
                                normalize,
                            ])

    return transform_fn


def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    """ From https://github.com/m-bain/frozen-in-time """
    
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs


def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = video_reader.get_batch(frame_idxs)
    return frames, frame_idxs


def read_frames_from_extracted_folder(video_path, total_frames, sample_num, sample='rand', fix_start=None): 
    frame_idxs = sample_frames(sample_num, total_frames, sample=sample, fix_start=fix_start)
    frames = []
    for idx in frame_idxs:
        frames.append(torch.from_numpy(np.asarray(Image.open(os.path.join(video_path, f'{idx}.jpg')))))
    frames = torch.stack(frames)
    return frames, frame_idxs