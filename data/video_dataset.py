import numpy as np
from torch.utils.data import Dataset
import torch
from torch import nn
from torch.nn.functional import upsample, interpolate
from torchvision.transforms.functional import resize
from torchvision.transforms import Normalize
import os
from os.path import join, basename, dirname
import sys
import imageio
from glob import glob
from copy import copy
from random import shuffle
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from functools import lru_cache
from tqdm import tqdm
import skvideo
from PIL import Image
class VideoDatasets(Dataset):
    def __init__(self, folder, transform=None, return_paths=False):
        self.file_list = glob(join(folder, '*.mp4'))
        self.num_videos = len(self.file_list)
        self.video_lengths = [self.get_video_length(self.get_video_reader(filename)) for filename in self.file_list]
        self.video_fps = [self.get_fps(self.get_video_reader(filename)) for filename in self.file_list]
        self.total_num_frames = sum(self.video_lengths)
        print(f'Found {self.num_videos} videos in folder with {self.total_num_frames} frames in total')
        self.transform = transform
        self.return_paths = return_paths
        return

    @lru_cache(maxsize=20)
    def get_fps(self, video_reader):
        return video_reader.get_meta_data()['fps']

    def __getitem__(self, idx):
        # assert isinstance(idx, tuple), f'input must be tuple specifying the file and the frame, {idx}'
        if isinstance(idx, tuple):
            file_idx, frame_idx = map(int, idx)
        
        filename = self.file_list[file_idx]
        video_reader = self.get_video_reader(filename)
        frame = self.get_frame(video_reader, frame_idx, file_idx, filename)
        frame = Image.fromarray(frame)

        if self.return_paths:
            return frame, filename + '_' + str(frame_idx)
        return frame

    def get_frame(self, reader, frame_idx, file_idx, filename):
        try:
            return reader.get_data(frame_idx)
        except Exception as e:
            print(f'exception : {file_idx}, {filename}, {frame_idx}, {reader.get_length()}')
            self.get_frame(reader, frame_idx-1, file_idx, filename)


    def get_video_length(self, reader):
        return reader.get_length()
                              
    # @lru_cache(maxsize=20)
    def get_video_reader(self, video_file):
        video_reader = imageio.get_reader(video_file, 'ffmpeg')
        return video_reader

    def __len__(self):
        return self.total_num_frames