from torch.utils.data import RandomSampler, BatchSampler, Sampler
import numpy as np
from random import shuffle

class VideoSampler(Sampler):
    """a batch sampler for sampling multiple video frames"""
    def __init__(self, dataset, batchSize, shuffle=True):
        self.dataset = dataset
        self.num_videos = len(self.dataset.file_list)
        self.batch_size = batchSize
        # self.available_videos = [i for i in range(self.num_videos)]

    def __iter__(self):
        while True:
            video_idx = np.random.randint(0, high=self.num_videos)
            n_frames = self.dataset.video_lengths[video_idx]
            batch = []
            for _ in range(self.batch_size):
                max_idx = n_frames - self.dataset.video_fps * 2 * 60
                min_idx = self.dataset.video_fps * 2 * 60
                frame_idx = np.random.randint(min_idx, max_idx)
                batch.append((video_idx, frame_idx))
            yield batch

    def __len__(self):
        return self.dataset.total_num_frames
        
class TwoVideoSampler(Sampler):
    """a batch sampler for sampling multiple video frames"""
    def __init__(self, dataset_a, dataset_b, batchSize, shuffle=True):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.num_videos = max(len(self.dataset_a.file_list), len(self.dataset_b.file_list))
        self.batch_size = batchSize

    def __iter__(self):
        while True:
            video_idx = np.random.randint(0, high=self.num_videos)
            A_video_idx = video_idx % self.dataset_a.num_videos
            B_video_idx = video_idx % self.dataset_b.num_videos
            max_frame_idx = max(self.dataset_a.video_lengths[A_video_idx], self.dataset_b.video_lengths[B_video_idx])
            max_frame_idx -= max(self.dataset_a.video_fps[A_video_idx], self.dataset_b.video_fps[B_video_idx]) * 4 * 60 # ditch the first and last 2 minutes
            batch = []
            for _ in range(self.batch_size):
                frame_idx = np.random.randint(0, high=max_frame_idx)
                A_frame_idx = frame_idx % (self.dataset_a.video_lengths[A_video_idx] - self.dataset_a.video_fps[A_video_idx] * 4 * 60)
                B_frame_idx = frame_idx % (self.dataset_b.video_lengths[B_video_idx] - self.dataset_b.video_fps[B_video_idx] * 4 * 60)
                A_frame_idx = int(A_frame_idx + self.dataset_a.video_fps[A_video_idx] * 2 * 60)
                B_frame_idx = int(B_frame_idx + self.dataset_b.video_fps[B_video_idx] * 2 * 60)
                # assert A_frame_idx < self.dataset_a.video_lengths[A_video_idx], f'{A_frame_idx} < {self.dataset_a.video_lengths[A_video_idx]}'
                # assert B_frame_idx < self.dataset_b.video_lengths[B_video_idx], f'{B_frame_idx} < {self.dataset_b.video_lengths[B_video_idx]}'
                batch.append((A_video_idx, A_frame_idx, B_video_idx, B_frame_idx))
            yield batch

    def __len__(self):
        return self.dataset.total_num_frames
        