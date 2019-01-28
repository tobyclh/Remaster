from argparse import ArgumentParser
from skimage.io import imsave
from skimage.transform import rescale
import skvideo.io

from pathlib import Path
from tqdm import tqdm
import skvideo.io
import skvideo.datasets
import json
parser = ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('--output_folder', type=str, default='outputs')
parser.add_argument('--FPS', type=int, default=1) 

opts = parser.parse_args()
output_path = Path(opts.output_folder)
# assert output_path.is_dir()
if not output_path.exists():
    output_path.mkdir()
print(f'Input : {opts.input}')
n_frames, length = skvideo.io.ffprobe(opts.input)['video']['@avg_frame_rate'].split('/')
avg_fps = float(n_frames) / float(length)
assert avg_fps > opts.FPS, f'Video FPS is only {avg_fps}, cannot provide {opts.FPS} as specified'
extract_every = int(avg_fps / opts.FPS) # extract one frame every
print(f'avg_fps : {avg_fps}, extract every : {extract_every}')

saved_count = 0
reader = skvideo.io.FFmpegReader(opts.input)
# iterate through the frames
accumulation = 0
for i, frame in enumerate(tqdm(reader.nextFrame())):
    if not i % extract_every == 0:
        continue
    imsave(output_path/f'{saved_count}.jpg', frame)
    saved_count += 1