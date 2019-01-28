import argparse
from pytube import YouTube
from pathlib import Path
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('list')
parser.add_argument('--folder')
# parser.add_argument('--itag')


opt = parser.parse_args()
with open(opt.list, mode='r') as f:
    lines = f.readlines()
output_path = Path(opt.folder)
if not output_path.exists():
    output_path.mkdir()
for line in tqdm(lines):
    url = line.rstrip()
    yt = YouTube(url)
    yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()