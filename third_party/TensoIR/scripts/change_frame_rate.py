import os
from glob import glob
import cv2
import imageio
import numpy as np
from tqdm import tqdm

input_videos_dir = 'nerfactor_cache4'
output_videos_dir = 'nerfactor_cache4_changed'

if not os.path.exists(output_videos_dir):
    os.mkdir(output_videos_dir)
# recursively find all the videos
input_videos = glob(os.path.join(input_videos_dir, '**', '*.mp4'), recursive=True)
input_videos.sort()

for input_video in tqdm(input_videos):
    # read video
    video = imageio.get_reader(input_video, 'ffmpeg')
    # get fps
    fps = video.get_meta_data()['fps']
    # get video name
    video_name = os.path.basename(input_video)
    # get video name without extension
    video_name = os.path.splitext(video_name)[0]
    # get video frames
    frames = []
    for frame in video:
        frames.append(frame)
    # change fps
    new_fps = 24
    # save to video
    imageio.mimsave(os.path.join(output_videos_dir, video_name + '.mp4'), frames, fps=new_fps)