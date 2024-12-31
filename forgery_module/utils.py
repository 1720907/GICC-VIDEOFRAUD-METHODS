import numpy as np
import cv2
# Shuffle the list of video names
def shuffle_videos(videos):
  np.random.shuffle(videos)
  n_videos = int(len(videos) // 2)
  # Split the shuffled list into two parts: manipulated and original
  forged_videos = videos[:n_videos]
  original_videos = videos[n_videos:]
  return forged_videos, original_videos

def resize_frames_single(list_frames, target_size=(64,64)):
  resized_frames = [cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA) for frame in list_frames]
  return resized_frames