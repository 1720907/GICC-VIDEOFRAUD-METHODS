import cv2
import numpy as np
import time
import argparse
import os
from os import listdir
from os.path import isfile, join
import pandas as pd

from utils import shuffle_videos

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-d','--data', type=str, default='', help='Path for original videos')
  parser.add_argument('-f','--forgery_data', type=str, default='', help='Path for forged videos')
  parser.add_argument('-l','--forgery_labels', type=str, default='', help='Path for labels')
  parser.add_argument('-fd','--forgery_info', type=str, default='', help='Path for saving forgery info in csv format')
  parser.add_argument('-p','--pm_deletion', type=float, default=0.1, help='Percentage maximum of deletion')
  parser.add_argument('-s', '--random_seed', type=int, default=30, help='Random seed for shuffling videos before forgery')
  parser.add_argument('-pt','--path_type', type=str, default='/', help='Path type for linux or windows system')
  # limit of deletion: 100 frames
  return parser.parse_args()

# Calculate lenght deletion for a video
def calculate_deletion_len(n_frames, pmd):
  max_deletion = int(n_frames*pmd)
  min_deletion = 10
  deletion_len = min_deletion
  if max_deletion > 10:
    deletion_len = np.random.randint(min_deletion, max_deletion+1)
  return deletion_len

def frame_original(video_path, output_path):
  
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print("Error: Could not open video.")
    return
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps_video = cap.get(cv2.CAP_PROP_FPS)
  n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  i=0
  n_error=0
  while True:
    ret, frame = cap.read()
    if not ret:
      if(i < n_frames):
        if(n_error == 1):
          break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        if(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != i):
          break
        n_error = 1
        continue
      else:
        break
    if(n_error == 1):
      n_error = 0
    i+=1
  cap.release()
  total_accesible_frames=i

  codec = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, codec, fps_video, (width, height))
  #deletion logic

  cap = cv2.VideoCapture(video_path)
  current_frame = 0
  n_error=0
  while True:
    ret, frame = cap.read()
    if not ret:
      if(current_frame<n_frames):
        if(n_error==1):
          break
        cap.set(cv2.CAP_PROP_POS_FRAMES,current_frame)
        if(int(cap.get(cv2.CAP_PROP_POS_FRAMES))!=current_frame):
          break
        n_error=1
        continue
      else:
        break
    out.write(frame)
    if(n_error==1):
      n_error=0
    current_frame += 1
  # Release resources
  cap.release()
  out.release()
  print("Video processed and saved to", output_path)

  return fps_video, n_frames

def frame_deletion(video_path, output_path, pmd):

  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print("Error: Could not open video.")
    return
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps_video = cap.get(cv2.CAP_PROP_FPS)
  n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  # Find accesible frames
  i=0
  n_error=0
  while True:
    ret,frame = cap.read()
    if not ret:
      if(i<n_frames):
        if(n_error==1):
          break
        cap.set(cv2.CAP_PROP_POS_FRAMES,i)
        if(int(cap.get(cv2.CAP_PROP_POS_FRAMES))!=i):
          break
        n_error=1
        continue
      else:
        break
    if(n_error==1):
      n_error=0
    i+=1
  cap.release()
  total_accesible_frames=i

  forgery_tuple = ()
  deletion_len = calculate_deletion_len(total_accesible_frames, pmd)
  #calculating index forgery tuple
  if total_accesible_frames >= 25:
    start_index = np.random.randint(1,total_accesible_frames-deletion_len) 
    end_index = start_index + deletion_len
    forgery_tuple = (start_index, end_index) #(49, 59)

  codec = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, codec, fps_video, (width, height))

  #deletion logic
  cap = cv2.VideoCapture(video_path)
  current_frame = 0
  n_error=0
  while True:
    ret, frame = cap.read()
    if not ret:
      if(current_frame<n_frames):
        if(n_error==1):
          break
        cap.set(cv2.CAP_PROP_POS_FRAMES,current_frame)
        if(int(cap.get(cv2.CAP_PROP_POS_FRAMES))!=current_frame):
          break
        n_error=1
        continue
      else:
        break
    # Save the frame only if it's not within the deletion range
    if not (forgery_tuple and forgery_tuple[0] <= current_frame < forgery_tuple[1]):
      out.write(frame)
    if(n_error==1):
      n_error=0
    current_frame += 1
  # Release resources
  cap.release()
  out.release()
  print("Video processed and saved to", output_path)

  return fps_video, total_accesible_frames, deletion_len, forgery_tuple

def run():
  args = parse_args()
  # Set path
  input_path = args.data
  forged_path = args.forgery_data
  labels_path = args.forgery_labels
  csv_path = args.forgery_info
  path_type = args.path_type
  pmd = args.pm_deletion
  r_seed = args.random_seed

  #For dataframe
  video_names = []
  video_fps = []
  original_frames = []
  deleted_frames = []
  output_frames = []
  sec_deletion = []
  index_deletion = []
  labels=[]

  videos = [f for f in listdir(input_path) if isfile(join(input_path, f))]
  videos = sorted(videos)
  if r_seed is not None:
    np.random.seed(r_seed)
  forged_videos, original_videos = shuffle_videos(videos)
  current_video = 1

  for filename in forged_videos:
    file_ext = filename[-4:]
    if (file_ext==".mp4" or ".avi"):
      input_video = os.path.join(input_path, filename)
      new_vname = f"deleted {filename[:-4]}"
      output_video = os.path.join(forged_path, new_vname+".mp4")
      fps, n_frames, frames_del, forgery_tuple = frame_deletion(input_video, output_video, pmd)
      start, end = (forgery_tuple[0]+1)/fps, (forgery_tuple[1]+1)/fps
      output = n_frames-frames_del

      #Generation of labels
      label = np.zeros(output)
      label[forgery_tuple[0]]=1
      np.save(f'{labels_path}{path_type}{new_vname}.npy', label)

      print(f"Forged video # {current_video}" 
            f"\n fps: {fps}" 
            f"\n original frames: {n_frames}"
            f"\n deleted frames: {frames_del}"
            f"\n output frames: {output}"
            f"\n deletion points (s): ({start, end})"
            f"\n deletion points (i): {forgery_tuple} \n")
      #Appending data to arrays for dataframe construction
      video_names.append(new_vname)
      labels.append(1)
      video_fps.append(fps)
      original_frames.append(n_frames)
      deleted_frames.append(frames_del)
      output_frames.append(output)
      sec_deletion.append((start, end))
      index_deletion.append(forgery_tuple)
    current_video+=1
    print((current_video*100)//len(videos))

  for filename in original_videos:
    file_ext = filename[-4:]
    if (file_ext==".mp4" or ".avi"):
      input_video = os.path.join(input_path, filename)
      output_video = os.path.join(forged_path, filename[:-4] + ".mp4")
      fps, n_frames = frame_original(input_video, output_video)

      #Generation of labels
      label = np.zeros(n_frames)
      np.save(f'{labels_path}{path_type}{filename[:-4]}.npy', label)

      print(f"Original video # {current_video}"
            f"\n fps: {fps}"
            f"\n original frames: {n_frames} \n")

      #Appending data to arrays for dataframe construction
      video_names.append(filename[:-4])
      labels.append(0)
      video_fps.append(fps)
      original_frames.append(n_frames)
      deleted_frames.append("0")
      output_frames.append("-")
      sec_deletion.append("-")
      index_deletion.append("-")
    current_video+=1
    print((current_video*100)//len(videos))

  # Create DataFrame
  data = {
    'name': video_names,
    'labels':labels,
    'fps': video_fps,
    'original_frames': original_frames,
    'deleted_frames': deleted_frames,
    'output_frames': output_frames,
    'deletion_point_s': sec_deletion,
    'deletion_point_i': index_deletion
  }

  df = pd.DataFrame(data)

  # Export DataFrame to CSV
  df.to_csv(f"{csv_path}{path_type}forgery_info.csv",encoding='utf-8-sig', index=False)

if __name__ == '__main__':
  # add timer
  inicio = time.time()
  run()
  fin = time.time()
  execution_time = fin - inicio
  hours = int(execution_time // 3600)
  minutes = int((execution_time % 3600) // 60)
  seconds = int(execution_time % 60)

  print(f"Time of execution: {hours}H:{minutes}M:{seconds}S")