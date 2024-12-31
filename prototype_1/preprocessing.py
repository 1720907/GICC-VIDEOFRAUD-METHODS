import os
import cv2
import numpy as np
import argparse
import time
from utils import resize_frames_single, frame_difference_single, generate_frame_batches_single, generate_label_batches_single
from os.path import isfile, join
from os import listdir
import pandas as pd

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-fd','--forgery_data', type=str, default='/mnt/c/users/Jorge-PC/desktop/forged', help='Path of videos')
  parser.add_argument('-fl','--forgery_labels', type=str, default='/mnt/c/users/Jorge-PC/desktop/labels', help='Path of labels')
  parser.add_argument('-db','--data_batches', type=str, default='./data_batches', help='Path to save preprocessed videos')
  parser.add_argument('-lb','--label_batches', type=str, default='./label_batches', help='Path to save label batches')
  return parser.parse_args()

def run():
  args = parse_args()
  video_dir = args.forgery_data
  labels_path = args.forgery_labels
  data_batches_path = args.data_batches
  label_batches_path = args.label_batches

  # Ordering videos
  input_videos = [f for f in listdir(video_dir) if isfile(join(video_dir, f))]
  input_videos = sorted(input_videos)

  
  # ordering labels
  input_labels = [f for f in listdir(labels_path) if isfile(join(labels_path, f))]
  input_labels = sorted(input_labels)
  
  # index to save batch
  current_video_index = 0
  #total list
  """
  total_list=[]
  labels = np.array([],dtype='uint8')
  """
  labels = pd.DataFrame(columns=['name','labels'])
  # Loop over each video file
  for video in input_videos:
  # for i, video_file in enumerate(video_files):
      # Read video frames
      print(f"Batching video: {video}...")
      video_path = os.path.join(video_dir, video)
      cap = cv2.VideoCapture(video_path)
      if not cap.isOpened():
          print(f"Error opening video file: {video_path}")
          continue

      fps = cap.get(cv2.CAP_PROP_FPS)
      frames = []
      # Loop over the frames in the video
      while True:
          ret, frame = cap.read()
          if not ret:
              break
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frames.append(frame_rgb)
      video_frames = np.array(frames)
      cap.release()
      # Resize
      processed_frames = resize_frames_single(video_frames, (64,64))
    
      # Read video label
      path = os.path.join(labels_path,input_labels[current_video_index])
      label = np.load(path)

      print(f"Label name: {input_labels[current_video_index]} \n")
      # Compute differences
      difference_frames, Y = frame_difference_single(processed_frames, label)

      # Normalization
      difference_frames = difference_frames.astype('float32')/255

      # Generate video frame batches
      video_frame_batches = generate_frame_batches_single(difference_frames, batch_size=36)
      video_frame_batches = np.array(video_frame_batches)

      # Generate video label batches
      video_label_batches = generate_label_batches_single(Y,batch_size=36)
      video_label_batches = np.array(video_label_batches)
      Ytrain = []
      names = []
      for batch in video_label_batches:
        isForged = any(forged == 1 for forged in batch)
        names.append(video[:-4])
        if isForged:
          Ytrain.append(1)
        else:
          Ytrain.append(0)

      data = {'name': names, 'labels': Ytrain}
      video_df = pd.DataFrame(data)
      labels = pd.concat([labels,video_df],ignore_index=True)


      # Save array of frames
      filename = os.path.splitext(video)[0]
      path = os.path.join(data_batches_path,f'{filename}.npy')
      np.save(path,video_frame_batches)


      # Update current_video_index to the next iteration
      current_video_index+=1
      print((current_video_index*100)//len(input_videos))


  #Saving labels
  labels.to_csv(join(label_batches_path,'label_batches.csv'),encoding='utf-8-sig', index=False)

if __name__ == '__main__':
  inicio = time.time()
  run()
  fin = time.time()
  execution_time = fin - inicio
  hours = int(execution_time // 3600)
  minutes = int((execution_time % 3600) // 60)
  seconds = int(execution_time % 60)

  print(f"Time of execution: {hours}H:{minutes}M:{seconds}S")