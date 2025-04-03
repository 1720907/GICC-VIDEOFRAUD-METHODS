import os
import cv2
import numpy as np
import argparse
import time
from model_3_utils import resize_frames, generate_frame_batches, generate_label_batches, compute_correlation_coefficients, compute_correlation_differences, label_diff
from os.path import isfile, join
from os import listdir
import pandas as pd

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-fd','--forgery_data', type=str, default='/mnt/c/users/Jorge-PC/desktop/forged', help='Path of videos')
  parser.add_argument('-fl','--forgery_labels', type=str, default='/mnt/c/users/Jorge-PC/desktop/labels', help='Path of labels')
  parser.add_argument('-db','--data_batches', type=str, default='./data_batches', help='Path to save preprocessed videos')
  parser.add_argument('-lb','--label_batches', type=str, default='./label_batches', help='Path to save label batches')
  parser.add_argument('-kar','--keep_aspect_ratio', type=bool, default=1, help='Whether keep aspect ratio or not when resizing videos')
  return parser.parse_args()

def run():
  args = parse_args()
  #definicion de argumentos
  video_dir = args.forgery_data
  labels_path = args.forgery_labels
  data_batches_path = args.data_batches
  label_batches_path = args.label_batches
  keep_aspect_ratio = args.keep_aspect_ratio

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
          frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          frames.append(frame_gray)
      video_frames = np.array(frames)
      cap.release()
      # Resize
      processed_frames = resize_frames(video_frames, keep_aspect_ratio, (250,250))
      processed_frames = np.array(processed_frames)

      # Normalize frames
      processed_frames = processed_frames.astype('float32')/255
    
      # Read video label
      path = os.path.join(labels_path,input_labels[current_video_index])
      label = np.load(path)

      print(f"Label name: {input_labels[current_video_index]} \n")

      # Generate video frame batches
      video_frame_batches = generate_frame_batches(processed_frames, batch_size=100)
      video_frame_batches = np.array(video_frame_batches)

      # Generate video label batches
      video_label_batches = generate_label_batches(label,batch_size=100)
      video_label_batches = np.array(video_label_batches)
      Ytrain = []
      names = []
      for batch in video_label_batches:
        isForged = any(forged == 1 for forged in batch)
        if isForged:
          Ytrain.append(1)
        else:
          Ytrain.append(0)
      # Adaptar Ytrain
      Ytrain = label_diff(Ytrain)
      # Second tranformation
      Ytrain = label_diff(Ytrain)
      for i in range(len(Ytrain)):
         names.append(video[:-4])

      # Compute correlation differences
      corr_differences_list = []
      for batch in video_frame_batches:
         corr_coefficients = compute_correlation_coefficients(batch)
         corr_differences = compute_correlation_differences(corr_coefficients)
         corr_differences = np.array(corr_differences)
         corr_differences = corr_differences.reshape(1,98)
         corr_differences_list.append(corr_differences)
      corr_differences_list = np.array(corr_differences_list)

      """
      # Add indexes in total_list
      for k in range(len(video_frame_batches)):
        total_list.append([current_video_index,k])
      """
      data = {'name': names, 'labels': Ytrain}
      video_df = pd.DataFrame(data)
      labels = pd.concat([labels,video_df],ignore_index=True)


      # Save array of frames
      filename = os.path.splitext(video)[0]
      path = os.path.join(data_batches_path,f'{filename}.npy')
      np.save(path,corr_differences_list)

      """
      # Save labels
      labels = np.append(labels, Ytrain)
      """

      # Update current_video_index to the next iteration
      current_video_index+=1
      print((current_video_index*100)//len(input_videos))

  """
  # Save labels
  path = os.path.join(label_batches_path, 'labels.npy')
  np.save(path,labels)

  # Saving indexes
  total_list=np.array(total_list)
  path = os.path.join(indexes_path,'indexes.npy')
  np.save(path, total_list)
  """
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