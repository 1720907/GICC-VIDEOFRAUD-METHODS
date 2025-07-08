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

# Search for the first occurrence of 1
def find_first_one(array):

    for index, value in enumerate(array):
        if value == 1:
            return index
    return -1

def run():
  args = parse_args()
  
  # Define args
  videos_path = args.forgery_data
  labels_path = args.forgery_labels
  data_batches_path = args.data_batches
  label_batches_path = args.label_batches
  keep_aspect_ratio = args.keep_aspect_ratio

  # Order videos
  input_videos = [f for f in listdir(videos_path) if isfile(join(videos_path, f))] # Create a list of video file names (without directories)
  input_videos = sorted(input_videos) # Sort the list in ascending order
  
  # Order labels
  input_labels = [f for f in listdir(labels_path) if isfile(join(labels_path, f))] # Create a list of labels (without directories)
  input_labels = sorted(input_labels) # Sort the list in ascending order
  
  # Index to save batch
  current_video_index = 0
  
  # Create a dataframe of labels
  labels = pd.DataFrame(columns=['name','labels'])
  
  for video in input_videos:
    
    print(f"Batching video: {video}...\n")
    video_path = os.path.join(videos_path, video)

    cap = cv2.VideoCapture(video_path)
    
    # Validate video availability
    if not cap.isOpened():
      print(f"Error opening video file: {video_path}")
      continue
    
    # Get frames per second
    fps = cap.get(cv2.CAP_PROP_FPS) 
    frames = []
    
    # Looping over the frames in the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Grayscale video frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame_gray)

    video_frames = np.array(frames)
    cap.release()

    # Resize video frames
    processed_frames = resize_frames(video_frames, keep_aspect_ratio, (250,250))
    processed_frames = np.array(processed_frames)

    # Normalize frames
    processed_frames = processed_frames.astype('float32')/255
  
    # Read video label
    path = os.path.join(labels_path, input_labels[current_video_index])
    label = np.load(path)

    # print(f"Label name: {input_labels[current_video_index]} \n")
    # print(f"Label length: {label.shape[0]}, Label: {label} \n")
    
    # Generate video frame batches
    video_frame_batches = generate_frame_batches(processed_frames, batch_size=100)
    video_frame_batches = np.array(video_frame_batches)

    # Generate video label batches
    video_label_batches = generate_label_batches(label, batch_size=100)
    video_label_batches = np.array(video_label_batches)
    y_train = []
    names = []
    
    # print(f"Log:\n Video frame batches length: {video_frame_batches.shape[0]} \n Video label batches length: {video_label_batches.shape[0]}\n")
    

    # # Adaptar y_train
    # # Second tranformation
    # y_train = label_diff(y_train)

    # print(f"Log:\n Video label batches length (y_train): {len(y_train)}\n\n")

    corr_differences_list = []
    labels_differences_list = []
    
    # Compute correlation differences per each batch of video frames
    for i in range(len(video_label_batches)):
      
      # Get video frame batch and labels batch
      batch = video_frame_batches[i]
      label_batch = video_label_batches[i]

      # Print log information before correlation coefficient calculation
      print(f"""\033[1mLog (Before correlation coef. calculation):\033[0m 
            \n Video batch length {len(batch)}...
            \n Labels batch length {len(label_batch)}...
            \n Labels batch {label_batch}...
            \n Position of 1: {find_first_one(label_batch)}...\n""")

      # Compute correlation coefficients
      corr_coefficients = compute_correlation_coefficients(batch)
      label_batch_coef = label_diff(label_batch)
      
      # Print log information after correlation coefficient calculation
      print(f"""\033[1mLog (After correlation coef. calculation):\033[0m
            \n Video batch length {len(corr_coefficients)}...
            \n Labels batch length {len(label_batch_coef)}...\n
            \n Labels batch {label_batch_coef}...
            \n Position of 1: {find_first_one(label_batch_coef)}...\n""")

      # Compute correlation differences
      corr_differences = compute_correlation_differences(corr_coefficients)
      label_batch_diff = label_diff(label_batch_coef)
      
      # Print log information after correlation coefficient difference calculation
      print(f"""\033[1mLog (After correlation coef. difference calculation):\033[0m  
            \n Video batch length {len(corr_differences)}...
            \n Labels batch length {len(label_batch_diff)}...
            \n Labels batch {label_batch_diff}...
            \n Position of 1: {find_first_one(label_batch_diff)}...\n""")

      corr_differences = np.array(corr_differences)
      corr_differences = corr_differences.reshape(1, 98)

      # Append correlation differences and labels to lists
      corr_differences_list.append(corr_differences)
      labels_differences_list.append(label_batch_diff)

    corr_differences_list = np.array(corr_differences_list)

    # Label per each batch
    for batch in labels_differences_list:
      is_forged = any(forged == 1 for forged in batch)
      if is_forged:
        y_train.append(1)
      else:
        y_train.append(0)
    
    for _ in range(len(y_train)):
        names.append(video[:-4])

    # Add labeled batches of a video to df "labels"
    data = {'name': names, 'labels': y_train}
    video_df = pd.DataFrame(data)
    labels = pd.concat([labels, video_df], ignore_index=True)

    # Save to path array of correlation differences list
    filename = os.path.splitext(video)[0]
    path = os.path.join(data_batches_path, f'{filename}.npy')
    np.save(path, corr_differences_list)

    # Update current_video_index to the next iteration
    current_video_index+=1
    # print(f"Current video: {(current_video_index*100)//len(input_videos)}")

  """
  # Save labels
  path = os.path.join(label_batches_path, 'labels.npy')
  np.save(path,labels)

  # Saving indexes
  total_list=np.array(total_list)
  path = os.path.join(indexes_path,'indexes.npy')
  np.save(path, total_list)
  """
  #Save labels
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