#Global imports
import cv2
import sys
import os
import numpy as np
import csv
from os.path import isfile, join, dirname, abspath, exists
from os import listdir
import argparse
import time

#For P1
from keras.models import load_model

#For P2
import tensorflow as tf
import time
from scipy.stats import pearsonr
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

sys.path.append(abspath(join(dirname(__file__), ".."))) # recognize s modeule utilities
from models.model_2.models_registry import VALID_MODELS
from models.model_2.utils import original_cnn, pretrained_cnn

#Function to detect the device (GPU or CPU)
def detect_device(args):
  use_gpu = args.cuda and tf.test.gpu_device_name()
  device = tf.device("GPU" if use_gpu else "CPU")

  if use_gpu:
      num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
      print(f"\nEnabled CUDA: [{num_gpus} GPU]\n")
  else:
      print("CUDA not available or not requested, Enabling CPU.\n")

  return device


#function defining arguments
def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-m','--model_name', type=str,default='',help='Name of the model')
  parser.add_argument('-vp','--video_path', type=str,default='',help='Path of videos to predict')
  parser.add_argument("--cuda", action='store_true', help="Use cuda or cpu")
  return parser.parse_args()

#functions for P1
def frame_difference_single(processed_frames):
  differences_video = []
  for j in range(len(processed_frames)-1):
    diff = cv2.absdiff(processed_frames[j], processed_frames[j+1])
    differences_video.append(diff)
  differences_video = np.array(differences_video)
  return differences_video
def generate_frame_batches_single(difference_frames, batch_size=36):
  batches = []
  n_frames = len(difference_frames)
  for start in range(0, n_frames, batch_size):
    end = start + batch_size
    # Ensure the batch has the required batch_size, otherwise fill with zeros
    if end <= n_frames:
      batch = difference_frames[start:end]
    else:
      # If there are not enough frames left for a full batch, fill the remainder with zeros
      remainder = end - n_frames
      batch = np.vstack((difference_frames[start:n_frames], np.zeros((remainder, difference_frames.shape[1], difference_frames.shape[2], difference_frames.shape[3]), dtype=difference_frames.dtype)))
    batches.append(batch)
  return batches

#function to preprocess videos
def preprocess_video(video_filename, model_name):

  cap = cv2.VideoCapture(video_filename)
  
  if not cap.isOpened():
      print(f"Error: Could not open video file {video_filename}")
      return None
  
  fps = cap.get(cv2.CAP_PROP_FPS)
  frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
  duration = frame_count / fps
  frames = []
  
  if model_name == "model2_original":
    while True:
      ret, frame = cap.read()
      if not ret:
          break
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # to grayscale
      resized_frame = cv2.resize(gray_frame, (250,250)) # resize the frame
      frames.append(resized_frame)
    cap.release() # release the video capture object
    frames_array = np.array(frames)
  elif model_name in VALID_MODELS:
    i = 0
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      resized_frame = cv2.resize(frame,(128,128))
      frames.append(resized_frame)
      i+=1
    cap.release()
    frames_array = np.array(frames)
  elif model_name == "model1":
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64,64), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    video_frames = np.array(frames)
    cap.release()
    #difference
    difference_frames = frame_difference_single(video_frames)
    #normalization
    difference_frames = difference_frames.astype('float32')/255
    #batches of 36 frames
    video_frame_batches = generate_frame_batches_single(difference_frames, batch_size = 36)
    video_frame_batches = np.array(video_frame_batches)
    frames_array = video_frame_batches
  else:
    raise ValueError(f"❌ Error: '{model_name}' is not a valid model. Choose from {list(VALID_MODELS.keys())}, 'model2_original' or 'model1'.")
  return frames_array, duration

def run():
  args = parse_args()

  #define args
  model_name = args.model_name
  video_path = args.video_path
  device = detect_device(args)
  inference_time_arr = []
  videonames_arr = []
  video_durations_arr=[]

  #Read the videos
  input_videos = [f for f in listdir(video_path) if isfile(join(video_path,f))]
  input_videos = sorted(input_videos)
  
  # Calculate the inference time according to the model
  if model_name == "model1":
    model_path = abspath(join(dirname(__file__), "..","..", "models", "model1_predictions", "D1+D2+D3", "model_fold_5.hdf5"))
    model = load_model(model_path, compile=False)
    #---------------warm up--------------
    with device:
      dummy_input = tf.random.uniform(shape = (1, 36, 64, 64, 3),dtype = tf.float32)
      _ = model(dummy_input, training=False)
    #---------------end of warm up-------
    for video in input_videos:
      video_filename = join(video_path, video)
      
      video_frame_batches, duration = preprocess_video(video_filename, model_name)
      
      # Calculate the inference time
      with device:

        begin = time.time()
        _ = model(video_frame_batches, training = False) # predict the video
        end = time.time()
      inference_time = end - begin
      print(f"Inference time for {video}: {inference_time:.2f} seconds")
      inference_time_arr.append(inference_time)
      videonames_arr.append(video)
      video_durations_arr.append(duration)

  elif model_name == "model2_original" or model_name in VALID_MODELS:
    # Define the model based on the model name
    if model_name == "model2_original":
      model = original_cnn((250,250,1), 40)
      dummy_input = tf.random.uniform(shape=(1,250,250,1), dtype=tf.float32)
    else:
      model = pretrained_cnn((128,128,3),model_name)
      dummy_input = tf.random.uniform(shape=(1,128,128,3), dtype=tf.float32)
    #--------warm up-----------------
    _ = model(dummy_input, training=False)
    #--------end of warm up----------

    for video in input_videos:
      video_filename = join(video_path, video)
      frames_array, duration = preprocess_video(video_filename, model_name)
      # Calculate the inference time
      with device:
        begin = time.time()
        model.predict(frames_array, verbose=0) # extract features
        end = time.time()
        inference_time = end - begin
        print(f"Inference time for {video}: {inference_time:.2f} seconds")
        inference_time_arr.append(inference_time)
        videonames_arr.append(video)
        video_durations_arr.append(duration)

  else:
    raise ValueError(f"❌ Error: '{model_name}' is not a valid model. Choose from {list(VALID_MODELS.keys())}, 'model2_original' or 'model1'.")

  ##  Calculate the average inference time
  # avg_inference_time = np.mean(inference_time_arr)
  # print(f"Average inference time of model {model_name}: {avg_inference_time:.2f} seconds")

  #Save in csv
  csv_path = join(dirname(__file__), "..", "..", "reports", "inference_time", "inference_time.csv")
  write_header = not exists(csv_path)

  with open(csv_path, mode = "a", newline = '') as csv_file:
    writer = csv.writer(csv_file)
    if write_header:
      writer.writerow(['model', 'video_name','video_duration(sec)', 'inference_time(sec)'])
    for i in range(len(inference_time_arr)):
      writer.writerow([model_name, videonames_arr[i], f"{video_durations_arr[i]:.4f}",f"{inference_time_arr[i]:.4f}"])
  
if __name__ == '__main__':
  inicio = time.time()
  run()
  fin = time.time()
  execution_time = fin-inicio
  hours = int(execution_time // 3600)
  minutes = int((execution_time % 3600) // 60)
  seconds = int(execution_time % 60)
  print(f"Time of execution: {hours}H:{minutes}M:{seconds}S")



