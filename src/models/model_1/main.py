import os
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
import argparse
import time
import tensorflow as tf
from tensorflow.data import Dataset
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from sklearn.metrics import accuracy_score
from utils import resize_frames, frame_difference, generate_frame_batches, generate_label_batches, create_3dcnn_model, split_data_indices

from os.path import isfile, join
from os import listdir

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-d','--data', type=str, default='/mnt/c/users/Jorge-PC/desktop/forged', help='Path of videos')
  parser.add_argument('-l','--labels', type=str, default='/mnt/c/users/Jorge-PC/desktop/labels', help='Path of labels')
  parser.add_argument('-ch','--checkpoint', type=str, default='/mnt/c/users/Jorge-PC/desktop/checkpoint', help='Path to save checkpoint of the model')
  parser.add_argument('-b_s','--batch_size', type=int, default=16, help='Batch size for training the model')
  parser.add_argument('-e','--epochs', type=int, default=20, help='Epochs for training')
  parser.add_argument('-t','--testing', type=float, default=0.1, help='Percentage for testing')
  parser.add_argument('-v','--validation', type=float, default=0.3, help='Percentage for validation')
  parser.add_argument('-s','--sample', type=int, default=None, help='Percentage for validation')
  parser.add_argument("--cuda", action='store_true', help="Use cuda or cpu")
  return parser.parse_args()

def detect_device(args):
  if args.cuda:
    if tf.test.gpu_device_name():
      device = tf.device("GPU")
      print("\nEnabled CUDA: [{} GPU]\n".format(len(tf.config.experimental.list_physical_devices('GPU'))))
    else:
      device = tf.device("CPU")
      print("CUDA not available, Enabling CPU.\n")
  else:
    device = tf.device("CPU")

  return device

def generator(indices, Xtrain, Ytrain):
  for i in indices:
    yield Xtrain[i], Ytrain[i]

def run():
  args = parse_args()
  #definicion de argumentos
  video_dir = args.data
  labels_path = args.labels
  device = detect_device(args)
  batch_s = args.batch_size
  checkpoint_path = args.checkpoint
  testing= args.testing
  validation = args.validation
  epochs=args.epochs

  # Create empty arrays to store frames, video names, and video indices
  video_frames_array = []

  # Ordering videos
  input_videos = [f for f in listdir(video_dir) if isfile(join(video_dir, f))]
  input_videos = sorted(input_videos)

  # Loop over each video file
  if args.sample is not None:
    total_sample = args.sample
    count = 0
  for video in input_videos:
  # for i, video_file in enumerate(video_files):
      if args.sample is not None:
        count+=1
        if count>total_sample:
          count=0
          break
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
      # Add the video frames, name, and index to the corresponding arrays
      video_frames_array.append(video_frames)
      cap.release()

  processed_frames = resize_frames(video_frames_array, (64, 64))
  #print("array of names", input_videos)

  labels = []
  
  # ordering labels
  input_labels = [f for f in listdir(labels_path) if isfile(join(labels_path, f))]
  input_labels = sorted(input_labels)

  for video_labels in input_labels:
      if args.sample is not None:
        count+=1
        if count>total_sample:
          total_sample=args.sample
          break
      labels.append(np.load(f"{labels_path}\{video_labels}"))
  # Compute frame differences
  #print("video labels: ", input_labels)
  difference_frames, Y = frame_difference(processed_frames, labels)

  for video in difference_frames:
    video=video.astype('float32')/255

  Xtrain= np.array(generate_frame_batches(difference_frames, batch_size=36))
  Ybatches=np.array(generate_label_batches(Y,batch_size=36))
  Ytrain = []
  for batch in Ybatches:
    isForged = any(forged == 1 for forged in batch)
    if isForged:
      Ytrain.append(1)
    else:
      Ytrain.append(0)

  Ytrain = np.array(Ytrain)
  


  train_indices, val_indices, test_indices = split_data_indices(len(Xtrain),testing, validation, random_seed=42)
  model = create_3dcnn_model()

  # Setup data generators or tf.data.Dataset here for both training and validation
  train_dataset = Dataset.from_generator(lambda: generator(train_indices,Xtrain, Ytrain),
                                          output_types=(tf.float32, tf.int32), output_shapes=([36, 64, 64, 3],[])).batch(batch_s)
  val_dataset = Dataset.from_generator(lambda: generator(val_indices, Xtrain, Ytrain),
                                              output_types=(tf.float32, tf.int32),output_shapes=([36, 64, 64, 3],[])).batch(batch_s)
  test_dataset = Dataset.from_generator(lambda: generator(test_indices, Xtrain, Ytrain),
                                              output_types=(tf.float32, tf.int32),output_shapes=([36, 64, 64, 3],[])).batch(batch_s)
                                                

  # Training
  with device:
    checkpoint = ModelCheckpoint(checkpoint_path+f"/model.hdf5",monitor = 'val_accuracy', save_best_only = True, mode='max', verbose=0)
    # early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, mode='max', restore_best_weights=True)

    hist = model.fit(train_dataset, epochs=epochs, callbacks = [checkpoint], validation_data=val_dataset) 

  # Testing
  model = load_model(checkpoint_path+f"/model.hdf5")
  # model = load_model(r"C:\Users\Jorge-PC\Proyecto\GICC_VideoFraud_Prototype\prototype_1\checkpoint\best\model_0.67.hdf5")
  with device:
    y_pred = model.predict(test_dataset)
  y_pred = y_pred.reshape((-1))

  # replacing values by one or zero
  for i in range(y_pred.shape[0]):
    if y_pred[i]>0.5:
      y_pred[i] = 1
    else:
      y_pred[i] = 0

  Ytest = []

  for _, etiquetas in test_dataset:
    Ytest.append(etiquetas.numpy())
  Ytest = np.concatenate(Ytest)

  acc = accuracy_score(Ytest, y_pred)

  print("Accuracy test: ", acc)

if __name__ == '__main__':
  inicio = time.time()
  run()
  fin = time.time()
  execution_time = fin - inicio
  hours = int(execution_time // 3600)
  minutes = int((execution_time % 3600) // 60)
  seconds = int(execution_time % 60)

  print(f"Time of execution: {hours}H:{minutes}M:{seconds}S")
