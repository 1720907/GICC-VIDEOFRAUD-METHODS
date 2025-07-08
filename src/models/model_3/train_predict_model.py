import numpy as np
import argparse
import time
import os
from os import listdir
from os.path import isfile, join
import sys
import tensorflow as tf
from utils import generate_cnn, generator
from tensorflow.data import Dataset
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score
import gc
from sklearn.model_selection import StratifiedKFold
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) # recognizes modeule utilities
from utilities.logging import log_execution_time

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('datasets', metavar='D', type=str, nargs='+', help='Directories containing their each respective batches and label_batches')
  parser.add_argument('-ch','--checkpoint', type=str, default='./checkpoint', help='Path to save checkpoint of the model')
  parser.add_argument("--cuda", action='store_true', help="Use cuda or cpu")
  parser.add_argument("-b_s", "--batch_size", type = int, default = 32,help = "Batch size for training the model")
  parser.add_argument('-e','--epochs', type=int, default=20, help='Epochs for training')
  parser.add_argument('-f', '--folds', type=int, default=10, help="Number of folds.")
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

def run():
  args = parse_args()
  device = detect_device(args)
  batch_s = args.batch_size
  checkpoint_path = args.checkpoint
  epochs = args.epochs
  n_folds = args.folds
  dataset_paths = [(args.datasets[i]) for i in range(len(args.datasets))]

  skf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)

  model = generate_cnn(input_shape=(98,1)) # To 1D

  complete_df_labels = pd.DataFrame(columns=['name','labels', 'dataset'])
  dataset_index=0

  # Get df_labels from label_batches.csv
  for dataset_path in dataset_paths:
    dataframe_path = join(dataset_path, "p3_label_batches", "label_batches.csv") # correct file name
    df = pd.read_csv(dataframe_path, usecols=['name','labels'], dtype={'name':str,'labels':int})
    df['dataset'] = dataset_index
    complete_df_labels = pd.concat([complete_df_labels, df], ignore_index=True)
    dataset_index+=1
  
  complete_df = pd.DataFrame(columns=['name', 'labels', 'dataset'])
  dataset_index = 0
  for dataset_path in dataset_paths:
    dataframe_path = join(dataset_path, "forgery_info", "forgery_info.csv")
    df = pd.read_csv(dataframe_path, usecols=['name', 'labels'], dtype={'name':str,'labels':int})
    df['dataset'] = dataset_index
    complete_df = pd.concat([complete_df, df], ignore_index=True)
    dataset_index+=1
  total_videos = len(complete_df)
  Y = complete_df['labels'].astype(int).to_numpy()
  folds = skf.split(np.arange(total_videos), Y)
  
  for fold, (train_indixes, test_indixes) in enumerate(folds,1):
    print(f"Fold: {fold}")
    Ytest_fold = Y[test_indixes]
    train_dataset = Dataset.from_generator(lambda: generator(train_indixes, dataset_paths, complete_df_labels, complete_df), output_types=(tf.float32, tf.float32), output_shapes=([98, 1], [2])).batch(batch_s) # To 1D
    test_dataset = Dataset.from_generator(lambda: generator(test_indixes, dataset_paths, complete_df_labels, complete_df), output_types=(tf.float32, tf.float32), output_shapes=([98, 1], [2])).batch(batch_s) # To 1D

    #Testing train_dataset
    # for data, label in train_dataset.take(1):
      # print(f"Data shape: {data.shape}, Label shape: {label.shape}, Label: {label}")
    # import keras
    # print(f"Keras version: {keras.__version__}")  
    # Training
    with device:
      checkpoint = ModelCheckpoint(checkpoint_path+f"/model_fold_{fold}.h5",monitor = 'accuracy', save_best_only = True, mode='max', verbose=0) # change of model format
      model.fit(train_dataset, epochs=epochs, callbacks = [checkpoint])

    # Testing
    model = load_model(checkpoint_path+f"/model_fold_{fold}.h5") # change of model format
    with device:
      y_pred = model.predict(test_dataset)
    y_pred = y_pred.reshape((-1))

    # Classifying
    for i in range(y_pred.shape[0]):
      if y_pred[i]>0.5:
        y_pred[i] = 1
      else:
        y_pred[i] = 0
    
    labels = []
    subset_df = complete_df.iloc[test_indixes]
    current=0
    for row in subset_df.itertuples():
      indexes = complete_df.index[complete_df['name']==row.name].to_numpy()
      length = len(indexes)
      if(any(l == 1 for l in y_pred[current:current+length])):
        labels.append(1)
      else:
        labels.append(0)
      current=current+length
    y_pred = np.array(labels)
    subset_df=None    


    acc = accuracy_score(Ytest_fold, y_pred)
    
    print("Accuracy test: ", acc)

    print("Report: \n", classification_report(Ytest_fold, y_pred))
    np.save(checkpoint_path+f"/y_pred_fold_{fold}.npy",y_pred)
    np.save(checkpoint_path+f"/y_test_fold_{fold}.npy", Ytest_fold)
    gc.collect()

if __name__ == '__main__':
  start_time = time.time()
  
  args = run() # ✅ Capture returned args
  
  end_time = time.time()
  log_execution_time("model3", args.datasets, start_time, end_time)