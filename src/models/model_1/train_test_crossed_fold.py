import os
import sys
import cv2
import numpy as np
import argparse
import time
import tensorflow as tf
from tensorflow.data import Dataset
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from sklearn.metrics import accuracy_score
from utils import create_3dcnn_model
from sklearn.metrics import classification_report
import pandas as pd
import gc
from sklearn.model_selection import StratifiedKFold
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) # recognizes modeule utilities
from utilities.logging import log_execution_time
from os.path import join

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('datasets', metavar='D', type=str, nargs='+', help='Directories containing their each respective data_batches and label_batches')
  parser.add_argument('-ch','--checkpoint', type=str, default='./checkpoint', help='Path to save checkpoint of the model')
  parser.add_argument('-b_s','--batch_size', type=int, default=32, help='Batch size for training the model')
  parser.add_argument('-e','--epochs', type=int, default=20, help='Epochs for training')
  parser.add_argument('-f', '--folds', type=int, default=10, help="Number of folds.")   
  parser.add_argument("--cuda", action='store_true', help="Use cuda or cpu")
  parser.add_argument("--batch", action='store_true', help="Use batch videos or not")
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

def generator(set_indexes, dataset_paths, complete_df_labels):
  subset_df = complete_df_labels.iloc[set_indexes]
  for row in subset_df.itertuples():
    path = join(dataset_paths[row.dataset],"p1_data_batches",f"{row.name}.npy")
    data = np.load(path)
    yield data, row.labels

def generator2(set_indexes, dataset_paths, complete_df_labels, complete_df):
  subset_df = complete_df.iloc[set_indexes]
  for row in subset_df.itertuples():
    labels = complete_df_labels.loc[complete_df_labels['name']==row.name, 'labels'].astype(int).to_numpy()
    path = join(dataset_paths[row.dataset],"p1_data_batches",f"{row.name}.npy")
    if(row.name[-4:]=='.mp4'):
      raise Exception(f'Nombre archivo: {row.name}')
    data = np.load(path)
    print(f"Data shape: {data.shape}, Data length: {len(data)}, Labels length: {len(labels)}") # compare data length with labels length
    for i in range(len(data)):
      yield data[i], labels[i]

def run():
  args = parse_args()
  #definicion de argumentos
  device = detect_device(args)
  batch_s = args.batch_size
  checkpoint_path = args.checkpoint
  epochs=args.epochs
  n_folds = args.folds
  batch = args.batch
  dataset_paths = [(args.datasets[i]) for i in range(len(args.datasets))] #array of dataset paths -> ["x/D1", "xx/D2"]

  # Iterating datasets
  complete_df_labels = pd.DataFrame(columns=['name', 'labels', 'dataset']) #creates an empty dataframe
  dataset_index =0
  for dataset_path in dataset_paths: #iterate over array of dataset paths
    dataframe_path = join(dataset_path,"p1_label_batches","label_batches.csv")
    df = pd.read_csv(dataframe_path, dtype={'name': str, 'labels': int})
    df['dataset'] = dataset_index #adds a new column
    complete_df_labels = pd.concat([complete_df_labels, df], ignore_index=True) #concatenate dataframes (datasets) in only one
    dataset_index+=1

  # Indexes from videos

  skf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)

  model = create_3dcnn_model()

  if batch:
    total_videos = len(complete_df_labels)
    Y = complete_df_labels['labels'].astype(int).to_numpy()
  else:
    complete_df = pd.DataFrame(columns=['name', 'labels', 'dataset'])
    dataset_index=0
    for dataset_path in dataset_paths:
      dataframe_path = join(dataset_path,"forgery_info","forgery_info.csv")
      df = pd.read_csv(dataframe_path, usecols=['name','labels'],dtype={'name': str, 'labels': int})
      df['dataset'] = dataset_index
      complete_df = pd.concat([complete_df, df], ignore_index=True)
      dataset_index+=1
    total_videos = len(complete_df)
    Y = complete_df['labels'].astype(int).to_numpy()
    
  for fold, (train_indices,test_indices) in enumerate(skf.split(np.arange(total_videos), Y),1):
    print(f"FOLD: {fold}")
    Ytest_fold = Y[test_indices]
    if batch:
      # Split with whole videos
      train_dataset = Dataset.from_generator(lambda: generator(train_indices, dataset_paths, complete_df_labels),
                                              output_types=(tf.float32, tf.int32), output_shapes=([36, 64, 64, 3],[])).batch(batch_s)
      test_dataset = Dataset.from_generator(lambda: generator(test_indices, dataset_paths, complete_df_labels),
                                                  output_types=(tf.float32, tf.int32),output_shapes=([36, 64, 64, 3],[])).batch(batch_s)
      print("Datasets defined")
    else:
      # Split with whole videos
      train_dataset = Dataset.from_generator(lambda: generator2(train_indices, dataset_paths, complete_df_labels, complete_df),
                                              output_types=(tf.float32, tf.int32), output_shapes=([36, 64, 64, 3],[])).batch(batch_s)
      test_dataset = Dataset.from_generator(lambda: generator2(test_indices, dataset_paths, complete_df_labels, complete_df),
                                                  output_types=(tf.float32, tf.int32),output_shapes=([36, 64, 64, 3],[])).batch(batch_s)
      print("Datasets defined")

    # Training
    with device:
      checkpoint = ModelCheckpoint(checkpoint_path+f"/model_fold_{fold}.keras",monitor = 'accuracy', save_best_only = True, mode='max', verbose=0)
      early_stopping = EarlyStopping(monitor='accuracy', patience=7, mode='max', verbose=1, restore_best_weights=True) # early stopping for control of accuracy performance
      model.fit(train_dataset, epochs=epochs, callbacks = [checkpoint, early_stopping])

    # Testing
    model = load_model(checkpoint_path+f"/model_fold_{fold}.keras")
    with device:
      y_pred = model.predict(test_dataset)
    y_pred = y_pred.reshape((-1))

    # Classifying
    for i in range(y_pred.shape[0]):
      if y_pred[i]>0.5:
        y_pred[i] = 1
      else:
        y_pred[i] = 0
    
    if not batch:
      labels = []
      subset_df = complete_df.iloc[test_indices]
      current=0
      for row in subset_df.itertuples():
        indexes = complete_df_labels.index[complete_df_labels['name']==row.name].to_numpy()
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
  return args

if __name__ == '__main__':
  start_time = time.time()
  
  args = run() # ✅ Capture returned args
  
  end_time = time.time()
  log_execution_time("model1", args.datasets, start_time, end_time)
