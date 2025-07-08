import os
import cv2
import numpy as np
import argparse
import time
import tensorflow as tf
from tensorflow.data import Dataset
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from sklearn.metrics import accuracy_score
from utils import create_3dcnn_model, split_data_indices
from sklearn.metrics import classification_report

from os.path import isfile, join
from os import listdir

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-d','--data', type=str, default='./data_batches', help='Path of videos')
  parser.add_argument('-l','--labels', type=str, default='./label_batches/labels.npy', help='Path of labels')
  parser.add_argument('-i','--indexes', type=str, default='./indexes.npy', help='Path of indexes')
  parser.add_argument('-ch','--checkpoint', type=str, default='./checkpoint', help='Path to save checkpoint of the model')
  parser.add_argument('-b_s','--batch_size', type=int, default=16, help='Batch size for training the model')
  parser.add_argument('-e','--epochs', type=int, default=20, help='Epochs for training')
  parser.add_argument('-t','--testing', type=float, default=0.26, help='Percentage for testing')
  parser.add_argument('-v','--validation', type=float, default=0.15, help='Percentage for validation')
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

def generator(set_indexes, indexes, arrays_paths, video_dir, Ytrain):
  for i in set_indexes:
    batch_indexes = indexes[i]
    path = os.path.join(video_dir,arrays_paths[batch_indexes[0]])
    data = np.load(path)
    yield data[batch_indexes[1]], Ytrain[i]

#set_indexes, indices aleatorios para el set (de 1 a 13320)
#indexes, indices que indica el nombre del video con sus correspondientes batchs
#arrays_paths, lista con nombres de videos
#video_dir, directorio que contiene los videos en npy q contiene a su vez los batches
#Ytrain array de labels
def generator_2(set_indexes, indexes, arrays_paths, video_dir, Y_train):
  for i in set_indexes:
    indexes_Y = np.where(indexes[:,0]==i)[0]
    video_name = arrays_paths[i]
    path = os.path.join(video_dir,video_name)
    data=np.load(path)
    for j in range(len(data)):
      yield data[j], Y_train[indexes_Y[j]] 


def run():
  args = parse_args()
  #definicion de argumentos
  video_dir = args.data
  labels_path = args.labels
  indexes_path = args.indexes
  device = detect_device(args)
  batch_s = args.batch_size
  checkpoint_path = args.checkpoint
  testing= args.testing
  validation = args.validation
  epochs=args.epochs

  # Load indexes
  indexes = np.load(indexes_path)

  # Ordering videos
  input_videos = [f for f in listdir(video_dir) if isfile(join(video_dir, f))]
  input_videos = sorted(input_videos)
  print("Files read")

  
  # ordering labels
  labels = np.load(labels_path)
  print("Label read")
  

  # indexes from batches
  # train_indices, val_indices, test_indices = split_data_indices(len(labels),testing, validation, random_seed=42)

  # Indexes from videos
  train_indices, val_indices, test_indices = split_data_indices(len(input_videos),testing, validation, random_seed=42)

  print("Indexes for test, validation and training created")
  model = create_3dcnn_model()

  # Setup data generators or tf.data.Dataset here for both training and validation
  # train_dataset = Dataset.from_generator(lambda: generator(train_indices, indexes, input_videos,video_dir, labels),
  #                                         output_types=(tf.float32, tf.int32), output_shapes=([36, 64, 64, 3],[])).batch(batch_s)
  # val_dataset = Dataset.from_generator(lambda: generator(val_indices, indexes, input_videos,video_dir, labels),
  #                                             output_types=(tf.float32, tf.int32),output_shapes=([36, 64, 64, 3],[])).batch(batch_s)
  # test_dataset = Dataset.from_generator(lambda: generator(test_indices, indexes, input_videos, video_dir, labels),
  #                                             output_types=(tf.float32, tf.int32),output_shapes=([36, 64, 64, 3],[])).batch(batch_s)

  # Split with whole videos
  train_dataset = Dataset.from_generator(lambda: generator_2(train_indices, indexes, input_videos,video_dir, labels),
                                          output_types=(tf.float32, tf.int32), output_shapes=([36, 64, 64, 3],[])).batch(batch_s)
  val_dataset = Dataset.from_generator(lambda: generator_2(val_indices, indexes, input_videos,video_dir, labels),
                                              output_types=(tf.float32, tf.int32),output_shapes=([36, 64, 64, 3],[])).batch(batch_s)
  test_dataset = Dataset.from_generator(lambda: generator_2(test_indices, indexes, input_videos, video_dir, labels),
                                              output_types=(tf.float32, tf.int32),output_shapes=([36, 64, 64, 3],[])).batch(batch_s)
  print("Datasets defined")
                                                

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

  # batches shuffle
  # Ytest = labels[test_indices]

  #Split with whole videos
  y_indexes = np.array([],dtype='uint8')
  for video_index in test_indices:
    indexes_Y = np.where(indexes[:,0]==video_index)[0]
    y_indexes = np.append(y_indexes,indexes_Y)
  Ytest = labels[y_indexes]

  acc = accuracy_score(Ytest, y_pred)
  print("Report: \n", classification_report(Ytest, y_pred))

  print("Accuracy test: ", acc)
  np.save(checkpoint_path+f"/y_pred.npy",y_pred)
  np.save(checkpoint_path+f"/y_test.npy", Ytest)

if __name__ == '__main__':
  inicio = time.time()
  run()
  fin = time.time()
  execution_time = fin - inicio
  hours = int(execution_time // 3600)
  minutes = int((execution_time % 3600) // 60)
  seconds = int(execution_time % 60)

  print(f"Time of execution: {hours}H:{minutes}M:{seconds}S")
