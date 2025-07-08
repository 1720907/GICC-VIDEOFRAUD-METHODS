from keras import layers, models
import numpy as np
import pandas as pd
from os.path import join
import tensorflow as tf
import keras

def generate_cnn(input_shape):
  # Create the model
  model = models.Sequential()

  # Layer 1: Convolutional layer
  model.add(layers.Conv1D(32, 3, activation='relu', input_shape=input_shape)) # To 1D
  model.add(layers.BatchNormalization())

  # Layer 2: Max Pooling layer of size 3x3
  model.add(layers.MaxPooling1D(3)) # To 1D

  # Layer 3: Convolutional layer 
  model.add(layers.Conv1D(16, 3, activation='relu')) # To 1D
  model.add(layers.BatchNormalization())

  # Layer 4: Max Pooling layer of size 3x3
  model.add(layers.MaxPooling1D(3)) # To 1D

  # Layer 5: Convolutional layer with 16 filters
  model.add(layers.Conv1D(16, 3, activation='relu')) # To 1D
  model.add(layers.BatchNormalization())

  # Layer 6: Max Pooling layer of size 3x3
  model.add(layers.MaxPooling1D(3)) # To 1D

  # Layer 7: Fully connected layer
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))

  # Output layer with softmax function for binary classification
  model.add(layers.Dense(2, activation='softmax'))

  # Compile the model with Adam optimizer and cross-entropy loss function
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  # return model
  return model

def generator(set_indexes, dataset_paths, complete_df_labels, complete_df):
  subset_df = complete_df.iloc[set_indexes]
  for row in subset_df.itertuples(): # iterate row per row
    labels = complete_df_labels.loc[complete_df_labels['name']==row.name, 'labels'].astype(int).to_numpy() # get labels per batch from video
    # print("Labels: ", labels, row.name) # View labels of batches of video
    path = join(dataset_paths[row.dataset],"p3_data_batches",f"{row.name}.npy") # get path of video (set of batches)
    if(row.name[:-4]=='.mp4'):
      raise Exception(f'Nombre de archivo: {row.name}')
    data = np.load(path) # get batches of video
    # print(f"Data shape: {data.shape}, Data length: {len(data)}, Labels length: {len(labels)}") # compare data length with labels length
    for i in range(len(labels)): # iterate batch per batch
      one_hot_label = keras.utils.to_categorical(labels[i], num_classes=2)    # One-hot encoding: Convert to [0,1] or [1,0]
      yield data[i].T, one_hot_label # One-hot encoding