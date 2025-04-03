from keras import layers, models
import numpy as np
import pandas as pd
from os.path import join

def generate_cnn(input_shape):
  # Create the model
  model = models.Sequential()

  # Layer 1: Convolutional layer with 32 filters of size 3x3, followed by Batch Normalization and ReLU
  model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
  model.add(layers.BatchNormalization())

  # Layer 2: Max Pooling layer of size 3x3
  model.add(layers.MaxPooling2D((3, 3)))

  # Layer 3: Convolutional layer with 16 filters of size 3x3, followed by Batch Normalization and ReLU
  model.add(layers.Conv2D(16, (3, 3), activation='relu'))
  model.add(layers.BatchNormalization())

  # Layer 4: Max Pooling layer of size 3x3
  model.add(layers.MaxPooling2D((3, 3)))

  # Layer 5: Convolutional layer with 16 filters of size 3x3, followed by Batch Normalization and ReLU
  model.add(layers.Conv2D(16, (3, 3), activation='relu'))
  model.add(layers.BatchNormalization())

  # Layer 6: Max Pooling layer of size 3x3
  model.add(layers.MaxPooling2D((3, 3)))

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
  for row in subset_df.itertuples():
    labels = complete_df_labels.loc[complete_df_labels['name']==row.name, 'labels'].astype(int).to_numpy()

    path = join(dataset_paths[row.dataset],"p3_data_batches",f"{row.name}.npy")
    if(row.name[:-4]=='.mp4'):
      raise Exception(f'Nombre de archivo: {row.name}')
    data = np.load(path)
    for i in range(len(data)):
        yield np.expand_dims(data[i], axis=-1), labels[i]   # <-- Agrega la dimensión del canal