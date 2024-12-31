import cv2
import numpy as np
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

def resize_frames(video_frames, target_size=(64, 64)):
  resized_videos=[]
  for video in video_frames:
    resized_frames = [cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA) for frame in video]
    resized_videos.append(resized_frames)
  return resized_videos

def resize_frames_single(list_frames, target_size=(64,64)):
  resized_frames = [cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA) for frame in list_frames]
  return resized_frames

def frame_difference(processed_frames, processed_labels):
  print("len(processed_frames): ", len(processed_frames))
  print("len(processed_labels): ", len(processed_labels))
  differences = []
  labels =[]
  for i in range(len(processed_frames)):
    differences_video = []
    labels_video = []
    for j in range(len(processed_frames[i]) - 1):
      # Compute the absolute difference between consecutive frames
      diff = cv2.absdiff(processed_frames[i][j], processed_frames[i][j+1])
      label = abs(processed_labels[i][j] - processed_labels[i][j+1])
      differences_video.append(diff)
      labels_video.append(label)
    differences_video = np.array(differences_video)
    labels_video = np.array(labels_video)
    differences.append(differences_video)
    labels.append(labels_video)
  return differences, labels

def frame_difference_single(processed_frames, processed_labels):
  differences_video = []
  labels_video = []
  for j in range(len(processed_frames)-1):
    diff = cv2.absdiff(processed_frames[j], processed_frames[j+1])
    label = abs(processed_labels[j] - processed_labels[j+1])
    differences_video.append(diff)
    labels_video.append(label)
  differences_video = np.array(differences_video)
  labels_video = np.array(labels_video)
  return differences_video, labels_video

def generate_frame_batches(videos, batch_size=36):
  batches = []
  for video in videos:
    n_frames = video.shape[0]
    for start in range(0, n_frames, batch_size):
      end = start + batch_size
      # Ensure the batch has the required batch_size, otherwise fill with zeros
      if end <= n_frames:
        batch = video[start:end]
      else:
        # If there are not enough frames left for a full batch, fill the remainder with zeros
        remainder = end - n_frames
        batch = np.vstack((video[start:n_frames], np.zeros((remainder, video.shape[1], video.shape[2], video.shape[3]), dtype=video.dtype)))
      batches.append(batch)
  return batches

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

def generate_label_batches(labels,batch_size=36):
  batches = []
  for video in labels:
    n_labels = video.shape[0]
    for start in range(0,n_labels,batch_size):
      end = start+batch_size
      if end <= n_labels:
        batch = video[start:end]
      else:
        remainder = end-n_labels
        batch = np.hstack((video[start:n_labels],np.zeros(remainder,dtype=video.dtype)))
      batches.append(batch)
  return batches

def generate_label_batches_single(labels,batch_size=36):
  batches = []
  n_labels = labels.shape[0]
  for start in range(0,n_labels,batch_size):
    end = start+batch_size
    if end <= n_labels:
      batch = labels[start:end]
    else:
      remainder = end-n_labels
      batch = np.hstack((labels[start:n_labels],np.zeros(remainder,dtype=labels.dtype)))
    batches.append(batch)

  return batches

def train_validation_test_split(X_data, Y_labels, test_size, val_size, random_state=42):
  val_size=val_size/(1-test_size)
  #primero dividir en train_temp, y test
  X_train_temp, X_test, Y_train_temp, Y_test = train_test_split(X_data, Y_labels, test_size=test_size, random_state=random_state)
  X_train, X_val, Y_train, Y_val = train_test_split(X_train_temp, Y_train_temp, test_size=val_size, random_state=random_state)
  return X_train,X_val,X_test,Y_train,Y_val,Y_test

def split_data_indices(total_size, test_percent, val_percent, random_seed=None):

    

    val_size = int(total_size * val_percent)
    test_size = int(total_size * test_percent)
    train_size = total_size - val_size - test_size
    

    indices = np.arange(total_size)
    if random_seed is not None:
        np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices

def create_3dcnn_model():
  # Define the input shape for the arbitrary crops mentioned
  input_shape = (36, 64, 64, 3)  # Adjusted to the cropping size mentioned for jittering

  model = models.Sequential()

  # Convolutional Layer 1
  model.add(layers.Conv3D(8, (3,3,3), strides=(1,1,1), padding='same', activation='relu', input_shape=input_shape))

  # Convolutional Layer 2
  model.add(layers.Conv3D(8, (3,3,3), strides=(1,1,1), padding='same', activation='relu'))
  # Pooling Layer 1 with temporal kernel depth d=1, spatial kernel size k=2

  model.add(layers.MaxPooling3D((1,2,2)))
  for _ in range(1):
   model.add(layers.Dropout(0.2))


  # Convolutional Layer 3
  model.add(layers.Conv3D(16, (3,3,3), strides=(1,1,1), padding='same', activation='relu'))

  model.add(layers.MaxPooling3D((2,2,2)))
  for _ in range(1):
   model.add(layers.Dropout(0.2))


  # Convolutional Layer 4
  model.add(layers.Conv3D(16, (3,3,3), strides=(1,1,1), padding='same', activation='relu'))

  # Pooling Layer 2 and 3 with 3D max-pooling kernel size 2x2x2
  model.add(layers.MaxPooling3D((2,2,2)))
  for _ in range(1):
   model.add(layers.Dropout(0.2))


 
  # Flatten the layers before the fully connected layers
  model.add(layers.Flatten())

  # Fully Connected Layer 1
  model.add(layers.Dense(1024, activation='relu'))
  # Dropout Layer 1 to 3
  for _ in range(2):
    model.add(layers.Dropout(0.2))

  # Fully Connected Layer 2
  model.add(layers.Dense(512, activation='relu'))
  # Dropout Layers 2 to 5
  # for _ in range(4):
  #   model.add(layers.Dropout(0.2))

  # Output layer with sigmoid activation for binary classification
  model.add(layers.Dense(1, activation='sigmoid'))

  # Compile the model with specified learning rate and momentum

  #-----------------optimizador---------------------------------
  optimizer = optimizers.SGD(learning_rate=0.009, momentum=0.9)
  # optimizer = optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

  return model
