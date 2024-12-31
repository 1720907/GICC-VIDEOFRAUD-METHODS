import cv2
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input
from skimage.feature import hog

# 4.1. PREPROCESSING
## Grayscale and resize video
def preprocess_video(video_path, counter):
  cap = cv2.VideoCapture(video_path)
  filename = video_path.split("/")[-1]
  print(f"Preprocessing video {counter}: ", filename)
  frames=[]
  i=0
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Grayscale
    resized_frame = cv2.resize(gray_frame, (250, 250)) # Resize to 250x250 pixels
    # resized_frame = cv2.resize(frame, (250, 250)) #,interpolation = cv2.INTER_AREA
    frames.append(resized_frame)
    i+=1
  cap.release()
  print(f"Number of frames: {i}"+'\n'+"Video preprocessed!\n")
  return np.array(frames), i

def preprocess_video_2(video_path, counter):
  cap = cv2.VideoCapture(video_path)
  filename = video_path.split("/")[-1]
  print(f"Preprocessing video {counter}: ", filename)
  frames=[]
  i=0
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Grayscale
    resized_frame = cv2.resize(gray_frame, (250, 250)) # Resize to 250x250 pixels
    # resized_frame = cv2.resize(frame, (128, 128)) #,interpolation = cv2.INTER_AREA
    # resized_frame =cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Parámetros de HOG
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)

    hog_features, hog_image = hog(
       resized_frame,
       orientations=orientations,
       pixels_per_cell=pixels_per_cell,
       cells_per_block=cells_per_block,
       visualize=True
    )
    frames.append(hog_image)
    i+=1
  cap.release()
  print(f"Number of frames: {i}"+'\n'+"Video preprocessed!")
  return np.array(frames), i

def preprocess_video_3(video_path, counter):
  cap = cv2.VideoCapture(video_path)
  filename = video_path.split("/")[-1]
  print(f"Preprocessing video {counter}: ", filename)
  frames=[]
  i=0
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Grayscale
    # resized_frame = cv2.resize(gray_frame, (250, 250)) # Resize to 250x250 pixels
    resized_frame = cv2.resize(frame, (250, 250)) #,interpolation = cv2.INTER_AREA
    resized_frame =cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Parámetros de HOG
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)

    hog_features, hog_image = hog(
       resized_frame,
       orientations=orientations,
       pixels_per_cell=pixels_per_cell,
       cells_per_block=cells_per_block,
       visualize=True
    )
    
    combined_image = cv2.hconcat([resized_frame,hog_image])
    resized_frame =  cv2.resize(combined_image, (250, 250))

    frames.append(resized_frame)
    i+=1
  cap.release()
  print(f"Number of frames: {i}"+'\n'+"Video preprocessed!")
  return np.array(frames), i

# 4.1.1 LABELLING
# Get the labels for Video Forgery Detection/Classification tasks
def get_vfd_vfc_labels(video_names):
  vfd_labels=[]
  vfc_labels=[]
  for v_name in video_names:
    vfd_label = ""
    first_word=v_name[0].split(" ")[0]
    if first_word=="insert":
      vfd_label, vfc_label=1, 1
    elif first_word=="deleted":
      vfd_label, vfc_label=1, 2
    else:
      vfd_label, vfc_label=0, 0
    vfd_labels.append(vfd_label)
    vfc_labels.append(vfc_label)

  return vfd_labels, vfc_labels

# 4.2 FEATURE EXTRACTION AND PCC CALCULATION

def build_cnn(input_shape, seed):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(16, (3, 3), activation='relu', strides=2, kernel_initializer=GlorotUniform(seed=seed)))
    model.add(Conv2D(16, (3, 3), activation='relu', strides=2, kernel_initializer=GlorotUniform(seed=seed)))
    model.add(MaxPooling2D((3, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=GlorotUniform(seed=seed)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=GlorotUniform(seed=seed)))
    model.add(Flatten())
    return model

# Transfer learning with vgg16
def build_vgg16(input_shape):
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in vgg_base.layers:
        layer.trainable = False
    x = Flatten()(vgg_base.output)
    model = Model(inputs = vgg_base.input, outputs=x)
    return model

#Extract features f.
def extract_features(model, frame):
    frame_reshaped = frame.reshape((1, frame.shape[0], frame.shape[1],1))
    features = model.predict(frame_reshaped, verbose=0)
    features = features.squeeze()
    return features

def extract_features_batch(model, frames_batch):
    # frames_reshaped = np.array(frames_batch).reshape((len(frames_batch), frames_batch[0].shape[0], frames_batch[0].shape[1], 1))
    features_batch = model.predict(frames_batch, verbose=0)
    return features_batch

#Pearson correlation coefficient f. (r)
def calculate_r(features1, features2):
    r, _ = pearsonr(features1, features2)
    if r == None:
        return 0
    else:
        return r

#Difference of the adjacent correlation coefficients (rs) f.
def calculate_rs(r_video):
    rs_video = np.abs(np.diff(r_video, n=1))
    return np.insert(rs_video, 0, 0)

#Second-order derivative f. (Rs)
def calculate_sod(differences):
    n = len(differences)
    second_deriv = np.zeros_like(differences)  # Initialize with zeros
    for x in range(1, n-1):
        second_deriv[x] = differences[x+1] + differences[x-1] - (2*differences[x]) #CHANGE
    second_deriv[0] = 0  # Setting boundary conditions based on your specific needs #CHANGE
    if n > 1:
        second_deriv[-1] = 0
    return second_deriv

# 4.3 ADAPTIVE THRESHOLDS
## Calculate (lb, ub) f.
def calculate_lb_ub(sod_videos, sigma_c): 
  i=1
  thresholds = []
  sigma_coefficient = sigma_c
  for sod in sod_videos:
    mean_sod = np.mean(sod) # Mean of Sod
    std_sod = np.std(sod, ddof=1) # standard dev. of Sod.
    ub = mean_sod + (sigma_coefficient * std_sod) #upper bound
    lb = mean_sod - (sigma_coefficient * std_sod) #lower bound
    thresholds.append((lb, ub))
    i+=1
  return thresholds

# 4.4 VFD & VFC
def vf_detection(rs, lambda_t):
      return np.array(rs) > lambda_t

def vf_classification(sod_v, ub, lb, y1, y2):
  class_results = [] #list with classification results from Rs video
  for sod in sod_v:
    if y1 * lb < sod < y1 * ub: #T1-Eq.7
      if y2 * lb < sod < y2 * ub: #T2-Eq.8
        class_results.append('O') 
      else:
        class_results.append('D')
    else:
      class_results.append('I')
  return class_results

# VFD-video result
def get_vfd_output(vfd_list):
    result=0
    for i in vfd_list:
        if(i == True):
            result = 1
            return result
    return result

# VFC-video result
def get_vfc_output(vfc_list):
  result=0
  for i in vfc_list:
    if(i == 'I'):
        result = 1
        return result
    elif(i== 'D'):
        result = 2
        return result
  return result
