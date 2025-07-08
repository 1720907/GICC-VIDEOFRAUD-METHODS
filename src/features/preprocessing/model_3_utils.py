import cv2
import numpy as np

def resize_frames(list_frames,keep_aspect_ratio, target_size=(128,128)):
  if keep_aspect_ratio:
    resized_frames=[]
    for frame in list_frames:
      # Option 1: Keep aspect ratio with padding
      h, w = frame.shape
      scale = min(target_size[0] / h, target_size[1] / w)
      new_w, new_h = int(w * scale), int(h * scale)
      resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
      pad_w = (target_size[1] - new_w) // 2
      pad_h = (target_size[0] - new_h) // 2
      processed_frame = cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
      resized_frames.append(processed_frame)
  else:
    # Option 2: Rescale proportionally without padding
    resized_frames = [cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA) for frame in list_frames]
  return resized_frames

def generate_frame_batches(processed_frames, batch_size=100):
  batches = []

  # Obtain length of video frames
  n_frames = len(processed_frames)

  # Iterate n_frames by batch_size
  for start in range(0, n_frames, batch_size):
    end = start + batch_size 
    
    # Ensure the batch has the required batch_size
    if end <= n_frames: 
      batch = processed_frames[start:end] 

    # Fill the remainder with zeros if there are not enough frames
    else:
      remainder = end - n_frames 
      batch = np.vstack((processed_frames[start:n_frames], np.zeros((remainder, processed_frames.shape[1], processed_frames.shape[2]), dtype=processed_frames.dtype)))

    batches.append(batch)

  return batches

def generate_label_batches(labels, batch_size=100):
  batches = []
  
  # Obtain legth of video labels
  n_labels = labels.shape[0]
  
  # Iterate n_labels by batch_size
  for start in range(0, n_labels, batch_size):
    end = start + batch_size
    
    # Ensure the batch has the required batch_size
    if end <= n_labels:
      batch = labels[start:end]

    # Fill the remainder with zeros if there are not enough frames
    else:
      remainder = end - n_labels
      batch = np.hstack((labels[start:n_labels], np.zeros(remainder,dtype=labels.dtype)))
    
    batches.append(batch)

  return batches

def label_diff(labels):
  L = []
  
  for i in range(len(labels)-1):
    L.append(abs(labels[i]-labels[i+1]))
  
  return L

def compute_correlation_coefficients(frames):
  correlations = []

  # Obtain legth of video batch
  n = len(frames)
  
  # Iterate n frames per each
  for k in range(n - 1):

    # Convert consecutive frames to float numbers
    Gk = frames[k].astype(np.float32)
    Gk1 = frames[k + 1].astype(np.float32)
    
    # Get mean of each consecutive frame
    Gk_mean = np.mean(Gk)
    Gk1_mean = np.mean(Gk1)
    
    # Get numerator and denominator for correlation coefficient calculation.
    numerator = np.sum((Gk - Gk_mean) * (Gk1 - Gk1_mean))
    denominator = np.sqrt(np.sum((Gk - Gk_mean) ** 2) * np.sum((Gk1 - Gk1_mean) ** 2))
    
    # Calculate correlation coefficient
    ck = numerator / denominator if denominator != 0 else 0
    correlations.append(ck)
  
  return correlations

def compute_correlation_differences(correlations):
  return [correlations[i + 1] - correlations[i] for i in range(len(correlations) - 1)]