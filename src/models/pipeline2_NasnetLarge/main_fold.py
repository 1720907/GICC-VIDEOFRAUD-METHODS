import cv2
import numpy as np
import argparse
import time
import os
from os import listdir
from os.path import isfile, join

import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import pandas as pd

from utils import preprocess_video, get_vfd_vfc_labels, extract_features_batch, calculate_r, calculate_rs, calculate_sod, calculate_lb_ub, vf_detection, get_vfd_output, vf_classification, get_vfc_output, build_vgg16

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', metavar='D', type=str, nargs='+', help='Directories containing their each respective batches and label_batches')
    parser.add_argument('-g','--graphs', type=str, default='./graphs', help='Path to save graphs')
    parser.add_argument('-o','--output', type=str, default='./output', help='Path to save y_pred and y_test')
    parser.add_argument('-lt','--lambda_t', type=float, default=0.08, help='Lambda threshold parameter')
    parser.add_argument('-y1','--gamma1', type=float, default=1.4, help='Gamma 1 for VFC')
    parser.add_argument('-y2','--gamma2', type=float, default=1, help='Gamma 2 for VFC')
    parser.add_argument("--cuda", action='store_true', help="Use cuda or cpu")
    parser.add_argument("--batch", action='store_true', help="Use batch videos or not")
    parser.add_argument('-b_s', '--batch_size', type=int, default=1000, help='Batch size for processing videos to manage memory usage')
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
  lambda_t = args.lambda_t
  y1 = args.gamma1
  y2 = args.gamma2
  graph_path = args.graphs
  batch_size = args.batch_size
  batch = args.batch
  output_path = args.output
  n_folds = args.folds
  dataset_paths = [(args.datasets[i]) for i in range(len(args.datasets))]

  # Reading datasets
  complete_df = pd.DataFrame(columns=['name', 'labels', 'dataset'])
  dataset_index =0
  for dataset_path in dataset_paths:
    if batch:
      dataframe_path = join(dataset_path,"label_batches","label_batches.csv")
      df = pd.read_csv(dataframe_path, dtype={'name': str, 'labels': int})
    else:
      dataframe_path = join(dataset_path,"forgery_info","forgery_info.csv")
      df = pd.read_csv(dataframe_path, usecols=['name','labels'],dtype={'name': str, 'labels': int})
    df['dataset'] = dataset_index
    complete_df = pd.concat([complete_df, df], ignore_index=True)
    dataset_index+=1

  # Instantiate pretrained model
  input_shape = (128, 128, 3)
  model = build_vgg16(input_shape = input_shape)
  
  # Folding dataset
  total_videos = len(complete_df)
  Y = complete_df['labels'].astype(int).to_numpy()
  skf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
  folds = skf.split(np.arange(total_videos), Y)
  Y = None

  # Cross-Fold validation 
  for fold, (train_indices,test_indices) in enumerate(folds, 1):
    print(f"Fold: {fold}")
    prepro_videos=[]  
    video_names=[] 
    vfd_outputs = []
    vfc_outputs = []
    vfd_labels = []
    vfc_labels = []
    fold_df = complete_df.iloc[test_indices] # videos for testing
    counter = 1
    batch_counter = 0
    for row in fold_df.itertuples():
      print(f"Video #{counter} of {len(fold_df)}\n")
      filename = row.name
      
      #Getting video path
      if(batch):
        if(row.labels==1):
          vfd_label, vfc_label = 1,2
        elif(row.labels==0):
          vfd_label, vfc_label = 0,0
        else:
          vfd_label, vfc_label = 1,1
        vfd_labels.append(vfd_label)
        vfc_labels.append(vfc_label)
        video_path = join(dataset_paths[row.dataset], "batches", f"{filename}.mp4")
      else:
        video_path = join(dataset_paths[row.dataset], "forgery_data", f"{filename}.mp4")
      
      # Preprocessing
      v_frames, n_frames = preprocess_video(video_path, counter)
      prepro_videos.append(v_frames)
      video_names.append((filename, n_frames))

      # Processing in batches (when the preprocessing batch or fold ends)
      if len(prepro_videos) >= batch_size or counter == len(fold_df):
        if not batch:
          batch_vfd_labels, batch_vfc_labels = get_vfd_vfc_labels(video_names)
          vfd_labels.extend(batch_vfd_labels)
          vfc_labels.extend(batch_vfc_labels)
        
        r_videos = [] # video correlations
        j=1

        #Feature extraction and correlations
        print("Extracting features of batch")
        for p_video in prepro_videos:          
          with device:
            features_batch = extract_features_batch(model, p_video)
          correlations = [calculate_r(features_batch[i], features_batch[i+1]) for i in range(len(p_video)-1)]
          r_videos.append(correlations)
          percentage = (j/len(prepro_videos))*100
          print(f"\rProgress: {percentage:.2f}%", end="")
          j+=1
        print("")

        ## Calculating rs and Rs
        rs_videos = [] 
        sod_videos=[]
        for i in range(len(r_videos)):
            differences = calculate_rs(r_videos[i])
            rs_videos.append(differences)
            sod = calculate_sod(differences)
            sod_videos.append(sod)

        ## Calculate adaptive thresholds (lb, ub)
        lb_ub_videos = calculate_lb_ub(sod_videos)

        ## Executing VFD and VFC
        for i, sod_v in enumerate(sod_videos):
          lb, ub = lb_ub_videos[i]
          vfd = vf_detection(rs_videos[i], lambda_t)
          fd_output = get_vfd_output(vfd) # forgery detection output (0 or 1)
          if fd_output == 0:
            fc_output=0
          else:
            vfc = vf_classification(sod_v, ub, lb, y1, y2)
            fc_output = get_vfc_output(vfc) # forgery classification output (0, 1 or 2)
            if fc_output == 0:
              fd_output=0
          vfd_outputs.append(fd_output)
          vfc_outputs.append(fc_output)

        # Clear the lists for the next batch
        prepro_videos = []
        video_names = []
        batch_counter += 1
      counter += 1

    ## VFD Report
    print("VFD Report: \n", classification_report(vfd_labels, vfd_outputs))
    np.save(output_path+f"/vfd_y_pred_fold_{fold}.npy", vfd_outputs)
    np.save(output_path+f"/vfd_y_test_fold_{fold}.npy", vfd_labels)
    ## VFD Confussion Matrix
    cnf_matrix = confusion_matrix(vfd_labels, vfd_outputs)
    plt.style.use('dark_background')
    disp=ConfusionMatrixDisplay(confusion_matrix = cnf_matrix, display_labels = ["Original","Forged"])
    disp.plot()
    plt.savefig(f"{graph_path}/vfd_cnf_matrix_fold_{fold}.png", format='png')
    
    ## VFC Report
    print("VFC Report: \n", classification_report(vfc_labels, vfc_outputs))
    np.save(output_path+f"/vfc_y_pred_fold_{fold}.npy", vfc_outputs)
    np.save(output_path+f"/vfc_y_test_fold_{fold}.npy", vfc_labels)
    ## VFC Confussion Matrix
    pred_labels = np.unique(vfc_outputs)
    labels = ["Original","Inserted", "Deleted"]
    labels_to_show = [labels[label] for label in pred_labels]
    cnf2_matrix = confusion_matrix(vfc_labels, vfc_outputs, labels = pred_labels)
    plt.style.use('dark_background')
    disp=ConfusionMatrixDisplay(confusion_matrix = cnf2_matrix, display_labels = labels_to_show)
    disp.plot()
    plt.savefig(f"{graph_path}/vfc_cnf_matrix_fold_{fold}.png", format='png')

if __name__ == '__main__':
    inicio = time.time()
    run()
    fin = time.time()
    execution_time = fin - inicio
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)

    print(f"Time of execution: {hours}H:{minutes}M:{seconds}S")
