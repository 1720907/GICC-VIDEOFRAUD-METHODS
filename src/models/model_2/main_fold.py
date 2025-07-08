import random
import cv2
import numpy as np
import argparse
import time
import sys
import os
from os.path import join
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from models_registry import VALID_MODELS  # Import just the model names
from utils import preprocess_video, get_vfd_vfc_labels, extract_features_batch, calculate_r, calculate_rs, calculate_sod, calculate_lb_ub, vf_detection, get_vfd_output, vf_classification, get_vfc_output, pretrained_cnn, original_cnn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) # recognizes modeule utilities
from utilities.logging import log_execution_time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', metavar='D', type=str, nargs='+', help='Directories containing their each respective batches and label_batches')
    parser.add_argument('-g','--graphs', type=str, default='./graphs', help='Path to save graphs')
    parser.add_argument('-o','--output', type=str, default='./output', help='Path to save y_pred and y_test')
    parser.add_argument('-m', '--model_name', type=str, default='model2_original', help='Name of model for feature extraction')
    parser.add_argument('-lt','--lambda_t', type=float, default=0.1, help='Lambda threshold for determine a video is original/forged for VFD')
    parser.add_argument('-sc','--sigma_c', type=float, default=2.6, help='Sigma coefficient for calculating adaptive thresholds for VFC')
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
  # Instantiating parameters

  dataset_paths = [(args.datasets[i]) for i in range(len(args.datasets))]
  output_path = args.output # correct in the last lines
  model_name = args.model_name
  lambda_t = args.lambda_t
  sigma_c = args.sigma_c
  y1 = args.gamma1
  y2 = args.gamma2
  batch_size = args.batch_size
  batch = args.batch
  n_folds = args.folds

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

  # Choosing the appropiate model
  if model_name == "model2_original":
    seed = 40
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    # Instantiate pretrained model
    model = original_cnn((250, 250, 1), seed)
  elif model_name in VALID_MODELS:
    model = pretrained_cnn((128, 128, 3), model_name)
  else:
    raise ValueError(f"❌ Error: '{model_name}' no es un modelo válido. Debe ser uno de {list(VALID_MODELS)}.")
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
    fold_df = complete_df.iloc[test_indices] # get videos for testing
    counter = 1
    batch_counter = 0
    for row in fold_df.itertuples(): #iterate videos for testing
      print(f"Video #{counter} of {len(fold_df)}")
      filename = row.name
      
      #Getting video path and labelling
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
        #the labelling ocurres in line #124
      
      # Preprocessing
      v_frames, n_frames = preprocess_video(video_path, counter, model_name)
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
        print("Extracting features of fold/batch and calculating PCC")
        for p_video in prepro_videos:          
          with device:
            #Feature extraction
            features_batch = extract_features_batch(model, p_video)
          #Pearson Correlation Coeficient (PCC)
          correlations = [calculate_r(features_batch[i], features_batch[i+1]) for i in range(len(p_video)-1)]
          r_videos.append(correlations)
          #Calculating progess of processing
          percentage = (j/len(prepro_videos))*100
          print(f"\rProgress: {percentage:.2f}%", end="")
          j+=1
        print("")

        ## Calculating Differences of PCC (rs) and Second Order Derivative (Rs/Sod)
        rs_videos = [] 
        sod_videos=[]
        for i in range(len(r_videos)):
            differences = calculate_rs(r_videos[i])
            rs_videos.append(differences)
            sod = calculate_sod(differences)
            sod_videos.append(sod)

        ## Calculate adaptive thresholds (lb, ub)
        lb_ub_videos = calculate_lb_ub(sod_videos, sigma_c)

        # Generating Graphics
        for i in range(len(r_videos)):
          print(f"video lenght: {len(prepro_videos[i])}")
          print(f"r (PCC) lenght: {len(r_videos[i])}")
          print(f"rs lenght: {len(rs_videos[i])}")
          print(f"Rs lenght: {len(sod_videos[i])}")

          fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 3))
          ax1.plot(r_videos[i])
          ax1.set_xlabel('Data points')
          ax1.set_ylabel('Value')
          ax1.set_title(f'PCC (r)')

          ax2.plot(rs_videos[i])
          ax2.set_xlabel('Data points')
          ax2.set_ylabel('Value')
          ax2.set_title(f'PCC Diferences (rs)')

          ax3.plot(sod_videos[i])
          ax3.set_xlabel('Data points')
          ax3.set_ylabel('Value')
          ax3.set_title(f'SOD with lb and ub')

        # Draw horizontal lines based on limits
          y_values = [lb_ub_videos[i][0]*y2, lb_ub_videos[i][1]*y2, lb_ub_videos[i][0]*y1, lb_ub_videos[i][1]*y1]
          colors = ['r', 'g', 'c', 'orange']
          labels = ['y2*lb', 'y2*ub', 'y1*lb', 'y1*ub']
          for j, (y, color, label) in enumerate(zip(y_values, colors, labels)):
            ax3.axhline(y=y, color=color, linestyle='-')
            if j % 2 == 0:  # Para las líneas inferiores (Y1, Y3), colocamos el texto encima de la línea
              ax3.text(50, y, f' {label}', va='bottom', ha='left')  # Ajusta la posición x según sea necesario
            else:  # Para las líneas superiores (Y2, Y4), colocamos el texto debajo de la línea
              ax3.text(50, y, f' {label}', va='top', ha='left')  # Ajusta la posición x según sea necesario
          fig.suptitle(f'video {i+1}: {video_names[i][0]}')
          # fig.subplots_adjust(wspace=0.3, hspace=0.3, top=0.80)  # Adjust space between subplots
          plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leaves space for suptitle

          # plt.show() 
          plt.savefig(f"{'/models/model2_vgg16_predictions/D3/graphs'}/{video_names[i][0]}.png", format='png')
        # Executing VFD and VFC
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
    #cnf_matrix = confusion_matrix(vfd_labels, vfd_outputs)
    #plt.style.use('dark_background')
    #disp=ConfusionMatrixDisplay(confusion_matrix = cnf_matrix, display_labels = ["Original","Forged"])
    #disp.plot()
    #plt.savefig(f"{graph_path}/vfd_cnf_matrix_fold_{fold}.png", format='png')
    
    ## VFC Report
    print("VFC Report: \n", classification_report(vfc_labels, vfc_outputs))
    np.save(output_path+f"/vfc_y_pred_fold_{fold}.npy", vfc_outputs)
    np.save(output_path+f"/vfc_y_test_fold_{fold}.npy", vfc_labels)
    ## VFC Confussion Matrix
    #pred_labels = np.unique(vfc_outputs)
    #labels = ["Original","Inserted", "Deleted"]
    #labels_to_show = [labels[label] for label in pred_labels]
    #cnf2_matrix = confusion_matrix(vfc_labels, vfc_outputs, labels = pred_labels)
    #plt.style.use('dark_background')
    #disp=ConfusionMatrixDisplay(confusion_matrix = cnf2_matrix, display_labels = labels_to_show)
    #disp.plot()
    #plt.savefig(f"{graph_path}/vfc_cnf_matrix_fold_{fold}.png", format='png')
  return args  # ✅ Now it returns the args

if __name__ == '__main__':
  start_time = time.time()
  
  args = run() # ✅ Capture returned args
  
  end_time = time.time()
  log_execution_time(args.model_name, args.datasets, start_time, end_time)
