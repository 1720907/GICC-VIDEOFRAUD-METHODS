import cv2
import random
import numpy as np
import argparse
import os
from os import listdir
from os.path import isfile, join
import pandas as pd

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-d','--data', type=str, default='', help='Path of original videos')
  parser.add_argument('-f','--forged', type=str, default='', help='Path of forged videos')
  parser.add_argument('-l','--labels', type=str, default='', help='Path to save labels')
  parser.add_argument('-fd','--forgery_data', type=str, default='', help='Path to save data of forgeries into a csv')
  parser.add_argument('-p','--p_forgery', type=float, default=0.2, help='Percentage of forgery')
  parser.add_argument('-pt','--path_type', type=str, default='/', help='Path type for linux or windows system')
  
  
  return parser.parse_args()

#Generate ranges of deletions
def generate_frame_deletions(frames_to_delete, total_frames, percent_deletion)

  #initialize array of frame deletions
  frame_deletions = []
  #set of ranges
  existing_ranges = set()

  left_after_delete = total_frames-frames_to_delete-20
  max_deletions = round((frames_to_delete/10)+0.5) 
  number_fill_ranges=max_deletions+1
  min_separation= round((left_after_delete/number_fill_ranges)-0.5)
  #counter to know the number of number of ranges to delete
  i=0
  while frames_to_delete>0:
    if(frames_to_delete<10):
      break
    elif (10<=frames_to_delete<=30):
      ordered = sorted(frame_deletions)
      control=False
      for index in range(len(ordered)-1):
        if((ordered[index+1][0] - ordered[index][1] - frames_to_delete) > 2*min_separation):
          control=True
          break
      if ((10+frames_to_delete+min_separation)>(ordered[0][0])) and ((ordered[-1][1]+min_separation)>(total_frames-41)) and not (control):
        max_separation=0
        lower = 0
        upper = 0
        for index in range(len(ordered)-1):
          val = ordered[index+1][0] - ordered[index][1]
          if val>max_separation:
            max_separation=val 
            lower = ordered[index][1]
            upper = ordered[index+1][0]
        separation = round(((upper-lower-frames_to_delete)/2)-0.5)
        start_frame = lower+separation
        end_frame = start_frame+frames_to_delete
        new_range = (start_frame, end_frame)

        frame_deletions.append(new_range)
        existing_ranges.add(new_range)
        i=i+1
        break
      else:  
        start_frame = random.randint(11-1, total_frames-1 - 40)
        end_frame = start_frame + frames_to_delete
        new_range = (start_frame, end_frame)
    else:
      start_frame = random.randint(11-1, total_frames-1 - 40) 
      end_frame = start_frame + random.randint(10, 30)
      new_range = (start_frame, end_frame)
    
    n_frames_to_del=end_frame-start_frame

    # Check if the new range overlaps with existing ones or if it's already generated
    if (new_range in frame_deletions) or any(value==value2 for start,end in existing_ranges for value in range(start_frame, end_frame+1) for value2 in range(start,end+1)):
      continue

    # Check if the new range is separated by at least min_separation value
    if any(abs(start - end_frame) < min_separation or abs(end - start_frame) < min_separation for start, end in existing_ranges):
      continue
    frames_to_delete=frames_to_delete-n_frames_to_del
    frame_deletions.append(new_range)
    existing_ranges.add(new_range)
    i=i+1
  return frame_deletions, i

def delete_and_concatenate_frames(video_path, percent_deletion, output_path):
  cap = cv2.VideoCapture(video_path)

  
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps_video = cap.get(cv2.CAP_PROP_FPS)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  #----------------------------------------------------
  print("Total de frames: ", total_frames)
  # Getting actual accesible frames
  i=0
  n_error=0
  while True:
    ret,frame = cap.read()
    if not ret:
      if(i<total_frames):
        if(n_error==1):
          break
        cap.set(cv2.CAP_PROP_POS_FRAMES,i)
        if(int(cap.get(cv2.CAP_PROP_POS_FRAMES))!=i):
          break
        n_error=1
        continue
      else:
        break
    if(n_error==1):
      n_error=0
    i+=1
  cap.release()
  total_accesible_frames=i
  #-------------------------------------------------------
  print("Total de frames accesibles: ", total_accesible_frames)

  frames_to_delete=int(percent_deletion*total_accesible_frames)
  frames_del=frames_to_delete

  frame_deletions, n_deletions = generate_frame_deletions(frames_to_delete, total_accesible_frames, percent_deletion)

  codec = cv2.VideoWriter_fourcc(*'mp4v')  # Adjust according to your needs

  out = cv2.VideoWriter(output_path, codec, fps_video, (width, height))

  current_frame_index = 0
  n_error=0
  current_forged_index=0
  #forged indexes of current video
  forged_indexes_v=[]
  cap = cv2.VideoCapture(video_path)
  while True:
    ret, frame = cap.read()
    if not ret:
      if(current_frame_index<total_frames):
        if(n_error==1):
          break
        cap.set(cv2.CAP_PROP_POS_FRAMES,current_frame_index)
        if(int(cap.get(cv2.CAP_PROP_POS_FRAMES))!=current_frame_index):
          break
        n_error=1
        continue
      else:
        break

    # Determine if the current frame should be deleted based on frame_deletions
    delete_frame = any(start <= current_frame_index <= end for start, end in frame_deletions)
    start_delete_frame=any(start==current_frame_index for start, end in frame_deletions)

    if start_delete_frame:
      forged_indexes_v.append(current_forged_index)
    if not delete_frame:
      out.write(frame)
      current_forged_index+=1
    if(n_error==1):
      n_error=0
    current_frame_index += 1
  cap.release()
  out.release()
  
  return forged_indexes_v, current_forged_index, current_frame_index, frames_del, n_deletions, fps_video

def run():
  args = parse_args()
  # Set path
  input_path = args.data
  forged_path = args.forged
  labels_path = args.labels
  percent_forged = args.p_forgery
  csv_path = args.forgery_data
  path_type = args.path_type

  #For dataframe
  # Create empty lists to store data
  n_frames_deleted = []
  filenames = []
  video_fps = []
  segundos_de_corte = []
  n_frames_original = []
  n_manipulaciones = []
  indices_de_manipulacion = []
  n_frames_video_manipulado = []

  input_files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
  input_files = sorted(input_files)

  i=0
  for filename in input_files:
    if filename.endswith(".mp4") or filename.endswith(".avi"):
      video_path = os.path.join(input_path, filename)

      output_path = os.path.join(forged_path, filename[:-4] + "_forged.mp4")
      forged_indexes, n_f_frames, n_frames, frames_del, n_deletions, fps_video = delete_and_concatenate_frames(
                                                                    video_path, percent_forged, output_path)

      forged_positions = np.array(forged_indexes)+1
      print(f'{filename}: cantidad de frames eliminados: {frames_del}')  
      print(f"{filename}: video fps: ", fps_video)
      print(f"{filename}: timestamps: ", forged_positions/fps_video)
      print(f"{filename}: n° de frames-v. original: ", n_frames)
      print(f"{filename}: n° de manipulaciones: ", n_deletions)
      print(f"{filename}: índices de manipulación:", forged_positions)
      print(f"{filename}: n° de frames-v. manipulado: ", n_f_frames)

      #Appending data to arrays for dataframe construction
      n_frames_deleted.append(frames_del)
      filenames.append(filename)
      video_fps.append(fps_video)
      segundos_de_corte.append(forged_positions/fps_video)
      n_frames_original.append(n_frames)
      n_manipulaciones.append(n_deletions)
      indices_de_manipulacion.append(forged_positions)
      n_frames_video_manipulado.append(n_f_frames)

      #Generation of labels
      label = np.zeros(n_f_frames)
      for index in forged_indexes:
        label[index]=1

      np.save(f'{labels_path}{path_type}{filename[:-4]}_label.npy', label)
      i+=1
      #----------------------------------------------------
      print("Videos procesados: ",i,"\n")

  # Create DataFrame
  data = {
    'Filename': filenames,
    'n_frames_deleted': n_frames_deleted,
    'Video FPS': video_fps,
    'Deletion points (s)': segundos_de_corte,
    'Num. of original frames': n_frames_original,
    'Num. of deletions': n_manipulaciones,
    'Forgery indexes': indices_de_manipulacion,
    'Num. of frames after deletion': n_frames_video_manipulado
  }

  df = pd.DataFrame(data)

  # Export DataFrame to CSV
  df.to_csv(f"{csv_path}{path_type}data.csv", index=False)

if __name__ == '__main__':
  run()
