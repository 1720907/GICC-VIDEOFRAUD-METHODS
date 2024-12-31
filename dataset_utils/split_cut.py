import moviepy.editor as mpe
from pathlib import Path
import random
import os
#------------------------------------------------------------------------------------------
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--data', type=str, default=r"C:\Users\Jorge-PC\Desktop\videos\sin_recortar", help='Path of videos to cut')
    parser.add_argument('-sd','--save_data', type=str, default=r"C:\Users\Jorge-PC\Desktop\videos\recortados", help='Path to export cutted videos')
    parser.add_argument('--min', type=int, default=5, help='Min duration of each final clip cutted')
    parser.add_argument('--max', type=int, default=72, help='Max duration of each final clip cutted')
    return parser.parse_args()
#---------------------------------------------------------------------------------------

#Get the duration of the video function
def get_duration(video):
   return video.duration

#Create timestamps function
def create_timestamps(video_dur, min, max):
   duration_vid = video_dur
   timestamps = []
   timestamps.append(0)
   while video_dur!=0:
      if video_dur>max:
         random_int = random.randint(min,max)
         while (video_dur-random_int)<min:
           random_int= random.randint(min,max)
         timestamp = timestamps[-1]+random_int
         timestamps.append(timestamp)
         video_dur-=random_int
      else:
         #the last timestamp will be the duration of the video:
         timestamps.append(duration_vid)
         video_dur=0
   return timestamps

#Main function  
def run():
  args = parse_args()
    
  # Set path
  data_path = args.data
  data_export=args.save_data
  #----------------------------------------------------------------
  p1 = Path(data_path)
  for fvids in p1.glob('*.mp4'):
    

    #video_name:
    file_name=os.path.basename(fvids)
    

    #get the video
    video = mpe.VideoFileClip(str(fvids))
    
    #obtaining timestamps
    duration = get_duration(video)
    timestamps=create_timestamps(duration, args.min, args.max)

    #cutting the video
    before_stamp=0
    for x in range(1,len(timestamps)):
       #Main video
       print(file_name)
       path = os.path.join(data_export,f"{file_name[:-4]}_{x}.mp4")
       
       sub_vid = video.subclip(before_stamp,timestamps[x])
       #creating cutted videoclip, naming resulting cutted video clips
       sub_vid.write_videofile(path)
       before_stamp=timestamps[x]


#---------------------------------------------------------------------------------------
if __name__ == '__main__':
    run()