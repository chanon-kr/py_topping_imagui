import pandas as pd
import cv2
import json

# Read Input File
try :
    with open('config/config.json', 'r') as f :
        input_df = json.loads(f.read())
except :
    video_source = str(input('Input video source : '))
    input_df = {'video_source' : video_source}

# Get Video Source
video_source = input_df.get('video_source','0')
if video_source.replace('.','',1).isdigit() : video_source = int(video_source)
reconnect_video = int(input_df.get('reconnect_video','1'))

# Read Video
video = cv2.VideoCapture(video_source)
while(video.isOpened()):
    # Check
    ret, frame = video.read()
    if not ret:
        if reconnect_video : continue
        else :
            print('Reached the end of the video!')
            break
    # Show
    frame = cv2.resize(frame, (400, 400))
    cv2.imshow('Object detector', frame)
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'): break

# Clean Up
video.release()
cv2.destroyAllWindows()