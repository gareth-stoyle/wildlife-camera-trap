from camera_trap import CamTrap
import video_processing
import db

import datetime
import os
import time

#
# get file and path details
#

current_date = datetime.datetime.now().strftime("%Y-%m-%d")
current_directory = os.getcwd()
path = os.path.abspath(os.path.join(current_directory, 'videos'))
video_file = f"unprocessed-{current_date}_footage.h264"
full_video_path = path + '/' + video_file

#
# Setup Camera Trap System
#

framerate = 8
resolution = '720x480'
duration = 10*60
end_time = current_date + duration
camtrap = CamTrap(framerate, resolution, current_date)

#
# Start recording
#

camtrap.start_recording(path, video_file)
start_time = datetime.datetime.now().strftime('%H:%M:%S')

print(f"Recording video to {video_file} in the path: {path}.\nPress 'q' and Enter to stop.")

#
# Handle end of recording/logging.
#

try:
    user_input = input()
    while datetime.datetime.now() < end_time:
        user_input = input()
    print('Exiting while loop...')
except Exception as e:
    print('[EXCEPTION]', e)
finally:
    print('Ending recording sessions...')
    camtrap.stop_recording()
    end_time = datetime.datetime.now().strftime('%H:%M:%S')
    time.sleep(0.1) # just in case there is a delay in finishing file writing

    conversion_status, mp4_path = video_processing.convert_h264_to_mp4(path, video_file, framerate)
    # Delete h264
    if conversion_status:
        video_processing.delete_file(full_video_path)
    print(f'Recording successfully captured in {mp4_path}')
