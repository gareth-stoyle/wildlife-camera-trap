import cv2
# from picamera2 import Picamera2
from fake_capture import FakeCamera
import time
from typing import Optional


class CamTrap:
    def __init__(self, path, duration):
        self.path: str = path
        self.duration: int = duration
        self.start_time: Optional[float] = None
        self.config: dict = get_config()

        # To use dummy footage instead of camera capture, move
        # fake_capture.py into src/ and instantiate FakeCamera()
        # instead of Picamera2()
        # self.camera: Picamera2 = Picamera2()
        self.camera = FakeCamera()
        video_config = self.camera.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
        self.camera.configure(video_config)

    def run(self) -> None:
        avg = None
        self.start_time = time.time()
        self.camera.start()
        time.sleep(2)
        frame = self.camera.capture_array()
        i = 0
        # loop until we run out of frames
        while frame is not None:
            i += 1
            detected = False
            # stop recording when duration met, but continue looping until
            # no frames left.
            if (time.time() - self.start_time) > self.duration:
                self.camera.stop()

            # take breaks to prevent overheating of camera & Pi
            if i % (24*60*10) == 0:
                self.camera.stop()
                time.sleep(10)
                self.camera.start()
            
            print(f'Captured frame, type: {type(frame)}, shape: {frame.shape}')
            avg, cnts = self._detect_motion(frame, self.config['delta_thresh'], avg)
            
            # Find the biggest cnt and check if it's big enough to register
            if cnts:
                big_cnt = max(cnts, key=cv2.contourArea)
                detected = cv2.contourArea(big_cnt) > self.config['min_area']
            else:
                detected = False

            if detected:
                # feed contours or the bigest contour into the model
                print("\n=======================================\n")
                print('Motion detected, feeding contours into model')
                print(f'cnts len: {len(big_cnt)}, cnts shape: {big_cnt.shape}')
                print('Model output X')
                print('drawing bounding box on frame')
                print('writing frame to path')
                print("\n=======================================\n")
                # Draw bounding box on the frame.
                (x, y, w, h) = cv2.boundingRect(big_cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (168, 98, 27), 3)
            else:
                # write frame and continue
                print('No motion detected, writing frame and continuing')
            
            cv2.imwrite(f"{self.path}_{i}.jpg", frame, [])
            print(f"Saved: {self.path}_{i}.jpg")
            frame = self.camera.capture_array()

    def _detect_motion(self, frame, delta_thresh, avg) -> tuple:
        '''Determines if motion was detected based on config variables'''
        # convert frame to grayscale, and blur it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        # if the average frame is None, initialize it
        if avg is None:
            avg = gray.copy().astype("float")
            return (avg, None)
        # accumulate the weighted average between the current frame and
        # previous frames, then compute the difference between the current
        # frame and running average
        cv2.accumulateWeighted(gray, avg, 0.5)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        
        # threshold the delta image, dilate the thresholded image to fill
        # in holes, then find contours on thresholded image
        thresh = cv2.threshold(frameDelta, delta_thresh, 255,
            cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
                    
        return (avg, cnts)


def get_config() -> dict:
    '''Return configuration variables in dictionary format'''
    config = {}
    
    # # get video details
    # config['fourcc'] = cv2.VideoWriter_fourcc(*'avc1')
    # config['fps'] = video.get(cv2.CAP_PROP_FPS)
    # config['total_frames'] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # config['frames_to_skip'] = int(config['fps']*60*15) # skip first x mins
    # config['frames_for_iteration'] = int(config['total_frames'] - config['frames_to_skip']) # skip last x mins for footage
    # config['frame_width'] = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # config['frame_height'] = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # motion detection config
    config['delta_thresh'] = 10
    config['min_area'] = 1000

    # # clip saving algo config
    # config['min_motion_frames'] = 4
    # config['min_clip_gap'] = config['fps']*30 # no motion enough to end clip and reset things
    # config['frames_to_shave'] = (config['min_clip_gap'] * 0.75)  # get rid of 75% of those last no motion frames

    # # drawing timestamp on frame
    # config['font_scale'] = 1  # Increase this value for bigger text
    # config['font_color'] = (255, 255, 255)  # White color
    # config['font_thickness'] = 2  # Thickness of the text
    # config['font'] = cv2.FONT_HERSHEY_SIMPLEX

    return config