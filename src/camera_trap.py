from animal_detection import detect_animal
import asyncio
import cv2
from picamera2 import Picamera2
# from fake_capture import FakeCamera
from logger import customLogger
import numpy as np
import time
from typing import Optional
from queue import Queue 

logger = customLogger("CamTrap", "outputs/app.log", debug=False)

DELTA_THRESH = 1
MIN_AREA = 5000

class CamTrap:
    def __init__(self, path, duration):
        self.path: str = path
        self.duration: int = duration
        self.start_time: Optional[float] = None

        # To use dummy footage instead of camera capture, move
        # fake_capture.py into src/ and instantiate FakeCamera()
        # instead of Picamera2()
        # self.camera = FakeCamera()
        self.camera: Picamera2 = Picamera2()
        video_config = self.camera.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
        still_config = self.camera.create_still_configuration(main={"size": (1920, 1080)})
        self.camera.set_controls({
            "NoiseReductionMode": 2,  # Enables denoising
            "AwbEnable": 1,           # Enables Auto White Balance
            "AeEnable": 1,            # Enables Auto Exposure
            "ExposureTime": 20000,    # Adjusts exposure for better brightness
            "AnalogueGain": 2.0       # Boosts brightness
        })
        self.camera.configure(still_config)

        self.frame_queue = asyncio.Queue() 
        self.capturing = True

    async def capture_frames(self):
        """Captures frames at 24 FPS and put them in the queue."""
        self.camera.start()
        frame_interval = 1 / 24
        next_frame_time = time.time()

        while self.capturing:
            now = time.time()
            if now >= next_frame_time:
                logger.info("capturing and adding to Queue")
                frame = self.camera.capture_array()
                # self.camera.capture_file("test.jpg")

                await self.frame_queue.put(frame)

                next_frame_time += frame_interval
                if next_frame_time < now:  # Prevent drift
                    next_frame_time = now + frame_interval

            await asyncio.sleep(0)
        
        self.camera.stop()
        logger.info("Camera capturing stopped.")

    async def process_frames(self):
        """Processes frames asynchronously."""
        avg = None
        self.start_time = time.time()
        self.camera.start()
        time.sleep(2)
        frame_times = Queue(maxsize=10) # counter for estimating FPS
        count = 0

        while self.capturing or not self.frame_queue.empty():
            try:
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=3.0)
                logger.warning(self.frame_queue.qsize())
                
                start = time.time()
                count += 1

                avg = await self._process_single_frame(frame, avg, count)

                frame_time = time.time() - start
                logger.debug(f"full processing for this frame took {frame_time} seconds")
                frame_times.put(frame_time)
                if frame_times.qsize() == 10:
                    frame_times.get()
                fps = int(1 / (np.mean(frame_times.queue)))
                logger.info(f"FPS: {fps}")
            except asyncio.QueueEmpty:
                await asyncio.sleep(0)  # Yield control if queue is empty
                logger.warning("Q is empty.")
            except asyncio.TimeoutError:
                if self.capturing:
                    raise
                logger.info("No more frames left to process, ending.")

    async def run(self) -> None:
        """Runs both capture and processing tasks concurrently."""
        capture_task = asyncio.create_task(self.capture_frames())
        process_task = asyncio.create_task(self.process_frames())

        try:
            await asyncio.sleep(self.duration)
            self.capturing = False
            await asyncio.gather(capture_task, process_task)
        except asyncio.CancelledError:
            self.capturing = False

    async def _process_single_frame(self, frame, avg, count):
        logger.info(f"======== Image: {count} ========")
        detected = False
        # Get contours if motion is detected
        avg, cnts = self._detect_motion(frame, DELTA_THRESH, avg)
        
        # Find the biggest cnt and check if it's big enough to register
        if cnts:
            big_cnt = max(cnts, key=cv2.contourArea)
            detected = cv2.contourArea(big_cnt) > MIN_AREA

        if detected:
            logger.info('Motion detected, feeding biggest contour into model')
            (x, y, w, h) = cv2.boundingRect(big_cnt)
            img = self._prep_img_for_inf(frame, x, y, w, h)
            species, confidence = detect_animal(img)
            # Draw bounding box on the frame with label
            cv2.putText(frame, f'{species}, {int(confidence)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (168, 98, 27), 3)
        else:
            await asyncio.sleep(0.02)
        
        cv2.imwrite(f"{self.path}_{count}.jpg", frame, [])
        logger.info(f"Saved: {self.path}_{count}.jpg")

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
    
    def _prep_img_for_inf(self, frame, x, y, w, h, target_size=300):
        """Take a bounding box of a given frame and extract a square img
        of 300x300 for inferencing"""
        max_side = max(w, h)

        # Center the square around bounding box
        cx, cy = x + w // 2, y + h // 2
        x_new = max(cx - max_side // 2, 0)
        y_new = max(cy - max_side // 2, 0)

        x_new = min(x_new, frame.shape[1] - max_side)
        y_new = min(y_new, frame.shape[0] - max_side)

        # Extract the square img
        square_img = frame[y_new:y_new + max_side, x_new:x_new + max_side]
        square_img = cv2.resize(square_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

        return square_img
    
