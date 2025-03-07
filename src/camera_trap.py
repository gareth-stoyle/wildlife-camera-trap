from animal_detection import detect_animal
import asyncio
import cv2
import datetime
from picamera2 import Picamera2
# from fake_capture import FakeCamera
from logger import customLogger
import numpy as np
from queue import Queue 
import time
from typing import Optional

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
        still_config = self.camera.create_still_configuration(main={"size": (1280, 720)})
        self.camera.configure(still_config)

        self.frame_queue = asyncio.Queue() 
        self.capturing = True

    async def capture_frames(self):
        """Captures frames at 24 FPS and put them in the queue."""
        self.camera.start()
        capture_count = 0
        frame_interval = 1 / 24
        next_frame_time = time.time()

        while self.capturing:
            now = time.time()
            if now >= next_frame_time:
                capture_count += 1
                logger.debug(f"Capturing image #{capture_count} and adding to Queue")
                frame = self.camera.capture_array()

                await self.frame_queue.put((capture_count, frame))
                logger.debug(f"Queue size: {self.frame_queue.qsize()}")

                next_frame_time += frame_interval
                if next_frame_time < now: # Prevent drift
                    next_frame_time = now + frame_interval
            
                # Put the kibosh on capturing for ~10s if our Queue in RAM
                # gets to about 4gb. This will only happen after ~500
                # frames of continuous motion detection (unlikely).
                if self.frame_queue.qsize() > 1100:
                    logger.warning(f"Pausing capture for 10s to reduce Queue.")
                    next_frame_time += 10

                # Log every 100 frames
                if capture_count % 100 == 0:
                    logger.info(f"Captured {capture_count} frames")

            await asyncio.sleep(0.0001)
        
        self.camera.stop()
        logger.info("Camera capturing stopped.")

    async def process_frames(self):
        """Processes frames asynchronously."""
        avg = None
        self.start_time = time.time()
        self.camera.start()
        time.sleep(2)
        frame_times = Queue(maxsize=10) # counter for estimating FPS

        while self.capturing or not self.frame_queue.empty():
            try:
                image_id, frame = await asyncio.wait_for(self.frame_queue.get(), timeout=3.0)
                logger.debug(f"Queue size: {self.frame_queue.qsize()}")
                start = time.time()

                avg = await self._process_single_frame(frame, avg, image_id)

                frame_time = time.time() - start
                logger.debug(f"full processing for frame #{image_id} took {frame_time} seconds")
                frame_times.put(frame_time)
                if frame_times.qsize() == 10:
                    frame_times.get()
                fps = int(1 / (np.mean(frame_times.queue)))
                logger.debug(f"FPS: {fps}")

                # Log every 100 frames
                if image_id % 100 == 0:
                    logger.info(f"Processed {image_id} frames")

            except asyncio.QueueEmpty:
                await asyncio.sleep(0)  # Yield control if queue is empty
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

    async def _process_single_frame(self, frame, avg, image_id):
        detected = False
        # Get contours if motion is detected
        avg, cnts = await self._detect_motion(frame, DELTA_THRESH, avg)
        
        # Find the biggest cnt and check if it's big enough to register
        if cnts:
            big_cnt = max(cnts, key=cv2.contourArea)
            detected = cv2.contourArea(big_cnt) > MIN_AREA

        if detected:
            logger.debug('Motion detected, feeding biggest contour into model')
            (x, y, w, h) = cv2.boundingRect(big_cnt)
            img = await self._prep_img_for_inf(frame, x, y, w, h)
            species, confidence = await detect_animal(img)
            # Draw bounding box on the frame with label
            cv2.putText(frame, f'{species}, {int(confidence)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 3)

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d---%H-%M-%S-%f')
        cv2.imwrite(f"{self.path}/{timestamp}.jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        logger.debug(f"Saved #{image_id}: {self.path}/{timestamp}.jpg")

        return avg

    async def _detect_motion(self, frame, delta_thresh, avg) -> tuple:
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
    
    async def _prep_img_for_inf(self, frame, x, y, w, h, target_size=300):
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
    
