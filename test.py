from picamera2 import Picamera2
import time

# Initialize Picamera2
picam2 = Picamera2()

# Configure the camera (optional)
picam2.configure(picam2.create_still_configuration())

# Start the camera
picam2.start()
time.sleep(2)  # Allow camera to adjust

# Capture an image and save it
picam2.capture_file("image.jpg")
picam2.capture_file("image.png")  # Save as PNG

print("Image captured and saved as 'image.jpg'")
picam2.close()