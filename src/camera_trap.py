import picamera2

class CamTrap:
    def __init__(self, framerate=24, resolution='720x480', flip=False):
        self.camera = picamera.PiCamera()
        self.camera.framerate = framerate
        self.camera.resolution = resolution
        self.camera.color_effects = (128,128)
        if flip:
            self.camera.rotation = 180

    def start_recording(self, path, video_file):
        output = path + '/' + video_file
        self.camera.start_recording(output, bitrate=self.bitrate, quality=self.quality)

    def stop_recording(self):
        
        self.camera.stop_recording()
