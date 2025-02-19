import cv2

class FakeCamera:
    def __init__(self):
        self.video_path = "test_images/fox.mp4"
        self.capture = cv2.VideoCapture(self.video_path)

        if not self.capture.isOpened():
            print("Error: Could not open video.")
            exit()

    def start(self):
        pass
    
    def capture_array(self):
        ret, frame = self.capture.read()
        return frame
    
    def stop(self):
        self.capture.release()
        cv2.destroyAllWindows()

    def create_video_configuration(self, main):
        pass

    def configure(self, config):
        pass