import threading
from threading import Lock
import cv2
import os
from py_topping.general_use import timeout

# Reference from https://medium.com/@teckyian/building-an-ai-security-camera-with-a-web-ui-in-100-lines-of-code-6d983586a9bf

class Camera:
    last_frame = None
    last_ready = None
    lock = Lock()

    @timeout(15)
    def __init__(self, rtsp_link):
        if os.path.isfile(rtsp_link) : self.isfile = True
        else : self.isfile = False
        self.capture = cv2.VideoCapture(rtsp_link)
        self.additional_config = { 'width' : self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                                 , 'height' : self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                                 }
        if not self.isfile :
            self.running = True
            self.thread = threading.Thread(target=self.rtsp_cam_buffer, name="rtsp_read_thread")
            self.thread.daemon = True
            self.thread.start()

    def rtsp_cam_buffer(self):
        while self.running :
            with self.lock:
                self.last_ready, self.last_frame = self.capture.read()
        self.capture.release()

    def getFrame(self):
        if not self.isfile :
            if (self.last_ready is not None) and (self.last_frame is not None):
                return True, self.last_frame.copy()
            else:
                return False, None
        else :
            return self.capture.read()

    def release(self) :
        self.running = False