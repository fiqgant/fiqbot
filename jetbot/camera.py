import traitlets
from traitlets.config.configurable import SingletonConfigurable
from traitlets import Int, Float, Unicode, Bool
import numpy as np
import cv2
import threading
import time

try:
    import jetcam
    from jetcam.csi_camera import CSICamera
    JETCAM_AVAILABLE = True
except ImportError:
    JETCAM_AVAILABLE = False
    print("Warning: jetcam not available, using OpenCV fallback")


class CSICameraWrapper(SingletonConfigurable):
    """Wrapper for CSI camera (IMX219) using jetcam or OpenCV fallback"""
    
    value = traitlets.Any()
    
    width = Int(default_value=224).tag(config=True)
    height = Int(default_value=224).tag(config=True)
    fps = Int(default_value=30).tag(config=True)
    capture_width = Int(default_value=3280).tag(config=True)
    capture_height = Int(default_value=2464).tag(config=True)
    
    def __init__(self, *args, **kwargs):
        super(CSICameraWrapper, self).__init__(*args, **kwargs)
        self.camera = None
        self.running = False
        self.thread = None
        
        if JETCAM_AVAILABLE:
            try:
                self.camera = CSICamera(
                    width=self.width,
                    height=self.height,
                    capture_width=self.capture_width,
                    capture_height=self.capture_height,
                    fps=self.fps
                )
                self.value = self.camera.value
                self.camera.observe(self._update_value, names='value')
            except Exception as e:
                print(f"Error initializing jetcam: {e}")
                self._init_opencv_camera()
        else:
            self._init_opencv_camera()
    
    def _init_opencv_camera(self):
        """Fallback to OpenCV for CSI camera"""
        print("Using OpenCV for CSI camera")
        # GStreamer pipeline for CSI camera on Jetson
        gstreamer_pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            f"width={self.capture_width}, height={self.capture_height}, "
            f"format=NV12, framerate={self.fps}/1 ! "
            "nvvidconv flip-method=0 ! "
            f"video/x-raw, width={self.width}, height={self.height}, "
            "format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink"
        )
        
        self.camera = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        if not self.camera.isOpened():
            # Try simpler pipeline
            self.camera = cv2.VideoCapture(0)
        
        self.running = True
        self.thread = threading.Thread(target=self._update_frame)
        self.thread.daemon = True
        self.thread.start()
    
    def _update_frame(self):
        """Update frame in background thread for OpenCV"""
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                self.value = frame
            time.sleep(1.0 / self.fps)
    
    def _update_value(self, change):
        """Update value when using jetcam"""
        self.value = change['new']
    
    def read(self):
        """Read current frame"""
        if JETCAM_AVAILABLE and hasattr(self.camera, 'value'):
            return self.camera.value
        return self.value
    
    def release(self):
        """Release camera resources"""
        self.running = False
        if self.thread:
            self.thread.join()
        if hasattr(self.camera, 'release'):
            self.camera.release()
        elif hasattr(self.camera, 'stop'):
            self.camera.stop()


# Alias for compatibility
Camera = CSICameraWrapper

