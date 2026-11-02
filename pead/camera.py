import os
import threading
import time
import cv2


class Camera:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.sequence = 0
        self.last_frame = 0.
        self.error = 'Camera disconnected'
        self.actual = {}
        self.thread = None
        self.stop_event = threading.Event()

    def start(self, index, width, height, fps):
        self.stop()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run,args=(index,width,height,fps),daemon=True)
        self.thread.start()

    def _run(self,index,width,height,fps):
        cap = cv2.VideoCapture(index,cv2.CAP_DSHOW if os.name=='nt' else cv2.CAP_ANY)
        try:
            if not cap.isOpened():
                raise RuntimeError('Cannot open camera; close other camera applications and check permissions')
            cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
            cap.set(cv2.CAP_PROP_FPS,fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            with self.lock:
                self.actual = {'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                               'fps':cap.get(cv2.CAP_PROP_FPS),'fourcc':int(cap.get(cv2.CAP_PROP_FOURCC))}
                self.error = None
            while not self.stop_event.is_set():
                ok,frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera frame read failed; reconnect the camera')
                with self.lock:
                    self.frame = frame
                    self.sequence += 1
                    self.last_frame = time.monotonic()
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            cap.release()
            with self.lock:
                self.frame = None

    def snapshot(self):
        with self.lock:
            if self.frame is None or time.monotonic()-self.last_frame>2:
                return None,self.sequence
            return self.frame.copy(),self.sequence

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=4)
            if self.thread.is_alive():
                raise RuntimeError('Camera driver is not responding; restart application before reconnecting')
        with self.lock:
            self.frame = None
            self.error = 'Camera disconnected'
            self.actual = {}
