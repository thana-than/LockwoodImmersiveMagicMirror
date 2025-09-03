import cv2
import time
import argparse
import tkinter
import threading
from PIL import Image, ImageTk
from queue import SimpleQueue, Empty

PROJECT_NAME = 'Magic Mirror'

# region Argument Parsing

parser = argparse.ArgumentParser(
    prog=f'python mirror.py',
    description='Magic Mirror prop used for Lockwood Immersive\'s HES Ball 2025 integration.',
    epilog='Created by Than | https://github.com/thana-than/LockwoodImmersiveMagicMirror')
parser.add_argument('-c', '--camera', default=0, help='Index of camera device. Default is 0 (the default camera device).')
parser.add_argument('-d', '--debug', action='store_true', default=False, help='Shows debug frame and detection state. Defaults to False.')
parser.add_argument('-f', '--fps', default=30, help='Set the desired Frames Per Second. Defaults to 30.')
parser.add_argument('-w', '--window', default=(1920, 1080), nargs=2, metavar=('WIDTH', 'HEIGHT'), help='Set the desired Window Size. Defaults to 1920 1080.')
args = parser.parse_args()

# endregion

# region Fields and Variables

CAM_DEVICE = int(args.camera) # Index of the camera device thats to be displayed
DEBUG = args.debug
FRAMES_PER_SECOND = int(args.fps)
WINDOW_SIZE = (int(args.window[0]), int(args.window[1]))

DETECT_ACCEL = 2.0 # how fast the face state is registered
DETECT_DECCEL = 0.5 # how fast the face state is deregistered
DETECTION_THRESHOLD =  8 # thresholds that eliminate false positives, lower if face detection is spotty
FACE_SIZE = 40 # minimum allowed face size
VIDEO_SCALE_FACTOR = 1.1 # reduce image size for optimization (1 / VIDEO_SCALE_FACTOR = scale percentage)

clock_overlay = cv2.imread('res/clock.png', cv2.IMREAD_UNCHANGED)
check_overlay = cv2.imread('res/check.png', cv2.IMREAD_UNCHANGED)

ms_delay = int(1.0 / float(FRAMES_PER_SECOND) * 1000)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# endregion

# region Methods
def detect_bounding_box(video_frame, classifier):
    gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY) # Greyscale for optimized detection
    faces = classifier.detectMultiScale(gray_image, VIDEO_SCALE_FACTOR, DETECTION_THRESHOLD, minSize=(FACE_SIZE,FACE_SIZE))
    return faces

def face_detected_update(video_frame, faces):
    return video_frame

def on_face_state_change(face_state):
    return

def debug_draw_detection(video_frame, faces):
    if DEBUG is False:
        return video_frame
    
    for (x, y, w, h) in faces:
        cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return video_frame

def debug_draw_overlay(overlay_image, video_frame):
    if DEBUG is False:
        return video_frame
    
    # Resize overlay if needed
    scale = 0.2  # 20% of original size
    overlay_resized = cv2.resize(
        overlay_image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )

    oh, ow = overlay_resized.shape[:2]
    vh, vw = video_frame.shape[:2]

    # Position: bottom-right
    x, y = vw - ow, vh - oh

    if overlay_resized.shape[2] == 4:
        # If overlay has alpha channel
        alpha = overlay_resized[:, :, 3] / 255.0
        for c in range(0, 3):
            video_frame[y:vh, x:vw, c] = (
                alpha * overlay_resized[:, :, c]
                + (1 - alpha) * video_frame[y:vh, x:vw, c]
            )
    else:
        # No alpha channel, just overwrite
        video_frame[y:vh, x:vw] = overlay_resized

    return video_frame

def wait_process_frame(queue, video_frame):
    while not queue.empty():
        try: queue.get_nowait()
        except Empty: break
    video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for Tkinter
    queue.put(video_frame)

def clamp(n, smallest, largest): return max(smallest, min(n, largest)) 

# endregion

# region Render Thread Classes

class VideoRender(threading.Thread):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.stop_event = None
        self.queue = SimpleQueue()
        self.img_id = None
        self.img = None
        self.video_capture = cv2.VideoCapture(self.source)
        self.personal_stop_event = threading.Event()

    def run(self):
        print('STARTING VIDEO RENDER THREAD')
        while not self.personal_stop_event.is_set() and (self.stop_event is None or not self.stop_event.is_set()):
            result, video_frame = self.video_capture.read()  # read frames from the video
            if result is False:
                break  # terminate the loop if the frame is not read successfully
            
            video_frame = self.process_frame(video_frame)

            # frame delay
            wait_process_frame(self.queue, video_frame)
        
        self.cleanup()

    def process_frame(self, video_frame):
        return video_frame

    def cleanup(self):
        self.personal_stop_event.set()
        self.video_capture.release()
        try:
            self.join(timeout=1)
        except RuntimeError:
            pass


class CV2_Detection(VideoRender):    
    def __init__(self, source, classifier):
        super().__init__(source)
        self.prev_time = time.time()
        self.detect_time = 0
        self.face_state = False
        self.classifier = classifier

    def process_frame(self, video_frame):
        current_time = time.time()
        delta_time = current_time - self.prev_time
        self.prev_time = current_time

        faces = detect_bounding_box(video_frame, self.classifier)  # apply the function we created to the video frame 
        video_frame = debug_draw_detection(video_frame, faces)

        # detection threshold
        detect_step = DETECT_ACCEL if len(faces) > 0 else -DETECT_DECCEL 

        self.detect_time += delta_time * detect_step
        self.detect_time = clamp(self.detect_time, 0.0, 1.0)

        # face state assignment
        current_face_state = self.face_state
        if (self.detect_time == 0.0):
            current_face_state = False
        elif (self.detect_time == 1.0):
            current_face_state = True

        if self.face_state != current_face_state:
            self.face_state = current_face_state
            on_face_state_change(current_face_state)

        # face state update methods
        if (current_face_state): #detected
            video_frame = face_detected_update(video_frame, faces)
            video_frame = debug_draw_overlay(check_overlay, video_frame)
        elif (detect_step > 0): # charging up
            video_frame = debug_draw_overlay(clock_overlay, video_frame)

        if DEBUG:
            print('DETECTION TIME: ' + str(self.detect_time))

        return video_frame
    
# endregion

# region Main

class App:
    def __init__(self, root, size=(320, 240)):
        self.root = root
        self.stop_event = threading.Event()

        self.size = size
        self.w, self.h = size
        self.canvas = tkinter.Canvas(root, width=self.w, height=self.h, bg="black")
        self.canvas.pack()

        # threads
        self.threads = []

        self.update()
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup)
        self.root.bind("<Escape>", lambda e: self.cleanup())

    def add(self, video_render):
        self.validate(video_render)

        self.threads.append(video_render)
        video_render.start()

    def validate(self, video_render):
        video_render.stop_event = self.stop_event
        if video_render.img_id is None:
            video_render.img_id = self.canvas.create_image(self.w / 2, self.h / 2, anchor=tkinter.CENTER)

    def remove(self, video_render):
        self.threads.remove(video_render)
        video_render.cleanup()

    def update(self):
        if self.stop_event.is_set():
            return
        
        for self.thread in self.threads:
            try:
                frame = self.thread.queue.get_nowait()
                self.thread.img = ImageTk.PhotoImage(Image.fromarray(frame))
                self.canvas.itemconfig(self.thread.img_id, image=self.thread.img)
            except Empty:
                pass

        self.root.after(ms_delay, self.update)

    def cleanup(self):
        self.stop_event.set()
        self.root.destroy()
        for self.thread in self.threads:
            self.thread.cleanup()
        self.threads.clear()

#####* PROGRAM START *#####
if __name__ == "__main__":
    # setup root window
    root = tkinter.Tk()
    root.title("Videos")

    # print opening message
    print(f'{PROJECT_NAME}. Created by Than.\nCamera Device: {CAM_DEVICE}\nFPS: {FRAMES_PER_SECOND}')
    if DEBUG:
        print('Debug mode active')
    
    app = App(root, size=WINDOW_SIZE)
    app.add(CV2_Detection(CAM_DEVICE, face_classifier))

    root.mainloop()

# endregion