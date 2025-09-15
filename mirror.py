import cv2
import time
import argparse
import tkinter
import threading
from PIL import Image, ImageTk
from queue import SimpleQueue, Empty
import numpy
import configparser

#TODO video speed (framerate?) fix
#TODO render scaling
# Old TODOS
## cleaner call for video overlay (currently in on_face_state_change)
## video position and scaling based on position of object (?) (probably wont work with mirror surface perspective)

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

# region Config File

config_path = 'config.ini'
config = configparser.ConfigParser()
config['DEFAULT'] = {
    'First_Low_RGB': '0, 82, 158',
    'First_High_RGB': '0, 197, 255',
    
    'Second_Low_RGB': '174, 77, 63',
    'Second_High_RGB': '255, 249, 168',
    
    'Third_Low_RGB': '24, 176, 119',
    'Third_High_RGB': '124, 240, 253'
    }

config['COOLDOWNS'] = {
    'Cooldown_Sequence_Correct': 5,
    'Cooldown_Sequence_Incorrect': 5,
    }

config.read(config_path)

with open(config_path, 'w') as configfile:
    config.write(configfile)

def config_to_rgb(parameter):
    str = config['DEFAULT'][parameter]
    rgb = [int(s.strip()) for s in str.split(',')]
    return (rgb[0],rgb[1],rgb[2])

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

correct_video_filename = 'res/sequence_correct.mp4'
incorrect_video_filename = 'res/sequence_incorrect.mp4'
video_colorkey = '#00FF00'

ms_delay = int(1.0 / float(FRAMES_PER_SECOND) * 1000)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

app = None

# endregion

# region Methods

def hex_to_hsv_bounds(hex_color, threshold=40):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    rgb_np = numpy.uint8([[rgb]])
    hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)[0][0]

    h, s, v = [int(x) for x in hsv]
    lower = numpy.array([
        max(h - threshold, 0),
        max(s - threshold, 0),
        max(v - threshold, 0)
    ])
    upper = numpy.array([
        min(h + threshold, 179),
        min(s + threshold, 255),
        min(v + threshold, 255)
    ])
    return lower, upper

def detect_bounding_box(video_frame, classifier):
    gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY) # Greyscale for optimized detection
    faces = classifier.detectMultiScale(gray_image, VIDEO_SCALE_FACTOR, DETECTION_THRESHOLD, minSize=(FACE_SIZE,FACE_SIZE))
    return faces

def find_color_bounds(in_color, low, high):
    # Threshold on brightness (tune values)
    mask = cv2.inRange(in_color, low, high)

    # Morphology to clean up
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, numpy.ones((5,5), numpy.uint8))

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h > 500:  # filter small noise
            return (x,y,w,h)

    return None

def debug_draw_rect_bounds(video_frame, bounds, draw_color):
    if not DEBUG:
        return
    x = bounds[0]
    y = bounds[1]
    w = bounds[2]
    h = bounds[3]
    d_r = draw_color[0]
    d_g = draw_color[1]
    d_b = draw_color[2]
    cv2.rectangle(video_frame, (x,y), (x+w,y+h), (d_b,d_g,d_r), 2)

def face_detected_update(video_frame, faces):
    return video_frame

def play_correct_video():
    video = CV2_Render(correct_video_filename)
    video.set_color_key(video_colorkey)
    app.add(video)
    return

def play_incorrect_video():
    video = CV2_Render(incorrect_video_filename)
    video.set_color_key(video_colorkey)
    app.add(video)
    return

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

def clamp(n, smallest, largest): return max(smallest, min(n, largest)) 

# endregion

# region Render Thread Classes

class Video_Render(threading.Thread):
    def __init__(self, source, x = 0, y = 0):
        super().__init__()
        self.stop_event = None
        self.source = source
        self.queue = SimpleQueue()
        self.active = True
        self.personal_stop_event = threading.Event()
        self.x = x
        self.y = y
        self.w = 0
        self.h = 0
    
    def read_frame(self):
        return None

    def run(self):
        print('STARTING VIDEO RENDER THREAD')
        while not self.personal_stop_event.is_set() and (self.stop_event is None or not self.stop_event.is_set()):
            video_frame = self.read_frame()
            if video_frame is None:
                break
            
            video_frame = self.post_process_frame(video_frame)

            # wait delay
            while not self.queue.empty():
                try: self.queue.get_nowait()
                except Empty: break

            self.queue.put(video_frame)
        
        print('ENDING VIDEO RENDER THREAD')
        self.cleanup()

    def post_process_frame(self, video_frame):
        return video_frame
    
    def cleanup(self):
        if not self.active:
            return

        self.active = False
        self.personal_stop_event.set()
        if self.exitCallback is not None:
            self.exitCallback()
    
        try:
            self.join(timeout=1)
        except RuntimeError:
            pass
        
class CV2_Render(Video_Render):
    def __init__(self, source, x = 0, y = 0):
        super().__init__(source, x, y)
        self.video_capture = cv2.VideoCapture(source)
        self.color_key = None
        self.w = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_frame(self):
        result, video_frame = self.video_capture.read()
        if result is False:
            return None
        return video_frame
    
    def set_color_key(self, hex_color, threshold=40):
        lower, upper = hex_to_hsv_bounds(hex_color, threshold)
        self.color_key = (lower, upper)
    
    def apply_color_key(self, video_frame):
        # HSV Value
        lower_color = numpy.array([35, 80, 80]) 
        upper_color = numpy.array([85, 255, 255])
        hsv = cv2.cvtColor(video_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Invert mask for foreground
        mask_inv = cv2.bitwise_not(mask)

        # Stack alpha to frame
        alpha = mask_inv
        bgr = video_frame if video_frame.shape[2] == 3 else video_frame[:, :, :3]
        rgba = cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], alpha])
        return rgba
    
    def post_process_frame(self, video_frame):
        if self.color_key is not None:
            video_frame = self.apply_color_key(video_frame)
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGRA2RGBA if video_frame.shape[2] == 4 else cv2.COLOR_BGR2RGBA)
        return video_frame
    
    def cleanup(self):
        super().cleanup()
        self.video_capture.release()
        

class CV2_Detection(CV2_Render):    
    def __init__(self, source, x = 0, y = 0, classifier = face_classifier):
        super().__init__(source, x, y)
        self.prev_time = time.time()
        self.detect_time = 0
        self.face_state = False
        self.classifier = classifier

    def post_process_frame(self, video_frame):
        current_time = time.time()
        delta_time = current_time - self.prev_time
        self.prev_time = current_time

        faces = detect_bounding_box(video_frame, self.classifier)  # apply the function we created to the video frame 
        items = faces# + candles
        video_frame = debug_draw_detection(video_frame, items)

        # detection threshold
        detect_step = DETECT_ACCEL if len(items) > 0 else -DETECT_DECCEL 

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
            video_frame = face_detected_update(video_frame, items)
            video_frame = debug_draw_overlay(check_overlay, video_frame)
        elif (detect_step > 0): # charging up
            video_frame = debug_draw_overlay(clock_overlay, video_frame)

        if DEBUG:
            return super().post_process_frame(video_frame)
        else:
            return None # hide frame if not debugging
    
# region Candle Sequencing

class Candle():
    def __init__(self, low_param, high_param):
        self.low = config_to_rgb(low_param)
        self.high = config_to_rgb(high_param)
        self.display_color = self.low

        self.detection = 0
        self.bounds = (0,0,0,0)

class CV2_Sequencer(CV2_Render):
    def __init__(self, source, x = 0, y = 0, candles = []):
        super().__init__(source, x, y)

        self.candles = candles
        self.current_sequence = []
        self.last_state_change = 0
        self.current_cooldown = 0

        self.cooldown_correct = float(config['COOLDOWNS']['Cooldown_Sequence_Correct'])
        self.cooldown_incorrect = float(config['COOLDOWNS']['Cooldown_Sequence_Incorrect'])
        self.cooldown_next = 0

        self.prev_time = time.time()
        self.detect_time = 0
        self.queue_clear = False

    def incorrect_response(self):
        print("SEQUENCE " + ",".join(map(str, self.current_sequence)) + " INCORRECT")
        self.queue_clear = True
        self.current_cooldown = self.cooldown_incorrect
        play_incorrect_video()

    def correct_response(self):
        print("SEQUENCE CORRECT")
        self.queue_clear = True
        self.current_cooldown = self.cooldown_correct
        play_correct_video()

    def is_in_sequence(self, candle_index):
        return candle_index in self.current_sequence

    def try_add_to_sequence(self, candle_index):
        if self.is_in_sequence(candle_index):
            return
        print("ADDED CANDLE " + str(candle_index) + " TO SEQUENCE")
        self.current_cooldown = self.cooldown_next
        self.current_sequence.append(candle_index)
        self.process_sequence_state()

    def clear_sequence(self):
        self.queue_clear = False
        self.current_sequence.clear()
        for candle in self.candles:
            candle.detection = 0

    def update_candle_bounds(self, video_frame):
        in_color = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                
        for candle in self.candles:
            candle.bounds = find_color_bounds(in_color, candle.low, candle.high)

    def candle_detection_step(self, delta_time):                
        for i in range(len(self.candles)):
            candle = self.candles[i]

            visible = candle.bounds != None
            detect_step = DETECT_ACCEL if visible else -DETECT_DECCEL
            candle.detection += detect_step

            candle.detection += delta_time * detect_step
            candle.detection = clamp(candle.detection, 0.0, 1.0)

            if candle.detection >= 1:
                self.try_add_to_sequence(i)

    def process_sequence_state(self):
        # Check if the candle sequence is finished and accurate
        if len(self.current_sequence) != len(self.candles):
            return
        
        for i in range(0, len(self.current_sequence)):
            if self.current_sequence[i] != i:
                self.incorrect_response()
                return
        self.correct_response()

    def debug_draw_detected_candles(self, video_frame, candles):
        if not DEBUG:
            return
        
        for candle in candles:
            if candle.bounds:
                debug_draw_rect_bounds(video_frame, candle.bounds, candle.display_color)

    def debug_draw_sequence_state(self, video_frame):
        if not DEBUG:
            return
        for i in range(0, len(self.current_sequence)):
            spacing = i * 50
            display_color = self.candles[self.current_sequence[i]].display_color
            cv2.circle(video_frame, center=(50 + spacing, 50), radius=20, color=(display_color[2], display_color[1], display_color[0]), thickness=-1)

    def sequence_update(self, video_frame, delta_time):
        self.current_cooldown -= delta_time

        self.update_candle_bounds(video_frame)
        
        if self.current_cooldown <= 0:
            self.current_cooldown = 0

            if self.queue_clear:
                self.clear_sequence()

            self.candle_detection_step(delta_time)

        self.debug_draw_detected_candles(video_frame, self.candles)
        self.debug_draw_sequence_state(video_frame)

    def post_process_frame(self, video_frame):
        current_time = time.time()
        delta_time = current_time - self.prev_time
        self.prev_time = current_time

        self.sequence_update(video_frame, delta_time)

        if DEBUG:
            return super().post_process_frame(video_frame)
        else:
            return None # hide frame if not debugging

#endregion

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

        self.image_id = self.canvas.create_image(self.w / 2, self.h / 2, anchor=tkinter.CENTER)

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
        video_render.exitCallback = lambda: self.remove(video_render)
        video_render.last_frame = None

    def remove(self, video_render):
        if video_render not in self.threads:
            return
        self.threads.remove(video_render)
        video_render.exitCallback = None
        video_render.cleanup()

    def update(self):
        if self.stop_event.is_set():
            return
        
        base_frame = Image.new('RGBA', (self.w, self.h))

        x_offset = self.w / 2
        y_offset = self.h / 2

        for thread in self.threads:
            try:
                frame = thread.queue.get_nowait()
                # frame caching
                thread.last_frame = frame
            except Empty:
                frame = thread.last_frame
                
            if frame is None:
                continue
            
            frame_pil = Image.fromarray(frame)
            if frame.shape[2] == 3:
                frame_pil = frame_pil.convert('RGBA')
            base_frame.alpha_composite(frame_pil, dest=(int(x_offset - (thread.w / 2) + thread.x), int(y_offset - (thread.h / 2) + thread.y)))

            
        self.img = ImageTk.PhotoImage(base_frame)
        self.canvas.itemconfig(self.image_id, image=self.img)

        self.root.after(ms_delay, self.update)

    def cleanup(self):
        self.stop_event.set()
        self.root.destroy()
        for thread in self.threads:
            thread.cleanup()
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
    
    #app.add(CV2_Detection(CAM_DEVICE, x=0,y=0, classifier=face_classifier))
    app.add(CV2_Sequencer(CAM_DEVICE, x=0,y=0, candles=[
        Candle('First_Low_RGB', 'First_High_RGB'),
        Candle('Second_Low_RGB', 'Second_High_RGB'),
        Candle('Third_Low_RGB', 'Third_High_RGB'),
    ]))

    root.mainloop()

# endregion