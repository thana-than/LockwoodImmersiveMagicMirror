import signal
import cv2
import time
import argparse
import tkinter
import threading
from PIL import Image, ImageTk, ImageColor
from queue import SimpleQueue, Empty
import numpy
import configparser
import json
import os
import socket

PROJECT_NAME = 'Magic Mirror'

# region Argument Parsing

parser = argparse.ArgumentParser(
    prog=f'python mirror.py',
    description='Magic Mirror prop used for Lockwood Immersive\'s HES Ball 2025 integration.',
    epilog='Created by Than | https://github.com/thana-than/LockwoodImmersiveMagicMirror')
parser.add_argument('-c', '--camera', default=0, help='Index of camera device. Default is 0 (the default camera device).')
parser.add_argument('-d', '--debug', action='store_true', default=False, help='Shows debug frame and detection state. Defaults to False.')
parser.add_argument('-f', '--fps', default=30, help='Set the desired Frames Per Second. Defaults to 30.')
parser.add_argument('-p', '--port', default=5005, help='Port where sequencing data is sent. Useful for external rendering.')
parser.add_argument('-m', '--model', default='res/cascade.xml', help='Path to the cascade model used in image detection.')
parser.add_argument('-a', '--aruco', action='store_true', default=False, help='Use aruco marker mode instead of image detection model.')
args = parser.parse_args()

# endregion

# region UDP

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
IP = "127.0.0.1" #localhost
PORT = int(args.port)


# endregion

# region JSON File

DEFAULT_JSON_SCHEMA = {
    "label": "blue",
    "display_color": "#0000FF",
    "detect_colors": [
        "#0078FD",
    ]
}

DEFAULT_JSON_DATA =[
        {
            "label": "blue",
            "display_color": "#0000FF",
            "detect_colors": [
                "#0078FD",
                "#0037AC",
                "#017CFE",
                "#03AEFE",
                "#0044C3",
                "#006CFB",
                "#003BA9",
                "#0045BC",
                "#D2F8FB",
                "#B4D9FC",
                "#CAFBFD",
                "#93D3F8",
                "#D0FBFA",
            ],
        },
        {
            "label": "red",
            "display_color": "#FF0000",
            "detect_colors": [
                "#F40700",
                "#970700",
                "#8B0600",
                "#FD9A39",
                "#FCF491",
                "#FBBB6D"
            ],
        },
        {
            "label": "green",
            "display_color": "#00FF00",
            "detect_colors": [
                "#03E572",
                "#00A04A",
                "#00914A",
                "#005829",
                "#03E773",
                "#86FABD",
                "#80D69C",
                "#9EFBFB",
                "#74FDDE",
                "#6DE9B0",
                "#A1FACA",
                "#E1FCFC",
                "#629270",
                "#7CC091",
                "#68F7C3",
                "#F7FDFD",
                "#E7FBFD",
                "#C9FCFD"
            ],
        },
    ]

json_file = 'candles.json'

def validate_json(data):
    for i in range(len(data)):
        schema = DEFAULT_JSON_SCHEMA
        if i < len(DEFAULT_JSON_DATA):
            schema = DEFAULT_JSON_DATA[i]

        for key in schema:
            data[i][key] = data[i].get(key, schema[key])

    return data
        

def load_candles_from_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = DEFAULT_JSON_DATA
    
    data = validate_json(data)

    candles = []
    for i in range(len(data)):
        obj = data[i]
        candles.append(Candle(obj["display_color"], obj["detect_colors"]))

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

    return candles

# endregion

# region Config File

config_path = 'config.ini'
config = configparser.ConfigParser()

config['DETECTION'] = {
    'Detection_Build_Speed': 3.0,
    'Detection_Reduce_Speed': .5,
    'Detection_Threshold': 4,
    'Detection_VideoScaleFactor': 1.2,
    }

config.read(config_path)

with open(config_path, 'w') as configfile:
    config.write(configfile)

# endregion

# region Fields and Variables
CAM_DEVICE = int(args.camera) # Index of the camera device thats to be displayed
DEBUG = args.debug
ARUCO = args.aruco
FRAMES_PER_SECOND = int(args.fps)

detect_accel = float(config['DETECTION']['Detection_Build_Speed']) # how fast the face state is registered
detect_deccel = float(config['DETECTION']['Detection_Reduce_Speed']) # how fast the face state is deregistered

detection_threshold =  int(config['DETECTION']['Detection_Threshold']) # thresholds that eliminate false positives, lower if detection is spotty
detection_video_scale_factor = float(config['DETECTION']['Detection_VideoScaleFactor']) # reduce image size for optimization (1 / VIDEO_SCALE_FACTOR = scale percentage)

seconds_delay = 1.0 / float(FRAMES_PER_SECOND)
ms_delay = int(seconds_delay * 1000)
app = None

if ARUCO:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_parameters = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_parameters)
else:
    model_classifier = cv2.CascadeClassifier('res/cascade.xml')

# endregion

# region Methods

def to_lab(bgr):
    bgr = numpy.uint8([[bgr]])
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0][0]
    return lab

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
    bounding_boxes = classifier.detectMultiScale(gray_image, detection_video_scale_factor, detection_threshold)
    return bounding_boxes

def scale_rect(rect, percent):
    x,y,w,h = rect

    c_x = x + w / 2
    c_y = y + h / 2

    w = int(w * percent)
    h = int(h * percent)

    x = int(c_x - w / 2)
    y = int(c_y - h / 2)

    return (x,y,w,h)

def sample_color_in_bounds(video_frame, bounds): #, brightness_threshold=210):
    x, y, w, h = bounds
    roi = video_frame[y : y + h, x : x + w]
    avg_color = numpy.mean(roi, axis=(0,1))
    return avg_color.astype(int).tolist()

def debug_draw_rect_bounds_rgb(video_frame, bounds, draw_color):
    debug_draw_rect_bounds_bgr(video_frame, bounds, (draw_color[2], draw_color[1], draw_color[0]))

def debug_draw_rect_bounds_bgr(video_frame, bounds, draw_color):
    if not DEBUG:
        return
    x = bounds[0]
    y = bounds[1]
    w = bounds[2]
    h = bounds[3]
    cv2.rectangle(video_frame, (x,y), (x+w,y+h), draw_color, 2)

def clamp(n, smallest, largest): return max(smallest, min(n, largest)) 

# endregion

# region Candle Class
class Candle():
    def __init__(self, display_color, detect_colors):
        self.display_color = ImageColor.getcolor(display_color, "RGB")
        self.detect_colors = [ImageColor.getcolor(color, "RGB") for color in detect_colors]
        self.lab_colors = [to_lab((rgb[2], rgb[1], rgb[0])) for rgb in self.detect_colors]
        self.detection = 0
        self.bounds = (0,0,0,0)
        self.force_detection = False

    def get_data(self):
        return {
            "detection" : self.detection,
            "bounds" : self.bounds
        }      
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
    
# region Candle Sequencing

class CV2_Sequencer(CV2_Render):
    def __init__(self, source, x = 0, y = 0, candles = []):
        super().__init__(source, x, y)

        self.candles = candles

        # build out color_labs tables
        labs = []
        self.candles_lab_index = []
        for i, candle in enumerate(self.candles):
            labs += candle.lab_colors
            self.candles_lab_index += [i] * len(candle.lab_colors)
        self.candle_labs = numpy.array(labs)
        self.lab_tests = self.candle_labs # could potentially add "duds" to the end of this array to create color weights that the test should avoid

        self.current_sequence = []
        self.last_state_change = 0

        self.prev_time = time.time()
        self.detect_time = 0
        self.queue_clear = False

        self.state_name = "DETECT"

    def incorrect_response(self):
        print("SEQUENCE " + ",".join(map(str, self.current_sequence)) + " INCORRECT")
        self.queue_clear = True
        self.state_name = "INCORRECT"
        # play_incorrect_video()

    def correct_response(self):
        print("SEQUENCE CORRECT")
        self.queue_clear = True
        self.state_name = "CORRECT"
        # play_correct_video()

    def is_in_sequence(self, candle_index):
        return candle_index in self.current_sequence
    
    def debug_toggle_candle(self, candle_index):
        if candle_index < 0 or candle_index >= len(self.candles):
            print("Candle at index {candle_index} does not exist!")
            return
        
        candle = self.candles[candle_index]
        candle.force_detection = not candle.force_detection
        print("Candle {candle_index} ", "on! " if candle.force_detection else "off!")

    def try_add_to_sequence(self, candle_index):        
        if self.is_in_sequence(candle_index):
            return
        print("ADDED CANDLE " + str(candle_index) + " TO SEQUENCE")
        self.current_sequence.append(candle_index)
        self.process_sequence_state()

    def clear_sequence(self):
        print("DETECT STATE")
        self.queue_clear = False
        self.state_name = "DETECT"
        self.current_sequence.clear()
        for candle in self.candles:
            candle.force_detection = False
            candle.detection = 0

    def get_sequence_data(self):
        return {
            "order": self.current_sequence,
            "state": self.state_name,
        }
    
    def find_closest_candle(self, input_color):
        # finds the closest candle lab color to the input_color and returns that specific candle
        input_lab = to_lab(input_color)

        color_labs = self.lab_tests
        distances = numpy.linalg.norm(color_labs - input_lab, axis=1)
        i = numpy.argmin(distances)
        if i >= len(self.candles_lab_index):
            return None
        return self.candles[self.candles_lab_index[i]]

    def model_update_candle_bounds(self, video_frame):
        bounds_collection = detect_bounding_box(video_frame, model_classifier)  # apply the function we created to the video frame 

        for candle in self.candles:
            candle.bounds = None

        for bounds in bounds_collection:
            bounds = scale_rect(bounds, .5)

            avg_color = sample_color_in_bounds(video_frame, bounds)
            debug_draw_rect_bounds_bgr(video_frame, bounds, avg_color)
            candle = self.find_closest_candle(avg_color)
            if candle:
                candle.bounds = bounds

    def aruco_update_candle_bounds(self, video_frame):
        in_color = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = aruco_detector.detectMarkers(in_color)

        for candle in self.candles:
            candle.bounds = None

        if ids is None:
            return
        
        ids = ids.flatten()
        for id, candle in enumerate(self.candles):
            if id in ids:
                idx = list(ids).index(id)
                pts = corners[idx].reshape((4, 2)).astype(int)
                candle.bounds = cv2.boundingRect(pts)

    def candle_detection_step(self, delta_time):                
        for i in range(len(self.candles)):
            candle = self.candles[i]

            visible = candle.force_detection or candle.bounds != None
            detect_step = detect_accel if visible else -detect_deccel

            candle.detection += delta_time * detect_step
            candle.detection = clamp(candle.detection, 0.0, 1.0)

    def candle_sequence_step(self):
        for i, candle in enumerate(self.candles):
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
                debug_draw_rect_bounds_rgb(video_frame, candle.bounds, candle.display_color)

    def debug_draw_sequence_state(self, video_frame):
        if not DEBUG:
            return
        for i in range(0, len(self.current_sequence)):
            spacing = i * 50
            display_color = self.candles[self.current_sequence[i]].display_color
            cv2.circle(video_frame, center=(50 + spacing, 50), radius=20, color=(display_color[2], display_color[1], display_color[0]), thickness=-1)

    def upload_state(self):
        candle_data = [c.get_data() for c in self.candles]
        sequence_data = self.get_sequence_data()
        data = {
            "candles": candle_data,
            "sequence": sequence_data
        }
        json_data = json.dumps(data)      
        sock.sendto(json_data.encode(), (IP, PORT))
        return

    def sequence_update(self, video_frame, delta_time):
        if ARUCO:
            self.aruco_update_candle_bounds(video_frame)
        else:
            self.model_update_candle_bounds(video_frame)

        self.candle_detection_step(delta_time)

        self.debug_draw_detected_candles(video_frame, self.candles)
        
        if self.queue_clear:
            for candle in self.candles:
                if candle.force_detection:
                    candle.detection = 1.0
                    candle.force_detection = False
                if candle.detection > 0:
                    return
            self.clear_sequence()

        self.candle_sequence_step()
        self.debug_draw_sequence_state(video_frame)
        self.upload_state()

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
    def __init__(self, window):
        self.running = True

        self.threads = []
        self.stop_event = threading.Event()

        self.hasWindow = window is not None
        if self.hasWindow:
            self.window = window
            self.canvas = tkinter.Canvas(window, width=self.window.winfo_reqwidth(), height=self.window.winfo_reqheight(), bg="black")
            self.width = 1
            self.height = 1
            self.canvas.pack(fill="both", expand=True)
            self.image_id = self.canvas.create_image(self.width / 2, self.height / 2, anchor=tkinter.CENTER)
            self.canvas.bind("<Configure>", self.on_resize)
            self.window.protocol("WM_DELETE_WINDOW", self.cleanup)
            self.window.bind("<Escape>", lambda e: self.cleanup())

        self.update()

    def on_resize(self,event):
        self.resize(event.x + event.width, event.y + event.height)

    def resize(self, width, height):
        self.width = width
        self.height = height

        self.image_id = self.canvas.create_image(self.width / 2, self.height / 2, anchor=tkinter.CENTER)

    def add(self, video_render):
        self.validate(video_render)

        self.threads.append(video_render)
        video_render.start()

        #* resize to this input if we haven't done a resize yet
        if self.hasWindow and self.width <= 1 and self.height <= 1:
            self.window.geometry(f'{video_render.w}x{video_render.h}')
            self.resize(video_render.w, video_render.h)

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
        
        if self.hasWindow:
            base_frame = Image.new('RGBA', (self.width, self.height))

        for thread in self.threads:
            try:
                frame = thread.queue.get_nowait()
                # frame caching
                thread.last_frame = frame
            except Empty:
                frame = thread.last_frame
                
            if not self.hasWindow or frame is None:
                continue
            
            f_height, f_width = frame.shape[:2]
            scale = min(self.width / f_width, self.height / f_height)
            new_w, new_h = int(f_width * scale), int(f_height * scale)

            x_offset = (self.width - new_w) // 2
            y_offset = (self.height - new_h) // 2

            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            frame_pil = Image.fromarray(frame)
            if frame.shape[2] == 3:
                frame_pil = frame_pil.convert('RGBA')
                
            base_frame.alpha_composite(frame_pil, dest=(x_offset, y_offset))

        if self.hasWindow:
            self.img = ImageTk.PhotoImage(base_frame)
            self.canvas.itemconfig(self.image_id, image=self.img)
            #* call next update
            self.window.after(ms_delay, self.update)
        else:
            time.sleep(seconds_delay)

    def cleanup(self):
        self.running = False
        self.stop_event.set()
        if self.hasWindow:
            self.window.destroy()
        for thread in self.threads:
            thread.cleanup()
        self.threads.clear()


def sequence_listener(sequencer):
    time.sleep(1)
    print("Program is running. Enter a number to add a candle.")
    while True:
        cmd = input("> ")
        if cmd.isdigit():
            sequencer.debug_toggle_candle(int(cmd))
        else:
            print(f"Unknown command: {cmd}")

def handle_exitsignal(sig, frame):
    print('')
    app.cleanup()
    exit(0)

#####* PROGRAM START *#####
if __name__ == "__main__":
    window = None
    # print opening message
    print(f'{PROJECT_NAME}. Created by Than.\nCamera Device: {CAM_DEVICE}\nFPS: {FRAMES_PER_SECOND}')
    if ARUCO:
        print('ARUCO marker mode active')
    if DEBUG:
        print('Debug mode active')
        # setup root window
        window = tkinter.Tk()
        window.title("Videos")

    app = App(window)
    
    sequencer = CV2_Sequencer(CAM_DEVICE, x=0,y=0, candles = load_candles_from_json(json_file))
    app.add(sequencer)

    signal.signal(signal.SIGINT, handle_exitsignal)
    listener = threading.Thread(target=sequence_listener, args=(sequencer,), daemon=True)
    listener.start()

    if window is not None:
        window.mainloop()
    else:
        while app.running:
            time.sleep(seconds_delay)
# endregion