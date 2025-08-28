import cv2
import time
import argparse

PROJECT_NAME = 'Magic Mirror'

# region Argument Parsing

parser = argparse.ArgumentParser(
    prog=f'python mirror.py',
    description='Magic Mirror prop used for Lockwood Immersive\'s HES Ball 2025 integration.',
    epilog='Created by Than | https://github.com/thana-than/LockwoodImmersiveMagicMirror')
parser.add_argument('-c', '--camera', default=0, help='Index of camera device. Default is 0 (the default camera device).')
parser.add_argument('-d', '--debug', action='store_true', default=False, help='Shows debug frame and detection state. Defaults to False.')
parser.add_argument('-f', '--fps', default=30, help='Set the desired Frames Per Second. Defaults to 30.')
args = parser.parse_args()

# endregion

# region Fields and Variables

CAM_DEVICE = int(args.camera) # Index of the camera device thats to be displayed
DEBUG = args.debug
FRAMES_PER_SECOND = int(args.fps)

DETECT_ACCEL = 2.0 # how fast the face state is registered
DETECT_DECCEL = 0.5 # how fast the face state is deregistered
DETECTION_THRESHOLD =  8 # thresholds that eliminate false positives, lower if face detection is spotty
FACE_SIZE = 40 # minimum allowed face size
VIDEO_SCALE_FACTOR = 1.1 # reduce image size for optimization (1 / VIDEO_SCALE_FACTOR = scale percentage)

clock_overlay = cv2.imread('res/clock.png', cv2.IMREAD_UNCHANGED)
check_overlay = cv2.imread('res/check.png', cv2.IMREAD_UNCHANGED)
video_capture = cv2.VideoCapture(CAM_DEVICE)

detect_time = 0
previous_time = time.time()
face_state = False
window_active = True

ms_delay = int(1.0 / float(FRAMES_PER_SECOND) * 1000)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# endregion

# region Methods
def detect_bounding_box(video_frame):
    gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY) # Greyscale for optimized detection
    faces = face_classifier.detectMultiScale(gray_image, VIDEO_SCALE_FACTOR, DETECTION_THRESHOLD, minSize=(FACE_SIZE,FACE_SIZE))
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

def clamp(n, smallest, largest): return max(smallest, min(n, largest)) 

# endregion

# region Main

#####* PROGRAM START *#####
# print opening message
print(f'{PROJECT_NAME}. Created by Than.\nCamera Device: {CAM_DEVICE}\nFPS: {FRAMES_PER_SECOND}')
if DEBUG:
    print('Debug mode active')

while window_active:
    current_time = time.time()
    delta_time = current_time - previous_time
    previous_time = current_time

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(video_frame)  # apply the function we created to the video frame 
    video_frame = debug_draw_detection(video_frame, faces)

    # detection threshold
    detect_step = DETECT_ACCEL if len(faces) > 0 else -DETECT_DECCEL 

    detect_time += delta_time * detect_step
    detect_time = clamp(detect_time, 0.0, 1.0)

    # face state assignment
    current_face_state = face_state
    if (detect_time == 0.0):
        current_face_state = False
    elif (detect_time == 1.0):
        current_face_state = True

    if face_state != current_face_state:
        face_state = current_face_state
        on_face_state_change(current_face_state)

    # face state update methods
    if (current_face_state): #detected
        video_frame = face_detected_update(video_frame, faces)
        video_frame = debug_draw_overlay(check_overlay, video_frame)
    elif (detect_step > 0): # charging up
        video_frame = debug_draw_overlay(clock_overlay, video_frame)

    if DEBUG:
        print('DETECTION TIME: ' + str(detect_time))

    # rendering
    cv2.imshow(PROJECT_NAME, video_frame)

    # frame delay
    keycode_input = cv2.waitKey(ms_delay)

    if keycode_input == 27:  # ESC key
        break
    
    #check if window is still active
    window_active = cv2.getWindowProperty(PROJECT_NAME, cv2.WND_PROP_VISIBLE) > 0

# release
video_capture.release()
cv2.destroyAllWindows()

# endregion