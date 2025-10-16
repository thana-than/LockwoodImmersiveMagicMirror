<br />
<div align="center">
  <h3 align="center">Lockwood Immersive's Magic Mirror</h3>

  <p align="center">
    Magic Mirror prop used for <a href="https://www.lockwoodimmersive.com/">Lockwood Immersive</a>'s <a href="https://www.handeyesociety.com/">Hand Eye Society</a> Ball 2025 integration.
    <br/>
    Created by <a href="https://thanathan.com">Than</a>.
  </p>
</div>

## Getting Started

### Prerequisites

<a href="https://www.python.org/downloads/release/python-3124/">Python</a> (v. 3.12.4 was used for this implementation)

### Renderer
Make sure you download the external <a href="https://github.com/thana-than/LockwoodImmersiveMagicMirrorRenderer">Godot Renderer</a> for better visuals!

### Installation

1. Clone the repo
```
git clone https://github.com/thana-than/LockwoodImmersiveMagicMirror.git
```
2. Install Python packages
```
pip install -r requirements.txt
```
3. Run command in project directory
```
python mirror.py --debug
```
   
## Usage
   ```
   python mirror.py [-h] [-c CAMERA] [-d] [-f FPS]

  -h, --help                                Shows this help message.
  -c CAMERA, --camera CAMERA                Index of camera device. Default is 0 (the default camera device).
  -d, --debug                               Shows debug frame and detection state. Defaults to False.
  -f FPS, --fps FPS                         Set the desired Frames Per Second. Defaults to 30.
  -p PORT, --port PORT                      Port where sequencing data is sent. Useful for external rendering.
  -w WIDTH HEIGHT, --window WIDTH HEIGHT    Set the desired Window Size. Defaults to 1920 1080.
  -m MODEL, --model MODEL                   Path to the cascade model used in image detection.
   ```
## Json Configuration
Array is in order of correct candle sequence. </br>
`candles.json` *(will appear after first run)*
```
[
        {
            "label": "blue",                Name for debugging only.
            "display_color": "#0000FF",   Color for debugging.
            "detect_colors": [              Colors used to determine which candle is in view. Ideal to have lots of color samples.
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
```
## Extra Configuration
`config.ini` *(will appear after first run)*
```
[DETECTION]
detection_build_speed = 3.0                 The speed at which a target is counted as detected.
detection_reduce_speed = 1.0                The speed at which a target is removed from detection (when out of view).
detection_threshold = 4                     Thresholds that eliminate false positives, higher means harsher threshold. Lower if detection is spotty.
detection_videoscalefactor = 1.2            Reduce image size for optimization (1 / VIDEO_SCALE_FACTOR = scale percentage)
```
## Contact

Than More - [thanathan.com](https://thanathan.com/) - than@thanathan.com
<br/>
Lockwood Immersive - [www.lockwoodimmersive.com](https://www.lockwoodimmersive.com/) - hello@lockwoodimmersive.com
<br/>
Hand Eye Society - [www.handeyesociety.com](https://www.handeyesociety.com/) - info@handeyesociety.com
<br/><br/>
Python Backend - [github.com/thana-than/LockwoodImmersiveMagicMirror](https://github.com/thana-than/LockwoodImmersiveMagicMirror)
<br/>
External Renderer - [github.com/thana-than/LockwoodImmersiveMagicMirrorRenderer](https://github.com/thana-than/LockwoodImmersiveMagicMirrorRenderer)

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Acknowledgments

* [Python](https://www.python.org/)
* [NumPy](https://pypi.org/project/numpy/)
* [OpenCV](https://pypi.org/project/opencv_python/)
* [Pillow](https://pypi.org/project/Pillow/)