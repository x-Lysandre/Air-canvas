# Air Canvas Project

## Overview

The Air Canvas project allows users to write or draw in the air using hand gestures, with the help of MediaPipe and OpenCV. This interactive tool detects hand movements from a webcam feed, enabling drawing, erasing, and color-changing functionalities.

## Features

- **Draw with Your Index Finger:** Write or draw by moving your right hand's index finger.
- **Erase Mode:** Toggle between writing and erasing modes by pinching your right hand's thumb and index finger together. The mode switches back and forth with each pinch.
- **Color Selection:** Use different finger combinations on your left hand to change the pen color:
  - **Index Finger:** Cyan color
  - **Index + Middle Finger:** Peach color
  - **Index + Middle + Ring Finger:** Lime color
  - **Index + Middle + Ring + Pinky Finger:** Silver color

## Requirements

- Python 3.11
- OpenCV
- MediaPipe
- NumPy

## Installation

### Clone the repository:
   ```bash
   git clone https://github.com/yourusername/air-canvas.git
   cd air-canvas
  ```


## How to Use

1. Run the air_canvas.py script:

```bash
python air_canvas.py
```
2. Use the webcam to draw or write in the air:

* Right-hand index finger for writing/drawing.
* Pinch with the right-hand thumb and index finger to toggle between write and erase modes.
* Use left-hand finger combinations to change colors.
    * Index: Cyan
    * Index + Middle: Peach
    * Index + Middle + Ring: Lime
    * Index + Middle + Ring + Pinky: Silver


