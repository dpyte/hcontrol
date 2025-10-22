# HCONTROL (Hand Control)

Using hand gestures to control my dimmable lamp.

Started this back in college when I wanted to learn computer vision and gesture recognition. The lamp control worked,
but now I'm expanding it to control the desktop too (primarily Windows).

## What it does

- Hand tracking using MediaPipe
- Gesture recognition for mouse control
- Click, drag, scroll using hand gestures
- Originally built for controlling a smart lamp, now works as a mouse

## Current Status

Work in progress. Reworking the codebase and adding Windows desktop control support.
The gesture recognition works. Mouse control works. Still tuning the ergonomics and adding features.

Need a webcam and decent lighting.

## Basic Gestures

- Point with index finger → move cursor
- Pinch (thumb + index) → click
- Pinch + hold → right-click
- Pinch + drag → drag objects
- Drop hand → rest mode

Press ESC to exit, R to recalibrate.

## Files

- `gesture_advanced.py` - State machine gesture recognizer
- `mouse_controller.py` - Mouse/keyboard control
- `example_integration.py` - Ready to run example
- `tracking.py` - MediaPipe hand tracking
- Old lamp control code - removed, was fun while it lasted

## Notes

The green box on screen shows the activation zone. Keep your hand there for tracking. Drop it below to rest.

Arm gets tired if you use it too long. That's normal. Take breaks or adjust the activation zone smaller.

## Changes

- 2025-21-10:
    - Removed the old code i.e., C++ code. The transition to pure Python from C++ was necessary as Python-C++ bridge was
      a performance bottleneck. (PS: I did use LLM to migrate the code)

- 2025-22-10:
    - Add logic for on-screen points calibration. Decided it wasn't worth the effort to make it work. The logic's still
      there, however, the final result is not used.
    - Added a new module(s) for gesture detection.

## License

Currently using MIT so, do whatever you want with it.
