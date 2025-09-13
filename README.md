<img width="1882" height="1050" alt="0" src="https://github.com/user-attachments/assets/cb3ceced-7ead-4736-9c10-0f09806398a3" />

 OSRS Lumbridge Fishing & Cooking Bot
Automates a full fishing → cooking → banking loop in Old School RuneScape using screen capture, classical image processing, and on-screen object detection. Includes a real-time dashboard UI, pathfinding on pre-defined map grids, and a finite-state-machine (FSM) to drive actions.
⚠️ Disclaimer
This project is for educational and personal use only. Automating gameplay violates OSRS’s Terms of Service. Use responsibly and at your own risk.
## Highlights
- End-to-end loop: fish at Lumbridge, cook at the range, bank cooked fish, and repeat
- Screen-based, no memory injection
- Minimap template matching to locate player on static world/top-floor maps
- A* pathfinding on discretized map grids
- YOLO-based on-screen object detection
- Real-time performance metrics overlay (FPS, inference time, template-matching time, display time)
- Simple GUI to view the pipeline and toggle bot actions


[![Watch the video](https://img.youtube.com/vi/OB_UskZ1ajc/maxresdefault.jpg)](https://youtu.be/OB_UskZ1ajc)
▶️ [**Watch the video on YouTube**](https://youtu.be/OB_UskZ1ajc)


## Demo (What you’ll see)
- Top-left: Live game feed with detections and chosen click points
- Top-right: Static map with current minimap location box
- Bottom-right: Minimap view with valid path cells, path overlay, and target path cells
- Bottom-left: Performance stats and last action info
- A button to Start/Stop the bot, and a keyboard toggle (key: x)

## Requirements
- OS: Windows 10/11 (required for win32-based capture)
- Python: 3.9.x (code targets 3.9.6)
- Display scaling: Set Windows “Scale” to 100% for accurate capture coordinates
- GPU: Optional, but supported for OpenCV CUDA template matching (if your OpenCV build has CUDA)

### Python dependencies
Create a virtual environment and install:
- opencv-python
- numpy
- pillow
- tkinter (usually included with Python on Windows)
- ultralytics (for YOLO)
- pygetwindow
- scipy, networkx, matplotlib (used in parts of the project)
- pyyaml, requests, sympy, six, jinja2, pillow, pyparsing

Example:

# Windows PowerShell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install opencv-python numpy pillow pygetwindow ultralytics scipy networkx matplotlib pyyaml requests sympy six jinja2 pyparsing


Note: If you want CUDA-accelerated template matching, you’ll need an OpenCV build with CUDA support

## Project structure (key modules)
- lumbridge_fishing_cooking_bot_gui.py — Main app with the GUI/dashboard and processing loop
- lumbridge_fishing_cooking_bot.py — FSM logic that turns detections into clicks (actions)
- image_processing.py — Template matching, grid creation, movement detection, detection utilities
- map_info.py — Map container and configs (grid, scale, start/end cells)
- a_star_pathfinding.py — A* search on grid (used to precompute optimal path)
- display_gui.py — Drawing/placement helpers for dashboard
- win_32_capture.py — Windows client-area screen capture (event-driven)
- performance_tracker.py — EMA-based performance metrics (overlay)
- screen_interact.py — Executes clicks/keystrokes on the game window
- settings.py — Centralized configuration, paths, and UI constants

There are also small utilities to help with dataset preparation and pixel coordinate checks (e.g., processing video frames to images, clicking an image to print original coordinates).
## Configuration
Open settings.py and review the following typical items:
- MODEL_PATH: Path to your YOLO model (e.g., a trained .pt file)
- APPLICATION_WINDOW: Partial title of the OSRS client window to capture
- FRAME_RATE: Target FPS for the app loop
- MINIMAP_*: Minimap crop rectangle (scaled relative to a nominal height)
- WORLD_MAP_PATH / TOP_FLOOR_MAP_PATH: Static map images used for template matching and grid/path overlays
- TEST_WITH_RECORDING / TEST_WITH_SCREENSHOT: Toggle test modes that bypass live capture
- COLORS, GRID_COLOR, CHOSEN_PATH_COLOR, VALID_CELLS_COLOR: Display colors

Notes:
- The code tries to resolve the OSRS window by partial title. Make sure APPLICATION_WINDOW matches what you see in your taskbar title.
- If you run at a different resolution or UI layout, you may need to adjust SCALE_FACTOR in map configs and MINIMAP_* coordinates.

## Quick start
1. Set Windows display scaling to 100%
Settings → System → Display → Scale: 100%
2. Create and activate a virtual environment, then install dependencies
See “Python dependencies” above.
3. Prepare assets and settings

- Place or set MODEL_PATH to a valid YOLO model file trained on your labels (player, fishing_spot, bank, cooking_range, stairs, interact_up, interact_down, textbox, bank_screen, bank_deposit, raw, cooked).
- Ensure the world and top-floor map image paths are correct in settings.py.

1. Start the GUI
python lumbridge_fishing_cooking_bot_gui.py


## How it works
- Capture: Win32Capture hooks the OSRS client window and streams frames to the pipeline.
- Detection: YOLO runs asynchronously (thread pool) on incoming frames to find key objects.
- Minimap localization: Template matching places the current minimap onto a static world/top-floor image.
- Grid & Path: Static maps are discretized into a grid; A* path and cost map are precomputed.
- FSM: The bot’s state (fishing, travel, cooking, banking, etc.) chooses the next action based on detections and progress trackers.
- Action execution: Clicks and keys are sent to the client window to interact (e.g., click fishing spot, climb stairs, deposit items).

## Tips and troubleshooting
- DPI/Scaling: On Windows with Python 3.9, Win32 capture is accurate only at 100% display scale. Using other scale factors causes mismatches.
- Window focus: Ensure the OSRS client is visible and not minimized. The capture targets the client area.
- Model performance: If inference is too slow, reduce model size, lower confidence, or run half-precision. Turn on CUDA if available.
- Minimap classification: A fast color-based classifier determines which static map to use; you can replace/tune it if you need more robustness.
- Template matching: The code uses a coarse-to-fine search with optional GPU acceleration and a localized ROI around the previous match.

## Roadmap
- Smarter minimap detection/tracking with fallback strategies
- Auto calibration of minimap rect and grid scaling
- More robust state recovery and anti-stall heuristics
- Cross-client support and additional OSRS locations via configurable map packs

## Contributing
Issues and PRs are welcome. If you add a new map/location:
- Provide the static map image
- Create a corresponding MapConfig with tuned SCALE_FACTOR, grid size, and start/end cells
- Update settings to reference your new assets

## License
This project is provided “as is” for educational purposes. This project is licensed under the MIT license. Permission is hereby granted, free of charge, to any person obtaining a copy
## Acknowledgments
- OpenCV for image and template matching
- Ultralytics YOLO for object detection
- NumPy and SciPy ecosystem for numerics and utilities

If you have questions or need help setting it up, open an issue.
