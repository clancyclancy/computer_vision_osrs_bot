
"""
OSRS Lumbridge Fishing & Cooking Bot 🎣🔥🏦
------------------------------------------
Automates a full fishing-cooking-banking loop in Old School RuneScape using pixel color detection
and screen interaction. Designed for Lumbridge, this bot performs the following tasks:

1. Locates and clicks fishing spots to catch fish.
2. Detects full inventory and travels to the nearby cooking range.
3. Cooks the caught fish using color-based interaction.
4. Deposits cooked fish at the Lumbridge bank.
5. Returns to the fishing area and repeats the cycle.

🛠️ Features:
- Color-based object recognition (no injection or memory reading)
- Inventory monitoring and conditional task switching
- Simple pathing logic for location transitions
- **Highly dynamic design** — can be adapted to other OSRS locations by only updating map images
  and retraining object detection parameters

⚠️ Disclaimer:
This bot is for educational and personal use only. Automating gameplay violates OSRS's
terms of service. Use responsibly and at your own risk.

Author: Michael Clancy
Created: 07/01/2025
"""

# TODO: win32 screen capture only works when Windows Display Scale = 100%, otherwise it will retrieve inaccurate frames
# Python 3.9 setting DPI aware seems to do nothing, Python 3.12 by default appears to be DPI aware and DPI setting insensitive
# Python 3.9 can retrieve the windows DPI settings (125%, 150%, etc), so could do some post processing on the frame given this
# information if desired

import numpy as np
import tkinter as tk
import pygetwindow
import PIL
import ultralytics
import threading
import queue
import concurrent.futures
import cv2
import time

import win_32_capture
import settings
import map_info
import performance_tracker
import display_gui
import image_processing
import a_star_pathfinding
import lumbridge_fishing_cooking_bot
import screen_interact

# Thread-safe queue for latest frame
frame_grab_queue    = queue.Queue(maxsize=1)
frame_display_queue = queue.Queue(maxsize=1)


executor = concurrent.futures.ThreadPoolExecutor()

DISPLAY_WINDOW = "OSRS Bot"

# Global toggle for bot
bot_enabled = False


def process():
    """
    Processes game state, manages frame processing, object detection, map analysis,
    and decision-making for a Lumbridge Fishing and Cooking Bot application.

    This function initializes the necessary settings, resources, and dependent
    components such as maps for navigation, object detection models, and performance
    tracking. It executes continuous game state analysis, including frame capture,
    minimap location detection, player navigation using predefined map configurations,
    and processing object detection results.

    The function implements preprocessing steps for map analysis, including
    discretizing the map, performing A* pathfinding for optimal path computation,
    and visualizing both the world map and the Lumbridge castle's top floor. During
    runtime, it continuously captures frames, detects objects using a machine learning
    model, tracks player movement, and updates map-related data.

    Control logic includes processing distinct maps based on minimap location and
    determining player actions such as navigating paths or selecting cells based on
    path cost scores. The bot operates in a loop to adaptively analyze the game state,
    update visualizations, and output performance metrics.

    :raises queue.Empty: Raised when the frame capture queue is empty, preventing
                         retrieval of a new frame at a particular instance.
    :raises Exception: Raised for any unexpected issues occurring during runtime,
                       dependent on individual components' functionality.

    :return: This function does not return any value as it operates on a continuous
             loop performing real-time processing for the bot.
    """
    # --------------------
    # STARTUP
    # --------------------
    process_settings = settings.Settings()

    bot = lumbridge_fishing_cooking_bot.LumbridgeFishingCookingBot()

    model = ultralytics.YOLO(process_settings.MODEL_PATH)

    perf = performance_tracker.PerformanceTracker()


    if process_settings.TEST_WITH_RECORDING:
        capture = cv2.VideoCapture(process_settings.TEST_WITH_RECORDING_PATH)
    elif process_settings.TEST_WITH_SCREENSHOT:
        capture = []
    else: #Nominal application capture
        capture = win_32_capture.Win32Capture(window_title_substring=process_settings.APPLICATION_WINDOW)
        #capture = windows_capture.WindowsCapture(window_name=process_settings.APPLICATION_WINDOW)

    # Start the thread for frame loading
    threading.Thread(target = frame_loader,
                     args   = (capture, process_settings),
                     daemon = True).start()

    # Store previous iteration vars
    prev_max_val              = None
    prev_max_loc              = None
    prev_minimap_img          = None
    prev_minimap_in_world_map = None
    previous_action           = None

    # Model
    model_future       = None
    last_model_objects = []

    process_settings.APPLICATION_WINDOW = find_windows_by_partial_title(process_settings.APPLICATION_WINDOW)

    # --------------------
    # PREPROCESS
    # --------------------
    """ 
    There are two static maps which the player will encounter. 
    1. The main world map, where the player will fish, cook, and travel. 
    2. The top floor of Lumbridge castle map, where the player will deposit their inventory in the bank
    The player's minimap will indicate which map the player is on, and where on the map they are.
    The optimal path to walk along each static map does not change, therefore the pathing can be precomputed
    """

    # create map objects
    image = cv2.imread(str(process_settings.WORLD_MAP_PATH))
    image = display_gui.resize_to_fit(image,process_settings.QUADRANT_WIDTH, process_settings.QUADRANT_HEIGHT)
    world_map     = map_info.Map(image)

    image = cv2.imread(str(process_settings.TOP_FLOOR_MAP_PATH))
    image = display_gui.resize_to_fit(image,process_settings.QUADRANT_WIDTH, process_settings.QUADRANT_HEIGHT)
    top_floor_map = map_info.Map(image)

    # create map configs
    world_map.config = map_info.WorldMapConfig()
    top_floor_map.config = map_info.TopFloorMapConfig()

    # discretize map and compute A* path
    world_map.valid_cells_grid = image_processing.create_valid_cells_grid(world_map.image, world_map.config)
    # A* pathing
    world_map.path, world_map.path_costs = a_star_pathfinding.a_star(grid=world_map.valid_cells_grid,
                                                                     start=world_map.config.MAP_START_CELL,
                                                                     end=world_map.config.MAP_END_CELL)
    # discretize map and compute A* path
    top_floor_map.valid_cells_grid = image_processing.create_valid_cells_grid(top_floor_map.image, top_floor_map.config)
    # A* pathing
    top_floor_map.path, top_floor_map.path_costs = a_star_pathfinding.a_star(grid=top_floor_map.valid_cells_grid,
                                                                             start=top_floor_map.config.MAP_START_CELL,
                                                                             end=top_floor_map.config.MAP_END_CELL)
    display_world_map = world_map.image.copy()
    display_gui.draw_static_objects(image       = display_world_map,
                                    path        = world_map.path,
                                    grid        = world_map.valid_cells_grid,
                                    grid_rows   = world_map.config.MAP_GRID_ROWS,
                                    grid_cols   = world_map.config.MAP_GRID_COLS,
                                    grid_color  = process_settings.GRID_COLOR,
                                    valid_color = process_settings.VALID_CELLS_COLOR,
                                    path_color  = process_settings.CHOSEN_PATH_COLOR)

    display_top_floor_map = top_floor_map.image.copy()
    display_gui.draw_static_objects(image       = display_top_floor_map,
                                    path        = top_floor_map.path,
                                    grid        = top_floor_map.valid_cells_grid,
                                    grid_rows   = top_floor_map.config.MAP_GRID_ROWS,
                                    grid_cols   = top_floor_map.config.MAP_GRID_COLS,
                                    grid_color  = process_settings.GRID_COLOR,
                                    valid_color = process_settings.VALID_CELLS_COLOR,
                                    path_color  = process_settings.CHOSEN_PATH_COLOR)


    world_map.display_image = display_world_map
    top_floor_map.display_image = display_top_floor_map


    # --------------------
    # PROCESS
    # --------------------
    while True:

        loop_start = time.perf_counter()

        # CAPTURE FRAME --------------------
        try:
            frame = frame_grab_queue.get_nowait()
        except queue.Empty:
            frame = None

        if frame is None:
            continue

        # RUN MODEL IN PARALLEL --------------------
        if model_future is None or model_future.done():
            if model_future is None:
                #If model does not exist submit new model
                model_future = executor.submit(image_processing.timed_model, model, frame, device=0, half=True, verbose=False)
            else:
                #If the model exists, grab results and begin processing the new frame
                last_model_result, model_time = model_future.result()
                perf.update_model_time(model_time)
                last_model_objects = image_processing.get_objects(last_model_result)
                model_future = executor.submit(image_processing.timed_model, model, frame, device=0, half=True, verbose=False)

        # GET MINIMAP FROM HARDCODED LOCATION IN FRAME --------------------
        # TODO: train model to detect minimap location if minimap location appears to move?

        frame_height, _ = frame.shape[:2]
        scale = frame_height / process_settings.MINIMAP_COORDS_NOMINAL_SCALE_HEIGHT

        frame_minimap = frame[int(process_settings.MINIMAP_y1*scale):int(process_settings.MINIMAP_y2*scale),
                              int(process_settings.MINIMAP_x1*scale):int(process_settings.MINIMAP_x2*scale)].copy()


        # GET LOCATION OF PLAYER (MAP1/MAP2) --------------------
        # TODO: if average image color is unreliable, can run two template matches, one for each map and compare correl, way slower but more reliable
        minimap_in_world_map = image_processing.fast_classify(image = frame_minimap, scale = 1.0)


        # assign object
        if minimap_in_world_map:
            map_obj = world_map
        else:
            map_obj = top_floor_map


        # RESIZE MINIMAP TO HAVE SAME RESOLUTION AS MAP --------------------
        # precomputed for speed in test_getAverageTemplateMatch_minimapScaler.py
        frame_minimap = cv2.resize(frame_minimap,
                                   dsize = (0,0),
                                   fx=map_obj.config.SCALE_FACTOR,
                                   fy=map_obj.config.SCALE_FACTOR,
                                   interpolation=cv2.INTER_AREA)

        # DETECT IF PLAYER IS MOVING --------------------
        if prev_minimap_img is not None:
            significant_movement = image_processing.significant_movement(image          = frame_minimap,
                                                                         previous_image = prev_minimap_img)
        else:
            significant_movement = True


        # TEMPLATE MATCH TO DETERMINE WHERE IN MAP MINIMAP IS --------------------
        start_time = time.perf_counter()

        if significant_movement:
            max_val, max_loc = image_processing.compute_template_match(map_img      = map_obj.image,
                                                                       minimap_img  = frame_minimap,
                                                                       prev_max_loc = prev_max_loc,
                                                                       prev_max_val = prev_max_val,
                                                                       on_same_map  = prev_minimap_in_world_map == minimap_in_world_map)
        else:
            max_val = prev_max_val
            max_loc = prev_max_loc

        perf.update_template_time(time.perf_counter() - start_time)


        # ASSIGN PREVIOUS VARS --------------------
        prev_minimap_img          = frame_minimap.copy()
        prev_max_val              = max_val
        prev_max_loc              = max_loc
        prev_minimap_in_world_map = minimap_in_world_map


        # GET GRID CELLS WITHIN CURRENT MINIMAP --------------------
        grid_within_minimap, shift_x, shift_y = image_processing.get_cells_within_minimap(image_map     = map_obj.image,
                                                                                           image_minimap = frame_minimap,
                                                                                           max_loc       = max_loc,
                                                                                           grid_cols     = map_obj.config.MAP_GRID_COLS,
                                                                                           grid_rows     = map_obj.config.MAP_GRID_ROWS)

        # GET LOWEST/HIGHEST PATH COST SCORES --------------------
        # click lowest cost cell to move towards the start of the path, click the highest to move towards the end
        lowest_cell, highest_cell = image_processing.get_path_bounds(path                = map_obj.path,
                                                                     costs               = map_obj.path_costs,
                                                                     grid_within_minimap = grid_within_minimap)

        # GET PIXEL LOCATION OF HIGHEST AND LOWEST CELL --------------------
        map_height, map_width = map_obj.image.shape[:2]
        lowest_cell_loc = image_processing.get_pixel_location(cell = lowest_cell,
                                                              grid_within_minimap = grid_within_minimap,
                                                              shift_x = shift_x,
                                                              shift_y = shift_y,
                                                              map_height = map_height,
                                                              map_width = map_width)

        highest_cell_loc = image_processing.get_pixel_location(cell = highest_cell,
                                                              grid_within_minimap = grid_within_minimap,
                                                              shift_x = shift_x,
                                                              shift_y = shift_y,
                                                              map_height = map_height,
                                                              map_width = map_width)

        # SCALE TO ORIGINAL LOADED FRAME SIZE --------------------
        lowest_cell_loc = (int((lowest_cell_loc[0] / map_obj.config.SCALE_FACTOR) + process_settings.MINIMAP_x1*scale),
                           int((lowest_cell_loc[1] / map_obj.config.SCALE_FACTOR) + process_settings.MINIMAP_y1*scale))

        highest_cell_loc = (int((highest_cell_loc[0] / map_obj.config.SCALE_FACTOR) + process_settings.MINIMAP_x1*scale),
                            int((highest_cell_loc[1] / map_obj.config.SCALE_FACTOR) + process_settings.MINIMAP_y1*scale))


        # GET MODEL OBJECTS --------------------
        while process_settings.WAIT_ON_MODEL and not model_future.done():
            time.sleep(0.0025)

        if model_future.done():
            last_model_result, model_time = model_future.result()
            perf.update_model_time(model_time)
            last_model_objects = image_processing.get_objects(last_model_result)
            model_future = None


        # ADD CLICKABLE POINTS ON MINIMAP TO OBJECTS --------------------
        image_processing.append_objects(objects    = last_model_objects,
                                        position   = lowest_cell_loc,
                                        label      = "lowest_cell")

        image_processing.append_objects(objects    = last_model_objects,
                                        position   = highest_cell_loc,
                                        label      = "highest_cell")


        # COMPUTE ACTION --------------------

        if bot_enabled:
            action = bot.request_action(last_model_objects, minimap_in_world_map)
        else:
            action = None

        if action is not None:

            #scale up the action location
            x,y = action.position
            x = int(x / scale)
            y = int(y / scale)
            action.position = (x,y)

            new_action = True
            previous_action = action
            perf.clicked_object_label = action.object["label"]
            perf.time_of_new_action = time.perf_counter()
        else:
            new_action = False
            action = previous_action


        # EXECUTE ACTION --------------------
        if new_action:
            screen_interact.execute_action(action, process_settings.APPLICATION_WINDOW, DISPLAY_WINDOW)
            action = None


        # DRAW FRAME --------------------
        display_start = time.perf_counter()

        # FULL FRAME -------------------------
        # bounding boxes on frame
        display_gui.draw_objects(image   = frame,
                                 objects = last_model_objects,
                                 colors  = process_settings.COLORS)

        # draw minimap snip region on frame
        cv2.rectangle(frame, (int(process_settings.MINIMAP_x1 * scale), int(process_settings.MINIMAP_y1 * scale)),
                             (int(process_settings.MINIMAP_x2 * scale), int(process_settings.MINIMAP_y2 * scale)), (155,255,0), 3)

        #draw points on minimap that can be clicked to maximize/minimize tile cost score (move closer to cooking range or move closer to fishing spot)
        display_gui.draw_point(image = frame,
                               point = lowest_cell_loc,
                               color = process_settings.CHOSEN_PATH_COLOR)

        display_gui.draw_point(image = frame,
                               point = highest_cell_loc,
                               color = process_settings.CHOSEN_PATH_COLOR)


        # MAP ----------------
        display_map = map_obj.display_image.copy()
        minimap_height, minimap_width = frame_minimap.shape[:2]
        display_gui.draw_minimap_location(image            = display_map,
                                          minimap_height   = minimap_height,
                                          minimap_width    = minimap_width,
                                          max_loc          = max_loc,
                                          thickness        = 3)

        cv2.putText(
            display_map,
            "World Map",
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )

        #MINIMAP ----------------
        map_height, map_width = map_obj.image.shape[:2]
        display_gui.draw_dynamic_minimap(image               = frame_minimap,
                                         grid_within_minimap = grid_within_minimap,
                                         valid_cells_grid    = map_obj.valid_cells_grid,
                                         grid_path           = map_obj.path,
                                         lowest_path_cost    = lowest_cell,
                                         highest_path_cost   = highest_cell,
                                         shift_x             = shift_x,
                                         shift_y             = shift_y,
                                         map_width           = map_width,
                                         map_height          = map_height,
                                         valid_color         = process_settings.VALID_CELLS_COLOR,
                                         path_color          = process_settings.CHOSEN_PATH_COLOR,
                                         max_min_cells_color = (155, 255, 155),
                                         thickness           = 1)



        # Add the thick blue border
        border_thickness = 3
        frame_minimap = cv2.copyMakeBorder(
            frame_minimap,
            top=border_thickness,
            bottom=border_thickness,
            left=border_thickness,
            right=border_thickness,
            borderType=cv2.BORDER_CONSTANT,
            value=(155,255,0)
        )

        ##Resize minimap to fully fill the quadrant
        frame_minimap = display_gui.resize_to_fit(
            image=frame_minimap,
            max_width=process_settings.QUADRANT_WIDTH/1.4,
            max_height=process_settings.QUADRANT_HEIGHT/1.4
        )


        # draw the last desired object to be clicked
        if action is not None and process_settings.DRAW_LAST_ACTION_LOCATION:
            display_gui.draw_objects(image        = frame,
                                     objects      = action.object,
                                     colors       = (255,255,255),
                                     display_text = True)

            display_gui.draw_point(image = frame,
                                   point = action.position,
                                   color = (255,255,255))

        # Create blank canvas
        dashboard = np.zeros((process_settings.CANVAS_HEIGHT, process_settings.CANVAS_WIDTH, 3), dtype=np.uint8)

        # Place top-left (main frame)
        display_gui.place_centered(target     = dashboard,
                                   image      = frame,
                                   top_left_x = 0,
                                   top_left_y = 0,
                                   quad_w     = process_settings.QUADRANT_WIDTH,
                                   quad_h     = process_settings.QUADRANT_HEIGHT)

        # Place top-right (static map)
        display_gui.place_centered(target     = dashboard,
                                   image      = display_map,
                                   top_left_x = process_settings.QUADRANT_WIDTH,
                                   top_left_y = 0,
                                   quad_w     = process_settings.QUADRANT_WIDTH,
                                   quad_h     = process_settings.QUADRANT_HEIGHT)


        # Place bottom-right (frame piece)
        display_gui.place_centered(target     = dashboard,
                                   image      = frame_minimap,
                                   top_left_x = process_settings.QUADRANT_WIDTH,
                                   top_left_y = process_settings.QUADRANT_HEIGHT,
                                   quad_w     = process_settings.QUADRANT_WIDTH,
                                   quad_h     = process_settings.QUADRANT_HEIGHT)

        # Placed text bottom-left
        offset = 30
        display_gui.place_text(target     = dashboard,
                               lines      = perf.get_overlay_stats(),
                               top_left_x = offset,
                               top_left_y = process_settings.QUADRANT_HEIGHT + offset)


        # DISPLAY FRAME --------------------
        final_img = dashboard
        #push to queue (drop old if full)
        if not frame_display_queue.empty():
            try:
                frame_display_queue.get_nowait()
            except queue.Empty:
                pass
        frame_display_queue.put(final_img)


        # TIME DRAW AND DISPLAY --------------------
        display_time = time.perf_counter() - display_start
        perf.update_display_time(display_time)


        # FPS LIMIT --------------------
        target_dt = 1.0 / process_settings.FRAME_RATE
        dt = time.perf_counter() - loop_start

        if dt < target_dt:
            time.sleep(target_dt - dt)

        dt = time.perf_counter() - loop_start
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        perf.update_fps(inst_fps)


def update_display():
    """
    Updates the GUI with the most recent frame from the frame display queue in
    a continuous loop.

    The function retrieves the latest frame from the queue, processes it into a
    format compatible with Tkinter GUI, and updates the label widget to display
    the image. This function also schedules itself to run again after a short
    interval to ensure continuous updates.

    :return: None
    """
    # Continuously updates the GUI with the latest frame from the queue

    if not frame_display_queue.empty():

        frame = frame_display_queue.get()

        # Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        img = PIL.Image.fromarray(frame_rgb)

        # Convert to Tkinter-compatible image
        tk_img = PIL.ImageTk.PhotoImage(image=img)

        # Update the label widget with the new image
        label.config(image=tk_img)

        # Keep a reference to avoid garbage collection
        label.image = tk_img

    root.after(1, update_display)  # Schedule the next update in 1 millisecond



def frame_loader(capture, process_settings):
    """
    Loads frames from various sources, processes them, and updates a frame queue for
    further usage. The function works with a capture object, including hooking into its
    frame event, or it can use predefined test settings such as loading frames from video
    recordings or screenshots. Frames are resized to fit defined dimensions before being added
    to the frame queue.

    :param capture: The source of frames, either a capture object with an event-driven
        system or None if test settings are used.
    :type capture: object or None
    :param process_settings: An object containing various configuration and testing options.
        Determines the behavior of frame loading and processing.
    :type process_settings: object
    :return: This function does not explicitly return a value. Its primary operation is to
        manage frame loading, resizing, and updating the shared frame queue.
    :rtype: None
    """
    latest_app_frame = [None]

    # If capture object is provided, hook into its event
    if capture is not None and not process_settings.TESTING:
        @capture.event
        def on_frame_arrived(frame_bgr,control):
            latest_app_frame[0] = frame_bgr

    while True:

        if process_settings.TEST_WITH_RECORDING:
            ret, frame = capture.read()
            if not ret:
                break

        elif process_settings.TEST_WITH_SCREENSHOT:
            frame = cv2.imread(str(process_settings.TEST_WITH_SCREENSHOT_PATH))

        # Load frame from the application screen capture
        else:
            frame = latest_app_frame[0]


        if frame is not None:

            frame = display_gui.resize_to_fit(
                image = frame,
                max_width = process_settings.QUADRANT_WIDTH,
                max_height = process_settings.QUADRANT_HEIGHT
            )

            try:
                frame_grab_queue.get_nowait()
            except queue.Empty:
                pass
            frame_grab_queue.put_nowait(frame)

        time.sleep(0.001)

def find_windows_by_partial_title(partial_title):
    """
    Find windows whose titles contain a given partial string.

    This function searches through all available window titles on the system
    and returns those that contain the specified partial title string. The search
    is case-insensitive.

    :param partial_title: A substring to search for within window titles.
    :type partial_title: str
    :return: A list of window titles that match the given partial title.
    :rtype: list[str]
    """
    matches = []
    for win in pygetwindow.getAllTitles():
        if partial_title.lower() in win.lower():
            matches.append(win)
    return matches

def toggle_bot():
    """
    Toggles the state of a global bot and updates its associated UI components.

    This function switches the value of the global variable `bot_enabled`
    between `True` and `False`. Based on the current state, it updates
    the text and background color of the button to reflect whether the
    bot is active or stopped.

    :raises NameError: If the global `bot_enabled` or `button` is not defined.
    """
    global bot_enabled
    bot_enabled = not bot_enabled
    button.config(text="🟥 STOP Bot" if bot_enabled else "🟩 START Bot",
                  bg="red" if bot_enabled else "green"
    )


# main loop
root = tk.Tk()
root.title(DISPLAY_WINDOW)
label = tk.Label(root)
label.pack()
button = tk.Button(root, text="🟩 START Bot", command=toggle_bot, bg="green", fg="white", font=("Arial", 10))
button.pack(pady=10)
root.bind("x", lambda event: toggle_bot())


# Start processing in the background thread
threading.Thread(target = process, daemon = True).start()

# Start display loop
update_display()
root.mainloop()
