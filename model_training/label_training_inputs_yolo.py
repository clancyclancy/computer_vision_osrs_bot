"""
Label Training Inputs for YOLO

Overview:
This script provides an interactive Tkinter-based GUI to quickly create YOLO-format
annotations for image frames. It is designed to streamline labeling for object
detection training by allowing you to click to place fixed-size or label-specific
bounding boxes and save them directly in YOLO .txt format alongside the images.

Key Features:
- Displays frames from a source folder and lets you step through them.
- Click-to-annotate with a selected label; supports a "sticky" label mode.
- Undo last annotation for current frame.
- Copy last frame’s annotations to the current frame to speed up repetitive work.
- Duplicate, skip, go back, and step forward through frames.
- Saves YOLO annotations (class_id x_center y_center width height, normalized) per image.

Configuration:
- frames_folder: Path to the folder containing the input frames (images).
- dataset_dir: Destination root where YOLO dataset structure and labels will be saved.
  The script will create/expect standard YOLO layout and save label files to the
  appropriate labels directory (e.g., dataset_dir/labels/...).
- labels_list: Ordered list of class names. The index in this list is the YOLO class_id.
- fixed_box_size or label_box_sizes: Controls the size of the placed bounding boxes.
- desired_num_frames, start_frame_index: Control labeling subset and where to start.
- max_display_width/height: Controls preview scaling in the GUI.

How to Use:
1) Open the script and set:
   - frames_folder to your images directory
   - dataset_dir to your output dataset root
   - labels_list to your class names in the desired order
   - (optional) box sizes and other options shown near the top of the file
2) Run the script:
   - python label_training_inputs_yolo.py
3) In the GUI:
   - Select a label button.
   - (Optional) Toggle Sticky to keep the current label selected after each click.
   - Click on the image to place a bounding box for the current label.
   - Use Undo to remove the most recent annotation in the current frame.
   - Use Next/Back/Skip/Step to navigate frames.
   - Use Duplicate to duplicate the current frame and annotations if needed.
   - Use Copy Last to copy previous frame’s annotations to current frame.
4) Saving:
   - Use the Save/Next controls; the script writes a YOLO .txt file per image with
     normalized coordinates corresponding to the annotations shown.
"""


import cv2, os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
frames_folder = os.path.join(BASE_DIR, "FRAMES")
dataset_dir = os.path.join(BASE_DIR, "allDataYOLO")


labels_list = [
    "bank",
    "player",
    "fishing_spot",
    "raw",
    "cooked",
    "cooking_range",
    "stairs",
    "interact_up",
    "interact_down",
    "textbox",
    "bank_screen",
    "bank_deposit"
]

fixed_box_size    = (65, 65)

textbox_position  = (550, 880)
bank_screen_position = (718,385)

# Individual box sizes per label
label_box_sizes = {
    "bank":           (67, 115),
    "player":         (55, 115),
    "fishing_spot":   (85, 70),
    "raw":            fixed_box_size,
    "cooked":         fixed_box_size,
    "cooking_range":  (74, 96),
    "stairs":         (106, 122),
    "interact_up":    (295, 210),
    "interact_down":  (295+45, 210),
    "textbox":        (1100, 300),
    "bank_screen":    (1025, 702),  # or whatever size fits your UI
    "bank_deposit":   (260, 335)
}



desired_num_frames = 2000
start_frame_index  = 2182

max_display_width  = 1300
max_display_height = 900

# prepare dataset dirs
os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "labels"), exist_ok=True)

# sample frames
all_frames  = sorted(f for f in os.listdir(frames_folder)
                     if f.lower().endswith((".png","jpg","jpeg")))
step         = max(1, len(all_frames) // desired_num_frames)
#frame_files  = all_frames[::step][:desired_num_frames]
frame_files  = all_frames

# === STATE ===
frame_index         = start_frame_index
frame_image_cv      = None
frame_filename      = None
image_w = image_h   = None
display_image_tk    = None
yolo_annotations    = []
current_click_label = None
saved_frame_count   = 0 + sum(1 for f in os.scandir(os.path.join(dataset_dir, "images")) if f.is_file())
sticky_label_mode   = False    # <-- sticky toggle
label_buttons       = {}
scale               = 1.0
last_frame_annotations = []


# === FUNCTIONS ===

def update_title():
    """
    Updates the root window's title with current frame index, total number of
    frames, and saved frame count compared to the desired frame count.

    :return: None
    """
    root.title(
        f"Frame {frame_index+1}/{len(frame_files)}  "
        f"|  Saved {saved_frame_count}/{desired_num_frames}"
    )

def add_annotation(label_name, x1, y1, x2, y2):
    """
    Adds an annotation to the YOLO format annotations list and updates the display.
    The function calculates the normalized center coordinates, width, and height
    of the bounding box relative to the image dimensions. It also updates the
    current label state if sticky label mode is off.

    :param label_name: The name of the label to add to the annotation.
    :type label_name: str
    :param x1: The x-coordinate of the first point of the bounding box.
    :type x1: float
    :param y1: The y-coordinate of the first point of the bounding box.
    :type y1: float
    :param x2: The x-coordinate of the second point of the bounding box.
    :type x2: float
    :param y2: The y-coordinate of the second point of the bounding box.
    :type y2: float
    :return: None
    """
    cls_id = labels_list.index(label_name)
    xc     = ((x1+x2)/2) / image_w
    yc     = ((y1+y2)/2) / image_h
    w_rel  = abs(x2-x1) / image_w
    h_rel  = abs(y2-y1) / image_h
    yolo_annotations.append((cls_id, xc, yc, w_rel, h_rel))
    draw_all_annotations()

    global current_click_label
    if not sticky_label_mode and current_click_label:
        label_buttons[current_click_label].config(style="TButton")
        current_click_label = None

def undo_last_annotation(event=None):
    """
    Undo the last annotation in the annotation list if available.

    This function removes the most recent annotation from the list of annotations
    and updates the visual display accordingly. If there are no annotations to
    remove, it notifies the user through a message.

    :param event: Optional parameter representing the event that triggers the
                  function. Can be left as None.
    :return: None
    """
    if yolo_annotations:
        yolo_annotations.pop()
        draw_all_annotations()
        print("Undid last annotation.")
    else:
        print("No annotation to undo.")

def draw_all_annotations():
    """
    Renders all annotations onto the current frame image and updates the display with a scaled version of the
    result.

    This function takes YOLO annotation coordinates and draws bounding boxes with the corresponding class
    labels on the frame image. It scales the annotated image to fit the maximum display dimensions while
    maintaining the aspect ratio. The scaled image is prepared for display on a graphical user interface.

    :param frame_image_cv: The current frame image in OpenCV format.
    :param yolo_annotations: A list of annotations, where each annotation is a tuple containing the
        class index, center x-coordinate (relative), center y-coordinate (relative), width (relative),
        and height (relative).
    :param image_w: The width of the original image in pixels.
    :param image_h: The height of the original image in pixels.
    :param labels_list: A list of labels corresponding to the class indices.
    :param max_display_width: The maximum display width for scaling the annotated image.
    :param max_display_height: The maximum display height for scaling the annotated image.
    :param canvas: The Tkinter Canvas widget used for displaying the scaled image.
    :return: None
    """
    img = frame_image_cv.copy()
    for cls, xc, yc, w, h in yolo_annotations:
        bw = int(w * image_w)
        bh = int(h * image_h)
        x1 = int(xc*image_w - bw/2)
        y1 = int(yc*image_h - bh/2)
        x2, y2 = x1+bw, y1+bh
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, labels_list[cls], (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),1)

    # resize to fit
    sw = max_display_width / image_w
    sh = max_display_height / image_h
    global scale
    scale = min(sw, sh, 1.0)
    small = cv2.resize(img, (int(image_w*scale), int(image_h*scale)))
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    tkimg = ImageTk.PhotoImage(Image.fromarray(rgb))

    global display_image_tk
    display_image_tk = tkimg
    canvas.config(width=small.shape[1], height=small.shape[0])
    canvas.create_image(0, 0, anchor="nw", image=tkimg)

def on_canvas_click(e):
    """
    Handles the user's click event on the canvas, calculates the bounding box
    coordinates for the click based on the current label, and adds an annotation.

    :param e: The event object from the canvas click event, containing positional
        data such as `e.x` and `e.y`.
    """
    global current_click_label
    if not current_click_label:
        return

    # noinspection PyUnreachableCode
    lbl = current_click_label
    # noinspection PyUnreachableCode
    if lbl in ["textbox", "bank_screen"]:
        w, h = label_box_sizes[lbl]
        xc, yc = textbox_position if lbl == "textbox" else bank_screen_position # example position for bank_screen
        x1 = max(0, xc - w // 2)
        y1 = max(0, yc - h // 2)
        x2 = min(image_w - 1, xc + w // 2)
        y2 = min(image_h - 1, yc + h // 2)
    else:
        w, h = label_box_sizes.get(lbl, fixed_box_size)
        cx = int(e.x/scale); cy = int(e.y/scale)
        x1 = max(0, cx - w//2); y1= max(0, cy - h//2)
        x2 = min(image_w-1, cx + w//2)
        y2 = min(image_h-1, cy + h//2)

    # noinspection PyUnreachableCode
    add_annotation(lbl, x1, y1, x2, y2)

def set_click_label(lbl):
    """
    Sets the current click label and updates the styling of label buttons. If the selected label
    is a "textbox" or "bank_screen", an annotation is automatically added to the image based on
    predefined sizes and positions.

    The method updates the global state for the current selected label, modifies the visual style
    of buttons, and handles auto-annotation logic for specific labels.

    :param lbl: The label to set as the current click label
    :type lbl: str
    :return: None
    """
    global current_click_label
    if current_click_label:
        label_buttons[current_click_label].config(style="TButton")

    current_click_label = lbl
    label_buttons[lbl].config(style="Selected.TButton")
    print("Selected label:", lbl)

    # If it's a textbox, auto-place immediately
    if lbl in ["textbox", "bank_screen"]:
        w, h = label_box_sizes[lbl]
        xc, yc = textbox_position if lbl == "textbox" else bank_screen_position
        x1 = max(0, xc - w // 2)
        y1 = max(0, yc - h // 2)
        x2 = min(image_w - 1, xc + w // 2)
        y2 = min(image_h - 1, yc + h // 2)
        add_annotation(lbl, x1, y1, x2, y2)
        current_click_label = None


def toggle_sticky():
    """
    Toggles the sticky label mode of the application.

    This function manages the state of the sticky label functionality.
    It changes the appearance of the currently selected label when sticky
    label mode is turned off and accurately updates the sticky label
    mode to be ON or OFF. The state change is reflected in the button's
    label and printed to the console.

    :return: None
    """
    global sticky_label_mode, current_click_label
    if sticky_label_mode:
        if current_click_label:  # Add this check
            label_buttons[current_click_label].config(style="TButton")
        current_click_label = None

    sticky_label_mode = not sticky_label_mode
    txt = "Sticky ON" if sticky_label_mode else "Sticky OFF"
    sticky_btn.config(text=txt)
    print(txt)


def save_current_frame_and_labels():
    """
    Saves the current frame and associated labels to separate files within the specified
    dataset directory. The function ensures unique filenames by appending an incrementing
    counter to the base name if necessary. Annotations are saved in the YOLO format.

    :return: None
    """
    global saved_frame_count

    # can save dupes of a frame, have a counter _01, _02, etc appended to end of filename
    name, ext = os.path.splitext(frame_filename)
    unique_filename = frame_filename
    counter = 1
    while os.path.exists(os.path.join(dataset_dir, "images", unique_filename)):
        unique_filename = f"{name}_{counter:02d}{ext}"
        counter += 1

    out_img = os.path.join(dataset_dir, "images", unique_filename)
    cv2.imwrite(out_img, frame_image_cv)

    out_txt = os.path.join(dataset_dir, "labels",
                           os.path.splitext(unique_filename)[0] + ".txt")
    with open(out_txt, "w") as f:
        for ann in yolo_annotations:
            f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} "
                    f"{ann[3]:.6f} {ann[4]:.6f}\n")

    saved_frame_count += 1
    print(f"Saved {unique_filename} ({len(yolo_annotations)} annotations)")
    update_title()


def load_frame():
    """
    Loads a frame image along with its YOLO annotations, sets its global dimensions, and
    updates the UI with associated titles. This function fetches the frame file based on the
    current frame index, reads the image, and initializes its height and width for subsequent
    processing. It also clears the existing annotations for the frame and redraws updated ones.

    :global yolo_annotations: A list that holds the YOLO annotations for the current frame.
    :global frame_filename: The filename of the current frame being processed.
    :global image_w: The width of the current frame image in pixels.
    :global image_h: The height of the current frame image in pixels.
    :global frame_image_cv: The OpenCV image object of the current frame.

    :return: None
    """
    global frame_image_cv, frame_filename, image_w, image_h, yolo_annotations
    yolo_annotations = []
    frame_filename   = frame_files[frame_index]
    path             = os.path.join(frames_folder, frame_filename)
    frame_image_cv   = cv2.imread(path)
    image_h, image_w = frame_image_cv.shape[:2]
    draw_all_annotations()
    update_title()

def next_frame(event=None):
    """
    Advances to the next frame, saving the current frame's labels and annotations, and
    loading the subsequent frame. Updates global tracking variables for frame index and
    annotations. Terminates the process if the end of the frames list is reached.

    :param event: Optional event that can be passed during invocation. Default is None.
    :return: None
    """
    global frame_index, last_frame_annotations
    save_current_frame_and_labels()
    last_frame_annotations = yolo_annotations.copy()
    frame_index += 1  # ← use the dynamic step size here
    if frame_index >= len(frame_files):
        root.quit()
    else:
        load_frame()

def step_frame(event=None):
    """
    Advances to the next frame in the sequence, saving the current frame's annotations
    and updating the frame index. If the end of the frame sequence is reached, the
    application quits. Otherwise, it loads the next frame.

    :param event: An optional event object, typically associated with a user interaction.
        Defaults to None.
    :return: None
    """
    global frame_index, last_frame_annotations
    save_current_frame_and_labels()
    last_frame_annotations = yolo_annotations.copy()
    frame_index += step
    if frame_index >= len(frame_files):
        root.quit()
    else:
        load_frame()

def duplicate_frame(event=None):
    """
    Duplicates the current frame along with its annotations, allowing users to save a
    copy of the current state of the frame for later use or comparison.

    This function serves as a utility for creating an identical copy of the active
    frame and its labels. The duplication process is handled by invoking the
    appropriate save functionality.

    :param event: Optional parameter that can be used for event-driven interactions.
        Defaults to None.
    :return: None
    """
    save_current_frame_and_labels()


def skip_frame(event=None):
    """
    Skips the current frame and progresses to the next frame based on the defined step.
    If the next frame index exceeds the total number of frames, it terminates the application.
    Otherwise, it loads the next frame.

    :param event: Optional. Event data provided when triggered by an event handler.
    :type event: Any or None
    :return: None
    """
    global frame_index
    frame_index += step
    if frame_index >= len(frame_files):
        root.quit()
    else:
        load_frame()

def back_frame(event=None):
    """
    Decreases the current frame index to move back to the previous frame. If the
    current frame index is already at the first frame, a message is printed
    indicating that it is the first frame.

    :param event: Optional; Information about the event triggering the function
        (typically passed when bound to a GUI event).
    :return: None
    """
    global frame_index
    if frame_index > 0:
        frame_index -= 1
        load_frame()
    else:
        print("At first frame")

def copy_last_annotations():
    """
    Copies the annotations from the last frame to the global variable `yolo_annotations`,
    retaining a copy of the previous frame's annotations. This ensures the latest set of
    annotations can be reused or modified as needed. Additionally, calls a function
    to redraw all annotations, immediately reflecting the updated state.

    :global yolo_annotations: A global list holding the current frame's annotations.
    :global last_frame_annotations: A global list containing the annotations from
        the previous frame.
    :return: None
    """
    global yolo_annotations, last_frame_annotations
    yolo_annotations = last_frame_annotations.copy()
    draw_all_annotations()
    print(f"Copied {len(yolo_annotations)} annotations from previous frame.")

# === UI SETUP ===
root = tk.Tk()
style = ttk.Style()
style.configure("TButton", padding=5)
style.configure("Selected.TButton", padding=5,
                background="red", foreground="black")

update_title()

first_toolbar = ttk.Frame(root)
first_toolbar.pack(side="top", fill="x")

# label buttons with hotkey numbers
for idx, lbl in enumerate(labels_list):
    btn = ttk.Button(first_toolbar,
                     text=f"{idx}: {lbl}",
                     command=lambda l=lbl: set_click_label(l))
    btn.pack(side="left")
    label_buttons[lbl] = btn
    root.bind(str(idx),  lambda e, l=lbl: set_click_label(l))

# sticky‐mode toggle
ttk.Button(first_toolbar, text="Copy Last (c)", command=copy_last_annotations).pack(side="right")


second_toolbar = ttk.Frame(root)
second_toolbar.pack(side="top", fill="x", pady=5)


sticky_btn = ttk.Button(first_toolbar, text="Sticky OFF",
                        command=toggle_sticky)
sticky_btn.pack(side="right", padx=5)

# frame controls
ttk.Button(second_toolbar, text=f"Step(+{step}) (x)", command=step_frame).pack(side="left")
ttk.Button(second_toolbar, text="Next(+1)(d)", command=next_frame).pack(side="left")
ttk.Button(second_toolbar, text=f"Skip(+{step})(s)", command=skip_frame).pack(side="left")
ttk.Button(second_toolbar, text="Back(-1)(b)", command=back_frame).pack(side="left")
ttk.Button(second_toolbar, text="Undo(Ctrl+z)", command=undo_last_annotation).pack(side="left")
ttk.Button(second_toolbar, text="dupe current frame (u)", command=duplicate_frame).pack(side="left")

canvas = tk.Canvas(root)
canvas.pack(fill="both", expand=True)
canvas.bind("<Button-1>", on_canvas_click)

# keyboard shortcuts
root.bind("x", step_frame)
root.bind("d", next_frame)
root.bind("s", skip_frame)
root.bind("b", back_frame)
root.bind("<Control-z>", undo_last_annotation)
root.bind("c", lambda e: copy_last_annotations())
root.bind("u", duplicate_frame)

load_frame()
root.mainloop()
