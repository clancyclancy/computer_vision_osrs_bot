import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

class DisplayGUI:
    def __init__(self, title="OSRS Fishing Bot Map"):
        # Create root window
        self.root = tk.Tk()
        self.root.title(title)
        self.root.state('zoomed')  # Windowed fullscreen
        #self.root.configure(bg="black")

        # Get screen dimensions
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()

        # Keep references so images don't get garbage collected
        self._images = {}
        # Store label widgets for each quadrant
        self._slots = {}

    def display_image(self, image, bottom_grid_text, box_w, box_h, loc_x1, loc_y1):
        """Display an image of box_w,box_h at the coordinates loc_x1,loc_y1 ."""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (box_w, box_h), interpolation=cv2.INTER_AREA)
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil, master=self.root)
        self._images['map'] = img_tk  # store reference

        frame = tk.Frame(self.root, bg="black")
        frame.place(
            x=loc_x1,
            y=loc_y1,
            width=box_w,
            height=box_h
        )
        map_label = tk.Label(frame, image=img_tk, bg="black")
        map_label.pack()

        text_label = tk.Label(frame, text=bottom_grid_text, font=("Arial", 14), fg="black", bg="white")
        text_label.pack()

    def _create_slot(self, name, box_w, box_h, loc_x1, loc_y1, initial_text=""):
        frame = tk.Frame(self.root, bg="white")
        frame.place(x=loc_x1, y=loc_y1, width=box_w, height=box_h + 30)

        img_label = tk.Label(frame, bg="black")
        img_label.pack()

        text_label = tk.Label(frame, text=initial_text, font=("Arial", 14), fg="black", bg="white")
        text_label.pack()

        # Store widget refs AND size for resizing later
        self._slots[name] = {
            "img_label": img_label,
            "text_label": text_label,
            "box_w": box_w,
            "box_h": box_h
        }

    def _update_slot(self, name, image_bgr, text=None):
        slot = self._slots[name]
        box_w, box_h = slot["box_w"], slot["box_h"]

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Get the height available for the image (subtract text label height)
        text_height = slot["text_label"].winfo_reqheight() or 30  # fallback if not rendered yet
        available_h = box_h - text_height

        # Scale to fit inside the available box while preserving aspect ratio
        scale = min(box_w / w, available_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil, master=self.root)

        self._images[name] = img_tk

        # Update the image label and center it in the frame
        slot["img_label"].config(image=img_tk, width=new_w, height=new_h)
        slot["img_label"].pack(expand=True)  # centers image vertically and horizontally

        if text is not None:
            slot["text_label"].config(text=text)

    def display_image_in_quadrant(self, name, image, bottom_grid_text,
                                  fraction_of_screen=2.0, location_on_screen="TOP_LEFT"):
        box_w = int(self.screen_w / fraction_of_screen)
        box_h = int(self.screen_h / fraction_of_screen)

        center_w = (self.screen_w / 2.0 - box_w) / 2.0
        center_h = (self.screen_h / 2.0 - box_h) / 2.0

        if location_on_screen == "TOP_LEFT":
            loc_x1 = center_w
            loc_y1 = center_h
        elif location_on_screen == "TOP_RIGHT":
            loc_x1 = self.screen_w / 2.0 + center_w
            loc_y1 = center_h
        elif location_on_screen == "BOTTOM_LEFT":
            loc_x1 = center_w
            loc_y1 = self.screen_h / 2.0 + center_h
        elif location_on_screen == "BOTTOM_RIGHT":
            loc_x1 = self.screen_w / 2.0 + center_w
            loc_y1 = self.screen_h / 2.0 + center_h
        else:
            loc_x1, loc_y1 = 0, 0

        if name not in self._slots:
            self._create_slot(name, box_w, box_h, loc_x1, loc_y1, bottom_grid_text)

        self._update_slot(name, image, bottom_grid_text)

        #self.display_image(image=image,bottom_grid_text=bottom_grid_text,box_w=box_w,box_h=box_h,loc_x1=loc_x1,loc_y1=loc_y1)

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()


def create_map_with_overlay(image, mini_map_image, max_loc, grid_cols, grid_rows):
    map_with_overlay = image.copy()

    h, w = mini_map_image.shape[:2]
    x0, y0 = max_loc

    # Clamp to image bounds
    x1 = min(x0 + w, map_with_overlay.shape[1])
    y1 = min(y0 + h, map_with_overlay.shape[0])

    cv2.rectangle(map_with_overlay, (x0, y0), (x1, y1), (0, 0, 0), 3)

    # Draw grid
    map_h, map_w = map_with_overlay.shape[:2]
    cell_w = map_w / grid_cols
    cell_h = map_h / grid_rows

    for i in range(1, grid_cols):
        x = int(i * cell_w)
        cv2.line(map_with_overlay, (x, 0), (x, map_h), (0, 0, 255), 2)

    for j in range(1, grid_rows):
        y = int(j * cell_h)
        cv2.line(map_with_overlay, (0, y), (map_w, y), (0, 0, 255), 2)

    return map_with_overlay

def draw_grid_box(image, x1 ,y1, x2, y2, color=(0, 0, 255), thickness=2):
    """
    Draws a rectangle around the specified grid cell.

    Args:
        image: Grayscale or color image (numpy array).
        rows, cols: Number of grid rows and columns.
        row, col: Grid cell coordinates to draw.
        color: BGR tuple for rectangle color.
        thickness: Rectangle line thickness.

    Returns:
        output_img: Copy of image with rectangle drawn.
    """

    output_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    cv2.rectangle(output_img, (x1, y1), (x2, y2), color, thickness)

    return output_img


def draw_valid_points(image, grid, grid_rows, grid_cols, color=(255, 155, 0), thickness=1):
    h, w = image.shape[:2]
    x_edges = np.linspace(0, w, grid_cols + 1, dtype=int)
    y_edges = np.linspace(0, h, grid_rows + 1, dtype=int)

    for r in range(grid_rows):
        for c in range(grid_cols):
            if grid[r][c] == 1:
                x0, x1 = x_edges[c], x_edges[c + 1]
                y0, y1 = y_edges[r], y_edges[r + 1]

                cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)
    return image


def draw_grid(image, grid_rows, grid_cols, color=(0, 0, 255), thickness=1):
    """
    Draws a perfectly aligned grid over the image using evenly spaced lines.

    Args:
        image: The image (NumPy array) to draw on.
        grid_rows: Number of horizontal divisions.
        grid_cols: Number of vertical divisions.
        color: BGR tuple for line color.
        thickness: Line thickness in pixels.

    Returns:
        The image with the grid drawn (modified in place).
    """
    map_h, map_w = image.shape[:2]

    # Compute exact pixel boundaries for columns and rows
    x_edges = np.linspace(0, map_w, grid_cols + 1, dtype=int)
    y_edges = np.linspace(0, map_h, grid_rows + 1, dtype=int)

    # Draw vertical lines
    for x in x_edges[1:-1]:  # skip first and last to avoid drawing border twice
        cv2.line(image, (x, 0), (x, map_h), color, thickness)

    # Draw horizontal lines
    for y in y_edges[1:-1]:
        cv2.line(image, (0, y), (map_w, y), color, thickness)

    return image

def draw_minimap_location(image, minimap_height, minimap_width, max_loc, color = (155,255,0), thickness = 3):

    x0, y0 = max_loc

    # Clamp to image bounds
    x1 = min(x0 + minimap_width, image.shape[1])
    y1 = min(y0 + minimap_height, image.shape[0])

    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)

    return image


def draw_astar_path(image, path, grid_rows, grid_cols, color = (0,255,0), thickness = -1):
    default_color = color
    h, w = image.shape[:2]
    x_edges = np.linspace(0, w, grid_cols + 1, dtype=int)
    y_edges = np.linspace(0, h, grid_rows + 1, dtype=int)

    for i, (r, c) in enumerate(path):
        if i == 0 or i == len(path) - 1:
            color = (0, 255, 0)
        else:
            color = default_color
        x0, x1 = x_edges[c], x_edges[c + 1]
        y0, y1 = y_edges[r], y_edges[r + 1]
        cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)


def draw_minimap_grid(minimap, valid_grid, grid_within_minimap, x_edges, y_edges, shift_x,
                              shift_y, color=(255, 155, 0), thickness=2):
    grid_rows, grid_cols = grid_within_minimap.shape


    # The total grid is relative to the total map
    # Want to draw the region of the total grid that is within the current minimap location
    # The mini map has the same resolution as the total map
    # find the distance from the first cell in the minimap to the first cell in the total grid
    # offset drawn cells by this distance

    # Find all non-zero positions
    positions = np.argwhere(grid_within_minimap != 0)
    first_row, first_col = positions[0]  # first match in row-major order
    shift_x -= x_edges[first_col]
    shift_y -= y_edges[first_row]


    for r in range(grid_rows):
        for c in range(grid_cols):

            if valid_grid[r][c] == 1: # valid
                if grid_within_minimap[r][c] == 1: # within minimap bounds

                    x0, x1 = x_edges[c], x_edges[c + 1]
                    y0, y1 = y_edges[r], y_edges[r + 1]

                    cv2.rectangle(minimap, (x0+shift_x, y0+shift_y), (x1+shift_x, y1+shift_y), color, thickness)


def draw_minimap_path(minimap, path, grid_within_minimap, shift_x,
                              shift_y, x_edges, y_edges, color=(255, 0, 0), thickness=2):

    positions = np.argwhere(grid_within_minimap != 0)
    first_row, first_col = positions[0]  # first match in row-major order
    shift_x -= x_edges[first_col]
    shift_y -= y_edges[first_row]

    for r,c in path:
            # within minimap bounds
            if grid_within_minimap[r][c] == 1:

                x0, x1 = x_edges[c], x_edges[c + 1]
                y0, y1 = y_edges[r], y_edges[r + 1]

                cv2.rectangle(minimap, (x0 + shift_x, y0 + shift_y), (x1 + shift_x, y1 + shift_y), color, thickness)


def draw_objects(image, objects, colors, display_text = False, font_scale = 0.3, thickness = 1):
    """
    Draws bounding boxes and labels on the frame
    """
    # TODO: make this better

    # if just one object, wrap into a list so for obj in objects: doesnt break
    if isinstance(objects, dict):
        objects = [objects]

    # Determine if colors is a single tuple or a dict
    is_single_color = isinstance(colors, tuple) and len(colors) == 3

    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        label = obj["label"]
        confidence = obj["confidence"]

        color = colors if is_single_color else colors.get(label, (255, 255, 255))

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        if display_text or (label != "raw" and label != "cooked"):
            # Draw label
            text = f"{label} {confidence:.2f}"
            cv2.putText(image, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def draw_static_objects(image, path, grid, grid_rows, grid_cols, grid_color, valid_color, path_color):
    """
    Draw all static objects on map
    """

    draw_grid(image    = image,
              grid_rows= grid_rows,
              grid_cols= grid_cols,
              color    = grid_color)

    draw_valid_points(image    = image,
                      grid     = grid,
                      grid_rows= grid_rows,
                      grid_cols= grid_cols,
                      color    = valid_color)

    draw_astar_path(image     = image,
                    path      = path,
                    grid_rows = grid_rows,
                    grid_cols = grid_cols,
                    color     = path_color)

import cv2
import numpy as np

def place_on_gpu_canvas(img_gpu, canvas_gpu, center_x, center_y):
    """Place img_gpu centered at (center_x, center_y) on canvas_gpu."""
    h, w = img_gpu.size()  # Note: size() returns (width, height) in GpuMat
    img_w, img_h = h, w  # careful: OpenCV CUDA swaps order

    # Calculate top-left corner for centering
    x0 = int(center_x - img_w // 2)
    y0 = int(center_y - img_h // 2)

    # Define ROI on the canvas
    roi = canvas_gpu.rowRange(y0, y0 + img_h).colRange(x0, x0 + img_w)

    # Copy image into ROI
    img_gpu.copyTo(roi)


def draw_dynamic_minimap(image, grid_within_minimap, valid_cells_grid, grid_path, lowest_path_cost, highest_path_cost, shift_x, shift_y, map_width, map_height, valid_color, path_color,
                         max_min_cells_color, thickness):

    grid_rows, grid_cols = grid_within_minimap.shape
    x_edges = np.linspace(0, map_width, grid_cols + 1, dtype=int)
    y_edges = np.linspace(0, map_height, grid_rows + 1, dtype=int)

    draw_minimap_path(image, grid_path, grid_within_minimap, shift_x,
                          shift_y, x_edges, y_edges, color=path_color, thickness=-1)

    draw_minimap_path(image, [lowest_path_cost, highest_path_cost], grid_within_minimap, shift_x,
                          shift_y, x_edges, y_edges, color=max_min_cells_color, thickness=-1)

    draw_minimap_grid(image, valid_cells_grid, grid_within_minimap, x_edges, y_edges, shift_x,
                          shift_y, color=valid_color, thickness=thickness)

def resize_to_fit(image, max_width, max_height):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)  # smallest scale to fit both dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def place_centered(target, image, top_left_x, top_left_y, quad_w, quad_h):
    h_img, w_img = image.shape[:2]
    offset_x = top_left_x + (quad_w - w_img) // 2
    offset_y = top_left_y + (quad_h - h_img) // 2
    target[offset_y:offset_y + h_img, offset_x:offset_x + w_img] = image


def draw_point(image, point, color, radius = 3, thickness = -1):
    cv2.circle(image,
               (int(point[0]), int(point[1])),
               radius=radius,
               color=color,
               thickness=thickness)


def place_text(target, lines, top_left_x, top_left_y, font = cv2.FONT_HERSHEY_SIMPLEX, scale = 0.5, color = (255,255,255), thickness = 1, line_height = 30):
    for i, line in enumerate(lines):
        cv2.putText(target, line,
                    (top_left_x, top_left_y + i * line_height),
                    font, scale, color, thickness)