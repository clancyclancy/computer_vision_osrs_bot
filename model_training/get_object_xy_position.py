import cv2
import tkinter as tk
import os

# 1. Load your image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(BASE_DIR, "FRAMES","frame_01316.jpg")
img_full = cv2.imread(IMG_PATH)
if img_full is None:
    raise FileNotFoundError(f"Cannot load image at {IMG_PATH}")

h_full, w_full = img_full.shape[:2]

# 2. Get screen resolution (to determine max display size)
root = tk.Tk()
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
root.destroy()

# 3. Compute uniform scale factor (never upscale)
scale = min(screen_w / w_full, screen_h / h_full, 1.0)

# 4. Resize for display
w_disp, h_disp = int(w_full * scale), int(h_full * scale)
img_disp = cv2.resize(img_full, (w_disp, h_disp), interpolation=cv2.INTER_AREA)

# 5. Mouse‐click callback
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        print(f"Display click: ({x}, {y})  →  Original: ({orig_x}, {orig_y})")

# 6. Show window and bind callback
win_name = "Click to get coordinates"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(win_name, on_mouse)

cv2.imshow(win_name, img_disp)
print("Click anywhere on the image. Press any key in the window to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()
