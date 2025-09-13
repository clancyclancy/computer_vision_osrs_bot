import win32gui
import win32ui
import win32con
import threading
import cv2
import time
import numpy as np


class Win32Capture:
    """
    This class facilitates capturing frames from a specified window in Microsoft
    Windows environments and provides an event mechanism for frame capture.

    The primary purpose of this class is to capture real-time frames from a window
    determined by a substring match of the window's title. It can be used in
    applications requiring screen capture, such as custom broadcasting, monitoring,
    or processing content visible in a particular window.

    :ivar hwnd: Handle of the window matched by the title substring.
    :type hwnd: int
    """
    def __init__(self, window_title_substring):
        self.hwnd = self.find_window(window_title_substring)

        if not self.hwnd:
            raise RuntimeError(f"Window containing '{window_title_substring}' not found.")
        self._callback = None
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)

        self._thread.start()

    def find_window(self, title_substring):
        def enum_handler(hwnd, result):
            if win32gui.IsWindowVisible(hwnd):
                if title_substring.lower() in win32gui.GetWindowText(hwnd).lower():
                    result.append(hwnd)
        result = []
        win32gui.EnumWindows(enum_handler, result)
        return result[0] if result else None

    def _capture_loop(self):
        while True:
            frame = self.capture_frame()
            if frame is not None and self._callback:
                self._callback(frame, self)
            time.sleep(1.0 / 30)  # 30 FPS (adjust if needed)

    def capture_frame(self):
        try:


            left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
            left, top = win32gui.ClientToScreen(self.hwnd, (left, top))
            right, bottom = win32gui.ClientToScreen(self.hwnd, (right, bottom))
            width = right - left
            height = bottom - top


            #hwnd_dc = win32gui.GetWindowDC(self.hwnd)
            hwnd_dc = win32gui.GetDC(self.hwnd)

            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(save_bitmap)
            save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)

            bmpinfo = save_bitmap.GetInfo()
            bmpstr = save_bitmap.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype=np.uint8)
            #img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((height, width, 4))
            img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)

            win32gui.DeleteObject(save_bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hwnd_dc)


            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            print(f"[ERROR] Failed to capture window: {e}")
            return None

    def event(self, func):
        self._callback = func
