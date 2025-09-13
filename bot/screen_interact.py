"""
Screen interaction utilities (Windows).

This module focuses an application window and synthesizes user input to execute
bot actions. It provides:
- execute_action: high-level entry that focuses the client, sends a click or
  space-bar press, then restores focus to a display window if appropriate.
- focus: brings a window to the foreground (with a safe ALT-tap workaround).
- press_space / press_mouse: low-level input helpers for key and mouse events.

Notes:
- Coordinates for mouse actions are expected in absolute screen space (pixels).
- Small randomized delays are used to mimic human interaction and avoid
  excessively robotic timing.
- Use responsibly; synthetic input can affect the active user session.
"""

import win32gui
import win32con
import time
import win32api
import random

def execute_action(action, application_window, display_model_window):
    """
    Executes a specified action by interacting with the UI elements of the given application
    window and display model window. Depending on the provided action, it either clicks
    on a specified position or simulates pressing the space bar. The function also manages
    the focus between the application and the GUI windows appropriately.

    :param action: Object representing the action to be executed. Contains details such as
        position and the type of click or keypress.
    :param application_window: List or tuple representing the application window to be
        brought into focus for executing the action.
    :param display_model_window: The window element representing the GUI, which is brought
        back into focus after the action is performed.
    :return: None
    """
    # bring client into focus
    focus(application_window[0])

    if action.click == "Space_Bar":
        press_space(click_delay = round(random.uniform(0.05, 0.1), 2))
    else:
        x_pos,y_pos = action.position
        press_mouse(x_pos       = x_pos,
                    y_pos       = y_pos,
                    click_type  = action.click,
                    click_delay = round(random.uniform(0.05, 0.1), 2))

    #bring GUI back into focus
    if action.click != "Right_Click":
        focus(display_model_window)


    pass


def focus(window_title):
    """
    Bring a specified window to the foreground and maximize it.

    This function utilizes the Win32 API to find a window by its title, bring it to
    the front of the screen in its normal window state, and ensure it becomes the
    foreground window.

    :param window_title: The title of the window as a string to be brought to the
        foreground.
    :type window_title: str
    :return: None
    """
    # get handle of window
    hwnd = win32gui.FindWindow( None,
                               window_title)

    # bring to front maximized
    #win32gui.ShowWindow(hwnd, win32con.SW_SHOWMAXIMIZED)
    win32gui.ShowWindow(hwnd, win32con.SW_SHOW)

    # Try to set foreground
    if not _set_foreground(hwnd):
        _press_alt()
        _set_foreground(hwnd)


def press_space(click_delay=0.05):
    """
    Simulates a spacebar press event with an optional delay between pressing and
    releasing the key.

    :param click_delay: The delay in seconds between pressing and releasing the
        spacebar. Defaults to 0.05 seconds.
    :type click_delay: float
    :return: None
    """
    # Press space
    win32api.keybd_event(win32con.VK_SPACE, 0, 0, 0)

    time.sleep(click_delay)

    # Release space
    win32api.keybd_event(win32con.VK_SPACE, 0, win32con.KEYEVENTF_KEYUP, 0)

def press_mouse(x_pos, y_pos, click_type, click_delay = 0.05):
    """
    Simulates a mouse press event at a specified position with a specified button type
    and optional delay. This function allows simulating left or right mouse button clicks
    at an arbitrary cursor position.

    :param x_pos: The x-coordinate of the position where the mouse click will occur.
    :type x_pos: int
    :param y_pos: The y-coordinate of the position where the mouse click will occur.
    :type y_pos: int
    :param click_type: The type of mouse button click to simulate. Acceptable values are
                       "Left_Click" for left-click and "Right_Click" for right-click.
    :type click_type: str
    :param click_delay: The delay in seconds between mouse button press and release events.
                        Default is 0.05 seconds.
    :type click_delay: float, optional
    :raises ValueError: If the `click_type` is not "Left_Click" or "Right_Click".
    :return: None
    """
    # Move the mouse
    win32api.SetCursorPos((int(x_pos), int(y_pos)))


    # Choose down/up events based on button
    if click_type == "Left_Click":
        down, up = win32con.MOUSEEVENTF_LEFTDOWN, win32con.MOUSEEVENTF_LEFTUP
    elif click_type == "Right_Click":
        down, up = win32con.MOUSEEVENTF_RIGHTDOWN, win32con.MOUSEEVENTF_RIGHTUP
    else:
        raise ValueError(" [ERROR] action mouse click must be Left_Click or Right_Click")

    # Click
    time.sleep(click_delay)
    win32api.mouse_event(down, 0, 0, 0, 0)

    time.sleep(click_delay)
    win32api.mouse_event(up, 0, 0, 0, 0)

def _set_foreground(hwnd):
    """
    Sets the specified window to the foreground, bringing it to the front and activating
    it. This method interacts with the system GUI to manipulate the focus of the
    windows manager. It returns a boolean indicating whether the operation was
    successful or not.

    :param hwnd: The handle to the window to be set as the foreground window.
    :type hwnd: int
    :return: True if the window was successfully set to foreground, False otherwise.
    :rtype: bool
    """
    try:
        win32gui.SetForegroundWindow(hwnd)
        return True
    except Exception:
        return False

def _press_alt():
    """
    Simulates an ALT key press event programmatically. This function uses the
    `win32api` and `win32con` libraries to emulate the pressing and releasing
    of the ALT key. It injects a small delay between the key press and release
    to mimic a natural key press event.

    :error:
        This function does not handle exceptions explicitly. If the `win32api`
        or `win32con` libraries are unavailable or fail, it will raise an
        error inherently.

    """
    win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)  # ALT down
    time.sleep(0.001)
    win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)  # ALT up
