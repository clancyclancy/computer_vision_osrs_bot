"""
Lumbridge Fishing & Cooking Bot logic.

This module implements the decision-making core of the bot using a simple finite
state machine (FSM). It converts model detections into structured snapshots,
tracks progress (e.g., fishing/cooking counters), and produces actionable clicks
with cooldowns based on the current state and on-screen context.

Key components:
- State: enumerates high-level phases (fishing, cooking, banking, navigation).
- EnvSnapshot: normalized view of detections for quick visibility checks.
- ProgressTracker: detects stalls to trigger corrective actions.
- LumbridgeFishingCookingBot: orchestrates state transitions and generates
  next actions (object + click position/type) from the latest snapshot.
"""


from enum import Enum, auto
import random
from typing import NamedTuple
from collections import Counter
import time


class State(Enum):
    """
    Enumeration representing different states of a system.

    This enumeration defines a set of states for various actions or phases within
    a broader workflow or system context. Each state is assigned a unique identifier
    automatically.

    .. note::
        To ensure compatibility and clarity, these states should be used to track
        specific actions within the system and for facilitating state transitions.
    """
    FISHING = auto()
    TRAVEL_TO_RANGE = auto()
    COOKING = auto()
    TRAVEL_UPSTAIRS = auto()
    TRAVEL_TO_BANK = auto()
    BANKING = auto()
    TRAVEL_DOWNSTAIRS = auto()
    TRAVEL_TO_FISHING = auto()

class EnvSnapshot(NamedTuple):
    """
    Represents a snapshot of the environment at a specific moment.

    This class is a NamedTuple that captures the state of various aspects 
    of the environment, including counts of resources, visibility of major 
    objects, and visibility of UI elements. It is used to provide a clear 
    and concise representation of the current state for further processing 
    or decision-making.

    :ivar raw_fish: The number of raw fish currently available.
    :type raw_fish: int
    :ivar cooked_fish: The number of cooked fish currently available.
    :type cooked_fish: int
    :ivar fishing_spot: The number of fishing spots currently detected.
    :type fishing_spot: int
    :ivar fishing_spot_visible: Whether the fishing spot is currently visible.
    :type fishing_spot_visible: bool
    :ivar cooking_range_visible: Whether the cooking range is currently visible.
    :type cooking_range_visible: bool
    :ivar bank_visible: Whether the bank is currently visible.
    :type bank_visible: bool
    :ivar stairs_visible: Whether the stairs are currently visible.
    :type stairs_visible: bool
    :ivar interact_up_visible: Whether the upward interaction UI element is visible.
    :type interact_up_visible: bool
    :ivar interact_down_visible: Whether the downward interaction UI element is visible.
    :type interact_down_visible: bool
    :ivar textbox_visible: Whether the textbox UI element is visible.
    :type textbox_visible: bool
    :ivar bank_screen_visible: Whether the bank screen UI is visible.
    :type bank_screen_visible: bool
    :ivar bank_deposit_visible: Whether the bank deposit UI is visible.
    :type bank_deposit_visible: bool
    """
    #counts
    raw_fish:     int
    cooked_fish:  int
    fishing_spot: int

    #major objects
    fishing_spot_visible:  bool
    cooking_range_visible: bool
    bank_visible:          bool
    stairs_visible:        bool

    #UI objects
    interact_up_visible:   bool
    interact_down_visible: bool
    textbox_visible:       bool
    bank_screen_visible:   bool
    bank_deposit_visible:  bool

class Action:
    """
    Represents an action that can be performed, such as interacting with an object
    or handling positional screen elements.

    This class provides a template for configuring actions, particularly defining
    an associated object, a position in the screen's coordinate space, and the type
    of click interaction. Use this as a base to develop actions involving GUI elements
    or positional clicks.

    :ivar object: The associated object with the action.
    :ivar position: The screen position where the action occurs, represented as
        coordinates (x, y).
    :ivar click: The type of click associated with the action, such as a left-click
        or right-click.
    """
    def __init__(self):
        self.object   = None
        self.position = None  # (x, y) screen coordinates
        self.click    = None



class ProgressTracker:
    """
    Tracks the progress of an operation by monitoring value changes and determines 
    whether the operation has stalled or not.

    The ProgressTracker class is used to monitor incremental progress of 
    a numeric counter and detect periods of inactivity (stalls). It resets 
    status when the counter is set to a lower value, assuming a reset in 
    the operation being tracked. This is useful in systems where a timeout 
    must trigger a specific action when progress halts.

    :ivar timeout: Timeout in seconds to determine if progress is stalled.
    :type timeout: int
    :ivar last_count: The last known count value for tracking progress.
    :type last_count: int
    :ivar last_increase_time: The last recorded time when the count 
        increased, used for stall detection.
    :type last_increase_time: float
    """
    def __init__(self, timeout=10):
        self.last_count = 0
        self.last_increase_time = time.time()
        self.timeout = timeout  # seconds

    def update(self, current_count):
        now = time.time()
        if current_count > self.last_count:
            #count increased
            self.last_increase_time = now
            self.last_count = current_count
            return True  # progress made
        elif current_count < self.last_count:
            # Inventory reset
            self.last_count = current_count
            self.last_increase_time = now
        return False  # no progress this tick

    def is_stalled(self):
        return (time.time() - self.last_increase_time) > self.timeout


class LumbridgeFishingCookingBot:
    """
    Handles the operation and state management of a bot designed for fishing and
    cooking in the Lumbridge area in a game.

    The LumbridgeFishingCookingBot class manages the action requests and decision-
    making for the bot based on its current state and environment. The bot can detect 
    and handle various in-game states and decide its next action such as fishing, 
    cooking, navigating, and interacting with objects like banks and cooking ranges. 
    It uses finite state machines and tracks progression for fishing and cooking 
    activities to manage its tasks effectively.

    :ivar action_request_object: Object requested for action processing.
    :type action_request_object: Any
    :ivar objects: Current viewable objects in the game environment.
    :type objects: Any
    :ivar initialized: Indicates whether the bot has been initialized.
    :type initialized: bool
    :ivar state: Current state of the bot derived from `State`.
    :type state: State
    :ivar fishing_tracker: Tracks progress related to fishing activities.
    :type fishing_tracker: ProgressTracker
    :ivar cooking_tracker: Tracks progress related to cooking activities.
    :type cooking_tracker: ProgressTracker
    :ivar snapshot: Snapshot of the current environment providing visibility of 
        interactable objects and counts of raw and cooked fish.
    :type snapshot: EnvSnapshot
    :ivar minimap_in_world_map: Current minimap-to-world mapping.
    :type minimap_in_world_map: Any
    :ivar action: Holds the current action request for the bot to perform.
    :type action: Action
    :ivar last_action_time: Tracks the time of the last performed action in seconds.
    :type last_action_time: float
    :ivar DEFAULT_ACTION_COOLDOWN: Default cooldown time for general actions.
    :type DEFAULT_ACTION_COOLDOWN: float
    :ivar DEFAULT_FAST_ACTION_COOLDOWN: Default cooldown time for faster actions.
    :type DEFAULT_FAST_ACTION_COOLDOWN: float
    :ivar DEFAULT_SLOW_ACTION_COOLDOWN: Default cooldown time for slower actions.
    :type DEFAULT_SLOW_ACTION_COOLDOWN: float
    :ivar last_action_cooldown: Cooldown time associated with the last action.
    :type last_action_cooldown: float
    :ivar pending_action: Indicates if there is any pending action request.
    :type pending_action: bool
    :ivar CLICK_TYPE: Maps object labels to their respective click types used for actions.
    :type CLICK_TYPE: dict
    :ivar ACTION_COOLDOWN: Maps object labels to their respective action cooldown durations.
    :type ACTION_COOLDOWN: dict
    :ivar INCREASE_CELL_COST: Determines whether object interaction increases the 
        cell movement cost for pathfinding purposes.
    :type INCREASE_CELL_COST: dict
    :ivar CLICK_Y_SHIFT: Maps object labels to their vertical shift value for click 
        positioning.
    :type CLICK_Y_SHIFT: dict
    """
    def __init__(self):

        self.action_request_object = None
        self.objects = None
        self.initialized = False

        self.state    = None

        self.fishing_tracker = ProgressTracker()
        self.cooking_tracker = ProgressTracker()

        self.snapshot = None

        self.minimap_in_world_map = None

        self.action = Action()
        self.last_action_time = 0
        self.DEFAULT_ACTION_COOLDOWN = 1.5  # seconds
        self.DEFAULT_FAST_ACTION_COOLDOWN = .75  # seconds
        self.DEFAULT_SLOW_ACTION_COOLDOWN = 5.0  # seconds

        self.last_action_cooldown = 0

        self.pending_action = False


        self.CLICK_TYPE = {
        "bank":          "Left_Click",
        "player":        "",
        "fishing_spot":  "Left_Click",
        "raw":           "",
        "cooked":        "Right_Click",
        "cooking_range": "Left_Click",
        "stairs":        "Right_Click",
        "interact_up":   "Left_Click",
        "interact_down": "Left_Click",
        "textbox":       "Space_Bar",
        "bank_screen":   "",
        "bank_deposit":  "Left_Click",
        "highest_cell":  "Left_Click",
        "lowest_cell":   "Left_Click"
        }
        
        self.ACTION_COOLDOWN = {
        "bank":          self.DEFAULT_SLOW_ACTION_COOLDOWN,
        "player":        self.DEFAULT_ACTION_COOLDOWN,
        "fishing_spot":  self.DEFAULT_SLOW_ACTION_COOLDOWN,
        "raw":           self.DEFAULT_ACTION_COOLDOWN,
        "cooked":        self.DEFAULT_FAST_ACTION_COOLDOWN,
        "cooking_range": self.DEFAULT_SLOW_ACTION_COOLDOWN,
        "stairs":        self.DEFAULT_FAST_ACTION_COOLDOWN,
        "interact_up":   self.DEFAULT_ACTION_COOLDOWN,
        "interact_down": self.DEFAULT_ACTION_COOLDOWN,
        "textbox":       self.DEFAULT_ACTION_COOLDOWN,
        "bank_screen":   self.DEFAULT_ACTION_COOLDOWN,
        "bank_deposit":  self.DEFAULT_FAST_ACTION_COOLDOWN,
        "highest_cell":  self.DEFAULT_ACTION_COOLDOWN,
        "lowest_cell":   self.DEFAULT_ACTION_COOLDOWN
        }

        self.INCREASE_CELL_COST = {
        "bank":          False,
        "fishing_spot":  True,
        "cooking_range": False,
        "stairs_to_go_upstairs":        False,
        "stairs_to_go_downstairs":        True,
        }


        self.CLICK_Y_SHIFT = {
        "bank":          0,
        "player":        0,
        "fishing_spot":  0,
        "raw":           0,
        "cooked":        0,
        "cooking_range": 0,
        "stairs":        0,
        "interact_up":   0.43 - 0.5,
        "interact_down": 0.42 - 0.5,
        "textbox":       0,
        "bank_screen":   0,
        "bank_deposit":  0.685 - 0.5,
        "highest_cell":  0,
        "lowest_cell":   0
        }


    def detect_initial_state(self):
        """Infer the starting FSM state from current visibility and inventory counters."""

        # in bank screen or near bank with items to deposit
        if (self.is_bank_interface_open() or self.is_bank_booth_visible()) and self.count_cooked_fish() > 0:
            return State.BANKING

        # near cooking range with fish to cook
        if self.is_cooking_range_visible() and self.count_raw_fish() > 0:
            return State.COOKING

        # on ground floor with alot of fish
        if self.count_raw_fish() > 22 and not self.is_upstairs():
            return State.TRAVEL_TO_RANGE

        # on ground floor with cooked fish (and no raw fish, previous check)
        if self.count_cooked_fish() > 0 and not self.is_upstairs():
            return State.TRAVEL_UPSTAIRS

        # on top floor with cooked fish
        if self.count_cooked_fish() > 0 and self.is_upstairs():
            return State.TRAVEL_TO_BANK

        # upstairs (and empty inventory)
        if self.is_upstairs():
            print("init travel downstairs")
            return State.TRAVEL_DOWNSTAIRS

        # if on ground floor
        if not self.is_upstairs():
            return State.TRAVEL_TO_FISHING

        # if near fishing spot
        if self.is_fishing_spot_visible():
            return State.FISHING
        # Fallback
        return State.TRAVEL_DOWNSTAIRS if self.is_upstairs() else State.TRAVEL_TO_FISHING

    def initialize(self):
        """Set initial FSM state once and mark the bot as initialized."""

        self.state = self.detect_initial_state()
        self.initialized = True


    def request_action(self, objects, minimap_in_world_map):
        """Consume detections and context to produce the next actionable click (or None if cooling down)."""


        # set objects
        self.objects = objects
        self.minimap_in_world_map = minimap_in_world_map

        #update trackers
        # Get current objects in view
        label_counts = Counter([obj['label'] for obj in self.objects])
        self.snapshot = EnvSnapshot(raw_fish              = label_counts.get('raw', 0),
                                    cooked_fish           = label_counts.get('cooked', 0),
                                    fishing_spot          = label_counts.get('fishing_spot', 0),
                                    fishing_spot_visible  = label_counts.get('fishing_spot', 0) > 0, #TODO: better way to do this?
                                    cooking_range_visible = label_counts.get('cooking_range', 0) > 0,
                                    bank_visible          = label_counts.get('bank', 0) > 0,
                                    stairs_visible        = label_counts.get('stairs', 0) > 0,
                                    interact_up_visible   = label_counts.get('interact_up', 0) > 0,
                                    interact_down_visible = label_counts.get('interact_down', 0) > 0,
                                    textbox_visible       = label_counts.get('textbox', 0) > 0,
                                    bank_screen_visible   = label_counts.get('bank_screen', 0) > 0,
                                    bank_deposit_visible  = label_counts.get('bank_deposit', 0) > 0,
                                    )

        # track trends
        self.fishing_tracker.update(self.snapshot.raw_fish)
        self.cooking_tracker.update(self.snapshot.cooked_fish)


        # don't call process until ready for an action
        if (time.perf_counter() - self.last_action_time) < self.last_action_cooldown:
            return None


        # process
        self.process()

        # create action
        if self.action_request_object is not None:
            self.process_action_request()
            return self.action

        return None

    def process_action_request(self):
        """Translate an action request object into a concrete click action and set cooldowns."""

        # store object just for drawing
        self.action.object = self.action_request_object

        # click type
        self.action.click = self.CLICK_TYPE.get(self.action.object["label"])

        #click location
        x,y = get_center(self.action.object["bbox"])
        width, height = get_width_height(self.action.object["bbox"])

        self.action.position = (x,y + height * self.CLICK_Y_SHIFT.get(self.action.object["label"]))

        #set times
        self.last_action_time = time.perf_counter()
        self.last_action_cooldown = self.ACTION_COOLDOWN.get(  self.action.object["label"]) * random.uniform(0.85, 1.15)

        #reset action request
        self.action_request_object = None

    def process(self):
        """Run one FSM tick: handle UI interactions, then advance state-specific behavior."""

        if not self.initialized:
            self.initialize()


        # PROCESS INTERACTABLE OBJECTS (textbox, drop down menus)
        if self.snapshot.textbox_visible:
            self.action_request_object = self.get_object("textbox")
            return

        if self.snapshot.bank_deposit_visible:
            self.action_request_object = self.get_object("bank_deposit")
            return

        if self.snapshot.interact_up_visible:
            self.action_request_object = self.get_object("interact_up")
            return

        if self.snapshot.interact_down_visible:
            self.action_request_object = self.get_object("interact_down")
            return


        # PROCESS FINITE STATE MACHINE
        if self.state == State.FISHING:
            self.fish_at_spot()
            if self.inventory_full(self.fishing_tracker.last_count):
                self.state = State.TRAVEL_TO_RANGE

        elif self.state == State.TRAVEL_TO_RANGE:
            self.navigate_to("cooking_range")
            if self.at_cooking_range():
                self.state = State.COOKING

        elif self.state == State.COOKING:
            self.cook_all()
            if self.inventory_empty(self.fishing_tracker.last_count):
                self.state = State.TRAVEL_UPSTAIRS

        elif self.state == State.TRAVEL_UPSTAIRS:
            self.climb_stairs(up=True)
            if self.at_upstairs():
                self.state = State.TRAVEL_TO_BANK

        elif self.state == State.TRAVEL_TO_BANK:
            self.navigate_to("bank")
            if self.at_bank():
                self.state = State.BANKING

        elif self.state == State.BANKING:
            self.deposit_all_cooked()
            if self.inventory_empty(self.cooking_tracker.last_count):
                self.state = State.TRAVEL_DOWNSTAIRS

        elif self.state == State.TRAVEL_DOWNSTAIRS:
            self.climb_stairs(up=False)
            if self.at_ground_floor():
                self.state = State.TRAVEL_TO_FISHING

        elif self.state == State.TRAVEL_TO_FISHING:
            self.navigate_to("fishing_spot")
            if self.at_fishing_spot():
                self.state = State.FISHING


    # -------------------------
    # Action Method Helpers
    # -------------------------

    def fish_at_spot(self):
        """If fishing stalls, click a fishing spot (closest when multiple are visible)."""
        
        # TODO: probably don't need the number of fishing spots check and can just have get_object() return the closest?
        if self.fishing_tracker.is_stalled():
            if self.snapshot.fishing_spot > 1:
                # click on nearest fishing spot to player
                self.action_request_object = self.get_closest_object("fishing_spot")
            else:
                # click on fishing spot
                self.action_request_object = self.get_object("fishing_spot")

    def inventory_full(self, count) -> bool:
        """Return True when (raw) inventory is considered full (threshold tuned for OSRS)."""

        #TODO: sometimes special items are fished, add a timer to see the last
        # time 26 fish were reached, if it was awhile ago, set the max to the most recent highest fish count?
        return count >= 26 #OSRS has a 7x4 inventory size, but 2 spots are used

    def navigate_to(self, location): 
        """Request movement toward a named location by clicking a path cell (low/high cost)."""

        if self.INCREASE_CELL_COST.get(location):
            object = "lowest_cell"
        else:
            object = "highest_cell"
        self.action_request_object = self.get_object(object)

    def at_cooking_range(self) -> bool:
        """Return True if the cooking range is currently visible."""

        return self.is_cooking_range_visible()

    def cook_all(self):
        """If cooking progress stalls, click the cooking range again to continue."""

        if self.cooking_tracker.is_stalled():
            self.action_request_object = self.get_object("cooking_range")

    def inventory_empty(self, count) -> bool:
        """Return True when the tracked count indicates inventory is empty."""

        return count == 0

    def climb_stairs(self, up: bool):
        """Navigate stairs: click stairs if visible; otherwise path toward the appropriate staircase."""

        #TODO: maybe redo this, only need self.is_stairs_visible() and is_player_on_ground_floor from main.process()?
        if self.is_stairs_to_go_upstairs_visible() and up:
            self.action_request_object = self.get_object("stairs")

        elif self.is_stairs_to_go_downstairs_visible() and not up:
            self.action_request_object = self.get_object("stairs")

        else:
            if up:
                self.navigate_to("stairs_to_go_upstairs")
            else:
                self.navigate_to("stairs_to_go_downstairs")


    def at_upstairs(self) -> bool:
        """Return True when the bot is on the upstairs map (based on minimap classification)."""

        return not self.minimap_in_world_map

    def at_bank(self) -> bool:
        """Return True when the bank booth is visible."""

        return self.is_bank_booth_visible()

    def deposit_all_cooked(self):
        """Handle bank deposit flow: click bank when visible, then deposit cooked items when in bank UI."""

        if self.is_bank_booth_visible():
            self.action_request_object = self.get_object("bank")

        elif self.is_bank_screen_visible():
            self.action_request_object = self.get_object("cooked")

    def at_ground_floor(self) -> bool:
        """Return True when the bot is on the ground floor map (world map)."""

        return self.minimap_in_world_map

    def at_fishing_spot(self) -> bool:
        """Return True if a fishing spot is currently visible."""

        return self.is_fishing_spot_visible()
    # -------------------------
    # Inventory Count Helpers
    # -------------------------
    def count_cooked_fish(self) -> int:
        """Get the latest tracked count of cooked fish."""

        return self.cooking_tracker.last_count

    def count_raw_fish(self) -> int:
        """Get the latest tracked count of raw fish."""
        
        return self.fishing_tracker.last_count


    # -------------------------
    # Object Visibility Helpers
    # -------------------------
    def is_bank_interface_open(self) -> bool:
        """Return True if the bank interface is currently open."""

        return self.snapshot.bank_screen_visible

    def is_bank_booth_visible(self) -> bool:
        """Return True if a bank booth is visible in the scene."""

        return self.snapshot.bank_visible

    def is_cooking_range_visible(self) -> bool:
        """Return True if a cooking range is visible in the scene."""

        return self.snapshot.cooking_range_visible

    def is_stairs_to_go_upstairs_visible(self) -> bool:
        """Return True if stairs are visible and the bot is on the ground floor (can go up)."""

        return self.snapshot.stairs_visible and not self.is_upstairs()

    def is_stairs_to_go_downstairs_visible(self) -> bool:
        """Return True if stairs are visible and the bot is upstairs (can go down)."""

        return self.snapshot.stairs_visible and self.is_upstairs()

    def is_fishing_spot_visible(self) -> bool:
        """Return True if a fishing spot label is currently visible."""

        return self.snapshot.fishing_spot_visible

    def is_interact_up_visible(self) -> bool:
        """Return True if the 'interact up' UI element is visible."""

        return self.snapshot.interact_up_visible

    def is_interact_down_visible(self) -> bool:
        """Return True if the 'interact down' UI element is visible."""

        return self.snapshot.interact_down_visible

    def is_textbox_visible(self) -> bool:
        """Return True if the textbox UI element is visible."""

        return self.snapshot.textbox_visible

    def is_bank_screen_visible(self) -> bool:
        """Return True if the bank screen UI is visible."""

        return self.snapshot.bank_screen_visible

    def is_bank_deposit_visible(self) -> bool:
        """Return True if the bank deposit button/area is visible."""

        return self.snapshot.bank_deposit_visible

    def is_upstairs(self):
        """Return True when the map context indicates the bot is upstairs."""

        return not self.minimap_in_world_map

    def get_object(self, object_label: str) -> object:
        """Return the first detected object matching the given label (or None if not found)."""

        return next((obj for obj in self.objects if obj["label"] == object_label), None)

    def get_closest_object(self, object_label: str) -> object:
        """Among objects of a given label, return the one with minimal distance to the player."""


        player = next((obj for obj in self.objects if obj["label"] == "player"), None)
        desired_objects = (obj for obj in self.objects if obj["label"] == object_label)

        #prevent Error
        if player is None or desired_objects is None:
            return None

        #find the desired_objects closest to the player
        closest = None
        min_dist = float("inf")

        for obj in desired_objects:
            dist = get_distance(player["bbox"], obj["bbox"])
            if dist < min_dist:
                min_dist = dist
                closest = obj
        return closest




def get_distance(bbox1, bbox2):
    """
    Calculates the Euclidean distance between the centers of two bounding boxes.

    The function computes the centers of the two provided bounding boxes and calculates
    the Euclidean distance between them. This can be useful for applications that rely
    on spatial measurements between object bounding boxes, such as object tracking or 
    spatial analysis.

    :param bbox1: The first bounding box specified as a tuple or list with coordinates.
    :param bbox2: The second bounding box specified as a tuple or list with coordinates.
    :return: The computed Euclidean distance as a float between the centers of bbox1 and bbox2.

    """
    x1, y1 = get_center(bbox1)
    x2, y2 = get_center(bbox2)
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def get_center(bbox):
    """
    Calculates and returns the center point of a bounding box.

    The function takes a bounding box (defined by its minimum and maximum x and 
    y coordinates) and computes the center point's coordinates as a tuple.

    :param bbox: A sequence of floats or integers representing the bounding 
        box coordinates in the order (x_min, y_min, x_max, y_max).
    :return: A tuple containing the x and y coordinates of the center point 
        as floats.
    """
    x_min, y_min, x_max, y_max = bbox
    return (x_min + x_max) / 2, (y_min + y_max) / 2

def get_width_height(bbox):
    """
    Calculates the width and height of a bounding box based on its given coordinates.

    :param bbox: A tuple containing bounding box coordinates in the order 
                 (x_min, y_min, x_max, y_max).
    :type bbox: tuple[float, float, float, float]
    :return: A tuple containing the width and height of the bounding box. 
             The width is calculated as (x_max - x_min), and the height as 
             (y_max - y_min).
    :rtype: tuple[float, float]
    """
    x_min, y_min, x_max, y_max = bbox
    return x_max - x_min, y_max - y_min
