from pathlib import Path


class Settings:
    """
    Contains configuration settings and constants for the application.

    This class defines a variety of settings and constants commonly used throughout
    the application. It includes predefined configurations for canvas dimensions,
    user interface elements, application behavior, and file paths. It serves as a central
    repository of static and configurable values, simplifying access and ensuring consistency.

    :ivar int CANVAS_WIDTH: The width of the main canvas in pixels.
    :ivar int CANVAS_HEIGHT: The height of the main canvas in pixels.
    :ivar int QUADRANT_WIDTH: Half the width of the main canvas, calculated as an integer.
    :ivar int QUADRANT_HEIGHT: Half the height of the main canvas, calculated as an integer.
    :ivar APPLICATION_WINDOW: The name of the application window.
    :ivar USE_GPU: Flag indicating whether GPU should be used.
    :ivar int MINIMAP_x1: The x-coordinate of the top-left corner of the minimap.
    :ivar int MINIMAP_y1: The y-coordinate of the top-left corner of the minimap.
    :ivar int MINIMAP_x2: The x-coordinate of the bottom-right corner of the minimap.
    :ivar int MINIMAP_y2: The y-coordinate of the bottom-right corner of the minimap.
    :ivar int MINIMAP_COORDS_NOMINAL_SCALE_WIDTH: The nominal width of the minimap scale in pixels.
    :ivar int MINIMAP_COORDS_NOMINAL_SCALE_HEIGHT: The nominal height of the minimap scale in pixels.
    :ivar bool WAIT_ON_MODEL: Indicates whether to always wait for the model to finish before using objects.
    :ivar FRAME_RATE: The expected frame rate of the application in frames per second.
    :ivar VALID_CELLS_COLOR: RGB color tuple for valid grid cells.
    :ivar CHOSEN_PATH_COLOR: RGB color tuple for the chosen path.
    :ivar GRID_COLOR: RGB color tuple for the grid display.
    :ivar COLORS: A dictionary of pre-defined category labels mapped to their respective RGB color codes.
    :ivar BASE_DIR: The base file directory path where the application files are located.
    :ivar WORLD_MAP_PATH: Path to the world map image file.
    :ivar TOP_FLOOR_MAP_PATH: Path to the top floor map image file.
    :ivar MODEL_PATH: Path to the machine learning model weights file.
    :ivar bool TEST_WITH_SCREENSHOT: Flag indicating whether to test using a static screenshot.
    :ivar TEST_WITH_SCREENSHOT_PATH: Path to the test screenshot when TEST_WITH_SCREENSHOT is enabled.
    :ivar bool TEST_WITH_RECORDING: Flag indicating whether to test using a video recording.
    :ivar TEST_WITH_RECORDING_PATH: Path to the test video when TEST_WITH_RECORDING is enabled.
    :ivar bool TESTING: Indicates whether testing is enabled, determined by the TEST_WITH_SCREENSHOT
        or TEST_WITH_RECORDING flags.

    """
    CANVAS_WIDTH  = 1280
    CANVAS_HEIGHT = 720
    QUADRANT_WIDTH  = CANVAS_WIDTH // 2
    QUADRANT_HEIGHT = CANVAS_HEIGHT // 2


    APPLICATION_WINDOW = "RuneLite"

    USE_GPU = True

    MINIMAP_x1 = 1586
    MINIMAP_y1 = 104
    MINIMAP_x2 = 1818
    MINIMAP_y2 = 296

    MINIMAP_COORDS_NOMINAL_SCALE_WIDTH  = 1920
    MINIMAP_COORDS_NOMINAL_SCALE_HEIGHT = 1080

    WAIT_ON_MODEL = True #Don't use stale objects, always wait for model to finish

    FRAME_RATE = 40

    VALID_CELLS_COLOR = (0, 155, 255)
    CHOSEN_PATH_COLOR = (255, 155, 0)
    GRID_COLOR        = (100, 100, 100)

    COLORS = {
        "bank":          (255, 155, 155),
        "player":        (155, 255, 155),
        "fishing_spot":  (0, 155, 155),
        "raw":           (0, 155, 255),
        "cooked":        (255, 155, 0),
        "cooking_range": (0, 255, 255),
        "stairs":        (155, 155, 255),
        "interact_up":   (155, 0, 0),
        "interact_down": (155, 0, 0),
        "textbox":       (155, 0, 0),
        "bank_screen":   (155, 0, 0),
        "bank_deposit":  (255, 0, 155)
    }


    BASE_DIR = Path(__file__).resolve().parent.parent

    WORLD_MAP_PATH     = BASE_DIR / "images" / "map.png"
    TOP_FLOOR_MAP_PATH = BASE_DIR / "images" / ("top_floor_map.png")
    #200 epochs 416x416 imagsz
    #MODEL_PATH         = BASE_DIR / "model_training" / "runs" / "detect" / "train8" / "weights" / "best.pt"
    #300 epochs 320x320 imagsz
    #MODEL_PATH         = BASE_DIR / "model_training" / "runs" / "detect" / "train9" / "weights" / "best.pt"
    #400 epochs 256x256 imagsz
    MODEL_PATH         = BASE_DIR / "model_training" / "runs" / "detect" / "train256" / "weights" / "best.pt"

    TEST_WITH_SCREENSHOT      = False
    TEST_WITH_SCREENSHOT_PATH = BASE_DIR / "images" / "test_frame_1.png"

    TEST_WITH_RECORDING = False
    TEST_WITH_RECORDING_PATH = r"C:\Users\mclan\Videos\OBS\OSRS_FISHINGBOTv2.mp4"

    TESTING = TEST_WITH_RECORDING or TEST_WITH_SCREENSHOT

    DRAW_LAST_ACTION_LOCATION = False
