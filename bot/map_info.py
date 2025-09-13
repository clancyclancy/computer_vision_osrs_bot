"""
Map data structures and configuration for navigation.

This module defines:
- Map: a lightweight container for a static map image plus derived artifacts
  used during navigation (display image, valid-cell grid, path, and costs).
- WorldMapConfig and TopFloorMapConfig: dataclass-based configurations that
  specify scaling factors, grid dimensions, manual grid overrides, and A*
  start/end cells for different map contexts.

These definitions decouple map-specific parameters from the navigation and
vision logic, making it easy to swap maps or tune pathing without touching
the bot’s core decision code.
"""


from dataclasses import dataclass, field
from typing import List, Tuple


class Map:
    """
    Represents a map for processing and displaying image-based grid data.

    Handles image processing, grid configuration, validation of cells,
    and pathfinding operations on a map system.

    :ivar image: The original input image to the map.
    :type image: Any
    :ivar display_image: The processed image used for display purposes.
    :type display_image: Any
    :ivar config: A dataclass containing configuration details for the map.
    :type config: dataclass
    :ivar valid_cells_grid: A list representing the valid cells on the map grid.
    :type valid_cells_grid: list
    :ivar path: A list representing the computed path on the map.
    :type path: list
    :ivar path_costs: A list representing the costs associated with the computed path.
    :type path_costs: list
    """
    def __init__(self,image):

        self.image         = image
        self.display_image = image

        self.config = [] #dataclass

        self.valid_cells_grid = []
        self.path             = []
        self.path_costs       = []


@dataclass
class WorldMapConfig:
    """
    Configuration for the World Map with adjustable parameters.

    This class defines the configuration settings for a 2D grid-based world map. It includes
    parameters for scaling, manual modifications of the map grid, and defining valid cells
    to be added or removed. Additionally, it specifies the start and end points for pathfinding
    algorithms like A*.

    :ivar SCALE_FACTOR: A scaling factor pre-calculated for a 1080x1920 screen. It adjusts
        the map's dimensions based on the required resolution.
    :type SCALE_FACTOR: float
    :ivar MAP_GRID_ROWS: The number of rows in the map grid.
    :type MAP_GRID_ROWS: int
    :ivar MAP_GRID_COLS: The number of columns in the map grid.
    :type MAP_GRID_COLS: int
    :ivar MANUALLY_MODIFY_GRID: A flag indicating whether the modification of the grid is
        governed manually or not.
    :type MANUALLY_MODIFY_GRID: bool
    :ivar MAP_ADD_VALID_CELLS: A list of coordinates representing the valid cells that
        should be added to the map grid explicitly during the map generation process.
    :type MAP_ADD_VALID_CELLS: List[Tuple[int, int]]
    :ivar MAP_REMOVE_VALID_CELLS: A list of coordinates representing the cells that
        should be removed from the map grid explicitly during the map generation process.
    :type MAP_REMOVE_VALID_CELLS: List[Tuple[int, int]]
    :ivar MAP_START_CELL: The starting cell's (row, column) coordinate for the A* algorithm
        or other relevant processes.
    :type MAP_START_CELL: Tuple[int, int]
    :ivar MAP_END_CELL: The end cell's (row, column) coordinate for the A* algorithm
        or other relevant processes.
    :type MAP_END_CELL: Tuple[int, int]
    """
    # SCALE_FACTOR: float = 0.9165
    SCALE_FACTOR: float = 1.1122 #pre-calculated for 1080x1920 screen

    MAP_GRID_ROWS: int = 75
    MAP_GRID_COLS: int = 75

    MANUALLY_MODIFY_GRID: bool = True

    MAP_ADD_VALID_CELLS: List[Tuple[int, int]] = field(default_factory=lambda: [
        (16, 45), (16, 46), (16, 47), (16, 48), (16, 49), (16, 50), (16, 51),
        (16, 52), (16, 53), (16, 54), (17, 44), (17, 45), (17, 46), (17, 47), (17, 48),
        (17, 49), (17, 50), (17, 51), (17, 52), (17, 53), (17, 54), (17, 55), (14, 53),
        (15, 52), (15, 53), (18, 44), (18, 45), (18, 46), (18, 55), (18, 56), (19, 44),
        (19, 45), (20, 44), (20, 45), (21, 44), (21, 45), (22, 44), (22, 45), (23, 44),
        (23, 45), (24, 44), (24, 45), (25, 44), (25, 45), (26, 44), (26, 45), (27, 44),
        (27, 45), (28, 44), (28, 45)
    ])
    MAP_REMOVE_VALID_CELLS: List[Tuple[int, int]] = field(default_factory=lambda: [
        (59, 18), (59, 19), (60, 18), (60, 19), (61, 18), (61, 19), (62, 18), (62, 19),
        (63, 18), (63, 19), (64, 18), (64, 19), (65, 18), (65, 19), (66, 15), (66, 16),
        (66, 17), (66, 18), (66, 19), (67, 15), (67, 16), (67, 17), (67, 18)
    ])
    # A* start and end points
    MAP_START_CELL: Tuple[int, int] = (28,45) #(17, 45)
    MAP_END_CELL: Tuple[int, int]   = (69, 14)

@dataclass
class TopFloorMapConfig:
    """
    Configuration for the top floor map.

    This class defines configuration parameters for creating and managing a grid map
    for the top floor. It allows customization of grid dimensions, scalability, and
    manual modifications such as adding or removing specific cells. This class can
    be used in scenarios where grid details are vital for navigation, pathfinding,
    or analysis.

    :ivar SCALE_FACTOR: Scale factor for the map grid.
    :type SCALE_FACTOR: float
    :ivar MAP_GRID_ROWS: Number of rows in the map grid.
    :type MAP_GRID_ROWS: int
    :ivar MAP_GRID_COLS: Number of columns in the map grid.
    :type MAP_GRID_COLS: int
    :ivar MANUALLY_MODIFY_GRID: Flag indicating if manual modifications are applied to the grid.
    :type MANUALLY_MODIFY_GRID: bool
    :ivar MAP_ADD_VALID_CELLS: List of cell coordinates to be manually added to the grid.
    :type MAP_ADD_VALID_CELLS: List[Tuple[int, int]]
    :ivar MAP_REMOVE_VALID_CELLS: List of cell coordinates to be manually removed from the grid.
    :type MAP_REMOVE_VALID_CELLS: List[Tuple[int, int]]
    :ivar MAP_START_CELL: Coordinates of the starting cell in the grid.
    :type MAP_START_CELL: Tuple[int, int]
    :ivar MAP_END_CELL: Coordinates of the ending cell in the grid.
    :type MAP_END_CELL: Tuple[int, int]
    """
    #SCALE_FACTOR: float = 1.2451
    SCALE_FACTOR: float = 1.5102

    MAP_GRID_ROWS: int = 40
    MAP_GRID_COLS: int = 40

    MANUALLY_MODIFY_GRID: bool = True

    MAP_ADD_VALID_CELLS: List[Tuple[int, int]] = field(default_factory=lambda: [

    ])
    MAP_REMOVE_VALID_CELLS: List[Tuple[int, int]] = field(default_factory=lambda: [
        (21, 22), (21, 23), (22, 22), (23, 22), (24, 22), (24, 23), (24, 24), (24, 25)
    ])

    MAP_START_CELL: Tuple[int, int] = (28, 20)
    MAP_END_CELL: Tuple[int, int]   = (20, 24)