"""
Image processing utilities and shared configuration.

This module centralizes low-level helpers (validation, color space conversion,
CUDA checks) and shared constants used across higher-level image processing
routines (e.g., template matching, grid creation).
"""


import cv2
import numpy as np
import time


def gpu_template_match(image, template, method = cv2.TM_CCOEFF_NORMED):
    """
    Performs template matching using GPU acceleration. It utilizes OpenCV's GPU modules
    to upload the input image and template to the GPU, matches them using a specified method,
    and then downloads the result back to the CPU.

    :param image: Input image, which will be matched against the template.
                  Expected to be a 2D or 3D NumPy array depending on the channel-type image.
    :param template: Input template image used for matching.
                     Expected to be a 2D or 3D NumPy array similar to the input image.
    :param method: Template matching method, specified as one of the predefined
                   methods in OpenCV (e.g., cv2.TM_CCOEFF_NORMED, cv2.TM_SQDIFF).
                   Default is cv2.TM_CCOEFF_NORMED.
    :return: The matching result as a NumPy array. The result contains a similarity
             score matrix where each location indicates the matching performance
             between the template and the input image at that position.
    """
    # --- GPU version ---

    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(image)
    gpu_tpl = cv2.cuda_GpuMat()
    gpu_tpl.upload(template)


    # Create the GPU template matcher
    matcher = cv2.cuda.createTemplateMatching(gpu_img.type(),method)

    # Perform matching on GPU
    gpu_result = matcher.match(gpu_img, gpu_tpl)


    result = gpu_result.download()

    return result



def get_cells_within_minimap(image_map, image_minimap, max_loc, grid_cols, grid_rows):
    """
    Finds and marks the grid cells within the 75x75 grid of the main `image_map` that are
    entirely encapsulated by the given `image_minimap`. Also calculates the offsets between
    the top-left corner of the first grid cell and the provided location.

    :param image_map: A 2D array representing the primary image divided into a 75x75 grid.
    :param image_minimap: A portion of the `image_map` that needs to be located and mapped to grid cells.
    :param max_loc: Tuple of two integers representing the top-left corner location of `image_minimap`
                    in the `image_map`.
    :param grid_cols: Total number of columns in the grid.
    :param grid_rows: Total number of rows in the grid.
    :return: A tuple containing:
               - `grid_within_minimap` (2D numpy array): A binary mask for the grid cells that are completely
                 within the `image_minimap`, where `1` indicates the cell is within the `image_minimap`.
               - `shift_x` (int): Horizontal offset from the top-left corner of the first cell to `max_loc`.
               - `shift_y` (int): Vertical offset from the top-left corner of the first cell to `max_loc`.
    """
    h, w = image_map.shape[:2]

    hmm, wmm = image_minimap.shape[:2]


    x_edges = np.linspace(0, w, grid_cols + 1, dtype=int)
    y_edges = np.linspace(0, h, grid_rows + 1, dtype=int)

    # cell that contains minimap upper left corner
    col = np.searchsorted(x_edges, max_loc[0], side="right") - 1
    row = np.searchsorted(y_edges, max_loc[1], side="right") - 1

    # first cell fully in minimap
    first_col = col + 1
    first_row = row + 1

    # distance from top left corner of first cell to max_loc
    shift_x = x_edges[first_col] - max_loc[0]
    shift_y = y_edges[first_row] - max_loc[1]

    # cell that contains bottom right corner
    col = np.searchsorted(x_edges, max_loc[0] + wmm, side="right")
    row = np.searchsorted(y_edges, max_loc[1] + hmm, side="right")

    # last cell fully within minimap
    last_col = col - 1
    last_row = row - 1

    # Create a grid of zeros
    grid_within_minimap = np.zeros((grid_rows, grid_cols), dtype=int)

    # Fill the rectangular region with 1s
    grid_within_minimap[first_row:last_row + 1, first_col:last_col + 1] = 1

    return grid_within_minimap, shift_x, shift_y



def gray_filter(image):
    """
    Apply a gray pixel filter to the given image. This function converts the input
    image to HSV color space to facilitate better color filtering. It identifies
    gray color ranges (low saturation and medium brightness) and creates a mask
    to isolate gray pixels. Finally, the function applies the mask to the input
    image to produce a filtered result containing only the gray pixels.

    :param image: The input image to which the gray filter will be applied.
                  It is assumed to be in the BGR color space.
    :type image: numpy.ndarray
    :return: A new image containing only the gray pixels from the input image,
             with the rest of the pixels masked out.
    :rtype: numpy.ndarray
    """
    # --------------------
    #GRAY PIXEL FILTER
    # --------------------

    # Convert to HSV for better color filtering
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define gray range (low saturation, medium brightness)
    lower_gray = np.array([0, 0, 50])  # H=any, S=0-50, V=50-200
    upper_gray = np.array([180, 50, 200])

    # Create mask
    mask_gray = cv2.inRange(hsv_image, lower_gray, upper_gray)

    # Apply mask to image
    gray_filtered_image = cv2.bitwise_and(image, image, mask=mask_gray)

    return gray_filtered_image

def create_grid(image, grid_rows, grid_cols, valid_point_gray_ratio = .65, gray_thresh_low=50, gray_thresh_high=200):
    """
    Create a binary grid representation of an image based on the ratio of gray pixels
    within regions defined by grid rows and columns.

    This function partitions the input image into a grid defined by the specified number
    of rows and columns. For each cell in the grid, it calculates the ratio of pixels
    falling within a specified gray intensity range. If this ratio exceeds a threshold,
    the cell is marked as valid (1); otherwise, it is marked as invalid (0).

    :param image: A 2D array or a grayscale image representing pixel intensities.
    :param grid_rows: Number of rows in the grid partition.
    :param grid_cols: Number of columns in the grid partition.
    :param valid_point_gray_ratio: Threshold value for determining whether a cell in the grid is valid,
        based on the ratio of gray pixels within the cell.
    :param gray_thresh_low: Lower bound of gray pixel intensity values.
    :param gray_thresh_high: Upper bound of gray pixel intensity values.
    :return: A 2D list representing the binary grid. Each element in the grid is 1 if the
        corresponding cell meets the valid_point_gray_ratio threshold based on gray intensity,
        otherwise it is 0.
    """
    h, w = image.shape[:2]

    # Boolean mask for gray pixels
    gray_mask = (image >= gray_thresh_low) & (image <= gray_thresh_high)

    # Precompute exact pixel boundaries
    x_edges = np.linspace(0, w, grid_cols + 1, dtype=int)
    y_edges = np.linspace(0, h, grid_rows + 1, dtype=int)

    grid = [[0 for _ in range(grid_cols)] for _ in range(grid_rows)]

    for r in range(grid_rows):
        for c in range(grid_cols):
            y1, y2 = y_edges[r], y_edges[r + 1]
            x1, x2 = x_edges[c], x_edges[c + 1]

            cell_mask = gray_mask[y1:y2, x1:x2]
            gray_ratio = np.sum(cell_mask) / cell_mask.size

            if gray_ratio > valid_point_gray_ratio:
                grid[r][c] = 1

    return grid


def modify_grid(grid, add_valid_cells, remove_valid_cells):
    """
    Modifies the given grid by adding and removing specified valid cells.

    This function updates a 2D grid based on the provided lists of valid cells
    to add and remove. Cells marked for addition are updated to 1, while cells
    marked for removal are updated to 0.

    :param grid: A 2D array representing the grid to be updated.
    :param add_valid_cells: A list of tuples containing the row and column indices
        of cells to be set to 1 in the grid.
    :param remove_valid_cells: A list of tuples containing the row and column
        indices of cells to be set to 0 in the grid.
    :return: None
    """
    # Add valid cells
    for r, c in add_valid_cells:
        grid[r][c] = 1

    for r, c in remove_valid_cells:
        grid[r][c] = 0


def get_path_bounds(path, costs, grid_within_minimap):
    """
    Determines the earliest and latest cells in a path based on the cost values and
    grid boundaries. The algorithm iterates through the path to find the cells
    within the minimap bounds, selecting the cells with the smallest and largest
    costs respectively.

    :param path: A list of tuples representing cell coordinates in the path.
    :param costs: A 2D list or array where each value represents the cost at a
        specific cell.
    :param grid_within_minimap: A 2D list or array of the same size as `costs`,
        where each value indicates whether the cell is within the minimap bounds
        (1 for inside, 0 for outside).
    :return: A tuple containing the coordinates (row, column) of the earliest
        cell (with the smallest cost) and the latest cell (with the largest cost)
        along the path that are within minimap bounds.
    """
    latest_cell = path[0]

    earliest_cell = path[-1]

    for r,c in path:

        #within minimap bounds
        if grid_within_minimap[r][c] == 1: # within minimap bounds
            if costs[r][c] < costs[earliest_cell[0]][earliest_cell[1]]:
                earliest_cell = (r,c)
            if costs[r][c] > costs[latest_cell[0]][latest_cell[1]]:
                latest_cell = (r,c)

    return earliest_cell, latest_cell


def create_valid_cells_grid(image, config):
    """
    Creates a grid of valid cells based on the given image and configuration. The function
    performs initial preprocessing of the image, such as applying a grayscale filter,
    creates a grid based on the dimensions specified in the configuration, and optionally
    allows manual adjustments to the grid by adding or removing specific valid cells. This
    is useful for tasks requiring tailored grid-based representations of the image.

    :param image: The input image to be processed.
    :param config: Configuration object containing parameters for grid creation and manual
                   modifications.
    :return: A grid of valid cells derived from the input image.
    :rtype: list
    """
    gray_filtered_image = gray_filter(image=image)

    # Get valid points in grid, 50x50?
    grid = create_grid(image=gray_filtered_image,
                       grid_rows=config.MAP_GRID_ROWS,
                       grid_cols=config.MAP_GRID_COLS)

    # Manually label start & end, add any extra valid points
    if config.MANUALLY_MODIFY_GRID:
        modify_grid(grid=grid,
                    add_valid_cells=config.MAP_ADD_VALID_CELLS,
                    remove_valid_cells=config.MAP_REMOVE_VALID_CELLS)

    return grid


def fast_classify(image, tolerance=10, scale=0.1, cutoff=15):
    """
    Classifies whether an image passes the specified threshold criteria based on the percentage of pixels classified as "black".
    A "black" pixel is identified as having all its RGB values below or equal to the tolerance threshold.
    The evaluation is done relative to a cutoff value.

    :param image: The input image represented as a NumPy array.
    :param tolerance: The threshold for classifying pixels as "black". Defaults to 10.
    :param scale: Currently unused; placeholder for future scaling implementation. Defaults to 0.1.
    :param cutoff: The percentage of black pixels above which the classification returns False. Defaults to 15.
    :return: A boolean indicating whether the image passes the classification criteria (True) or not (False).
    """
    black_mask = np.all(image <= tolerance, axis=-1)
    black_pct = np.count_nonzero(black_mask) / black_mask.size * 100
    return False if black_pct > cutoff else True




def timed_model(model, frame, device=0, half=True, verbose=False):
    """
    Executes the provided model's prediction on a given frame while measuring the
    execution time. The method utilizes additional settings such as device, half
    precision mode, and verbosity, while also applying confidence and IoU thresholds
    to filter predictions.

    :param model: The model instance used to perform predictions on the input frame.
    :type model: Any
    :param frame: The input data (e.g., image or data frame) to be processed by the model.
    :type frame: Any
    :param device: The device ID on which the computation should be executed.
                   Defaults to 0 (typically the CPU or first available GPU).
    :type device: int, optional
    :param half: Whether to use half-precision mode for faster inference.
                 Defaults to True.
    :type half: bool, optional
    :param verbose: If True, enables verbose logging while performing predictions.
                    Defaults to False.
    :type verbose: bool, optional
    :return: A tuple containing the model prediction result and the elapsed time
             for the inference process.
    :rtype: tuple
    """
    start = time.perf_counter()
    result = model.predict(frame,
                           device=device,
                           half=half,
                           verbose=verbose,
                           conf=0.5,  # filter low-confidence boxes
                           iou=0.3,  # aggressive NMS
                           agnostic_nms=True  # merge across classes
                           )
    end = time.perf_counter()

    return result, end-start


def get_objects(model_results):
    """
    Extracts detected objects from model results, filtering out overlapping objects
    based on Intersection over Union (IoU) threshold and avoiding label duplication.
    Each object is represented with label, confidence score, and bounding box.

    :param model_results: A list of detection results, where each result contains
        detected boxes with properties like bounding box coordinates, class IDs,
        confidence scores, and associated labels.
    :type model_results: list
    :return: A list of unique objects detected, each represented as a dictionary
        containing the label, confidence score, and bounding box coordinates.
    :rtype: list[dict]
    """
    objects = []
    for result in model_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = result.names[cls_id]

            new_obj = {
                "label": label,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2)
            }

            if any(o["label"] == label and iou(o["bbox"], new_obj["bbox"]) > 0.3 for o in objects):
                continue

            objects.append(new_obj)

    return objects


def compute_template_match(map_img,
                           minimap_img,
                           use_gpu              = True,
                           prev_max_loc         = None,
                           prev_max_val         = None,
                           on_same_map          = False,
                           scale_factor         = 0.25,
                           search_region_factor = 2,
                           use_color            = True,
                           method               = cv2.TM_CCOEFF_NORMED
                           ):
    """
    Compute the best match location of a template image within a larger image using
    multi-scale and optimized template matching.

    Prioritizes a coarse-to-fine searching approach to improve performance,
    and optionally leverages GPU acceleration. If a previous match with high
    correlation exists on the same map, a tighter region is searched around
    the previous match to improve efficiency further.

    :param map_img: The larger reference image in which the template should be located.
    :type map_img: numpy.ndarray
    :param minimap_img: The template image to locate within the reference image.
    :type minimap_img: numpy.ndarray
    :param use_gpu: Indicates if GPU acceleration should be utilized during
        template matching when supported.
    :type use_gpu: bool
    :param prev_max_loc: The location of the previous best match in the reference
        image, used to optimize the search region if a prior match exists.
    :type prev_max_loc: tuple[int, int] | None
    :param prev_max_val: The correlation value of the previous best match, used when
        deciding the similarity threshold for skipping computation on the same map.
    :type prev_max_val: float | None
    :param on_same_map: Indicates if the current image is the same as the previous
        map image, which allows optimization for re-using prior match information.
    :type on_same_map: bool
    :param scale_factor: The downscale factor to coarsely locate the template before refining
        the match, allowing an improved search efficiency. A value in (0, 1].
    :type scale_factor: float
    :param search_region_factor: The multiplier factor for determining the size of the
        search region area around the previously matched area or coarse location.
    :type search_region_factor: float
    :param use_color: Indicates whether to retain color channels during the matching process
        or convert them to grayscale for computation.
    :type use_color: bool
    :param method: The similarity metric used for template matching. Defaults to cv2.TM_CCOEFF_NORMED.
    :type method: int
    :return: Returns the maximum correlation value and its corresponding location
        (x, y) in the reference image. If a valid region or match cannot be established,
        returns (None, None).
    :rtype: tuple[float | None, tuple[int, int] | None]
    """
    if not use_color:
        # 1. Convert to grayscale
        map_img     = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
        minimap_img = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2GRAY)


    # 1. Use region where previous minimap was found, if the correlation is similar enough, return
    if prev_max_loc is not None and on_same_map:
        minimap_height, minimap_width = minimap_img.shape[:2]

        center_x = prev_max_loc[0] + (minimap_width  // 2)
        center_y = prev_max_loc[1] + (minimap_height // 2)

        # Define a tighter ROI around coarse match
        # Estimate region bounds
        search_x1 = int(center_x - (minimap_width  // 2) * search_region_factor)
        search_y1 = int(center_y - (minimap_height // 2) * search_region_factor)

        search_x2 = int(center_x + (minimap_width  // 2) * search_region_factor)
        search_y2 = int(center_y + (minimap_height // 2) * search_region_factor)

        # Clamp to map dimensions
        map_height, map_width = map_img.shape[:2]

        search_x1 = max(0, search_x1)
        search_y1 = max(0, search_y1)
        search_x2 = min(map_width, search_x2)
        search_y2 = min(map_height, search_y2)

        # Make sure the region is valid
        if search_x2 <= search_x1 or search_y2 <= search_y1:
            print("❌ Invalid ROI: Skipping template match------------------")
            return None, None

        search_region = map_img[search_y1:search_y2, search_x1:search_x2]

        if cv2.cuda.getCudaEnabledDeviceCount() > 0 and use_gpu:
            result = gpu_template_match(search_region, minimap_img, method=method)
        else:
            result = cv2.matchTemplate(search_region, minimap_img, method=method)

        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        max_loc = (max_loc[0] + search_x1, max_loc[1] + search_y1)


        if (max_val-prev_max_val)/prev_max_val < 0.05:
            return max_val, max_loc


    # 2. Coarse to fine search
    if scale_factor < 1.0:
        small_map = cv2.resize(map_img, (0, 0), fx=scale_factor, fy=scale_factor)
        small_template = cv2.resize(minimap_img, (0, 0), fx=scale_factor, fy=scale_factor)

        # Check if GPU is available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0 and use_gpu:
            res_small = gpu_template_match(small_map, small_template, method = method)
        else:
            res_small = cv2.matchTemplate(small_map, small_template, method = method)

        _, _, _, max_loc_small = cv2.minMaxLoc(res_small)
        # Map coarse location back to full-res ROI
        start_x = int(max_loc_small[0] / scale_factor)
        start_y = int(max_loc_small[1] / scale_factor)

        # Define a tighter ROI around coarse match
        minimap_height, minimap_width = minimap_img.shape[:2]

        center_x = start_x + (minimap_width // 2)
        center_y = start_y + (minimap_height // 2)


        # Estimate region bounds
        search_x1 = int(center_x - (minimap_width  // 2) * search_region_factor)
        search_y1 = int(center_y - (minimap_height // 2) * search_region_factor)

        search_x2 = int(center_x + (minimap_width  // 2) * search_region_factor)
        search_y2 = int(center_y + (minimap_height // 2) * search_region_factor)

        # Clamp to map dimensions
        map_height, map_width = map_img.shape[:2]

        search_x1 = max(0, search_x1)
        search_y1 = max(0, search_y1)
        search_x2 = min(map_width, search_x2)
        search_y2 = min(map_height, search_y2)

        # Make sure the region is valid
        if search_x2 <= search_x1 or search_y2 <= search_y1:
            print("❌ Invalid ROI: Skipping template match coarse to fine------------------")
            return None, None

        search_region = map_img[search_y1:search_y2, search_x1:search_x2]

        if cv2.cuda.getCudaEnabledDeviceCount() > 0 and use_gpu:
            result = gpu_template_match(search_region, minimap_img, method = method)
        else:
            result = cv2.matchTemplate(search_region, minimap_img, method = method)

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Adjust location to full image coords
        max_loc = (max_loc[0] + search_x1, max_loc[1] + search_y1)

        return max_val, max_loc

    #3. Nominal full size template matching
    if cv2.cuda.getCudaEnabledDeviceCount() > 0 and use_gpu:
        result = gpu_template_match(map_img, minimap_img, method=method)
    else:
        result = cv2.matchTemplate(map_img, minimap_img, method=method)

    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_val, max_loc


def significant_movement(image, previous_image):
    """
    Determines if there is a significant movement between two images by comparing their
    dimensions and average color differences.

    This function first checks if the dimensions of the two input images are identical.
    If they differ, significant movement is assumed, and the function returns `True`.
    If the dimensions match, the function calculates the per-channel absolute color
    difference between the two images and computes the Euclidean difference per pixel.
    The average of these differences is then calculated, and a threshold is applied to
    evaluate if the movement is significant.

    :param image: The first input image to compare. It is expected to be a NumPy array
        representing an image, including its height, width, and color channels.
    :param previous_image: The second input image to compare. It is expected to be a
        NumPy array representing an image, with its height, width, and color channels.
    :return: A boolean indicating whether significant movement has been detected
        between the two images. Returns `True` if the movement is significant and
        `False` otherwise.
    """
    h1, w1 = image.shape[:2]
    h2, w2 = previous_image.shape[:2]

    if (h1, w1) != (h2, w2):
        return True

    # Compute per-channel absolute difference
    diff = cv2.absdiff(image, previous_image).astype(np.float32)

    # Euclidean color difference per pixel
    diff_magnitude = np.linalg.norm(diff, axis=2)  # shape: (H, W)

    # Average difference across all pixels
    avg_diff = np.mean(diff_magnitude)

    return avg_diff > 10


def get_pixel_location(cell, grid_within_minimap, shift_x, shift_y, map_height, map_width):
    """
    Calculates the pixel location (x, y) of a cell within a grid on a minimap. The function
    uses the dimensions of the grid, the grid's position within the minimap, and the
    offsets (shifts) to determine the exact pixel location.

    :param cell: A tuple representing the cell's row and column within the grid.
    :param grid_within_minimap: A 2D numpy array representing the grid layout within
        the minimap.
    :param shift_x: Horizontal shift offset in pixels.
    :param shift_y: Vertical shift offset in pixels.
    :param map_height: The total height of the map in pixels.
    :param map_width: The total width of the map in pixels.
    :return: A tuple (x_center, y_center) representing the center pixel location in
        the minimap for the given cell.
    """
    row, col = cell

    # Get the pixel sizes of each cell in map
    grid_rows, grid_cols = grid_within_minimap.shape
    x_edges = np.linspace(0, map_width, grid_cols + 1, dtype=int)
    y_edges = np.linspace(0, map_height, grid_rows + 1, dtype=int)

    # get the shift from the map 0,0 to minimap location
    positions = np.argwhere(grid_within_minimap != 0)
    first_row, first_col = positions[0]  # first match in row-major order
    shift_x -= x_edges[first_col]
    shift_y -= y_edges[first_row]


    x0, x1 = x_edges[col], x_edges[col + 1]
    y0, y1 = y_edges[row], y_edges[row + 1]

    width = x1 - x0
    height = y1 - y0

    x_center = x0 + shift_x + width / 2
    y_center = y0 + shift_y + height / 2

    return x_center, y_center



def iou(box1, box2):
    """
    This function computes the Intersection over Union (IoU) between two rectangular
    bounding boxes. The IoU is a measure of the overlap between two boxes, defined as
    the ratio of their intersection area to their union area. The function accepts
    the coordinates of the top-left and bottom-right corners of the boxes, calculates
    the intersection and union areas, and returns the IoU value. If the union area
    is zero, the function returns 0.0 to avoid division by zero.

    :param box1: A tuple (x1, y1, x2, y2) representing the coordinates of the first
        bounding box. (x1, y1) specifies the top-left corner, and (x2, y2) specifies
        the bottom-right corner.
    :param box2: A tuple (X1, Y1, X2, Y2) representing the coordinates of the second
        bounding box. (X1, Y1) specifies the top-left corner, and (X2, Y2) specifies
        the bottom-right corner.
    :return: A float value representing the IoU between the two bounding boxes. If
        there is no overlap or if the union area is zero, the function returns 0.0.
    :rtype: float
    """
    x1, y1, x2, y2 = box1
    X1, Y1, X2, Y2 = box2

    # Determine the coordinates of the intersection rectangle
    inter_x1 = max(x1, X1)
    inter_y1 = max(y1, Y1)
    inter_x2 = min(x2, X2)
    inter_y2 = min(y2, Y2)

    # Compute intersection area
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Compute each box's area
    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, X2 - X1) * max(0, Y2 - Y1)

    # Compute union area
    union_area = area1 + area2 - inter_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    return inter_area / union_area


def append_objects(objects, position, label, confidence = 1.0, width = 2.0, height = 2.0):
    """
    Appends an object with a specified label, confidence, and bounding box
    dimensions to the provided list of objects. The bounding box is calculated
    centered around the given position with the specified width and height.

    :param objects: List to which the object will be appended.
    :type objects: list.
    :param position: Center position of the bounding box.
    :type position: tuple[int, int].
    :param label: Label identifying the object.
    :type label: str.
    :param confidence: Confidence level of the object, defaults to 1.0.
    :type confidence: float, optional.
    :param width: Width of the bounding box, defaults to 2.0.
    :type width: float, optional.
    :param height: Height of the bounding box, defaults to 2.0.
    :type height: float, optional.
    :return: None.
    """
    x = int(width / 2)
    y = int(height / 2)
    objects.append({
        "label": label,
        "confidence": confidence,
        "bbox": (position[0] - x, position[1] - y,
                 position[0] + x, position[1] + y)
    })
