# a_star_pathfinding.py
import heapq
import math

def heuristic(a, b):
    """
    Euclidean distance heuristic for 8-directional movement.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])

def a_star(grid, start, end):
    """
    Perform A* pathfinding on a 2D grid with 8-directional movement.

    Args:
        grid (list[list[int]]): 1 = walkable, 0 = blocked
        start (tuple[int, int]): (row, col) start position
        end (tuple[int, int]): (row, col) goal position

    Returns:
        path (list[tuple[int, int]]): List of coordinates from start to end
        cost_grid (list[list[float]]): Cost to reach each cell from start
    """
    rows, cols = len(grid), len(grid[0])
    cost_grid = [[math.inf for _ in range(cols)] for _ in range(rows)]
    came_from = {}

    # Min-heap priority queue: (f_score, g_score, (row, col))
    open_set = []
    heapq.heappush(open_set, (heuristic(start, end), 0, start))
    cost_grid[start[0]][start[1]] = 0

    # 8 possible moves: N, S, E, W, NE, NW, SE, SW
    directions = [
        (-1, 0),  (1, 0),  (0, -1), (0, 1),   # straight moves
        (-1, -1), (-1, 1), (1, -1), (1, 1)    # diagonals
    ]

    while open_set:
        _, g, current = heapq.heappop(open_set)

        if current == end:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, cost_grid

        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                # Diagonal moves cost sqrt(2), straight moves cost 1
                step_cost = math.hypot(dr, dc)
                new_cost = g + step_cost
                if new_cost < cost_grid[nr][nc]:
                    cost_grid[nr][nc] = new_cost
                    priority = new_cost + heuristic((nr, nc), end)
                    heapq.heappush(open_set, (priority, new_cost, (nr, nc)))
                    came_from[(nr, nc)] = current

    # No path found
    return [], cost_grid

def print_grid(grid, path):
    for i in range(len(grid)):
        row = ""
        for j in range(len(grid[0])):
            if (i, j) in path:
                row += "🟩 "
            elif grid[i][j] == 1:
                row += "⬛ "
            else:
                row += "⬜ "
        print(row)

if __name__ == "__main__":
    # Example usage
    example_grid = [
        [0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    start_pos = (0, 0)
    end_pos = (4, 4)

    path, costs = a_star(example_grid, start_pos, end_pos)

    print("Path:", path)
    print("Cost Grid:")
    for row in costs:
        print(["{:.2f}".format(c) if c != math.inf else "inf" for c in row])

    if path:
        print("✅ Path found:")
        print(path)
        print_grid(example_grid, path)
    else:
        print("❌ No path found.")
