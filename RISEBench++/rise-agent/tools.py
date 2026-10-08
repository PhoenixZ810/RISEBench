from __future__ import annotations

import ast
import heapq
import json
import math
import operator
import re
from collections import Counter, deque
from datetime import datetime, timedelta
from typing import Any, Callable

import requests
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFont


class ToolError(ValueError):
    pass


class TavilySearch:
    def __init__(self, api_key: str | None, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        if not self.api_key:
            raise ToolError("TAVILY_API_KEY is not configured")
        query = query.strip()
        if not query:
            raise ToolError("Search query cannot be empty")
        result_limit = max(1, min(int(max_results), 8))
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "query": query[:1000],
                "search_depth": "basic",
                "max_results": result_limit,
                "include_answer": "basic",
                "include_raw_content": False,
                "safe_search": True,
            },
            timeout=self.timeout,
            allow_redirects=False,
        )
        response.raise_for_status()
        payload = response.json()
        return {
            "query": payload.get("query", query),
            "answer": payload.get("answer", ""),
            "results": [
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                }
                for result in payload.get("results", [])[:result_limit]
            ],
        }


_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_ALLOWED_FUNCS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "radians": math.radians,
    "degrees": math.degrees,
    "gcd": math.gcd,
}


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ToolError(f"{name} must be finite")
    return number


def _safe_eval(expression: str) -> int | float:
    if len(expression) > 500:
        raise ToolError("Arithmetic expression is too long")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ToolError("Arithmetic expression is too complex")

    def visit(node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 20:
                raise ToolError("Exponent is too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](visit(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = _ALLOWED_FUNCS.get(node.func.id)
            if function and not node.keywords and len(node.args) <= 4:
                return function(*(visit(arg) for arg in node.args))
        raise ToolError(f"Unsupported arithmetic syntax: {ast.dump(node, include_attributes=False)}")

    result = visit(tree.body)
    if not math.isfinite(float(result)):
        raise ToolError("Arithmetic result is not finite")
    return result


def _grid_shortest_path(spec: dict[str, Any]) -> dict[str, Any]:
    grid = spec.get("grid")
    if not isinstance(grid, list) or not grid:
        raise ToolError("grid must be a non-empty list")
    if len(grid) > 100:
        raise ToolError("grid has too many rows")
    try:
        rows = [list(row) for row in grid]
    except TypeError as exc:
        raise ToolError("Each grid row must be iterable") from exc
    width = len(rows[0])
    if width == 0 or width > 100 or any(len(row) != width for row in rows):
        raise ToolError("grid must be rectangular and at most 100x100")

    def coordinate(name: str) -> tuple[int, int]:
        value = spec.get(name)
        if not isinstance(value, list) or len(value) != 2 or not all(isinstance(x, int) for x in value):
            raise ToolError(f"{name} must be [row, column] integers")
        point = (value[0], value[1])
        if not (0 <= point[0] < len(rows) and 0 <= point[1] < width):
            raise ToolError(f"{name} is outside the grid")
        return point

    start, goal = coordinate("start"), coordinate("goal")
    blocked_values = spec.get("blocked", ["#", 1])
    if not isinstance(blocked_values, list):
        raise ToolError("blocked must be a list")
    if rows[start[0]][start[1]] in blocked_values or rows[goal[0]][goal[1]] in blocked_values:
        raise ToolError("start and goal must be walkable")
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if spec.get("diagonal", False):
        directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for dr, dc in directions:
            nxt = (current[0] + dr, current[1] + dc)
            if not (0 <= nxt[0] < len(rows) and 0 <= nxt[1] < width):
                continue
            if rows[nxt[0]][nxt[1]] in blocked_values or nxt in parent:
                continue
            parent[nxt] = current
            queue.append(nxt)
    if goal not in parent:
        return {"reachable": False, "distance": None, "path": []}
    path: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = goal
    while cursor is not None:
        path.append(cursor)
        cursor = parent[cursor]
    path.reverse()
    return {"reachable": True, "distance": len(path) - 1, "path": [list(point) for point in path]}


def _graph_shortest_path(spec: dict[str, Any]) -> dict[str, Any]:
    edges = spec.get("edges", [])
    if not isinstance(edges, list) or len(edges) > 10_000:
        raise ToolError("edges must be a list with at most 10000 entries")
    start, goal = str(spec.get("start", "")), str(spec.get("goal", ""))
    if not start or not goal:
        raise ToolError("start and goal node names are required")
    weighted = any(isinstance(edge, list) and len(edge) == 3 for edge in edges)
    graph: dict[str, list[tuple[str, float]]] = {}
    for edge in edges:
        if not isinstance(edge, list) or len(edge) not in (2, 3):
            raise ToolError("Each edge must be [u, v] or [u, v, weight]")
        u, v = str(edge[0]), str(edge[1])
        weight = _finite_number(edge[2], "edge weight") if len(edge) == 3 else 1.0
        if weight < 0:
            raise ToolError("Negative edge weights are unsupported")
        graph.setdefault(u, []).append((v, weight))
        if not spec.get("directed", False):
            graph.setdefault(v, []).append((u, weight))

    heap = [(0.0, start, [start])]
    best = {start: 0.0}
    while heap:
        distance, node, path = heapq.heappop(heap)
        if node == goal:
            return {"reachable": True, "distance": distance if weighted else int(distance), "path": path}
        if distance != best.get(node):
            continue
        for neighbor, weight in graph.get(node, []):
            candidate = distance + weight
            if candidate < best.get(neighbor, float("inf")):
                best[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor, path + [neighbor]))
    return {"reachable": False, "distance": None, "path": []}


def _solve_sudoku(spec: dict[str, Any]) -> dict[str, Any]:
    raw_board = spec.get("board", [])
    if not isinstance(raw_board, list) or not raw_board:
        raise ToolError("Sudoku board must be a non-empty square matrix")
    if any(not isinstance(row, list) for row in raw_board):
        raise ToolError("Each Sudoku row must be a JSON array")
    if any(type(value) is not int for row in raw_board for value in row):
        raise ToolError("Sudoku cells must be integers, using 0 for blanks")
    board = [row.copy() for row in raw_board]
    size = len(board)
    if size > 16 or any(len(row) != size for row in board):
        raise ToolError("Sudoku board must be square with size at most 16")
    if any(value < 0 or value > size for row in board for value in row):
        raise ToolError(f"Sudoku values must be between 0 and {size}")

    box_rows = spec.get("box_rows")
    box_cols = spec.get("box_cols")
    if (box_rows is None) != (box_cols is None):
        raise ToolError("box_rows and box_cols must be provided together")
    if box_rows is not None and (not isinstance(box_rows, int) or not isinstance(box_cols, int)):
        raise ToolError("box_rows and box_cols must both be integers")
    if box_rows is not None and (box_rows <= 0 or box_cols <= 0 or box_rows * box_cols != size):
        raise ToolError("box_rows * box_cols must equal the board size")

    def units_for(row: int, col: int) -> list[list[int]]:
        units = [board[row], [board[r][col] for r in range(size)]]
        if box_rows is not None and box_cols is not None:
            row0, col0 = box_rows * (row // box_rows), box_cols * (col // box_cols)
            units.append(
                [board[r][c] for r in range(row0, row0 + box_rows) for c in range(col0, col0 + box_cols)]
            )
        return units

    for row in range(size):
        for col in range(size):
            value = board[row][col]
            if value == 0:
                continue
            board[row][col] = 0
            if any(value in unit for unit in units_for(row, col)):
                raise ToolError("Sudoku clues violate row, column, or box constraints")
            board[row][col] = value

    symbols = set(range(1, size + 1))

    def candidates(row: int, col: int) -> set[int]:
        used = {value for unit in units_for(row, col) for value in unit if value}
        return symbols - used

    def solve() -> bool:
        choices = [
            (len(options), row, col, options)
            for row in range(size)
            for col in range(size)
            if board[row][col] == 0
            for options in [candidates(row, col)]
        ]
        if not choices:
            return True
        _, row, col, options = min(choices, key=lambda value: value[0])
        if not options:
            return False
        for value in sorted(options):
            board[row][col] = value
            if solve():
                return True
            board[row][col] = 0
        return False

    solved = solve()
    return {
        "solved": solved,
        "board": board if solved else None,
        "box_rows": box_rows,
        "box_cols": box_cols,
    }


def _point(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ToolError(f"{name} must be [x, y]")
    return _finite_number(value[0], f"{name}.x"), _finite_number(value[1], f"{name}.y")


def _coordinate_geometry(spec: dict[str, Any]) -> dict[str, Any]:
    action = spec.get("action")
    if action in {"distance", "midpoint", "slope"}:
        p1, p2 = _point(spec.get("p1"), "p1"), _point(spec.get("p2"), "p2")
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        if action == "distance":
            return {"distance": math.hypot(dx, dy)}
        if action == "midpoint":
            return {"point": [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]}
        return {"slope": None if dx == 0 else dy / dx, "vertical": dx == 0}
    if action == "rotate":
        point = _point(spec.get("point"), "point")
        center = _point(spec.get("center", [0, 0]), "center")
        angle = math.radians(_finite_number(spec.get("degrees"), "degrees"))
        x, y = point[0] - center[0], point[1] - center[1]
        return {
            "point": [
                x * math.cos(angle) - y * math.sin(angle) + center[0],
                x * math.sin(angle) + y * math.cos(angle) + center[1],
            ]
        }
    raise ToolError(f"Unsupported geometry action: {action}")


_MAX_MATRIX_SIDE = 12


def _read_matrix(spec: dict[str, Any], key: str = "matrix") -> list[list[Any]]:
    matrix = spec.get(key)
    if not isinstance(matrix, list) or not matrix:
        raise ToolError(f"{key} must be a non-empty list of rows")
    if len(matrix) > _MAX_MATRIX_SIDE:
        raise ToolError(f"{key} may have at most {_MAX_MATRIX_SIDE} rows")
    rows: list[list[Any]] = []
    for index, row in enumerate(matrix):
        if not isinstance(row, list):
            raise ToolError(f"{key}[{index}] must be a list")
        rows.append(list(row))
    width = len(rows[0])
    if width == 0 or width > _MAX_MATRIX_SIDE:
        raise ToolError(f"{key} rows must hold 1..{_MAX_MATRIX_SIDE} cells")
    if any(len(row) != width for row in rows):
        raise ToolError(f"{key} must be rectangular")
    return rows


def _cell_numbers(rows: list[list[Any]]) -> list[list[float]]:
    numbers: list[list[float]] = []
    for r, row in enumerate(rows):
        numbers.append([_finite_number(value, f"matrix[{r}][{c}]") for c, value in enumerate(row)])
    return numbers


def _spiral_order(rows: int, cols: int, start: str, directions: list[str]) -> list[tuple[int, int]]:
    steps = {"left": (0, -1), "right": (0, 1), "up": (-1, 0), "down": (1, 0)}
    corners = {
        "top_left": (0, 0),
        "top_right": (0, cols - 1),
        "bottom_left": (rows - 1, 0),
        "bottom_right": (rows - 1, cols - 1),
    }
    if start not in corners:
        raise ToolError(f"start_corner must be one of: {', '.join(sorted(corners))}")
    for name in directions:
        if name not in steps:
            raise ToolError(f"direction_cycle entries must be one of: {', '.join(sorted(steps))}")

    position = corners[start]
    seen = {position}
    order = [position]
    heading = 0
    turns = 0
    while len(order) < rows * cols:
        dr, dc = steps[directions[heading]]
        nxt = (position[0] + dr, position[1] + dc)
        if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and nxt not in seen:
            position = nxt
            seen.add(nxt)
            order.append(nxt)
            turns = 0
            continue
        heading = (heading + 1) % len(directions)
        turns += 1
        if turns >= len(directions):
            raise ToolError("Traversal became stuck before covering every cell")
    return order


def _knight_tour_relabel(values: list[list[float]]) -> list[list[int]]:
    """Assign visit orders 0..n-1 to cells so consecutive orders are a knight's move apart.

    Cells keep their current contents; only which cell holds which visit index changes,
    and the search maximises the number of cells that keep their original index so the
    "minimise relocations" requirement is respected.
    """
    rows, cols = len(values), len(values[0])
    total = rows * cols
    moves = ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1))
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    neighbours = {
        cell: [
            (cell[0] + dr, cell[1] + dc)
            for dr, dc in moves
            if 0 <= cell[0] + dr < rows and 0 <= cell[1] + dc < cols
        ]
        for cell in cells
    }
    # Where each visit order currently sits, so a candidate tour can be scored by how
    # many cells would not have to move at all.
    current: dict[int, tuple[int, int]] = {}
    for r in range(rows):
        for c in range(cols):
            order = values[r][c]
            if float(order).is_integer() and 0 <= int(order) < total:
                current.setdefault(int(order), (r, c))

    best: list[tuple[int, int]] | None = None
    best_kept = -1
    budget = 400_000

    def search(path: list[tuple[int, int]], visited: set[tuple[int, int]], kept: int) -> None:
        nonlocal best, best_kept, budget
        if budget <= 0:
            return
        budget -= 1
        if len(path) == total:
            if kept > best_kept:
                best, best_kept = list(path), kept
            return
        # Warnsdorff ordering keeps the search tractable; cells that would stay put are
        # tried first so a high-scoring tour is found early.
        options = sorted(
            (cell for cell in neighbours[path[-1]] if cell not in visited),
            key=lambda cell: (len([n for n in neighbours[cell] if n not in visited]),
                              current.get(len(path)) != cell),
        )
        for cell in options:
            visited.add(cell)
            path.append(cell)
            search(path, visited, kept + (1 if current.get(len(path) - 1) == cell else 0))
            path.pop()
            visited.discard(cell)
            if best_kept == total:
                return

    starts = sorted(cells, key=lambda cell: current.get(0) != cell)
    for start in starts:
        search([start], {start}, 1 if current.get(0) == start else 0)
        if best is not None:
            break
    if best is None:
        raise ToolError(f"No knight's tour exists on a {rows}x{cols} board")
    placement = [[0] * cols for _ in range(rows)]
    for order, (r, c) in enumerate(best):
        placement[r][c] = order
    return placement


def _matrix_transform(spec: dict[str, Any]) -> dict[str, Any]:
    """Compute a cell permutation for a grid rearrangement.

    Every action returns `moves`, a list of `{"from": [r, c], "to": [r, c]}` entries that
    map straight onto the renderer's `source_cell` / `target_cell` fields, so the model
    never has to derive cell coordinates or the permutation itself.
    """
    rows_data = _read_matrix(spec)
    rows, cols = len(rows_data), len(rows_data[0])
    action = str(spec.get("action", ""))

    # source[r][c] = which original cell ends up at (r, c)
    if action == "rotate":
        degrees = int(_finite_number(spec.get("degrees", 180), "degrees")) % 360
        if degrees not in {0, 90, 180, 270}:
            raise ToolError("degrees must be 0, 90, 180 or 270")
        if degrees in {90, 270} and rows != cols:
            raise ToolError("90/270 degree rotation requires a square matrix")
        if degrees == 0:
            source = [[(r, c) for c in range(cols)] for r in range(rows)]
        elif degrees == 180:
            source = [[(rows - 1 - r, cols - 1 - c) for c in range(cols)] for r in range(rows)]
        elif degrees == 90:  # clockwise
            source = [[(rows - 1 - c, r) for c in range(cols)] for r in range(rows)]
        else:
            source = [[(c, cols - 1 - r) for c in range(cols)] for r in range(rows)]

    elif action == "transpose":
        if rows != cols:
            raise ToolError("transpose requires a square matrix")
        source = [[(c, r) for c in range(cols)] for r in range(rows)]

    elif action in {"flip_horizontal", "flip_vertical"}:
        if action == "flip_horizontal":
            source = [[(r, cols - 1 - c) for c in range(cols)] for r in range(rows)]
        else:
            source = [[(rows - 1 - r, c) for c in range(cols)] for r in range(rows)]

    elif action in {"swap_rows", "swap_columns"}:
        by_sum = spec.get("select") == "extreme_sums"
        numbers = _cell_numbers(rows_data)
        limit = rows if action == "swap_rows" else cols
        if by_sum:
            totals = (
                [sum(numbers[r]) for r in range(rows)]
                if action == "swap_rows"
                else [sum(numbers[r][c] for r in range(rows)) for c in range(cols)]
            )
            # Ties resolve to the lowest index, matching the usual task wording.
            first = max(range(limit), key=lambda i: (totals[i], -i))
            second = min(range(limit), key=lambda i: (totals[i], i))
            sums = totals
        else:
            indices = spec.get("indices")
            if not isinstance(indices, list) or len(indices) != 2:
                raise ToolError("indices must be [i, j], or set select='extreme_sums'")
            if not all(isinstance(i, int) and not isinstance(i, bool) for i in indices):
                raise ToolError("indices must be integers")
            first, second = indices
            sums = None
            if not (0 <= first < limit and 0 <= second < limit):
                raise ToolError(f"indices must lie in 0..{limit - 1}")

        def remap(index: int) -> int:
            if index == first:
                return second
            return first if index == second else index

        if action == "swap_rows":
            source = [[(remap(r), c) for c in range(cols)] for r in range(rows)]
        else:
            source = [[(r, remap(c)) for c in range(cols)] for r in range(rows)]

    elif action == "spiral_rearrange":
        cycle = spec.get("direction_cycle", ["left", "down", "right", "up"])
        if not isinstance(cycle, list) or not cycle:
            raise ToolError("direction_cycle must be a non-empty list")
        order = _spiral_order(
            rows, cols, str(spec.get("start_corner", "top_right")), [str(d) for d in cycle]
        )
        source = [[order[r * cols + c] for c in range(cols)] for r in range(rows)]

    elif action == "knight_tour":
        placement = _knight_tour_relabel(_cell_numbers(rows_data))
        # Cell holding visit order k must end up where the tour puts k.
        origin: dict[int, tuple[int, int]] = {}
        for r in range(rows):
            for c in range(cols):
                value = rows_data[r][c]
                number = _finite_number(value, f"matrix[{r}][{c}]")
                if not float(number).is_integer():
                    raise ToolError("knight_tour requires integer visit orders")
                origin[int(number)] = (r, c)
        expected = set(range(rows * cols))
        if set(origin) != expected:
            raise ToolError(f"knight_tour requires each of 0..{rows * cols - 1} exactly once")
        source = [[origin[placement[r][c]] for c in range(cols)] for r in range(rows)]

    else:
        raise ToolError(
            "action must be one of: rotate, transpose, flip_horizontal, flip_vertical, "
            "swap_rows, swap_columns, spiral_rearrange, knight_tour"
        )

    moves = [
        {"from": [source[r][c][0], source[r][c][1]], "to": [r, c]}
        for r in range(rows)
        for c in range(cols)
        if source[r][c] != (r, c)
    ]
    result: dict[str, Any] = {
        "rows": rows,
        "cols": cols,
        "matrix": [[rows_data[source[r][c][0]][source[r][c][1]] for c in range(cols)] for r in range(rows)],
        "moves": moves,
        "unchanged_cells": rows * cols - len(moves),
        "note": (
            "Feed `moves` straight to the renderer: each entry becomes one paste_region "
            "with source_cell [grid, from[0], from[1]] and target_cell [grid, to[0], to[1]]."
        ),
    }
    if action in {"swap_rows", "swap_columns"} and spec.get("select") == "extreme_sums":
        # Report whole numbers when the inputs were whole, so the planner does not have
        # to reason about "12.0 vs 12" when it echoes the sums into its rationale.
        result["sums"] = [int(value) if float(value).is_integer() else value for value in sums]
        result["swapped"] = sorted((first, second))
    return result


def _date_arithmetic(spec: dict[str, Any]) -> dict[str, Any]:
    try:
        start = datetime.fromisoformat(str(spec["start"]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as exc:
        raise ToolError("start must be a valid ISO-8601 datetime") from exc
    delta = spec.get("delta", {})
    if not isinstance(delta, dict):
        raise ToolError("delta must be an object")
    delta_values = {
        key: _finite_number(delta.get(key, 0), f"delta.{key}")
        for key in ("weeks", "days", "hours", "minutes", "seconds")
    }
    if sum(abs(value) for value in delta_values.values()) > 10_000_000:
        raise ToolError("Date delta is too large")
    operation = spec.get("operation", "add")
    if operation not in {"add", "subtract"}:
        raise ToolError("operation must be add or subtract")
    multiplier = -1 if operation == "subtract" else 1
    result = start + multiplier * timedelta(**delta_values)
    return {"result": result.isoformat(), "weekday": result.strftime("%A")}


class SafeCodeSolver:
    """Deterministic solver with a fixed operation allowlist; model-written code is never executed."""

    SCHEMAS: dict[str, dict[str, Any]] = {
        "arithmetic": {"expression": "numeric expression using +,-,*,/,//,%,**,sqrt,..."},
        "date_arithmetic": {
            "start": "ISO-8601 datetime",
            "operation": "add|subtract",
            "delta": {"weeks": 0, "days": 0, "hours": 0, "minutes": 0, "seconds": 0},
        },
        "grid_shortest_path": {
            "grid": ["...", ".#.", "..."],
            "start": [0, 0],
            "goal": [2, 2],
            "blocked": ["#"],
            "diagonal": False,
        },
        "graph_shortest_path": {
            "edges": [["A", "B", 2], ["B", "C", 1]],
            "start": "A",
            "goal": "C",
            "directed": False,
        },
        "sudoku": {
            "board": "NxN integer matrix with 0 for blanks",
            "box_rows": "omit for row/column-only Latin grids; use 3 for standard 9x9 or 2 for 6x6 with 2x3 boxes",
            "box_cols": "omit for row/column-only Latin grids; use 3 for standard 9x9 or 3 for 6x6 with 2x3 boxes",
        },
        "coordinate_geometry": {
            "action": "distance|midpoint|slope|rotate",
            "p1": [0, 0],
            "p2": [1, 1],
            "point": [1, 0],
            "center": [0, 0],
            "degrees": 90,
        },
        "matrix_transform": {
            "_note": (
                "Rearranges grid cells and returns `moves`, a ready-made cell permutation "
                "for the renderer. Use this for any matrix/board rearrangement instead of "
                "working out cell positions yourself. Transcribe the cell numbers row by "
                "row from the image into `matrix`."
            ),
            "matrix": [[1, 2], [3, 4]],
            "action": (
                "rotate|transpose|flip_horizontal|flip_vertical|swap_rows|swap_columns|"
                "spiral_rearrange|knight_tour"
            ),
            "degrees": "for rotate: 90, 180 or 270, clockwise",
            "indices": "for swap_rows/swap_columns: [i, j]",
            "select": "for swap_rows/swap_columns: 'extreme_sums' to swap largest-sum with smallest-sum",
            "start_corner": "for spiral_rearrange: top_left|top_right|bottom_left|bottom_right",
            "direction_cycle": "for spiral_rearrange, e.g. ['left','down','right','up']",
        },
    }
    OPERATIONS = tuple(SCHEMAS)

    @classmethod
    def protocol(cls) -> str:
        return json.dumps(cls.SCHEMAS, ensure_ascii=False)

    def solve(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ToolError("code_solver request must be a JSON object")
        operation = str(request.get("operation", ""))
        spec = request.get("input", {})
        if not isinstance(spec, dict):
            raise ToolError("code_solver input must be a JSON object")
        handlers = {
            "arithmetic": lambda value: {"result": _safe_eval(str(value["expression"]))},
            "date_arithmetic": _date_arithmetic,
            "grid_shortest_path": _grid_shortest_path,
            "graph_shortest_path": _graph_shortest_path,
            "sudoku": _solve_sudoku,
            "coordinate_geometry": _coordinate_geometry,
            "matrix_transform": _matrix_transform,
        }
        if operation not in handlers:
            raise ToolError(f"Unsupported operation {operation!r}; allowed: {', '.join(self.OPERATIONS)}")
        try:
            result = handlers[operation](spec)
        except KeyError as exc:
            raise ToolError(f"Missing required field: {exc.args[0]}") from exc
        return {"operation": operation, "result": result}


_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
# CJK-capable faces come first: several benchmark figures label rooms and axes in
# Chinese, and a Latin-only face renders those as blank tofu boxes with no error
# anywhere. `index` selects the face inside a .ttc collection (SC = Simplified).
_FONT_CANDIDATES: tuple[tuple[str, int], ...] = (
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 2),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc", 2),
    ("/System/Library/Fonts/Supplemental/PingFang.ttc", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 0),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
)
_MAX_OPS = 200
_MAX_POINTS = 1000
_MAX_TEXT = 200
_MAX_GRIDS = 8
_MAX_GRID_SIDE = 64


def _load_font(size_px: int) -> Any:
    size_px = max(6, min(512, int(size_px)))
    for path, index in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size_px, index=index)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size_px)
    except TypeError:
        return ImageFont.load_default()


def _star_vertices(
    center: tuple[float, float],
    radius: float,
    tips: int,
    inner_ratio: float,
    rotation_deg: float,
) -> list[tuple[float, float]]:
    """Vertices of a regular star, first tip pointing up unless rotated."""
    inner = radius * inner_ratio
    start = math.radians(rotation_deg) - math.pi / 2
    step = math.pi / tips
    return [
        (
            center[0] + (radius if i % 2 == 0 else inner) * math.cos(start + i * step),
            center[1] + (radius if i % 2 == 0 else inner) * math.sin(start + i * step),
        )
        for i in range(2 * tips)
    ]


_DASH_PATTERNS: dict[str, tuple[float, float]] = {
    # (on, off) as multiples of the stroke width, so a dash stays proportional to
    # the line it belongs to instead of vanishing on a thick stroke.
    "dashed": (4.0, 3.0),
    "dotted": (1.0, 2.0),
}


def _line_style(value: Any, field: str) -> str:
    style = "solid" if value is None else str(value).strip().lower()
    if style not in {"solid", *_DASH_PATTERNS}:
        raise ToolError(f"{field} must be solid, dashed or dotted")
    return style


def _dash_segments(
    points: list[tuple[float, float]], stroke: int, style: str
) -> list[list[tuple[float, float]]]:
    """Split a polyline into the drawn pieces of a dash pattern.

    Walking the path by arc length keeps the pattern continuous across corners, so a
    dashed route through a maze does not restart its rhythm at every turn.
    """
    on_length, off_length = (value * max(1, stroke) for value in _DASH_PATTERNS[style])
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = [points[0]]
    drawing = True
    remaining = on_length
    for start, end in zip(points, points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        ux, uy = dx / length, dy / length
        travelled = 0.0
        while length - travelled > remaining:
            travelled += remaining
            cut = (start[0] + ux * travelled, start[1] + uy * travelled)
            if drawing:
                current.append(cut)
                segments.append(current)
                current = []
            else:
                current = [cut]
            drawing = not drawing
            remaining = on_length if drawing else off_length
        remaining -= length - travelled
        if drawing:
            current.append(end)
    if drawing and len(current) > 1:
        segments.append(current)
    return segments


def _smooth_path(points: list[tuple[float, float]], steps: int = 12) -> list[tuple[float, float]]:
    """A Catmull-Rom spline through `points`, sampled densely enough to look smooth.

    The renderer offers no curve primitive, yet several tasks ask for a curved arrow
    or a rounded trajectory. Interpolating the planner's own waypoints keeps the
    result anchored to coordinates it can actually see, unlike a control-point spline.
    """
    if len(points) < 3:
        return points
    padded = [points[0], *points, points[-1]]
    curve: list[tuple[float, float]] = [points[0]]
    for p0, p1, p2, p3 in zip(padded, padded[1:], padded[2:], padded[3:]):
        for step in range(1, steps + 1):
            t = step / steps
            t2, t3 = t * t, t * t * t
            curve.append(
                (
                    0.5
                    * (
                        2 * p1[0]
                        + (-p0[0] + p2[0]) * t
                        + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                        + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
                    ),
                    0.5
                    * (
                        2 * p1[1]
                        + (-p0[1] + p2[1]) * t
                        + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                        + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
                    ),
                )
            )
    return curve


class _Grid:
    """A declared grid, so per-cell geometry is computed instead of guessed.

    Each axis is described one of two ways, and exactly one of them per axis:
      * a count (`rows` / `cols`), for an evenly divided axis, or
      * explicit boundaries (`row_edges` / `col_edges`), for an axis whose cells are
        not all the same size.
    The two axes are independent, so a figure with even columns but uneven rows
    declares `cols` and `row_edges` and does not have to transcribe the columns it
    could have had for free.

    Explicit boundaries exist because "looks like a grid" and "is an even grid" are
    different claims. A benchmark figure whose last row is 520px tall while the other
    four are 487px cannot be expressed by a count at all: dividing the outer box by
    five silently addresses the wrong rectangle, and the error grows down the grid
    while every operation still succeeds. Being able to state the real boundaries is
    what makes the correct answer expressible; validating them strictly is what turns
    a wrong guess into a repairable error instead of a plausible-looking output.

    Edges are stored as integers so that neighbouring cells share an edge exactly:
    cell (r, c)'s right edge *is* cell (r, c + 1)'s left edge, leaving neither a gap
    nor an overlap. Cell sizes may still differ by one pixel when an even span is not
    divisible by its count (2858 / 5 = 571.6), which callers that move pixels between
    cells must absorb rather than ignore.
    """

    __slots__ = ("x0", "y0", "x1", "y1", "rows", "cols", "name", "col_edges", "row_edges")

    def __init__(self, name: str, spec: Any, sx: float, sy: float) -> None:
        if not isinstance(spec, dict):
            raise ToolError(f"grids[{name!r}] must be a JSON object")
        bounds = spec.get("bbox")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
            raise ToolError(f"grids[{name!r}].bbox must be [x0, y0, x1, y1]")
        numbers = [_finite_number(bounds[i], f"grids[{name!r}].bbox[{i}]") for i in range(4)]
        self.x0, self.x1 = sorted((numbers[0] * sx, numbers[2] * sx))
        self.y0, self.y1 = sorted((numbers[1] * sy, numbers[3] * sy))
        if self.x1 - self.x0 < 1 or self.y1 - self.y0 < 1:
            raise ToolError(f"grids[{name!r}].bbox is degenerate")
        self.name = name
        self.col_edges = self._axis(
            spec, "cols", "col_edges", self.x0, self.x1, sx, f"grids[{name!r}]"
        )
        self.row_edges = self._axis(
            spec, "rows", "row_edges", self.y0, self.y1, sy, f"grids[{name!r}]"
        )
        self.cols = len(self.col_edges) - 1
        self.rows = len(self.row_edges) - 1
        # Explicit boundaries override the outer box on their own axis, so keep the
        # recorded extent consistent with the edges actually in use rather than with
        # a bbox component that is no longer authoritative.
        self.x0, self.x1 = float(self.col_edges[0]), float(self.col_edges[-1])
        self.y0, self.y1 = float(self.row_edges[0]), float(self.row_edges[-1])

    @classmethod
    def _axis(
        cls,
        spec: dict[str, Any],
        count_key: str,
        edges_key: str,
        start: float,
        end: float,
        scale: float,
        prefix: str,
    ) -> tuple[int, ...]:
        """Boundaries for one axis, from either a count or an explicit edge list."""
        count, edges = spec.get(count_key), spec.get(edges_key)
        if count is not None and edges is not None:
            raise ToolError(
                f"{prefix} may set either {count_key} or {edges_key} for that axis, "
                f"not both; they would describe two different sets of boundaries"
            )
        if count is None and edges is None:
            raise ToolError(f"{prefix} must set either {count_key} or {edges_key}")
        if edges is None:
            return cls._even_edges(start, end, cls._count(count, f"{prefix}.{count_key}"))
        return cls._explicit_edges(edges, scale, f"{prefix}.{edges_key}")

    @staticmethod
    def _count(value: Any, field: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ToolError(f"{field} must be an integer")
        if not 1 <= value <= _MAX_GRID_SIDE:
            raise ToolError(f"{field} must be between 1 and {_MAX_GRID_SIDE}")
        return value

    @staticmethod
    def _even_edges(start: float, end: float, count: int) -> tuple[int, ...]:
        """`count + 1` integer boundaries that tile [start, end] evenly.

        Each boundary is rounded exactly once and then shared by the two cells on
        either side of it, so no cell can be computed a pixel wider than its
        neighbour believes it to be.
        """
        span = end - start
        edges = [round(start + index * span / count) for index in range(count + 1)]
        # A grid finer than its own pixel span would otherwise produce zero-width
        # cells; nudge each boundary just enough to keep every cell non-empty.
        for index in range(1, len(edges)):
            if edges[index] <= edges[index - 1]:
                edges[index] = edges[index - 1] + 1
        return tuple(edges)

    @staticmethod
    def _explicit_edges(value: Any, scale: float, field: str) -> tuple[int, ...]:
        """Validated boundaries as supplied, in the request's own coordinate space.

        Rejecting a non-increasing list matters more than it looks: a duplicated or
        swapped boundary is exactly what a misread of the figure produces, and it
        would otherwise yield an empty or inverted cell that quietly addresses the
        wrong pixels. Failing here instead puts a named error in front of the repair
        loop, which can then re-read the boundaries off the image.
        """
        if not isinstance(value, (list, tuple)):
            raise ToolError(f"{field} must be a list of boundary positions")
        if len(value) < 2:
            raise ToolError(
                f"{field} needs at least two boundaries to describe one cell "
                f"(n cells require n + 1 boundaries)"
            )
        if len(value) > _MAX_GRID_SIDE + 1:
            raise ToolError(
                f"{field} describes more than {_MAX_GRID_SIDE} cells "
                f"(at most {_MAX_GRID_SIDE + 1} boundaries)"
            )
        edges = [
            round(_finite_number(item, f"{field}[{index}]") * scale)
            for index, item in enumerate(value)
        ]
        for index in range(1, len(edges)):
            if edges[index] <= edges[index - 1]:
                raise ToolError(
                    f"{field} must be strictly increasing, but boundary {index} "
                    f"({edges[index]}px) does not exceed boundary {index - 1} "
                    f"({edges[index - 1]}px); list the cell divisions in order, "
                    f"from the top or left edge to the opposite one"
                )
        if edges[0] < 0:
            raise ToolError(f"{field} starts before the image at {edges[0]}px")
        return tuple(edges)

    def cell(self, row: Any, col: Any, field: str) -> tuple[float, float, float, float]:
        for value, name, limit in ((row, "row", self.rows), (col, "column", self.cols)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ToolError(f"{field} {name} must be an integer")
            if not 0 <= value < limit:
                raise ToolError(
                    f"{field} {name} {value} is outside grid {self.name!r} (0..{limit - 1})"
                )
        return (
            float(self.col_edges[col]),
            float(self.row_edges[row]),
            float(self.col_edges[col + 1]),
            float(self.row_edges[row + 1]),
        )


class _RenderScale:
    """Per-image coordinate scaling, so the renderer itself stays stateless."""

    def __init__(self, size: tuple[int, int], normalized: bool) -> None:
        width, height = size
        self.sx = width if normalized else 1.0
        self.sy = height if normalized else 1.0
        self.ss = min(width, height) if normalized else 1.0
        # A stroke wider than a quarter of the image is never a legitimate annotation,
        # but a radius or glyph may legitimately span the whole image, so the two get
        # different ceilings.
        self.max_stroke = max(2, min(width, height) // 4)
        self.max_extent = max(2, min(width, height))
        self.grids: dict[str, _Grid] = {}
        self._grids_spec: Any = None
        self._normalized = normalized

    def load_grids(self, spec: Any) -> None:
        if spec is None:
            return
        if not isinstance(spec, dict):
            raise ToolError("'grids' must be a JSON object mapping names to grid definitions")
        if len(spec) > _MAX_GRIDS:
            raise ToolError(f"at most {_MAX_GRIDS} grids may be declared")
        self._grids_spec = spec
        self.grids = {
            str(name): _Grid(str(name), definition, self.sx, self.sy)
            for name, definition in spec.items()
        }

    def for_size(self, size: tuple[int, int]) -> "_RenderScale":
        """The same declared grids, resolved against a differently sized image.

        Grid bounds are relative to their own image, so a source image of another
        resolution still addresses the same logical cells.
        """
        other = _RenderScale(size, self._normalized)
        other.load_grids(self._grids_spec)
        return other

    def grid_cell(self, value: Any, field: str) -> tuple[float, float, float, float]:
        """Resolve a `[grid_name, row, col]` reference to a pixel rectangle."""
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ToolError(f"{field} must be [grid_name, row, col]")
        name = str(value[0])
        grid = self.grids.get(name)
        if grid is None:
            known = ", ".join(sorted(self.grids)) or "none declared"
            raise ToolError(f"{field} references unknown grid {name!r}; declared: {known}")
        return grid.cell(value[1], value[2], field)

    def point(self, value: Any, field: str) -> tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ToolError(f"{field} must be [x, y]")
        return (
            _finite_number(value[0], f"{field}.x") * self.sx,
            _finite_number(value[1], f"{field}.y") * self.sy,
        )

    def bbox(self, value: Any, field: str) -> tuple[float, float, float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ToolError(f"{field} must be [x0, y0, x1, y1]")
        numbers = [_finite_number(value[i], f"{field}[{i}]") for i in range(4)]
        x0, x1 = sorted((numbers[0] * self.sx, numbers[2] * self.sx))
        y0, y1 = sorted((numbers[1] * self.sy, numbers[3] * self.sy))
        return x0, y0, x1, y1

    def resolve_bbox(self, spec: dict[str, Any], field: str, *, cell_key: str = "cell") -> tuple[float, float, float, float]:
        """A rectangle given either explicitly as `bbox` or as a grid `cell` reference."""
        if spec.get(cell_key) is not None:
            return self.grid_cell(spec.get(cell_key), f"{field}.{cell_key}")
        bbox_key = "bbox" if cell_key == "cell" else cell_key.replace("_cell", "_bbox")
        return self.bbox(spec.get(bbox_key), f"{field}.{bbox_key}")

    def resolve_point(self, spec: dict[str, Any], field: str, key: str, cell_key: str) -> tuple[float, float]:
        """A point given either explicitly, or as the centre of a grid cell."""
        if spec.get(cell_key) is not None:
            x0, y0, x1, y1 = self.grid_cell(spec.get(cell_key), f"{field}.{cell_key}")
            return (x0 + x1) / 2, (y0 + y1) / 2
        return self.point(spec.get(key), f"{field}.{key}")

    def scalar(
        self,
        value: Any,
        field: str,
        default: float,
        minimum: int = 1,
        ceiling: int | None = None,
    ) -> int:
        number = default if value is None else _finite_number(value, field)
        limit = self.max_stroke if ceiling is None else ceiling
        return max(minimum, min(limit, round(number * self.ss)))


class ImageRenderer:
    """Deterministic image annotation.

    Applies a validated, fixed allowlist of drawing primitives to a source image.
    Nothing here executes model-authored code: the model may only emit declarative
    operations whose every field is range-checked before it reaches Pillow.

    Positions are normalised to [0, 1] of the image width/height by default, so a
    plan stays valid regardless of the resolution the planner actually saw.
    Scalars (`width`, `size`, `radius`) are fractions of min(width, height); stroke
    widths are capped at a quarter of that, while radii and glyph sizes may span it.
    """

    OPS = (
        "line",
        "polyline",
        "radial_line",
        "polygon",
        "rect",
        "circle",
        "arc",
        "star",
        "text",
        "clear_region",
        "recolor_region",
        "paste_region",
    )

    SCHEMA: dict[str, Any] = {
        "coord_space": "normalized (default, all coordinates in [0,1]) | pixel",
        "grids": {
            "_note": (
                "Optional. Declare a grid ONCE, then address its cells by integer "
                "[name, row, col] instead of computing per-cell coordinates. row 0 is the "
                "top row, col 0 the left column. Strongly preferred for matrices, boards, "
                "mazes, Sudoku and puzzle grids: it removes all coordinate arithmetic. "
                "Each axis is described EITHER by a count (`rows` / `cols`) when its cells "
                "are evenly sized, OR by explicit boundaries (`row_edges` / `col_edges`) "
                "when they are not. Give exactly one of the two per axis: supplying both "
                "for the same axis is rejected, because they would define conflicting "
                "boundaries. The axes are independent, so a figure with even columns and "
                "uneven rows uses `cols` together with `row_edges`."
            ),
            "matrix": {"bbox": [0.04, 0.07, 0.96, 0.98], "rows": 5, "cols": 5},
            "_uneven_example": {
                "bbox": [0.044, 0.067, 0.956, 0.933],
                "cols": 5,
                "row_edges": [0.067, 0.239, 0.409, 0.579, 0.750, 0.933],
                "_why": (
                    "Five rows whose boundaries are listed because the last row is taller "
                    "than the other four. n cells need n + 1 boundaries, strictly "
                    "increasing, in the same coordinate space as `bbox`. Boundaries "
                    "override `bbox` on their own axis, so `bbox` may still be given in "
                    "full. Before assuming an even grid, check the figure: if the cell "
                    "divisions are not equally spaced, a count addresses the wrong "
                    "rectangles and the error grows across the grid without any error "
                    "being reported."
                ),
            },
        },
        "cell_reference": (
            "Any op taking `bbox` also accepts `cell: [grid, row, col]` for that rectangle; "
            "any op taking `center`/`position` also accepts `cell` for the cell centre; "
            "`paste_region` accepts `source_cell` and `target_cell`."
        ),
        "ops": [
            {
                "op": "line",
                "start": [0.1, 0.2],
                "end": [0.8, 0.5],
                "color": "red",
                "width": 0.008,
                "arrow": False,
                "opacity": 1.0,
                "style": "solid",
            },
            {
                "op": "polyline",
                "points": [[0.1, 0.2], [0.4, 0.3], [0.8, 0.5]],
                "color": "red",
                "width": 0.008,
                "arrow": False,
                "style": "solid",
                "smooth": False,
                "_note": (
                    "`style` may be solid|dashed|dotted on any stroked op. Set `smooth` "
                    "true to round the polyline into a curve through its points, which is "
                    "how a curved arrow or a freehand-looking path is drawn."
                ),
            },
            {
                "op": "radial_line",
                "center": [0.5, 0.5],
                "angle": 90,
                "length": 0.3,
                "color": "black",
                "width": 0.01,
                "arrow": False,
                "_note": (
                    "A line from `center` outwards at `angle` degrees, measured clockwise "
                    "with 0 pointing straight UP (12 o'clock). This is the op for clock "
                    "hands, gauge and meter needles, compass arrows and any radial spoke: "
                    "use it instead of computing an endpoint with sin/cos yourself. "
                    "`length` is a fraction of the shorter image side. Also accepts `cell` "
                    "for the centre. For a clock, the hour hand is at "
                    "30*hour + 0.5*minute degrees and the minute hand at 6*minute degrees."
                ),
            },
            {
                "op": "polygon",
                "points": [[0.1, 0.2], [0.4, 0.3], [0.3, 0.6]],
                "color": "red",
                "width": 0.006,
                "fill": None,
            },
            {
                "op": "rect",
                "bbox": [0.1, 0.1, 0.4, 0.4],
                "color": "red",
                "width": 0.006,
                "fill": None,
            },
            {"op": "circle", "center": [0.5, 0.5], "radius": 0.05, "color": "red", "width": 0.006, "fill": None},
            {
                "op": "arc",
                "bbox": [0.1, 0.1, 0.4, 0.4],
                "start_angle": 0,
                "end_angle": 90,
                "color": "red",
                "width": 0.006,
                "fill": None,
                "_note": "angles in degrees clockwise from 3 o'clock; set fill to draw a filled pie slice",
            },
            {
                "op": "star",
                "center": [0.5, 0.5],
                "radius": 0.05,
                "color": "red",
                "fill": "red",
                "tips": 5,
                "inner_ratio": 0.42,
                "rotation": 0,
                "width": 0.004,
            },
            {
                "op": "text",
                "position": [0.5, 0.5],
                "text": "7",
                "color": "black",
                "size": 0.05,
                "anchor": "mm",
                "halo": "white",
                "_note": "CJK text is supported",
            },
            {
                "op": "clear_region",
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "color": None,
                "_note": (
                    "Erases: fills the rectangle so whatever was there disappears. Leave "
                    "`color` null and the surrounding background colour is sampled and "
                    "reused automatically, which is what you want on a diagram. Also "
                    "accepts `cell`. Use this before drawing a replacement on top of an "
                    "existing mark."
                ),
            },
            {
                "op": "recolor_region",
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "from": "#ffffff",
                "to": "#000000",
                "tolerance": 12,
                "_note": (
                    "Repaints only the pixels already close to `from`, leaving everything "
                    "else in the rectangle untouched. This is how a grid cell is filled or "
                    "cleared without destroying its borders, digits or letters: Nonogram "
                    "and Picross cells, lights-out and B/W flips, shading a chart bar, "
                    "recolouring a region of a map. `tolerance` is a per-channel distance "
                    "in 0..255. Omit `from` to repaint the region's own dominant colour, "
                    "which is the usual way to swap a cell's background. Also accepts "
                    "`cell`. Prefer this over rect+fill whenever content inside the "
                    "rectangle must survive."
                ),
            },
            {
                "op": "paste_region",
                "source_bbox": [0.0, 0.0, 0.2, 0.2],
                "target_bbox": [0.8, 0.8, 1.0, 1.0],
                "source_index": 0,
                "clear_source": False,
                "rotate": 0,
                "flip": None,
                "mask": None,
                "_note": (
                    "bboxes are [x0, y0, x1, y1] corners, NOT [x, y, width, height]. "
                    "`source_index` is 0-based over the source images as supplied, "
                    "independent of which one is the canvas. This op COPIES: set "
                    "clear_source true to erase the vacated region as well, i.e. to move "
                    "rather than duplicate. With a declared grid, prefer "
                    "source_cell/target_cell, e.g. "
                    "{\"op\": \"paste_region\", \"source_cell\": [\"matrix\", 4, 4], "
                    "\"target_cell\": [\"matrix\", 0, 0]}. "
                    "`rotate` is 0, 90, 180 or 270 degrees clockwise and `flip` is "
                    "horizontal or vertical, applied to the copied pixels before they land: "
                    "this is how an object is turned around, a shape rotated about its own "
                    "centre, or a pattern mirrored across an axis for a symmetry task. "
                    "Rotating by 90 or 270 swaps width and height, so give a target whose "
                    "aspect matches or the patch is resized to fit. `mask` keeps the "
                    "target's own background where the source is blank: set it to "
                    "\"nonwhite\" to transfer only non-white pixels, or to a colour such as "
                    "\"#ffffff\" to treat exactly that colour as transparent. Use a mask "
                    "when pasting a shape onto a textured or coloured background, otherwise "
                    "the source's rectangular background is copied along with it."
                ),
            },
        ],
    }

    @classmethod
    def protocol(cls) -> str:
        return json.dumps(cls.SCHEMA, ensure_ascii=False)

    # -- validation helpers -------------------------------------------------
    @staticmethod
    def _color(value: Any, field: str, default: str | None = "red") -> str | None:
        if value is None:
            return default
        text = str(value).strip().lower()
        if _HEX_COLOR.fullmatch(text):
            return text
        if text in ImageColor.colormap:
            return text
        raise ToolError(f"{field} must be a CSS colour name or #rrggbb, got {value!r}")

    @staticmethod
    def _opacity(value: Any) -> float:
        if value is None:
            return 1.0
        return max(0.0, min(1.0, _finite_number(value, "opacity")))

    # -- rendering ----------------------------------------------------------
    def render(
        self,
        images: list[Image.Image],
        request: dict[str, Any],
        *,
        base_index: int = 0,
    ) -> Image.Image:
        """Draw `request` onto `images[base_index]`, leaving every other pixel intact.

        The whole source list is taken rather than a base plus extras so that
        `paste_region.source_index` always means "the n-th image the agent was given",
        whichever of them the plan chose to draw on.
        """
        if not images:
            raise ToolError("render requires at least one source image")
        if not 0 <= base_index < len(images):
            raise ToolError(
                f"base_image_index {base_index} is out of range for {len(images)} source image(s)"
            )
        if not isinstance(request, dict):
            raise ToolError("render request must be a JSON object")
        ops = request.get("ops", [])
        if not isinstance(ops, list) or not ops:
            raise ToolError("render request must contain a non-empty 'ops' list")
        if len(ops) > _MAX_OPS:
            raise ToolError(f"render request has too many ops (max {_MAX_OPS})")

        space = str(request.get("coord_space", "normalized")).strip().lower()
        if space not in {"normalized", "pixel"}:
            raise ToolError("coord_space must be 'normalized' or 'pixel'")
        normalized = space == "normalized"

        canvas = images[base_index].convert("RGB").copy()
        scale = _RenderScale(canvas.size, normalized)
        scale.load_grids(request.get("grids"))
        snapshots = [image.convert("RGB") for image in images]

        for position, raw in enumerate(ops):
            if not isinstance(raw, dict):
                raise ToolError(f"ops[{position}] must be a JSON object")
            name = str(raw.get("op", "")).strip().lower()
            if name not in self.OPS:
                raise ToolError(
                    f"ops[{position}] uses unsupported op {name!r}; allowed: {', '.join(self.OPS)}"
                )
            canvas = self._apply(
                canvas, scale, name, raw, position, snapshots, normalized, base_index
            )
        if ImageChops.difference(canvas, snapshots[base_index]).getbbox() is None:
            raise ToolError("Drawing program left the image unchanged")
        return canvas

    def _apply(
        self,
        canvas: Image.Image,
        scale: _RenderScale,
        name: str,
        spec: dict[str, Any],
        position: int,
        snapshots: list[Image.Image],
        normalized: bool,
        base_index: int,
    ) -> Image.Image:
        if name == "paste_region":
            return self._paste_region(
                canvas, scale, spec, position, snapshots, normalized, base_index
            )
        if name == "clear_region":
            return self._clear_region(canvas, scale, spec, position, snapshots, base_index)
        if name == "recolor_region":
            return self._recolor_region(canvas, scale, spec, position, snapshots, base_index)

        opacity = self._opacity(spec.get("opacity"))
        if opacity <= 0:
            return canvas
        if opacity >= 1.0:
            self._draw(ImageDraw.Draw(canvas), scale, name, spec, position)
            return canvas
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        self._draw(ImageDraw.Draw(overlay), scale, name, spec, position)
        box = overlay.getbbox()
        if box is None:
            return canvas
        patch = overlay.crop(box)
        patch.putalpha(patch.getchannel("A").point(lambda value: int(value * opacity)))
        blended = Image.alpha_composite(canvas.crop(box).convert("RGBA"), patch)
        canvas.paste(blended.convert("RGB"), box[:2])
        return canvas

    def _draw(
        self,
        draw: ImageDraw.ImageDraw,
        scale: _RenderScale,
        name: str,
        spec: dict[str, Any],
        position: int,
    ) -> None:
        field = f"ops[{position}]"
        color = self._color(spec.get("color"), f"{field}.color")
        stroke = scale.scalar(spec.get("width"), f"{field}.width", 0.008)

        if name in {"line", "polyline", "radial_line"}:
            if name == "line":
                points = [
                    scale.point(spec.get("start"), f"{field}.start"),
                    scale.point(spec.get("end"), f"{field}.end"),
                ]
            elif name == "radial_line":
                center = scale.resolve_point(spec, field, "center", "cell")
                angle = math.radians(
                    _finite_number(spec.get("angle"), f"{field}.angle") - 90.0
                )
                length = scale.scalar(
                    spec.get("length"), f"{field}.length", 0.3, minimum=2, ceiling=scale.max_extent
                )
                points = [
                    center,
                    (
                        center[0] + length * math.cos(angle),
                        center[1] + length * math.sin(angle),
                    ),
                ]
            else:
                raw_points = spec.get("points")
                if not isinstance(raw_points, list) or len(raw_points) < 2:
                    raise ToolError(f"{field}.points needs at least two points")
                if len(raw_points) > _MAX_POINTS:
                    raise ToolError(f"{field}.points has too many points (max {_MAX_POINTS})")
                points = [scale.point(p, f"{field}.points[{i}]") for i, p in enumerate(raw_points)]
                if bool(spec.get("smooth", False)):
                    points = _smooth_path(points)
            style = _line_style(spec.get("style"), f"{field}.style")
            if style == "solid":
                strokes = [points]
            else:
                strokes = _dash_segments(points, stroke, style)
                if not strokes:
                    raise ToolError(
                        f"{field} is too short for a {style} line; use style 'solid' "
                        "or a longer path"
                    )
            for piece in strokes:
                draw.line(piece, fill=color, width=stroke, joint="curve")
                radius = stroke / 2
                for point in piece[1:-1]:
                    draw.ellipse(
                        [
                            point[0] - radius,
                            point[1] - radius,
                            point[0] + radius,
                            point[1] + radius,
                        ],
                        fill=color,
                    )
            # The head belongs to the path, not to the last dash, so it is drawn from
            # the true final direction whatever the dash pattern did.
            if bool(spec.get("arrow", False)):
                self._arrow_head(draw, points[-2], points[-1], color, stroke)
            return

        if name in {"rect", "circle"}:
            if name == "circle":
                center = scale.resolve_point(spec, field, "center", "cell")
                radius = scale.scalar(
                    spec.get("radius"), f"{field}.radius", 0.05, ceiling=scale.max_extent
                )
                box = (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)
            else:
                box = scale.resolve_bbox(spec, field)
            fill = self._color(spec.get("fill"), f"{field}.fill", default=None)
            shape = draw.rectangle if name == "rect" else draw.ellipse
            wants_outline = spec.get("color") is not None or spec.get("width") is not None
            outline = color if (wants_outline or fill is None) else None
            shape(box, outline=outline, width=stroke if outline else 0, fill=fill)
            return

        if name in {"polygon", "star"}:
            if name == "star":
                center = scale.resolve_point(spec, field, "center", "cell")
                radius = scale.scalar(
                    spec.get("radius"), f"{field}.radius", 0.05, minimum=3, ceiling=scale.max_extent
                )
                tips = spec.get("tips", 5)
                if not isinstance(tips, int) or not 3 <= tips <= 24:
                    raise ToolError(f"{field}.tips must be an integer between 3 and 24")
                inner_ratio = _finite_number(spec.get("inner_ratio", 0.42), f"{field}.inner_ratio")
                if not 0.05 <= inner_ratio <= 0.95:
                    raise ToolError(f"{field}.inner_ratio must be between 0.05 and 0.95")
                points = _star_vertices(
                    center,
                    radius,
                    tips,
                    inner_ratio,
                    _finite_number(spec.get("rotation", 0), f"{field}.rotation"),
                )
                fill = self._color(
                    spec.get("fill"), f"{field}.fill", default=None if "fill" in spec else color
                )
            else:
                raw_points = spec.get("points")
                if not isinstance(raw_points, list) or len(raw_points) < 3:
                    raise ToolError(f"{field}.points needs at least three points")
                if len(raw_points) > _MAX_POINTS:
                    raise ToolError(f"{field}.points has too many points (max {_MAX_POINTS})")
                points = [scale.point(p, f"{field}.points[{i}]") for i, p in enumerate(raw_points)]
                fill = self._color(spec.get("fill"), f"{field}.fill", default=None)
            wants_outline = spec.get("color") is not None or spec.get("width") is not None
            outline = color if (wants_outline or fill is None) else None
            draw.polygon(points, fill=fill, outline=outline, width=stroke if outline else 0)
            return

        if name == "arc":
            box = scale.resolve_bbox(spec, field)
            start_angle = _finite_number(spec.get("start_angle", 0), f"{field}.start_angle")
            end_angle = _finite_number(spec.get("end_angle", 90), f"{field}.end_angle")
            if abs(end_angle - start_angle) < 1e-6:
                raise ToolError(f"{field}.start_angle and end_angle must differ")
            fill = self._color(spec.get("fill"), f"{field}.fill", default=None)
            if fill is not None:
                wants_outline = spec.get("color") is not None or spec.get("width") is not None
                draw.pieslice(
                    box,
                    start_angle,
                    end_angle,
                    fill=fill,
                    outline=color if wants_outline else None,
                    width=stroke if wants_outline else 0,
                )
            else:
                draw.arc(box, start_angle, end_angle, fill=color, width=stroke)
            return

        if name == "text":
            text = str(spec.get("text", ""))
            if not text:
                raise ToolError(f"{field}.text cannot be empty")
            if len(text) > _MAX_TEXT:
                raise ToolError(f"{field}.text is too long (max {_MAX_TEXT} characters)")
            anchor = str(spec.get("anchor", "mm"))
            if anchor not in {"lt", "mt", "rt", "lm", "mm", "rm", "lb", "mb", "rb", "ls", "ms", "rs"}:
                raise ToolError(f"{field}.anchor is not a valid Pillow anchor")
            size = scale.scalar(
                spec.get("size"), f"{field}.size", 0.05, minimum=6, ceiling=scale.max_extent
            )
            halo = self._color(spec.get("halo"), f"{field}.halo", default=None)
            draw.text(
                scale.resolve_point(spec, field, "position", "cell"),
                text,
                fill=self._color(spec.get("color"), f"{field}.color", default="black"),
                font=_load_font(size),
                anchor=anchor,
                stroke_width=max(1, size // 12) if halo else 0,
                stroke_fill=halo,
            )
            return

        raise ToolError(f"{field} uses unsupported op {name!r}")

    @staticmethod
    def _arrow_head(
        draw: ImageDraw.ImageDraw,
        start: tuple[float, float],
        end: tuple[float, float],
        color: str | None,
        stroke: int,
    ) -> None:
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        ux, uy = dx / length, dy / length
        size = max(stroke * 3.0, 6.0)
        base = (end[0] - ux * size, end[1] - uy * size)
        half = size * 0.5
        draw.polygon(
            [end, (base[0] - uy * half, base[1] + ux * half), (base[0] + uy * half, base[1] - ux * half)],
            fill=color,
        )

    @staticmethod
    def _clamp_box(
        box: tuple[float, float, float, float], image: Image.Image, field: str
    ) -> tuple[int, int, int, int]:
        clamped = (
            max(0, int(round(box[0]))),
            max(0, int(round(box[1]))),
            min(image.width, int(round(box[2]))),
            min(image.height, int(round(box[3]))),
        )
        if clamped[2] - clamped[0] < 1 or clamped[3] - clamped[1] < 1:
            raise ToolError(f"{field} is empty after clamping to the image")
        return clamped

    @staticmethod
    def _surrounding_colour(
        source: Image.Image, box: tuple[int, int, int, int], field: str, ring: int = 2
    ) -> tuple[int, int, int]:
        """The dominant colour of a thin ring just outside `box`.

        Erasing needs the local background, and the planner cannot sample pixels: asking
        it to name a hex value produces a coloured patch wherever it guesses wrong, which
        is worse than not erasing at all. Reading the ring from the pre-op snapshot keeps
        the answer stable no matter what earlier ops painted.
        """
        x0, y0, x1, y1 = box
        strips = [
            (x0 - ring, y0 - ring, x1 + ring, y0),
            (x0 - ring, y1, x1 + ring, y1 + ring),
            (x0 - ring, y0, x0, y1),
            (x1, y0, x1 + ring, y1 + ring),
        ]
        counts: Counter[tuple[int, int, int]] = Counter()
        for strip in strips:
            crop = (
                max(0, strip[0]),
                max(0, strip[1]),
                min(source.width, strip[2]),
                min(source.height, strip[3]),
            )
            if crop[2] - crop[0] < 1 or crop[3] - crop[1] < 1:
                continue
            patch = source.crop(crop)
            for count, colour in patch.getcolors(maxcolors=patch.width * patch.height) or []:
                counts[colour] += count
        if not counts:
            raise ToolError(
                f"{field} touches every edge of the image, so no surrounding background "
                "can be sampled; set an explicit 'color' instead"
            )
        return counts.most_common(1)[0][0]

    def _clear_region(
        self,
        canvas: Image.Image,
        scale: _RenderScale,
        spec: dict[str, Any],
        position: int,
        snapshots: list[Image.Image],
        base_index: int,
    ) -> Image.Image:
        field = f"ops[{position}]"
        box = self._clamp_box(scale.resolve_bbox(spec, field), canvas, f"{field}.bbox")
        colour = self._color(spec.get("color"), f"{field}.color", default=None)
        fill = (
            ImageColor.getrgb(colour)
            if colour is not None
            else self._surrounding_colour(snapshots[base_index], box, field)
        )
        canvas.paste(fill, box)
        return canvas

    @staticmethod
    def _dominant_colour(image: Image.Image) -> tuple[int, int, int]:
        colours = image.getcolors(maxcolors=image.width * image.height) or []
        if not colours:
            raise ToolError("region has no sampleable pixels")
        return max(colours)[1]

    def _recolor_region(
        self,
        canvas: Image.Image,
        scale: _RenderScale,
        spec: dict[str, Any],
        position: int,
        snapshots: list[Image.Image],
        base_index: int,
    ) -> Image.Image:
        """Repaint only the pixels matching one colour, leaving the rest of the box alone.

        `rect` with a fill is the wrong tool for "fill this cell": it also covers the
        cell's borders, its digit and its letter, which is precisely the content a
        Nonogram, lights-out or magic-square answer has to keep. Selecting by colour
        instead means the grid survives the edit and only the background changes.
        """
        field = f"ops[{position}]"
        box = self._clamp_box(scale.resolve_bbox(spec, field), canvas, f"{field}.bbox")
        target_colour = self._color(spec.get("to"), f"{field}.to", default=None)
        if target_colour is None:
            raise ToolError(f"{field}.to is required and must be a colour")
        replacement = ImageColor.getrgb(target_colour)[:3]

        tolerance = int(_finite_number(spec.get("tolerance", 12), f"{field}.tolerance"))
        if not 0 <= tolerance <= 255:
            raise ToolError(f"{field}.tolerance must be between 0 and 255")

        region = canvas.crop(box)
        source_colour = self._color(spec.get("from"), f"{field}.from", default=None)
        if source_colour is not None:
            match = ImageColor.getrgb(source_colour)[:3]
        else:
            match = self._dominant_colour(snapshots[base_index].crop(box))

        pixels = region.load()
        changed = 0
        for y in range(region.height):
            for x in range(region.width):
                pixel = pixels[x, y]
                if all(abs(pixel[i] - match[i]) <= tolerance for i in range(3)):
                    pixels[x, y] = replacement
                    changed += 1
        if not changed:
            raise ToolError(
                f"{field} matched no pixels: no colour within tolerance {tolerance} of "
                f"{match} is present in the region"
            )
        canvas.paste(region, box[:2])
        return canvas

    @staticmethod
    def _orient_patch(
        patch: Image.Image, spec: dict[str, Any], field: str
    ) -> Image.Image:
        """Apply the requested rotation and mirroring to copied pixels.

        Rotation is in whole quarter turns only, so it is a lossless pixel permutation:
        an arbitrary angle would resample and blur the very content the programmatic
        executor exists to preserve.
        """
        rotate = int(_finite_number(spec.get("rotate", 0), f"{field}.rotate")) % 360
        if rotate not in {0, 90, 180, 270}:
            raise ToolError(f"{field}.rotate must be 0, 90, 180 or 270")
        if rotate:
            # Pillow's `transpose` rotates counter-clockwise, while the schema and every
            # task wording use clockwise, so the constant is mirrored here.
            patch = patch.transpose(
                {
                    90: Image.Transpose.ROTATE_270,
                    180: Image.Transpose.ROTATE_180,
                    270: Image.Transpose.ROTATE_90,
                }[rotate]
            )
        flip = spec.get("flip")
        if flip is not None:
            name = str(flip).strip().lower()
            if name not in {"horizontal", "vertical"}:
                raise ToolError(f"{field}.flip must be 'horizontal' or 'vertical'")
            patch = patch.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
                if name == "horizontal"
                else Image.Transpose.FLIP_TOP_BOTTOM
            )
        return patch

    def _paste_mask(
        self, patch: Image.Image, spec: dict[str, Any], field: str
    ) -> Image.Image | None:
        """An alpha mask that hides the source's own background, or None for a full copy."""
        raw = spec.get("mask")
        if raw is None:
            return None
        name = str(raw).strip().lower()
        if name in {"none", "false"}:
            return None
        if name == "nonwhite":
            transparent = (255, 255, 255)
            tolerance = 24
        else:
            colour = self._color(raw, f"{field}.mask", default=None)
            if colour is None:
                raise ToolError(f"{field}.mask must be 'nonwhite' or a colour")
            transparent = ImageColor.getrgb(colour)[:3]
            tolerance = int(_finite_number(spec.get("mask_tolerance", 24), f"{field}.mask_tolerance"))
            if not 0 <= tolerance <= 255:
                raise ToolError(f"{field}.mask_tolerance must be between 0 and 255")
        mask = Image.new("L", patch.size, 255)
        mask_pixels = mask.load()
        patch_pixels = patch.load()
        for y in range(patch.height):
            for x in range(patch.width):
                pixel = patch_pixels[x, y]
                if all(abs(pixel[i] - transparent[i]) <= tolerance for i in range(3)):
                    mask_pixels[x, y] = 0
        if not mask.getbbox():
            raise ToolError(
                f"{field}.mask hid every pixel of the source region, so the paste would "
                "do nothing; widen the source or drop the mask"
            )
        return mask

    def _paste_region(
        self,
        canvas: Image.Image,
        scale: _RenderScale,
        spec: dict[str, Any],
        position: int,
        snapshots: list[Image.Image],
        normalized: bool,
        base_index: int,
    ) -> Image.Image:
        field = f"ops[{position}]"
        try:
            source_index = int(spec.get("source_index", 0))
        except (TypeError, ValueError) as exc:
            raise ToolError(f"{field}.source_index must be an integer") from exc
        if not 0 <= source_index < len(snapshots):
            raise ToolError(f"{field}.source_index {source_index} is out of range")
        source = snapshots[source_index]

        # A cell reference resolves against the grid geometry of the image it addresses:
        # the source snapshot may be a different size from the canvas.
        source_scale = scale.for_size(source.size)
        if spec.get("source_cell") is not None:
            src = source_scale.grid_cell(spec.get("source_cell"), f"{field}.source_cell")
        else:
            src = source_scale.bbox(spec.get("source_bbox"), f"{field}.source_bbox")
        if spec.get("target_cell") is not None:
            target = scale.grid_cell(spec.get("target_cell"), f"{field}.target_cell")
        else:
            target = scale.bbox(spec.get("target_bbox"), f"{field}.target_bbox")

        src_box = self._clamp_box(src, source, f"{field}.source_bbox")
        target_box = self._clamp_box(target, canvas, f"{field}.target_bbox")
        target_size = (target_box[2] - target_box[0], target_box[3] - target_box[1])
        patch = self._orient_patch(source.crop(src_box), spec, field)
        if patch.size != target_size:
            cell_to_cell = (
                spec.get("source_cell") is not None and spec.get("target_cell") is not None
            )
            if cell_to_cell and self._is_off_by_one(patch.size, target_size):
                patch = self._pad_to(patch, target_size)
            else:
                patch = patch.resize(target_size, Image.Resampling.LANCZOS)
        mask = self._paste_mask(patch, spec, field)
        if bool(spec.get("clear_source", False)):
            if source_index != base_index:
                raise ToolError(
                    f"{field}.clear_source only applies when the region comes from the canvas "
                    f"itself (source_index {base_index}), not from another source image"
                )
            canvas.paste(self._surrounding_colour(snapshots[base_index], src_box, field), src_box)
        canvas.paste(patch, target_box[:2], mask)
        return canvas

    @staticmethod
    def _is_off_by_one(size: tuple[int, int], target: tuple[int, int]) -> bool:
        """Whether two cell sizes differ only by grid rounding, not by intent."""
        return abs(size[0] - target[0]) <= 1 and abs(size[1] - target[1]) <= 1

    @staticmethod
    def _pad_to(patch: Image.Image, size: tuple[int, int]) -> Image.Image:
        """`patch` resized to `size` by repeating its edge pixels, never resampling.

        Only ever called for a one-pixel discrepancy, so at most a single row and
        column of duplicated edge pixels is introduced - invisible in the output, and
        crucially it covers the whole target rather than leaving a seam.
        """
        width, height = size
        result = patch.crop((0, 0, min(patch.width, width), min(patch.height, height)))
        if result.width < width:
            edge = result.crop((result.width - 1, 0, result.width, result.height))
            grown = Image.new(result.mode, (width, result.height))
            grown.paste(result, (0, 0))
            for x in range(result.width, width):
                grown.paste(edge, (x, 0))
            result = grown
        if result.height < height:
            edge = result.crop((0, result.height - 1, result.width, result.height))
            grown = Image.new(result.mode, (result.width, height))
            grown.paste(result, (0, 0))
            for y in range(result.height, height):
                grown.paste(edge, (0, y))
            result = grown
        return result

