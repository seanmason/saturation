from typing import Tuple, List, Optional

import pandas as pd
import numpy as np
import numba as nb
from numba import njit

from saturation.datatypes import Location, Arc


arc_type = nb.types.UniTuple(nb.float64, 2)


@njit(fastmath=True)
def get_xy_intersection(center1: Location,
                        center2: Location,
                        r1: float,
                        r2: float) -> Tuple[Location, Location]:
    """
    Returns the (x, y) intersection locations of two circles.
    Does not handle the case where the circles do not intersect.
    Adapted from https://gist.github.com/jupdike/bfe5eb23d1c395d8a0a1a4ddd94882ac
    """
    if center1 == center2:
        return (0.0, 0.0), (0.0, 0.0)

    x1, y1 = center1
    x2, y2 = center2

    center_dx = x1 - x2
    center_dy = y1 - y2
    R = np.sqrt(center_dx * center_dx + center_dy * center_dy)
    R2 = R * R
    R4 = R2 * R2
    a = (r1 * r1 - r2 * r2) / (2 * R2)
    r2r2 = (r1 * r1 - r2 * r2)
    c = np.sqrt(2 * (r1 * r1 + r2 * r2) / R2 - (r2r2 * r2r2) / R4 - 1)
    fx = (x1 + x2) / 2 + a * (x2 - x1)
    gx = c * (y2 - y1) / 2
    fy = (y1 + y2) / 2 + a * (y2 - y1)
    gy = c * (x1 - x2) / 2

    return (fx + gx, fy + gy), (fx - gx, fy - gy)


@njit(fastmath=True)
def get_intersection_arc(center1: Location,
                         r1: float,
                         center2: Location,
                         r2: float) -> Arc:
    """
    Returns the intersection arc (in radians) of the circle defined by center2 and r2 on the circle
    defined by center1 and r1.
    """
    # Circle 1 is completely encompassed by circle 2
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    if distance + r1 < r2:
        return 0.0, 2 * np.pi

    # Circle 2 is completely encompassed by circle 1
    if distance + r2 < r1:
        return 0.0, 0.0

    i1, i2 = get_xy_intersection(center1, center2, r1, r2)

    # np.arctan2 gets us the result in the correct quadrant
    # Note that arctan2 takes arguments (y, x)
    theta1 = np.arctan2(i1[1] - center1[1], i1[0] - center1[0])
    theta2 = np.arctan2(i2[1] - center1[1], i2[0] - center1[0])

    # Adjust both thetas to be in (0, 2*pi)
    theta1 = theta1 if theta1 > 0 else theta1 + 2 * np.pi
    theta2 = theta2 if theta2 > 0 else theta2 + 2 * np.pi
    if theta1 > theta2:
        tmp = theta1
        theta1 = theta2
        theta2 = tmp

    # Test if the midpoint of (theta1, theta2) on circle1 is inside circle2
    midpoint = (theta1 + theta2) / 2
    midpoint_x = center1[0] + r1 * np.cos(midpoint)
    midpoint_y = center1[1] + r1 * np.sin(midpoint)
    test_value = (center2[0] - midpoint_x) ** 2 + (center2[1] - midpoint_y) ** 2

    # If our test point is not within circle2, reverse our thetas
    if test_value >= r2 ** 2:
        tmp = theta1
        theta1 = theta2
        theta2 = tmp

    return theta1, theta2


def solve_quadratic(a: float, b: float, c: float) -> Tuple[float, float]:
    first = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    second = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    return first, second


def find_intersections_with_study_region_bounds(center: Location,
                                                radius: float,
                                                study_region_size: int,
                                                study_region_padding: int) -> List[Location]:
    result = []

    low_boundary = study_region_padding
    high_boundary = study_region_size + study_region_padding

    # The circle goes off the left side of the study region
    if center[0] - radius < low_boundary:
        offset = low_boundary
        test_x = center[0] - offset
        test_y = center[1] - offset

        a = 1
        b = -2 * test_y
        c = test_y ** 2 - radius ** 2 - test_x ** 2
        solutions = solve_quadratic(a, b, c)
        if solutions[0] > 0:
            result.append((offset, solutions[0] + offset))
        if solutions[1] > 0:
            result.append((offset, solutions[1] + offset))

    # The circle goes off the right side of the study region
    if center[0] + radius > high_boundary:
        offset = high_boundary
        test_x = center[0] - offset
        test_y = center[1] - offset

        a = 1
        b = -2 * test_y
        c = test_y ** 2 - radius ** 2 - test_x ** 2
        solutions = solve_quadratic(a, b, c)
        if solutions[0] > 0:
            result.append((offset, solutions[0] + offset))
        if solutions[1] > 0:
            result.append((offset, solutions[1] + offset))

    # The circle goes off the bottom side of the study region
    if  center[1] - radius < low_boundary:
        offset = low_boundary
        test_x = center[0] - offset
        test_y = center[1] - offset

        a = 1
        b = -2 * test_x
        c = test_x ** 2 - radius ** 2 - test_y ** 2
        solutions = solve_quadratic(a, b, c)
        if solutions[0] > 0:
            result.append((solutions[0] + offset, offset))
        if solutions[1] > 0:
            result.append((solutions[1] + offset, offset))

    # The circle goes off the top side of the study region
    if center[1] + radius > high_boundary:
        offset = high_boundary
        test_x = center[0] - offset
        test_y = center[1] - offset

        a = 1
        b = -2 * test_x
        c = test_x ** 2 - radius ** 2 - test_y ** 2
        solutions = solve_quadratic(a, b, c)
        if solutions[0] > 0:
            result.append((solutions[0] + offset, offset))
        if solutions[1] > 0:
            result.append((solutions[1] + offset, offset))

    return result


def get_study_region_boundary_intersection_arc(center: Location,
                                               radius: float,
                                               study_region_size: int,
                                               study_region_padding: int) -> Optional[Arc]:
    """
    Returns the intersection arc (in radians) of the specified circle with the study region's bounds.
    """
    intersections = find_intersections_with_study_region_bounds(center, radius, study_region_size, study_region_padding)

    if len(intersections) != 2:
        return None

    # np.arctan2 gets us the result in the correct quadrant
    # Note that arctan2 takes arguments (y, x)
    theta1 = np.arctan2(intersections[0][1] - center[1], intersections[0][0] - center[0])
    theta2 = np.arctan2(intersections[1][1] - center[1], intersections[1][0] - center[0])

    # Adjust both thetas to be in (0, 2*pi)
    theta1 = theta1 if theta1 > 0 else theta1 + 2 * np.pi
    theta2 = theta2 if theta2 > 0 else theta2 + 2 * np.pi
    if theta1 > theta2:
        tmp = theta1
        theta1 = theta2
        theta2 = tmp

    # Test if the midpoint of (theta1, theta2) on the circle is in bounds
    midpoint = (theta1 + theta2) / 2
    midpoint_x = center[0] + radius * np.cos(midpoint)
    midpoint_y = center[1] + radius * np.sin(midpoint)

    # If our test point is not within circle2, reverse our thetas
    if not (
            study_region_padding <= midpoint_x <= study_region_size + study_region_padding and study_region_padding <= midpoint_y <= study_region_size + study_region_padding):
        tmp = theta1
        theta1 = theta2
        theta2 = tmp

    return theta1, theta2


def get_erased_rim_arcs(craters: pd.DataFrame,
                        min_crater_radius: float,
                        rmult: float) -> pd.DataFrame:
    """
    Returns the erased rim arcs resulting from a supplied sequence of craters.
    rmult specifies a multiplier on the size of a newly-formed crater at removing other craters' rims.
    Erased rim arcs will only result for craters with radii greater than min_crater_radius
    """
    min_id = min(craters.index)
    max_id = max(craters.index)

    crater_ids_larger_than_min = craters[craters.radius >= min_crater_radius].index

    erased_arcs = []
    for new_id in range(min_id, max_id + 1):
        new_crater = craters.loc[new_id]
        new_x = new_crater.x
        new_y = new_crater.y
        new_radius = new_crater.radius * rmult

        # Filter to only circles with radius greater than the threshold
        filtered = craters.loc[crater_ids_larger_than_min[crater_ids_larger_than_min < new_id]]

        # Further filter to only those that intersect
        distance = np.sqrt((filtered.x - new_x) ** 2 + (filtered.y - new_y) ** 2)
        filtered = filtered[(distance < filtered.radius + new_radius)
                            & (distance + new_radius > filtered.radius)].reset_index()

        for row in filtered[['x', 'y', 'radius', 'id']].values:
            arc = get_intersection_arc((row[0], row[1]),
                                       row[2],
                                       (new_x, new_y),
                                       new_radius)

            erased_arcs.append({
                'new_id': new_id,
                'old_id': int(row[3]),
                'theta1': arc[0],
                'theta2': arc[1]
            })

    return pd.DataFrame(erased_arcs)


@njit(fastmath=True)
def add_arc(new_arc: Arc, existing_arcs: List[Arc]) -> None:
    """
    Adds a new arc and merges as necessary, accounting for overlaps.
    The result will contain no arcs that cross 0 or 2*pi.
    existing_arcs is modified in-place.
    """
    # Normalize the new arc to not cross 0 or 2*pi
    if new_arc[0] > new_arc[1]:
        existing_arcs.append((new_arc[0], 2 * np.pi))
        existing_arcs.append((0, new_arc[1]))
    else:
        existing_arcs.append(new_arc)

    # Sort before merging
    existing_arcs.sort(key=lambda x: x[0])

    # Scan arcs and merge if there are overlaps
    index = 0
    while index < len(existing_arcs) - 1:
        current_arc = existing_arcs[index]
        next_arc = existing_arcs[index + 1]

        if current_arc[1] >= next_arc[0]:
            existing_arcs[index] = (current_arc[0], max(current_arc[1], next_arc[1]))
            del existing_arcs[index + 1]
        else:
            index += 1


@njit(fastmath=True)
def calculate_rim_percentage_remaining(erased_arcs: List[Arc]) -> float:
    """
    Calculates the percentage of rim remaining, given a set of erased arcs.
    """
    arcs = nb.typed.List.empty_list(arc_type)
    for erased_arc in erased_arcs:
        add_arc(erased_arc, arcs)

    return 1.0 - sum([x[1] - x[0] for x in arcs]) / (2 * np.pi)
