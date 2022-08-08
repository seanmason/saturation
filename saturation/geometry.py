from typing import Tuple, List

import pandas as pd
import numpy as np

from saturation.datatypes import Location, Arc, SortedArcList


def get_xy_intersection(center1: Location,
                        center2: Location,
                        r1: float,
                        r2: float) -> Tuple[Location, Location]:
    """
    Returns the (x, y) intersection locations of two circles.
    Does not handle the case where the circles do not intersect.
    Adapted from https://gist.github.com/jupdike/bfe5eb23d1c395d8a0a1a4ddd94882ac
    """
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
        return 0, 2 * np.pi

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


def find_intersections_with_terrain_bounds(center: Location,
                                           radius: float,
                                           observed_terrain_size: int,
                                           terrain_padding: int) -> List[Location]:
    result = []

    low_boundary = terrain_padding
    high_boundary = observed_terrain_size + terrain_padding

    # The circle goes off the left side of the terrain
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

    # The circle goes off the right side of the terrain
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

    # The circle goes off the bottom side of the terrain
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

    # The circle goes off the top side of the terrain
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


def get_terrain_boundary_intersection_arc(center: Location,
                                          radius: float,
                                          observed_terrain_size: int,
                                          terrain_padding: int) -> Arc:
    """
    Returns the intersection arc (in radians) of the specified circle with the terrain bounds.
    """
    intersections = find_intersections_with_terrain_bounds(center, radius, observed_terrain_size, terrain_padding)

    if len(intersections) != 2:
        return 0.0, 0.0

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
            terrain_padding <= midpoint_x <= observed_terrain_size + terrain_padding and terrain_padding <= midpoint_y <= observed_terrain_size + terrain_padding):
        tmp = theta1
        theta1 = theta2
        theta2 = tmp

    return theta1, theta2


def get_erased_rim_arcs(craters: pd.DataFrame,
                        min_crater_radius: float,
                        effective_radius_multiplier: float) -> pd.DataFrame:
    """
    Returns the erased rim arcs resulting from a supplied sequence of craters.
    effective_radius_multiplier specifies a multiplier on the size of a newly-formed
    crater at removing other craters' rims.
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
        new_radius = new_crater.radius * effective_radius_multiplier

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


def normalize_arcs(arcs: List[Arc]) -> SortedArcList:
    """
    Splits arcs that cross zero into two arcs.
    Assumes all arcs are in range [0, 2*pi]
    """
    result = SortedArcList()
    for arc in arcs:
        if arc[0] > arc[1]:
            result.add((arc[0], 2 * np.pi))
            result.add((0, arc[1]))
        else:
            result.add(arc)

    return result


def merge_arcs(arcs: SortedArcList) -> SortedArcList:
    """
    Merges arcs, accounting for overlaps.
    Assumes that arcs do not cross 0/2*pi
    """
    result = SortedArcList()
    for arc in arcs:
        if not result or result[-1][1] < arc[0]:
            result.add(arc)
        else:
            last = result[-1]
            del result[-1]
            result.add((last[0], max(last[1], arc[1])))

    return result


def calculate_rim_percentage_remaining(erased_arcs: List[Arc]) -> float:
    """
    Calculates the percentage of rim remaining, given a set of erased arcs.
    """
    normalized_arcs = SortedArcList(normalize_arcs(erased_arcs))
    merged_arcs = merge_arcs(normalized_arcs)
    return 1 - sum([x[1] - x[0] for x in merged_arcs]) / (2 * np.pi)
