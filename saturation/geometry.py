from typing import Tuple

import numpy as np

from saturation.simulation import Location


# Type definitions
# Arc in radians
Arc = Tuple[float, float]


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
                         center2: Location,
                         r1: float,
                         r2: float) -> Arc:
    """
    Returns the intersection arc (in radians) of the circle defined by center2 and r2 on the circle
    defined by center1 and r1.
    """
    # Circle 1 is completely encompassed by circle 2
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
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
    test_value = (center2[0] - midpoint_x)**2 + (center2[1] - midpoint_y)**2

    # If our test point is not within circle2, reverse our thetas
    if test_value >= r2**2:
        tmp = theta1
        theta1 = theta2
        theta2 = tmp

    return theta1, theta2
