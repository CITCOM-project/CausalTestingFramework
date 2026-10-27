"""
An implementation of the Poisson line process from R. Guderlei and J. Mayer,
"Statistical Metamorphic Testing Testing Programs with Random Output by Means of
Statistical Hypothesis Tests and Metamorphic Testing," Seventh International
Conference on Quality Software (QSIC 2007), Portland, OR, USA, 2007,
pp. 404-409, doi: 10.1109/QSIC.2007.4385527.
"""

import numpy as np
from numpy.random import default_rng
from scipy.stats import uniform
from shapely.geometry import MultiLineString, MultiPolygon, Polygon


def realize_poisson_line_process(width=1, height=1, intensity=1, orientation=uniform(loc=0, scale=np.pi)):
    """Draw a realization from a poisson line process
    inside a rectangle centered around (0,0) with given
    extent, intensity and orientation distributions.

    The pairs of distances of the lines to the origin and
    their orientations are drawn from a poisson process
    on [0,infty) x [0,2*pi).

    Parameters
    ----------
    width : float, optional
        The width of the rectangle, default is 1.0
    height : float, optional
        The height of the rectangle, default is 1.0
    intensity : float, optional
        The intensity of the poisson line process,
        default is 1.0.
    orientation : {`scipy.stats.rv_continuous`,
                   `scipy.stats.rv_discrete`}, optional
        The line orientation distribution, default
        is uniform distribution on [0, pi)

    Returns
    -------
    lines : list
        list of poisson lines which intersect the given
        rectangle. Each line is given as a pair of
        ``(slope, intercept)``.
    """
    if intensity == 0 or width == 0 or height == 0:
        return []

    rng = default_rng()

    # points within inner_radius are always inside the rectangle
    inner_radius = min(width / 2, height / 2)

    # points outside outer_radius are never inside the rectangle
    outer_radius = 0.5 * np.sqrt(width * width + height * height)

    lines = []

    distance = 0

    # radially simulate the distances between the lines and the origin
    while distance < outer_radius:
        distance -= np.log(rng.random()) / (2 * np.pi * intensity)

        # Draw alpha from the given orientation distribution.
        alpha = orientation.rvs()

        if rng.random() > 0.5:
            alpha = alpha + np.pi

        current_line = convert_line_representation(distance, alpha)

        # Check if lines between inner_radius and outer_radius intersect the rectangle
        if distance > inner_radius:
            if current_line[0] > 0:
                # Positive gradient => Check if lower right and upper left
                # corner are on the same side of the line
                side_1 = side(current_line, (width / 2, -height / 2))
                side_2 = side(current_line, (-width / 2, height / 2))
            else:
                # Negative gradient => Check if upper right and lower left left
                # corner are on the same side of the line
                side_1 = side(current_line, (width / 2, height / 2))
                side_2 = side(current_line, (-width / 2, -height / 2))
            if side_1 == side_2:
                continue

        lines.append(current_line)

    return lines


def convert_line_representation(distance_to_origin, angle):
    """
    Convert the representation of a line given by the distance to the origin and
    the angle to y=f(x)=m*x+t
    """
    m = -np.cos(angle) / np.sin(angle)
    x = distance_to_origin * np.cos(angle)
    y = distance_to_origin * np.sin(angle)
    t = y - m * x
    return m, t


def side(line, point):
    """
    Determines on which side (relative to the directional vector (dx,dy) of the
    given line the point p lies. We assume that line is a tuple of m and t
    (f(x)=m*x+t).
    """
    m, t = line
    px, py = point
    return py - (m * px + t) > 0


def detect_shapes(lines, width, height):
    """
    Detect shapes formed by intersecting lines in the sampling window.
    """
    high_x = width / 2
    low_x = -high_x
    high_y = height / 2
    low_y = -high_y
    line_pairs = []
    for current_line in lines:
        y_min = current_line[0] * low_x + current_line[1]
        y_max = current_line[0] * high_x + current_line[1]
        line_pairs.append(((low_x, y_min), (high_x, y_max)))

    mls = MultiLineString(line_pairs)
    # Convert Lines to Polygons by applying a tiny buffer
    mls = mls.buffer(0.0000000000001)
    # Get outer boundary of the lines as a polygon
    boundary = mls.convex_hull
    # Get a difference to generate a multipolygon
    multipolygons = boundary.difference(mls)

    area = Polygon(
        [
            [low_x, low_y],
            [high_x, low_y],
            [high_x, high_y],
            [low_x, high_y],
            [low_x, low_y],
        ]
    )
    if isinstance(multipolygons, Polygon):
        polygon = multipolygons.intersection(area)
        if polygon.is_empty:
            return []
        return [polygon]

    polygons = []
    for polygon in multipolygons.geoms:
        polygon = polygon.intersection(area)
        if polygon.is_empty:
            continue
        if isinstance(polygon, Polygon):
            polygons.append(polygon)
        elif isinstance(polygon, MultiPolygon):
            polygons += list(polygon.geoms)

    return polygons


def run(width, height, intensity):
    """
    Run the Poisson process for the given width and hight sampling window with
    the given intensity.
    """
    orientation_distribution = uniform(loc=0, scale=np.pi)
    lines = realize_poisson_line_process(width, height, intensity, orientation_distribution)
    polygons = detect_shapes(lines, width, height)
    num_lines = len(lines)
    num_polygons = len(polygons)
    area = width * height
    return (num_lines, num_polygons, num_lines / area, num_polygons / area)
