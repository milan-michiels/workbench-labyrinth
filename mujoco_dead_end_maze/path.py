import numpy as np

path_coords = [
    [-0.4, -0.4],  # path1
    [-0.4, -0.29],  # path2
    [-0.05, -0.29],  # path3
    [-0.05, -0.43],  # path4
    [0.42, -0.43],  # path5
    [0.42, -0.10],  # path6
    [0.20, -0.10],  # path7
    [0.20, 0.07],  # path8
    [-0.17, 0.07],  # path9
    [-0.17, 0.27],  # path10
    [-0.31, 0.27],  # path11
    [-0.31, -0.12],  # path12
    [-0.43, -0.12],  # path13
    [-0.43, 0.42],  # path14
    [0.18, 0.42],  # path15
    [0.18, 0.20],  # path16
    [0.43, 0.20],  # path17
    [0.40, 0.40]  # path18
]


def distance(p1, p2):
    """ Calculate the Euclidean distance between two points. """
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def closest_point_on_segment(px, py, x1, y1, x2, y2):
    """ Find the closest point on a line segment to the given point. """
    # Bereken de vector van het lijnsegment
    segment_vector = np.array([x2 - x1, y2 - y1])
    point_vector = np.array([px - x1, py - y1])

    # Projecteer het punt op het lijnsegment
    segment_length = np.linalg.norm(segment_vector)
    if segment_length == 0:
        return x1, y1  # het lijnsegment is een punt, return het beginpunt

    projection = np.dot(point_vector, segment_vector) / segment_length
    projection = max(0, min(projection, segment_length))  # zorg dat het tussen 0 en de lengte van het segment valt

    closest_point = np.array([x1, y1]) + (projection / segment_length) * segment_vector
    return closest_point[0], closest_point[1]


def get_next_targets(last_known_point, num_next_points):
    """ Get the next num_next_points targets along the path. """
    last_index = find_closest_path_index(last_known_point)
    next_targets = path_coords[last_index + 1:last_index + 1 + num_next_points]
    return next_targets


def find_closest_path_index(point, closest=False):
    """ Find the nearest index point in the path_coords list. """
    if closest:
        index = min(range(len(path_coords)), key=lambda i: distance(path_coords[i], point))
    else:
        index = None
        for i in range(len(path_coords) - 1):
            # Check if point is between two consecutive path coordinates
            print(f"X: {point[0]} Y: {point[1]}")
            print(f"Path X: {path_coords[i][0]} Y: {path_coords[i][1]}")
            print(f"Nex Path X: {path_coords[i + 1][0]} Y: {path_coords[i + 1][1]}")
            if (point[0] == path_coords[i][0] and point[0] == path_coords[i + 1][0]) or (
                    point[1] == path_coords[i][1] and point[1] == path_coords[i + 1][1]):
                index = i + 1
                print(f"Index: {index}")
    return index


def closest_point_on_path(px, py, last_known_point, search_range=1):
    """ Find the closest point on the path to the given point. """
    closest_dist = float('inf')
    closest_point = None

    last_index = find_closest_path_index(last_known_point, closest=True)
    start_index = max(0, last_index - search_range)
    end_index = min(len(path_coords) - 1, last_index + search_range)

    for i in range(start_index, end_index):
        start = path_coords[i]
        end = path_coords[i + 1]

        cx, cy = closest_point_on_segment(px, py, start[0], start[1], end[0], end[1])
        dist = distance((px, py), (cx, cy))

        if dist < closest_dist:
            closest_dist = dist
            closest_point = (cx, cy)

    return closest_point, closest_dist


def distance_along_path(start_point):
    """ Calculate the distance along the path from the given point to the goal. """
    start_index = find_closest_path_index(start_point)

    # Bereken de afstand vanaf het startpunt tot het volgende knooppunt
    next_point = path_coords[start_index + 1]
    segment_dist = distance(start_point, next_point)

    # Bereken de resterende afstand vanaf dat knooppunt tot de goal
    remaining_distance = sum(
        distance(path_coords[i], path_coords[i + 1]) for i in range(start_index + 1, len(path_coords) - 1)
    )

    return segment_dist + remaining_distance
