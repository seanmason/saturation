import datetime
import random
import collections
from itertools import islice

import numpy as np

from data_structures import kdtree
from data_structures.kdtree import KDNode


def random_tree(nodes=20):
    points = list(islice(random_points(), 0, nodes))
    tree = kdtree.create(points)
    return tree


def random_point(dimensions=3, minval=0, maxval=100):
    return tuple(random.randint(minval, maxval) for _ in range(dimensions))


def random_points(dimensions=3, minval=0, maxval=100):
    while True:
        yield random_point(dimensions, minval, maxval)


def do_random_add():
    points = list(set(islice(random_points(), 0, 10)))
    tree = kdtree.create(dimensions=len(points[0]))
    for n, point in enumerate(points, 1):
        tree.add(point)

        assert tree.is_valid()
        assert point in [node.data for node in tree.inorder()]

        nodes_in_tree = len(list(tree.inorder()))
        assert nodes_in_tree == n


def find_best(tree, point):
    best = None
    best_dist = None
    for p in tree.inorder():
        dist = p.dist(point)
        if best is None or dist < best_dist:
            best = p
            best_dist = dist
    return best, best_dist


def test_remove_duplicates():
    """ creates a tree with only duplicate points, and removes them all """
    points = [(1, 1)] * 100
    tree = kdtree.create(points)
    assert tree.is_valid()

    random.shuffle(points)
    while points:
        point = points.pop(0)

        tree = tree.remove(point)

        # Check if the Tree is valid after the removal
        assert tree.is_valid()

        # Check if the removal reduced the number of nodes by 1 (not more, not less)
        remaining_points = len(points)
        nodes_in_tree = len(list(tree.inorder()))
        assert nodes_in_tree == remaining_points


def test_remove():
    """ Tests random removal from a tree, multiple times """
    for i in range(10):
        do_random_remove()


def do_random_remove():
    """ Creates a random tree, removes all points in random order """
    points = list(set(islice(random_points(), 0, 20)))
    tree = kdtree.create(points)
    assert tree.is_valid()

    random.shuffle(points)
    while points:
        point = points.pop(0)

        tree = tree.remove(point)

        # Check if the Tree is valid after the removal
        assert tree.is_valid()

        # Check if the point has actually been removed
        assert point not in [n.data for n in tree.inorder()]

        # Check if the removal reduced the number of nodes by 1 (not more, not less)
        remaining_points = len(points)
        nodes_in_tree = len(list(tree.inorder()))
        assert nodes_in_tree == remaining_points


def test_remove_empty_tree():
    tree = kdtree.create(dimensions=2)
    tree.remove((1, 2))
    assert not bool(tree)


def test_add():
    """ Tests random additions to a tree, multiple times """

    for i in range(10):
        do_random_add()


def test_invalid_child():
    """ Children on wrong subtree invalidate Tree """
    child = kdtree.KDNode((3, 2))
    child.axis = 2
    tree = kdtree.create([(2, 3)])
    tree.left = child
    assert not tree.is_valid()

    tree = kdtree.create([(4, 1)])
    tree.right = child
    assert not tree.is_valid()


def test_different_dimensions():
    """ Can't create Tree for Points of different dimensions """
    points = [(1, 2), (2, 3, 4)]
    try:
        kdtree.create(points)
        assert False
    except:
        pass


def test_same_length():
    tree = random_tree(nodes=10)

    inorder_len = len(list(tree.inorder()))

    assert inorder_len == 10


def test_rebalance():
    tree = random_tree(1)
    while tree.is_balanced:
        tree.add(random_point())

    tree = tree.rebalance()
    assert tree.is_balanced


def test_search_knn():
    points = [(50, 20), (51, 19), (1, 80)]
    tree = kdtree.create(points)
    point = (48, 18)

    all_dist = []
    for p in tree.inorder():
        dist = p.dist(point)
        all_dist.append([p, dist])

    all_dist = sorted(all_dist, key=lambda n: n[1])

    result = tree.search_knn(point, 1)
    assert result[0][1] == all_dist[0][1]

    result = tree.search_knn(point, 2)
    assert result[0][1] == all_dist[0][1]
    assert result[1][1] == all_dist[1][1]

    result = tree.search_knn(point, 3)
    assert result[0][1] == all_dist[0][1]
    assert result[1][1] == all_dist[1][1]
    assert result[2][1] == all_dist[2][1]


def test_search_nn():
    points = list(islice(random_points(), 0, 10))
    tree = kdtree.create(points)
    point = random_point()

    nn, dist = tree.search_nn(point)
    best, best_dist = find_best(tree, point)
    assert best_dist == dist


def test_search_nn2():
    points = [(1, 2, 3), (5, 1, 2), (9, 3, 4), (3, 9, 1), (4, 8, 3), (9, 1, 1), (5, 0, 0),
              (1, 1, 1), (7, 2, 2), (5, 9, 1), (1, 1, 9), (9, 8, 7), (2, 3, 4), (4, 5, 4.01)]
    tree = kdtree.create(points)
    point = (2, 5, 6)

    nn, dist = tree.search_nn(point)
    best, best_dist = find_best(tree, point)
    assert best_dist == dist


def test_search_nn3():
    points = [(0, 25, 73), (1, 91, 85), (1, 47, 12), (2, 90, 20),
              (2, 66, 79), (2, 46, 27), (4, 48, 99), (5, 73, 64), (7, 42, 70),
              (7, 34, 60), (8, 86, 80), (10, 27, 14), (15, 64, 39), (17, 74, 24),
              (18, 58, 12), (18, 58, 5), (19, 14, 2), (20, 88, 11), (20, 28, 58),
              (20, 79, 48), (21, 32, 8), (21, 46, 41), (22, 6, 4), (22, 42, 68),
              (22, 62, 42), (24, 70, 96), (27, 77, 57), (27, 47, 39), (28, 61, 19),
              (30, 28, 22), (34, 13, 85), (34, 39, 96), (34, 90, 32), (39, 7, 45),
              (40, 61, 53), (40, 69, 50), (41, 45, 16), (41, 15, 44), (42, 40, 19),
              (45, 6, 68), (46, 79, 91), (47, 91, 86), (47, 50, 24), (48, 57, 64),
              (49, 21, 72), (49, 87, 21), (49, 41, 62), (54, 94, 32), (56, 14, 54),
              (56, 93, 2), (58, 34, 44), (58, 27, 42), (59, 62, 80), (60, 69, 69),
              (61, 67, 35), (62, 31, 50), (63, 9, 93), (63, 46, 95), (64, 31, 2),
              (64, 2, 36), (65, 23, 96), (66, 94, 69), (67, 98, 10), (67, 40, 88),
              (68, 4, 15), (68, 1, 6), (68, 88, 72), (70, 24, 53), (70, 31, 87),
              (71, 95, 26), (74, 80, 34), (75, 59, 99), (75, 15, 25), (76, 90, 99),
              (77, 75, 19), (77, 68, 26), (80, 19, 98), (82, 90, 50), (82, 87, 37),
              (84, 88, 59), (85, 76, 61), (85, 89, 20), (85, 64, 64), (86, 55, 92),
              (86, 15, 69), (87, 48, 46), (87, 67, 47), (89, 81, 65), (89, 87, 39),
              (89, 87, 3), (91, 65, 87), (94, 37, 74), (94, 20, 92), (95, 95, 49),
              (96, 15, 80), (96, 27, 39), (97, 87, 32), (97, 43, 7), (98, 78, 10),
              (99, 64, 55)]

    tree = kdtree.create(points)
    point = (66, 54, 29)

    nn, dist = tree.search_nn(point)
    best, best_dist = find_best(tree, point)
    assert best_dist == dist


def test_get_nn_dist():
    points = [
        (1, 1),
        (2, 2),
        (3.1, 3.1)
    ]
    tree = kdtree.create(points)

    dist = tree.get_nn_dist((1, 2))
    assert dist == 1


def test_search_nn_dist():
    """ tests search_nn_dist() according to bug #8 """

    points = [(x, y) for x in range(10) for y in range(10)]
    tree = kdtree.create(points)
    nn = tree.search_nn_dist((5, 5), 2.5)

    assert len(nn) == 9
    assert (4, 4) in nn
    assert (4, 5) in nn
    assert (4, 6) in nn
    assert (5, 4) in nn
    assert (6, 4) in nn
    assert (6, 6) in nn
    assert (5, 5) in nn
    assert (5, 6) in nn
    assert (6, 5) in nn


def test_search_nn_dist2():
    """ Test case from #36 """
    points = [[0.25, 0.25, 1.600000023841858], [0.75, 0.25, 1.600000023841858], [1.25, 0.25, 1.600000023841858],
              [1.75, 0.25, 1.600000023841858], [2.25, 0.25, 1.600000023841858], [2.75, 0.25, 1.600000023841858]]

    expected = [0.25, 0.25, 1.600000023841858]
    tree = kdtree.create(points)
    rmax = 1.0
    search_p = [0.42621034383773804, 0.18793821334838867, 1.44510018825531]
    results = tree.search_nn_dist(search_p, rmax)
    found = False
    for result in results:
        if result == expected:
            found = True
            break
    assert found


def test_search_nn_dist3():
    """ Test case from #36 """
    points_list = [
        (0.25, 0.25, 1.600000023841858),
        (0.75, 0.25, 1.600000023841858),
        (1.25, 0.25, 1.600000023841858),
        (1.75, 0.25, 1.600000023841858),
        (2.25, 0.25, 1.600000023841858),
        (2.75, 0.25, 1.600000023841858),
    ]

    tree = kdtree.create(points_list)
    point = (0.42621034383773804, 0.18793821334838867, 1.44510018825531)

    points = tree.inorder()
    points = sorted(points, key=lambda p: p.dist(point))

    for p in points:
        dist = p.dist(point)
        nn = tree.search_nn_dist(point, dist)

        for pn in points:
            if pn in nn:
                msg = '{} in {} but {} < {}'.format(
                    pn, nn, pn.dist(point), dist)
                assert pn.dist(point) < dist
            else:
                msg = '{} not in {} but {} >= {}'.format(
                    pn, nn, pn.dist(point), dist)
                assert pn.dist(point) >= dist


def test_search_nn_dist_random():
    for n in range(50):
        tree = random_tree()
        point = random_point()
        points = tree.inorder()

        points = sorted(points, key=lambda p: p.dist(point))

        for p in points:
            dist = p.dist(point)
            nn = tree.search_nn_dist(point, dist)

            for pn in points:
                if pn in nn:
                    assert pn.dist(point) < dist
                else:
                    assert pn.dist(point) >= dist


def test_point_types():
    point1 = (2, 3, 4)
    point2 = [4, 5, 6]
    Point = collections.namedtuple('Point', 'x y z')
    point3 = Point(5, 3, 2)
    tree = kdtree.create([point1, point2, point3])
    res, dist = tree.search_nn((1, 2, 3))

    assert res == kdtree.KDNode((2, 3, 4))


def test_payload():
    points = list(islice(random_points(dimensions=3), 0, 100))
    tree = kdtree.create(dimensions=3)

    for i, p in enumerate(points):
        tree.add(p).payload = i

    for i, p in enumerate(points):
        assert i == tree.search_nn(p)[0].payload


# def test_load():
#     """
#     Test how long it takes to add, remove, and search for nodes.
#     """
#     N_NODES = 1000
#     N_ITERATIONS = 1
#
#     start = datetime.datetime.now()
#     for i in range(N_ITERATIONS):
#         tree = KDNode(
#             data=(100, 100, 1.2345),
#             sel_axis=lambda prev_axis: (prev_axis + 1) % 2,
#             axis=0,
#             dimensions=2
#         )
#
#         points = (np.random.random(size=(N_NODES, 2)) * 200).astype("int")
#         radii = (np.random.random(size=N_NODES) * 1000).astype("int")
#         for x in range(N_NODES):
#             tree.add((points[x][0], points[x][1], radii[x]))
#
#             if x % 5 == 1:
#                 r = np.random.choice(list(range(x)))
#                 tree.remove((points[r][0], points[r][1], radii[r]))
#
#         for x in range(N_NODES):
#             tree.search_knn((points[x][0], points[x][1], radii[x]),
#                             100,
#                             filter_func=lambda origin, dest: origin[2] + dest[2] < np.sqrt((origin[1] - dest[1]) ** 2 + (origin[0] - dest[0]) ** 2))
#
#     elapsed = datetime.datetime.now() - start
#     print(elapsed)
#     print(len(list(tree.inorder())))
#     print(tree.left.height())
#     print(tree.right.height())
