import math
import random
import time
import turtle

from pandas import *


def traveling_salesman(distance_matrix):
    """
    Calculates minimum cost circuit with dynamic programming approach
    :param distance_matrix: 2D matrix of distance costs to each point
    :return: list of optimum path nodes and the minimum distance cost
    """
    # Number of nodes
    n = len(distance_matrix)
    # Set containing all nodes
    set_of_nodes = set(range(n))

    # dictionary containing all subproblem solutions
    dict_of_subproblems = {(tuple([node]), node): tuple([0, None]) for node in set_of_nodes}

    # queue of subproblems to solve, starting at node 0
    queue = [((0,), 0)]

    while len(queue) > 0:
        # Take next problem to solve from queue
        already_visited_nodes, old_previous_node = queue.pop(0)

        # Get previous best distance from dictionary
        prev_dist, _ = dict_of_subproblems[(already_visited_nodes, old_previous_node)]

        # Create set of nodes to visit
        nodes_to_visit = set_of_nodes.difference(set(already_visited_nodes))

        for new_previous_node in nodes_to_visit:
            # Mark that each node in nodes_to_visit has been already visited
            new_visited = tuple(sorted(list(already_visited_nodes) + [new_previous_node]))

            # Previous best distance stored in dict + matrix lookup for distance from previous to new node
            new_dist = (prev_dist + distance_matrix[old_previous_node][new_previous_node])

            # If subproblem not solved, add to the queue to solve it
            if (new_visited, new_previous_node) not in dict_of_subproblems:
                dict_of_subproblems[(new_visited, new_previous_node)] = (new_dist, old_previous_node)
                queue += [(new_visited, new_previous_node)]

            # If new distance is better, update previously stored solution
            else:
                if new_dist < dict_of_subproblems[(new_visited, new_previous_node)][0]:
                    dict_of_subproblems[(new_visited, new_previous_node)] = (new_dist, old_previous_node)

    path, cost = retrace_optimal_path(dict_of_subproblems, n, distance_matrix)
    return path, cost


def retrace_optimal_path(dict_of_subproblems: dict, n: int, dist_matrix: [[int]]) -> [[int], float]:
    """
    Solves backwards through stored solutions of subproblems
    :param dict_of_subproblems: stored solutions of subproblems
    :param n: number of nodes
    :param dist_matrix: 2D matrix of distance costs to each point
    :return: optimal path, optimal cost
    """
    node_set = tuple(range(n))

    # Finds all penultimate subproblem solutions and stores them in a dictionary
    penultimate_path_dict = dict((visited_nodes, previous_node) for visited_nodes, previous_node in
                                 dict_of_subproblems.items() if visited_nodes[0] == node_set)

    # List of keys to penultimate_path_dict
    path_keys = list(penultimate_path_dict.keys())

    total_distance_list = []
    # Calculate total distances from penultimate node back to 0
    for path in penultimate_path_dict:
        penultimate_distance = penultimate_path_dict[path][0]
        distance_to_0 = dist_matrix[path[1]][0]
        total_distance = penultimate_distance + distance_to_0
        total_distance_list.append(total_distance)

    # Get path key of shortest path
    final_distance = min(total_distance_list)
    path_key = path_keys[total_distance_list.index(final_distance)]

    # Start with penultimate node (since last node, 0, is known)
    penultimate_node = path_key[1]
    # Get node before penultimate node from it's associated dictionary value
    _, pre_penultimate_node = dict_of_subproblems[path_key]

    final_path = [penultimate_node]
    # node_set gets penultimate node removed. Used as part of key for dict lookup
    node_set = tuple(sorted(set(node_set).difference({penultimate_node})))

    # Keeps looking up best cost backwards until node_set is empty
    while pre_penultimate_node is not None:
        penultimate_node = pre_penultimate_node
        path_key = (node_set, penultimate_node)
        _, pre_penultimate_node = dict_of_subproblems[path_key]

        # Prepends node to final path list
        final_path = [penultimate_node] + final_path
        # Remove node from set of remaining nodes to backtrace through
        node_set = tuple(sorted(set(node_set).difference({penultimate_node})))

    final_path.append(0)

    return final_path, final_distance


def create_random_points(n):
    x_range = (0, n * 3)
    y_range = (0, n * 3)

    random_points = set({})
    i = 0
    while i < n:
        x = random.randrange(*x_range)
        y = random.randrange(*y_range)
        point = (x, y)
        random_points.add(point)
        i = len(random_points)
    return list(random_points)


def create_matrix(points):
    """
    Creates a matrix from a list of point tuples
    :param points: list of x, y point tuples
    :return: 2D matrix of distances between points
    """
    number_of_points = int(len(points))
    distance_matrix = [[0 for _ in range(number_of_points)] for _ in range(number_of_points)]
    for i in range(number_of_points - 1):
        for j in range(i + 1, number_of_points):
            point1 = points[i]
            point2 = points[j]
            distance_matrix[i][j] = dist(point1, point2)
            distance_matrix[j][i] = distance_matrix[i][j]

    return distance_matrix


def dist(point1, point2):
    """
    Computes integer distance between two points
    :param point1: x, y point tuple
    :param point2: x, y point tuple
    :return: integer distance value
    """
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    return math.sqrt(delta_x ** 2 + delta_y ** 2)


def visualize(points, path, cost, r_time):
    """
    Creates a window to display points and the path that traverses them
    :param points: list of points
    :param path: list of optimal path
    :param cost: optimal cost
    :param r_time: total runtime
    """
    screen = turtle.Screen()
    draw = turtle.Turtle(visible=False)

    screen.colormode(255)
    small, big = extreme_points(points)
    screen.setworldcoordinates(small[0] - 10, small[1] - 10, big[0] + 10, big[1] + 10)

    draw.pensize(3)
    draw.speed(0)
    draw.penup()

    color_change = int((255 / len(points)) - .5)
    r = 255
    g = 0

    draw.goto(small[0] - 5, small[1] - 5)
    draw.write(f"Optimal Cost: {cost}\nAlgorithm Runtime: {r_time}", font=("Arial", 16, "normal",))

    for i, point in enumerate(points):
        draw.goto(point[0], point[1])
        draw.dot(10)
        draw.write(f"  {i}", font=("Arial", 16, "normal"))

    draw.speed(6)

    for node in path:
        draw.goto(points[node])
        draw.pendown()
        color = (r, g, 0)
        draw.pencolor(color)
        r -= color_change
        g += color_change

    turtle.Screen().exitonclick()
    return


def extreme_points(points):
    """
    Finds the extreme x and y values in a set of points, and returns them as min and max points
    :param points: list of x, y points
    :return: two tuple points, min_point and max_point
    """
    min_x = max_x = points[0][0]
    min_y = max_y = points[0][1]
    for point in points:
        if point[0] < min_x:
            min_x = point[0]
        if point[0] > max_x:
            max_x = point[0]

        if point[1] < min_y:
            min_y = point[1]
        if point[1] > max_y:
            max_y = point[1]

    min_point = (min_x, min_y)
    max_point = (max_x, max_y)

    return min_point, max_point


print()
print("Input problem size:")
print("(Problem sizes under 20 recommended)")
size = int(input())
coordinates = create_random_points(size)

matrix = create_matrix(coordinates)

# Print the matrix with formatting
mtrx = DataFrame(matrix)
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', None)
pandas.set_option('display.max_colwidth', None)
print(DataFrame(mtrx).round(2))

start = time.time()
optimal_path, optimal_cost = traveling_salesman(matrix)
runtime = round(time.time() - start, 3)

print()
print(f"Algorithm runtime: {runtime} seconds.")
print(f"Optimal cost: {round(optimal_cost, 3)}")
print(f"Optimal path: {optimal_path}")

visualize(coordinates, optimal_path, optimal_cost, runtime)
