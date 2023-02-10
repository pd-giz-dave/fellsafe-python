""" chatGPT suggested code!
    question was: "write a python function for 2d k-d tree"
    suggested code is actually dimension agnostic
"""

class Node:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right

def build_kdtree(points, depth=0):
    if not points:
        return None

    k = len(points[0])  # k is dimension of points
    axis = depth % k  # axis alternates as depth increases
    points.sort(key=lambda x: x[axis])  # O(n.log(n)) but n is shrinking as we recurse down
    median = len(points) >> 1  # chatGPT suggested "// 2" rather than ">> 1"

    return Node(point=points[median],
                left=build_kdtree(points[:median], depth+1),  # recurse on first half
                right=build_kdtree(points[median+1:], depth+1))  # recurse on second half
