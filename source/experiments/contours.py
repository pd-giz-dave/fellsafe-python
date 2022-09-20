"""
    Find contours of 4-connected components.
    This module is a re-implementation of blob.h from https://github.com/BlockoS/blob
    Which is a C implementation of:
        "A linear-time component-labeling algorithm using contour tracing technique"
        by Fu Chang, Chun-Jen Chen, and Chi-Jen Lu.
    It has been tweaked to reduce the connectivity searched from 8 to 4 (i.e. direct neighbours only).
    It is also extended to compute the area of the components found.
    Blame the original for any weird looking logic in here!
"""

import math
import numpy as np
from typing import List

# co-ordinates of pixel neighbours relative to that pixel, clockwise from 'east':
#   [5][6][7]
#   [4][-][0]
#   [3][2][1]
dx = [1, 1, 0, -1, -1, -1,  0,  1]       # x-offset
dy = [0, 1, 1,  1,  0, -1, -1, -1]       # .. y-offset

# Reject codes for blobs being ignored (must be sequential)
REJECT_NONE        = 0                   # not rejected (must be zero)
REJECT_UNKNOWN     = 1                   # do not know why it was rejected
REJECT_TOO_SMALL   = 2                   # area below minimum
REJECT_TOO_BIG     = 3                   # area above maximum
REJECT_IRREGULAR   = 4                   # shape too irregular
REJECT_RADIUS      = 5                   # radius too small
REJECT_MARGIN      = 6                   # margin to image edge too small
REJECT_CODE_LIMIT  = 7                   # +1 on last code

BLACK: int = 0
WHITE: int = 255
GREY: int = (WHITE - BLACK) >> 1

class Logger:
    context: List[str] = []

    def __init__(self, context: str):
        self.push(context)

    def log(self, msg: str):
        print("{}: {}".format(self.context[0], msg))

    def push(self, context):
        self.context.insert(0, context)

    def pop(self):
        self.context.pop(0)


class Point:
    x: float = None
    y: float = None

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        return "{:.2f},{:.2f}".format(self.x, self.y)


class Circle:
    centre: Point = None
    radius: float = None

    def __init__(self, centre: Point, radius: float):
        self.centre = centre
        self.radius = radius

    def __str__(self):
        return "centre:{}, radius:{:.2f}".format(self.centre, self.radius)


class Contour:
    points: List[Point] = None
    unique_points: List[Point] = None
    area: float = None
    top_left: Point = None
    bottom_right: Point = None
    external: bool = None

    def __init__(self, external: bool = True):
        self.external = external

    def __str__(self):
        if self.points is None:
            return "None"
        return "start:{}, perimeter:{}, size:{}, inside:{}, area:{:.2f}, circularity:{:.2f}".\
            format(self.points[0], self.get_perimeter(), self.get_size(),
                   self.area, self.get_area(), self.get_circularity())

    def add_point(self, point: Point):
        """ add a point to the contour """
        if self.points is None:
            self.points = [point]
        else:
            self.points.append(point)
        if self.top_left is None:
            self.top_left = Point(point.x, point.y)
        else:
            if point.x < self.top_left.x:
                self.top_left.x = point.x
            if point.y < self.top_left.y:
                self.top_left.y = point.y
        if self.bottom_right is None:
            self.bottom_right = Point(point.x, point.y)
        else:
            if point.x > self.bottom_right.x:
                self.bottom_right.x = point.x
            if point.y > self.bottom_right.y:
                self.bottom_right.y = point.y

    def add_area(self, area: int = 1):
        """ add pixel(s) to the area"""
        if self.area is None:
            self.area = area
        else:
            self.area += area

    def get_unique_points(self):
        # NB: this looses the ordering, but we don't care
        # This is very inefficient, but don't care.
        # If a Point was just a tuple (x, y) it could be done by: list(set(self.points))
        unique_set = set()
        for point in self.points:
            unique_set.add((point.x, point.y))
        unique_points = []
        for item in unique_set:
            unique_points.append(Point(item[0], item[1]))
        return unique_points

    def get_size(self) -> Point:
        width: float = self.bottom_right.x - self.top_left.x + 1
        height: float = self.bottom_right.y - self.top_left.y + 1
        return Point(width, height)

    def get_axis(self):
        size = self.get_size()
        return Point(size.x / 2, size.y / 2)

    def get_area(self) -> float:
        """ when external is False the area is an approximation on the assumption it is circular and enclosed
            when external is True the area is the internal area plus the perimeter minus one (end is same as start)
            """
        if self.points is None:
            return None
        if self.external:
            if self.area is None:
                return self.get_perimeter()
            else:
                return self.area + self.get_perimeter()
        else:
            # internal area does not include the perimeter
            # we have no actual area info, so we just assume its approx circular (valid-ish for our use case)
            circle = self.get_enclosing_circle()
            radius = max(circle.radius - 1, 1)  # -1 to exclude the perimeter
            return math.pi * radius * radius

    def get_ellipse_area(self):
        """ the area assuming the enclosing box covers an ellipse,
            worst case error is if the ellipse is rotated 45 degrees, in which case width and height are approx
            the same and the answer then is the area of a circle, tough!
            """
        axis = self.get_axis()
        return math.pi * axis.x * axis.y

    def get_enclosing_circle(self) -> Circle:
        """ when external is False get (approximate) centre and radius of the maximum enclosed circle
            when external is True get (approximate) centre and radius of the minimum enclosing circle """
        if self.top_left is None or self.bottom_right is None:
            return None
        axis = self.get_axis()
        x: float = self.top_left.x + axis.x
        y: float = self.top_left.y + axis.y
        if self.external:
            r: float = max(axis.x, axis.y)
        else:
            r: float = min(axis.x, axis.y)
        return Circle(Point(x, y), r)

    def get_perimeter(self) -> int:
        if self.points is None:
            return None
        if self.unique_points is None:
            self.unique_points = self.get_unique_points()
        return len(self.unique_points)

    def get_circularity(self, tolerance: float = 1.0, aspect_ratio: float = 0.66) -> float:
        """ get the (approximate) circularity of the contour as (area * tolerance) / ellipse area
            to be considered circular the circularity must be close to one,
            also its aspect ratio must be close to square
            """
        size = self.get_size()
        squareness: float = min(size.x, size.y) / max(size.x, size.y)  # in range 0..1
        if squareness < aspect_ratio:
            # width to height ratio > 3:2, consider that a very weak circle
            return 0.0
        return (self.get_area() * tolerance) / self.get_ellipse_area()


class Blob:
    label: int = None
    external: Contour = None
    internal: List[Contour] = None
    rejected: int = REJECT_UNKNOWN

    def __init__(self, label: int):
        self.label = label
        self.id = id

    def __str__(self):
        msg = "label:{}".format(self.label)
        if self.external is None:
            return msg
        msg = "{}, area:{:.2f}, {}".format(msg, self.get_area(), self.external)
        if self.internal is None:
            return msg
        return "{}, holes:{}".format(msg, len(self.internal))

    def add_contour(self, contour: Contour):
        """ add a contour to the blob, the first contour is the external one,
            subsequent contours are internal,
        """
        if self.external is None:
            self.external = contour
        elif self.internal is None:
            self.internal = [contour]
        else:
            self.internal.append(contour)

    def get_area(self) -> float:
        """ the area is approximated to the area of the external contour
            plus the area of all its internal contours """
        return self.external.get_area()

    def get_circularity(self, hole_tolerance, aspect_ratio) -> float:
        if self.external is None:
            return None
        if self.internal is not None:
            return self.external.get_circularity(1.0 + hole_tolerance, aspect_ratio)
        else:
            return self.external.get_circularity(1.0, aspect_ratio)


class Labels:
    """ label to blob map """
    blobs = None

    def add_label(self, label: int, blob: Blob):
        if self.blobs is None:
            self.blobs = {}
        self.blobs[label] = blob

    def get_blob(self, label: int):
        if self.blobs is None:
            return None
        else:
            return self.blobs[label]


class Targets:
    """ a holder for the parameters required by find_targets and its result """
    source = None                        # the source greyscale image
    binary = None                        # the binarized image
    integration_width: int = 32          # width of integration area as fraction of image width
    integration_height: int = None       # height of integration area as fraction of image height (None==same as width)
    black_threshold: float = 0.01        # make +ve to get more white, -ve to get more black, range +100%..-100%
                                         # NB: Make a small +ve number to ensure totally white stays white
    white_threshold: float = None        # grey/white threshold, None == same as black (i.e. binary)
    direct_neighbours: bool = True       # True == 4-connected, False == 8-connected
    min_area: float = 9
    max_area: float = 200000
    max_hole_tolerance: float = 0.1
    max_aspect_ratio: float = 0.6
    min_roundness: float = 0.84
    max_roundness: float = 1.3           # NB: max poss roundness is 1.273 (4/pi)
    min_radius: float = 2.5
    min_margin: float = 4.0              # min margin to image edge of target in units of target radius
    targets: List[tuple] = None          # the result


def make_binary(source, width: float=8, height: float=None, black: float=15, white: float=None):
    """ create a binary (or tertiary) image of source using an adaptive threshold,
        width is the fraction of the image width to use as the integration area,
        width is the fraction of the image height to use as the integration area (None==same as width)
        black is the % below the average that is considered to be the black/grey boundary,
        white is the % above the average that is considered to be the grey/white boundary,
        white of None means same as black and will yield a binary image,
        the s fraction applies to the minimum of the image width or height.
        See the adaptive-threshold-algorithm.pdf paper for algorithm details.
        """

    # get the image metrics
    max_y = source.shape[0]
    max_x = source.shape[1]
    width_pixels = int(max_x / width)  # we want this to be odd so that there is a centre
    width_plus = max(width_pixels >> 1, 2)  # offset for going forward
    width_minus = width_plus - 1  # offset for going backward
    if height is None:
        height = width
    height_pixels = int(max_y / height)  # we want this to be odd so that there is a centre
    height_plus = max(height_pixels >> 1, 2)  # offset for going forward
    height_minus = height_plus - 1  # offset for going backward

    # make an empty buffer to accumulate our integral in
    integral = np.zeros((max_y, max_x), np.int32)

    # accumulate the image integral
    for y in range(max_y):
        for x in range(max_x):
            if x == 0:
                acc = int(source[y, x])
            else:
                acc += int(source[y, x])
            if y == 0:
                integral[y][x] = acc
            else:
                integral[y][x] = acc + integral[y-1][x]

    # do the threshold on a new image buffer
    binary = np.zeros((max_y, max_x), np.uint8)
    black_limit = (100-black)/100    # convert % to a ratio
    if white is None:
        white_limit = black_limit
    else:
        white_limit = (100+white)/100  # convert % to a ratio
    for x in range(max_x):
        x1 = int(max(x - width_minus, 0))
        x2 = int(min(x + width_plus, max_x - 1))
        for y in range(max_y):
            y1 = int(max(y - height_minus, 0))
            y2 = int(min(y + height_plus, max_y - 1))
            count = int((x2 - x1) * (y2 - y1))  # how many samples in the integration area
            # sum = bottom right (x2,y2) + top left (x1,y1) - top right (x2,y1) - bottom left (x1,y2)
            # where all but bottom right are *outside* the integration window
            acc = integral[y2][x2] + integral[y1][x1] - integral[y1][x2] - integral[y2][x1]
            if (int(source[y, x]) * count) >= (acc * white_limit):
                binary[y, x] = WHITE
            elif (int(source[y, x]) * count) <= (acc * black_limit):
                binary[y, x] = BLACK
            else:
                binary[y, x] = GREY

    return binary


def contour_trace(image, buffer, label: int, x: int, y: int, external: bool = True, direct: bool = True) -> Contour:
    """ follow the contour at x,y in image giving it label in buffer,
        if external is True follow an external contour, else internal,
        if direct is True use 4-connected neighbours else 8,
        both image and buffer must be the same shape, have a zero border and x,y must never be in it
        """
    if external:
        i: int = 7
    else:
        i: int = 3
    if direct:
        offset: int = 1
        i = (i + 1) & 7
    else:
        offset: int = 0
    x0: int = x
    y0: int = y
    xx: int = -1  # 2nd white pixel visited is saved here
    yy: int = -1  # ..
    buffer[y, x] = label
    contour: Contour = Contour(external)
    done: bool = False
    while not done:
        contour.add_point(Point(x0, y0))
        # scan around current pixel in clockwise order starting after last white hit
        j: int = 0
        while j < 8:
            x1: int = x0 + dx[i]
            y1: int = y0 + dy[i]
            if image[y1, x1] == WHITE:
                buffer[y1, x1] = label
                if xx < 0 and yy < 0:
                    xx = x1  # note 2nd white pixel visited
                    yy = y1  # ..
                else:
                    # we are done if we crossed the first 2 contour points again
                    done = (x == x0) and (y == y0) and (xx == x1) and (yy == y1)
                x0 = x1
                y0 = y1
                break
            else:
                buffer[y1, x1] = -1
            j += (1 + offset)
            i = (i + 1 + offset) & 7
        if j == 8:
            # isolated point
            done = True
        else:
            # compute next start position
            previous: int = (i + 4) & 7
            i = (previous + 2) & 7
    return contour


def find_blobs(image, direct: bool = True, debug: bool = False) -> List[Blob]:
    """ find_blobs in the given image,
        returns a list of Blob's or None if failed
    """
    width = image.shape[1]
    height = image.shape[0]
    if width < 3 or height < 3:
        return None
    # allocate a label buffer
    buffer = np.zeros((height, width), np.int32)  # NB: must be signed values
    # put a black border around the image edge 1 pixel wide
    # put a -1 border around the buffer edges 1 pixel wide
    # this means we can scan from 1,1 to max-1,max-1 and not need to bother about the edges
    for x in range(width):
        image[0, x] = 0
        image[height-1, x] = 0
        buffer[0, x] = -1
        buffer[height - 1, x] = -1
    for y in range(height):
        image[y, 0] = 0
        image[y, width-1] = 0
        buffer[y, 0] = -1
        buffer[y, width - 1] = -1
    # scan for start points and follow each
    blobs = []                           # blob list is built in here
    labels = Labels()                    # label to blob map is in here
    label = 0                            # current label (0=background)
    for y in range(1, height-1):         # ignore first and last row (don't care about image edge)
        for x in range(1, width-1):      # ..ditto for first and last column
            # find a start point as the next non-zero pixel
            here_in = image[y, x]
            if here_in == BLACK:
                continue
            above_in = image[y-1, x]
            below_in = image[y+1, x]
            here_label = buffer[y, x]
            before_label = buffer[y, x-1]
            below_label = buffer[y+1, x]
            if here_label == 0 and above_in == BLACK:
                # found a new start
                label += 1               # assign next label
                blob = Blob(label)
                blob.add_contour(contour_trace(image, buffer, label, x, y, external=True, direct=direct))
                blobs.append(blob)
                labels.add_label(label, blob)
            elif below_label == 0 and below_in == BLACK:
                # found a new internal contour (a hole)
                current_blob = labels.get_blob(before_label)
                current_blob.add_contour(contour_trace(image, buffer, before_label, x, y, external=False, direct=direct))
            elif here_label == 0:
                # found an internal element of an external contour
                buffer[y, x] = before_label
                if before_label > 0:
                    # we're inside a component, so up its area
                    current_blob = labels.get_blob(before_label)
                    current_blob.external.add_area()

    if debug:
        return blobs, buffer, labels
    else:
        return blobs


def filter_blobs(blobs: List[Blob],
                 min_area: float, max_area: float,
                 max_hole: float, max_ratio: float,
                 min_round: float, max_round: float,
                 logger = None) -> List[Blob]:
    """ filter out blobs that are too small, too big or not round enough,
        marks *all* blobs with an appropriate reject code and returns a list of good ones
        """
    if logger is not None:
        logger.push("filter_blobs")
    good_blobs = []                      # good blobs are accumulated in here
    for blob in blobs:
        area = blob.get_area()
        if area < min_area:
            reason_code = REJECT_TOO_SMALL
            if logger is not None:
                reason = "area {:.2f} below minimum {:.2f}".format(area, min_area)
        elif area > max_area:
            reason_code = REJECT_TOO_BIG
            if logger is not None:
                reason = "area {:.2f} above maximum {:.2f}".format(area, max_area)
        else:
            roundness = blob.get_circularity(max_hole, max_ratio)
            if roundness < min_round:
                reason_code = REJECT_IRREGULAR
                if logger is not None:
                    reason = "circularity {:.2f} below minimum {:.2f}".format(roundness, min_round)
            elif roundness > max_round:
                reason_code = REJECT_IRREGULAR
                if logger is not None:
                    reason = "circularity {:.2f} above maximum {:.2f}".format(roundness, max_round)
            else:
                reason_code = REJECT_NONE
                good_blobs.append(blob)
                if logger is not None:
                    reason = ""
        blob.rejected = reason_code
        if logger is not None and reason != "":
            logger.log("Rejected:{}, {}".format(reason, blob))
    if logger is not None:
        logger.log("Accepted blobs: {}, rejected {} of {}".format(len(good_blobs), len(blobs) - len(good_blobs), len(blobs)))
        logger.pop()
    return good_blobs


def find_targets(source, params: Targets, logger=None):
    """ given a monochrome image find all potential targets,
        returns a list of targets where each is a tuple of x:float, y:float, radius:float, label:int
        x,y is the pixel address of the centre of the target and radius is its radius in pixels,
        all may be fractional
        label is the label number assigned to the blob the target was detected within,
        NB: The presence of a logger implies we are debugging
        """
    if logger is not None:
        logger.push("find_targets")
    params.source = source
    params.binary = make_binary(params.source,
                                params.integration_width,
                                params.integration_height,
                                params.black_threshold,
                                params.white_threshold)
    if logger is None:
        blobs = find_blobs(params.binary, params.direct_neighbours)
    else:
        blobs, buffer, labels = find_blobs(params.binary, params.direct_neighbours, debug=True)
    passed = filter_blobs(blobs,
                          params.min_area, params.max_area,
                          params.max_hole_tolerance, params.max_aspect_ratio,
                          params.min_roundness, params.max_roundness,
                          logger=logger)
    max_x = params.source.shape[1]
    max_y = params.source.shape[0]
    params.targets = []
    for blob in passed:
        circle = blob.external.get_enclosing_circle()
        if circle.radius < params.min_radius:
            blob.rejected = REJECT_RADIUS
            if logger is not None:
                reason = "radius {:.2f} below minimum {:.2f}".format(circle.radius, params.min_radius)
                logger.log("Rejected:{}, {}".format(reason, blob))
            continue
        margin_x_max = (max_x - circle.centre.x) / circle.radius
        margin_x_min = circle.centre.x / circle.radius
        margin_x = min(margin_x_max, margin_x_min)
        margin_y_max = (max_y - circle.centre.y) / circle.radius
        margin_y_min = circle.centre.y / circle.radius
        margin_y = min(margin_y_max, margin_y_min)
        margin = min(margin_x, margin_y)
        if margin < params.min_margin:
            blob.rejected = REJECT_MARGIN
            if logger is not None:
                reason = "margin {:.2f} below minimum {:.2f}".format(margin, params.min_margin)
                logger.log("Rejected:{}, {}".format(reason, blob))
            continue
        params.targets.append([circle.centre.x, circle.centre.y, circle.radius, blob.label])
    if logger is None:
        return params.targets
    else:
        logger.pop()
        return params.binary, blobs, buffer, labels, params.targets


def get_targets(source, params=None, logger=None) -> List[tuple]:
    """ find targets in the given image using default parameters """
    if params is None:
        params = Targets()  # use defaults
    result = find_targets(source, params, logger)
    return params.targets, params.binary


def _test(src):
    """ ************** TEST **************** """
    import cv2

    logger = Logger("_test")

    logger.log("\nPreparing image:")
    source = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    # Downsize it
    width, height = source.shape
    aspect_ratio = width / height
    new_width = 1152
    new_height = int(new_width * aspect_ratio)
    shrunk = cv2.resize(source, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    logger.log("\nDetecting blobs")
    params = Targets()
    image, blobs, buffer, labels, _ = find_targets(shrunk, params, logger=logger)
    draw = cv2.merge([image, image, image])
    logger.log("\n")
    cv2.imwrite("contours_binary.png", draw)
    logger.log("binary image shown in contours_binary.png")
    max_x = buffer.shape[1]
    max_y = buffer.shape[0]
    draw_bad = cv2.merge([shrunk, shrunk, shrunk])
    draw_good = cv2.merge([shrunk, shrunk, shrunk])
    for x in range(max_x):
        for y in range(max_y):
            label = buffer[y, x]
            if label > 0:
                blob = labels.get_blob(label)
                if blob.rejected == REJECT_NONE:
                    draw_good[y, x] = (0, 255, 0)   # green
                elif blob.rejected == REJECT_IRREGULAR:
                    draw_bad[y, x] = (255, 0, 255)  # magenta
                elif blob.rejected == REJECT_TOO_SMALL:
                    draw_bad[y, x] = (255, 255, 0)  # cyan
                elif blob.rejected == REJECT_TOO_BIG:
                    draw_bad[y, x] = (255, 0, 0)    # blue
                elif blob.rejected == REJECT_RADIUS:
                    draw_bad[y, x] = (0, 255, 255)  # yellow
                elif blob.rejected == REJECT_MARGIN:
                    draw_bad[y, x] = (64, 96, 255)  # orange
                else:
                    draw_bad[y, x] = (0, 0, 255)    # red

    cv2.imwrite("contours_accepted.png", draw_good)
    logger.log("accepted contours shown in contours_accepted.png")
    cv2.imwrite("contours_rejected.png", draw_bad)
    logger.log("rejected contours shown in contours_rejected.png")
    draw = cv2.merge([shrunk, shrunk, shrunk])
    for target in params.targets:
        cv2.circle(draw, (int(round(target[0])), int(round(target[1]))), int(round(target[2])), (0, 255, 0), 1)
    cv2.imwrite("contours_blobs.png", draw)
    logger.log("detected blobs shown in contours_blobs.png")
    logger.log("\n")

    logger.log("\nAll accepted blobs:")
    for b, blob in enumerate(blobs):
        if blob.rejected != REJECT_NONE:
            continue
        logger.log("  {}: {}".format(b, blob))
        if blob.internal is not None:
            for i, contour in enumerate(blob.internal):
                logger.log("    {}: {}".format(i, contour))
    logger.log("\nAll detected targets:")
    for t, target in enumerate(params.targets):
        logger.log("  {}: centre: {:.2f}, {:.2f}  radius: {:.2f}  label: {}".
                   format(t, target[0], target[1], target[2], target[3]))

    logger.log("\nBlob colours:")
    logger.log("    green=good, red=unknown")
    logger.log("    magenta=shape too irregular, cyan=area too small, blue=area too big")
    logger.log("    yellow=radius too small, orange=margin too small")


if __name__ == "__main__":
    #src = "/home/dave/blob-extractor/test/data/checker.png"
    #src = "/home/dave/blob-extractor/test/data/diffract.png"
    #src = "/home/dave/blob-extractor/test/data/dummy.png"
    #src = "/home/dave/blob-extractor/test/data/labyrinth.png"
    #src = "/home/dave/blob-extractor/test/data/lines.png"
    #src = "/home/dave/blob-extractor/test/data/simple.png"
    #src = "/home/dave/precious/fellsafe/fellsafe-image/codes/test-code-101.png"
    src = "/home/dave/precious/fellsafe/fellsafe-image/media/lead-head-ratio-codes/photo-101.jpg"
    #src = "/home/dave/precious/fellsafe/fellsafe-image/projected-101.png"
    #src = "/home/dave/precious/fellsafe/fellsafe-image/media/old-codes/photo-101.jpg"
    #src = "/home/dave/precious/fellsafe/fellsafe-image/media/old-codes/photo-101-102-182-247-301-424-448-500-537-565-v2.jpg"
    #src = "/home/dave/precious/fellsafe/fellsafe-image/media/old-codes/photo-101-102-182-247-301-424-448-500-537-565-v5.jpg"
    #src = "/home/dave/precious/fellsafe/fellsafe-image/media/photo-332-222-555-800-574-371-757-611-620-132.jpg"
    #src = "/home/dave/precious/fellsafe/fellsafe-image/media/photo-332-222-555-800-574-371-757-611-620-132-mid.jpg"
    #src = "/home/dave/precious/fellsafe/fellsafe-image/media/photo-332-222-555-800-574-371-757-611-620-132-distant.jpg"
    #src = "/home/dave/precious/fellsafe/fellsafe-image/media/photo-332-222-555-800-574-371-757-611-620-132-near.jpg"
    #src = "/home/dave/precious/fellsafe/fellsafe-image/media/photo-332-222-555-800-574-371-757-611-620-132-near-blurry.jpg"
    # src = "targets.jpg"

    # shape = [[0,0,0,0,0,0,0,0,0,0],
    #          [0,0,0,0,0,0,0,0,0,0],
    #          [0,1,1,0,0,0,0,1,1,0],
    #          [0,1,0,1,1,1,1,0,1,0],
    #          [0,0,1,1,1,1,1,0,0,0],
    #          [0,0,1,1,1,1,1,0,0,0],
    #          [0,0,1,1,1,1,1,0,0,0],
    #          [0,0,1,1,1,1,0,1,0,0],
    #          [0,0,0,0,0,0,1,1,0,0],
    #          [0,0,0,0,0,0,0,0,0,0]]
    # image = np.zeros((len(shape), len(shape[0])), np.uint8)
    # for y, row in enumerate(shape):
    #     for x, pixel in enumerate(row):
    #         image[y, x] = pixel * 255
    # blobs, buffer, labels = find_blobs(image)

    _test(src)
