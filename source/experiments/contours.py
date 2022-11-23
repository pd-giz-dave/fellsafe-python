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

# Reject codes for blobs being ignored
REJECT_NONE            = 'accepted'
REJECT_UNKNOWN         = 'unknown'
REJECT_TOO_FEW_POINTS  = 'too few contour points'
REJECT_TOO_MANY_POINTS = 'too many contour points'
REJECT_TOO_SMALL       = 'area below minimum'
REJECT_TOO_BIG         = 'area above maximum'
REJECT_ROUNDNESS       = 'not round enough'
REJECT_RADIUS          = 'radius too small'
REJECT_MARGIN          = 'margin to image edge too small'
REJECT_INTERNALS       = 'too many internal contours'
REJECT_WHITENESS       = 'not enough white'
REJECT_SQUARENESS      = 'not square enough'
REJECT_WAVYNESS        = 'perimeter too wavy'
REJECT_QUALITY         = 'quality too low'

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

    def __init__(self, x: float, y: float):
        self.x: float = x
        self.y: float = y

    def __str__(self):
        return "({:.2f}, {:.2f})".format(self.x, self.y)


class Circle:

    def __init__(self, centre: Point, radius: float):
        self.centre = centre
        self.radius = radius

    def __str__(self):
        return "(centre:{}, radius:{:.2f}, area:{:.2f})".format(self.centre, self.radius, self.area())

    def area(self):
        return math.pi * self.radius * self.radius


class Contour:
    """ properties of a contour and methods to create/access them,
        NB: most of the metrics of a contour are in the range 0..1 where 0 is good and 1 is very bad
        """

    def __init__(self, external: bool = True):
        self.external: bool = external
        self.points: [Point] = None  # points that make up the contour (NB: contours are a 'closed' set of points)
        self.top_left: Point = None
        self.bottom_right: Point = None
        self.reset()

    def __str__(self):
        return self.show()

    def reset(self):
        """ reset the cached stuff """
        self.perimeter: {Point} = None
        self.x_slices: [tuple] = None
        self.y_slices: [tuple] = None
        self.area: float = None
        self.circle: Circle = None

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

    def show(self, verbose: bool = False, prefix: str = '    '):
        """ produce a string describing the contour for printing purposes,
            if verbose is True a multi-line response is made that describes all properties,
            lines after the first are prefixed by prfix
            """
        if self.points is None:
            return "None"
        first_line = 'box:{}..{}, size:{}, area:{:.2f}, wavyness:{:.2f}, external:{}'.\
                     format(self.top_left, self.bottom_right, self.get_size(),
                            self.get_area(), self.get_wavyness(), self.external)
        if not verbose:
            return first_line
        second_line = 'start:{}, points:{}, perimeter:{}, squareness:{:.2f}, roundness:{:.2f}, circle:{}'.\
                      format(self.points[0], len(self.points), len(self.get_perimeter()), self.get_squareness(),
                             self.get_roundness(), self.get_enclosing_circle())
        return '{}\n{}{}'.format(first_line, prefix, second_line)

    def get_size(self) -> Point:
        """ the size is the maximum width and height of the contour """
        width: float = self.bottom_right.x - self.top_left.x + 1
        height: float = self.bottom_right.y - self.top_left.y + 1
        return Point(width, height)

    def get_x_slices(self):
        """ get the slices in x array,
            for every x co-ord find the y extent at that x,
            this function is lazy
            """
        if self.x_slices is not None:
            # already been done
            return self.x_slices
        x_slices = dict()
        for point in self.points:
            if x_slices.get(point.x) is None:
                x_slices[point.x] = dict()
            x_slices[point.x][point.y] = True
        self.x_slices = []
        for x in x_slices:
            min_y = None
            max_y = None
            for y in x_slices[x]:
                if min_y is None or y < min_y:
                    min_y = y
                if max_y is None or y > max_y:
                    max_y = y
            self.x_slices.append((x, min_y, max_y))
        return self.x_slices

    def get_y_slices(self):
        """ get the slices in y array,
            for every y co-ord find the x extent at that y,
            this function is lazy
            """
        if self.y_slices is not None:
            # already been done
            return self.y_slices
        y_slices = dict()
        for point in self.points:
            if y_slices.get(point.y) is None:
                y_slices[point.y] = dict()
            y_slices[point.y][point.x] = True
        self.y_slices = []
        for y in y_slices:
            min_x = None
            max_x = None
            for x in y_slices[y]:
                if min_x is None or x < min_x:
                    min_x = x
                if max_x is None or x > max_x:
                    max_x = x
            self.y_slices.append((y, min_x, max_x))
        return self.y_slices

    def get_perimeter(self):
        """ get the unique contour perimeter points,
            this function is lazy
            """
        if self.perimeter is not None:
            return self.perimeter
        self.perimeter = {}
        for point in self.points:
            self.perimeter[(point.x, point.y)] = True  # NB: do NOT use point as the key, its an object not a tuple
        return self.perimeter

    def get_circle_area(self):
        """ get the area of the enclosing circle """
        circle = self.get_enclosing_circle()
        return circle.area()

    def get_enclosing_circle(self) -> Circle:
        """ the enclosing circle of a contour is the centre and the radius required to cover (most of) it,
            the centre of the blob is calculated as the average x and y of its points, centre accuracy is
            critical when translating from polar to cartesian co-ordinates, the centre x,y are fractional
            when external is False get (approximate) centre and radius of the maximum enclosed circle
            when external is True get (approximate) centre and radius of the minimum enclosing circle,
            this is an expensive operation so its lazy, calculated on demand and then cached
            """
        if self.circle is not None:
            return self.circle
        x_slices = self.get_x_slices()
        y_slices = self.get_y_slices()
        # calculate the average x as the centre x
        centre_x = 0
        samples = 0
        for x, min_y, max_y in x_slices:
            count = max_y - min_y + 1
            centre_x += x * count
            samples += count
        centre_x /= samples
        # calculate the average y as the centre y
        centre_y = 0
        samples = 0
        for y, min_x, max_x in y_slices:
            count = max_x - min_x + 1
            centre_y += y * count
            samples += count
        centre_y /= samples
        # the centre is the top left of a 1x1 pixel square, an accurate centre is critical
        # the radius is calculated as the mean distance from the centre to the perimeter points
        # the centre is the top-left of a 1x1 pixel square, so the actual centre is +0.5 on this
        # we want the distance to the *outside* of the perimeter, so when the perimeter is ahead
        # of the centre we add the pixel width, i.e. 1, when the perimeter is behind its as is
        perimeter = self.get_perimeter()
        mean_distance_squared = 0
        for x, y in perimeter:
            if x < centre_x:
                x_distance = (centre_x + 0.5) - x
            else:
                x_distance = (x + 1) - (centre_x + 0.5)
            x_distance *= x_distance
            if y < centre_y:
                y_distance = (centre_y + 0.5) - y
            else:
                y_distance = (y + 1) - (centre_y + 0.5)
            y_distance *= y_distance
            mean_distance_squared += (x_distance + y_distance)
        mean_distance_squared /= len(perimeter)
        r = math.sqrt(mean_distance_squared)
        # make the circle
        self.circle = Circle(Point(centre_x, centre_y), r)
        return self.circle

    def get_area(self) -> float:
        """ an area is a 'count' of the pixels enclosed by the contour
            when external is True the area includes the perimeter
            when external is False the area excludes the perimeter
            """
        if self.area is not None:
            return self.area
        if self.points is None:
            return 0.0
        x_slices = self.get_x_slices()  # could equally well use y_slices
        self.area = 0
        for x, min_y, max_y in x_slices:
            self.area += (max_y - min_y + 1)
        if not self.external:
            # exclude the perimeter
            perimeter = len(self.get_perimeter())
            self.area -= perimeter
            if self.area < 1:
                raise Exception('internal area {} too small (points={}, perimeter={})'.
                                format(self.area, len(self.points), perimeter))
        return self.area

    def get_squareness(self) -> float:
        """ squareness is a measure of how square the enclosing box is,
            result is in range 0..1, where 0 is perfect square, 1 is very thin rectangle
            """
        size = self.get_size()
        ratio = min(size.x, size.y) / max(size.x, size.y)  # in range 0..1, 0=bad, 1=good
        return 1 - ratio  # in range 0..1, 0=square, 1=very thin rectangle

    def get_roundness(self) -> float:
        """ roundness is a measure of how close to a circle the contour is,
            result is in range 0..1, where 0 is a perfect circle and 1 is not at all circular
            """
        # NB: the enclosing circle area could be bigger or smaller than the contour area
        contour_area = self.get_area()
        circle_area = self.get_circle_area()
        max_area = max(contour_area, circle_area)
        if max_area < 1:
            # not a circle if got no area
            return 0.0
        min_area = min(contour_area, circle_area)
        if min_area < 1:
            # not a circle if got no area
            return 0.0
        delta = max_area - min_area  # range 0..max_area
        ratio = delta / max_area  # range 0..1, 0=good, 1=bad
        return ratio

    def get_wavyness(self):
        """ wavyness is a measure of how different the length of the perimeter is to the number of contour points,
            result is in range 0..1, where 0 is not wavy and 1 is very wavy
            """
        if self.points is None:
            return 1.0
        # NB: number of points is always more than the perimeter length
        return 1 - (len(self.get_perimeter()) / len(self.points))


class Blob:
    """ a blob is an external contour and its properties """

    def __init__(self, label: int, image):
        self.label: int = label
        self.image = image  # the binary image the blob was found within
        self.external: Contour = None
        self.internal: List[Contour] = []
        self.rejected = REJECT_UNKNOWN
        self.quality = None
        self.reset()

    def __str__(self):
        return self.show()

    def reset(self):
        """ reset all the cached stuff """
        self.holes = None

    def add_contour(self, contour: Contour):
        """ add a contour to the blob, the first contour is the external one,
            subsequent contours are internal,
        """
        if self.external is None:
            self.external = contour
        else:
            self.internal.append(contour)

    def show(self, verbose: bool = False, prefix: str = '    '):
        """ describe the blob for printing purposes,
            if verbose is True a multi-line response is made that describes all properties,
            lines after the first are prefixed by prefix
            """
        header = "label:{}".format(self.label)
        if self.external is None:
            return header
        body = '{}, {}'.format(header, self.external.show(verbose, prefix))
        if verbose:
            body = '{}\n{}internals:{}, holes:{}, whiteness:{:.2f}, quality:{}'.\
                   format(body, prefix, len(self.internal), self.get_holes(), self.get_whiteness(), self.quality)
        return body

    def get_holes(self):
        """ get the total holes area within the perimeter of the blob """
        if self.external is None:
            return None
        if self.holes is not None:
            return self.holes
        self.holes = count_black(self.image, self.external.get_x_slices())
        return self.holes

    def get_area(self) -> float:
        return self.external.get_area()

    def get_whiteness(self) -> float:
        """ whiteness is a measure of how 'white' the area of the contour is,
            result is in range 0..1, where 0 is all white and 1 is all black
            """
        if self.external is None:
            return None
        return self.get_holes() / self.get_area()

    def get_squareness(self) -> float:
        if self.external is None:
            return None
        return self.external.get_squareness()

    def get_roundness(self) -> float:
        if self.external is None:
            return None
        return self.external.get_roundness()

    def get_wavyness(self) -> float:
        if self.external is None:
            return None
        return self.external.get_wavyness()


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
    integration_width: int = 48          # width of integration area as fraction of image width
    integration_height: int = None       # height of integration area as fraction of image height (None==same as width)
    black_threshold: float = 0.01        # make +ve to get more white, -ve to get more black, range +100%..-100%
                                         # NB: Make a small +ve number to ensure totally white stays white
    white_threshold: float = None        # grey/white threshold, None == same as black (i.e. binary)
    direct_neighbours: bool = True       # True == 4-connected, False == 8-connected
    min_points: int = 12                 # minimum points in the contour (around a 3x4 rectangle)
    max_points: int = 1600               # maximum points in the contour (should be > pi*2R, R=sqrt(max_area/pi))
    min_area: float = 3 * 4              # this should be small to detect far away targets
    min_small_area: float = 5 * 5        # area below which thresholds change to the 'small' values
    max_area: float = 200000             # this has to be big to cater for testing drawn targets (rather than photos)
    max_internals: int = 1               # max number of internal contours that is tolerated to be a blob
    max_wavyness: float = (0.2, 0.2)     # how close to not wavy a contour perimeter must be, 0=not wavy, >0=more wavy
    max_whiteness: float = (0.1, 0.1)    # how close to fully white a blob has to be, 0=all white, >0=less white
    max_roundness: float = (0.3, 0.1)    # how close to circular a blob has to be, 0=perfect, >0=less perfect
    max_squareness: float = (0.25, 0.25) # how close to square the bounding box has to be, 0=perfect, >0=less perfect
    max_quality: float = (0.65, 0.5)     # how close to perfect quality a blob has to be, 0=perfect, >0=less perfect
    min_radius: float = 2.0              # less than this and we're into less than 1 pixel per ring!
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


def count_black(image, perimeter):
    """ count how many pixels in the area bounded by perimeter within image are black,
        perimeter is a list of x co-ords and the max/min y at that x
        """
    black = 0
    for x, min_y, max_y in perimeter:
        for y in range(min_y, max_y + 1):
            if image[y, x] == BLACK:
                black += 1
    return black


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
                blob = Blob(label, image)
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

    if debug:
        return blobs, buffer, labels
    else:
        return blobs


def filter_blobs(blobs: List[Blob], params: Targets, logger=None) -> List[Blob]:
    """ filter out blobs that are too small, too big or not round enough, et al,
        marks *all* blobs with an appropriate reject code and returns a list of good ones
        """

    if logger is not None:
        logger.push("filter_blobs")

    good_blobs = []                      # good blobs are accumulated in here
    for blob in blobs:
        while True:
            quality = None  # range 0..1, 0=perfect, >0=less perfect
            # check internals and points early as it's cheap and gets rid of a lot of junk
            if len(blob.internal) > params.max_internals:
                reason_code = REJECT_INTERNALS
                break
            if len(blob.external.points) < params.min_points:
                reason_code = REJECT_TOO_FEW_POINTS
                break
            if len(blob.external.points) > params.max_points:
                reason_code = REJECT_TOO_MANY_POINTS
                break
            area = blob.get_area()
            if area < params.min_area:
                reason_code = REJECT_TOO_SMALL
                break
            if area > params.max_area:
                reason_code = REJECT_TOO_BIG
                break
            if area > params.min_small_area:
                threshold = 1  # index into the threshold tuples for a big blob
            else:
                threshold = 0  # index into the threshold tuples for a small blob
            squareness = blob.get_squareness()
            if squareness > params.max_squareness[threshold]:
                reason_code = REJECT_SQUARENESS
                break
            wavyness = blob.get_wavyness()
            if wavyness > params.max_wavyness[threshold]:
                reason_code = REJECT_WAVYNESS
                break
            roundness = blob.get_roundness()
            if roundness > params.max_roundness[threshold]:
                reason_code = REJECT_ROUNDNESS
                break
            # this one is really expensive, so do it last
            whiteness = blob.get_whiteness()
            if whiteness > params.max_whiteness[threshold]:
                reason_code = REJECT_WHITENESS
                break
            # calculate a 'quality' of our 'ness' measures scaled to their threshold
            squareness /= params.max_squareness[threshold]  # now in range 0..1
            wavyness   /= params.max_wavyness[threshold]    # ..
            roundness  /= params.max_roundness[threshold]   # ....
            whiteness  /= params.max_whiteness[threshold]   # ......
            # quality = squareness * wavyness * roundness * whiteness  # still in range 0..1, 0=perfect, >0=less so
            quality = (squareness + wavyness + roundness + whiteness) / 4
            if quality > params.max_quality[threshold]:
                reason_code = REJECT_QUALITY
                break
            # all filters passed
            reason_code = REJECT_NONE
            good_blobs.append(blob)
            break
        blob.quality = quality
        blob.rejected = reason_code
        if logger is not None and reason_code != REJECT_NONE:
            logger.log("Rejected:{}, {}".format(reason_code, blob.show(verbose=True)))
    if logger is not None:
        logger.log("Accepted blobs: {}, rejected {} of {}".
                   format(len(good_blobs), len(blobs) - len(good_blobs), len(blobs)))
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
    passed = filter_blobs(blobs, params, logger=logger)
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
        params.targets.append([circle.centre.x, circle.centre.y, circle.radius, blob.label, blob.quality])
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

    def downsize(source, new_size):
        """ downsize the given image such that either its width or height is at most that given,
            the aspect ratio is preserved,
            """
        height, width = source.shape
        if width <= new_size or height <= new_size:
            # its already small enough
            return source
        if width > height:
            # bring height down to new size
            new_height = new_size
            new_width = int(width / (height / new_size))
        else:
            # bring width down to new size
            new_width = new_size
            new_height = int(height / (width / new_size))
        shrunk = cv2.resize(source, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return shrunk

    logger = Logger("_test")

    logger.log("\nPreparing image:")
    source = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    # Downsize it
    shrunk = downsize(source, 1152)
    logger.log("\nDetecting blobs")
    params = Targets()
    # params.integration_width = 48  # Scan.PROXIMITY_FAR
    # params.min_area = 10  # Scan.MIN_BLOB_AREA
    # params.min_radius = 2  # Scan.MIN_BLOB_RADIUS
    # do the actual detection
    image, blobs, buffer, labels, _ = find_targets(shrunk, params, logger=logger)
    # show what happenned
    draw = cv2.merge([image, image, image])
    logger.log("\n")
    cv2.imwrite("contours_binary.png", draw)
    logger.log("binary image shown in contours_binary.png")
    max_x = buffer.shape[1]
    max_y = buffer.shape[0]
    draw_bad = cv2.merge([shrunk, shrunk, shrunk])
    draw_good = cv2.merge([shrunk, shrunk, shrunk])
    # NB: cv2 colour order is BGR not RGB
    colours = {REJECT_NONE:            ((0, 255, 0),    'green'),
               REJECT_UNKNOWN:         ((0, 0, 255),    'red'),
               REJECT_TOO_FEW_POINTS:  ((80, 127, 255), 'orange'),
               REJECT_TOO_MANY_POINTS: ((80, 127, 255), 'orange'),
               REJECT_TOO_SMALL:       ((0, 255, 255),  'yellow'),
               REJECT_TOO_BIG:         ((0, 255, 255),  'yellow'),
               REJECT_ROUNDNESS:       ((255, 0, 0),    'blue'),
               REJECT_RADIUS:          ((255, 255, 0),  'cyan'),
               REJECT_MARGIN:          ((0, 0, 128),    'maroon'),
               REJECT_INTERNALS:       ((0, 128, 128),  'olive'),
               REJECT_WHITENESS:       ((128, 128, 0),  'teal'),
               REJECT_SQUARENESS:      ((128, 0, 0),    'navy'),
               REJECT_WAVYNESS:        ((255, 0, 255),  'magenta'),
               REJECT_QUALITY:         ((128, 0, 128),  'purple'),
               }
    for x in range(max_x):
        for y in range(max_y):
            label = buffer[y, x]
            if label > 0:
                blob = labels.get_blob(label)
                colour, _ = colours[blob.rejected]
                if blob.rejected == REJECT_NONE:
                    draw_good[y, x] = colour
                else:
                    draw_bad[y, x] = colour

    # draw enclosing circles in blue and bounding boxes in red on the detected targets
    for blob in blobs:
        if blob.rejected != REJECT_NONE:
            continue
        circle = blob.external.get_enclosing_circle()
        centre_x = int(round(circle.centre.x))
        centre_y = int(round(circle.centre.y))
        radius = int(round(circle.radius))
        draw_good = cv2.circle(draw_good, (centre_x, centre_y), radius, (255, 0, 0))
        top_left = blob.external.top_left
        bottom_right = blob.external.bottom_right
        start = (int(round(top_left.x)), int(round(top_left.y)))
        end = (int(round(bottom_right.x)), int(round(bottom_right.y)))
        draw_good = cv2.rectangle(draw_good, start, end, (0, 0, 255), 1)

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
    blobs.sort(key=lambda k: (k.external.top_left.x, k.external.top_left.y))
    for b, blob in enumerate(blobs):
        if blob.rejected != REJECT_NONE:
            continue
        logger.log("  {}: {}".format(b, blob.show(verbose=True)))
        for i, contour in enumerate(blob.internal):
            logger.log("    {}: {}".format(i, contour.show(verbose=True)))
    logger.log("\nAll detected targets:")
    params.targets.sort(key=lambda k: (k[0], k[1]))
    for t, target in enumerate(params.targets):
        logger.log("  {}: centre: {:.2f}, {:.2f}  radius: {:.2f}  label: {}, quality: {}".
                   format(t, target[0], target[1], target[2], target[3], target[4]))

    logger.log("\nBlob colours:")
    for reason, (_, name) in colours.items():
        logger.log('  {}: {}'.format(name, reason))


if __name__ == "__main__":
    #src = "/home/dave/blob-extractor/test/data/checker.png"
    #src = "/home/dave/blob-extractor/test/data/diffract.png"
    #src = "/home/dave/blob-extractor/test/data/dummy.png"
    #src = "/home/dave/blob-extractor/test/data/labyrinth.png"
    #src = "/home/dave/blob-extractor/test/data/lines.png"
    #src = "/home/dave/blob-extractor/test/data/simple.png"
    #src = "/home/dave/precious/fellsafe/fellsafe-image/codes/test-code-101.png"
    src = '/home/dave/precious/fellsafe/fellsafe-image/media/' \
          'distant-101-111-126-159-205-209-222-223-225-252-333-360-366-383-412-427-444-454-497-518.jpg'
    #src = "/home/dave/precious/fellsafe/fellsafe-image/media/lead-head-ratio-codes/photo-101.jpg"
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
