"""
    Find contours of 4-connected components.
    This module is a re-implementation of blob.h from https://github.com/BlockoS/blob
    Which is a C implementation of:
        "A linear-time component-labeling algorithm using contour tracing technique"
        by Fu Chang, Chun-Jen Chen, and Chi-Jen Lu.
    It has been tweaked to reduce the connectivity searched from 8 to 4 (i.e. direct neighbours only).
    It is also extended to compute the area of the components found and various other properties.
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
        self.points = None  # will become perimeter points on-demand

    def __str__(self):
        return "(centre:{}, radius:{:.2f}, area:{:.2f})".format(self.centre, self.radius, self.area())

    def area(self):
        return math.pi * self.radius * self.radius

    def perimeter(self):
        """ get the x,y co-ordinates of the perimeter of the circle,
            the co-ords are returned as a list of triplets
            where each triplet is the x co-ord and its two y co-ords,
            this format is compatible with count_black()
            """
        if self.points is not None:
            # already done it
            return self.points
        # this is expensive, so only do it once
        points = circumference(self.centre.x, self.centre.y, self.radius)
        # we want a min_y/max_y value pair for every x
        y_limits = {}
        min_x = None
        max_x = None
        for x, y in points:
            if min_x is None or x < min_x:
                min_x = x
            if max_x is None or x > max_x:
                max_x = x
            if y_limits.get(x) is None:
                y_limits[x] = [y, y]
            elif y < y_limits[x][0]:
                y_limits[x][0] = y
            elif y > y_limits[x][1]:
                y_limits[x][1] = y
        # build our tuple set
        self.points = []
        for x in range(min_x, max_x + 1):
            limits = y_limits.get(x)
            if limits is None:
                raise Exception('perimeter does not include x of {} when range is {}..{}'.format(x, min_x, max_x))
            min_y, max_y = limits
            min_y += 1  # we want exclusive co-ords
            max_y -= 1  # ..
            # NB: min_y > max_y can happen at the x extremes, but we chuck those, so we don't care
            self.points.append((x, min_y, max_y))
        # we want exclusive x co-ords, so knock the first and last out (there should be at least 3 entries)
        if len(self.points) < 3:
            raise Exception('perimeter has less than 3 x co-ords')
        del self.points[0]
        del self.points[-1]
        return self.points


class Contour:
    """ properties of a contour and methods to create/access them,
        NB: most of the metrics of a contour are in the range 0..1 where 0 is good and 1 is very bad
        """

    def __init__(self):
        self.points: [Point] = None  # points that make up the contour (NB: contours are a 'closed' set of points)
        self.top_left: Point = None
        self.bottom_right: Point = None
        self.reset()

    def __str__(self):
        return self.show()

    def reset(self):
        """ reset the cached stuff """
        self.blob_perimeter: {Point} = None
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
        first_line = 'start:{}, box:{}..{}, size:{}, points:{}, area:{:.2f}'.\
                     format(self.points[0], self.top_left, self.bottom_right,
                            self.get_size(), len(self.points), self.get_blob_area())
        if not verbose:
            return first_line
        second_line = 'perimeter:{}, circle:{}, squareness:{:.2f}, wavyness:{:.2f}'.\
                      format(len(self.get_blob_perimeter()), self.get_enclosing_circle(),
                             self.get_squareness(), self.get_wavyness())
        return '{}\n{}{}'.format(first_line, prefix, second_line)

    def get_squareness(self) -> float:
        """ squareness is a measure of how square the enclosing box is,
            result is in range 0..1, where 0 is perfect square, 1 is very thin rectangle,
            this is a very cheap metric that can be used as a pre-filter to quickly drop junk
            """
        size = self.get_size()
        ratio = min(size.x, size.y) / max(size.x, size.y)  # in range 0..1, 0=bad, 1=good
        return 1 - ratio  # in range 0..1, 0=square, 1=very thin rectangle

    def get_wavyness(self):
        """ wavyness is a measure of how different the length of the perimeter is to the number of contour points,
            result is in range 0..1, where 0 is not wavy and 1 is very wavy,
            this is a very cheap metric that can be used to quickly drop junk
            """
        if self.points is None:
            return 1.0
        # NB: number of points is always more than the perimeter length
        return 1 - (len(self.get_blob_perimeter()) / len(self.points))

    def get_size(self) -> Point:
        """ the size is the maximum width and height of the contour """
        width: float = self.bottom_right.x - self.top_left.x + 1
        height: float = self.bottom_right.y - self.top_left.y + 1
        return Point(width, height)

    def get_x_slices(self):
        """ get the slices in x,
            for every x co-ord find the y extent at that x,
            this function is lazy
            """
        if self.x_slices is not None:
            # already been done
            return self.x_slices
        x_slices = {}
        for point in self.points:
            if x_slices.get(point.x) is None:
                x_slices[point.x] = {}
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

    def get_blob_perimeter(self):
        """ get the unique contour perimeter points,
            this function is lazy
            """
        if self.blob_perimeter is not None:
            return self.blob_perimeter
        self.blob_perimeter = {}
        for point in self.points:
            self.blob_perimeter[(point.x, point.y)] = True  # NB: do NOT use point as the key, its an object not a tuple
        return self.blob_perimeter

    def get_circle_perimeter(self):
        """ get the perimeter of the enclosing circle,
            NB: the circle perimeter is expected to be cached by the Circle instance
            """
        circle = self.get_enclosing_circle()
        return circle.perimeter()

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
        perimeter = self.get_blob_perimeter()
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

    def get_blob_area(self) -> float:
        """ the area is the 'count' of all pixels (black or white) enclosed by the contour """
        if self.area is not None:
            return self.area
        if self.points is None:
            return 0.0
        x_slices = self.get_x_slices()  # could equally well use y_slices
        self.area = 0
        for x, min_y, max_y in x_slices:
            self.area += (max_y - min_y + 1)
        return self.area


class Blob:
    """ a blob is an external contour and its properties """

    def __init__(self, label: int, image):
        self.label: int = label
        self.image = image  # the binary image the blob was found within
        self.external: Contour = None
        self.internal: List[Contour] = []
        self.rejected = REJECT_UNKNOWN
        self.reset()

    def __str__(self):
        return self.show()

    def reset(self):
        """ reset all the cached stuff """
        self.blob_black = None
        self.blob_white = None
        self.circle_black = None
        self.circle_white = None
        self.margin = None

    def add_contour(self, contour: Contour):
        """ add a contour to the blob, the first contour is the external one,
            subsequent contours are internal,
        """
        if self.external is None:
            self.external = contour
        else:
            self.internal.append(contour)

    @staticmethod
    def format_float(value):
        """ produce a formatted float that may be None """
        if value is None:
            return 'None'
        return '{:.2f}'.format(value)

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
            body = '{}\n{}internals:{}, blob_pixels:{}, circle_pixels:{}, ' \
                   'roundness:{}, whiteness:{}, margin:{}'.\
                   format(body, prefix, len(self.internal), self.get_blob_pixels(), self.get_circle_pixels(),
                          self.format_float(self.get_roundness()),
                          self.format_float(self.get_whiteness()),
                          self.format_float(self.get_margin()))
        return body

    def get_quality_stats(self):
        """ get all the 'quality' statistics for a blob """
        return self.get_squareness(), self.get_wavyness(), self.get_whiteness(), self.get_roundness()

    def get_blob_pixels(self):
        """ get the total holes area within the perimeter of the blob """
        if self.external is None:
            return None
        if self.blob_black is not None:
            return self.blob_black, self.blob_white
        self.blob_black, self.blob_white = count_pixels(self.image, self.external.get_x_slices())
        return self.blob_black, self.blob_white

    def get_circle_pixels(self):
        """ get the total holes area within the enclosing circle """
        if self.external is None:
            return None
        if self.circle_black is not None:
            return self.circle_black, self.circle_white
        self.circle_black, self.circle_white = count_pixels(self.image, self.external.get_circle_perimeter())
        return self.circle_black, self.circle_white

    def get_blob_area(self) -> float:
        if self.external is None:
            return None
        # NB: not using get_blob_pixels 'cos its slower and this function is very busy
        return self.external.get_blob_area()

    def get_squareness(self) -> float:
        if self.external is None:
            return None
        return self.external.get_squareness()

    def get_radius(self):
        if self.external is None:
            return None
        return self.external.get_enclosing_circle().radius

    def get_wavyness(self) -> float:
        if self.external is None:
            return None
        return self.external.get_wavyness()

    def get_margin(self):
        """ get the distance of the blob centre from the image edge in terms of radius multiples """
        if self.external is None:
            return None
        if self.margin is not None:
            return self.margin
        max_y = self.image.shape[0]
        max_x = self.image.shape[1]
        circle = self.external.get_enclosing_circle()
        margin_x_max = (max_x - circle.centre.x) / circle.radius
        margin_x_min = circle.centre.x / circle.radius
        margin_x = min(margin_x_max, margin_x_min)
        margin_y_max = (max_y - circle.centre.y) / circle.radius
        margin_y_min = circle.centre.y / circle.radius
        margin_y = min(margin_y_max, margin_y_min)
        self.margin = min(margin_x, margin_y)
        return self.margin

    def get_roundness(self):
        """ roundness is a measure of how circular the blob is,
            result is in the range 0..1, where 0 is a perfect circle and >0 is less so
            """
        if self.external is None:
            return None

        # for a perfect circle circumference/area is 2/radius and 2/radius is the minimum possible,
        # area=pi*r*r, circumference=2*pi*r, (2*pi*r)/(pi*r*r)==2/r
        # circumference is the length of the blob perimeter, area is the blob area, radius is enclosing circle radius
        circumference = len(self.external.get_blob_perimeter())
        area = self.external.get_blob_area()
        radius = int(round(self.get_radius() + 0.5))  # +0.5 to allow for quantisation effects
        if radius < 2:
            # too small to be meaningful, treat as circular
            circularity = 0.0
        else:
            circularity = (circumference / area) - (2 / radius)  # range -?..+?
        if circularity < -0.3:
            # this should not happen!
            raise Exception('circularity ({:.2f}) < 0 when circumference {}, area {}, radius {}'.
                            format(circularity, circumference, area, radius))
        else:
            circularity = max(circularity, 0.0)  # ignore small error (can happen due to quantisation effects)
        return circularity

    def get_whiteness(self) -> float:
        """ whiteness is a measure of how 'white' the area of the enclosing circle is,
            result is in range 0..1, where 0 is all white and 1 is all black
            """
        if self.external is None:
            return None
        circle_black, circle_white = self.get_circle_pixels()
        return circle_black / (circle_black + circle_white)


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
    min_points: int = 10                 # minimum points in the contour (around a 3x4 rectangle)
    max_points: int = 1600               # maximum points in the contour (should be > pi*2R, R=sqrt(max_area/pi))
    min_area: float = 3 * 4              # this should be small to detect far away targets
    min_small_area: float = 6 * 6        # area below which thresholds change to the 'small' values
    max_area: float = 200000             # this has to be big to cater for testing drawn targets (rather than photos)
    max_internals: int = 1               # max number of internal contours that is tolerated to be a blob
    # all these 'ness' thresholds are a tuple of small area threshold and big area threshold
    max_squareness = (0.4 , 0.3 )        # how close to square the bounding box has to be, 0=perfect, >0=less perfect
    max_wavyness   = (0.2 , 0.15)        # how close to not wavy a contour perimeter must be, 0=not wavy, >0=more wavy
    max_whiteness  = (0.2 , 0.15)        # how close to fully white a blob has to be, 0=all white, >0=less white
    max_roundness  = (0.3 , 0.18)        # how close to circular a blob has to be, 0=perfect, >0=less perfect
    min_radius: float = 2.1              # less than this and we're into less than 1 pixel per ring!
    min_margin: float = 4.0              # min margin to image edge of target in units of target radius
    targets: List[tuple] = None          # the result


def make_binary(source, width: float=8, height: float=None, black: float=15, white: float=None):
    """ create a binary (or tertiary) image of source using an adaptive threshold,
        width is the fraction of the image width to use as the integration area,
        height is the fraction of the image height to use as the integration area (None==same as width in pixels)
        black is the % below the average that is considered to be the black/grey boundary,
        white is the % above the average that is considered to be the grey/white boundary,
        white of None means same as black and will yield a binary image,
        See the adaptive-threshold-algorithm.pdf paper for algorithm details.
        """

    # get the image metrics
    max_y = source.shape[0]
    max_x = source.shape[1]
    width_pixels = int(max_x / width)  # we want this to be odd so that there is a centre
    width_plus = max(width_pixels >> 1, 2)  # offset for going forward
    width_minus = width_plus - 1  # offset for going backward
    if height is None:
        height_pixels = width_pixels  # make it square
    else:
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
    contour: Contour = Contour()
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


def count_pixels(image, perimeter):
    """ count how many pixels in the area bounded by perimeter within image are black and how many are white,
        perimeter is a list of x co-ords and the max/min y at that x,
        co-ords outside the image bounds are considered to be black
        """
    limit_y = image.shape[0]
    limit_x = image.shape[1]
    black = 0
    white = 0
    for x, min_y, max_y in perimeter:
        for y in range(min_y, max_y + 1):
            if y >= limit_y or y < 0:
                black += 1
            elif x >= limit_x or x < 0:
                black += 1
            elif image[y, x] == BLACK:
                black += 1
            else:
                white += 1
    return black, white


def circumference(centre_x: float, centre_y: float, r: float) -> [(int, int)]:
    """ return a list of co-ordinates of a circle centred on x,y of radius r,
        x,y,r do not need to be integer but the returned co-ordinates will be integers,
        the co-ordinates returned are suitable for drawing the circle,
        co-ordinates returned are unique but in a random order,
        the algorithm here was inspired by: https://www.cs.helsinki.fi/group/goa/mallinnus/ympyrat/ymp1.html
        """

    centre_x: int = int(round(centre_x))
    centre_y: int = int(round(centre_y))

    points = []  # list of x,y tuples for a circle of radius r centred on centre_x,centre_y

    def circle_points(x: int, y: int):
        """ make all 8 quadrant points from the one point given
            from https://www.cs.helsinki.fi/group/goa/mallinnus/ympyrat/ymp1.html
                Procedure Circle_Points(x,y: Integer);
                Begin
                    Plot(x,y);
                    Plot(y,x);
                    Plot(y,-x);
                    Plot(x,-y);
                    Plot(-x,-y);
                    Plot(-y,-x);
                    Plot(-y,x);
                    Plot(-x,y)
                End;
        """

        def plot(x_offset: int, y_offset: int):
            """ add the circle point for the given x,y offsets from the centre """
            points.append((centre_x + x_offset, centre_y + y_offset))

        # NB: when a co-ord is 0, x and -x are the same, ditto for y

        if x == 0 and y == 0:
            plot(0, 0)
        elif x == 0:
            plot(0, y)
            plot(y, 0)
            plot(0, -y)
            plot(-y, 0)
        elif y == 0:
            plot(x, 0)
            plot(0, x)
            plot(0, -x)
            plot(-x, 0)
        elif x == y:
            plot(x, x)
            plot(x, -x)
            plot(-x, -x)
            plot(-x, x)
        elif x == -y:
            plot(x, -x)
            plot(-x, x)
            plot(-x, -x)
            plot(x, x)
        else:
            plot(x, y)
            plot(y, x)
            plot(y, -x)
            plot(x, -y)
            plot(-x, -y)
            plot(-y, -x)
            plot(-y, x)
            plot(-x, y)

    """ from https://www.cs.helsinki.fi/group/goa/mallinnus/ympyrat/ymp1.html
        Begin {Circle}
        x := r;
        y := 0;
        d := 1 - r;
        Repeat
            Circle_Points(x,y);
            y := y + 1;
            if d < 0 Then
                d := d + 2*y + 1
            Else Begin
                x := x - 1;
                d := d + 2*(y-x) + 1
            End
        Until x < y
        End; {Circle}
    """
    x = int(round(r))
    y = 0
    d = 1 - x
    while True:
        circle_points(x, y)
        y += 1
        if d < 0:
            d += (2 * y + 1)
        else:
            x -= 1
            d += (2 * (y - x) + 1)
        if x < y:
            break

    return points


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
            threshold = None  # we do not know yet
            quality = None  # range 0..1, 0=perfect, >0=less perfect
            # do the cheap checks first
            if len(blob.internal) > params.max_internals:
                reason_code = REJECT_INTERNALS
                break
            if len(blob.external.points) < params.min_points:
                reason_code = REJECT_TOO_FEW_POINTS
                break
            if len(blob.external.points) > params.max_points:
                reason_code = REJECT_TOO_MANY_POINTS
                break
            squareness = blob.get_squareness()
            if squareness > max(params.max_squareness[0], params.max_squareness[1]):
                # worse than both, we do this early 'cos its cheap and effective
                reason_code = REJECT_SQUARENESS
                break
            wavyness = blob.get_wavyness()
            if wavyness > max(params.max_wavyness[0], params.max_wavyness[1]):
                # worse than both, we do this early 'cos its cheap and effective
                reason_code = REJECT_WAVYNESS
                break
            # now do the expensive checks, in cheapest first order
            radius = blob.get_radius()
            if radius < params.min_radius:
                reason_code = REJECT_RADIUS
                break
            margin = blob.get_margin()
            if margin < params.min_margin:
                reason_code = REJECT_MARGIN
                break
            area = blob.get_blob_area()
            if area < params.min_area:
                reason_code = REJECT_TOO_SMALL
                break
            if area > params.max_area:
                reason_code = REJECT_TOO_BIG
                break
            # setup which threshold to apply to subsequent checks (area dependant)
            if area > params.min_small_area:
                threshold = 1  # index into the threshold tuples for a big blob
            else:
                threshold = 0  # index into the threshold tuples for a small blob
            if squareness > params.max_squareness[threshold]:
                reason_code = REJECT_SQUARENESS
                break
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
            # all filters passed
            reason_code = REJECT_NONE
            good_blobs.append(blob)
            break
        blob.quality = quality
        blob.rejected = reason_code
        if logger is not None and reason_code != REJECT_NONE:
            if threshold is None:
                size = ''
            elif threshold == 0:
                size = '(small)'
            else:
                size = '(big)'
            logger.log("Rejected:{}{}, {}".format(reason_code, size, blob.show(verbose=True)))
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
    params.targets = []
    for blob in passed:
        circle = blob.external.get_enclosing_circle()
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


def _test(src, size, proximity, black):
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

    logger.log("\nPreparing image: size={}, proximity={}".format(size, proximity))
    source = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    # Downsize it
    shrunk = downsize(source, size)
    logger.log("\nDetecting blobs")
    params = Targets()
    params.integration_width = proximity
    params.black_threshold = black
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
    colours = {REJECT_NONE:            ((0, 255, 0),    'lime'),
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

    def plot(buffer, points, colour):
        for x, y in points:
            if x < 0 or x >= max_x or y < 0 or y >= max_y:
                continue
            buffer[y, x] = colour  # NB: cv2 x, y are reversed
        return buffer

    # draw enclosing circles in blue and bounding boxes in red on the detected targets
    for blob in blobs:
        if blob.rejected != REJECT_NONE:
            continue
        # top_left = blob.external.top_left
        # bottom_right = blob.external.bottom_right
        # start = (int(round(top_left.x)), int(round(top_left.y)))
        # end = (int(round(bottom_right.x)), int(round(bottom_right.y)))
        # draw_good = cv2.rectangle(draw_good, start, end, (0, 0, 255), 1)
        circle = blob.external.get_enclosing_circle()
        points = circumference(circle.centre.x, circle.centre.y, circle.radius)
        draw_good = plot(draw_good, points, (255, 0, 0))

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
    stats_range = 20
    stats_span = 100/stats_range/100
    all_squareness_stats = [[0, 0] for _ in range(stats_range + 1)]
    all_wavyness_stats = [[0, 0] for _ in range(stats_range + 1)]
    all_whiteness_stats = [[0, 0] for _ in range(stats_range + 1)]
    all_roundness_stats = [[0, 0] for _ in range(stats_range + 1)]
    squareness_stats = [[0, 0] for _ in range(stats_range + 1)]
    wavyness_stats = [[0, 0] for _ in range(stats_range + 1)]
    whiteness_stats = [[0, 0] for _ in range(stats_range + 1)]
    roundness_stats = [[0, 0] for _ in range(stats_range + 1)]
    all_size_stats = [0, 0]
    size_stats = [0, 0]
    reject_stats = {}
    good_blobs = 0
    blobs.sort(key=lambda k: (k.external.top_left.x, k.external.top_left.y))

    def update_count(stats, size, value):
        if value is None:
            return
        stats[int(value * stats_range)][size] += 1

    def log_stats(name, stats):
        for size in range(len(stats[0])):
            msg = ''
            for i, stat in enumerate(stats):
                num = stat[size]
                if num == 0:
                    continue
                msg = '{}, {:.2f}-{}'.format(msg, i*stats_span, num)
            logger.log('  {:10}({}): {}'.format(name, size, msg[2:]))

    for b, blob in enumerate(blobs):
        if reject_stats.get(blob.rejected) is None:
            reject_stats[blob.rejected] = 1
        else:
            reject_stats[blob.rejected] += 1
        area = blob.get_blob_area()
        if area > params.min_small_area:
            size = 1
        else:
            size = 0
        all_size_stats[size] += 1
        squareness, wavyness, whiteness, roundness = blob.get_quality_stats()
        update_count(all_squareness_stats, size, squareness)
        update_count(all_wavyness_stats, size, wavyness)
        update_count(all_whiteness_stats, size, whiteness)
        update_count(all_roundness_stats, size, roundness)
        if blob.rejected != REJECT_NONE:
            continue
        good_blobs += 1
        size_stats[size] += 1
        update_count(squareness_stats, size, squareness)
        update_count(wavyness_stats, size, wavyness)
        update_count(whiteness_stats, size, whiteness)
        update_count(roundness_stats, size, roundness)
        logger.log("  {}: {}".format(b, blob.show(verbose=True)))
    # show stats
    logger.log("\nAll reject frequencies (across {} blobs):".format(len(blobs)))
    for reason, count in reject_stats.items():
        logger.log("  {}: {} ({:.2f}%)".format(reason, count, (count/len(blobs))*100))
    logger.log("\nAll blobs stats (across {} blobs):".format(len(blobs)))
    log_stats("squareness", all_squareness_stats)
    log_stats("wavyness", all_wavyness_stats)
    log_stats("whiteness", all_whiteness_stats)
    log_stats("roundness", all_roundness_stats)
    logger.log("  sizes:{}".format(all_size_stats))

    logger.log("\nAll accepted blobs stats (across {} blobs):".format(good_blobs))
    log_stats("squareness", squareness_stats)
    log_stats("wavyness", wavyness_stats)
    log_stats("whiteness", whiteness_stats)
    log_stats("roundness", roundness_stats)
    logger.log("  sizes:{}".format(size_stats))

    logger.log("\nAll detected targets:")
    params.targets.sort(key=lambda k: (k[0], k[1]))
    for t, target in enumerate(params.targets):
        logger.log("  {}: centre: {:.2f}, {:.2f}  radius: {:.2f}  label: {}".
                   format(t, target[0], target[1], target[2], target[3]))

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
    src = "/home/dave/precious/fellsafe/fellsafe-image/codes/test-code-101.png"
    # src = '/home/dave/precious/fellsafe/fellsafe-image/media/' \
    #       'distant-101-102-111-116-222-298-333-387-401-444-555-666-673-732-746-756-777-888-892-999.jpg'
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

    # region test shape...
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
    # endregion

    # size 1152 is 2K, 2160 is 4K, proximity 16 is close, 48 is far, black=-5 for far, 0.01 for close
    _test(src, size=1152, proximity=16, black=0.01)
