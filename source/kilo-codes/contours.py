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

import const
import utils
import math
import numpy as np
import cv2

# co-ordinates of pixel neighbours relative to that pixel, clockwise from 'east':
#   [5][6][7]
#   [4][-][0]
#   [3][2][1]
dx = [1, 1, 0, -1, -1, -1,  0,  1]       # x-offset
dy = [0, 1, 1,  1,  0, -1, -1, -1]       # .. y-offset

# region Reject codes for blobs being ignored...
REJECT_NONE            = 'accepted'
REJECT_UNKNOWN         = 'unknown'
REJECT_TOO_SMALL       = 'size below minimum'
REJECT_TOO_BIG         = 'size above maximum'
REJECT_INTERNALS       = 'too many internal contours'
REJECT_WHITENESS       = 'not enough circle white'
REJECT_BLACKNESS       = 'not enough box white'
REJECT_SQUARENESS      = 'not square enough'
REJECT_WAVYNESS        = 'perimeter too wavy'
REJECT_OFFSETNESS      = 'centroid too offset'
# endregion

BLACK = const.MIN_LUMINANCE
WHITE = const.MAX_LUMINANCE
GREY  = const.MID_LUMINANCE

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
            this format is compatible with count_pixels()
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
        the contour only knows the co-ordinates of its points, it knows nothing of the underlying image,
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
        self.circle: Circle = None
        self.centroid: Point = None
        self.offset: float = None

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
        first_line = 'start:{}, box:{}..{}, size:{}, points:{}'.\
                     format(self.points[0], self.top_left, self.bottom_right, self.get_size(), len(self.points))
        if not verbose:
            return first_line
        second_line = 'centroid:{}, offsetness:{:.2f}, squareness:{:.2f}, wavyness:{:.2f}'.\
                      format(self.get_centroid(), self.get_offsetness(),
                             self.get_squareness(), self.get_wavyness())
        return '{}\n{}{}'.format(first_line, prefix, second_line)

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
        """ the size is the maximum width and height of the contour,
            i.e. the size of the enclosing box
            this is a very cheap metric that can be used to quickly drop junk
            """
        width: float = self.bottom_right.x - self.top_left.x + 1
        height: float = self.bottom_right.y - self.top_left.y + 1
        return Point(width, height)

    def get_squareness(self) -> float:
        """ squareness is a measure of how square the enclosing box is,
            result is in range 0..1, where 0 is perfect square, 1 is very thin rectangle,
            this is a very cheap metric that can be used to quickly drop junk
            """
        size = self.get_size()
        ratio = min(size.x, size.y) / max(size.x, size.y)  # in range 0..1, 0=bad, 1=good
        return 1 - ratio  # in range 0..1, 0=square, 1=very thin rectangle

    def get_offsetness(self) -> float:
        """ offsetness is a measure of the distance from the centroid to the enclosing box centre,
            result is in range 0..1, where 0 is exactly coincident, 1 is very far apart
            """
        if self.offset is None:
            box_size = self.get_size()
            box_centre = Point(self.top_left.x + (box_size.x / 2), self.top_left.y + (box_size.y / 2))
            centroid = self.get_centroid()  # NB: this cannot be outside the enclosing box
            x_diff = box_centre.x - centroid.x  # max this can be is box_size.x
            x_diff *= x_diff
            y_diff = box_centre.y - centroid.y  # max this can be is box_size.y
            y_diff *= y_diff
            distance = x_diff + y_diff  # most this can be is box_size.x^2 + box_size.y^2
            limit = max((box_size.x * box_size.x) + (box_size.y * box_size.y), 1)
            self.offset = distance / limit
        return self.offset

    def get_x_slices(self):
        """ get the slices in x,
            for every unique x co-ord find the y extent at that x,
            this function is lazy
            """
        # ToDo: extend to remove 1 pixel gaps
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
            for every unique y co-ord find the x extent at that y,
            this function is lazy
            """
        # ToDo: extend to remove 1 pixel gaps
        if self.y_slices is not None:
            # already been done
            return self.y_slices
        y_slices = {}
        for point in self.points:
            if y_slices.get(point.y) is None:
                y_slices[point.y] = {}
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

    def get_enclosing_circle(self) -> Circle:
        """ the enclosing circle of a contour is the centre and the radius required to cover (most of) it,
            this is an expensive operation so its lazy, calculated on demand and then cached
            """
        if self.circle is not None:
            return self.circle
        centre = self.get_centroid()  # the centre of mass of the blob
        # the centre is the top left of a 1x1 pixel square, an accurate centre is critical,
        # the radius is calculated as the mean distance from the centre to the perimeter points,
        # the centre is the top-left of a 1x1 pixel square, so the actual centre is +0.5 on this,
        # we want the distance to the *outside* of the perimeter, so when the perimeter is ahead
        # of the centre we add the pixel width, i.e. 1, when the perimeter is behind its as is
        perimeter = self.get_blob_perimeter()
        mean_distance_squared = 0
        for x, y in perimeter:
            if x < centre.x:
                x_distance = (centre.x + 0.5) - x
            else:
                x_distance = (x + 1) - (centre.x + 0.5)
            x_distance *= x_distance
            if y < centre.y:
                y_distance = (centre.y + 0.5) - y
            else:
                y_distance = (y + 1) - (centre.y + 0.5)
            y_distance *= y_distance
            mean_distance_squared += (x_distance + y_distance)
        if len(perimeter) > 0:
            mean_distance_squared /= len(perimeter)
        r = math.sqrt(mean_distance_squared)
        # make the circle
        self.circle = Circle(centre, r)
        return self.circle

    def get_circle_perimeter(self):
        """ get the perimeter of the enclosing circle,
            NB: the circle perimeter is expected to be cached by the Circle instance
            """
        circle = self.get_enclosing_circle()
        return circle.perimeter()

    def trim_edges(self):
        """ remove single pixel extension or indentation in x or y slices at the x or y extremes,
            e.g: XXX                            XXXX
                  XX                            XXX   <-- x indentation
                  XX                            XXXX
                   X  <-- y extension           X XX
                ^                                ^
                |                                |
                +-- x extension                  +-- y indentation
            the algorithm is iterative eating away at the edges until there are no extensions or indentations
            the result may be nothing when dealing with small blobs!
            this is intended to tidy up target edges
            """
        # ToDo: implement the above then hook it into the properties (in particular wavyness)
        #       indentation is better done in x_slices and y_slices
        # an indentation is a slice with
        dropped = True
        while dropped:
            dropped = False
            x_slices = self.get_x_slices()
            while len(x_slices) > 0:
                _, min_y, max_y = x_slices[0]
                if min_y == max_y:
                    # got a protrusion at min-x, drop it
                    del x_slices[0]
                    dropped = True
            while len(x_slices) > 0:
                _, min_y, max_y = x_slices[-1]
                if min_y == max_y:
                    # got a protrusion at max-x, drop it
                    del x_slices[-1]
                    dropped = True
            y_slices = self.get_y_slices()
            while len(y_slices) > 0:
                _, min_x, max_x = y_slices[0]
                if min_x == max_x:
                    # got a protrusion at min-y, drop it
                    del y_slices[0]
                    dropped = True
            while len(y_slices) > 0:
                _, min_x, max_x = y_slices[-1]
                if min_x == max_x:
                    # got a protrusion at max-y, drop it
                    del y_slices[-1]
                    dropped = True

    def get_centroid(self) -> Point:
        """ get the centroid of the blob as: sum(points)/num(points) """
        if self.centroid is None:
            sum_x = 0
            num_x = 0
            x_slices = self.get_x_slices()
            for x, min_y, max_y in x_slices:
                samples = max_y - min_y + 1
                sum_x += samples * x
                num_x += samples
            sum_y = 0
            num_y = 0
            y_slices = self.get_y_slices()
            for y, min_x, max_x in y_slices:
                samples = max_x - min_x + 1
                sum_y += samples * y
                num_y += samples
            self.centroid = Point(sum_x / num_x, sum_y / num_y)
        return self.centroid


class Blob:
    """ a blob is an external contour and its properties,
        a blob has access to the underlying image (unlike a Contour)
        """

    def __init__(self, label: int, image, inverted: bool):
        self.label: int = label
        self.image = image  # the binary image buffer the blob was found within
        self.inverted = inverted  # True if its a balck blob, else a white blob
        self.external: Contour = None
        self.internal: [Contour] = []
        self.rejected = REJECT_UNKNOWN
        self.reset()

    def __str__(self):
        return self.show()

    def reset(self):
        """ reset all the cached stuff """
        self.blob_black = None
        self.blob_white = None
        self.box_black = None
        self.box_white = None
        self.circle_black = None
        self.circle_white = None

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
            body = '{}\n{}internals:{}, blob_pixels:{}, box_pixels:{}, whiteness:{}, blackness:{}'.\
                   format(body, prefix, len(self.internal), self.get_blob_pixels(), self.get_box_pixels(),
                          self.format_float(self.get_whiteness()), self.format_float(self.get_blackness()))
        return body

    def get_quality_stats(self):
        """ get all the 'quality' statistics for a blob """
        return self.get_squareness(), self.get_wavyness(),\
               self.get_whiteness(), self.get_blackness(), self.get_offsetness()

    def get_circle_pixels(self):
        """ get the total white area and black area within the enclosing circle """
        if self.external is None:
            return None
        if self.circle_black is not None:
            return self.circle_black, self.circle_white
        self.circle_black, self.circle_white = count_pixels(self.image, self.external.get_circle_perimeter())
        return self.circle_black, self.circle_white

    def get_blob_pixels(self):
        """ get the total white area and black area within the perimeter of the blob """
        if self.external is None:
            return None
        if self.blob_black is not None:
            return self.blob_black, self.blob_white
        self.blob_black, self.blob_white = count_pixels(self.image, self.external.get_x_slices())
        return self.blob_black, self.blob_white

    def get_box_pixels(self):
        """ get the total white area and black area within the enclosing box of the blob """
        if self.external is None:
            return None
        if self.box_black is not None:
            return self.box_black, self.box_white
        # build 'x-slices' for the box
        x_slices = []
        for x in range(self.external.top_left.x, self.external.bottom_right.x + 1):
            x_slices.append((x, self.external.top_left.y, self.external.bottom_right.y))
        self.box_black, self.box_white = count_pixels(self.image, x_slices)
        return self.box_black, self.box_white

    def get_squareness(self) -> float:
        if self.external is None:
            return None
        return self.external.get_squareness()

    def get_size(self) -> float:
        """ the size of a blob is the average of the width + height of the bounding box """
        if self.external is None:
            return None
        point = self.external.get_size()
        return (point.x + point.y) / 2

    def get_wavyness(self) -> float:
        """ this allows for filtering very irregular blobs """
        if self.external is None:
            return None
        return self.external.get_wavyness()

    def get_offsetness(self) -> float:
        """ this measure how far the centre of mass is from the centre of the enclosing box,
            this allows for filtering elongated blobs (e.g. a 'saucepan')
            """
        if self.external is None:
            return None
        return self.external.get_offsetness()

    def get_whiteness(self) -> float:
        """ whiteness is a measure of how 'white' the area covered by the enclosing circle is,
            (contrast this with blackness which covers the whole enclosing box)
            when inverted is false:
                result is in range 0..1, where 0 is all white and 1 is all black,
            when inverted is True:
                result is in range 0..1, where 0 is all black and 1 is all white,
            this allows for filtering out blobs with lots of holes in it
            """
        if self.external is None:
            return None
        black, white = self.get_circle_pixels()
        if self.inverted:
            return white / (black + white)
        else:
            return black / (black + white)

    def get_blackness(self) -> float:
        """ blackness is a measure of how 'white' the area covered by the enclosing box is,
            (contrast this with whiteness which covers the enclosing circle)
            when inverted is false:
                result is in range 0..1, where 0 is all white and 1 is all black,
            when inverted is True:
                result is in range 0..1, where 0 is all black and 1 is all white,
            this allows for filtering out sparse symmetrical blobs (e.g. a 'star')
            """
        if self.external is None:
            return None
        black, white = self.get_box_pixels()
        if self.inverted:
            return white / (black + white)
        else:
            return black / (black + white)


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
        elif label in self.blobs:
            return self.blobs[label]
        else:
            # no such label
            return None


class Targets:
    """ a holder for the parameters required by find_targets and its result """
    source_file = None                   # file name of originating source image (for diagnostic naming purposes only)
    source = None                        # the source greyscale image buffer
    blurred = None                       # the blurred greyscale image buffer
    binary = None                        # the binarized blurred image buffer
    box = None                           # when not None the box within the image to process, else all of it
    inverted = False                     # if True look for black blobs, else white blobs
    integration_width: int = 48          # width of integration area as fraction of image width
    integration_height: int = None       # height of integration area as fraction of image height (None==same as width)
    black_threshold: float = 0.01        # make +ve to get more white, -ve to get more black, range +100%..-100%
                                         # NB: Make a small +ve number to ensure totally white stays white
    white_threshold: float = None        # grey/white threshold, None == same as black (i.e. binary)
    direct_neighbours: bool = True       # True == 4-connected, False == 8-connected
    min_area: float = 3 * 3              # this should be small to detect far away targets
    max_area: float = 200000             # this has to be big to cater for testing drawn targets (rather than photos)
    max_internals: int = 1               # max number of internal contours that is tolerated to be a blob
    min_size: float = 3                  # min number of pixels across width/height
    max_size: float = 100                # max number of pixels across width/height
    blur_kernel_size = 3                 # blur kernel size to apply, must be odd, None or < 3 == do not blur
    # all these 'ness' parameters are in the range 0..1, where 0 is perfect and 1 is utter crap
    max_squareness = 0.25                # how close to square the bounding box has to be (0.5 is a 2:1 rectangle)
    max_wavyness   = 0.25                # how close to not wavy a contour perimeter must be
    max_offsetness = 0.03                # how close the centroid has to be to the enclosing box centre
    max_whiteness  = 0.25                # whiteness of the enclosing circle
    max_blackness  = 0.5                 # whiteness of the enclosing box (0.5 is worst case for a 45 deg rotated sq)
    targets: [tuple]  = None             # the result


def blur_image(source, kernel_size):
    """ return the blurred image as a mean blur over the given kernel size """
    # we do this by integrating then calculating the average via integral differences,
    # this means we only visit each pixel once irrespective of the kernel size
    if kernel_size is not None:
        kernel_size = kernel_size | 1  # must be odd (so there is a centre)
    else:
        kernel_size = 0
    if kernel_size < 3:
        return source  # pointless

    # integrate the image
    integral = integrate(source)

    # get image geometry
    width  = len(integral[0])
    height = len(integral)

    # set kernel geometry
    kernel_plus  = kernel_size >> 1  # offset for going forward
    kernel_minus = kernel_size - 1   # offset for going backward

    # blur the image by averaging over the given kernel size
    blurred = np.zeros((height, width), np.uint8)
    for x in range(width):
        x1 = int(max(x - kernel_minus, 0))
        x2 = int(min(x + kernel_plus, width - 1))
        for y in range(height):
            y1 = int(max(y - kernel_minus, 0))
            y2 = int(min(y + kernel_plus, height - 1))
            count = int((x2 - x1) * (y2 - y1))  # how many samples in the integration area
            # sum = bottom right (x2,y2) + top left (x1,y1) - top right (x2,y1) - bottom left (x1,y2)
            # where all but bottom right are *outside* the integration window
            average = int((integral[y2][x2] + integral[y1][x1] - integral[y1][x2] - integral[y2][x1]) / count)
            blurred[y][x] = average

    return blurred

def integrate(source, box=None):
    """ generate the integral of the given box within the given image """

    if box is None:
        box_min_x = 0
        box_max_x = len(source[0])
        box_min_y = 0
        box_max_y = len(source)
    else:
        box_min_x = box[0][0]
        box_max_x = box[1][0] + 1
        box_min_y = box[0][1]
        box_max_y = box[1][1] + 1
    box_width  = box_max_x - box_min_x
    box_height = box_max_y - box_min_y

    # make an empty buffer to accumulate our integral in
    integral = np.zeros((box_height, box_width), np.int32)

    for y in range(box_min_y, box_max_y):
        for x in range(box_min_x, box_max_x):
            if x == box_min_x:
                acc = int(source[y, x])
            else:
                acc += int(source[y, x])
            ix = x - box_min_x
            iy = y - box_min_y
            if iy == 0:
                integral[iy][ix] = acc
            else:
                integral[iy][ix] = acc + integral[iy - 1][ix]

    return integral

def make_binary(source, box=None, width: float=8, height: float=None, black: float=15, white: float=None):
    """ create a binary (or tertiary) image of the source image within the box using an adaptive threshold,
        if box is None the whole image is processed, otherwise just the area within the given box,
        width is the fraction of the source/box width to use as the integration area,
        height is the fraction of the source/box height to use as the integration area (None==same as width in pixels)
        black is the % below the average that is considered to be the black/grey boundary,
        white is the % above the average that is considered to be the grey/white boundary,
        white of None means same as black and will yield a binary image,
        See the adaptive-threshold-algorithm.pdf paper for algorithm details.
        the image returned is the same size as the box (or the source iff no box given)
        """

    # region get the source and box metrics...
    source_min_x = 0
    source_max_x = source.shape[1]
    source_min_y = 0
    source_max_y = source.shape[0]
    if box is None:
        box_min_x = source_min_x
        box_max_x = source_max_x
        box_min_y = source_min_y
        box_max_y = source_max_y
    else:
        box_min_x = box[0][0]
        box_max_x = box[1][0] + 1
        box_min_y = box[0][1]
        box_max_y = box[1][1] + 1
    box_width  = box_max_x - box_min_x
    box_height = box_max_y - box_min_y
    # endregion

    # region set the integration size...
    width_pixels = int(box_width / width)            # we want this to be odd so that there is a centre
    width_plus   = max(width_pixels >> 1, 2)         # offset for going forward
    width_minus  = width_plus - 1                    # offset for going backward
    if height is None:
        height_pixels = width_pixels                 # make it square
    else:
        height_pixels = int(box_height / height)     # we want this to be odd so that there is a centre
    height_plus  = max(height_pixels >> 1, 2)        # offset for going forward
    height_minus = height_plus - 1                   # offset for going backward
    # endregion

    # integrate the image
    integral = integrate(source, box)

    # region do the threshold on a new image buffer...
    binary = np.zeros((box_height, box_width), np.uint8)
    black_limit = (100-black)/100    # convert % to a ratio
    if white is None:
        white_limit = black_limit
    else:
        white_limit = (100+white)/100  # convert % to a ratio
    for x in range(box_width):
        x1 = int(max(x - width_minus, 0))
        x2 = int(min(x + width_plus, box_width - 1))
        for y in range(box_height):
            y1 = int(max(y - height_minus, 0))
            y2 = int(min(y + height_plus, box_height - 1))
            count = int((x2 - x1) * (y2 - y1))  # how many samples in the integration area
            # sum = bottom right (x2,y2) + top left (x1,y1) - top right (x2,y1) - bottom left (x1,y2)
            # where all but bottom right are *outside* the integration window
            acc = integral[y2][x2] + integral[y1][x1] - integral[y1][x2] - integral[y2][x1]
            src = int(source[box_min_y + y, box_min_x + x]) * count  # NB: need int 'cos source is uint8 (unsigned)
            if src >= (acc * white_limit):
                binary[y, x] = WHITE
            elif src <= (acc * black_limit):
                binary[y, x] = BLACK
            else:
                binary[y, x] = GREY
    # endregion

    return binary

def contour_trace(image, buffer, label: int, x: int, y: int,
                  external: bool = True, direct: bool = True, inverted: bool=False) -> Contour:
    """ follow the contour at x,y in image giving it label in buffer,
        if external is True follow an external contour, else internal,
        if direct is True use 4-connected neighbours else 8,
        if inverted is True follow black contours, else white,
        both image and buffer must be the same shape, have a zero border and x,y must never be in it
        """
    if inverted:
        follow = BLACK
    else:
        follow = WHITE
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
            if image[y1, x1] == follow:
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

    def plot(x_offset: int, y_offset: int):
        """ add the circle point for the given x,y offsets from the centre """
        points.append((centre_x + x_offset, centre_y + y_offset))

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
    if x == 0:
        # special case
        plot(0,0)
    else:
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

def find_blobs(image, direct: bool = True, inverted: bool=False, debug: bool = False) -> [Blob]:
    """ find_blobs in the given image,
        if inverted is True look for black blobs, else white
        returns a list of Blob's or None if failed
    """
    if inverted:
        find_outside = WHITE  # the colour *outside* the blob we look for
    else:
        find_outside = BLACK  # the colour *outside* the blob we look for
    width = image.shape[1]
    height = image.shape[0]
    if width < 3 or height < 3:
        return None
    # allocate a label buffer
    buffer = np.zeros((height, width), np.int32)  # NB: must be signed values, 0==background
    # put a 'follow' border around the image edge 1 pixel wide
    # put a -1 border around the buffer edges 1 pixel wide
    # this means we can scan from 1,1 to max-1,max-1 and not need to bother about the edges
    for x in range(width):
        image[0, x] = find_outside
        image[height-1, x] = find_outside
        buffer[0, x] = -1
        buffer[height - 1, x] = -1
    for y in range(height):
        image[y, 0] = find_outside
        image[y, width-1] = find_outside
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
            if here_in == find_outside:
                continue
            above_in = image[y-1, x]
            below_in = image[y+1, x]
            here_label = buffer[y, x]
            before_label = buffer[y, x-1]
            below_label = buffer[y+1, x]
            if here_label == 0 and above_in == find_outside:
                # found a new start
                label += 1               # assign next label
                blob = Blob(label, image, inverted)
                blob.add_contour(contour_trace(image, buffer, label, x, y,
                                               external=True, direct=direct, inverted=inverted))
                blobs.append(blob)
                labels.add_label(label, blob)
            elif below_label == 0 and below_in == find_outside:
                # found a new internal contour (a hole)
                if before_label < 1:
                    # this means its a hole in 'here'
                    before_label = here_label
                current_blob = labels.get_blob(before_label)
                if current_blob is None:
                    raise Exception('Cannot find current blob when before label is {} at {}x{}'.
                                    format(before_label, x, y))
                current_blob.add_contour(contour_trace(image, buffer, before_label, x, y,
                                                       external=False, direct=direct, inverted=inverted))
            elif here_label == 0:
                # found an internal element of an external contour
                buffer[y, x] = before_label

    if debug:
        return blobs, buffer, labels
    else:
        return blobs

def filter_blobs(blobs: [Blob], params: Targets, logger=None) -> [Blob]:
    """ filter out blobs that do no meet the target criteria,
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
            size = blob.get_size()
            if size < params.min_size:
                reason_code = REJECT_TOO_SMALL
                break
            if size > params.max_size:
                reason_code = REJECT_TOO_BIG
                break
            squareness = blob.get_squareness()
            if squareness > params.max_squareness:
                reason_code = REJECT_SQUARENESS
                break
            wavyness = blob.get_wavyness()
            if wavyness > params.max_wavyness:
                reason_code = REJECT_WAVYNESS
                break
            # now do the expensive checks, in cheapest first order
            offsetness = blob.get_offsetness()
            if offsetness > params.max_offsetness:
                reason_code = REJECT_OFFSETNESS
                break
            blackness = blob.get_blackness()
            if blackness > params.max_blackness:
                reason_code = REJECT_BLACKNESS
                break
            whiteness = blob.get_whiteness()
            if whiteness > params.max_whiteness:
                reason_code = REJECT_WHITENESS
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
        rejected = len(blobs) - len(good_blobs)
        logger.log("Accepted blobs: {}, rejected {}({:.2f}%) of {}".
                   format(len(good_blobs), rejected, (rejected / len(blobs)) * 100, len(blobs)))
        logger.pop()
    return good_blobs

def find_targets(source, params: Targets, logger=None):
    """ given a monochrome image find all potential targets,
        returns a list of targets where each is a tuple of x:float, y:float, radius:float, label:int
        x,y is the pixel address of the centre of the target and radius is its radius in pixels,
        all may be fractional,
        label is the label number assigned to the blob the target was detected within,
        although the ideal target consists of square blobs, we report them as if they are circles,
        this is so we do not have to deal with shape rotation, we are only interested in relative
        distances not shapes, so this is fine,
        NB: The presence of a logger implies we are debugging
        """
    if logger is not None:
        logger.push("find_targets")
    params.source  = source
    params.blurred = blur_image(source, params.blur_kernel_size)
    params.binary  = make_binary(params.blurred,
                                 params.box,
                                 params.integration_width,
                                 params.integration_height,
                                 params.black_threshold,
                                 params.white_threshold)
    if logger is None:
        blobs = find_blobs(params.binary, params.direct_neighbours, inverted=params.inverted)
    else:
        blobs, buffer, labels = find_blobs(params.binary, params.direct_neighbours,
                                           inverted=params.inverted, debug=True)
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

def get_targets(source, params=None, logger=None) -> [tuple]:
    """ find targets in the given image using default parameters """
    if params is None:
        params = Targets()  # use defaults
    result = find_targets(source, params, logger)
    if logger is not None:
        show_result(params, result, logger)
    return params.targets, params.binary

def extract_box(image, box: ((int, int), (int, int))):
    """ extract a box of the given image """
    tl_x, tl_y = box[0]
    br_x, br_y = box[1]
    width  = br_x - tl_x + 1
    height = br_y - tl_y + 1
    buffer = np.zeros((height, width), np.uint8)
    for x in range(width):
        for y in range(height):
            buffer[y, x] = image[tl_y + y, tl_x + x]
    return buffer

def show_result(params, result, logger):
    # show what happened
    image, blobs, buffer, labels, _ = result
    if params.blur_kernel_size is not None and params.blur_kernel_size >= 3:
        logger.draw(params.blurred, file='blurred')
    draw = cv2.merge([image, image, image])
    logger.draw(draw, file='binary')
    max_x = buffer.shape[1]
    max_y = buffer.shape[0]
    if params.box is not None:
        source_part = extract_box(params.source, params.box)
    else:
        source_part = params.source
    logger.draw(source_part, file='grayscale')
    draw_bad = cv2.merge([source_part, source_part, source_part])
    draw_good = cv2.merge([source_part, source_part, source_part])
    # NB: cv2 colour order is BGR not RGB
    colours = {REJECT_NONE: (const.LIME, 'lime'),
               REJECT_UNKNOWN: (const.RED, 'red'),
               REJECT_TOO_SMALL: (const.YELLOW, 'yellow'),
               REJECT_TOO_BIG: (const.YELLOW, 'yellow'),
               REJECT_WHITENESS: (const.MAROON, 'maroon'),
               REJECT_BLACKNESS: (const.CYAN, 'cyan'),
               REJECT_INTERNALS: (const.OLIVE, 'olive'),
               REJECT_SQUARENESS: (const.NAVY, 'navy'),
               REJECT_WAVYNESS: (const.MAGENTA, 'magenta'),
               REJECT_OFFSETNESS: (const.ORANGE, 'orange'),
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

    # draw enclosing circles in blue on the detected targets
    for blob in blobs:
        if blob.rejected != REJECT_NONE:
            continue
        circle = blob.external.get_enclosing_circle()
        points = circumference(circle.centre.x, circle.centre.y, circle.radius)
        draw_good = plot(draw_good, points, const.BLUE)

    logger.draw(draw_good, file='accepted')
    logger.draw(draw_bad, file='rejected')
    draw = cv2.merge([source_part, source_part, source_part])
    for target in params.targets:
        x = int(round(target[0]))
        y = int(round(target[1]))
        cv2.circle(draw, (x, y), int(round(target[2])), (0, 255, 0), 1)
    logger.draw(draw, file='blobs')

    logger.log("\nAll accepted blobs:")
    stats_range = 20
    stats_span = 100 / stats_range / 100
    all_squareness_stats = [0 for _ in range(stats_range + 1)]
    all_wavyness_stats = [0 for _ in range(stats_range + 1)]
    all_whiteness_stats = [0 for _ in range(stats_range + 1)]
    all_blackness_stats = [0 for _ in range(stats_range + 1)]
    all_offsetness_stats = [0 for _ in range(stats_range + 1)]
    squareness_stats = [0 for _ in range(stats_range + 1)]
    wavyness_stats = [0 for _ in range(stats_range + 1)]
    whiteness_stats = [0 for _ in range(stats_range + 1)]
    blackness_stats = [0 for _ in range(stats_range + 1)]
    offsetness_stats = [0 for _ in range(stats_range + 1)]
    reject_stats = {}
    good_blobs = 0
    blobs.sort(key=lambda k: (k.external.top_left.x, k.external.top_left.y))

    def update_count(stats, value):
        if value is None:
            return
        stats[int(value * stats_range)] += 1

    def log_stats(name, stats):
        msg = ''
        for i, num in enumerate(stats):
            if num == 0:
                continue
            msg = '{}, {:.2f}-{}'.format(msg, i * stats_span, num)
        logger.log('  {:10}: {}'.format(name, msg[2:]))

    for b, blob in enumerate(blobs):
        if reject_stats.get(blob.rejected) is None:
            reject_stats[blob.rejected] = 1
        else:
            reject_stats[blob.rejected] += 1
        squareness, wavyness, whiteness, blackness, offsetness = blob.get_quality_stats()
        update_count(all_squareness_stats, squareness)
        update_count(all_wavyness_stats, wavyness)
        update_count(all_whiteness_stats, whiteness)
        update_count(all_blackness_stats, blackness)
        update_count(all_offsetness_stats, offsetness)
        if blob.rejected != REJECT_NONE:
            continue
        good_blobs += 1
        update_count(squareness_stats, squareness)
        update_count(wavyness_stats, wavyness)
        update_count(whiteness_stats, whiteness)
        update_count(blackness_stats, blackness)
        update_count(offsetness_stats, offsetness)
        logger.log("  {}: {}".format(b, blob.show(verbose=True)))
    # show stats
    logger.log("\nAll reject frequencies (across {} blobs):".format(len(blobs)))
    for reason, count in reject_stats.items():
        logger.log("  {}: {} ({:.2f}%)".format(reason, count, (count / len(blobs)) * 100))
    logger.log("\nAll blobs stats (across {} blobs):".format(len(blobs)))
    log_stats("squareness", all_squareness_stats)
    log_stats("wavyness", all_wavyness_stats)
    log_stats("whiteness", all_whiteness_stats)
    log_stats("blackness", all_blackness_stats)
    log_stats("offsetness", all_offsetness_stats)

    logger.log("\nAll accepted blobs stats (across {} blobs):".format(good_blobs))
    log_stats("squareness", squareness_stats)
    log_stats("wavyness", wavyness_stats)
    log_stats("whiteness", whiteness_stats)
    log_stats("blackness", blackness_stats)
    log_stats("offsetness", offsetness_stats)

    logger.log("\nAll detected targets:")
    params.targets.sort(key=lambda k: (k[0], k[1]))
    for t, target in enumerate(params.targets):
        logger.log("  {}: centre: {:.2f}, {:.2f}  radius: {:.2f}  label: {}".
                   format(t, target[0], target[1], target[2], target[3]))

    logger.log("\nBlob colours:")
    for reason, (_, name) in colours.items():
        logger.log('  {}: {}'.format(name, reason))

def downsize(source, new_size):
    """ downsize the given image such that either its width or height is at most that given,
        the aspect ratio is preserved, its a no-op if image already small enough,
        returns the modified image,
        this is purely a diagnostic aid to simulate low-resolution cameras
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
    shrunk = cv2.resize(source, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    return shrunk

def load_image(src):
    """ load the given image file as a grayscale image """
    return cv2.imread(src, cv2.IMREAD_GRAYSCALE)

def _test(src, size, proximity, black, inverted, blur, logger, params=None):
    """ ************** TEST **************** """

    logger.log("\nPreparing image: size={}, proximity={}, blur={}".format(size, proximity, blur))
    source = load_image(src)
    # Downsize it (to simulate low quality smartphone cameras)
    shrunk = downsize(source, size)
    logger.log("\nDetecting blobs")
    if params is None:
        params = Targets()
    params.source_file = src
    params.integration_width = proximity
    params.black_threshold = black
    params.inverted = inverted
    params.blur_kernel_size = blur
    # do the actual detection
    result = get_targets(shrunk, params, logger=logger)

    return params


if __name__ == "__main__":
    #src = "targets.jpg"
    #src = "/home/dave/blob-extractor/test/data/checker.png"
    #src = "/home/dave/blob-extractor/test/data/diffract.png"
    #src = "/home/dave/blob-extractor/test/data/dummy.png"
    #src = "/home/dave/blob-extractor/test/data/labyrinth.png"
    #src = "/home/dave/blob-extractor/test/data/lines.png"
    #src = "/home/dave/blob-extractor/test/data/simple.png"
    # src = "/home/dave/precious/fellsafe/fellsafe-image/codes/test-code-101.png"
    # src = '/home/dave/precious/fellsafe/fellsafe-image/media/' \
    #       'far-101-102-111-116-222-298-333-387-401-444-555-666-673-732-746-756-777-888-892-999.jpg'
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
    #src = "/home/dave/precious/fellsafe/fellsafe-image/codes/test-code-101.png"
    #src = "/home/dave/precious/fellsafe/fellsafe-image/media/close-101-111-124-172-222-281-333-337-354-444-555-594-655-666-710-740-777-819-888-900.jpg"
    src = "/home/dave/precious/fellsafe/fellsafe-image/media/square-codes/square-codes-distant.jpg"
    #src = "/home/dave/precious/fellsafe/fellsafe-image/source/kilo-codes/test-alt-bits.png"
    #proximity = const.PROXIMITY_CLOSE
    proximity = const.PROXIMITY_FAR

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

    logger = utils.Logger('contours.log', 'contours')
    logger.log('_test')

    _test(src, size=const.VIDEO_2K, proximity=proximity, black=const.BLACK_LEVEL[proximity],
          inverted=True, blur=3, logger=logger)
