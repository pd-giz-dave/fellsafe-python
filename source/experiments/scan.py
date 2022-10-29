
import codec
import ring
import angle
import frame
import contours
import math
import os
from typing import List
import traceback
import time
import numpy as np

# colours
MAX_LUMINANCE = 255
MIN_LUMINANCE = 0
MID_LUMINANCE = (MAX_LUMINANCE - MIN_LUMINANCE) >> 1


def nstr(number, fmt='.2f'):
    """ given a number that may be None, return an appropriate string """
    if number is None:
        return 'None'
    else:
        fmt = '{:' + fmt + '}'
        return fmt.format(number)


def vstr(vector, fmt='.2f', open='[', close=']'):
    """ given a list of numbers return a string representing them """
    if vector is None:
        return 'None'
    result = ''
    for pt in vector:
        result += ', ' + nstr(pt)
    return open + result[2:] + close


def wrapped_gap(x1, x2, limit_x):
    """ given two x co-ords that may be wrapped return the gap between the two,
        a legitimate gap must be less than half the limit,
        the returned gap may be +ve or -ve and represents the gap from x1 to x2,
        i.e. (x1 + gap) % limit_x = x2
        """

    dx = x2 - x1
    if dx < 0:
        # x1 bigger than x2, this is OK provided the gap is less than half the limit
        if (0 - dx) > (limit_x / 2):
            # gap too big, so its a wrap, so true gap is (x2 + limit) - x1
            dx = (x2 + limit_x) - x1
    else:
        # x1 smaller than x2
        if dx > (limit_x / 2):
            # gap too big, so its wrap, so true gap is  x2 - (x1 + limit)
            dx = x2 - (x1 + limit_x)
    return dx


class Scan:
    """ this class provides functions to scan an image and extract any codes in it,
        algorithm summary:
            1. blob detect our bullseye
            2. warp polar around blob centres to create a cartesian co-ordinate rectangle
            3. edge detect all the ring and bit boundaries
            4. extract the bits for each ring segment
            5. decode those bits
        the algorithm has been developed more by experimentation than theory!
        """

    # region Constants...
    # our target shape
    NUM_RINGS = ring.Ring.NUM_RINGS  # total number of rings in the whole code (ring==cell in height)
    BULLSEYE_RINGS = ring.Ring.BULLSEYE_RINGS  # number of rings inside the inner edge
    NUM_DATA_RINGS = codec.Codec.RINGS_PER_DIGIT  # how many data rings in our codes
    INNER_BLACK_RINGS = codec.Codec.INNER_BLACK_RINGS  # black rings from inner to first data ring
    OUTER_BLACK_RINGS = codec.Codec.OUTER_BLACK_RINGS  # black rings from outer to last data ring
    INNER_OUTER_SPAN_RINGS = codec.Codec.SPAN  # how many rings between the inner and outer edges
    NUM_SEGMENTS = codec.Codec.DIGITS  # total number of segments in a ring (segment==cell in length)
    DIGITS_PER_NUM = codec.Codec.DIGITS_PER_WORD  # how many digits per encoded number
    COPIES = codec.Codec.COPIES_PER_BLOCK  # number of copies in a code-word

    # image 'segment' and 'ring' constraints,
    # a 'segment' is the angular division in a ring,
    # a 'ring' is a radial division,
    # a 'cell' is the intersection of a segment and a ring
    # these constraints set minimums that override the cells() property given to Scan
    MIN_PIXELS_PER_CELL = 4  # min pixels making up a cell length
    MIN_PIXELS_PER_RING = 4  # min pixels making up a ring width

    # region Tuning constants...
    MIN_BLOB_AREA = 10  # min area of a blob we want (in pixels) (default 9)
    MIN_BLOB_RADIUS = 2  # min radius of a blob we want (in pixels) (default 2.0)
    BLOB_RADIUS_STRETCH = 1.3  # how much to stretch blob radius to ensure always cover everything when projecting
    MIN_CONTRAST = 0.15  # minimum luminance variation of a valid blob projection relative to the max luminance
    THRESHOLD_WIDTH = 6  # the fraction of the projected image width to use as the integration area when binarizing
    THRESHOLD_HEIGHT = 2  # the fraction of the projected image height to use as the integration area (None=as width)
    THRESHOLD_BLACK = 10  # the % below the average luminance in a projected image that is considered to be black
    THRESHOLD_WHITE = 0  # the % above the average luminance in a projected image that is considered to be white
    MIN_EDGE_SAMPLES = 2  # minimum samples in an edge to be considered a valid edge
    MAX_NEIGHBOUR_ANGLE_INNER = 0.4  # ~=22 degrees, tan of the max acceptable angle when joining inner edge fragments
    MAX_NEIGHBOUR_ANGLE_OUTER = 0.6  # ~=31 degrees, tan of the max acceptable angle when joining outer edge fragments
    MAX_NEIGHBOUR_HEIGHT_GAP = 1  # max x or y jump allowed when following an edge
    MAX_NEIGHBOUR_HEIGHT_GAP_SQUARED = MAX_NEIGHBOUR_HEIGHT_GAP * MAX_NEIGHBOUR_HEIGHT_GAP
    MAX_NEIGHBOUR_LENGTH_JUMP = 10  # max x jump, in pixels, between edge fragments when joining
    MAX_NEIGHBOUR_HEIGHT_JUMP = 3  # max y jump, in pixels, between edge fragments when joining
    MAX_NEIGHBOUR_OVERLAP = 4  # max edge overlap, in pixels, between edge fragments when joining
    MAX_EDGE_GAP_SIZE = 3 / NUM_SEGMENTS  # max gap tolerated between edge fragments (as fraction of image width)
    MAX_EDGE_HEIGHT_JUMP = 8  # max jump in y, in pixels, allowed along an edge, more than this is an edge 'failure'
    MAX_DIGIT_WIDTH = 3  # maximum width of a digit relative to the nominal width (image / all digits)
    # endregion

    # region Video modes image height...
    VIDEO_SD = 480
    VIDEO_HD = 720
    VIDEO_FHD = 1080
    VIDEO_2K = 1152
    VIDEO_4K = 2160
    # endregion

    # region Proximity options
    # these control the contour detection, for big targets that cover the whole image a bigger
    # integration area is required (i.e. smaller image fraction), this is used for testing
    # print images
    PROXIMITY_FAR = 48  # suitable for most images (photos and videos)
    PROXIMITY_CLOSE = 16  # suitable for print images
    # end region
    # region Debug options...
    DEBUG_NONE = 0  # no debug output
    DEBUG_IMAGE = 1  # just write debug annotated image files
    DEBUG_VERBOSE = 2  # do everything - generates a *lot* of output
    # endregion

    # region Step/Edge/Corner types...
    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'
    RISING = 'rising'
    FALLING = 'falling'
    # endregion

    # region Diagnostic image colours...
    BLACK = (0, 0, 0)
    GREY = (64, 64, 64)
    WHITE = (255, 255, 255)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    DARK_GREEN = (0, 128, 0)
    BLUE = (255, 0, 0)
    DARK_BLUE = (64, 0, 0)
    YELLOW = (0, 255, 255)
    PURPLE = (255, 0, 255)
    PINK = (128, 0, 128)
    CYAN = (128, 128, 0)
    PALE_RED = (0, 0, 128)
    PALE_BLUE = (128, 0, 0)
    PALE_GREEN = (0, 128, 0)
    # endregion
    # endregion
    # endregion

    # region Structures...
    class Step:
        """ a Step is a description of a luminance change
            (orientation of vertical or horizontal is defined by the context)
            """

        def __init__(self, where, type, from_pixel, to_pixel):
            self.where = where  # the y co-ord of the step
            self.type = type  # the step type, rising or falling
            self.from_pixel = from_pixel  # the 'from' pixel level
            self.to_pixel = to_pixel  # the 'to' pixel level

        def __str__(self):
            return '({} at {} {}->{})'.format(self.type, self.where, self.from_pixel, self.to_pixel)

    class Edge:
        """ an Edge is a sequence of joined steps """

        def __init__(self, where, type, samples):
            self.where = where  # the x co-ord of the start of this edge
            self.type = type  # the type of the edge, falling or rising
            self.samples = samples  # the list of connected y's making up this edge

        def __str__(self):
            self.show()

        def show(self, in_line=10, max_x=None):
            """ generate a readable string describing the edge """
            if in_line == 0:
                # do not want to see samples
                samples = ''
            elif len(self.samples) > in_line:
                # too many samples to see all, so just show first and last few
                samples = ': {}..{}'.format(self.samples[:int(in_line/2)], self.samples[-int(in_line/2):])
            else:
                # can show all the samples
                samples = ': {}'.format(self.samples)
            from_x = self.where
            to_x = from_x + len(self.samples) - 1
            if max_x is not None:
                to_x %= max_x
            from_y = self.samples[0]
            to_y = self.samples[-1]
            return '{} at ({},{}) to ({},{}) for {}{}'.\
                   format(self.type, from_x, from_y, to_x, to_y, len(self.samples), samples)

    class Extent:
        """ an Extent is the inner edge co-ordinates of a projected image along with
            the horizontal and vertical edge fragments it was built from """

        def __init__(self, inner=None, outer=None, inner_fail=None, outer_fail=None,
                     buckets=None, rising_edges=None, falling_edges=None, slices=None):
            self.inner: [int] = inner  # list of y co-ords for the inner edge
            self.inner_fail = inner_fail  # reason if failed to find inner edge or None if OK
            self.outer: [int] = outer  # list of y co-ords for the outer edge
            self.outer_fail = outer_fail  # reason if failed to find outer edge or None if OK
            self.rising_edges: [Scan.Edge] = rising_edges  # rising edge list used to create this extent
            self.falling_edges: [Scan.Edge] = falling_edges  # falling edge list used to create this extent
            self.buckets = buckets  # the binarized image the extent was created from
            self.slices = slices  # the slices extracted from the extent (by _find_all_digits)

    class Digit:
        """ a digit is a decode of a sequence of slice samples into the most likely digit """

        def __init__(self, digit, error, start, samples):
            self.digit = digit  # the most likely digit
            self.error = error  # the average error across its samples
            self.start = start  # start x co-ord of this digit
            self.samples = samples  # the number of samples in this digit

        def __str__(self):
            return '({}, {:.2f}, at {} for {})'.format(self.digit, self.error, self.start, self.samples)

    class Box:
        """ a 'box' is an area that represents one black or white data bit in the projected image """

        def __init__(self, start: int, span: int):
            self.start = start  # x co-ord of start of box
            self.span = span  # the x co-ord span of the digit
            self.edges = None  # list of y co-ords pairs for a box top/bottom edge

        def add_edge(self, edge: [[int, int]]):
            if self.edges is None:
                self.edges = []
            self.edges.append(edge)  # start/end y co-ords of a ring (exclusive)

    class Result:
        """ a result is the result of a number decode and its associated error/confidence level """

        def __init__(self, number, doubt, code, digits):
            self.number = number  # the number found
            self.doubt = doubt  # integer part is sum of error digits (i.e. where not all three copies agree)
                                # fractional part is average bit error across all the slices
            self.code = code  # the code used for the number lookup
            self.digits = digits  # the digit pattern used to create the code

    class Target:
        """ structure to hold detected target information """

        def __init__(self, centre_x, centre_y, blob_size, target_size, result):
            self.centre_x = centre_x  # x co-ord of target in original image
            self.centre_y = centre_y  # y co-ord of target in original image
            self.blob_size = blob_size  # blob size originally detected by the blob detector
            self.target_size = target_size  # target size scaled to the original image (==outer edge average Y)
            self.result = result  # the number, doubt and digits of the target

    class Reject:
        """ struct to hold info about rejected targets """

        def __init__(self, centre_x, centre_y, blob_size, target_size, reason):
            self.centre_x = centre_x
            self.centre_y = centre_y
            self.blob_size = blob_size
            self.target_size = target_size
            self.reason = reason

    class Detection:
        """ struct to hold info about a Scan detected code """

        def __init__(self, result, centre_x, centre_y, target_size, blob_size):
            self.result = result  # the result of the detection
            self.centre_x = centre_x  # where it is in the original image
            self.centre_y = centre_y  # ..
            self.blob_size = blob_size  # the size of the blob as detected by opencv
            self.target_size = target_size  # the size of the target in the original image (used for relative distance)
    # endregion

    def __init__(self, codec, frame, transform, cells=(MIN_PIXELS_PER_CELL, MIN_PIXELS_PER_RING),
                 video_mode=VIDEO_FHD, proximity=PROXIMITY_FAR,
                 debug=DEBUG_NONE, log=None):
        """ codec is the codec instance defining the code structure,
            frame is the frame instance containing the image to be scanned,
            transform is the class implmenting various image transforms (mostly for diagnostic purposes),
            cells is the angular/radial resolution to use,
            video_mode is the maximum resolution to work at, the image is downsized to this if required,
            proximity is the contour detection integration area parameter
            """

        # set debug options
        if debug == self.DEBUG_IMAGE:
            self.show_log = False
            self.save_images = True
        elif debug == self.DEBUG_VERBOSE:
            self.show_log = True
            self.save_images = True
        else:
            self.show_log = False
            self.save_images = False

        # params
        self.proximity = proximity  # contour detection image fraction to integrate over
        self.cells = cells  # (segment length x ring height) size of segment cells to use when decoding
        self.video_mode = video_mode  # actually the downsized image height
        self.original = frame
        self.decoder = codec  # class to decode what we find

        # set warped image width/height
        self.angle_steps = int(round(Scan.NUM_SEGMENTS * max(self.cells[0], Scan.MIN_PIXELS_PER_CELL)))
        self.radial_steps = int(round(Scan.NUM_RINGS * Scan.BLOB_RADIUS_STRETCH *
                                      max(self.cells[1], Scan.MIN_PIXELS_PER_RING)))
        # NB: stretching radial steps by BLOB_RADIUS_STRETCH to ensure each ring achieves the min size we want

        # opencv wrapper functions
        self.transform = transform

        # decoding context (used for logging and image saving)
        self.centre_x = 0
        self.centre_y = 0

        # logging context
        self._log_file = None
        self._log_folder = log
        self._log_prefix = '{}: Blob'.format(self.original.source)

        self.logging = self.show_log or (self._log_folder is not None)

        # prepare image
        self.image = self.transform.downsize(self.original, self.video_mode)  # re-size to given video mode
        self.binary = None     # binarized image put here by _blobs function (debug aid)

        # needed by _project
        max_x, max_y = self.image.size()
        max_radius = min(max_x, max_y) / 2
        max_circumference = min(2 * math.pi * max_radius, 360)  # good enough for 1 degree resolution
        angles = angle.Angle(max_circumference, max_radius)
        self.angle_xy = angles.polarToCart

    def __del__(self):
        """ close our log file """
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _blobs(self) -> List[tuple]:
        """ find the target blobs in our image,
            this must be the first function called to process our image,
            creates a blob list each of which is a 'keypoint' tuple of:
                .pt[1] = x co-ord of the centre
                .pt[0] = y co-ord of centre
                .size = diameter of blob
            returns a list of unique blobs found
            """

        if False:  # ToDo: generates a *lot* of logs-->self.show_log:
            logger = contours.Logger('_blobs')
        else:
            logger = None
        params = contours.Targets()
        params.integration_width = self.proximity
        params.min_area = Scan.MIN_BLOB_AREA
        params.min_radius = Scan.MIN_BLOB_RADIUS
        # ToDo: get a better estimate of the blob centre - see openCV eg's
        blobs, binary = contours.get_targets(self.image.buffer, params=params, logger=logger)
        self.binary = self.image.instance()
        self.binary.set(binary)

        # just so the processing order is deterministic (helps debugging)
        blobs.sort(key=lambda e: (int(round(e[0])), int(round(e[1]))))

        if self.logging:
            self._log('blob-detect: found {} blobs'.format(len(blobs)), 0, 0)
            for blob in blobs:
                self._log("    x:{:.2f}, y:{:.2f}, size:{:.2f}".format(blob[0], blob[1], blob[2]))

        if self.save_images:
            plot = self.binary
            for blob in blobs:
                plot = self.transform.label(plot, blob, Scan.GREEN)
            self._unload(plot, 'contours', 0, 0)

        return blobs

    def _radius(self, centre_x, centre_y, blob_size) -> int:
        """ determine the image radius to extract around the given blob position and size
            blob_size is used as a guide to limit the radius projected,
            we assume the blob-size is (roughly) the radius of the inner two white rings
            but err on the side of going too big
            """

        max_x, max_y = self.image.size()
        edge_top = centre_y
        edge_left = centre_x
        edge_bottom = max_y - centre_y
        edge_right = max_x - centre_x
        ring_width = blob_size / 2
        limit_radius = max(min(edge_top, edge_bottom, edge_left, edge_right), 1)
        blob_radius = ring_width * Scan.NUM_RINGS * Scan.BLOB_RADIUS_STRETCH
        if blob_radius < limit_radius:
            # max possible size is less than the image edge, so use the blob size
            limit_radius = blob_radius

        limit_radius = int(math.ceil(limit_radius))

        if self.logging:
            self._log('radius: limit radius {}'.format(limit_radius))

        return limit_radius

    def _project(self, blob_size) -> (frame.Frame, float):
        """ 'project' a potential target from its circular shape to a rectangle of radius (y) by angle (x),
            returns the projection or None if its not worth pursuing further and a 'stretch-factor'
            """
        def get_pixel(x: float, y: float) -> int:
            """ get the interpolated pixel value at offset x,y from the centre,
                x,y are fractional so the pixel value returned is a mixture of the 4 pixels around x,y
                the mixture is based on the ratio of the neighbours to include, the ratio of all 4 is 1
                code based on:
                    void interpolateColorPixel(double x, double y) {
                        int xL, yL;
                        xL = (int) Math.floor(x);
                        yL = (int) Math.floor(y);
                        xLyL = ipInitial.getPixel(xL, yL, xLyL);
                        xLyH = ipInitial.getPixel(xL, yL + 1, xLyH);
                        xHyL = ipInitial.getPixel(xL + 1, yL, xHyL);
                        xHyH = ipInitial.getPixel(xL + 1, yL + 1, xHyH);
                        for (int rr = 0; rr < 3; rr++) {
                            double newValue = (xL + 1 - x) * (yL + 1 - y) * xLyL[rr];
                            newValue += (x - xL) * (yL + 1 - y) * xHyL[rr];
                            newValue += (xL + 1 - x) * (y - yL) * xLyH[rr];
                            newValue += (x - xL) * (y - yL) * xHyH[rr];
                            rgbArray[rr] = (int) newValue;
                        }
                    }
                from here: https://imagej.nih.gov/ij/plugins/download/Polar_Transformer.java
                explanation:
                x,y represent the top-left of a 1x1 pixel
                if x or y are not whole numbers the 1x1 pixel area overlaps its neighbours,
                the returned pixel value is the sum of the overlap fractions of its neighbour
                pixel squares, P is the fractional pixel address in its pixel, 1, 2 and 3 are
                its neighbours, dotted area is contribution from neighbours:
                    +------+------+
                    |  P   |   1  |
                    |  ....|....  |  Ax = 1 - (Px - int(Px) = 1 - Px + int(Px) = (int(Px) + 1) - Px
                    |  . A | B .  |  Ay = 1 - (Py - int(Py) = 1 - Py + int(Py) = (int(Py) + 1) - Py
                    +------+------+  et al for B, C, D
                    |  . D | C .  |
                    |  ....|....  |
                    |  3   |   2  |
                    +----- +------+
                """
            cX: float = self.centre_x + x
            cY: float = self.centre_y + y
            xL: int = int(cX)
            yL: int = int(cY)
            xH: int = xL + 1
            yH: int = yL + 1
            pixel_xLyL = self.image.getpixel(xL, yL)
            pixel_xLyH = self.image.getpixel(xL, yH)
            pixel_xHyL = self.image.getpixel(xH, yL)
            pixel_xHyH = self.image.getpixel(xH, yH)
            if pixel_xLyL is None:
                pixel_xLyL = MIN_LUMINANCE
            if pixel_xLyH is None:
                pixel_xLyH = MIN_LUMINANCE
            if pixel_xHyL is None:
                pixel_xHyL = MIN_LUMINANCE
            if pixel_xHyH is None:
                pixel_xHyH = MIN_LUMINANCE
            ratio_xLyL = (xH - cX) * (yH - cY)
            ratio_xHyL = (cX - xL) * (yH - cY)
            ratio_xLyH = (xH - cX) * (cY - yL)
            ratio_xHyH = (cX - xL) * (cY - yL)
            part_xLyL = pixel_xLyL * ratio_xLyL
            part_xHyL = pixel_xHyL * ratio_xHyL
            part_xLyH = pixel_xLyH * ratio_xLyH
            part_xHyH = pixel_xHyH * ratio_xHyH
            pixel = int(round(part_xLyL + part_xHyL + part_xLyH + part_xHyH))
            return pixel

        limit_radius = self._radius(self.centre_x, self.centre_y, blob_size)

        # for detecting luminance variation for filtering purposes
        # make a new black image to build the projection in
        angle_delta = 360 / self.angle_steps
        code = self.original.instance().new(self.angle_steps, limit_radius, MIN_LUMINANCE)
        min_level = MAX_LUMINANCE
        max_level = MIN_LUMINANCE
        for radius in range(limit_radius):
            for angle in range(self.angle_steps):
                degrees = angle * angle_delta
                x, y = self.angle_xy(degrees, radius)
                if x is not None:
                    c = get_pixel(x, y)  # centre_x/y a in here
                    if c > MIN_LUMINANCE:
                        code.putpixel(angle, radius, c)
                        max_level = max(c, max_level)
                        min_level = min(c, min_level)

        # chuck out targets that do not have enough black/white contrast
        contrast = (max_level - min_level) / MAX_LUMINANCE
        if contrast < Scan.MIN_CONTRAST:
            if self.logging:
                self._log('project: dropping blob - contrast {:.2f} below minimum ({:.2f})'.
                          format(contrast, Scan.MIN_CONTRAST))
            return None, None

        # normalise image size
        max_x, max_y = code.size()
        if max_y < self.radial_steps:
            # increase image height to meet our min pixel per ring requirement
            code = self.transform.upheight(code, self.radial_steps)
            stretch_factor = self.radial_steps / max_y  # NB: >1
        elif max_y > self.radial_steps:
            # decrease image height to meet our min pixel per ring requirement
            code = self.transform.downheight(code, self.radial_steps)
            stretch_factor = self.radial_steps / max_y  # NB: <1
        else:
            # exact size - how likely is this?
            stretch_factor = 1

        if self.logging:
            self._log('project: projected image size {}x {}y (stretch factor {:.2f})'.
                      format(max_x, max_y, stretch_factor))

        if self.save_images:
            # draw cropped binary image of just this blob
            max_x, max_y = self.image.size()
            start_x = max(int(self.centre_x - limit_radius), 0)
            end_x = min(int(self.centre_x + limit_radius), max_x)
            start_y = max(int(self.centre_y - limit_radius), 0)
            end_y = min(int(self.centre_y + limit_radius), max_y)
            blob = self.transform.crop(self.image, start_x, start_y, end_x, end_y)
            # draw the detected blob in red
            k = (limit_radius, limit_radius, blob_size)
            blob = self.transform.label(blob, k, Scan.RED)
            self._unload(blob, '01-target')
            # draw the corresponding projected image
            self._unload(code, '02-projected')

        return code, stretch_factor

    def _binarize(self, target: frame.Frame,
                  width: float = 2, height: float = None,
                  black: float = -1, white: float = None, clean=True, suffix='') -> frame.Frame:
        """ binarize the given projected image,
            width is the fraction of the image width to use as the integration area,
            height is the fraction of the image height to use as the integration area (None==same as width),
            black is the % below the average that is considered to be the black/grey boundary,
            white is the % above the average that is considered to be the grey/white boundary,
            white of None means same as black and will yield a binary image,
            iff clean=True, 'tidy' it by removing short pixel sequences (1 or 2 'alone'),
            suffix is used to modify diagnostic image names
            """

        MAX_CLEAN_PASSES = 8  # cleaning can sometimes be unstable, so limit to this many passes

        max_x, max_y = target.size()

        def make_binary(image: frame.Frame, width: float=8, height: float=None, black: float=15, white: float=None) -> frame.Frame:
            """ given a greyscale image return a binary image using an adaptive threshold.
                width, height, black, white - see description of overall method
                See the adaptive-threshold-algorithm.pdf paper for algorithm details.
                """

            # the image being thresholded wraps in x, to allow for this when binarizing, we extend the
            # image by a half of its height or width to the left and right, then remove it from the binary

            # make a bigger buffer to copy image into
            x_extra = min(int(round(max_y / 2)), int(round(max_x / 2)))
            extended = np.zeros((max_y, max_x + (2 * x_extra)), np.uint8)

            # create the extended image
            right_extra_src_start = (max_x - 1) - x_extra + 1
            right_extra_dest_start = max_x + x_extra
            left_extra_src_start = 0
            left_extra_dest_start = 0
            for x in range(x_extra):
                for y in range(max_y):
                    right_extra_src = right_extra_src_start + x
                    left_extra_src = left_extra_src_start + x
                    right_extra_dest = right_extra_dest_start + x
                    left_extra_dest = left_extra_dest_start + x
                    extended[y, left_extra_dest] = image.getpixel(right_extra_src, y)
                    extended[y, right_extra_dest] = image.getpixel(left_extra_src, y)
            for x in range(max_x):
                for y in range(max_y):
                    extended[y, x + x_extra] = image.getpixel(x, y)

            # binarize our extended image buffer
            thresholded = contours.make_binary(extended, width, height, black, white)

            # extract the bit we want into a new buffer
            buffer = np.zeros((max_y, max_x), np.uint8)
            for x in range(max_x):
                for y in range(max_y):
                    buffer[y, x] = thresholded[y, x + x_extra]

            # make new image
            binary: frame.Frame = image.instance()
            binary.set(buffer)

            if self.save_images:
                self._unload(binary, '03-binary{}'.format(suffix))

            return binary

        buckets = make_binary(target, width, height, black, white)

        if clean:
            # clean the pixels - BWB or WBW sequences are changed to BBB or WWW
            # pixels wrap in the x direction but not in the y direction
            if self.logging:
                header = 'binarize:'
            passes = 0
            h_tot_black = 0
            h_tot_white = 0
            h_tot_grey = 0
            v_tot_black = 0
            v_tot_white = 0
            v_tot_grey = 0
            pass_changes = 1
            while pass_changes > 0 and passes < MAX_CLEAN_PASSES:
                passes += 1
                pass_changes = 0
                h_black = 0
                h_white = 0
                h_grey = 0
                v_black = 0
                v_white = 0
                v_grey = 0
                for x in range(max_x):  # x wraps, so consider them all
                    for y in range(1, max_y - 1):  # NB: ignoring first and last y - they do not matter
                        above = buckets.getpixel(x, y - 1)
                        left = buckets.getpixel((x - 1) % max_x, y)
                        this = buckets.getpixel(x, y)
                        right = buckets.getpixel((x + 1) % max_x, y)
                        below = buckets.getpixel(x, y + 1)
                        if left == right:
                            if this != left:
                                # got a horizontal loner
                                buckets.putpixel(x, y, left)
                                pass_changes += 1
                                if left == MIN_LUMINANCE:
                                    h_black += 1
                                elif left == MAX_LUMINANCE:
                                    h_white += 1
                                else:
                                    h_grey += 1
                        elif False:  # ToDo: makes it unstable-->left != this and this != right and left != right and this != MID_LUMINANCE:
                            # all different and middle not grey, make middle grey
                            buckets.putpixel(x, y, MID_LUMINANCE)
                            pass_changes += 1
                            h_grey += 1
                        elif above == below:
                            # only look for vertical when there is no horizontal candidate, else it can oscillate
                            if this != above:
                                # got a vertical loner
                                # this condition is lower priority than above
                                buckets.putpixel(x, y, above)
                                pass_changes += 1
                                if above == MIN_LUMINANCE:
                                    v_black += 1
                                elif above == MAX_LUMINANCE:
                                    v_white += 1
                                else:
                                    v_grey += 1
                        elif above != this and this != below and above != below and this != MID_LUMINANCE:
                            # all different and middle not grey, make middle grey
                            buckets.putpixel(x, y, MID_LUMINANCE)
                            pass_changes += 1
                            v_grey += 1
                if self.logging and pass_changes > 0:
                    if header is not None:
                        self._log(header)
                        header = None
                    self._log('    pass {}: cleaning lone pixels: changes this pass: {}'.
                              format(passes, pass_changes))
                    self._log('        horizontal: changed {} to white, {} to black, {} to grey'.
                              format(h_white, h_black, h_grey))
                    self._log('        vertical: changed {} to white, {} to black, {} to grey'.
                              format(v_white, v_black, v_grey))
                h_tot_black += h_black
                h_tot_white += h_white
                h_tot_grey += h_grey
                v_tot_black += v_black
                v_tot_white += v_white
                v_tot_grey += v_grey
            if self.logging:
                if header is not None:
                    self._log(header)
                    header = None
                if pass_changes > 0:
                    # this means cleaning oscillated
                    self._log('    **** cleaning incomplete, {} changes not cleaned'.format(pass_changes))
                self._log('    cleaned lone pixels in {} passes'.format(passes))
                self._log('        horizontal: total changed {} to white, {} to black, {} to grey'.
                          format(h_tot_white, h_tot_black, h_tot_grey))
                self._log('        vertical: total changed {} to white, {} to black, {} to grey'.
                          format(v_tot_white, v_tot_black, v_tot_grey))

            if self.save_images:
                self._unload(buckets, '04-buckets{}'.format(suffix))

        return buckets

    def _slices(self, buckets: frame.Frame) -> List[List[Step]]:
        """ detect radial (across the target rings) luminance steps in the given binary/tertiary image,
            returns the step list for the image,
            each slice is expected to consist of a leading falling edge, then a further rising and falling edge,
            these represent ring boundaries,
            if there is no leading falling edge its an all white slice and steps are created to indicate that,
            if there is no trailing rising edge its an all black slice and steps are created to indicate that,
            it is guaranteed that at least one step is created for every x co-ord in the image,
            the y co-ord of a step is always that of the higher luminance pixel
            """

        max_x, max_y = buckets.size()

        # build list of transitions
        slices = [[] for _ in range(max_x)]
        for x in range(max_x):
            last_pixel = buckets.getpixel(x, 0)
            transitions = 0
            for y in range(1, max_y):
                pixel = buckets.getpixel(x, y)
                if pixel < last_pixel:
                    # falling step
                    slices[x].append(Scan.Step(y-1, Scan.FALLING, last_pixel, pixel))
                    transitions += 1
                elif pixel > last_pixel:
                    # rising step
                    slices[x].append(Scan.Step(y, Scan.RISING, last_pixel, pixel))
                    transitions += 1
                last_pixel = pixel
            if transitions == 0:
                # this probably means a big pulse has merged with the inner and the outer edge,
                if last_pixel == MAX_LUMINANCE:
                    # its all white - not possible?
                    # create a rising step at 0 and a falling step at max_y
                    slices[x].append(Scan.Step(0, Scan.RISING, MIN_LUMINANCE, MAX_LUMINANCE))
                    slices[x].append(Scan.Step(max_y-1, Scan.FALLING, MAX_LUMINANCE, MIN_LUMINANCE))
                    if self.logging:
                        self._log('slices: {}: all white and no transitions, creating {}, {}'.
                                  format(x, slices[x][-2], slices[x][-1]))
                elif last_pixel == MIN_LUMINANCE:
                    # its all black - not possible?
                    # create a falling step at 0 and a rising step at max_y
                    slices[x].append(Scan.Step(0, Scan.FALLING, MAX_LUMINANCE, MIN_LUMINANCE))
                    slices[x].append(Scan.Step(max_y-1, Scan.RISING, MIN_LUMINANCE, MAX_LUMINANCE))
                    if self.logging:
                        self._log('slices: {}: all black and no transitions, creating {}, {}'.
                                  format(x, slices[x][-2], slices[x][-1]))
                else:
                    # its all grey - this means all pixels are nearly the same in the integration area
                    # treat as if all white
                    # create a rising step at 0 and a falling step at max_y
                    slices[x].append(Scan.Step(0, Scan.RISING, MIN_LUMINANCE, MAX_LUMINANCE))
                    slices[x].append(Scan.Step(max_y-1, Scan.FALLING, MAX_LUMINANCE, MIN_LUMINANCE))
                    if self.logging:
                        self._log('slices: {}: all grey and no transitions, creating {}, {}'.
                                  format(x, slices[x][-2], slices[x][-1]))

        if self.logging:
            self._log('slices: {}'.format(len(slices)))
            for x, slice in enumerate(slices):
                steps = ''
                for step in slice:
                    steps += ', {}'.format(step)
                steps = steps[2:]
                self._log('    {:3d}: {}'.format(x, steps))

        return slices

    def _edges(self, slices: List[List[Step]], mode, max_y, from_y:[int]=None) -> [Edge]:
        """ build a list of falling or rising edges of our target, mode is FALLING or RISING,
            iff from_y is given it is a list of the starting y co-ords for each x,
            FALLING is optimised for detecting the inner white->black edge and
            RISING is optimised for detecting the outer black->white edge,
            returns the falling or rising edges list in increasing co-ordinate order,
            an 'edge' here is a sequence of connected rising or falling Steps,
            """

        if mode == Scan.FALLING:
            # we want any fall to min, do not want a half fall to mid
            # so max to min or mid to min qualifies, but max to mid does not,
            # we achieve this by only considering falling edges where the 'to' pixel is min
            context = 'falling-'
        elif mode == Scan.RISING:
            # we want any rise, including a half rise
            # so min to mid or min to max qualifies, but mid to max does not,
            # we achieve this by only considering rising edges where the 'from' pixel is min
            context = 'rising-'
        else:
            raise Exception('_edges: mode must be {} or {} not {}'.format(Scan.FALLING, Scan.RISING, mode))

        max_x = len(slices)
        used = [[False for _ in range(max_y)] for _ in range(max_x)]
        edges = []

        def make_candidates(start_x, start_y, edge_type):
            """ make candidate edges from the step from start_x,
                pixel pairs at x and x +/- 1 and x and x +/- 2 are considered,
                the next y is the closest of those 2 pairs,
                returns one or more instances of Edge or None
                """

            def get_nearest_y(x, y, step_type):
                """ find the y with the minimum acceptable gap to the given y at x for the given step type,
                    returns y or None iff nothing close enough
                    """

                if used[x][y]:
                    # already been here
                    return None

                min_y = None
                min_gap = max_y * max_y

                slice = slices[x]
                if slice is not None:
                    for step in slice:
                        if step.type != step_type:
                            continue
                        if used[x][step.where]:
                            # already been here so not a candidate for another edge
                            continue
                        if step.type == Scan.FALLING:
                            to_pixel = step.to_pixel
                            if to_pixel == MID_LUMINANCE:
                                to_pixel = treat_grey_as
                            if to_pixel == MIN_LUMINANCE:
                                # got a qualifying falling step
                                pass
                            else:
                                # ignore this step
                                continue
                        else:  # step.type == Scan.RISING
                            to_pixel = step.to_pixel
                            if to_pixel == MID_LUMINANCE:
                                to_pixel = treat_grey_as
                            if to_pixel == MAX_LUMINANCE:
                                # got a qualifying rising step
                                pass
                            else:
                                # ignore this step
                                continue
                        gap = wrapped_gap(y, step.where, max_y)
                        gap *= gap
                        if gap < min_gap:
                            min_gap = gap
                            min_y = step.where

                if min_gap > Scan.MAX_NEIGHBOUR_HEIGHT_GAP_SQUARED:
                    min_y = None

                return min_y

            def make_edge(sequence):
                """ make an Edge instance from the given y co-ord sequence """
                # find the edge ends (==between None to not None and not None to None in candidate)
                start_x = None
                end_x = None
                for x in range(max_x):
                    prev_x = (x - 1) % max_x
                    this_y = sequence[x]
                    prev_y = sequence[prev_x]
                    if prev_y is None and this_y is not None:  # None to not-None is a start
                        start_x = x
                    elif prev_y is not None and this_y is None:  # not-None to None is an end
                        end_x = prev_x
                    if start_x is not None and end_x is not None:
                        # there can only be 1 sequence, so we're done when got both ends
                        break
                # make the Edge instance
                if start_x is None:
                    # this means this edge goes all the way around
                    return Scan.Edge(0, edge_type, sequence)
                elif end_x is None:
                    # this means the edge ends at the end
                    return Scan.Edge(start_x, edge_type, sequence[start_x:max_x])
                elif end_x < start_x:  # NB: end cannot be < start unless wrapping is allowed
                    # this means the edge wraps
                    return Scan.Edge(start_x, edge_type, sequence[start_x:max_x] + sequence[0:end_x+1])
                else:
                    # normal edge away from either extreme
                    return Scan.Edge(start_x, edge_type, sequence[start_x:end_x+1])

            # follow the edge as far as we can (in both directions)
            backwards = [None for _ in range(max_x)]
            forwards = [None for _ in range(max_x)]
            samples = 0
            for offset, increment, candidate in ((1, 1, forwards), (0, -1, backwards)):
                x = start_x - offset
                y = start_y
                for _ in range(max_x):  # do up to one revolution either backwards or forwards
                    x += increment
                    x %= max_x
                    this_y = get_nearest_y(x, y, edge_type)
                    if this_y is None:
                        # direct neighbour not connected, see if indirect one is
                        # this allows us to skip a slice (it may be noise)
                        dx = x + increment
                        dx %= max_x
                        next_y = get_nearest_y(dx, y, edge_type)
                        if next_y is None:
                            # indirect not connected either, so that's it
                            break
                        # indirect neighbour is connected, use extrapolation of that from where we are
                        y = int(round((y + next_y) / 2))
                    else:
                        # direct neighbour connected, use that
                        y = this_y
                    candidate[x] = y
                    used[x][y] = True  # note used this x,y (to stop it being considered again)
                    samples += 1

            if samples >= Scan.MIN_EDGE_SAMPLES:
                # we've got an edge that is long enough to consider,
                # forwards now contains all the connected y co-ords going forward from start_x + 1 and
                # backwards contains all the connected y co-ords going backward from start_x, they may
                # overlap if the edge 'spirals', in which case we have ambiguity as we do not know which
                # overlap region is best, we resolve this by removing the overlap from the main sequence
                # and creating two short sequences for the overlap region
                edges = []
                if samples > max_x:
                    # we've got an overlap, split things up
                    main_sequence = [None for _ in range(max_x)]
                    forward_tail = [None for _ in range(max_x)]
                    backward_tail = [None for _ in range(max_x)]
                    main_samples = 0
                    for x in range(max_x):
                        forward_sample = forwards[x]
                        backward_sample = backwards[x]
                        if forward_sample is not None and backward_sample is not None:
                            # got an overlap, add to tails
                            backward_tail[x] = backward_sample
                            forward_tail[x] = forward_sample
                        elif forward_sample is not None:
                            # no overlap, add to main sequence
                            main_sequence[x] = forward_sample
                            main_samples += 1
                        elif backward_sample is not None:
                            # no overlap, add to main sequence
                            main_sequence[x] = backward_sample
                            main_samples += 1
                        else:
                            # eh?
                            raise Exception('sample missing at {} in spiralled edge'.format(x))
                    if main_samples > 0:
                        # this means there is a separation between the tails
                        edges.append(make_edge(main_sequence))
                        edges.append(make_edge(forward_tail))
                        edges.append(make_edge(backward_tail))
                    else:
                        # this means both forwards and backwards go right round, they must be the same in this case
                        edges.append(make_edge(forward_tail))
                else:
                    # no overlap, make a single candidate
                    candidate = backwards
                    for x, sample in enumerate(forwards):
                        if sample is not None:
                            candidate[x] = sample
                    edges.append(make_edge(candidate))
                return edges
            else:
                # did not find anything long enough to be useful
                return None

        # build the edges list
        for treat_grey_as in [MIN_LUMINANCE, MAX_LUMINANCE]:
            for x, slice in enumerate(slices):
                for step in slice:
                    if step.type != mode:
                        # not the step type we are looking for
                        continue
                    if from_y is not None:
                        if step.where <= from_y[x]:
                            # too close to top
                            continue
                    candidates = make_candidates(x, step.where, step.type)
                    if candidates is not None:
                        for candidate in candidates:
                            edges.append(candidate)

        if self.logging:
            self._log('{}edges: {} edges'.format(context, len(edges)))
            for edge in edges:
                if len(edge.samples) > 16:
                    self._log('    {}'.format(edge.show(0, max_x)))
                    self._log_edge(edge.samples, '        ')
                else:
                    self._log('    {}'.format(edge.show(16, max_x)))

        return edges

    def _extent(self, max_x, edges: [Edge]) -> ([int], str):
        """ determine the target inner or outer edge, the Edge type determines which,
            there should be a consistent set of falling/rising edges for the inner/outer black ring,
            edges that are within a few pixels of each other going right round is what we want,
            returns a list of y co-ords and fail reason (None iff not failed)
            """

        # region helpers...
        def make_full(edge):
            """ expand the given edge to fully encompass all x's """
            full_edge = [None for _ in range(max_x)]
            return merge(full_edge, edge)

        def merge(full_edge, edge):
            """ merge the given edge into full_edge, its assumed it will 'fit' """
            for dx, y in enumerate(edge.samples):
                if y is not None:
                    x = (edge.where + dx) % max_x
                    full_edge[x] = y
            return full_edge

        def delta(this_xy, that_xy):
            """ calculate the x and y difference between this_xy and that_xy,
                x' is this x - that x and y' is this y - that y, x' cannot be zero,
                this x and that x may also wrap (ie. that < this),
                this_xy must precede that_xy in x, if not a wrap is assumed,
                """
            if that_xy[0] < this_xy[0]:
                # wraps in x
                x_dash = (that_xy[0] + max_x) - this_xy[0]
            else:
                x_dash = that_xy[0] - this_xy[0]
            if that_xy[1] < this_xy[1]:
                y_dash = this_xy[1] - that_xy[1]
            else:
                y_dash = that_xy[1] - this_xy[1]
            return x_dash, y_dash

        def distance_OK(this_xy, that_xy, max_distance):
            """ check the distance between this_xy and that_xy is acceptable,
                this_xy must precede that_xy in x, if not a wrap is assumed,
                """
            xy_dash = delta(this_xy, that_xy)
            if xy_dash[0] > max_distance:
                return False
            return True

        def angle_OK(this_xy, that_xy, max_angle):
            """ check the angle of the slope between this_xy and that_xy is acceptable,
                an approximate tan of the angle is calculated as y'/x' with a few special cases,
                the special cases are when y' is within Scan.MAX_NEIGHBOUR_HEIGHT_GAP,
                these are all considered OK,
                this_xy must precede that_xy in x, if not a wrap is assumed,
                """
            xy_dash = delta(this_xy, that_xy)
            if xy_dash[1] <= Scan.MAX_NEIGHBOUR_HEIGHT_JUMP:
                # always OK
                return True
            if xy_dash[0] == 0:
                # this is angle 0, which is OK
                return True
            angle = xy_dash[1] / xy_dash[0]  # 1==45 degrees, 0.6==30, 1.2==50, 1.7==60, 2.7==70
            if angle > max_angle:
                # too steep
                return False
            return True

        def find_nearest_y(full_edge, start_x, direction):
            """ find the nearest y, and its position, in full_edge from x, in the given direction,
                direction is +1 to look forward in full_edge, or -1 to look backward
                we assume full_edge at start_x is empty
                """
            dx = start_x + direction
            for _ in range(max_x):
                x = int(dx % max_x)
                y = full_edge[x]
                if y is not None:
                    return x, y
                dx += direction
            return None

        def intersection(full_edge, edge):
            """ get the intersection between full_edge and edge,
                returns a list for every x with None or a y co-ord tuple when there is an overlap
                and a count of overlaps
                """
            overlaps = [None for _ in range(max_x)]
            samples = 0
            for dx, edge_y in enumerate(edge.samples):
                if edge_y is None:
                    continue
                x = (edge.where + dx) % max_x
                full_edge_y = full_edge[x]
                if full_edge_y is None:
                    continue
                if edge_y == full_edge_y:
                    continue
                overlaps[x] = (full_edge_y, edge_y)
                samples += 1
            return samples, overlaps

        def can_merge(full_edge, edge, max_distance, max_angle):
            """ check if edge can be merged into full_edge,
                full_edge is a list of y's or None's for every x, edge is the candidate to check,
                max_distance is the distance beyond which a merge is not allowed,
                max_angle is the maximum tolerated slope angle between two edge ends,
                returns True if it can be merged,
                to be allowed its x, y values must not to too far from what's there already,
                the overlap condition is assumed to have already been checked (by trim_overlap)
                """

            if len(edge.samples) == 0:
                # its been trimmed away, so nothing left to merge with
                return False

            # check at least one edge end is 'close' to existing ends
            # close is in terms of the slope angle across the gap and the width of the gap,
            edge_start = edge.where
            edge_end = edge_start + len(edge.samples) - 1
            nearest_back_xy = find_nearest_y(full_edge, edge_start, -1)  # backwards form our start
            nearest_next_xy = find_nearest_y(full_edge, edge_end, +1)  # forwards from our end
            if nearest_back_xy is None or nearest_next_xy is None:
                # means full_edge is empty - always OK
                return True

            our_start_xy = (edge.where, edge.samples[0])
            our_end_xy = (edge_end, edge.samples[-1])
            if angle_OK(nearest_back_xy, our_start_xy, max_angle) \
                    and distance_OK(nearest_back_xy, our_start_xy, max_distance):
                # angle and distance OK from our start to end of full
                return True
            if angle_OK(our_end_xy, nearest_next_xy, max_angle) \
                    and distance_OK(our_end_xy, nearest_next_xy, max_distance):
                # angle and distance OK from our end to start of full
                return True

            # too far away and/or too steep, so not allowed
            return False

        def trim_overlap(full_edge, edge, direction=None):
            """ where there is a small overlap between edge and full_edge return modified
                versions with the overlap removed, it returns *copies*, the originals are
                left as is, if the overlap constraint is not met the function returns None,
                overlaps can occur in heavily distorted images where the inner or outer edge
                has collided with a data edge,
                when the overlap is 'small' we either take it out of full_edge or edge,
                which is taken out is controlled by the direction param, if its FALLING the
                edge lower in the image (i.e. higher y) is taken out, otherwise higher is removed
                """

            def remove_sample(x, trimmed_edge):
                """ remove the sample at x from trimmed_edge,
                    returns True iff succeeded, else False
                    """
                if x == trimmed_edge.where:
                    # overlap at the front
                    trimmed_edge.where = (trimmed_edge.where + 1) % max_x  # move x past the overlap
                    del trimmed_edge.samples[0]  # remove the overlapping sample
                elif x == (trimmed_edge.where + len(trimmed_edge.samples) - 1):
                    # overlap at the back
                    del trimmed_edge.samples[-1]  # remove the overlapping sample
                else:
                    # overlap in the middle - not allowed
                    return False
                    # trimmed_edge.samples[(x - trimmed_edge.where) % max_x] = None
                return True

            samples, overlaps = intersection(full_edge, edge)
            if samples == 0:
                # no overlap
                return full_edge, edge
            if samples > Scan.MAX_NEIGHBOUR_OVERLAP:
                # too many overlaps
                return None

            if direction is None:
                # being told not to do it
                return full_edge, edge

            trimmed_full_edge = full_edge.copy()
            trimmed_edge = Scan.Edge(edge.where, edge.type, edge.samples.copy())
            for x, samples in enumerate(overlaps):
                if samples is None:
                    continue
                full_edge_y, edge_y = samples
                if edge_y > full_edge_y:
                    if direction == Scan.RISING:
                        # take out the full_edge sample - easy case
                        trimmed_full_edge[x] = None
                    else:
                        # take out the edge sample
                        if not remove_sample(x, trimmed_edge):
                            return None
                elif edge_y < full_edge_y:
                    if direction == Scan.FALLING:
                        # take out the full_edge sample - easy case
                        trimmed_full_edge[x] = None
                    else:
                        # take out the edge sample
                        if not remove_sample(x, trimmed_edge):
                            return None
                else:  # edge_y == full_edge_y:
                    # do nothing
                    continue

            return trimmed_full_edge, trimmed_edge

        def smooth(edge, direction):
            """ smooth out any excessive y 'steps',
                direction RISING when doing an outer edge or FALLING when doing an inner,
                when joining edges we only check one end is close to another, the other end
                may be a long way off (due to overlaps caused by messy edges merging in the image),
                we detect these and 'smooth' them out, we detect them by finding successive y's
                with a difference of more than two, on detection the correction is to move the two y's
                such that the y gaps' equally span the gap, e.g.:
                ---------1.           ---------..
                         ..                    ..
                         ..                    1.
                (A)      ..       -->          ..
                         ..                    .2
                         ..                    ..
                         .2---------           ..--------
                or
                         .2---------           ..--------
                         ..                    ..
                         ..                    .2
                (B)      ..       -->          ..
                         ..                    1.
                         ..                    ..
                ---------1.           ---------..
                we also remove 'nipples', this is an 'x' where its left and right neighbours have
                the same 'y', we make 'x' the same as its neighbours,
                returns the smoothed edge and None if OK or partially smoothed and a reason if failed
                """

            # phase 1 - smooth jumps in y
            reason = None
            max_edge_jump = Scan.MAX_EDGE_HEIGHT_JUMP * Scan.MAX_EDGE_HEIGHT_JUMP
            max_x = len(edge)
            for this_x in range(max_x):
                last_x = (this_x - 1) % max_x
                next_x = (this_x + 1) % max_x
                last_y = edge[last_x]
                this_y = edge[this_x]
                next_y = edge[next_x]
                diff_y = this_y - last_y
                jump = diff_y * diff_y  # get rid of the sign
                if jump > max_edge_jump:
                    # too big a jump to tolerate
                    return edge, 'edge broken'
                elif jump > 4:  # i.e. diff_y > 2
                    # 'smooth' this small jump
                    gap = int(round(diff_y / 3))
                    last_y += gap
                    this_y -= gap
                elif jump > 1:  # i.e. diff_y == 2
                    # we move one end by 1, which end?
                    gap = int(round(diff_y / 2))  # we do this to propagate the sign of diff_y
                    this_y -= gap  # arbitrarily choosing this end
                else:
                    # no smoothing required
                    continue
                edge[last_x] = last_y
                edge[this_x] = this_y
            # phase 2 - remove 'nipples'
            for this_x in range(max_x):
                last_x = (this_x - 1) % max_x
                next_x = (this_x + 1) % max_x
                last_y = edge[last_x]
                next_y = edge[next_x]
                if last_y == next_y:
                    # got a 'nipple', remove it
                    edge[this_x] = next_y
                else:
                    # no nipple
                    continue

            return edge, reason

        def extrapolate(edge):
            """ extrapolate across the gaps in the given edge,
                this can fail if a gap is too big,
                returns the updated edge and None if OK or the partial edge and a fail reason if not OK
                """

            def fill_gap(edge, size, start_x, stop_x, start_y, stop_y):
                """ fill a gap by linear extrapolation across it """
                if size < 1:
                    # nothing to fill
                    return edge
                delta_y = (stop_y - start_y) / (size + 1)
                x = start_x - 1
                y = start_y
                for _ in range(max_x):
                    x = (x + 1) % max_x
                    if x == stop_x:
                        break
                    y += delta_y
                    edge[x] = int(round(y))
                return edge

            max_gap = max_x * Scan.MAX_EDGE_GAP_SIZE

            # we need to start from a known position, so find the first non gap
            reason = None
            start_x = None
            for x in range(max_x):
                if edge[x] is None:
                    # found a gap, find the other end as our start point
                    start_gap = x
                    for _ in range(max_x):
                        x = (x + 1) % max_x
                        if x == start_gap:
                            # gone right round, eh?
                            break
                        if edge[x] is None:
                            # still in the gap, keep looking
                            continue
                        # found end of a gap, start our scan from there
                        start_x = x
                        break
                else:
                    start_x = x
                    break
            if start_x is not None:
                # found end of a gap, so there is at least one extrapolation to do
                start_y = edge[start_x]  # we know this is not None due to the above loop
                x = start_x
                gap_start = None
                gap_size = 0
                for _ in range(max_x):
                    x = (x + 1) % max_x
                    y = edge[x]
                    if x == start_x:
                        # got back to the start, so that's it
                        if gap_start is not None:
                            edge = fill_gap(edge, gap_size, gap_start, x, start_y, y)  # fill any final gap
                        break
                    if y is None:
                        if gap_start is None:
                            # start of a new gap
                            gap_start = x
                            gap_size = 1
                        else:
                            # current gap getting bigger
                            gap_size += 1
                            if gap_size > max_gap:
                                reason = 'edge gap: {}+'.format(gap_size)
                                break
                        continue
                    if gap_start is not None:
                        # we're coming out of a gap, extrapolate across it
                        edge = fill_gap(edge, gap_size, gap_start, x, start_y, y)
                    # no longer in a gap
                    gap_start = None
                    gap_size = 0
                    start_y = y

            return edge, reason

        def compose(edges, direction):
            """ we attempt to compose a complete edge by starting at every edge, then adding
                near neighbours until no more merge, then pick the longest and extrapolate
                across any remaining gaps,
                returns the edge and None or the partial edge and a fail reason if not OK
                """

            distance_span = range(2, Scan.MAX_NEIGHBOUR_LENGTH_JUMP + 1)

            if direction == Scan.FALLING:
                max_angle = Scan.MAX_NEIGHBOUR_ANGLE_INNER
            else:
                max_angle = Scan.MAX_NEIGHBOUR_ANGLE_OUTER

            full_edges = []
            for start_idx, start_edge in enumerate(edges):
                full_edge = make_full(start_edge)
                full_edge_samples = len(start_edge.samples)
                full_edge_fragments = [start_idx]  # note edges we've used
                for distance in distance_span:
                    if full_edge_samples == max_x:
                        # this is as big as its possible to get
                        break
                    merged = True
                    while merged and full_edge_samples < max_x:
                        merged = False
                        for edge_idx in range(start_idx + 1, len(edges)):
                            if edge_idx in full_edge_fragments:
                                # ignore if already used this one
                                continue
                            edge = edges[edge_idx]
                            trimmed = trim_overlap(full_edge, edge, direction)
                            if trimmed is None:
                                # fails overlap constraint
                                continue
                            if can_merge(trimmed[0], trimmed[1], distance, max_angle):
                                full_edge = merge(trimmed[0], trimmed[1])
                                full_edge_fragments.append(edge_idx)
                                merged = True
                        if merged:
                            full_edge_samples = 0
                            for x in full_edge:
                                if x is not None:
                                    full_edge_samples += 1
                # full_edge is now as big as we can get within our distance limit
                full_edges.append((full_edge_samples, len(full_edge_fragments), full_edge))

            # sort into longest order with fewest fragments
            full_edges.sort(key=lambda e: (e[0], e[0] - e[1]), reverse=True)

            if self.logging:
                for e, edge in enumerate(full_edges):
                    self._log('extent: {} #{} of {}: full edge candidate length {}, fragments {}'.
                              format(direction, e + 1, len(full_edges), edge[0], edge[1]))
                    self._log_edge(edge[2], '    ')

            if len(full_edges) == 0:
                # no edges detected!
                return [None for _ in range(max_x)], 'no edges'

            # extrapolate across any remaining gaps in the longest edge
            composed, reason = extrapolate(full_edges[0][2])
            if reason is not None:
                # failed
                return composed, reason

            # remove y 'steps'
            smoothed, reason = smooth(composed, direction)

            return smoothed, reason
        # endregion

        if len(edges) > 0:
            direction = edges[0].type  # they must all be the same
            # make the edge
            edge, fail = compose(edges, direction)
        else:
            # nothing to find the extent of
            direction = 'unknown'
            edge = None
            fail = 'no edges'

        if self.logging:
            self._log('extent: {} (fail={})'.format(direction, fail))
            self._log_edge(edge, '    ')

        return edge, fail

    def _flatten(self, image: frame.Frame, extent: Extent) -> (frame.Frame, float):
        """ remove perspective distortions from the given projected image and its inner/outer edges,
            this helps to mitigate ring width distortions which is significant when we analyse the image,
            a circle when not viewed straight on appears as an ellipse, when that is projected into a rectangle
            the radius edges become 'wavy' (a sine wave), this function straightens those wavy edges, other
            distortions can arise if the target is curved (e.g. if it is wrapped around someone's leg), in
            this case the outer rings are even more 'wavy', for the purposes of this function the distortion
            is assumed to be proportional to the variance in the inner and outer edge positions, we know between
            the inner and outer edges there are N rings, we apply a stretching factor to each ring that is a
            function of the inner and outer edge variance,
            the returned image is re-scaled to just cover the target rings
            """

        def get_extent(data_width):
            """ given a data width return the corresponding inner start, outer end, and image height,
                the start of the inner edge is where the first black pixel is expected,
                the end of the outer edge is where the first white pixel is expected,
                the image height is the resultant image height required to encompass all the elements
                """

            ring_size = data_width / Scan.INNER_OUTER_SPAN_RINGS
            min_ring_size = max(self.cells[1], Scan.MIN_PIXELS_PER_RING)
            ring_size = max(ring_size, min_ring_size)

            new_data_width = int(math.ceil(ring_size * Scan.INNER_OUTER_SPAN_RINGS))
            new_inner_width = int(math.ceil(ring_size * Scan.BULLSEYE_RINGS))
            new_outer_width = int(math.ceil(ring_size))

            new_inner_start = new_inner_width
            new_outer_end = new_inner_start + new_data_width
            new_image_height = new_outer_end + new_outer_width

            return new_inner_start, new_outer_end, new_image_height

        def stretch_pixels(pixels, size):
            """ given a vector of pixels stretch/shrink them such that the resulting pixels is size long,
                linear interpolation is done between stretched pixels,
                see https://towardsdatascience.com/image-processing-image-scaling-algorithms-ae29aaa6b36c
                """

            dest = [None for _ in range(size)]
            scale_y = size / len(pixels)

            if size < len(pixels):
                # we're shrinking
                for y in range(size):
                    y_scaled = min(int(round(y / scale_y)), len(pixels) - 1)
                    pixel = pixels[y_scaled]
                    dest[y] = pixel

            else:
                # we're stretching

                for y in range(size):
                    y_scaled = y / scale_y

                    y_above = math.floor(y_scaled)
                    y_below = min(math.ceil(y_scaled), len(pixels) - 1)

                    if y_above == y_below:
                        pixel = pixels[y_above]
                    else:
                        pixel_above = pixels[y_above]
                        pixel_below = pixels[y_below]
                        # Interpolating P
                        pixel = ((y_below - y_scaled) * pixel_above) + ((y_scaled - y_above) * pixel_below)

                    dest[y] = int(round(pixel))

            return dest

        def stretch_slice(x, src_image, src_start_y, src_end_y, dst_image, dst_start_y, dst_end_y):
            """ stretch a slice of pixels at x co-ord from src_image to dst_image,
                pixels between src_start_y and src_end_y (inclusive) are
                stretched between dst_start_y and dst_end_y (also inclusive)
                """
            in_range = range(src_start_y, src_end_y + 1)
            in_pixels = [None for _ in in_range]
            out_y = 0
            for in_y in in_range:
                in_pixels[out_y] = src_image.getpixel(x, in_y)
                out_y += 1
            # stretch to the out pixels
            out_pixels = stretch_pixels(in_pixels, dst_end_y - dst_start_y + 1)
            # move out pixels to our image
            out_y = dst_start_y
            for y in range(len(out_pixels)):
                dst_image.putpixel(x, out_y, out_pixels[y])
                out_y += 1

        ring_inner_edge = extent.inner
        ring_outer_edge = extent.outer

        # get the edge limits we need
        max_x, projected_y = image.size()

        # we stretch the data to the max distance between inner and outer,
        # the inner and outer are stretched/shrunk to fit the ring sizes implied by the data,
        # go find the stretch size
        max_inner_outer_distance = 0
        for x in range(max_x):
            inner_edge = ring_inner_edge[x]
            outer_edge = ring_outer_edge[x]
            inner_outer_distance = outer_edge - inner_edge
            if inner_outer_distance > max_inner_outer_distance:
                max_inner_outer_distance = inner_outer_distance

        extent = get_extent(max_inner_outer_distance)

        if self.logging:
            self._log('flatten: max inner/outer distance is {}, new extent is {}'.
                      format(max_inner_outer_distance, extent))

        # create a new image to flatten into
        flat_y = extent[2]
        code = self.original.instance().new(max_x, flat_y, MID_LUMINANCE)

        # build flat image
        for x in range(max_x):
            # stretch/shrink the inner pixels
            stretch_slice(x,
                          image, 0, ring_inner_edge[x],
                          code, 0, extent[0] - 1)
            # stretch the data pixels
            stretch_slice(x,
                          image, ring_inner_edge[x] + 1, ring_outer_edge[x] - 1,
                          code, extent[0], extent[1] - 1)
            # stretch/shrink the outer pixels
            stretch_slice(x,
                          image, ring_outer_edge[x], projected_y - 1,
                          code, extent[1], extent[2] - 1)

        # calculate the flattened image stretch factor
        new_stretch = flat_y / projected_y

        if self.save_images:
            self._unload(code, '05-flat')
        if self.logging:
            self._log('flatten: flattened image size {}x {}y, stretch factor={:.2f}, '.
                      format(max_x, flat_y, new_stretch))

        # return flattened image and its stretch factor
        return code, new_stretch

    def _identify(self, blob_size):
        """ identify the target in the current image with the given blob size (its radius) """

        def make_extent(target, clean=True, context='') -> Scan.Extent:
            """ binarize and inner/outer edge detect the given image """

            if self.logging:
                self._log('identify: phase{}'.format(context))

            max_x, max_y = target.size()

            buckets = self._binarize(target,
                                     width=Scan.THRESHOLD_WIDTH, height=Scan.THRESHOLD_HEIGHT,
                                     black=Scan.THRESHOLD_BLACK, white=Scan.THRESHOLD_WHITE,
                                     clean=clean, suffix=context)
            slices = self._slices(buckets)
            falling_edges = self._edges(slices, Scan.FALLING, max_y)
            inner, inner_fail = self._extent(max_x, falling_edges)
            if inner_fail is None:
                rising_edges = self._edges(slices, Scan.RISING, max_y, from_y=inner)
                outer, outer_fail = self._extent(max_x, rising_edges)
            else:
                rising_edges = None
                outer = None
                outer_fail = None

            extent = Scan.Extent(inner=inner, inner_fail=inner_fail,
                                 outer=outer, outer_fail=outer_fail,
                                 buckets=buckets,
                                 falling_edges=falling_edges, rising_edges=rising_edges)

            if self.save_images:
                plot = target
                plot = self._draw_edges((extent.falling_edges, extent.rising_edges), plot, extent)
                self._unload(plot, '04-edges{}'.format(context))

            return extent

        # do the polar to cartesian projection
        target, stretch_factor = self._project(blob_size)
        if target is None:
            # its been rejected
            return None

        # do the initial edge detection
        extent = make_extent(target, clean=True, context='-warped')

        # if extent.inner_fail is None and extent.outer_fail is None:
        #     flat, flat_stretch = self._flatten(target, extent)
        #     extent = make_extent(flat, clean=True, context='-flat')
        #     max_x, max_y = flat.size()
        #     return max_x, max_y, stretch_factor * flat_stretch, extent

        max_x, max_y = target.size()
        return max_x, max_y, stretch_factor, extent

    def _measure(self, extent: Extent, stretch_factor=1, log=True):
        """ get a measure of the target size by examining the extent,
            stretch_factor is how much the image height was stretched during projection,
            its used to re-scale the target size such that all are consistent wrt the original image
            """

        max_x = len(extent.inner)
        inner_average = 0
        outer_average = 0
        for x in range(max_x):
            inner_average += extent.inner[x]
            outer_average += extent.outer[x]
        inner_average /= max_x
        outer_average /= max_x
        inner_size = inner_average / stretch_factor
        outer_size = outer_average / stretch_factor

        if log and self.logging:
            self._log('measure: inner size: {:.2f}, outer size: {:.2f} (stretch factor: {:.2f})'.
                      format(inner_size, outer_size, stretch_factor))

        return inner_size, outer_size

    def _make_slice(self, digit, extent):
        """ given a digit return the equivalent slice that fits in the given extent,
            this is a diagnostic helper for drawing digits
            """
        ideal = self.decoder.to_ratio(digit)
        if ideal is None:
            return None
        inner_average, outer_average = self._measure(extent, log=False)
        start_y = int(round(inner_average + 1))
        end_y = int(round(outer_average))
        span = end_y - start_y
        lead_length = span * ideal.lead_ratio()
        head1_length = span * ideal.head1_ratio()
        gap_length = span * ideal.gap_ratio()
        head2_length = span * ideal.head2_ratio()
        tail_length = span * ideal.tail_ratio()
        # total = sum(ideal.parts)  # this covers span and is in range 0..?
        # scale = span / total  # so scale by this to get 0..span
        # lead_length *= scale
        # head1_length *= scale
        # gap_length *= scale
        # head2_length *= scale
        # tail_length *= scale
        lead_start = start_y
        head1_start = lead_start + lead_length
        gap_start = head1_start + head1_length
        head2_start = gap_start + gap_length
        tail_start = head2_start + head2_length
        tail_end = tail_start + tail_length
        return int(round(lead_start)), \
               int(round(head1_start)), \
               int(round(gap_start)), \
               int(round(head2_start)), \
               int(round(tail_start)), \
               int(round(tail_end))

    def _find_all_digits(self, extent: Extent) -> ([Digit], str):
        """ find all the digits from an analysis of the given extent,
            returns a digit list and None or a partial list and a fail reason,
            the extent is also updated with the slices involved
            """

        # ToDo: if flip to the 'box detection' method this can be simplified to just finding zeroes

        if self.logging:
            header = 'find_all_digits:'

        buckets = extent.buckets
        max_x, max_y = buckets.size()
        reason = None

        def show_options(options):
            """ produce a formatted string to describe the given digit options """
            if options is None:
                return 'None'
            msg = ''
            for option in options:
                msg = '{}, ({}, {})'.format(msg, option[0], option[1])
            return msg[2:]

        # region generate likely digit choices for each x...
        inner = extent.inner
        outer = extent.outer
        slices = [[None, None] for _ in range(max_x)]
        for x in range(max_x):
            start_y = inner[x] + 1  # get past the white bullseye, so start at first black
            end_y   = outer[x]      # this is the first white after the inner black
            # calculate the nominal ring height
            # the lead and tail are assumed to be at least this
            # we only scan the data ring area for the pulses
            ring_height = (end_y - start_y) / Scan.INNER_OUTER_SPAN_RINGS
            if ring_height < 1:
                # data area too small, so this is junk
                reason = 'data too narrow'
                break
            data_start = int(math.floor(start_y + (ring_height * Scan.INNER_BLACK_RINGS)))  # assumed earliest head start
            data_end = int(math.ceil(end_y - (ring_height * Scan.OUTER_BLACK_RINGS)))  # assumed first outer black
            # region scan down for lead/head/tail edges...
            down_lead_start_at = start_y
            down_head_start_at = None
            down_tail_start_at = None
            for y in range(data_start, data_end):
                pixel = buckets.getpixel(x, y)
                if pixel == MID_LUMINANCE:
                    # treat like white when in the data rings (which we know we are here)
                    pixel = MAX_LUMINANCE
                if pixel == MAX_LUMINANCE and down_head_start_at is None:
                    # start of head, end of lead
                    down_head_start_at = y  # first white after last black
                    continue
                if pixel == MIN_LUMINANCE and down_head_start_at is not None:
                    # end of head, begin of tail
                    down_tail_start_at = y  # first black after last white
                    break
            if down_head_start_at is None:
                # all black (this is a 0)
                down_head_start_at = end_y
                down_tail_start_at = end_y
            elif down_tail_start_at is None:
                # white to the end
                down_tail_start_at = data_end
            # endregion
            # region scan up for tail/head/lead edges...
            up_head_start_at = None
            up_tail_start_at = None
            up_tail_end_at = end_y - 1
            for y in range(data_end - 1, data_start - 1, -1):  # NB: scanning backwards, from bottom to top
                pixel = buckets.getpixel(x, y)
                if pixel == MID_LUMINANCE:
                    # treat like white when in the data rings (which we know we are here)
                    pixel = MAX_LUMINANCE
                if pixel == MAX_LUMINANCE and up_tail_start_at is None:
                    # start of tail, end of head
                    up_tail_start_at = y + 1  # get back to the black (first black after last head white)
                    continue
                if pixel == MIN_LUMINANCE and up_tail_start_at is not None:
                    # start of head
                    up_head_start_at = y + 1  # get back to the white (last head white)
                    break
            if up_tail_start_at is None:
                # all black (this is a 0)
                up_tail_start_at = end_y
                up_head_start_at = end_y
            elif up_head_start_at is None:
                # white to the end
                up_head_start_at = data_start
            # endregion
            # calculate ratios using the discovered lead/head and head/tail edges
            # we calculate the overall data width (outer - inner edges), and the
            # lengths of the components discovered, this is passed to our decode
            # to determine the ratios (this delegates knowledge of pulse shapes
            # to the decoded)
            down_lead_length = down_head_start_at - down_lead_start_at
            down_head_length = down_tail_start_at - down_head_start_at
            up_head_length = up_tail_start_at - up_head_start_at
            up_tail_length = up_tail_end_at - up_tail_start_at + 1
            data_length = end_y - start_y
            slices[x][0] = self.decoder.make_ratio(down_lead_length, up_tail_length, down_head_length,
                                                   up_head_length, data_length)
            slices[x][1] = self.decoder.classify(slices[x][0])
        if self.logging:
            if header is not None:
                self._log(header)
                header = None
            self._log('    initial slices (fail reason {}):'.format(reason))
            for x, (ratio, options) in enumerate(slices):
                self._log('        {}: ratio={}, options={}'.format(x, ratio, show_options(options)))
        # endregion

        if reason is None:
            # region repeat removing singles, doubles and triples until nothing changes...
            singles = 0
            doubles = 0
            triples = 0
            changes = True
            passes = 0
            if self.logging:
                sub_header = 'remove singles, doubles and triples'
            while changes:
                changes = False
                # adjust singles by removing its first choice
                for x, (ratio, this) in enumerate(slices):
                    if ratio is None:
                        continue
                    ratio, pred = slices[(x - 1) % max_x]
                    if ratio is None:
                        continue
                    ratio, succ = slices[(x + 1) % max_x]
                    if ratio is None:
                        continue
                    removed = []
                    while len(this) > 0:
                        if this[0][0] != pred[0][0] and this[0][0] != succ[0][0]:
                            # got a single, remove top choice
                            if len(this) > 1:
                                # got choices, so remove one
                                removed.append(this[0])
                                del this[0]
                            else:
                                # no choices left, morph to one of our neighbours, which one?
                                removed.append(this[0])
                                this[0] = pred[0]
                        else:
                            # not, or no longer, a single
                            break
                    if len(removed) > 0:
                        # we made a change
                        changes = True
                        singles += 1
                        if self.logging:
                            if header is not None:
                                self._log(header)
                                header = None
                            if sub_header is not None:
                                self._log('    {}:'.format(sub_header))
                                sub_header = None
                            self._log('        pass {} at {}: (single) removed {} leaving {}'.
                                      format(passes, x, show_options(removed), show_options(this)))
                if changes:
                    passes += 1
                    continue

                # no singles left, adjust doubles by making one of them into a single
                # for x, (ratio, this) in enumerate(slices):
                #     if ratio is None:
                #         continue
                #     ratio, pred = slices[(x - 1) % max_x]
                #     if ratio is None:
                #         continue
                #     ratio, that = slices[(x + 1) % max_x]
                #     if ratio is None:
                #         continue
                #     ratio, succ = slices[(x + 2) % max_x]
                #     if ratio is None:
                #         continue
                #     removed = []
                #     while len(this) > 0 or len(that) > 0:
                #         if this[0][0] == that[0][0] and this[0][0] != pred[0][0] and this[0][0] != succ[0][0]:
                #             # got a double, remove top choices
                #             if len(this) > 1:
                #                 # there is a choice
                #                 removed.append(this[0])
                #                 del this[0]
                #             elif len(that) > 1:
                #                 # there is a choice
                #                 removed.append(that[0])
                #                 del that[0]
                #             else:
                #                 # no choices left, morph to one of our neighbours, which one?
                #                 removed.append(this[0])
                #                 this[0] = pred[0]
                #         else:
                #             # not or no longer a double
                #             break
                #     if len(removed) > 0:
                #         # we made a change
                #         changes = True
                #         doubles += 1
                #         if self.logging:
                #             if header is not None:
                #                 self._log(header)
                #                 header = None
                #             if sub_header is not None:
                #                 self._log('    {}:'.format(sub_header))
                #                 sub_header = None
                #             self._log('        pass {} at {}: (double) removed {} leaving {}'.
                #                       format(passes, x, show_options(removed), show_options(this)))
                # if changes:
                #     passes += 1
                #     continue

                # no doubles left, adjust triples by making one of them into a single
                # for x, (ratio, this) in enumerate(slices):
                #     if ratio is None:
                #         continue
                #     ratio, pred = slices[(x - 2) % max_x]
                #     if ratio is None:
                #         continue
                #     ratio, left = slices[(x - 1) % max_x]
                #     if ratio is None:
                #         continue
                #     ratio, right = slices[(x + 1) % max_x]
                #     if ratio is None:
                #         continue
                #     ratio, succ = slices[(x + 2) % max_x]
                #     if ratio is None:
                #         continue
                #     removed = []
                #     while len(left) > 0 or len(this) > 0 or len(right) > 0:
                #         if this[0][0] == left[0][0] and this[0][0] == right[0][0]:
                #             # got a potential triple
                #             if this[0][0] != pred[0][0] and this[0][0] != succ[0][0]:
                #                 # got a triple, remove top choice
                #                 if len(this) > 1:
                #                     # there is a choice
                #                     removed.append(this[0])
                #                     del this[0]
                #                 elif len(left) > 1:
                #                     # there is a choice
                #                     removed.append(left[0])
                #                     del left[0]
                #                 elif len(right) > 1:
                #                     # there is a choice
                #                     removed.append(right[0])
                #                     del right[0]
                #                 else:
                #                     # no choices left, morph to one of our neighbours, which one?
                #                     removed.append(this[0])
                #                     this[0] = pred[0]
                #             else:
                #                 # not or no longer a triple
                #                 break
                #         else:
                #             # not or no longer a triple
                #             break
                #     if len(removed) > 0:
                #         # we made a change
                #         changes = True
                #         triples += 1
                #         if self.logging:
                #             if header is not None:
                #                 self._log(header)
                #                 header = None
                #             if sub_header is not None:
                #                 self._log('    {}:'.format(sub_header))
                #                 sub_header = None
                #             self._log('        pass {} at {}: (triple) removed {} leaving {}'.
                #                       format(passes, x, show_options(removed), show_options(this)))
                # if changes:
                #     passes += 1
                #     continue

            if self.logging:
                if header is not None:
                    self._log(header)
                    header = None
                self._log('    {} passes, {} singles adjusted, {} doubles adjusted, {} triples adjusted'.
                          format(passes, singles, doubles, triples))
                self._log('    final slices:')
                for x, (ratio, digit) in enumerate(slices):
                    self._log('        {}: ratio={}, options={}'.format(x, ratio, show_options(digit)))
            #endregion

        # region build digit list...
        digits = []
        last_digit = None
        for x, (ratio, this) in enumerate(slices):
            if ratio is None:
                # this is junk - ignore it
                continue
            if last_digit is None:
                # first digit
                last_digit = Scan.Digit(this[0][0], this[0][1].error, x, 1)
            elif this[0][0] == last_digit.digit:
                # continue with this digit
                last_digit.error += this[0][1].error  # accumulate error
                last_digit.samples += 1         # and sample count
            else:
                # save last digit
                last_digit.error /= last_digit.samples  # set average error
                digits.append(last_digit)
                # start a new digit
                last_digit = Scan.Digit(this[0][0], this[0][1].error, x, 1)
        # deal with last digit
        if last_digit is None:
            # nothing to see here...
            reason = 'no digits'
        elif len(digits) == 0:
            # its all the same digit - this must be junk
            last_digit.error /= last_digit.samples  # set average error
            digits = [last_digit]
            reason = 'single digit'
        elif last_digit.digit == digits[0].digit:
            # its part of the first digit
            last_digit.error /= last_digit.samples  # set average error
            digits[0].error = (digits[0].error + last_digit.error) / 2
            digits[0].start = last_digit.start
            digits[0].samples += last_digit.samples
        else:
            # its a separate digit
            last_digit.error /= last_digit.samples  # set average error
            digits.append(last_digit)
        # endregion

        if reason is None:
            # region check digits not too big...
            # check for reasonable digit widths, digits too big mean we are looking at junk
            max_digit_width = (max_x / Scan.NUM_SEGMENTS) * Scan.MAX_DIGIT_WIDTH
            for x, digit in enumerate(digits):
                if digit.samples > max_digit_width:
                    # too big
                    reason = 'digit too big'
                    break
            # end region

        # save slices in the extent for others to use
        extent.slices = slices

        if self.logging:
            if header is not None:
                self._log(header)
                header = None
            self._log('    {} digits (fail reason {}):'.format(len(digits), reason))
            for x, digit in enumerate(digits):
                self._log('        {}: {}'.format(x, digit))

        if self.save_images:
            plot = self.transform.copy(buckets)
            lead_lines = []
            head1_lines = []
            gap_lines = []
            head2_lines = []
            tail_lines = []
            for x, (ratio, slice) in enumerate(slices):
                if ratio is None:
                    # this is junk
                    continue
                x_slice = self._make_slice(slice[0][0], extent)
                if x_slice is None:
                    # not a valid digit
                    continue
                lead_start, head1_start, gap_start, head2_start, tail_start, tail_end = x_slice
                lead_lines.append((x, lead_start, x, head1_start - 1))
                head1_lines.append((x, head1_start, x, gap_start - 1))
                gap_lines.append((x, gap_start, x, head2_start - 1))
                head2_lines.append((x, head2_start, x, tail_start - 1))
                tail_lines.append((x, tail_start, x, tail_end - 1))
            plot = self._draw_lines(plot, lead_lines, Scan.GREEN)
            plot = self._draw_lines(plot, head1_lines, Scan.RED)
            plot = self._draw_lines(plot, gap_lines, Scan.GREEN)
            plot = self._draw_lines(plot, head2_lines, Scan.RED)
            plot = self._draw_lines(plot, tail_lines, Scan.GREEN)
            self._unload(plot, '05-slices')

        return digits, reason

    def _find_best_digits(self, digits: [Digit], extent: Extent = None) -> ([Digit], str):
        """ analyse the given digits to isolate the 'best' ones,
            extent provides the slices found by _find_all_digits,
            the 'best' digits are those that conform to the known code structure,
            specifically one zero per copy and correct number of digits per copy,
            returns the revised digit list and None if succeeded
            or a partial list and a reason if failed
            """

        if self.logging:
            header = 'find_best_digits:'

        def find_best_2nd_choice(slices, slice_start, slice_end):
            """ find the best 2nd choice in the given slices,
                return the digit, how many there are and the average error,
                the return info is sufficient to create a Scan.Digit
                """
            second_choice = [[0, 0] for _ in range(self.decoder.base)]
            for x in range(slice_start, slice_end):
                _, options = slices[x % len(slices)]
                if len(options) > 1:
                    second_choice[options[1][0]][0] += 1
                    second_choice[options[1][0]][1] += options[1][1].error
            best_digit = None
            best_count = 0
            best_error = 0
            for digit, (count, error) in enumerate(second_choice):
                if count > best_count:
                    best_digit = digit
                    best_count = count
                    best_error = error
            return best_digit, best_count, best_error / best_count

        def shrink_digit(slices, digit, start, samples) -> Scan.Digit:
            """ shrink the indicated digit to the start and size given,
                this involves moving the start, updating the samples and adjusting the error,
                the revised digit is returned
                """
            # calculate the revised error
            error = 0
            for x in range(start, start + samples):
                _, options = slices[x % len(slices)]
                error += options[0][1].error
            error /= samples
            # create a new digit
            new_digit = Scan.Digit(digit.digit, error, start % len(slices), samples)
            return new_digit

        # translate digits from [Digit] to [[Digit]] so we can mess with it and not change indices
        # indices into digits_list must remain constant even while we are adding/removing digits
        # we achieve this by having a list of digits for each 'digit', that list is emptied when
        # a digit is removed or extended when a digit is added, for any index 'x' digits[x] is
        # the original digit and digits_list[x] is a list of 1 or 2 digits or None, None=removed,
        # 2=split, and 1=no change
        digits_list = [[options] for options in digits]

        # we expect to find Scan.COPIES of the sync digit
        copies = []
        for x, digit in enumerate(digits):
            if self.decoder.is_sync_digit(digit.digit):
                copies.append([x])
        if len(copies) < Scan.COPIES:
            # not enough zeroes - that's a show-stopper
            reason = 'too few syncs'
        else:
            # if too many sync digits - dump the smallest with the most error
            while len(copies) > Scan.COPIES:
                smallest_x = 0  # index into copy
                for x in range(1, len(copies)):
                    xx = copies[x][0]
                    smallest_xx = copies[smallest_x][0]
                    if digits[xx].samples < digits[smallest_xx].samples:
                        # less samples
                        smallest_x = x
                    elif digits[xx].samples == digits[smallest_xx].samples:
                        if digits[xx].error > digits[smallest_xx].error:
                            # same samples but bigger error
                            smallest_x = x
                smallest_xx = copies[smallest_x][0]
                if self.logging:
                    if header is not None:
                        self._log(header)
                        header = None
                    self._log('    {}: dropping excess sync {}'.
                              format(smallest_xx, digits[smallest_xx]))
                digits_list[smallest_xx] = None
                del copies[smallest_x]
            reason = None
        if reason is None:
            boxes, reason = self._find_data_boxes(extent, digits, copies)  # ToDo: HACK
            if reason is None:
                bits = self._decode_data_boxes(boxes, extent)  # ToDo: HACK
            # find the actual digits between the syncs
            reason = None
            for copy in copies:
                xx = copy[0]
                while True:
                    xx = (xx + 1) % len(digits)
                    if digits_list[xx] is None:
                        # this one has been dumped
                        continue
                    if self.decoder.is_sync_digit(digits[xx].digit):
                        # ran into next copy
                        break
                    copy.append(xx)
                digits_required = Scan.DIGITS_PER_NUM
                while len(copy) < digits_required:
                    # not enough digits - split the biggest with the biggest error
                    biggest_x = None
                    for x in range(1, len(copy)):  # never consider the initial 0
                        xx = copy[x]
                        digits_xx = digits_list[xx]
                        if digits_xx is None:
                            # this one has been dumped
                            continue
                        if len(digits_xx) > 1:
                            # this one has already been split
                            continue
                        if biggest_x is None:
                            # found first split candidate
                            biggest_x = x
                            continue
                        biggest_digit = digits[copy[biggest_x]]
                        if digits[xx].samples > biggest_digit.samples:
                            biggest_x = x
                        elif digits[xx].samples == biggest_digit.samples:
                            if digits[xx].error > biggest_digit.error:
                                biggest_x = x
                    if biggest_x is None:
                        # everything has been split and still not enough - this is a show stopper
                        reason = 'too few digits'
                        break
                    biggest_xx = copy[biggest_x]
                    biggest_digit = digits[biggest_xx]
                    # we want to split the biggest using the second choice in the spanned slices
                    # count 2nd choices in the first half and second half of the biggest span
                    # use the option with the biggest count, this represents the least error 2nd choice
                    # this algorithm is very crude and is only reliable when we are one digit short
                    # this is the most common case, eg. when a 100 smudges into a 010
                    # we only split a sequence once so digits[x] and digits_list[x][0] are the same here
                    slices = extent.slices
                    slice_start = digits[biggest_xx].start
                    slice_full_span = digits[biggest_xx].samples
                    slice_first_span = int(round(slice_full_span / 2))
                    slice_second_span = slice_full_span - slice_first_span
                    best_1 = find_best_2nd_choice(slices, slice_start, slice_start + slice_first_span)
                    best_2 = find_best_2nd_choice(slices, slice_start + slice_first_span, slice_start + slice_full_span)
                    if best_1[1] > best_2[1]:
                        # first half is better, create a digit for that, insert before the other and shrink the other
                        digit_1 = Scan.Digit(best_1[0], best_1[2], slice_start, slice_first_span)
                        digit_2 = shrink_digit(slices, biggest_digit, slice_start + slice_first_span, slice_second_span)
                    else:
                        # second half is better, create a digit for that
                        digit_1 = shrink_digit(slices, biggest_digit, slice_start, slice_first_span)
                        digit_2 = Scan.Digit(best_2[0], best_2[2],
                                             (slice_start + slice_first_span) % len(slices), slice_second_span)
                    if self.logging:
                        if header is not None:
                            self._log(header)
                            header = None
                        self._log('    {}: splitting {} into {} and {}'.
                                  format(biggest_xx, biggest_digit, digit_1, digit_2))
                    digits_list[biggest_xx] = [digit_1, digit_2]
                    digits_required -= 1
                while len(copy) > digits_required:
                    # too many digits - drop the smallest with the biggest error that is not a zero
                    smallest_x = None
                    for x in range(1, len(copy)):  # never consider the initial 0
                        xx = copy[x]
                        digits_xx = digits_list[xx]
                        if digits_xx is None:
                            # this one has been dumped
                            continue
                        if len(digits_xx) > 1:
                            # this one has been split - not possible to see that here!
                            continue
                        if smallest_x is None:
                            # found first dump candidate
                            smallest_x = x
                            continue
                        smallest_digit = digits[copy[smallest_x]]
                        if digits[xx].samples < smallest_digit.samples:
                            smallest_x = x
                        elif digits[xx].samples == smallest_digit.samples:
                            if digits[xx].error > smallest_digit.error:
                                smallest_x = x
                    smallest_xx = copy[smallest_x]
                    if self.logging:
                        if header is not None:
                            self._log(header)
                            header = None
                        self._log('    {}: dropping excess digit {}'.
                                  format(smallest_xx, digits[smallest_xx]))
                    digits_list[smallest_xx] = None
                    digits_required += 1
                if reason is not None:
                    # we've given up someplace
                    break

        # build the final digit list
        best_digits = []
        for digits in digits_list:
            if digits is None:
                # this has been deleted
                continue
            for digit in digits:
                best_digits.append(digit)

        if self.logging:
            if header is not None:
                self._log(header)
                header = None
            self._log('    {} digits (fail reason: {}):'.format(len(best_digits), reason))
            for x, digit in enumerate(best_digits):
                self._log('        {}: {}'.format(x, digit))

        if self.save_images:
            buckets = extent.buckets
            max_x, max_y = buckets.size()
            plot = self.transform.copy(buckets)
            lead_lines = []
            head1_lines = []
            gap_lines = []
            head2_lines = []
            tail_lines = []
            for digit in best_digits:
                digit_slice = self._make_slice(digit.digit, extent)
                if digit_slice is None:
                    # not a valid digit
                    continue
                lead_start, head1_start, gap_start, head2_start, tail_start, tail_end = digit_slice
                for sample in range(digit.samples):
                    x = (digit.start + sample) % max_x
                    lead_lines.append((x, lead_start, x, head1_start - 1))
                    head1_lines.append((x, head1_start, x, gap_start - 1))
                    gap_lines.append((x, gap_start, x, head2_start - 1))
                    head2_lines.append((x, head2_start, x, tail_start - 1))
                    tail_lines.append((x, tail_start, x, tail_end - 1))
            plot = self._draw_lines(plot, lead_lines, Scan.GREEN)
            plot = self._draw_lines(plot, head1_lines, Scan.RED)
            plot = self._draw_lines(plot, gap_lines, Scan.GREEN)
            plot = self._draw_lines(plot, head2_lines, Scan.RED)
            plot = self._draw_lines(plot, tail_lines, Scan.GREEN)
            self._unload(plot, '06-digits')

        return best_digits, reason

    def _find_boundaries(self, extent):
        """ given an extent, return a list of x co-ords representing the maxima/minima lead/tail lengths,
            these represent the centre of (some of) the data cells, this information is used to guide
            subsequent logic in determining the cell boundaries
            """

        def de_nipple(sequence):
            """ change sequences that jiggle up/down to flat,
                if sequence N-1 == N+1 then set N to N-1,
                returns the de-nippled sequence
                """

            return sequence

            max_x = len(sequence)
            for x in range(max_x):
                prev = sequence[(x - 1) % max_x]
                next = sequence[(x + 1) % max_x]
                if prev == next:
                    sequence[x] = prev
            return sequence

        def get_slope(sequence, threshold=0):
            """ get_slope the given sequence, threshold determines the threshold to trigger a 'change',
                None values are ignored
                """

            max_x = len(sequence)
            delta_threshold = threshold * threshold  # square it to remove sign consideration

            slope = [0 for _ in range(max_x)]
            for x in range(max_x):
                prev_x = (x - 2) % max_x
                left_x = (x - 1) % max_x
                this_x = x
                right_x = (x + 1) % max_x
                next_x = (x + 2) % max_x
                behind = (sequence[prev_x] + sequence[left_x]) / 2
                ahead = (sequence[right_x] + sequence[next_x]) / 2
                delta = behind - ahead
                if (delta * delta) > delta_threshold:
                    slope[this_x] = delta

            return slope

        def find_knees(sequence, threshold=0):
            """ given a list of values that represent some sort of sequence find the changes in its slope,
                threshold is the value change required to qualify as a significant slope change,
                returns a list of co-ordinates into the sequence that represent the knees
                """

            slope = get_slope(sequence, threshold)
            slope_slope = get_slope(slope, threshold*2)

            max_x = len(sequence)

            # the knees are +ve to -ve or -ve to +ve or 0 to not-0 or not-0 to 0 slope changes
            # however the slope may be 'noisy' so we ignore single 0's
            knees = []
            for x in range(max_x):
                prev_x = (x - 1) % max_x
                this_x = x
                next_x = (x + 1) % max_x
                prev_sample = sequence[prev_x]
                this_sample = sequence[this_x]
                next_sample = sequence[next_x]
                if this_sample < 100 and prev_sample == 100:
                    # this is a zero edge for a lead or a tail
                    knees.append(this_x)
                    continue
                if this_sample < 100 and next_sample == 100:
                    # this is a zero edge for a lead or a tail
                    knees.append(this_x)
                    continue
                if this_sample > 0 and prev_sample == 0:
                    # this is a zero edge for a head
                    knees.append(this_x)
                    continue
                if this_sample > 0 and next_sample == 0:
                    # this is a zero edge for a head
                    knees.append(this_x)
                    continue
                if this_sample == 100 or this_sample == 0:
                    # ignore zeroes
                    continue
                # prev_slope_slope = slope_slope[prev_x]
                # this_slope_slope = slope_slope[this_x]
                # next_slope_slope = slope_slope[next_x]
                # if this_slope_slope == 0 and next_slope_slope != 0:
                #     knees.append(x)
                #     continue
                # if this_slope_slope == 0 and prev_slope_slope != 0:
                #     knees.append(x)
                #     continue
                # if this_slope_slope != 0 and next_slope_slope != 0:
                #     if this_slope_slope * next_slope_slope < 0:
                #         knees.append(x)
                #         continue
                # if this_slope_slope != 0 and prev_slope_slope != 0:
                #     if this_slope_slope * prev_slope_slope < 0:
                #         knees.append(x)
                #         continue
                prev_slope = slope[prev_x]
                this_slope = slope[this_x]
                next_slope = slope[next_x]
                if this_slope == 0 and next_slope != 0:
                    # becoming flat
                    knees.append(next_x)
                    continue
                if this_slope == 0 and prev_slope != 0:
                    # leaving flat
                    knees.append(prev_x)
                    continue
                if this_slope != 0 and next_slope != 0:
                    if this_slope * next_slope < 0:
                        # at a peak
                        knees.append(this_x)
                        continue
            return knees, slope, slope_slope

        inner = extent.inner
        outer = extent.outer
        slices = extent.slices  # the ratio elements in each slice contains the lead/head/tail lengths

        leads = [None for _ in range(len(slices))]
        heads = [None for _ in range(len(slices))]
        tails = [None for _ in range(len(slices))]
        for x, (ratio, _) in enumerate(slices):
            leads[x] = int(round(ratio.lead_ratio() * 100))
            heads[x] = int(round(ratio.head_ratio() * 100))
            tails[x] = int(round(ratio.tail_ratio() * 100))
        lead_knees, lead_knees_slope, lead_knees_slope_slope = find_knees(de_nipple(leads), 10)
        head_knees, head_knees_slope, head_knees_slope_slope = find_knees(de_nipple(heads), 10)
        tail_knees, tail_knees_slope, tail_knees_slope_slope = find_knees(de_nipple(tails), 10)

        if self.save_images:
            buckets = extent.buckets
            lead_lines = []
            head_lines = []
            tail_lines = []
            for x in lead_knees:
                lead_lines.append([x, inner[x], x, outer[x]])
            for x in head_knees:
                head_lines.append([x, inner[x], x, outer[x]])
            for x in tail_knees:
                tail_lines.append([x, inner[x], x, outer[x]])
            plot = self._draw_lines(buckets, lead_lines, colour=Scan.GREEN)
            plot = self._draw_lines(buckets, head_lines, colour=Scan.RED)
            plot = self._draw_lines(plot, tail_lines, colour=Scan.BLUE)
            self._unload(plot, '06-knees')

        if self.logging:
            self._log('find_boundaries: leads:{}'.format(leads))
            self._log('find_boundaries:     slope:{}'.format(lead_knees_slope))
            self._log('find_boundaries:     slope slope:{}'.format(lead_knees_slope_slope))
            self._log('find_boundaries:     knees:{}'.format(lead_knees))
            self._log('find_boundaries: heads:{}'.format(heads))
            self._log('find_boundaries:     slope:{}'.format(head_knees_slope))
            self._log('find_boundaries:     slope slope:{}'.format(head_knees_slope_slope))
            self._log('find_boundaries:     knees:{}'.format(head_knees))
            self._log('find_boundaries: tails:{}'.format(tails))
            self._log('find_boundaries:     slope:{}'.format(tail_knees_slope))
            self._log('find_boundaries:     slope slope:{}'.format(tail_knees_slope_slope))
            self._log('find_boundaries:     knees:{}'.format(tail_knees))

    def _find_data_boxes(self, extent: Extent, digits: [Digit], zeroes: [[int]]) -> ([Box], str):
        """ find the boundaries for all the data boxes given an extent a digit list and zero positions,
            returns a box list and None if OK, or a partial list and a fail reason if not,
            the y co-ords of the edges are exclusive
            """

        # ToDo: move this to a Scan constant
        MIN_VERTICAL_EDGE_LENGTH = 2  # vertical edges of less than this many pixels are ignored
        MAX_VERT_X_GAP = 2.5  # max x pixel gap between neighbour vertical edges where merging is allowed
        MAX_VERT_Y_GAP = 1  # max y pixel gap between neighbour vertical edges where merging is allowed
        MIN_HORIZONTAL_EDGE_LENGTH = 2  # horizontal edges of less than this many pixels are ignored
        HORIZONTAL_Y_OFFSETS = [0]  # ToDo: HACK-->[0, -1, +1]  # y offsets used for following horizontal edges
        MAX_HORIZ_Y_GAP = 2.5  # max y pixel gap between neighbour horizontal edge points where merging is allowed
        EXTENT_BORDER = 2  # pixels at inner/outer extent that are considered to be black

        inner = extent.inner
        outer = extent.outer
        buckets = extent.buckets
        max_x, max_y = buckets.size()

        def merge_y_edges(digit_edges, this_idx, next_idx):
            """ given two sets of vertical edges, merge those of the same type,
                digit_edges is the complete digit edge list, this_idx and next_idx are an index pair of edge sets in x,
                returns True iff at least one merge took place, result is reflected in digit_edges
                """

            def merge_y_spans(this_start_y, this_end_y, that_start_y, that_end_y, this_x, that_x):
                """ given two y spans, merge them if they (nearly) abutt or overlap,
                    this_x and that_x must also be sufficiently close to each,
                    returns None iff they do not merge or a (merged_x, merged_start_y, merged_end_y) tuple iff they do
                    """
                x_gap = wrapped_gap(that_x, this_x, max_x)
                if x_gap < 0:
                    x_gap = 0 - x_gap  # we've wrapped
                if x_gap > MAX_VERT_X_GAP:
                    # too far way to merge
                    return None
                if (this_end_y + MAX_VERT_Y_GAP) < that_start_y or (that_end_y + MAX_VERT_Y_GAP) < this_start_y:
                    # they do not overlap or (nearly) abutt
                    return None
                # merge these
                # the merged x is this_x plus the weighted difference between this_x and that_x
                # the weighted difference is based on the edge lengths, longer edges have more weight
                this_size = this_end_y - this_start_y
                that_size = that_end_y - that_start_y
                this_weight = 1 - (this_size / (this_size + that_size))  # 0..1, 0==this_x, 1==that_x
                merged_x = int(round(this_x + (this_weight * x_gap))) % max_x
                merged_start = min(this_start_y, that_start_y)
                merged_end = max(this_end_y, that_end_y)
                return merged_x, merged_start, merged_end

            # allow for wrapping
            this_idx = this_idx % len(digit_edges)
            next_idx = next_idx % len(digit_edges)

            this_x, this_edges = digit_edges[this_idx]
            next_x, next_edges = digit_edges[next_idx]

            for this_edge, (this_type, this_start_y, this_end_y) in enumerate(this_edges):
                for next_edge, (next_type, next_start_y, next_end_y) in enumerate(next_edges):
                    if this_idx == next_idx and this_edge == next_edge:
                        # ignore self
                        continue
                    if this_type != next_type:
                        # not same type
                        continue
                    merged = merge_y_spans(this_start_y, this_end_y, next_start_y, next_end_y, this_x, next_x)
                    if merged is None:
                        # they do not overlap or abutt
                        continue
                    merged_x, merged_start_y, merged_end_y = merged
                    if merged_x == this_x:
                        # add merged to this_edges, remove next
                        this_edges[this_edge] = (this_type, merged_start_y, merged_end_y)
                        del next_edges[next_edge]
                    elif merged_x == next_x:
                        # add merged to next_edges, remove this
                        next_edges[next_edge] = (this_type, merged_start_y, merged_end_y)
                        del this_edges[this_edge]
                    else:
                        # not the simple cases, so...
                        # remove both originals
                        del this_edges[this_edge]
                        del next_edges[next_edge]
                        # merged_x may exist in digit_edges between this_idx and next_idx,
                        # in which case we update it, otherwise we create a new x in digit_edges
                        # find index in digit_edges to put this merged x
                        target_idx = None
                        for idx in range(this_idx, next_idx + 1):
                            idx = idx % len(digit_edges)
                            target_x, target_edges = digit_edges[idx]
                            if target_x == merged_x:
                                # found an existing list, extend it
                                target_edges.append((this_type, merged_start_y, merged_end_y))
                                target_idx = idx
                                break
                            elif target_x > merged_x:
                                # we're guaranteed to get here if there is no exact match,
                                # 'cos we know this_x <= merged_x <= next_x and digit_edges is in x order,
                                # so insert a new entry here
                                digit_edges.insert(idx, [merged_x, [(this_type, merged_start_y, merged_end_y)]])
                                target_idx = idx
                                break
                            else:
                                # carry on looking
                                pass
                        if target_idx is None:
                            # eh?
                            raise Exception('digit_edges between {} and {} does not cover {}'.
                                            format(this_x, next_x, merged_x))
                    # note change for caller
                    return True

            # nothing merged
            return False

        # region force black pixels at the inner and outer edges (provides a boundary extreme for edge detection)...
        if EXTENT_BORDER > 0:
            for x in range(max_x):
                for dx in range(1, EXTENT_BORDER + 1):  # +1 'cos extents mark the white pixels, need to get past those
                    buckets.putpixel(x, inner[x] + dx, MIN_LUMINANCE)
                    buckets.putpixel(x, outer[x] - dx, MIN_LUMINANCE)
            if self.save_images:
                self._unload(buckets, '06-buckets-isolated')
        # endregion

        reason = None
        if reason is None:
            # region Phase 1 - find vertical edges...
            # vertical edges represent potential digit boundaries
            # we do two passes, treating gray as black in one pass and as white in the other
            digit_edges = []  # vertical edges as [x, [type, start y, end y]]
            for grey_as in [MIN_LUMINANCE, MAX_LUMINANCE]:
                for x in range(max_x):
                    good_edges = []  # edge at this x as [type, start y, end y]
                    start_y = inner[x] + 1  # get into the first black pixel (inner edge is the last white pixel)
                    end_y = outer[x]  # end_y is exclusive (outer edge is the first white pixel)
                    # Don't do the below 2 lines, it obscures legitimate edges close to the extent edges
                    # start_y += EXTENT_BORDER  # get over the no-go zone
                    # end_y -= EXTENT_BORDER  # get over the no-go zone
                    span_type = None
                    span_start = None
                    span_size = None
                    for y in range(start_y, end_y):
                        here_pixel = buckets.getpixel(x, y)
                        if here_pixel == MID_LUMINANCE:
                            # consider grey as directed
                            here_pixel = grey_as
                        left_pixel = buckets.getpixel((x - 1) % max_x, y)
                        if left_pixel == MID_LUMINANCE:
                            # consider grey as directed
                            left_pixel = grey_as
                        if left_pixel != here_pixel:
                            # got the beginnings or continuation of a vertical edge
                            if span_start is None:
                                # its a new edge
                                span_start = y
                                span_size = 1
                                if left_pixel > here_pixel:
                                    span_type = Scan.FALLING
                                else:
                                    span_type = Scan.RISING
                                continue
                            if left_pixel > here_pixel and span_type == Scan.FALLING:
                                # still in a falling edge
                                span_size += 1
                                continue
                            if here_pixel > left_pixel and span_type == Scan.RISING:
                                # still in a rising edge
                                span_size += 1
                                continue
                        if span_start is None:  # and left_pixel == here_pixel:
                            # we're nowhere (yet)
                            continue
                        # got end of a vertical edge, keep it if its long enough
                        if span_size >= MIN_VERTICAL_EDGE_LENGTH:
                            # qualifies
                            good_edges.append((span_type, span_start, span_start + span_size))
                        # look for next
                        span_type = None
                        span_start = None
                        span_size = None
                    if span_start is not None:
                        # got end of a vertical edge at the end
                        # does this qualify?
                        if span_size >= MIN_VERTICAL_EDGE_LENGTH:
                            # qualifies
                            good_edges.append((span_type, span_start, span_start + span_size))
                    if len(good_edges) > 0:
                        digit_edges.append([x, good_edges])
            # digit edges is now a list of all vertical edges
            # put into x,y for merging purposes
            digit_edges.sort(key=lambda e: (e[0], e[1][0][1]))
            # region remove duplicates...
            # if there are no greys we'll end of up with every edge duplicated, find and drop those
            for edge in range(len(digit_edges) - 1, 0, -1):  # NB: stop 1 short 'cos we address -1
                if digit_edges[edge] == digit_edges[edge - 1]:
                    # got a dup
                    del digit_edges[edge]
            # endregion
            if self.logging:
                self._log('find_data_boxes: discovered digit_edges (before merging):')
                for x, edges in digit_edges:
                    msg = ''
                    for edge_type, start_y, end_y in edges:
                        msg = '{}, [{} {}..{} ({})]'.format(msg, edge_type, start_y, end_y - 1, end_y - start_y)
                    self._log('    x={}: {}'.format(x, msg[2:]))
            # endregion
        if reason is None:
            # region Phase 2 - merge neighbour vertical edges...
            # any edges within some limit of each other are merged wrt their start and length
            # digit_edges is a list of vertical edges in x order (each is a list of [type, start_y,end_y])
            # this is a crude N*N algorithm, but N is small, so OK
            changed = True
            while changed:
                changed = False
                for this_idx in range(len(digit_edges)):
                    if len(digit_edges[this_idx]) == 0:
                        # ignore empty list
                        continue
                    for next_idx in range(len(digit_edges)):
                        if len(digit_edges[next_idx]) == 0:
                            # ignore empty list
                            continue
                        changed = merge_y_edges(digit_edges, this_idx, next_idx)
                        if changed:
                            break
                    if changed:
                        break
            if self.logging:
                self._log('find_data_boxes: discovered digit edges (after merging):')
                for x, edges in digit_edges:
                    msg = ''
                    for edge_type, start_y, end_y in edges:
                        msg = '{}, [{} {}..{} ({})]'.format(msg, edge_type, start_y, end_y - 1, end_y - start_y)
                    self._log('    x={}: {}'.format(x, msg[2:]))
            # endregion
        if reason is None:
            # region Phase 3 - find horizontal edges...
            # horizontal edges represent potential ring boundaries
            # we do two passes, treating gray as black in one pass and as white in the other
            ring_edges = []  # discovered edges
            for grey_as in [MIN_LUMINANCE, MAX_LUMINANCE]:
                used = [[False for _ in range(max_y)] for _ in range(max_x)]  # been here before detector
                if EXTENT_BORDER > 0:
                    # region consider all extent border pixels to be 'used' to stop them being considered...
                    for x in range(max_x):
                        for y in range(max_y):
                            if y <= inner[x] + (EXTENT_BORDER - 1):  # -1 so that one forced black is visible
                                used[x][y] = True
                            if y >= outer[x] - (EXTENT_BORDER - 1):  # -1 so that one forced black is visible
                                used[x][y] = True
                    # endregion
                # region find an initial x to scan from...
                # this must be a place where there is no edge, i.e. within a zero,
                # we look for 2 consecutive x's where there is nothing between the inner and outer edge,
                # this is required to allow for edges that wrap, we must not start in the middle of a wrapped edge
                start_x = None
                for x in range(max_x):
                    empty = True
                    for dx in [0, 1]:
                        try_x = (x + dx) % max_x
                        for y in range(inner[try_x] + EXTENT_BORDER + 1, outer[try_x] - EXTENT_BORDER - 1):
                            pixel = buckets.getpixel(x, y)
                            if pixel == MID_LUMINANCE:
                                # treat grey as directed
                                pixel = grey_as
                            if pixel != MIN_LUMINANCE:
                                # something here, so look at next x
                                empty = False
                                break
                        if not empty:
                            break
                    if empty:
                        start_x = x
                        break
                if start_x is None:
                    # this means there are no zeroes, so we're looking at junk
                    reason = 'no zeroes'
                # endregion
                if reason is None:
                    for dx in range(start_x, start_x + max_x):
                        dx = dx % max_x
                        start_y = inner[dx] + 1  # +1 to get past the inner edge white
                        end_y = outer[dx]  # not +1 'cos want exclusive y range
                        # look for a black to white or white to black transition in y at this x
                        for y in range(start_y + 1, end_y):  # +1 'cos we look above
                            edge_begin = x
                            edge_points = []
                            edge_type = None
                            last_y = y  # note our starting point
                            for x in range(dx, dx + max_x):
                                x %= max_x
                                use_y = None
                                for dy in HORIZONTAL_Y_OFFSETS:
                                    y = last_y + dy
                                    above_pixel = buckets.getpixel(x, y - 1)
                                    if above_pixel == MID_LUMINANCE:
                                        # treat grey as directed
                                        above_pixel = grey_as
                                    here_pixel = buckets.getpixel(x, y)
                                    if here_pixel == MID_LUMINANCE:
                                        # treat grey as directed
                                        here_pixel = grey_as
                                    if here_pixel != above_pixel:
                                        use_y = y  # note this candidate
                                        break
                                if use_y is None:
                                    # no edge candidate here
                                    break
                                # found the next y for this edge
                                if used[x][use_y]:
                                    # been here before
                                    break
                                # region check/set edge type
                                if above_pixel > here_pixel:
                                    # falling edge
                                    if edge_type is None:
                                        # lock onto a falling edge
                                        edge_type = Scan.FALLING
                                    elif edge_type != Scan.FALLING:
                                        # run into a different edge type
                                        use_y = None
                                        break
                                else:
                                    # rising edge
                                    if edge_type is None:
                                        # lock onto a rising edge
                                        edge_type = Scan.RISING
                                    elif edge_type != Scan.RISING:
                                        # run into a different edge type
                                        use_y = None
                                        break
                                # endregion
                                edge_points.append(use_y)
                                used[x][use_y] = True  # note been here, so we don't start a new edge from here again
                                last_y = use_y  # note where we are for looking at the next y
                            if len(edge_points) < MIN_HORIZONTAL_EDGE_LENGTH:
                                # too small to consider
                                pass
                            else:
                                # note this edge
                                ring_edges.append([edge_type, edge_begin, edge_points])
                            # carry on looking for more edges at this x
                            continue
                        # continue looking for edges at the next x
                        continue
            # region remove duplicates...
            # if there are no greys we'll end of up with every edge duplicated, find and drop those
            # sort into x order, so we can easily find dups
            ring_edges.sort(key=lambda e: (e[1], e[2][0]))
            for edge in range(len(ring_edges)-1, 0, -1):  # NB: stop 1 short 'cos we address -1
                if ring_edges[edge] == ring_edges[edge-1]:
                    # got a dup
                    del ring_edges[edge]
            # endregion
            if self.logging:
                self._log('find_data_boxes: discovered ring edges (before merging):')
                for edge in ring_edges:
                    self._log('    {}'.format(edge))
            # endregion
        if reason is None:
            # region Phase 4 - merge neighbour horizontal edges...
            # any edges within some limit of each other are merged wrt their start and points
            # for each edge we search for an overlapping edge and try to merge it
            # this is a crude N*N loop, but N is small so OK
            passes = 0
            changed = True
            while changed:
                passes += 1
                changed = False
                for edge_1 in range(len(ring_edges)):
                    # get this edge stats
                    edge_1_edge = ring_edges[edge_1]
                    edge_1_type, edge_1_start, edge_1_points = edge_1_edge
                    edge_1_size = len(edge_1_points)
                    edge_1_end = edge_1_start + edge_1_size  # do not wrap this
                    # look for another edge that overlaps this one in x
                    for edge_2 in range((edge_1 + 1) % len(ring_edges), len(ring_edges) - 1):  # scan all but edge_1
                        edge_2_edge = ring_edges[edge_2 % len(ring_edges)]
                        edge_2_type, edge_2_start, edge_2_points = edge_2_edge
                        edge_2_size = len(edge_2_points)
                        edge_2_end = edge_2_start + edge_2_size  # do not wrap this
                        # pre-filter for type
                        if edge_1_type != edge_2_type:
                            # different type
                            continue
                        # pre-filter for no overlap
                        # if edge_1_end < edge_2_start:  # NB: edge_1_end == edge_2_start means they abutt, we want those
                        #     # does not overlap
                        #     continue
                        # if edge_2_end < edge_1_start:  # NB: edge_2_end == edge_1_start means they abutt, we want those
                        #     # does not overlap
                        #     continue
                        # got a potential overlap in x, try to merge
                        # build combined points list
                        line = [[None, None] for _ in range(max_x)]
                        for dx in range(len(edge_1_points)):
                            line[(edge_1_start + dx) % max_x][0] = edge_1_points[dx]
                        for dx in range(len(edge_2_points)):
                            line[(edge_2_start + dx) % max_x][1] = edge_2_points[dx]
                        # merge overlapping 'close' points
                        # find the start scan point (first None, None position)
                        start_x = None
                        for x, (y1, y2) in enumerate(line):
                            if y1 is None and y2 is None:
                                # found it
                                start_x = x
                                break
                        if start_x is None:
                            # got a full line!!
                            start_x = 0
                        # now scan for our points
                        merged = 0
                        for x in range(start_x, start_x + max_x):
                            y1, y2 = line[x % max_x]
                            # merge if y1 'close' to y2
                            if y1 is None and y2 is None:
                                # not in either edge
                                continue
                            prev_y1, prev_y2 = line[(x - 1) % max_x]
                            if y1 is None:
                                # propagate previous when run off the end
                                y1 = prev_y1
                            if y2 is None:
                                # propagate previous when run off the end
                                y2 = prev_y2
                            if y1 is not None and y2 is not None:
                                gap = max(y1, y2) - min(y1, y2)
                                if gap > MAX_HORIZ_Y_GAP:
                                    # points do not merge
                                    continue
                                # merge as the average
                                merged_y = (y1 + y2) / 2
                                # only merge if the merge point is not too far from its neighbours
                                if prev_y1 is not None:
                                    gap = max(merged_y, prev_y1) - min(merged_y, prev_y1)
                                    if gap > MAX_HORIZ_Y_GAP:
                                        # neighbour gap too big
                                        continue
                                if prev_y2 is not None:
                                    gap = max(merged_y, prev_y2) - min(merged_y, prev_y2)
                                    if gap > MAX_HORIZ_Y_GAP:
                                        # neighbour gap too big
                                        continue
                                # merged point OK
                                line[x % max_x] = [merged_y, merged_y]
                                merged += 1
                        if merged == 0:
                            # edge does not merge
                            continue
                        # got a merged pair, keep longest, break up the other
                        # we may have stretched one of the edges, so go count them again
                        counts = [0, 0]
                        for x, ys in enumerate(line):
                            for y in [0, 1]:
                                if ys[y] is not None:
                                    counts[y] += 1
                        # we haven't changed the start positions
                        starts = [edge_1_start, edge_2_start]
                        # find longest
                        if counts[0] >= counts[1]:
                            keep = 0
                            break_up = 1
                        else:
                            keep = 1
                            break_up = 0
                        # get longest points as merged set
                        merged_points = []
                        for dx in range(counts[keep]):
                            merged_points.append(line[(starts[keep] + dx) % max_x][keep])
                        # replace edge_1 with merged, drop edge_2
                        ring_edges[edge_1] = [edge_1_type, starts[keep], merged_points]
                        del ring_edges[edge_2 % len(ring_edges)]
                        # find leading and trailing points in the other edge
                        leading_points = []
                        trailing_points = []
                        trailing_start = None
                        for dx in range(counts[break_up]):
                            x = (starts[break_up] + dx) % max_x
                            ys = line[x]
                            if ys[keep] is None:
                                if trailing_start is None:
                                    leading_points.append(ys[break_up])
                                else:
                                    trailing_points.append(ys[break_up])
                            else:
                                # found the merge point, switch ends
                                trailing_start = x
                        # insert edge residues if they are long enough
                        if len(leading_points) >= MIN_HORIZONTAL_EDGE_LENGTH:
                            # insert leading points
                            ring_edges.append([edge_1_type, starts[break_up], leading_points])
                        if len(trailing_points) > MIN_HORIZONTAL_EDGE_LENGTH:
                            # insert trailing points
                            ring_edges.append([edge_1_type, trailing_start, trailing_points])
                        # go round again
                        changed = True
                        break
                    if changed:
                        break
            if self.logging:
                # sort into x,y to make looking at logs easier
                ring_edges.sort(key=lambda e: (e[1], e[2][0]))
                self._log('find_data_boxes: discovered ring edges (after merging):')
                for edge in ring_edges:
                    self._log('    {}'.format(edge))
            # endregion
        if reason is None:
            # region Phase 5 - find likely digit edges
            pass  # ToDo: find likely digit edges
            # endregion
        if reason is None:
            # region Phase 6 - find likely ring edges
            pass  # ToDo: find likely ring edges
            # endregion
        if self.save_images:
            # show ideal boxes (as in blocks and ring_edge_estimates),
            # and discovered boxes (as in digit_edges and ring_edges)
            found_digits_rising = []
            found_digits_falling = []
            for x, edges in digit_edges:
                for edge_type, start_y, end_y in edges:
                    if edge_type == Scan.RISING:
                        found_digits_rising.append((x, start_y, x, end_y - 1))
                    else:
                        found_digits_falling.append((x, start_y, x, end_y - 1))
            found_rings_rising = []
            found_rings_falling = []
            for edge_type, edge_start, edge_points in ring_edges:
                if edge_type == Scan.RISING:
                    found_rings_rising.append((edge_start, edge_points))
                else:
                    found_rings_falling.append((edge_start, edge_points))
            plot = self._draw_extent(extent, buckets)
            plot = self._draw_lines(plot, found_digits_rising, Scan.BLUE, bleed=0.6)
            plot = self._draw_lines(plot, found_digits_falling, Scan.GREEN, bleed=0.6)
            plot = self._draw_plots(plot, plots_x=found_rings_rising, colour=Scan.BLUE, bleed=0.6)
            plot = self._draw_plots(plot, plots_x=found_rings_falling, colour=Scan.GREEN, bleed=0.6)
            self._unload(plot, '06-boxes-found')
        ###############################################
        # ToDo: HACK old scheme based on guessing
        ###############################################
        return self.__find_data_boxes(extent, digits, zeroes)

    def __find_data_boxes(self, extent: Extent, digits: [Digit], zeroes: [[int]]) -> ([Box], str):
        """ find the boundaries for all the data boxes given an extent a digit list and zero positions,
            bit boxes are delineated horizontally by inner/outer edge divided by 5 (inner/outer black + 3 data rings)
            and vertically by the zero gap divided by 4 (digits in a code less the zero) for each copy,
            the horizontal edge is a y co-ord for each x co-ord, the vertical edge is an x co-ord for each digit,
            each bit box is defined by a start x co-ord and a list of y edges for each ring,
            returns a box list and None if OK, or a partial list and a fail reason if not,
            the y co-ords of the edges are exclusive
            """

        # self._find_boundaries(extent)  # ToDo: HACK

        inner = extent.inner
        outer = extent.outer
        buckets = extent.buckets
        max_x, max_y = buckets.size()
        reason = None
        boxes = []
        if reason is None:
            # region Phase 1 - estimate where the bit boundaries should be based on the zero positions...
            for z in range(len(zeroes)):
                zero = digits[zeroes[z][0]]
                digit_start = (zero.start + zero.samples) % max_x
                last_digit = digits[zeroes[(z + 1) % len(zeroes)][0]]
                if last_digit.start < digit_start:
                    # we've wrapped
                    digit_width = (last_digit.start + max_x) - digit_start
                else:
                    digit_width = last_digit.start - digit_start
                digit_width /= (Scan.DIGITS_PER_NUM - 1)  # -1 'cos we're excluding the zero
                if digit_width < 1:
                    # not enough room for digits, so this is junk (it means the zeroes are too big)
                    reason = 'zeroes too big'
                    break
                boxes.append(Scan.Box(zero.start, zero.samples))
                edge = []
                for _ in range(Scan.DIGITS_PER_NUM - 1):  # -1 'cos already done the zero
                    edge.append(int(round(digit_start)))  # ignore wrapping for now
                    digit_start += digit_width
                edge.append(last_digit.start)  # so we can always use index+1 to calculate true width
                for digit in range(Scan.DIGITS_PER_NUM - 1):  # -1 'cos already done the zero
                    boxes.append(Scan.Box(edge[digit] % max_x, (edge[digit + 1] - edge[digit]) % max_x))
            # endregion
        if reason is None:
            # region Phase 2 - estimate where the ring boundaries should be based on the inner and outer edge positions...
            for box in boxes:
                edges = [[] for _ in range(Scan.NUM_DATA_RINGS)]
                for x in range(box.start, box.start + box.span):
                    y_start = inner[x % max_x] + 1  # +1 to get into the black
                    y_end = outer[x % max_x]
                    ring_width = (y_end - y_start) / Scan.INNER_OUTER_SPAN_RINGS
                    if ring_width < 1:
                        # not enough room for rings, so this is junk
                        reason = 'data too narrow'
                        break
                    edge = []
                    for _ in range(Scan.INNER_BLACK_RINGS):  # skip inner black ring(s)
                        y_start += ring_width
                    for _ in range(Scan.NUM_DATA_RINGS + 1):  # +1 to get end of final ring
                        edge.append(int(round(y_start)))  # leading edge
                        y_start += ring_width
                    for ring in range(Scan.NUM_DATA_RINGS):
                        edges[ring].append([edge[ring], edge[ring + 1]])  # trailing edge
                for edge in edges:
                    box.add_edge(edge)
            # endregion

        if self.logging:
            # put boxes into start order - makes matching diagnostic images with the logs easier
            boxes.sort(key=lambda b: b.start)

        if self.save_images:
            # draw data boxes
            plot = self._draw_edges((extent.falling_edges, extent.rising_edges), buckets, extent)
            plot = self._draw_boxes(boxes, plot)
            self._unload(plot, '06-boxes')

        if self.logging:
            if reason is not None:
                self._log('find_data_boxes: failed: {}'.format(reason))

        return boxes, reason

    def _decode_data_boxes(self, boxes: [Box], extent: Extent):
        """ given a set of data boxes, decode the bits they represent,
            a data box is a '1' if it is mostly white or a '0' if it is mostly black,
            however, due to 'smudging' effects what is considered 'mostly' is dependent
            on the neighbour boxes, 'smudging' is white pixels bleeding into neighbouring
            black pixels, each box is considered to consist of N x N blocks, in a 2D gird
            like this:
                  +-----+------+-...-+--------+
                  |   0 |    1 | ... | N-1    |
                  +-----+------+-...-+--------+
                  |   N |  N+1 | ... | N+N-1  |
                  +-----+---- -+-----+--------+
                  |  2N | 2N+1 | ... | 2N+N-1 |
                  +-----+------+-----+--------+
                  |  .. | .... | ... | ...... |
                  +-----+------+-----+--------+
                  |  .. | .... | ... | N*N-1  |
                  +-----+------+-----+--------+
            and each box has 4 neighbours:
                  | above |
                --+-------+--
                  |       |
             left | self  | right
                  |       |
                --+-------+--
                    below
            left bleed source are all except left-most blocks
            above bleed source blocks are all except top-most blocks
            right bleed source blocks are all except right-most blocks
            below bleed source blocks are all except bottom-most blocks
            when there is a bleeding source that is white on two sides the intersecting corner is ignored in self
            left plus above white ignores top-left corner
            above plus right white ignores top-right corner
            right plus below white ignores bottom-right corner
            below plus left white ignores bottom-left corner
            if all blocks are being ignored then self cell is white (as it implies both above and below are white)
            if non-ignored blocks are white then self cell is white
            if above bottom half and self top half are white then self is white if all self whiter than all above
            if below top half and self bottom half are white then self is white if all self whiter than all below
            otherwise self cell is black
        """

        buckets = extent.buckets
        max_x, max_y = buckets.size()

        # ToDo: move to Scan constants
        NEIGHBOUR_WHITE_THRESHOLD = 0.7  # white ratio above which a neighbour block is considered to be white
        SELF_WHITE_THRESHOLD = 0.45  # white ratio above which a self block is considered to be white
        CELL_BOUNDARY_PIXELS = 1  # this many pixels around a cell boundary are ignored for determining a cell whiteness

        class Ratios:
            """ whiteness ratios for the blocks of a cell """

            def __init__(self, blocks: [[int, int]]):
                """ white coverage of the N blocks as white count and total count for each """
                self.blocks = blocks
                self.ratios = []
                for block in self.blocks:
                    self.ratios.append(self._get_ratio(block))

            def __str__(self):
                msg = ''
                for x, ratio in enumerate(self.ratios):
                    msg = '{} {:1X}={:02d} {}'.format(msg, x, int(round(ratio * 10)), self.blocks[x])
                return '[{}]'.format(msg[1:])

            def _get_ratio(self, parts: (int, int)) -> float:
                if parts[1] == 0:
                    return 0
                else:
                    return parts[0] / parts[1]

            def get_whiteness(self, blocks: [int] = None) -> float:
                """ get the average whiteness for the given blocks,
                    the ratios are the white coverage in the range 0..1,
                    if blocks is None all are considered,
                    if blocks is empty 0 is returned
                    """
                if blocks is None:
                    blocks = [block for block in range(len(self.ratios))]
                if len(blocks) == 0:
                    return 0
                whiteness = 0
                samples = 0
                for block in blocks:
                    whiteness += self.ratios[block]
                    samples += 1
                return whiteness / samples  # return average across the addressed quadrants

        class Criterion:
            """ criterion for a set of blocks to be considered white """

            def __init__(self, address: (float, float), blocks: [int], threshold: float):
                self.address   = address    # x,y co-ord offsets of the box the criterion applies to
                self.blocks = blocks  # the blocks to check within that addressed box
                self.threshold = threshold  # the block whiteness threshold for the criterion (0..1)

            def satisfied(self, segments: [[Ratios]], x: int, y: int) -> bool:
                """ test if the criterion is satisfied within the segments at x,y,
                    x wraps, y does not
                    to be satisfied the addressed blocks average must be at least the white threshold
                    """
                whiteness = self.get_whiteness(segments, x, y)
                if whiteness is None:
                    return None
                if whiteness >= self.threshold:
                    return True
                else:
                    return False

            def get_whiteness(self, segments: [[Ratios]], x: int, y: int) -> float:
                """ get the whiteness ratio for our criterion """
                ratios = self.get_ratios(segments, x, y)
                if ratios is None:
                    return None
                return ratios.get_whiteness(self.blocks)

            def get_ratios(self, segments: [[Ratios]], x: int, y: int) -> Ratios:
                """ get the whiteness ratios for our criterion,
                    returns the Ratios or None if criterion address does not exist
                    """
                dy = y + self.address[1]
                if dy < 0 or dy >= len(segments[0]):
                    # y does not wrap
                    return None
                dx = (x + self.address[0]) % len(segments)  # x wraps
                return segments[dx][dy]

        # region criteria for black/white discrimination...
        # blocks per cell
        # BLOCKS_PER_ROW = 4
        # BLOCKS_PER_COL = 4
        # BLOCKS_PER_BOX = BLOCKS_PER_ROW * BLOCKS_PER_COL
        # # criteria block sets
        # TO_LEFT_BLOCKS  = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15}
        # TO_BELOW_BLOCKS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
        # TO_RIGHT_BLOCKS = {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14}
        # TO_ABOVE_BLOCKS = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
        # TOP_BLOCKS      = {0, 1, 2, 3, 4, 5, 6, 7}
        # BOTTOM_BLOCKS   = {8, 9, 10, 11, 12, 13, 14, 15}
        # ALL_BLOCKS      = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
        # # ignore block sets
        # TOP_LEFT_CORNER     = {0, 1, 2, 4, 5, 8}
        # TOP_RIGHT_CORNER    = {1, 2, 3, 6, 7, 11}
        # BOTTOM_RIGHT_CORNER = {7, 10, 11, 13, 14, 15}
        # BOTTOM_LEFT_CORNER  = {4, 8, 9, 12, 13, 14}
        BLOCKS_PER_ROW = 3
        BLOCKS_PER_COL = 3
        BLOCKS_PER_BOX = BLOCKS_PER_ROW * BLOCKS_PER_COL
        # criteria block sets
        TO_LEFT_BLOCKS  = {1, 2, 4, 5, 7, 8}
        TO_BELOW_BLOCKS = {0, 1, 2, 3, 4, 5}
        TO_RIGHT_BLOCKS = {0, 1, 3, 4, 6, 7}
        TO_ABOVE_BLOCKS = {3, 4, 5, 6, 7, 8}
        TOP_BLOCKS      = {0, 1, 2, 3, 4, 5}
        BOTTOM_BLOCKS   = {3, 4, 5, 6, 7, 8}
        ALL_BLOCKS      = {0, 1, 2, 3, 4, 5, 6, 7, 8}
        # ignore block sets
        TOP_LEFT_CORNER     = {0, 1, 2, 3, 4, 6}
        TOP_RIGHT_CORNER    = {0, 1, 2, 4, 5, 8}
        BOTTOM_RIGHT_CORNER = {2, 4, 5, 6, 7, 8}
        BOTTOM_LEFT_CORNER  = {0, 3, 4, 6, 7, 8}
        # neighbour addresses
        ABOVE = [0, -1]  # x,y co-ord offsets to get to the 'above' box
        BELOW = [0, +1]  # x,y co-ord offsets to get to the 'below' box
        HERE  = [0,  0]  # x,y co-ord offsets to get to the 'self' box
        LEFT  = [-1, 0]  # x,y co-ord offsets to get to the 'left' box
        RIGHT = [+1, 0]  # x,y co-ord offsets to get to the 'left' box
        # neighbour criteria
        TO_LEFT_IS_WHITE  = Criterion(LEFT , TO_LEFT_BLOCKS , NEIGHBOUR_WHITE_THRESHOLD)
        TO_ABOVE_IS_WHITE = Criterion(ABOVE, TO_ABOVE_BLOCKS, NEIGHBOUR_WHITE_THRESHOLD)
        TO_RIGHT_IS_WHITE = Criterion(RIGHT, TO_RIGHT_BLOCKS, NEIGHBOUR_WHITE_THRESHOLD)
        TO_BELOW_IS_WHITE = Criterion(BELOW, TO_BELOW_BLOCKS, NEIGHBOUR_WHITE_THRESHOLD)
        # self criteria
        ABOVE_BOTTOM_IS_WHITE = Criterion(ABOVE, BOTTOM_BLOCKS, SELF_WHITE_THRESHOLD)
        SELF_TOP_IS_WHITE     = Criterion(HERE , TOP_BLOCKS   , SELF_WHITE_THRESHOLD)
        SELF_BOTTOM_IS_WHITE  = Criterion(HERE , BOTTOM_BLOCKS, SELF_WHITE_THRESHOLD)
        BELOW_TOP_IS_WHITE    = Criterion(BELOW, TOP_BLOCKS   , SELF_WHITE_THRESHOLD)
        # all blocks criteria
        ALL_ABOVE = Criterion(ABOVE, ALL_BLOCKS, SELF_WHITE_THRESHOLD)
        ALL_SELF  = Criterion(HERE , ALL_BLOCKS, SELF_WHITE_THRESHOLD)
        ALL_BELOW = Criterion(BELOW, ALL_BLOCKS, SELF_WHITE_THRESHOLD)
        # endregion

        def get_box_white_ratios(ring, segment) -> Ratios:
            """ get the white block coverage of the given box,
                returns the ratio of white to black for each block,
                all ratios are in the range 0..1,
                to mitigate against off-by-one boundary estimate errors the boundary pixels
                of each cell are ignored
                """
            blocks = [[0, 0] for _ in range(BLOCKS_PER_BOX)]  # white count, total count for each block
            box = boxes[segment]
            edge = box.edges[ring]
            start_x = box.start
            x_span = box.span - (2 * CELL_BOUNDARY_PIXELS)
            for dx in range(CELL_BOUNDARY_PIXELS, len(edge) - CELL_BOUNDARY_PIXELS):
                start_y, end_y = edge[dx]
                start_y += CELL_BOUNDARY_PIXELS
                end_y   -= CELL_BOUNDARY_PIXELS
                y_span = end_y - start_y
                for y in range(start_y, end_y):  # end_y is exclusive, so this range is correct
                    # get the pixel colour
                    pixel = buckets.getpixel((start_x + dx) % max_x, y)
                    if pixel == MIN_LUMINANCE:
                        white = 0
                    else:
                        white = 1
                    # which block(s) are we in?
                    x_here = ((dx - CELL_BOUNDARY_PIXELS) / x_span) * BLOCKS_PER_ROW  # range 0..BLOCKS_PER_ROW
                    y_here = ((y - start_y) / y_span) * BLOCKS_PER_COL  # range 0..BLOCKS_PER_COL
                    x_next = int(round(x_here))
                    y_next = int(round(y_here))
                    x_here = int(x_here)
                    y_here = int(y_here)
                    update_blocks = [(x_here, y_here)]  # always update x_here, y_here
                    if x_here != x_next:
                        # on an x pixel boundary
                        update_blocks.append((x_next, y_here))
                    if y_here != y_next:
                        # on a y pixel boundary
                        update_blocks.append((x_here, y_next))
                    if x_here != x_next and y_here != y_next:
                        # on both x and y pixel boundaries
                        update_blocks.append((x_next, y_next))
                    # update block counts
                    for _x, _y in update_blocks:
                        _x = min(_x, BLOCKS_PER_ROW - 1)
                        _y = min(_y, BLOCKS_PER_COL - 1)
                        block = (_y * BLOCKS_PER_ROW) + _x
                        blocks[block][0] += white
                        blocks[block][1] += 1
            result = Ratios(blocks)
            return result

        def is_both_satisfied(segment: int, ring: int, criterion_1: Criterion, criterion_2: Criterion) -> bool:
            """ determine if both the criterion given are satisfied """
            is_satisfied_1 = criterion_1.satisfied(segments, segment, ring)
            is_satisfied_2 = criterion_2.satisfied(segments, segment, ring)
            if is_satisfied_1 is not None and is_satisfied_1 and is_satisfied_2 is not None and is_satisfied_2:
                return True
            else:
                return False

        def show_set(this_set):
            """ format the given block set in a form suitable for logging """
            blocks = ['_' for _ in range(BLOCKS_PER_BOX)]
            for block in this_set:
                blocks[block] = '{:1X}'.format(block)
            return '[' + ''.join(blocks) + ']'

        segments = [[None for _ in range(Scan.NUM_DATA_RINGS)] for _ in range(Scan.NUM_SEGMENTS)]
        # region Phase 1 - get the black coverage of every quadrant of every box...
        for segment in range(Scan.NUM_SEGMENTS):
            for ring in range(Scan.NUM_DATA_RINGS):
                segments[segment][ring] = get_box_white_ratios(ring, segment)
        # endregion
        # region Phase 2 - decode coverage...
        bits    = [[None for _ in range(Scan.NUM_DATA_RINGS)] for _ in range(Scan.NUM_SEGMENTS)]
        reasons = [[None for _ in range(Scan.NUM_DATA_RINGS)] for _ in range(Scan.NUM_SEGMENTS)]
        for segment in range(Scan.NUM_SEGMENTS):
            for ring in range(Scan.NUM_DATA_RINGS):
                # region determine what to sample...
                sample_set = {x for x in range(BLOCKS_PER_BOX)}
                why = []
                if is_both_satisfied(segment, ring, TO_LEFT_IS_WHITE, TO_BELOW_IS_WHITE):
                    sample_set.difference_update(BOTTOM_LEFT_CORNER)
                    why.append('ignore bottom left corner {}'.format(show_set(BOTTOM_LEFT_CORNER)))
                if is_both_satisfied(segment, ring, TO_RIGHT_IS_WHITE, TO_BELOW_IS_WHITE):
                    sample_set.difference_update(BOTTOM_RIGHT_CORNER)
                    why.append('ignore bottom right corner {}'.format(show_set(BOTTOM_RIGHT_CORNER)))
                if is_both_satisfied(segment, ring, TO_LEFT_IS_WHITE, TO_ABOVE_IS_WHITE):
                    sample_set.difference_update(TOP_LEFT_CORNER)
                    why.append('ignore top left corner {}'.format(show_set(TOP_LEFT_CORNER)))
                if is_both_satisfied(segment, ring, TO_RIGHT_IS_WHITE, TO_ABOVE_IS_WHITE):
                    sample_set.difference_update(TOP_RIGHT_CORNER)
                    why.append('ignore top right corner {}'.format(show_set(TOP_RIGHT_CORNER)))
                # endregion
                # region determine what the bit should be...
                bit = None
                if len(sample_set) == 0:
                    # if all blocks are being ignored then self cell is white
                    bit = 1
                    why.append('all ignored so white')
                if bit is None:
                    here_is_white = Criterion(HERE, sample_set, SELF_WHITE_THRESHOLD)
                    if here_is_white.satisfied(segments, segment, ring):
                        # if non-ignored blocks are white then self cell is white
                        bit = 1
                        why.append('non-ignored white {}'.format(show_set(sample_set)))
                if bit is None:
                    if is_both_satisfied(segment, ring, ABOVE_BOTTOM_IS_WHITE, SELF_TOP_IS_WHITE):
                        above_whiteness = ALL_ABOVE.get_whiteness(segments, segment, ring)
                        self_whiteness  = ALL_SELF.get_whiteness(segments, segment, ring)
                        if self_whiteness > above_whiteness:
                            # if above bottom and self top then self is white if all self whiter than all above
                            bit = 1
                            why.append('above bottom {} and self top {} white'.
                                    format(show_set(BOTTOM_BLOCKS), show_set(TOP_BLOCKS)))
                if bit is None:
                    if is_both_satisfied(segment, ring,BELOW_TOP_IS_WHITE, SELF_BOTTOM_IS_WHITE):
                        below_whiteness = ALL_BELOW.get_whiteness(segments, segment, ring)
                        self_whiteness = ALL_SELF.get_whiteness(segments, segment, ring)
                        if self_whiteness >= below_whiteness:  # NB: condition must be exclusive with above
                            # if below top and self bottom then self is white if all self whiter than all below
                            bit = 1
                            why.append('self bottom {} and below top {} white'.
                                    format(show_set(BOTTOM_BLOCKS), show_set(TOP_BLOCKS)))
                if bit is None:
                    # otherwise self cell is black
                    bit = 0
                    why.append('non-ignored black {}'.format(show_set(sample_set)))
                bits[segment][ring] = bit
                reasons[segment][ring] = why
                # endregion
        # endregion
        # region Phase 3 - translate bits to digits...
        digits = []
        for segment in bits:
            digits.append(self.decoder.digit(segment))
        # endregion
        # region diagnostics...
        if self.save_images:
            plot = self._draw_bits(bits, boxes, buckets)
            self._unload(plot, '07-bits')

        if self.logging:
            self._log('decode_data_boxes:')
            for segment in range(Scan.NUM_SEGMENTS):
                self._log('    segment {}'.format(segment))
                for ring in range(Scan.NUM_DATA_RINGS):
                    ratios = segments[segment][ring]
                    bit = bits[segment][ring]
                    why = reasons[segment][ring]
                    self._log('        ring {}: {} <--{}--> {}'.format(ring, bit, ratios, why))
            self._log('    digits: {}'.format(digits))
            code, doubt = self.decoder.unbuild(digits)
            num = self.decoder.decode(code)
            self._log('    num: {}, doubt: {}, code:{}'.format(num, doubt, code))
        # endregion

        return digits

    def _decode_digits(self, digits: [Digit]) -> [Result]:
        """ decode the digits into their corresponding code and doubt """

        bits = []
        error = 0
        samples = 0
        for digit in digits:
            bits.append(digit.digit)
            error += digit.error
            samples += digit.samples
        error /= samples  # average error per sample - 0..1
        code, doubt = self.decoder.unbuild(bits)
        number = self.decoder.decode(code)
        if self.logging:
            msg = ''
            for digit in digits:
                msg = '{}, {}'.format(msg, digit)
            self._log('decode_digits: num:{}, doubt:{}, code:{}, error:{:.3f} from: {}'.
                      format(number, doubt, code, error, msg[2:]))
        return Scan.Result(number, doubt + error, code, digits)

    def _find_codes(self) -> ([Target], frame.Frame):
        """ find the codes within each blob in our image,
            returns a list of potential targets
            """

        # find the blobs in the image
        blobs = self._blobs()
        if len(blobs) == 0:
            # no blobs here
            return [], None

        targets = []
        rejects = []
        for blob in blobs:
            self.centre_x = blob[0]
            self.centre_y = blob[1]
            blob_size = blob[2]

            if self.logging:
                self._log('***************************')
                self._log('processing candidate target')

            # find the extent of our target
            result = self._identify(blob_size)
            if result is None:
                # this means the blob has insufficient contrast (already logged)
                # tag it as deleted
                blob[2] = 0
                continue

            max_x, max_y, stretch_factor, extent = result
            if extent.inner_fail is not None or extent.outer_fail is not None:
                # failed - this means we did not find its inner and/or outer edge
                if self.save_images:
                    # add to reject list for labelling on the original image
                    reason = extent.inner_fail
                    if reason is None:
                        reason = extent.outer_fail
                    rejects.append(Scan.Reject(self.centre_x, self.centre_y, blob_size, blob_size*4, reason))
                continue

            digits, reason = self._find_all_digits(extent)
            if reason is None:
                digits, reason = self._find_best_digits(digits, extent)
            if reason is not None:
                # we failed to find required digits
                if self.save_images:
                    # add to reject list for labelling on the original image
                    rejects.append(Scan.Reject(self.centre_x, self.centre_y, blob_size, blob_size*4, reason))
                continue

            result = self._decode_digits(digits)
            target_size, _ = self._measure(extent, stretch_factor)
            targets.append(Scan.Target(self.centre_x, self.centre_y, blob_size, target_size, result))

        if self.save_images:
            # draw the accepted blobs
            grid = self._draw_blobs(self.image, blobs)
            self._unload(grid, 'blobs', 0, 0)
            # label all the blobs we processed that were rejected
            labels = self.transform.copy(self.image)
            for reject in rejects:
                x = reject.centre_x
                y = reject.centre_y
                blob_size = reject.blob_size
                target_size = reject.target_size
                reason = reject.reason
                # show blob detected
                labels = self.transform.label(labels, (x, y, blob_size), Scan.BLUE)
                # show reject reason
                labels = self.transform.label(labels, (x, y, target_size), Scan.RED,
                                              '{:.0f}x{:.0f}y {}'.format(x, y, reason))
        else:
            labels = None

        return targets, labels

    def decode_targets(self):
        """ find and decode the targets in the source image,
            returns a list of x,y blob co-ordinates, the encoded number there (or None) and the level of doubt
            """

        if self.logging:
            self._log('Scan starting...', None, None)

        targets, labels = self._find_codes()
        if len(targets) == 0:
            if self.logging:
                self._log('{}: - decode_targets: image does not contain any target candidates'.format(self.original.source),
                          0, 0, prefix=False)
            if self.save_images:
                if labels is not None:
                    self._unload(labels, 'targets', 0, 0)
                else:
                    self._unload(self.image, 'targets', 0, 0)
            return []

        if self.save_images:
            detections = self.transform.copy(self.image)

        numbers = []
        for target in targets:
            self.centre_x = target.centre_x    # for logging and labelling
            self.centre_y = target.centre_y    # ..
            blob_size = target.blob_size
            target_size = target.target_size
            result = target.result

            # add this result
            numbers.append(Scan.Detection(result, self.centre_x, self.centre_y, target_size, blob_size))

            if self.save_images:
                detection = numbers[-1]
                result = detection.result
                if result.number is None:
                    colour = Scan.RED
                    label = 'invalid ({:.4f})'.format(result.doubt)
                else:
                    colour = Scan.GREEN
                    label = 'number is {} ({:.4f})'.format(result.number, result.doubt)
                # draw the detected blob in blue
                k = (detection.centre_x, detection.centre_y, detection.blob_size)
                detections = self.transform.label(detections, k, Scan.BLUE)
                # draw the result
                k = (detection.centre_x, detection.centre_y, detection.target_size)
                detections = self.transform.label(detections, k, colour, '{:.0f}x{:.0f}y {}'.
                                                  format(detection.centre_x, detection.centre_y, label))

        if self.save_images:
            self._unload(detections, 'targets', 0, 0)
            if labels is not None:
                self._unload(labels, 'rejects', 0, 0)

        return numbers

    # region Helpers...
    def _remove_file(self, f, silent=False):
        try:
            os.remove(f)
        except:
            if silent:
                pass
            else:
                traceback.print_exc()
                self._log('Could not remove {}'.format(f))

    def _log(self, message, centre_x=None, centre_y=None, fatal=False, console=False, prefix=True):
        """ print a debug log message
            centre_x/y are the co-ordinates of the centre of the associated blob, if None use decoding context
            centre_x/y of 0,0 means no x/y identification in the log
            iff fatal is True an exception is raised, else the message is just printed
            iff console is True the message is logged to console regardless of other settings
            iff prefix is True the message is prefixed with the current log prefix
            """
        if centre_x is None:
            centre_x = self.centre_x
        if centre_y is None:
            centre_y = self.centre_y
        if centre_x > 0 and centre_y > 0:
            message = '{:.0f}x{:.0f}y - {}'.format(centre_x, centre_y, message)
        if prefix:
            message = '{} {}'.format(self._log_prefix, message)
        message = '[{:08.4f}] {}'.format(time.thread_time(), message)
        if self._log_folder:
            # we're logging to a file
            if self._log_file is None:
                filename, ext = os.path.splitext(self.original.source)
                log_file = '{}/{}.log'.format(self._log_folder, filename)
                self._remove_file(log_file, silent=True)
                self._log_file = open(log_file, 'w')
            self._log_file.write('{}\n'.format(message))
        if fatal:
            raise Exception(message)
        elif self.show_log or console:
            # we're logging to the console
            print(message)

    def _log_edge(self, edge: [int], prefix: str):
        """ log the given edge in 32 byte chunks, prefix is prepended to each line,
            this is a logging helper for logging long sequences of small integers in a readable manner
            """
        if edge is None:
            return
        max_edge = len(edge)
        block_size = 32
        blocks = int(max_edge / block_size)
        for block in range(blocks):
            start_block = block * block_size
            end_block = start_block + block_size
            self._log('{}{:3d}: {}'.format(prefix, start_block, edge[start_block : end_block]))
        residue = max_edge - (blocks * block_size)
        if residue > 0:
            self._log('{}{:3d}: {}'.format(prefix, len(edge) - residue, edge[-residue:]))

    def _unload(self, image, suffix, centre_x=None, centre_y=None):
        """ unload the given image with a name that indicates its source and context,
            suffix is the file name suffix (to indicate context),
            centre_x/y identify the blob the image represents, if None use decoding context,
            centre_x/y of 0,0 means no x/y identification on the image,
            as a diagnostic aid to find co-ordinates, small tick marks are added along the edges every 10 pixels,
            if a centre_x/y is given a folder for that is created and the image saved in it,
            otherwise a folder for the source is created and the image saved in that
            """

        # add tick marks
        max_x, max_y = image.size()
        lines = []
        for x in range(10, max_x, 10):
            lines.append([x, 0, x, 1])
            lines.append([x, max_y - 2, x, max_y - 1])
        image = self._draw_lines(image, lines, Scan.PINK)
        lines = []
        for y in range(10, max_y, 10):
            lines.append([0, y, 1, y])
            lines.append([max_x - 2, y, max_x - 1, y])
        image = self._draw_lines(image, lines, Scan.PINK)

        # construct parent folder to save images in for this source
        filename, _ = os.path.splitext(self.original.source)
        parent = '_{}'.format(filename)

        # construct the file name
        if centre_x is None:
            centre_x = self.centre_x
        if centre_y is None:
            centre_y = self.centre_y
        if centre_x > 0 and centre_y > 0:
            # use a sub-folder for this blob
            folder = '{}/_{}-{:.0f}x{:.0f}y'.format(parent, filename, centre_x, centre_y)
            name = '{:.0f}x{:.0f}y-'.format(centre_x, centre_y)
        else:
            # use given folder for non-blobs
            folder = parent
            name = ''

        # save the image
        filename = image.unload(self.original.source, '{}{}'.format(name, suffix), folder=folder)
        if self.logging:
            self._log('{}: image saved as: {}'.format(suffix, filename), centre_x, centre_y)

    def _draw_blobs(self, source, blobs: List[tuple], colour=GREEN):
        """ draw circular blobs in the given colour, each blob is a centre x,y and a size (radius),
            blobs with no size are not drawn
            returns a new colour image of the result
            """
        objects = []
        for blob in blobs:
            if blob[2] > 0:
                objects.append({"colour": colour,
                                "type": self.transform.CIRCLE,
                                "centre": (int(round(blob[0])), int(round(blob[1]))),
                                "radius": int(round(blob[2]))})
        target = self.transform.copy(source)
        return self.transform.annotate(target, objects)

    def _draw_plots(self, source, plots_x=None, plots_y=None, colour=RED, bleed=0.5):
        """ draw plots in the given colour, each plot is a set of points and a start x or y,
            returns a new colour image of the result
            """
        objects = []
        if plots_x is not None:
            for plot in plots_x:
                objects.append({"colour": colour,
                                "type": self.transform.PLOTX,
                                "start": plot[0],
                                "bleed": (bleed, bleed, bleed),
                                "points": plot[1]})
        if plots_y is not None:
            for plot in plots_y:
                objects.append({"colour": colour,
                                "type": self.transform.PLOTY,
                                "start": plot[0],
                                "bleed": (bleed, bleed, bleed),
                                "points": plot[1]})
        target = self.transform.copy(source)
        return self.transform.annotate(target, objects)

    def _draw_lines(self, source, lines, colour=RED, bleed=0.5, h_wrap=False, v_wrap=False):
        """ draw lines in given colour,
            lines param is an array of start-x,start-y,end-x,end-y tuples,
            for horizontal lines start-y and end-y are the same,
            for vertical lines start-x and end-x are the same,
            for horizontal and vertical lines bleed defines how much background bleeds through.
            for horizontal lines h_wrap is True iff they wrap in the x-direction,
            for vertical lines v_wrap is True iff they wrap in the y-direction,
            """
        dlines = []
        vlines = []
        hlines = []
        max_x, max_y = source.size()
        for line in lines:
            if line[0] == line[2]:
                # vertical line
                if v_wrap and line[3] < line[1]:
                    vlines.append([line[1], [line[0] for _ in range(line[1], max_y)]])
                    vlines.append([0, [line[0] for _ in range(0, line[3]+1)]])
                else:
                    vlines.append([line[1], [line[0] for _ in range(line[1], line[3]+1)]])
            elif line[1] == line[3]:
                # horizontal line
                if h_wrap and line[2] < line[0]:
                    hlines.append([line[0], [line[1] for _ in range(line[0], max_x)]])
                    hlines.append([0, [line[1] for _ in range(0, line[2]+1)]])
                else:
                    hlines.append([line[0], [line[1] for _ in range(line[0], line[2]+1)]])
            else:
                dlines.append({"colour": colour,
                                "type": self.transform.LINE,
                                "start": [line[0], line[1]],
                                "end": [line[2], line[3]]})
        target = source
        if len(dlines) > 0:
            target = self.transform.copy(target)
            target = self.transform.annotate(target, dlines)
        if len(vlines) > 0 or len(hlines) > 0:
            target = self._draw_plots(target, hlines, vlines, colour, bleed)
        return target

    def _draw_extent(self, extent: Extent, target, bleed=0.8):
        """ make the area outside the inner and outer edges on the given target visible """

        max_x, max_y = target.size()
        inner = extent.inner
        outer = extent.outer

        inner_lines = []
        outer_lines = []
        for x in range(max_x):
            if inner is not None and inner[x] is not None:
                inner_lines.append((x, 0, x, inner[x]))  # inner edge is on the last white
            if outer is not None and outer[x] is not None:
                outer_lines.append((x, outer[x], x, max_y - 1))  # outer edge is on first white
        target = self._draw_lines(target, inner_lines, colour=Scan.RED, bleed=bleed)
        target = self._draw_lines(target, outer_lines, colour=Scan.RED, bleed=bleed)

        target = self._draw_plots(target, plots_x=[[0, inner]], colour=Scan.RED, bleed=bleed/2)
        target = self._draw_plots(target, plots_x=[[0, outer]], colour=Scan.RED, bleed=bleed/2)

        return target

    def _draw_edges(self, edges, target: frame.Frame, extent: Extent=None, bleed=0.5):
        """ draw the edges and the extent on the given target image """

        falling_edges = edges[0]
        rising_edges = edges[1]

        plot = target

        # plot falling and rising edges
        falling_points = []
        rising_points = []
        if falling_edges is not None:
            for edge in falling_edges:
                falling_points.append((edge.where, edge.samples))
        if rising_edges is not None:
            for edge in rising_edges:
                rising_points.append((edge.where, edge.samples))
        plot = self._draw_plots(plot, plots_x=falling_points, colour=Scan.GREEN, bleed=bleed)
        plot = self._draw_plots(plot, plots_x=rising_points, colour=Scan.BLUE, bleed=bleed)

        if extent is not None:
            # mark the image area outside the inner and outer extent
            plot = self._draw_extent(extent, plot, bleed=0.8)

        return plot

    def _draw_boxes(self, boxes: [Box], target: frame.Frame):
        """ draw the data box boundaries on the given image """
        box_sides = []
        box_edges = []
        for box in boxes:
            if box.edges is None:
                continue
            else:
                box_sides.append((box.start, box.edges[0][0][0], box.start, box.edges[-1][0][1]))
                for edge in box.edges:
                    box_top = []
                    box_bottom = []
                    for y_start, y_end in edge:
                        box_top.append(y_start)
                        box_bottom.append(y_end)
                    box_edges.append([box.start, box_top])
                    box_edges.append([box.start, box_bottom])
        plot = self._draw_lines(target, box_sides, colour=Scan.RED)
        plot = self._draw_plots(plot, plots_x=box_edges, colour=Scan.RED)
        return plot

    def _draw_bits(self, bits: [[int]], boxes: [Box], target: frame.Frame):
        """ draw the given bits within the given boxes on the given image """
        max_x, _ = target.size()
        black_boxes = []
        white_boxes = []
        for segment, box in enumerate(boxes):
            if box.edges is None:
                continue
            for ring, edge in enumerate(box.edges):
                if bits[segment][ring] == 0:
                    # got a black box
                    fill = black_boxes
                elif bits[segment][ring] == 1:
                    # got a white box
                    fill = white_boxes
                else:
                    # got crap
                    continue
                x = box.start
                for y_start, y_end in edge:
                    fill.append((x % max_x, y_start, x % max_x, y_end-1))
                    x += 1
        plot = self._draw_lines(target, black_boxes, colour=Scan.BLUE)
        plot = self._draw_lines(plot, white_boxes, colour=Scan.GREEN)
        return plot

    # endregion
