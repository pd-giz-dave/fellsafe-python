
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

# ToDo: split this module into 3: bit level, digit level, code level

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

    # region Proximity options
    # these control the contour detection, for big targets that cover the whole image a bigger
    # integration area is required (i.e. smaller image fraction), this is used for testing
    # print images
    PROXIMITY_FAR = 48  # suitable for most images (photos and videos)
    PROXIMITY_CLOSE = 16  # suitable for print images
    BLACK_LEVEL = {PROXIMITY_FAR: -5, PROXIMITY_CLOSE: 0.01}  # black threshold for binarising contours
    # end region

    # our target shape
    NUM_RINGS = ring.Ring.NUM_RINGS  # total number of rings in the whole code (ring==cell in height)
    BULLSEYE_RINGS = ring.Ring.BULLSEYE_RINGS  # number of rings inside the inner edge
    DIGIT_BASE = codec.Codec.DIGIT_BASE  # number base for our digits
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
    BLOB_RADIUS_STRETCH = 1.3  # how much to stretch blob radius to ensure always cover everything when projecting
    MIN_CONTRAST = 0.35  # minimum luminance variation of a valid blob projection relative to the max luminance
    THRESHOLD_WIDTH = 8  # the fraction of the projected image width to use as the integration area when binarizing
    THRESHOLD_HEIGHT = 3.5  # the fraction of the projected image height to use as the integration area (None=as width)
    THRESHOLD_BLACK = 10  # the % below the average luminance in a projected image that is considered to be black
    THRESHOLD_WHITE = 0  # the % above the average luminance in a projected image that is considered to be white
    MIN_EDGE_SAMPLES = 2  # minimum samples in an edge to be considered a valid edge
    INNER_EDGE_GAP = 1.0  # fraction of inner edge y co-ord to add to inner edge when looking for the outer edge
    MAX_NEIGHBOUR_ANGLE_INNER = 0.4  # ~=22 degrees, tan of the max acceptable angle when joining inner edge fragments
    MAX_NEIGHBOUR_ANGLE_OUTER = 0.9  # ~=42 degrees, tan of the max acceptable angle when joining outer edge fragments
    MAX_NEIGHBOUR_HEIGHT_GAP = 1  # max x or y jump allowed when following an edge
    MAX_NEIGHBOUR_LENGTH_JUMP = 10  # max x jump, in pixels, between edge fragments when joining (arbitrary)
    MAX_NEIGHBOUR_HEIGHT_JUMP = 3  # max y jump, in pixels, between edge fragments when joining (arbitrary)
    MAX_NEIGHBOUR_OVERLAP = 4  # max edge overlap, in pixels, between edge fragments when joining (arbitrary)
    MAX_EDGE_GAP_SIZE = 3 / NUM_SEGMENTS  # max gap tolerated between edge fragments (as fraction of image width)
    SMOOTHING_WINDOW = 8  # samples in the moving average (we average the centre, so the full window is +/- this)
    MAX_NOT_ALLOWED_ERROR_DIFF = 0.15  # a not-allowed choice error within this of its neighbour is noise, else junk
    MAX_DIGIT_ERROR = 0.5  # digits with an error of more than this are dropped
    MAX_ZERO_ERROR_DIFF = 0.25  # a zero with a choice with a smaller error difference than this is dropped
    MAX_DIGIT_ERROR_DIFF = 0.05  # if two digit choices have an error difference less than this its ambiguous
    MAX_DIGIT_WIDTH = 2.0  # maximum width of a keep-able digit relative to the nominal digit width
    MIN_DIGIT_WIDTH = 0.3  # minimum width of a keep-able digit relative to the nominal digit width
    MIN_DROPPABLE_WIDTH = MIN_DIGIT_WIDTH * 3  # minimum width of a droppable digit
    MIN_SPLITTABLE_DIGIT_WIDTH = MIN_DIGIT_WIDTH * 2  # minimum width of a splittable digit (/2 must be >=1)
    # endregion

    # region Video modes image height...
    VIDEO_SD = 480
    VIDEO_HD = 720
    VIDEO_FHD = 1080
    VIDEO_2K = 1152
    VIDEO_4K = 2160
    # endregion

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

        def __init__(self, where, type, samples, grey_as):
            self.where = where  # the x co-ord of the start of this edge
            self.type = type  # the type of the edge, falling or rising
            self.samples = samples  # the list of connected y's making up this edge
            self.grey_as = grey_as

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
            return '{}(grey={}) at ({},{}) to ({},{}) for {}{}'.\
                   format(self.type, self.grey_as, from_x, from_y, to_x, to_y, len(self.samples), samples)

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
        params.black_threshold = Scan.BLACK_LEVEL[self.proximity]
        blobs, binary = contours.get_targets(self.image.buffer, params=params, logger=logger)
        self.binary = self.image.instance()
        self.binary.set(binary)

        if self.logging:
            # just so the processing order is deterministic (helps when viewing logs)
            blobs.sort(key=lambda e: (int(round(e[0])), int(round(e[1]))))
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
            self._log('project: projected image size {}x {}y (stretch factor {:.2f}, contrast {:.2f})'.
                      format(max_x, max_y, stretch_factor, contrast))

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

                if min_gap > (Scan.MAX_NEIGHBOUR_HEIGHT_GAP * Scan.MAX_NEIGHBOUR_HEIGHT_GAP):
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
                    return Scan.Edge(0, edge_type, sequence, treat_grey_as)
                elif end_x is None:
                    # this means the edge ends at the end
                    return Scan.Edge(start_x, edge_type, sequence[start_x:max_x], treat_grey_as)
                elif end_x < start_x:  # NB: end cannot be < start unless wrapping is allowed
                    # this means the edge wraps
                    return Scan.Edge(start_x, edge_type, sequence[start_x:max_x] + sequence[0:end_x+1], treat_grey_as)
                else:
                    # normal edge away from either extreme
                    return Scan.Edge(start_x, edge_type, sequence[start_x:end_x+1], treat_grey_as)

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
                x' is this_x - that_x and y' is this_y - that_y, x' cannot be zero,
                this_x and that_x may also wrap (ie. that < this),
                this_xy must precede that_xy in x, if not a wrap is assumed,
                the resultant x,y are always +ve
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
            if x_dash == 0:
                raise Exception('delta x is 0 between {} and {}'.format(this_xy, that_xy))
            return x_dash, y_dash

        def gap_OK(this_xy, that_xy, max_angle=None, max_x_distance=None, max_y_distance=None):
            """ check the angle of the slope and the distance between this_xy and that_xy are acceptable,
                an approximate tan of the angle is calculated as y'/x' with a few special cases,
                this_xy must precede that_xy in x, if not a wrap is assumed,
                angle 0 is considered to be along the x-axis, 90 degrees is straight up/down the y-axis,
                max_angle of None means do not check the angle,
                max_x/y_distance of None means do not check that distance.
                if all are None (silly) returns True
                """
            xy_dash = delta(this_xy, that_xy)
            if max_x_distance is not None:
                if xy_dash[0] > max_x_distance:
                    # too far apart
                    return False
            if max_y_distance is not None:
                if xy_dash[1] > max_y_distance:
                    # too far apart
                    return False
            if max_angle is not None:
                if xy_dash[1] <= Scan.MAX_NEIGHBOUR_HEIGHT_JUMP:
                    # always OK
                    return True
                if xy_dash[1] == 0:
                    # no difference in y, this is angle 0, which is OK
                    return True
                angle = xy_dash[1] / xy_dash[0]  # 1==45 degrees, 0.6==30, 1.2==50, 1.7==60, 2.7==70
                if angle > max_angle:
                    # too steep
                    return False
            return True

        def find_nearest_y(full_edge, start_x, direction):
            """ find the nearest y, and its position, in full_edge from start_x, in the given direction,
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
            nearest_back_xy = find_nearest_y(full_edge, edge_start, -1)  # backwards from our start
            nearest_next_xy = find_nearest_y(full_edge, edge_end, +1)  # forwards from our end
            if nearest_back_xy is None or nearest_next_xy is None:
                # means full_edge is empty - always OK
                return True

            our_start_xy = (edge.where, edge.samples[0])
            our_end_xy = (edge_end, edge.samples[-1])

            if gap_OK(nearest_back_xy, our_start_xy, max_angle, max_distance, max_distance):
                # angle and distance OK from our start to end of full
                # need to check the other end too, if our end is within max distance of next it must OK too
                if gap_OK(our_end_xy, nearest_next_xy, None, max_distance, None):
                    # other end is reachable in x, so its angle must be OK too
                    if gap_OK(our_end_xy, nearest_next_xy, max_angle, max_distance, max_distance):
                        # its OK
                        return True
                    else:
                        # other end no good
                        return False
                else:
                    # other end too far away, so OK
                    return True

            if gap_OK(our_end_xy, nearest_next_xy, max_angle, max_distance, max_distance):
                # angle and distance OK from our end to start of full
                # need to check the other end too, if our start is within max distance of back it must OK too
                if gap_OK(nearest_back_xy, our_start_xy, None, max_distance, None):
                    # other end is reachable in x, so its angle must be OK too
                    if gap_OK(nearest_back_xy, our_start_xy, max_angle, max_distance, max_distance):
                        # its OK
                        return True
                    else:
                        # other end no good
                        return False
                else:
                    # other end too far away, so OK
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
                edge lower in the image (i.e. higher y) is taken out, otherwise higher is removed,
                if a residue of less than MIN_EDGE_SAMPLES remains, that is removed too,
                if the resulting edge is empty the trim fails, an empty full_edge is acceptable
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
            trimmed_edge = Scan.Edge(edge.where, edge.type, edge.samples.copy(), edge.grey_as)
            trimmed_at = []
            for x, samples in enumerate(overlaps):
                if samples is None:
                    continue
                full_edge_y, edge_y = samples
                if edge_y > full_edge_y:
                    if direction == Scan.RISING:
                        # take out the full_edge sample - easy case
                        trimmed_full_edge[x] = None
                        trimmed_at.append(x)  # note where we changed it for clean-up later
                    else:
                        # take out the edge sample
                        if not remove_sample(x, trimmed_edge):
                            return None
                elif edge_y < full_edge_y:
                    if direction == Scan.FALLING:
                        # take out the full_edge sample - easy case
                        trimmed_full_edge[x] = None
                        trimmed_at.append(x)  # note where we changed it for clean-up later
                    else:
                        # take out the edge sample
                        if not remove_sample(x, trimmed_edge):
                            return None
                else:  # edge_y == full_edge_y:
                    # do nothing - this is benign, the merge process will just put in there what is already there
                    continue

            if len(trimmed_edge.samples) < Scan.MIN_EDGE_SAMPLES:
                # edge residue too small
                return None

            # remove small residue in trimmed_full_edge,
            # these can only be around places we trimmed the full edge
            def clean(x, dx):
                """ remove sequences that are below the minimum starting from x and going is direction dx,
                    returns True if one was found and cleaned out
                    """
                if trimmed_full_edge[x % max_x] is not None:
                    # full edge has not been tweaked here, so nothing to clean
                    return False
                residue = []
                for _ in range(max_x):
                    x = (x + dx) % max_x
                    if trimmed_full_edge[x] is None:
                        break
                    residue.append(x)
                    if len(residue) > Scan.MIN_EDGE_SAMPLES:
                        # got more than enough, so look no further
                        break
                if len(residue) < Scan.MIN_EDGE_SAMPLES:
                    # got a small residue, drop it
                    for x in residue:
                        trimmed_full_edge[x] = None
                    return True
                return False

            for x in trimmed_at:
                clean(x, -1)
                clean(x, +1)

            return trimmed_full_edge, trimmed_edge

        def smooth(edge):
            """ smooth the edge using a "simple moving average" - see https://en.wikipedia.org/wiki/Moving_average
                returns the smoothed edge
                """

            max_x = len(edge)

            # step 1 - integrate the edge (with overlap for x wrapping)
            integrated  = edge[max_x - Scan.SMOOTHING_WINDOW:]
            integrated += edge
            integrated += edge[:Scan.SMOOTHING_WINDOW]
            sum = 0
            for x in range(len(integrated)):
                sum += integrated[x]
                integrated[x] = sum

            # step 2 - generate the SMA
            sma = [0 for _ in range(max_x)]
            for x in range(max_x):
                ix = x + Scan.SMOOTHING_WINDOW  # add the intgration offset (to allow for x wrapping)
                ahead  = integrated[ix + Scan.SMOOTHING_WINDOW]
                behind = integrated[ix - Scan.SMOOTHING_WINDOW]
                sma[x] = int(round((ahead - behind) / (Scan.SMOOTHING_WINDOW * 2)))

            return sma

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
                for distance in distance_span:  # look for closest options first
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

            # smooth y gradients
            smoothed = smooth(composed)

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
                from_y = [y + (y * Scan.INNER_EDGE_GAP) for y in inner]
                rising_edges = self._edges(slices, Scan.RISING, max_y, from_y=from_y)
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
        extent = make_extent(target, clean=False, context='-warped')

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

    @staticmethod
    def drop_illegal_digits(digits):
        """ the initial digit classification takes no regard of not allowed digits,
            this is so we can differentiate junk from noise, we filter those here,
            returns the modified digits list and a count of digits dropped,
            this is public to allow test harnesses access
            """
        dropped = 0
        not_allowed = True
        while not_allowed and len(digits) > 0:
            not_allowed = False
            for choice, (digit, error) in enumerate(digits):
                coding = codec.Codec.coding(digit)
                if coding is None:
                    # we've got a not-allowed classification,
                    # if the error difference between this and some other choices is small-ish
                    # consider it noise and drop just that classification,
                    # if the error difference is large-ish it means we're looking at junk, so drop the lot
                    if choice == 0:
                        if choice < (len(digits) - 1):
                            # we're the first choice and there is a following choice, check it
                            digit2, error2 = digits[choice + 1]
                            coding2 = codec.Codec.coding(digit2)
                            if coding2 is None:
                                # got another illegal choice, drop that
                                dropped += 1
                                del digits[choice + 1]
                            else:
                                diff = max(error, error2) - min(error, error2)
                                if diff < Scan.MAX_NOT_ALLOWED_ERROR_DIFF:
                                    # its not a confident classification, so just drop this choice (as noise)
                                    dropped += 1
                                    del digits[choice]
                                else:
                                    # its confidently a not allowed digit, so drop the lot ('cos we're looking at junk)
                                    dropped += len(digits)
                                    digits = []
                        else:
                            # we're the only choice and illegal, drop it (not silent)
                            dropped += 1
                            del digits[choice]
                    else:
                        # not first choice, just drop it as a choice (silently)
                        del digits[choice]
                    not_allowed = True
                    break
        return digits, dropped

    @staticmethod
    def drop_bad_digits(digits):
        """ check the error of the given digits and drop those that are 'excessive',
            returns the modified digits list and a count of digits dropped,
            this is public to allow test harnesses access
            """
        dropped = 0
        for digit in range(len(digits) - 1, 0, -1):
            if digits[digit][1] > Scan.MAX_DIGIT_ERROR:
                # error too big to be considered as real
                dropped += 1
                del digits[digit]
        return digits, dropped

    @staticmethod
    def drop_bad_zero_digit(digits):
        """ drop a zero if it has a choice with a small-ish error,
            returns the modified digits list and a count of digits dropped,
            this is public to allow test harnesses access
            """
        dropped = 0
        if len(digits) > 1:
            # there is a choice, check if first is a 0 with a 'close' choice
            digit1, error1 = digits[0]
            if digit1 == 0:
                digit2, error2 = digits[1]  # NB: we know error2 is >= error1 (due to sort above)
                diff = error2 - error1
                if diff < Scan.MAX_ZERO_ERROR_DIFF:
                    # second choice is 'close' to first of 0, so treat 0 as dodgy and drop it
                    dropped += 1
                    del digits[0]
                    # there can only be one zero digit, so we're now done
        return digits, dropped

    @staticmethod
    def is_ambiguous(slice):
        """ test if the top choices in the given slice are ambiguous,
            this is public to allow test harnesses access
            """
        if len(slice) > 1:
            error1 = slice[0][1]
            error2 = slice[1][1]
            diff = max(error1, error2) - min(error1, error2)
            if diff < Scan.MAX_DIGIT_ERROR_DIFF:
                # its ambiguous when only a small error difference
                return True
        return False

    @staticmethod
    def show_options(options):
        """ produce a formatted string to describe the given digit options,
            this is public to allow test harnesses access
            """
        if options is None:
            return 'None'
        if len(options) > 0:
            msg = ''
            for digit, error in options:
                msg = '{}, ({}, {:.2f})'.format(msg, digit, error)
            return msg[2:]
        else:
            return '()'

    @staticmethod
    def show_bits(slice):
        """ produce a formatted string for the given slice of bits,
            this is public to allow test harnesses access
            """
        ring_width = int(round(len(slice) / codec.Codec.SPAN))
        bits = ''
        for ring in range(codec.Codec.SPAN):
            block = ''
            for dx in range(ring_width):
                x = (ring * ring_width) + dx
                if x >= len(slice):
                    block = '{}.'.format(block)
                else:
                    block = '{}{}'.format(block, slice[x])
            bits = '{} {}'.format(bits, block)
        return '[{}]'.format(bits[1:])

    def _find_all_digits(self, extent: Extent) -> ([Digit], str):
        """ find all the digits from an analysis of the given extent,
            returns a digit list and None or a partial list and a fail reason,
            the extent is also updated with the slices involved
            """

        if self.logging:
            header = 'find_all_digits:'

        buckets = extent.buckets
        max_x, max_y = buckets.size()
        reason = None

        # a 0 in these bit positions mean treat grey as black, else treat as white
        # these are AND masks, when we get a grey in the 'before', 'centre' or 'after' context in a slice
        # the result of the AND on the option being performed is used to determine what to do,
        # 0=treat as black, 1=treat as white
        GREY_BEFORE = 0  # 1  # grey option bit mask to specify how to handle grey before the first white
        GREY_CENTRE = 0  # 2  # grey option bit mask to specify how to handle grey between before and after white
        GREY_AFTER  = 0  # 4  # grey option bit mask to specify how to handle grey after the last white
        GREY_ONLY   = 0  # 8  # grey option bit mask to specify how to handle grey when there is no white

        def drop_dodgy_zero(x, option, digits, logging=False):
            nonlocal header
            if logging:
                before = self.show_options(digits)
            digits, dropped = self.drop_bad_zero_digit(digits)
            if logging and dropped > 0:
                if header is not None:
                    self._log(header)
                    header = None
                self._log('    {} (option {}): dropping {} zero choices from {} leaving {}'.
                          format(x, option, dropped, before, self.show_options(digits)))
            return dropped

        def first_good_choice(slice):
            """ get the first choice in the given slice,
                if there are two or more choices and they are ambiguous, return None,
                if the top two choices are not ambiguous return the first choice (of the 2),
                else return None,
                NB: returning None for only 1 choice is important to stop ambiguity resolutions propagating.
                    A single choice is only possible as a result of ambiguity resolution. ToDo: this is not always true!
                """
            if self.is_ambiguous(slice):
                return None
            if len(slice) > 1:
                return slice[0]
            else:
                return None

        def show_grey_option(option):
            """ option is a bit mask of 3 bits, bit 0 = before white, 1 = between, 2 = after
                a 1 in that bit means treat grey as white and a 0 means treat grey as black
                """
            if option & GREY_BEFORE == 0:
                before = '0'
            else:
                before = '1'
            if option & GREY_CENTRE == 0:
                centre = '0'
            else:
                centre = '1'
            if option & GREY_AFTER == 0:
                after = '0'
            else:
                after = '1'
            if option & GREY_ONLY == 0:
                only = '0'
            else:
                only = '1'
            return '(grey {}{}{}{})'.format(before, centre, after, only)

        # region generate likely digit choices for each x...
        inner = extent.inner
        outer = extent.outer
        slices = []
        for x in range(max_x):
            start_y = inner[x] + 1  # get past the white bullseye, so start at first black
            end_y   = outer[x]      # this is the first white after the inner black
            # region get the raw pixels and their luminance edge extremes...
            pixels = []
            has_white = 0
            has_grey = 0
            first_white = None  # location of first transition from black or grey to white
            last_white = None  # location of last transition from white to black or grey
            this_pixel = buckets.getpixel(x, start_y)  # make sure we do not see a transition on the first pixel
            for y in range(start_y, end_y):
                prev_pixel = this_pixel
                dy = y - start_y
                this_pixel = buckets.getpixel(x, y)
                pixels.append(this_pixel)
                if this_pixel != prev_pixel:
                    # got a transition
                    if prev_pixel == MAX_LUMINANCE:
                        # got a from white transition
                        last_white = dy - 1
                    if this_pixel == MAX_LUMINANCE and first_white is None:
                        # got first to white transition
                        first_white = dy
                if this_pixel == MAX_LUMINANCE:
                    has_white += 1
                elif this_pixel == MID_LUMINANCE:
                    has_grey += 1
            # adjust for leading/trailing white
            if first_white is None:
                # there was no transition to white
                if last_white is None:
                    # there is no transition from white either
                    if has_white > 0:
                        # this means its all white
                        first_white = 0
                        last_white = len(pixels) - 1
                    else:
                        # this means its all black or grey
                        # set an out of range value, so we do not have to check for None in our later loops
                        first_white = len(pixels) + 1
                        last_white = first_white
                else:
                    # there is transition from white but not to white, this means its all white from the start
                    first_white = 0
            elif last_white is None:
                # there is a transition to white but not from white, that means it ran into the end
                last_white = len(pixels) - 1
            # endregion
            # region build the options for the grey treatment...
            # the options are: grey before first white as black or white
            #                  grey after last white as black or white
            #                  grey between first and last white as black or white
            #                  grey when no white
            # we classify the pixels adjusted for these options, then pick the best,
            # this has the effect of sliding the central pulse up/down around the grey boundaries
            # there are 16 combinations of greys - before b/w * between b/w * after b/w * no white
            options = []
            for option in range(16):
                # option is a bit mask of 4 bits, bit 0 = before white, 1 = between, 2 = after, 3 = no white
                # a 1 in that bit means treat grey as white and a 0 means treat grey as black
                # NB: option 0 must be to consider *all* greys as black
                slice = []
                for y, pixel in enumerate(pixels):
                    if pixel == MID_LUMINANCE:
                        if has_white == 0:
                            if option & GREY_ONLY == 0:
                                # treat as black
                                pixel = MIN_LUMINANCE
                            else:
                                # treat as white
                                pixel = MAX_LUMINANCE
                        elif y < first_white:
                            if option & GREY_BEFORE == 0:
                                # treat before as black
                                pixel = MIN_LUMINANCE
                            else:
                                # treat before as white
                                pixel = MAX_LUMINANCE
                        elif y > last_white:
                            if option & GREY_AFTER == 0:
                                # treat after as black
                                pixel = MIN_LUMINANCE
                            else:
                                # treat after as white
                                pixel = MAX_LUMINANCE
                        else:  # if y >= first_white and y <= last_white:
                            if option & GREY_CENTRE == 0:
                                # treat between as black
                                pixel = MIN_LUMINANCE
                            else:
                                # treat between as white
                                pixel = MAX_LUMINANCE
                    if pixel == MIN_LUMINANCE:
                        slice.append(0)
                    elif pixel == MAX_LUMINANCE:
                        slice.append(1)
                    else:
                        # can't get here
                        raise Exception('Got unexpected MID_LUMINANCE')
                options.append((option, slice))
                if has_grey == 0:
                    # there are no greys, so don't bother with the rest
                    break
            # get rid of duplicates (this can happen if there are no greys before but some after, etc)
            if len(options) > 1:
                for option in range(len(options)-1, 0, -1):
                    _, this_bits = options[option]
                    for other in range(option):
                        _, other_bits = options[other]
                        if this_bits == other_bits:
                            del options[option]
                            break
            # endregion
            # region build the digits for each option...
            slice = []
            if self.logging:
                prefix = '{:3n}:'.format(x)
                disqualified = []
                illegal = []
                big_error = []
                bad_zero = []
            for option, (mask, bits) in enumerate(options):
                if not self.decoder.qualify(bits):
                    if self.logging:
                        disqualified.append(option)
                    slice.append([])  # set an empty digit list
                    continue
                raw_digits = self.decoder.classify(bits)
                # drop illegal digits
                digits, dropped = self.drop_illegal_digits(raw_digits.copy())
                if self.logging and dropped > 0:
                    illegal.append((option, raw_digits, dropped))
                # drop digits with an excessive error
                digits, dropped = self.drop_bad_digits(digits)
                if self.logging and dropped > 0:
                    big_error.append((option, raw_digits, dropped))
                # drop a zero if it has a choice with a small-ish error
                dropped = drop_dodgy_zero(x, option, digits)
                if self.logging and dropped > 0:
                    bad_zero.append((option, raw_digits, dropped))
                slice.append(digits)
            # log what happened
            if self.logging:
                if len(disqualified) > 0 or len(illegal) > 0 or len(big_error) > 0 or len(bad_zero) > 0:
                    if header is not None:
                        self._log(header)
                        header = None
                    for option in disqualified:
                        mask, bits = options[option]
                        self._log('    {}{} ignoring non-qualifying bits {}'.
                                  format(prefix, show_grey_option(mask), self.show_bits(bits)))
                        prefix = '    '
                    for option, raw_digits, dropped in illegal:
                        mask, bits = options[option]
                        self._log('    {}{} dropping {} illegal choices from {}'.
                                  format(prefix, show_grey_option(mask), dropped, self.show_options(raw_digits)))
                        prefix = '    '
                    for option, raw_digits, dropped in big_error:
                        mask, bits = options[option]
                        self._log('    {}{} dropping {} big error choices from {}'.
                                  format(prefix, show_grey_option(mask), dropped, self.show_options(raw_digits)))
                        prefix = '    '
                    for option, raw_digits, dropped in bad_zero:
                        mask, bits = options[option]
                        self._log('    {}{} dropping {} zero choices from {}'.
                                  format(prefix, show_grey_option(mask), dropped, self.show_options(raw_digits)))
                        prefix = '    '
            # endregion
            # region merge the grey options...
            # get rid of dodgy 0's, a 'dodgy' 0 is one where option 0 is black but some other is not
            # NB: option 0 is where all greys are considered to be black
            grey_as_black = slice[0]
            if len(grey_as_black) > 0 and grey_as_black[0][0] == 0:
                # it thinks its a zero, so check others
                invalid = 0
                for option in range(1, len(slice)):
                    if len(slice[option]) == 0:
                        # this means this option had no valid digits, but for the classifier to say this there
                        # must be white or grey pixels present, so the 0 is in doubt, if all other options are
                        # also invalid, we drop the 0, so just count them here
                        invalid += 1
                        continue
                    if slice[option][0][0] != 0:
                        # some other option thinks it is not 0, so its a dodgy 0, drop all in that slice
                        if self.logging:
                            if header is not None:
                                self._log(header)
                                header = None
                            self._log('    {}{} dropping zero slice: {} in favour of {}'.
                                      format(prefix, show_grey_option(option), self.show_options(slice[0]), self.show_options(slice[option])))
                            prefix = '    '
                        del slice[0]
                        break
                if invalid > 0 and invalid == (len(slice) - 1):
                    # all other options are invalid, so do not trust the initial 0, drop it
                    if self.logging:
                        if header is not None:
                            self._log(header)
                            header = None
                        self._log('    {}{} dropping zero: {} all its other {} options are invalid'.
                                  format(prefix, show_grey_option(0), self.show_options(slice[0]), invalid))
                        prefix = '    '
                    del slice[0]
            # join all the (remaining) slices so we can find the best choice for each digit
            choices = []
            for option in slice:
                choices += option
            choices.sort(key=lambda d: (d[0], d[1]))  # put like digits together in least error order
            for choice in range(len(choices) - 1, 0, -1):
                if choices[choice][0] == choices[choice-1][0]:
                    # duplicate digit, we're on the worst one, so dump that
                    del choices[choice]
            choices.sort(key=lambda d: d[1])  # put merged list into least error order
            drop_dodgy_zero(x, 0, choices, self.logging)  # JIC some percolated up
            # endregion
            slices.append(choices)
        # region check for and resolve ambiguous choices...
        # ambiguity can lead to false positives, which we want to avoid,
        # an ambiguous choice is when a digit has (nearly) equal error choices,
        # we try to find a choice that matches one of its non-ambiguous neighbours,
        # if found, resolve to that, otherwise drop the digit
        for x, slice in enumerate(slices):
            best_choice = first_good_choice(slice)
            if best_choice is None:
                # got no, or an ambiguous, choice, check its neighbours
                left_x = (x - 1) % max_x
                right_x = (x + 1) % max_x
                left_choice = first_good_choice(slices[left_x])
                right_choice = first_good_choice(slices[right_x])
                if left_choice is None and right_choice is None:
                    # both neighbours are ambiguous too, so just drop this one
                    if len(slice) > 0:
                        # there is something to drop
                        if self.logging:
                            if header is not None:
                                self._log(header)
                                header = None
                            self._log('    {}: dropping ambiguous choices: {}'.format(x, self.show_options(slices[x])))
                        slices[x] = []
                    continue
                # there is scope to resolve this ambiguity
                if self.logging:
                    if header is not None:
                        self._log(header)
                        header = None
                # if only 1 choice, inherit the other (that makes them the same in comparisons below)
                if left_choice is None:
                    left_choice = right_choice
                    left_x = right_x
                if right_choice is None:
                    right_choice = left_choice
                    right_x = left_x
                # choose best neighbour
                left_digit, left_error = left_choice
                right_digit, right_error = right_choice
                if len(slice) < 2:
                    # we've got no choice, so just inherit our best neighbour
                    if left_error < right_error:
                        if self.logging:
                            self._log('    {}: resolving ambiguity of {} with neighbour {}: {}'.
                                      format(x, self.show_options(slices[x]), left_x, self.show_options(slices[left_x])))
                        slices[x] = [left_choice]
                    else:
                        if self.logging:
                            self._log('    {}: resolving ambiguity of {} with neighbour {}: {}'.
                                      format(x, self.show_options(slices[x]), right_x, self.show_options(slices[right_x])))
                        slices[x] = [right_choice]
                    continue
                # we have choices to resolve
                digit1, error1 = slice[0]
                digit2, error2 = slice[1]
                if left_error < right_error:
                    # try left neighbour first
                    if digit1 == left_digit or digit2 == left_digit:
                        # got a match use it
                        if self.logging:
                            self._log('    {}: resolving ambiguity of {} with neighbour {}: {}'.
                                      format(x, self.show_options(slices[x]), left_x, self.show_options(slices[left_x])))
                        slice[0] = left_choice
                        del slice[1]
                        continue
                # try right neighbour
                if digit1 == right_digit or digit2 == right_digit:
                    # got a match use it
                    if self.logging:
                        self._log('    {}: resolving ambiguity of {} with neighbour {}: {}'.
                                  format(x, self.show_options(slices[x]), right_x, self.show_options(slices[right_x])))
                    slice[0] = right_choice
                    del slice[1]
                    continue
                # neither side matches, just drop it
                if self.logging:
                    self._log('    {}: dropping ambiguous choices: {}'.format(x, self.show_options(slices[x])))
                slices[x] = []
                continue
        # endregion
        # region resolve singletons...
        # a singleton is a single slice bordered both sides by some valid digit
        # these are considered to be noise and inherit their best neighbour
        for x, this_slice in enumerate(slices):
            left_x = (x - 1) % max_x
            right_x = (x + 1) % max_x
            left_slice = slices[left_x]
            right_slice = slices[right_x]
            if len(left_slice) == 0 or len(right_slice) == 0:
                # we have not got both neighbours
                continue
            left_digit, left_error = left_slice[0]
            right_digit, right_error = right_slice[0]
            if len(this_slice) == 0:
                # inherit best neighbour
                pass
            else:
                this_digit, this_error = this_slice[0]
                if left_digit == this_digit or right_digit == this_digit:
                    # not a potential singleton
                    continue
            # inherit best neighbour
            # do not inherit sync unless both sides are sync
            if left_digit == codec.Codec.SYNC_DIGIT and right_digit == codec.Codec.SYNC_DIGIT:
                # both sides are sync, so allow inherit
                pass
            elif left_digit == codec.Codec.SYNC_DIGIT:
                # do not allow inherit of left
                left_slice = right_slice
                left_x = right_x
            else:  # right_digit == codec.Codec.SYNC_DIGIT:
                # do not allow inherit of right
                right_slice = left_slice
                right_x = left_x
            if self.logging:
                if header is not None:
                    self._log(header)
                    header = None
            if left_error < right_error:
                if self.logging:
                    self._log('    {}: resolve singleton {} to {}: {}'.
                              format(x, self.show_options(this_slice), left_x, self.show_options(left_slice)))
                slices[x] = left_slice
            else:
                if self.logging:
                    self._log('    {}: resolve singleton {} to {}: {}'.
                              format(x, self.show_options(this_slice), right_x, self.show_options(right_slice)))
                slices[x] = right_slice
            continue
        # endregion
        if self.logging:
            if header is not None:
                self._log(header)
                header = None
            self._log('    initial slices (fail reason {}):'.format(reason))
            for x, options in enumerate(slices):
                self._log('        {}: options={}'.format(x, self.show_options(options)))
        # endregion
        # region build digit list...
        digits = []
        last_digit = None
        for x, options in enumerate(slices):
            if len(options) == 0:
                # this is junk - treat like an unknown digit
                best_digit = None
                best_error = 1.0
            else:
                # we only look at the best choice and only iff the next best has a worse error
                best_digit, best_error = options[0]
            if last_digit is None:
                # first digit
                last_digit = Scan.Digit(best_digit, best_error, x, 1)
            elif best_digit == last_digit.digit:
                # continue with this digit
                last_digit.error += best_error  # accumulate error
                last_digit.samples += 1         # and sample count
            else:
                # save last digit
                last_digit.error /= last_digit.samples  # set average error
                digits.append(last_digit)
                # start a new digit
                last_digit = Scan.Digit(best_digit, best_error, x, 1)
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

        # save slices in the extent for others to use
        extent.slices = slices

        if self.logging:
            if header is not None:
                self._log(header)
                header = None
            self._log('    {} digits (fail reason {}):'.format(len(digits), reason))
            for item, digit in enumerate(digits):
                self._log('        {}: {}'.format(item, digit))

        if self.save_images:
            plot = self.transform.copy(buckets)
            ones = []
            zeroes = []
            for x, options in enumerate(slices):
                if len(options) == 0:
                    continue
                start_y = inner[x] + 1
                end_y = outer[x]
                slice = self.decoder.make_slice(options[0][0], end_y - start_y)
                if slice is None:
                    continue
                last_bit = None
                for dy, bit in enumerate(slice):
                    if bit is None:
                        continue
                    if last_bit is None:
                        last_bit = (dy, bit)
                    elif bit != last_bit[1]:
                        # end of a run
                        if last_bit[1] == 0:
                            zeroes.append((x, last_bit[0] + start_y, x, start_y + dy - 1))
                        else:
                            ones.append((x, last_bit[0] + start_y, x, start_y + dy - 1))
                        last_bit = (dy, bit)
                if last_bit[1] == 0:
                    zeroes.append((x, last_bit[0] + start_y, x, end_y - 1))
                else:
                    ones.append((x, last_bit[0] + start_y, x, end_y - 1))
            plot = self._draw_lines(plot, ones, colour=Scan.RED)
            plot = self._draw_lines(plot, zeroes, colour=Scan.GREEN)
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

        buckets = extent.buckets
        max_x, max_y = buckets.size()
        nominal_digit_width = (max_x / Scan.NUM_SEGMENTS)
        max_digit_width = nominal_digit_width * Scan.MAX_DIGIT_WIDTH
        min_digit_width = nominal_digit_width * Scan.MIN_DIGIT_WIDTH
        min_splittable_digit_width = nominal_digit_width * Scan.MIN_SPLITTABLE_DIGIT_WIDTH
        min_droppable_digit_width = nominal_digit_width * Scan.MIN_DROPPABLE_WIDTH
        reason = None  # this is set to a reason mnemonic if we fail

        if self.logging:
            header = 'find_best_digits: digit width: nominal={:.2f}, limits={:.2f}..{:.2f}, ' \
                     'min splittable: {:.2f}, droppable: {:.2f}'.\
                     format(nominal_digit_width, min_digit_width, max_digit_width,
                            min_splittable_digit_width, min_droppable_digit_width)

        def find_best_2nd_choice(slices, slice_start, slice_end):
            """ find the best 2nd choice in the given slices,
                return the digit, how many there are and the average error,
                the return info is sufficient to create a Scan.Digit
                """
            second_choice = [[0, 0] for _ in range(Scan.DIGIT_BASE)]
            for x in range(slice_start, slice_end):
                options = slices[x % len(slices)]
                if len(options) > 1:
                    # there is a 2nd choice
                    digit, error = options[1]
                    second_choice[digit][0] += 1
                    second_choice[digit][1] += error
                elif len(options) > 0:
                    # no second choice, use the first
                    digit, error = options[0]
                else:
                    # nothing here at all
                    continue
                second_choice[digit][0] += 1
                second_choice[digit][1] += error
            best_digit = None
            best_count = 0
            best_error = 0
            for digit, (count, error) in enumerate(second_choice):
                if count > best_count:
                    best_digit = digit
                    best_count = count
                    best_error = error
            if best_count > 0:
                best_error /= best_count
            return best_digit, best_count, best_error

        def shrink_digit(slices, digit, start, samples) -> Scan.Digit:
            """ shrink the indicated digit to the start and size given,
                this involves moving the start, updating the samples and adjusting the error,
                the revised digit is returned
                """
            # calculate the revised error
            error = 0
            for x in range(start, start + samples):
                options = slices[x % len(slices)]
                if len(options) == 0:
                    # treat nothing as a worst case error
                    error += 1.0
                else:
                    error += options[0][1]
            if samples > 0:
                error /= samples
            # create a new digit
            new_digit = Scan.Digit(digit.digit, error, start % len(slices), samples)
            return new_digit

        # translate digits from [Digit] to [[Digit]] so we can mess with it and not change indices,
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
            # not enough sync digits - that's a show-stopper
            reason = 'too few syncs'
        else:
            # if too many sync digits - dump the smallest droppable sync digit with the biggest error
            while len(copies) > Scan.COPIES:
                smallest_x = None  # index into copy of the smallest sync digit
                for x in range(len(copies)):
                    xx = copies[x][0]
                    if digits[xx].samples >= min_droppable_digit_width:
                        # too big to drop
                        continue
                    if smallest_x is None:
                        smallest_x = x
                        continue
                    smallest_xx = copies[smallest_x][0]
                    if digits[xx].samples < digits[smallest_xx].samples:
                        # less samples
                        smallest_x = x
                    elif digits[xx].samples == digits[smallest_xx].samples:
                        if digits[xx].error > digits[smallest_xx].error:
                            # same samples but bigger error
                            smallest_x = x
                if smallest_x is None:
                    # nothing small enough to drop - that's a show stopper
                    reason = 'too many syncs'
                    break
                smallest_xx = copies[smallest_x][0]
                if self.logging:
                    if header is not None:
                        self._log(header)
                        header = None
                    self._log('    {}: dropping excess sync {}'.
                              format(smallest_xx, digits[smallest_xx]))
                digits_list[smallest_xx] = None
                del copies[smallest_x]
        if reason is None:
            # find the actual digits between the syncs
            for copy_num, copy in enumerate(copies):
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
                            biggest_digit = digits[copy[biggest_x]]
                            if biggest_digit.samples < min_splittable_digit_width:
                                # too small to split:
                                biggest_x = None
                            continue
                        biggest_digit = digits[copy[biggest_x]]
                        if digits[xx].samples > biggest_digit.samples:
                            # found a bigger candidate
                            biggest_x = x
                        elif digits[xx].samples == biggest_digit.samples:
                            if digits[xx].error > biggest_digit.error:
                                # find a same size candidate with a bigger error
                                biggest_x = x
                    if biggest_x is None:
                        # everything splittable has been split and still not enough - this is a show stopper
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
                    # too many digits - drop the smallest with the biggest error that is not a sync digit
                    smallest_x = None
                    for x in range(1, len(copy)):  # never consider the initial sync digit
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
                            smallest_digit = digits[copy[smallest_x]]
                            if smallest_digit.samples >= min_droppable_digit_width:
                                # too big to drop
                                smallest_x = None
                            continue
                        smallest_digit = digits[copy[smallest_x]]
                        if digits[xx].samples < smallest_digit.samples:
                            # found a smaller candidate
                            smallest_x = x
                        elif digits[xx].samples == smallest_digit.samples:
                            if digits[xx].error > smallest_digit.error:
                                # found a similar size candidate with a bigger error
                                smallest_x = x
                    if smallest_x is None:
                        # nothing (left) small enough to drop, that's a show stopper
                        reason = 'too many digits'
                        break
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
                    if self.logging:
                        continue  # carry on to check other copies so logs are more intelligible
                    else:
                        break  # no point looking at other copies

        # build the final digit list
        best_digits = []
        for digits in digits_list:
            if digits is None:
                # this has been deleted
                continue
            for digit in digits:
                best_digits.append(digit)

        if reason is None:
            # check we're not left with digits that are too small or too big
            for digit in best_digits:
                if digit.samples > max_digit_width:
                    # too big
                    reason = 'digit too big'
                    break
                elif digit.samples < min_digit_width:
                    if self.decoder.is_sync_digit(digit.digit):
                        # we tolerate small syncs
                        pass
                    else:
                        # too small
                        reason = 'digit too small'
                        break

        if self.logging:
            if header is not None:
                self._log(header)
                header = None
            self._log('    {} digits (fail reason: {}):'.format(len(best_digits), reason))
            for x, digit in enumerate(best_digits):
                self._log('        {}: {}'.format(x, digit))

        if self.save_images:
            buckets = extent.buckets
            max_x, _ = buckets.size()
            plot = self.transform.copy(buckets)
            ones = []
            zeroes = []
            nones = []
            for digit in best_digits:
                for x in range(digit.start, digit.start + digit.samples):
                    x %= max_x
                    start_y = extent.inner[x] + 1
                    end_y = extent.outer[x]
                    digit_slice = self.decoder.make_slice(digit.digit, end_y - start_y)
                    if digit_slice is None:
                        # this is a 'None' digit - draw those as blue
                        nones.append((x, start_y, x, end_y - 1))
                        continue
                    last_bit = None
                    for dy, bit in enumerate(digit_slice):
                        if bit is None:
                            continue
                        if last_bit is None:
                            last_bit = (dy, bit)
                        elif bit != last_bit[1]:
                            # end of a run
                            if last_bit[1] == 0:
                                zeroes.append((x, last_bit[0] + start_y, x, start_y + dy - 1))
                            else:
                                ones.append((x, last_bit[0] + start_y, x, start_y + dy - 1))
                            last_bit = (dy, bit)
                    if last_bit[1] == 0:
                        zeroes.append((x, last_bit[0] + start_y, x, end_y - 1))
                    else:
                        ones.append((x, last_bit[0] + start_y, x, end_y - 1))
            plot = self._draw_lines(plot, nones, colour=Scan.BLUE)
            plot = self._draw_lines(plot, ones, colour=Scan.RED)
            plot = self._draw_lines(plot, zeroes, colour=Scan.GREEN)
            self._unload(plot, '06-digits')
            self._log('find_best_digits: 06-digits: green==zeroes, red==ones, blue==nones, black/white==ignored')

        return best_digits, reason

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
            self._log('decode_digits: num:{}, code:{}, doubt:{}, error:{:.3f} from: {}'.
                      format(number, code, doubt, error, msg[2:]))
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

    # endregion
