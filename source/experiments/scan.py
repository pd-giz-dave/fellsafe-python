
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
        result += ',' + nstr(pt)
    return open + result[1:] + close


def wrapped_gap(x1, x2, limit_x):
    """ given two x co-ords that may be wrapped return the gap between the two,
        a legitimate gap must be less than half the limit,
        the returned gap may be +ve or -ve and represents the gap from x1 to x2,
        i.e. (x1 + gap) % limit_x = x2
        """
    dx = x1 - x2
    if dx < 0:
        if 0 - dx > (limit_x / 2):
            # its wrapped
            dx = 0 - ((x1 + limit_x) - x2)
    elif dx > 0:
        if dx > (limit_x / 2):
            # its wrapped
            dx = 0 - (x1 - (x2 + limit_x))
    return dx


class Scan:
    """ this class provides functions to scan an image and extract any codes in it,
        algorithm summary:
            1. blob detect our bullseye
            2. warp polar around its centre to create a cartesian co-ordinate rectangle
            3. edge detect all the ring and bit boundaries
            4. extract the bits for each ring segment
            5. decode those bits
        the algorithm has been developed more by experimentation than theory!
        """

    # region Constants...
    # our target shape
    NUM_RINGS = ring.Ring.NUM_RINGS  # total number of rings in the whole code (ring==cell in height)
    NUM_DATA_RINGS = codec.Codec.RINGS_PER_DIGIT  # how many data rings in our codes
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
    # ToDo: de-tune blob area and radius if new scheme cannot get valid detections from ...-distant.jpg
    MIN_BLOB_AREA = 8  # min area of a blob we want (in pixels) (default 9)
    MIN_BLOB_RADIUS = 2  # min radius of a blob we want (in pixels) (default 2.5)
    BLOB_RADIUS_STRETCH = 1.1  # how much to stretch blob radius to ensure always cover everything when projecting
    MIN_CONTRAST = 0.5  # minimum luminance variation of a valid blob projection relative to the mean luminance
    THRESHOLD_SIZE = 2.5  # the fraction of the projected image size to use as the integration area when binarizing
    THRESHOLD_BLACK = 2  # the % below the average luminance in a projected image that is considered to be black
    THRESHOLD_WHITE = 3  # the % above the average luminance in a projected image that is considered to be white
    MIN_EDGE_SAMPLES = 2  # minimum samples in an edge to be considered a valid edge
    MAX_NEIGHBOUR_ANGLE_INNER = 0.4  # ~=22 degrees, tan of the max acceptable angle when joining inner edge fragments
    MAX_NEIGHBOUR_ANGLE_OUTER = 0.6  # ~=31 degrees, tan of the max acceptable angle when joining outer edge fragments
    MAX_NEIGHBOUR_HEIGHT_GAP = 1  # max x or y jump allowed when following an edge
    MAX_NEIGHBOUR_HEIGHT_GAP_SQUARED = MAX_NEIGHBOUR_HEIGHT_GAP * MAX_NEIGHBOUR_HEIGHT_GAP
    MAX_NEIGHBOUR_LENGTH_JUMP = 10  # max x jump, in pixels, between edge fragments when joining
    MAX_NEIGHBOUR_HEIGHT_JUMP = 3  # max y jump, in pixels, between edge fragments when joining
    MAX_NEIGHBOUR_OVERLAP = 4  # max edge overlap, in pixels, between edge fragments when joining
    MAX_EDGE_GAP_SIZE = 3 / NUM_SEGMENTS  # max gap tolerated between edge fragments (as fraction of image width)
    MAX_EDGE_HEIGHT_JUMP = 2  # max jump in y, in pixels, along an edge before smoothing is triggered
    INNER_GUARD = 2  # minimum pixel width of inner black ring
    OUTER_GUARD = 2  # minimum pixel width of outer black ring
    MIN_PULSE_HEAD = 3  # minimum pixels for a valid pulse head period, pulse ignored if less than this
    MIN_SEGMENT_EDGE_DIFFERENCE = 3  # min white pixel difference (in y) between slices for a segment edge
    MIN_SEGMENT_SAMPLES = 3  # minimum samples (in x) in a valid segment, segments of this or less are dropped,
                             # this must be >2 to ensure there is a lead area when first and last slice are dropped
    MAX_SEGMENT_WIDTH = 3.0  # maximum samples in a digit relative to the nominal segment size
    STRONG_ZERO_LIMIT = 0.8  # width of a strong zero digit relative to the width of the widest one
    ZERO_WHITE_THRESHOLD = MIN_PULSE_HEAD - 1  # maximum white pixels tolerated in a 'zero'
    ZERO_GREY_THRESHOLD = ZERO_WHITE_THRESHOLD  # maximum grey pixels tolerated in a 'zero'
    LEAD_GRAY_TO_HEAD = 0.2  # what fraction of lead grey pixels to assign to the head (the rest is given to lead)
    TAIL_GRAY_TO_HEAD = 0.3  # what fraction of tail grey pixels to assign to the head (the rest is given to tail)
    MIN_ZERO_GAP = DIGITS_PER_NUM * MIN_SEGMENT_SAMPLES  # minimum samples between consecutive zeroes
    MAX_ZERO_GAP = 1.7  # maximum samples between consecutive zeroes as a ratio of a copy width within the image
    MIN_SEGMENT_WIDTH = 0.4  # minimum width of a digit segment as a ratio of the nominal width for the group
    PULSE_ERROR_RANGE = 99  # pulse component error range 0..PULSE_ERROR_RANGE
    MAX_CHOICE_ERROR_DIFF = 3  # if a bit choice error is more than this worse than the best, chuck the choice
    MAX_BIT_CHOICES = 1024*8  # max bit choices to allow/explore when decoding bits to the associated number
    CHOICE_DOUBT_DIFF_LIMIT = 1  # choices with a digit doubt difference of less than this are ambiguous
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

    # region Pulse classifications...
    DIGITS = codec.Codec.ENCODING
    SPAN = codec.Codec.SPAN  # total span of the target rings
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

    # region Structures...
    class Step:
        """ a Step is a description of a luminance change
            (orientation of vertical or horizontal is defined by the context)
            """

        def __init__(self, where, type, pixel):
            self.where = where  # the y co-ord of the step
            self.type = type  # the step type, rising or falling
            self.pixel = pixel  # the pixel level falling from or rising to

        def __str__(self):
            if self.type == Scan.RISING:
                edge_dir = 'to'
            else:
                edge_dir = 'from'
            return '({} at {} {} {})'.format(self.type, self.where, edge_dir, self.pixel)

    class Edge:
        """ an Edge is a sequence of joined steps, either horizontally or vertically """

        def __init__(self, where, type, samples=None, direction=None):
            self.where = where  # the x or y co-ord of the start of this edge
            self.type = type  # the type of the edge, falling or rising
            self.samples = samples  # the list of connected y's or x's making up this edge
            if direction is None:
                self.direction = Scan.HORIZONTAL
            else:
                self.direction = direction  # the direction of the edge

        def __str__(self):
            if self.samples is None:
                samples = ''
                y = '?'
            elif len(self.samples) > 10:
                samples = ': {}..{}'.format(self.samples[:5], self.samples[-5:])
                y = self.samples[0]
            else:
                samples = ': {}'.format(self.samples)
                y = self.samples[0]
            if self.direction == Scan.HORIZONTAL:
                # x,y are correct
                x = self.where
            else:
                # x,y are reversed
                x = y
                y = self.where
            return '(({}): {} at {},{} for {}{})'.format(self.direction, self.type, x, y, len(self.samples), samples)

    class Extent:
        """ an Extent is the inner edge co-ordinates of a projected image along with
            the horizontal and vertical edge fragments it was built from """

        def __init__(self, h_edges, v_edges, inner, inner_fail=None):
            self.h_edges: [Scan.Edge] = h_edges  # the horizontal edge fragments of the image or None
            self.v_edges: [Scan.Edge] = v_edges  # the vertical edge fragments of the image or None
            self.inner: [int] = inner  # list of y co-ords for the inner edge
            self.inner_fail = inner_fail  # reason if failed to find inner edge or None if OK

    class Pulse:
        """ a Pulse describes the pixels in a radius """

        def __init__(self, start=None, stop=None, lead=0, head=0, begin=None, end=None):
            self.start = start  # y co-ord where the pulse starts in the radius
            self.stop = stop  # y co-ord where the pulse ends in the radius
            self.lead = lead  # length of the lead '0' pixels (including the inner black)
            self.head = head  # length of the '1' pixels
            self.begin = begin  # x-coord of start of the pulse samples
            self.end = end  # x-coord of end of the pulse samples
            self.bits = []  # list of bit candidates and their error in least error order

        def ratio(self):
            """ return the back/white ratio that represents this pulse """
            if self.lead == 0 and self.head == 0:
                # this is nothing
                return None
            elif self.lead == 0:
                # this is white all the way down
                return 1 / self.head
            elif self.head == 0:
                # this is black all the way down
                return self.lead / 1
            else:
                return self.lead / self.head

        def __str__(self):
            bits = ''
            if len(self.bits) > 0:
                for bit in self.bits:
                    bits = '{}, {}'.format(bits, bit)
                bits = ', bits={}'.format(bits[2:])
            return '(x={}..{}, y={}..{}, lead={}, head={}, ratio={})'.\
                   format(nstr(self.begin), nstr(self.end), nstr(self.start), nstr(self.stop),
                          nstr(self.lead), nstr(self.head), nstr(self.ratio()), bits)

    class Bits:
        """ this encapsulates a bit sequence for a digit and its error """

        def __init__(self, bits, error, actual=None, ideal=None):
            self.bits = bits  # the bits across the data rings
            self.error = error  # the error actual vis ideal
            self.actual = actual  # actual pulse head, top, tail measured
            self.ideal = ideal  # the ideal head, top, tail for these bits

        def format(self, short=False):
            if short or self.actual is None:
                actual = ''
            else:
                actual = ' = actual:{}'.format(vstr(self.actual))
            if short or self.ideal is None:
                ideal = ''
            else:
                ideal = ', ideal:{}'.format(vstr(self.ideal))
            return '({}, {:.2f}{}{})'.format(self.bits, self.error, actual, ideal)

        def __str__(self):
            return self.format()

    class Segment:
        """ a Segment describes the decoded data rings (for diagnostic purposes) """

        def __init__(self, start, bits, size):
            self.start = start  # the start x of this sequence
            self.bits = bits  # the bit pattern for this sequence
            self.size = size  # how many pixels in it

        def __str__(self):
            return '(at {} bits={}, size={})'.format(self.start, self.bits, self.size)

    class Result:
        """ a result is the result of a number decode and its associated error/confidence level """

        def __init__(self, number, digit_doubt, digits, bit_doubt, bit_error, choice, count=1):
            self.number = number  # the code found
            self.digit_doubt = digit_doubt  # sum of error digits (i.e. where not all three copies agree)
            self.digits = digits  # the digit pattern used for the result
            self.bit_doubt = bit_doubt  # sum of choices across bit segments when more than one choice
            self.bit_error = bit_error  # sum of errors across bit segments
            self.choice = choice  # the segments choice this result was decoded from (diag only)
            self.count = count  # how many results with the same number found
            self.doubt_level = None  # (see self.doubt())
            self.max_bit_error = None  # (see self.doubt())

        def doubt(self, max_bit_error=None):
            """ calculate the doubt for this number, more doubt means less confidence,
                the doubt is an amalgamation of digit_doubt, bit_doubt and bit_error,
                the bit_error is scaled by max_bit_error to give a relative ratio of 0..100,
                if a max_bit_error is given it is used to (re-)calculate the doubt, otherwise
                the previous doubt is returned, the max_bit_error must be >= any result bit_error
                the doubt returned is a float consisting of 3 parts:
                    integral part is digit_doubt (i.e. mis-match between the 3 copies)
                    first two decimal places is the bit_doubt (i.e. choices in pulse interpretation)
                    second two decimal places is the relative bit error (i.e. error of actual pulse to ideal one)
                """
            if max_bit_error is not None:
                self.max_bit_error = max_bit_error
                if self.max_bit_error == 0:
                    # this means there are no errors
                    bit_error_ratio = 0
                else:
                    bit_error_ratio = self.bit_error / self.max_bit_error  # in range 0..1 (0=good, 1=crap)
                self.doubt_level = (min(self.digit_doubt, 99)) + \
                                   (min(self.bit_doubt, 99) / 100) + \
                                   (min(bit_error_ratio, 0.99) / 100)
            return self.doubt_level

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

        def __init__(self, number, doubt, centre_x, centre_y, target_size, blob_size, digits):
            self.centre_x = centre_x  # where it is in the original image
            self.centre_y = centre_y  # ..
            self.blob_size = blob_size  # the size of the blob as detected by opencv
            self.number = number  # the code number we found
            self.doubt = doubt  # how many bit errors there are in it
            self.target_size = target_size  # the size of the target in the original image (used for relative distance)
            self.digits = digits  # the digits as decoded by the codec (shows where the bit errors are)
    # endregion

    def __init__(self, codec, frame, transform, cells=(8, 4), video_mode=VIDEO_FHD, debug=DEBUG_NONE, log=None):
        """ codec is the codec instance defining the code structure,
            frame is the frame instance containing the image to be scanned,
            transform is the class implmenting various image transforms (mostly for diagnostic purposes),
            cells is the angular/radial resolution to use,
            video_mode is the maximum resolution to work at, the image is downsized to this if required
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
        self.cells = cells  # (segment length x ring height) size of segment cells to use when decoding
        self.video_mode = video_mode  # actually the downsized image height
        self.original = frame
        self.decoder = codec  # class to decode what we find

        # set warped image width/height
        self.angle_steps = int(round(Scan.NUM_SEGMENTS * max(self.cells[0], Scan.MIN_PIXELS_PER_CELL)))
        self.radial_steps = int(round(Scan.NUM_RINGS * max(self.cells[1], Scan.MIN_PIXELS_PER_RING)))

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
        max_circumference = min(2 * math.pi * max_radius, 3600)  # good enough for at least 0.1 degree resolution
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

        if self.show_log:
            logger = contours.Logger('_blobs')
        else:
            logger = None
        params = contours.Targets()
        params.min_area = Scan.MIN_BLOB_AREA
        params.min_radius = Scan.MIN_BLOB_RADIUS
        blobs, binary = contours.get_targets(self.image.buffer, params=params, logger=logger)
        self.binary = self.image.instance()
        self.binary.set(binary)

        blobs.sort(key=lambda e: (e[0], e[1]))  # just so the processing order is deterministic (helps debugging)

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

    def _project(self, centre_x, centre_y, blob_size) -> (frame.Frame, float):
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
                    +------+-----+
                    |  P   |   1  |
                    |   ...|...   |
                    |   .  |  .   |
                    +------+------+
                    |   .  |  .   |
                    |   ...|...   |
                    |  3   |   2  |
                    +------+------+
                """
            cX: float = centre_x + x
            cY: float = centre_y + y
            xL: int = int(cX)
            yL: int = int(cY)
            xH: int = xL + 1
            yH: int = yL + 1
            xLyL = self.image.getpixel(xL, yL)
            xLyH = self.image.getpixel(xL, yH)
            xHyL = self.image.getpixel(xH, yL)
            xHyH = self.image.getpixel(xH, yH)
            if xLyL is None:
                xLyL = MIN_LUMINANCE
            if xLyH is None:
                xLyH = MIN_LUMINANCE
            if xHyL is None:
                xHyL = MIN_LUMINANCE
            if xHyH is None:
                xHyH = MIN_LUMINANCE
            pixel: float = (xH - cX) * (yH - cY) * xLyL
            pixel += (cX - xL) * (yH - cY) * xHyL
            pixel += (xH - cX) * (cY - yL) * xLyH
            pixel += (cX - xL) * (cY - yL) * xHyH
            return int(round(pixel))

        limit_radius = self._radius(centre_x, centre_y, blob_size)

        # for detecting luminance variation for filtering purposes
        min_level = MAX_LUMINANCE
        max_level = MIN_LUMINANCE
        avg_level: float = 0
        samples = 0
        # make a new black image to build the projection in
        angle_delta = 360 / self.angle_steps
        code = self.original.instance().new(self.angle_steps, limit_radius, MIN_LUMINANCE)
        for radius in range(limit_radius):
            for angle in range(self.angle_steps):
                degrees = angle * angle_delta
                x, y = self.angle_xy(degrees, radius)
                if x is not None:
                    c = get_pixel(x, y)  # centre_x/y applied in here
                    if c > MIN_LUMINANCE:
                        code.putpixel(angle, radius, c)
                        avg_level += c
                        samples += 1
                        if c > max_level:
                            max_level = c
                        if c < min_level:
                            min_level = c
        avg_level /= samples

        # chuck out targets that do not have enough black/white contrast
        contrast = (max_level - min_level) / avg_level
        if contrast < Scan.MIN_CONTRAST:
            if self.logging:
                self._log('project: dropping blob at {:.1f} {:.1f} - contrast {:.2f} below minimum ({:.2f})'.
                          format(centre_x, centre_y, contrast, Scan.MIN_CONTRAST))
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
            start_x = max(int(centre_x - limit_radius), 0)
            end_x = min(int(centre_x + limit_radius), max_x)
            start_y = max(int(centre_y - limit_radius), 0)
            end_y = min(int(centre_y + limit_radius), max_y)
            blob = self.transform.crop(self.image, start_x, start_y, end_x, end_y)
            # draw the detected blob in red
            k = (limit_radius, limit_radius, blob_size)
            blob = self.transform.label(blob, k, Scan.RED)
            self._unload(blob, '01-target', centre_x, centre_y)
            # draw the corresponding projected image
            self._unload(code, '02-projected')

        return code, stretch_factor

    def _binarize(self, target: frame.Frame, s: float=2, black: float=-1, white: float=None, clean=True, suffix='') -> frame.Frame:
        """ binarize the given projected image,
            s is the fraction of the image size to use as the integration area (square) in pixels,
            black is the % below the average that is considered to be the black/grey boundary,
            white is the % above the average that is considered to be the grey/white boundary,
            white of None means same as black and will yield a binary image,
            also, iff clean=True, 'tidy' it by changing pixel sequences of BWB or WBW sequences to BBB or WWW
            """

        max_x, max_y = target.size()

        def make_binary(image: frame.Frame, s: float=8, black: float=15, white: float=None) -> frame.Frame:
            """ given a greyscale image return a binary image using an adaptive threshold.
                s, black, white - see description of overall method
                See the adaptive-threshold-algorithm.pdf paper for algorithm details.
                """

            # the image being thresholded wraps in x, to allow for this when binarizing we extend
            # the image by a half height to the left and right, then remove it from the binary

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
            thresholded = contours.make_binary(extended, s, black, white)

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

        buckets = make_binary(target, s, black, white)

        if clean:
            # clean the pixels - BWB or WBW sequences are changed to BBB or WWW
            # pixels wrap in the x direction but not in the y direction
            passes = 0
            total_to_black_changes = 0
            total_to_white_changes = 0
            pass_changes = 1
            while pass_changes > 0:
                passes += 1
                pass_changes = 0
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
                                if this == MIN_LUMINANCE:
                                    total_to_white_changes += 1
                                else:
                                    total_to_black_changes += 1
                        elif above == below:
                            # only look for vertical when there is no horizontal candidate, else it can oscillate
                            if this != above:
                                # got a vertical loner
                                # this condition is lower priority than above
                                buckets.putpixel(x, y, above)
                                pass_changes += 1
                                if this == MIN_LUMINANCE:
                                    total_to_white_changes += 1
                                else:
                                    total_to_black_changes += 1
            if self.logging:
                self._log('binarize: cleaned lone pixels in {} passes, changing {} pixels to white and {} to black'.
                          format(passes, total_to_white_changes, total_to_black_changes))
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
                    slices[x].append(Scan.Step(y-1, Scan.FALLING, last_pixel))
                    transitions += 1
                elif pixel > last_pixel:
                    # rising step
                    slices[x].append(Scan.Step(y, Scan.RISING, pixel))
                    transitions += 1
                last_pixel = pixel
            if transitions == 0:
                # this probably means a big pulse has merged with the inner and the outer edge,
                if last_pixel == MAX_LUMINANCE:
                    # its all white - not possible?
                    # create a rising step at 0 and a falling step at max_y
                    slices[x].append(Scan.Step(0, Scan.RISING, MAX_LUMINANCE))
                    slices[x].append(Scan.Step(max_y-1, Scan.FALLING, MAX_LUMINANCE))
                    if self.logging:
                        self._log('slices: {}: all white and no transitions, creating {}, {}'.
                                  format(x, slices[x][-2], slices[x][-1]))
                elif last_pixel == MIN_LUMINANCE:
                    # its all black - not possible?
                    # create a falling step at 0 and a rising step at max_y
                    slices[x].append(Scan.Step(0, Scan.FALLING, MAX_LUMINANCE))
                    slices[x].append(Scan.Step(max_y-1, Scan.RISING, MAX_LUMINANCE))
                    if self.logging:
                        self._log('slices: {}: all black and no transitions, creating {}, {}'.
                                  format(x, slices[x][-2], slices[x][-1]))
                else:
                    # its all grey - this means all pixels are nearly the same in the integration area
                    # treat as if all white
                    # create a rising step at 0 and a falling step at max_y
                    slices[x].append(Scan.Step(0, Scan.RISING, MAX_LUMINANCE))
                    slices[x].append(Scan.Step(max_y-1, Scan.FALLING, MAX_LUMINANCE))
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
                self._log('    {}: {}'.format(x, steps))

        return slices

    def _rings(self, buckets: frame.Frame) -> List[List[Step]]:
        """ detect axial (around the target rings) luminance steps in the given binary/tertiary image,
            returns the step list for the image,
            each ring is expected to consist of N rising and falling edges, these represent digit boundaries,
            the x co-ord of a step is always that of the higher luminance pixel
            """

        max_x, max_y = buckets.size()

        # build list of transitions
        # NB: x wraps y does not
        slices = [[] for _ in range(max_y)]
        for y in range(max_y):
            last_pixel = buckets.getpixel(max_x - 1, y)
            transitions = 0
            for x in range(max_x):
                pixel = buckets.getpixel(x, y)
                if pixel < last_pixel:
                    # falling step
                    slices[y].append(Scan.Step((x-1) % max_x, Scan.FALLING, last_pixel))
                    transitions += 1
                elif pixel > last_pixel:
                    # rising step
                    slices[y].append(Scan.Step(x, Scan.RISING, pixel))
                    transitions += 1
                last_pixel = pixel
            if transitions == 0:
                # this means we're in one of the continuous marker rings
                if self.logging:
                    if last_pixel == MAX_LUMINANCE:
                        self._log('rings: at {}: all white'.format(y))
                    elif last_pixel == MIN_LUMINANCE:
                        self._log('rings: at {}: all black'.format(y))
                    else:
                        # is this possible?
                        self._log('rings: at {}: all grey'.format(y))

        if self.logging:
            self._log('rings: total={}'.format(len(slices)))
            for y, slice in enumerate(slices):
                if len(slice) == 0:
                    # nothing here
                    continue
                steps = ''
                for step in slice:
                    steps += ', {}'.format(step)
                steps = steps[2:]
                self._log('    {}: {}'.format(y, steps))

        return slices

    def _edges(self, slices: List[List[Step]], limit_x, limit_y, mode) -> ([Edge], [Edge]):
        """ build a list of falling and rising edges of our target,
            mode is horizontal or vertical and just modifies the edge type created,
            this function is orientation agnostic, the use of the terms x and y are purely convenience,
            returns the falling and rising edges list in increasing co-ordinate order,
            an 'edge' here is a sequence of connected rising or falling Steps,
            white to not-white is a falling edge,
            not-white to white is a rising edge,
            'not-white' is black or grey (when dealing with a tertiary image)
            """

        if mode == Scan.HORIZONTAL:
            max_x = limit_x
            max_y = limit_y
            x_wraps = True
            y_wraps = False
            context = 'h-'
        elif mode == Scan.VERTICAL:
            max_x = limit_y
            max_y = limit_x
            x_wraps = False
            y_wraps = True
            context = 'v-'
        else:
            raise Exception('_edges: mode must be {} or {} not {}'.format(Scan.HORIZONTAL, Scan.VERTICAL, mode))

        used = [[False for _ in range(max_y)] for _ in range(max_x)]

        def make_candidate(start_x, start_y, edge_type):
            """ make a candidate edge from the step from start_x,
                pixel pairs at x and x +/- 1 and x and x +/- 2 are considered,
                the next y is the closest of those 2 pairs,
                returns an instance of Edge or None
                """

            def get_nearest_y(x, y, step_type):
                """ find the y with the minimum acceptable gap to the given y at x """

                if used[x][y]:
                    # already been here
                    return None

                min_y = None
                min_gap = max_y * max_y

                slice = slices[x]
                if slice is not None:
                    for step in slice:
                        if step.pixel != MAX_LUMINANCE:
                            # this is a half-step, ignore those
                            continue
                        if step.type != step_type:
                            continue
                        if used[x][step.where]:
                            # already been here so not a candidate for another edge
                            continue
                        gap = wrapped_gap(y, step.where, max_y)
                        gap *= gap
                        if gap < min_gap:
                            min_gap = gap
                            min_y = step.where

                if min_gap > Scan.MAX_NEIGHBOUR_HEIGHT_GAP_SQUARED:
                    min_y = None

                return min_y

            candidate = [None for _ in range(max_x)]
            samples = 0
            for offset, increment in ((1, 1), (0, -1)):  # explore in both directions
                x = start_x - offset
                y = start_y
                for _ in range(max_x):
                    x += increment
                    if x >= max_x and not x_wraps:
                        # we're done
                        break
                    x %= max_x
                    this_y = get_nearest_y(x, y, edge_type)
                    if this_y is None:
                        # direct neighbour not connected, see if indirect one is
                        # this allows us to skip a slice (it may be noise)
                        dx = x + increment
                        if dx >= max_x and not x_wraps:
                            # no indirect either, so we're done
                            break
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
                # find the edge (==between None to not None and not None to None in candidate)
                start_x = None
                end_x = None
                if x_wraps:
                    scan_range = range(max_x)
                else:
                    scan_range = range(1, max_x)
                for x in scan_range:
                    prev_x = (x - 1) % max_x
                    this_y = candidate[x]
                    prev_y = candidate[prev_x]
                    if prev_y is None and this_y is not None:  # None to not-None is a start
                        start_x = x
                    elif prev_y is not None and this_y is None:  # not-None to None is an end
                        end_x = prev_x
                    if start_x is not None and end_x is not None:
                        # there can only be 1 sequence, so we're done when got both ends
                        break
                # make the Edge instance
                if start_x is None:
                    if x_wraps:
                        # this means this edge goes all the way around
                        edge = Scan.Edge(0, edge_type, candidate, direction=mode)
                    else:
                        # this means nothing here
                        edge = None
                elif end_x is None:
                    # this means the edge ends at the end
                    edge = Scan.Edge(start_x, edge_type, candidate[start_x:max_x], direction=mode)
                elif end_x < start_x:  # NB: end cannot be < start unless wrapping is allowed
                    # this means the edge wraps
                    edge = Scan.Edge(start_x, edge_type, candidate[start_x:max_x] + candidate[0:end_x+1], direction=mode)
                else:
                    edge = Scan.Edge(start_x, edge_type, candidate[start_x:end_x+1], direction=mode)
            else:
                edge = None
            return edge

        # build the edges list
        falling_edges = []
        rising_edges = []
        for x, slice in enumerate(slices):
            for step in slice:
                if step.pixel != MAX_LUMINANCE:
                    # this is a half-step, ignore those
                    continue
                candidate = make_candidate(x, step.where, step.type)
                if candidate is not None:
                    if step.type == Scan.FALLING:
                        falling_edges.append(candidate)
                    else:
                        rising_edges.append(candidate)

        if self.logging:
            self._log('{}edges: {} falling edges'.format(context, len(falling_edges)))
            for edge in falling_edges:
                self._log('    {}'.format(edge))
            self._log('{}edges: {} rising edges'.format(context, len(rising_edges)))
            for edge in rising_edges:
                self._log('    {}'.format(edge))

        return falling_edges, rising_edges

    def _extent(self, h_edges, v_edges, max_x, max_y) -> Extent:
        """ determine the target inner edge,
            there should be a consistent set of falling edges for the inner black ring,
            edges that are within a few pixels of each other going right round is what we want,
            returns the Extent (which defines the edge and/or a fail reason),
            the inner edge is the y where y-1=white and y=black,
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
                versions with the overlap removed, it returns *copies* the originals are
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

        def nipple(edge, x, y):
            """ check if there is a 'nipple' at x in the given edge,
                a 'nipple' is a y value with both its x neighbours having the same but different y value,
                the x value given here is the centre and the y value is that of its left neighbour,
                if the given x is a nipple the given y is returned, else the actual y at x is returned
                """

            next_y = edge[(x + 1) % max_x]
            if next_y == y:
                return y
            else:
                return edge[x]

        def smooth(edge, direction):
            """ smooth out any excessive y 'steps',
                direction RISING when doing an outer edge or FALLING when doing an inner,
                when joining edges we only check one end is close to another, the other end
                may be a long way off (due to overlaps caused by messy edges merging in the image),
                we detect these and 'smooth' them out, we detect them by finding successive y's
                with a difference of more than two, on detection the correction is to extrapolate
                a 45 degree slope until one meets the other, for an inner edge the lowest y is
                extrapolated towards the higher, for an outer edge its the other way round, e.g.:
                ----------.           ----------.           ----------.
                          .                     \                  \
                (A)       .       -->            \        or        \
                          .                       \                  \
                          ----------            ----------            ----------
                                                inner                 outer
                          ----------            ----------            ----------
                          .                    /                        /
                (B)       .       -->         /           or           /
                          .                  /                        /
                ----------.            ---------.            ---------.
                NB: a 45 degree slope is just advancing or retarding y by 1 on each x step
                """

            max_x = len(edge)
            last_y = edge[-1]
            x = -1
            while x < (max_x - 1):
                x += 1
                edge[x] = nipple(edge, x, last_y)  # get rid of nipple
                this_y = edge[x]
                diff_y = this_y - last_y
                if diff_y > 0+Scan.MAX_EDGE_HEIGHT_JUMP:
                    if direction == Scan.FALLING:
                        # (A) inner
                        x -= 1
                        while True:
                            x = (x + 1) % max_x
                            diff_y = edge[x] - last_y
                            if diff_y > Scan.MAX_EDGE_HEIGHT_JUMP:
                                last_y += 1
                                edge[x] = last_y
                            else:
                                break
                    else:
                        # (A) outer
                        last_y = this_y
                        while True:
                            x = (x - 1) % max_x
                            diff_y = last_y - edge[x]
                            if diff_y > Scan.MAX_EDGE_HEIGHT_JUMP:
                                last_y -= 1
                                edge[x] = last_y
                            else:
                                break
                elif diff_y < 0-Scan.MAX_EDGE_HEIGHT_JUMP:
                    if direction == Scan.FALLING:
                        # (B) inner
                        last_y = this_y
                        while True:
                            x = (x - 1) % max_x
                            diff_y = edge[x] - last_y
                            if diff_y > Scan.MAX_EDGE_HEIGHT_JUMP:
                                last_y += 1
                                edge[x] = last_y
                            else:
                                break
                    else:
                        # (B) outer
                        x -= 1
                        while True:
                            x = (x + 1) % max_x
                            diff_y = last_y - edge[x]
                            if diff_y > Scan.MAX_EDGE_HEIGHT_JUMP:
                                last_y -= 1
                                edge[x] = last_y
                            else:
                                break
                else:
                    # no smoothing required
                    last_y = this_y

            return edge

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
                if OK returns the edge and None or the partial edge and a fail reason if not OK
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
                full_edges.append((full_edge_samples, full_edge))

            # sort into longest order
            full_edges.sort(key=lambda e: e[0], reverse=True)

            if self.logging:
                for e, edge in enumerate(full_edges):
                    self._log('extent: #{} of {}: full edge candidate length {}'.
                              format(e + 1, len(full_edges), edge[0]))
                    log_edge(edge[1])

            if len(full_edges) == 0:
                # no edges detected!
                return [None for _ in range(max_x)], 'no edges'

            # extrapolate across any remaining gaps in the longest edge
            composed, reason = extrapolate(full_edges[0][1])
            if reason is not None:
                # failed
                return composed, reason

            # remove y 'steps'
            smoothed = smooth(composed, direction)

            return smoothed, None

        def log_edge(edge):
            """ log the edge in 32 byte chunks """
            max_edge = len(edge)
            block_size = 32
            blocks = int(max_edge / block_size)
            for block in range(blocks):
                start_block = block * block_size
                end_block = start_block + block_size
                self._log('    {}: {}'.format(start_block, edge[start_block : end_block]))
            residue = max_edge - (blocks * block_size)
            if residue > 0:
                self._log('    {}: {}'.format(len(edge) - residue, edge[-residue:]))
        # endregion

        falling_edges, rising_edges = h_edges

        # make the inner edge
        inner, inner_fail = compose(falling_edges, Scan.FALLING)
        if self.logging:
            self._log('extent: inner (fail={})'.format(inner_fail))
            log_edge(inner)

        return Scan.Extent(h_edges, v_edges, inner, inner_fail)

    def _identify(self, blob_size):
        """ identify the target in the current image with the given blob size (its radius) """

        # do the polar to cartesian projection
        target, stretch_factor = self._project(self.centre_x, self.centre_y, blob_size)
        if target is None:
            # its been rejected
            return None

        max_x, max_y = target.size()

        # do the edge detection
        buckets = self._binarize(target, s=Scan.THRESHOLD_SIZE,
                                 black=Scan.THRESHOLD_BLACK, white=Scan.THRESHOLD_WHITE, clean=True)
        slices = self._slices(buckets)
        h_edges = self._edges(slices, max_x, max_y, mode=Scan.HORIZONTAL)
        rings = self._rings(buckets)
        v_edges = None  # self._edges(rings, max_x, max_y, mode=Scan.VERTICAL)
        extent = self._extent(h_edges, v_edges, max_x, max_y)

        if self.save_images:
            plot = target
            # plot = self._draw_edges(v_edges, plot)
            plot = self._draw_edges(h_edges, plot, extent)
            self._unload(plot, '05-edges')

        return max_x, max_y, stretch_factor, buckets, slices, extent

    def _measure(self, extent: Extent, stretch_factor):
        """ get a measure of the target size by examining the extent,
            stretch_factor is how much the image height was stretched during projection,
            its used to re-scale the target size such that all are consistent wrt the original image
            """

        max_x = len(extent.inner)
        target_size = 0  # set as the average of the inner
        for x in range(max_x):
            target_size += extent.inner[x]
        target_size /= max_x
        target_size /= stretch_factor

        if self.logging:
            self._log('measure: target size is {:.2f} (with stretch compensation of {:.2f})'.
                      format(target_size, stretch_factor))

        return target_size

    def _find_segment_edges(self, buckets: frame.Frame, extent: Extent) -> ([int], str):
        """ find the segment edges from an analysis of the given buckets and extent,
            returns an edge list and None if succeeded, or partial list and a reason if failed,
            the extent contains the horizontal edges in the image,
            """

        if self.logging:
            header = 'find_segment_edges:'

        max_x, max_y = buckets.size()
        reason = None

        # ToDo: HACK
        # generate a normalised black/white strip for each x for visual testing
        inner = extent.inner
        falling_edges, rising_edges = extent.h_edges
        strips = buckets.instance().new(max_x, max_y, MID_LUMINANCE)
        digits = [[None, None] for _ in range(max_x)]
        ring_width = int(round(max_y / Scan.NUM_RINGS))
        strip_start_y = ring_width * 2
        strip_end_y = max_y - ring_width
        for x in range(max_x):
            start_y = inner[x] + 1  # get past the white bullseye
            lead_at = None
            head_at = None
            tail_at = None
            for y in range(start_y, max_y):
                pixel = buckets.getpixel(x, y)
                if pixel == MID_LUMINANCE:
                    # ignore grey areas
                    # ToDo: do something better than just ignore it? another option with it?
                    continue
                if pixel == MAX_LUMINANCE and lead_at is None:
                    # ignore until we see a black
                    continue
                if pixel == MAX_LUMINANCE and head_at is None:
                    # start of head
                    head_at = y
                    continue
                if pixel == MIN_LUMINANCE and lead_at is None:
                    # start of lead
                    lead_at = start_y  # use real start
                    continue
                if pixel == MIN_LUMINANCE and head_at is not None:
                    # end of head
                    tail_at = y
                    break
            if lead_at is None:
                # all white, do 1 black and the rest white
                lead_at = start_y
                head_at = lead_at + 1
                tail_at = max_y
            if head_at is None:
                # all black, do a trailing white
                lead_at = start_y
                head_at = max_y - 1
                tail_at = max_y
            if tail_at is None:
                # white to the end, do a trailing black
                tail_at = max_y
            for y in range(lead_at, head_at):
                strips.putpixel(x, (y - start_y) + strip_start_y, MIN_LUMINANCE)
            for y in range(head_at, tail_at):
                strips.putpixel(x, (y - start_y) + strip_start_y, MAX_LUMINANCE)
            for y in range(strip_start_y):
                strips.putpixel(x, y, MAX_LUMINANCE)
            for y in range(strip_end_y, max_y):
                strips.putpixel(x, y, MIN_LUMINANCE)
            lead_length = head_at - lead_at
            head_length = tail_at - head_at
            ratio = lead_length / head_length
            digits[x][0] = ratio
            digits[x][1] = self.decoder.classify(ratio)
        self._unload(strips, '05-strips')
        self._log(header)
        for x, (ratio, digit) in enumerate(digits):
            if ratio is None or digit is None:
                continue
            msg = ''
            for option in digit:
                msg = '{}, ({}, {:.2f})'.format(msg, option[0], option[1])
            self._log('    {}: ratio={:.2f}, options={}'.format(x, ratio, msg[2:]))
        return None, 'not yet'
        # ToDo: HACKEND

        # region helpers...
        def show_edge(edge):
            """ given an edge tuple return a string suitable for printing """
            return '({}, {})'.format(edge[0], edge[1])

        def show_edge_list(edges):
            """ given a list of edge tuples return a string suitable for printing """
            msg = ''
            for edge in edges:
                msg = '{}, {}'.format(msg, show_edge(edge))
            msg = msg[2:]
            return msg

        def make_zero_edges(zeroes):
            """ make a list of edges (for diagnostic purposes) from a list of zeroes """
            edges = []
            if self.logging:
                for start_0, end_0 in zeroes:
                    edges.append(start_0)
                    edges.append(end_0)
            return edges

        def find_zeroes(start_x=0, end_x=max_x, ) -> [[int, int]]:
            """ find zeroes between start_x and end_x, returns a list of zero start/end x co-ords,
                a zero is either mostly black or partially grey between the inner and outer edges,
                the guard areas are ignored
                """
            zeroes = []
            # find all the 0 runs
            start_0 = None
            for x in range(start_x, end_x):
                blacks = 0
                greys = 0
                whites = 0
                for y in range(extent.inner[x] + Scan.INNER_GUARD, extent.outer[x] - Scan.OUTER_GUARD):
                    pixel = buckets.getpixel(x, y)
                    if pixel == MAX_LUMINANCE:
                        whites += 1
                    elif pixel == MIN_LUMINANCE:
                        blacks += 1
                    else:
                        greys += 1
                if whites > Scan.ZERO_WHITE_THRESHOLD:
                    is_zero = False
                elif greys > Scan.ZERO_GREY_THRESHOLD:
                    is_zero = False
                else:
                    is_zero = True
                if is_zero:
                    if start_0 is None:
                        # this is the start of a 0 run
                        start_0 = x
                else:
                    if start_0 is not None:
                        # this is the end of a 0 run
                        zeroes.append([start_0, x - 1])
                        start_0 = None
            # deal with a run that wraps
            if start_0 is not None:
                # we ended in a 0 run,
                # if the first run starts at 0 then its a wrap,
                # otherwise the run ends at the end
                if len(zeroes) == 0:
                    # this means all the slices are 0!
                    pass
                elif zeroes[0][0] == 0:
                    # started on 0, so it really started on the last start
                    zeroes[0][0] = start_0
                else:
                    # add a final one that ended here
                    zeroes.append([start_0, max_x - 1])
            return zeroes

        def find_edges(start_x, end_x):
            """ between slices from start_x to end_x find the most likely segment edges,
                start_x and end_x given here are the first and last non-zero slices,
                NB: edges at start_x and end_x are assumed, we only look for edges between those,
                an 'edge' is detected as the white pixel differences across three consecutive slices
                """

            def get_difference(slice1, slice2, slice3):
                """ measure the difference between the three given slices,
                    the difference is the number of non-overlapping white pixels,
                    also returns a bool to indicate if the middle slice was a 'zero'
                    """

                def add_whites(slice, pixels):
                    """ accumulate white pixels from slice between the inner and outer extent """
                    is_zero = True
                    for y in range(extent.inner[slice], extent.outer[slice]):
                        pixel = buckets.getpixel(slice, y)
                        if pixel == MAX_LUMINANCE:
                            pixels[y] += 1
                            is_zero = False
                    return pixels, is_zero

                pixels = [0 for _ in range(max_y)]
                pixels, zero_1 = add_whites(slice1, pixels)
                pixels, zero_2 = add_whites(slice2, pixels)
                pixels, zero_3 = add_whites(slice3, pixels)
                difference = 0
                for y in range(max_y):
                    if pixels[y] == 3 or pixels[y] == 0:
                        # this means all 3 are the same
                        pass
                    else:
                        # there is a change across these 3,
                        # it could be 100, 110, 001, or 011 (101 is not possible due to cleaning)
                        difference += 1
                return difference, zero_2

            candidates = []
            if end_x < start_x:
                # we're wrapping
                end_x += max_x
            in_zero = None
            for x in range(start_x, end_x - 1):  # NB: stopping short to ensure next_slice is not over the end
                last_slice = (x - 1) % max_x
                this_slice = x % max_x
                next_slice = (x + 1) % max_x
                # measure difference between last, this and the next slice
                difference, is_zero = get_difference(last_slice, this_slice, next_slice)
                if is_zero:
                    if in_zero is None:
                        # found the start of a zero run - the middle of such a thing is an edge not its ends
                        in_zero = x
                    continue  # continue to find the other end
                else:
                    if in_zero is not None:
                        # got to the end of a zero run - set the middle as an edge candidate
                        midpoint = int(round(in_zero + ((x - in_zero) / 2)))
                        candidates.append((midpoint % max_x, max_y - 1))  # set as a semi-precious edge
                        in_zero = None
                        continue  # carry on to find the next edge
                if difference >= Scan.MIN_SEGMENT_EDGE_DIFFERENCE:
                    # got a big change - consider it to be an edge
                    candidates.append((x % max_x, difference))
            if in_zero:
                # we ended in a zero run
                # the end cannot be in a zero - so the end is also the end of a zero run
                midpoint = int(round(in_zero + ((end_x - in_zero) / 2)))
                candidates.append((midpoint % max_x, max_y - 1))  # set as a semi-precious edge

            return candidates

        def merge_smallest_segment(copy_edges, limit):
            """ merge segments that are within limit of each other,
                returns True if found one and copy_edges updated
                """
            nonlocal header
            smallest = limit
            smallest_at = None
            start_digit, _ = copy_edges[0]
            for x in range(1, len(copy_edges)):
                end_digit, _ = copy_edges[x]
                if end_digit < start_digit:
                    # this one wraps
                    width = end_digit + max_x - start_digit
                else:
                    width = end_digit - start_digit
                if width < smallest:
                    smallest = width
                    smallest_at = x
                start_digit = end_digit
            if smallest_at is None:
                # did not find one
                return False
            # smallest_at is the copy edge at the end of the smallest gap,
            # so its predecessor is the start,
            # we move the predecessor and delete smallest_at,
            # however, precious edge are never deleted, a precious edge is one with a difference of max_y
            # NB: copy_edges has >2 entries so first and last always have something in between
            start_at, start_difference = copy_edges[smallest_at - 1]  # NB: smallest_at cannot be 0
            end_at, end_difference = copy_edges[smallest_at]
            if start_difference == max_y:
                # start is precious, if end is precious too it means we found a zero near the start, tough
                if self.logging:
                    if header is not None:
                        self._log(header)
                        header = None
                    self._log('        removing edge at x:{}, gap of {} from precious start too small (limit is {:.2f})'.
                              format(show_edge(copy_edges[smallest_at]), smallest, limit))
                del copy_edges[smallest_at]
            elif end_difference == max_y:
                # end is precious, we know start
                if self.logging:
                    if header is not None:
                        self._log(header)
                        header = None
                    self._log('        removing edge at x:{}, gap of {} to precious end too small (limit is {:.2f})'.
                              format(show_edge(copy_edges[smallest_at - 1]), smallest, limit))
                del copy_edges[smallest_at - 1]
            else:
                # neither are precious, so merge them
                if end_at < start_at:
                    # it wraps
                    end_at += max_x
                span = end_at - start_at
                total_difference = min(start_difference + end_difference, max_y - 1)  # don't let it become precious
                offset = (end_difference / total_difference) * span
                midpoint = int(round(start_at + offset)) % max_x
                new_edge = (midpoint, total_difference)
                if self.logging:
                    if header is not None:
                        self._log(header)
                        header = None
                    self._log('        merging edge at x:{} and x:{} to x:{}, gap of {} too small (limit is {:.2f})'.
                              format(show_edge((start_at, start_difference)),
                                     show_edge((end_at, end_difference)),
                                     show_edge(new_edge), smallest, limit))
                copy_edges[smallest_at - 1] = new_edge
                del copy_edges[smallest_at]
            return True

        def split_biggest_segment(copy_edges, limit):
            """ split segments that are further apart than the given limit,
                returns True if found one and copy_edges updated
                """
            # we split off the nominal segment width each time,
            # the nominal segment width is last-edge minus first-edge (both of which we inserted above)
            # divided by number of expected edges (number of digits)
            nonlocal header
            first_at = copy_edges[0][0]
            last_at = copy_edges[-1][0]
            if last_at < first_at:
                # this copy wraps
                last_at += max_x
            nominal_width = (last_at - first_at) / Scan.DIGITS_PER_NUM
            biggest = limit
            biggest_at = None
            start_digit, _ = copy_edges[0]
            start_at = None
            for x in range(1, len(copy_edges)):
                end_digit, _ = copy_edges[x]
                if end_digit < start_digit:
                    # this one wraps
                    width = end_digit + max_x - start_digit
                else:
                    width = end_digit - start_digit
                if width >= biggest:
                    biggest = width
                    biggest_at = x
                    start_at = start_digit
                start_digit = end_digit
            if start_at is None:
                # did not find one
                return False
            new_edge = (int(round((start_at + nominal_width) % max_x)), 0)
            if self.logging:
                if header is not None:
                    self._log(header)
                    header = None
                self._log('        inserting edge at x:{}, gap of {} too big (limit is {:.2f})'.
                          format(new_edge, biggest, limit))
            copy_edges.insert(biggest_at, new_edge)
            return True
        # endregion

        # region find enough zeroes...
        zeroes = find_zeroes()
        if self.logging:
            if header is not None:
                self._log(header)
                header = None
            self._log('    zero candidates: {}'.format(zeroes))
        # dump zeroes that are too narrow or too wide
        max_actual_width = 0
        max_digit_width = (max_x / Scan.NUM_SEGMENTS) * Scan.MAX_SEGMENT_WIDTH
        for z in range(len(zeroes)-1, -1, -1):
            start_0, end_0 = zeroes[z]
            if end_0 < start_0:
                width = (end_0 + max_x) - start_0
            else:
                width = end_0 - start_0
            width += 1  # +1 'cos start/end is inclusive
            if width < Scan.MIN_SEGMENT_SAMPLES:
                # too narrow
                if self.logging:
                    if header is not None:
                        self._log(header)
                        header = None
                    self._log('        dropping narrow ({}) zero {} (limit {})'.
                              format(width, zeroes[z], Scan.MIN_SEGMENT_SAMPLES))
                del zeroes[z]
            elif width > max_digit_width:
                # too wide - this means we're looking at junk
                if self.logging:
                    if header is not None:
                        self._log(header)
                        header = None
                    self._log('    zero {} too wide at {} (limit {})'.format(zeroes[z], width, max_digit_width))
                edges = make_zero_edges(zeroes)
                return edges, 'zero too wide'
            else:
                if width > max_actual_width:
                    max_actual_width = width  # this is used to pick 'strong' zeroes
        # if we do not find at least COPIES zeroes we're looking at junk
        if len(zeroes) < Scan.COPIES:
            # cannot be a valid target
            if self.logging:
                if header is not None:
                    self._log(header)
                    header = None
                self._log('        only found {} zeroes, need {}'.format(len(zeroes), Scan.COPIES))
                self._log('    final zeroes={}'.format(zeroes))
                edges = []
                for start_0, end_0 in zeroes:
                    edges.append(start_0)
                    edges.append(end_0)
            else:
                edges = []
            return edges, 'missing 0''s'
        # if we find too many dump the 'worst' ones
        # first find the 'strong' zeroes - these are those close to the max_actual_width
        # strong zeroes are used as reference points for the weak ones
        # we make sure there is always at least Scan.COPIES 'strong' zeroes
        if len(zeroes) > Scan.COPIES:
            strong_zeroes = []
            strong_limit = max_actual_width * Scan.STRONG_ZERO_LIMIT
            ignore_limit = max_x
            while len(strong_zeroes) < Scan.COPIES:
                for z, (start_0, end_0) in enumerate(zeroes):
                    if end_0 < start_0:
                        width = (end_0 + max_x) - start_0
                    else:
                        width = end_0 - start_0
                    if width >= ignore_limit:
                        # this is bypassing ones we found last time
                        pass
                    elif width >= strong_limit:
                        strong_zeroes.append((start_0, end_0))
                ignore_limit = strong_limit  # do not want to find these again
                strong_limit -= 1  # bring limit down in case did not enough
            # create ideals for all our strong zeroes
            ideals = []
            ideal_gap = max_x / Scan.COPIES
            for start_0, end_0 in strong_zeroes:
                for copy in range(Scan.COPIES):
                    ideals.append(int(round(start_0 + (copy * ideal_gap))) % max_x)
            if self.logging:
                ideals.sort()  # sorting helps when looking at logs, not required by the code
                if header is not None:
                    self._log(header)
                    header = None
                self._log('    strong zeroes: {}'.format(strong_zeroes))
                self._log('    ideal strong zero edges: {}'.format(ideals))
            while len(zeroes) > Scan.COPIES:
                # got too many, dump the 'worst' one,
                # the worst is that farthest from an ideal position of a 'strong' zero
                # for each 0, calculate the smallest difference to an 'ideal' 0
                # dump the 0 with the biggest difference
                gaps = []
                for z, (start_0, end_0) in enumerate(zeroes):
                    smallest_gap = max_x
                    smallest_at = None
                    for ideal in ideals:
                        if ideal == start_0:
                            # this is self, ignore it
                            continue
                        # we need to allow for wrapping
                        # a gap of > +/-ideal_gap is either junk or a wrap
                        if start_0 < ideal:
                            gap = ideal - start_0
                            if gap > ideal_gap:
                                # assume a wrap
                                gap = start_0 + max_x - ideal
                        else:
                            gap = start_0 - ideal
                            if gap > ideal_gap:
                                # assume a wrap
                                gap = ideal + max_x - start_0
                        if gap < smallest_gap:
                            smallest_gap = gap
                            smallest_at = z
                    gaps.append((smallest_gap, smallest_at))
                biggest_gap = 0
                biggest_at = None
                ambiguous = None
                for gap in gaps:
                    if gap[0] > biggest_gap:
                        biggest_gap = gap[0]
                        biggest_at = gap[1]
                        ambiguous = None
                    elif gap[0] == biggest_gap:
                        ambiguous = gap[1]
                if ambiguous is not None and (len(zeroes) - 1) == Scan.COPIES:
                    # ToDo: deal with ambiguity somehow
                    if self.logging:
                        if header is not None:
                            self._log(header)
                            header = None
                        self._log('        got excess zeroes with ambiguity at x:{} and x:{}'.
                                  format(zeroes[biggest_at][0], zeroes[ambiguous][0]))
                if self.logging:
                    if header is not None:
                        self._log(header)
                        header = None
                    self._log('        found {} zeroes, only want {}, dropping {} (gap to ideal is {})'.
                              format(len(zeroes), Scan.COPIES, zeroes[biggest_at], biggest_gap))
                # NB: This may delete a 'strong' zero and thereby invalidate its ideals.
                #     If we removed the ideals it would just make subsequent gaps even bigger.
                #     It could also invalidate already accepted zeroes.
                #     What to do?
                #     Nothing! If things are that iffy we're probably looking at junk anyway.
                zeroes.pop(biggest_at)
        # if gap between zeroes is too small or too big we're looking at junk
        # too small is DIGITS_PER_NUM * MIN_SEGMENT_SAMPLES
        # too big is relative to a 'copy width'
        min_gap = Scan.MIN_ZERO_GAP
        max_gap = (max_x / Scan.COPIES) * Scan.MAX_ZERO_GAP
        if self.logging:
            edges = make_zero_edges(zeroes)
        else:
            edges = []
        for copy in range(Scan.COPIES):
            _, start_x = zeroes[copy]
            end_x, _ = zeroes[(copy + 1) % Scan.COPIES]
            if end_x < start_x:
                # we've wrapped
                gap = end_x + max_x - start_x
            else:
                gap = end_x - start_x
            if gap < min_gap:
                # too small, so this is junk
                if self.logging:
                    if header is not None:
                        self._log(header)
                        header = None
                    self._log('    gap ({}) between zeroes at x:{} to x:{} too small (limit is {})'.
                              format(gap, start_x, end_x, min_gap))
                return edges, '0 gap''s too small'
            elif gap > max_gap:
                # too big, so this is junk
                if self.logging:
                    if header is not None:
                        self._log(header)
                        header = None
                    self._log('    gap ({}) between zeroes at x:{} to x:{} too big (limit is {})'.
                              format(gap, start_x, end_x, max_gap))
                return edges, '0 gap''s too big'

        if self.logging:
            if header is not None:
                self._log(header)
                header = None
            self._log('    final zeroes={}'.format(zeroes))
        # endregion

        # find segment edges between the 0's
        edges = []
        candidates = []
        for copy in range(Scan.COPIES):
            _, start_x = zeroes[copy]
            end_x, _ = zeroes[(copy + 1) % Scan.COPIES]
            # find edge candidates
            copy_edges = find_edges(start_x + 1, end_x - 2)  # keep away from the zero edges
            copy_edges.insert(0, (start_x, max_y))      # include the initial (precious) edge (end of this 0)
            copy_edges.append((end_x, max_y))           # ..and the final (precious) edge (start of next 0)
            if self.logging:
                if header is not None:
                    self._log(header)
                    header = None
                self._log('    copy {}: edge candidates=[{}]'.format(copy, show_edge_list(copy_edges)))
            # merge edges that are too narrow
            # too narrow here is relative to the nominal segment width,
            # nominal segment width is the span between the zeroes divided by number of digits in that span
            if end_x < start_x:
                # we've wrapped
                span = (end_x + max_x) - start_x
            else:
                span = end_x - start_x
            min_segment = int(round((span / Scan.DIGITS_PER_NUM) * Scan.MIN_SEGMENT_WIDTH))
            while merge_smallest_segment(copy_edges, min_segment):
                pass
            while len(copy_edges) > (Scan.DIGITS_PER_NUM + 1):
                # got too many, merge two with the smallest gap
                merge_smallest_segment(copy_edges, max_x)
            while len(copy_edges) < (Scan.DIGITS_PER_NUM + 1):
                # haven't got enough, split the biggest
                if not split_biggest_segment(copy_edges, 0):
                    # this means we're looking at crap
                    if self.logging:
                        if header is not None:
                            self._log(header)
                            header = None
                        self._log('    found {} edges, need {}, cannot split, giving up'.
                                  format(len(copy_edges), Scan.DIGITS_PER_NUM + 1))
                    reason = 'no edges'
                    break
            # add the segment edges for this copy
            if self.logging:
                if header is not None:
                    self._log(header)
                    header = None
                self._log('    copy {}: final edges=[{}]'.format(copy, show_edge_list(copy_edges)))
            for edge in copy_edges:
                edges.append(edge[0])

        # we're done
        if self.logging:
            edges.sort()  # not necessary but helps when viewing the logs
            if header is not None:
                self._log(header)
                header = None
            self._log('    final edges={}'.format(edges))

        return edges, reason

    def _find_pulses(self, edges: [int], buckets: frame.Frame, extent: Extent) -> ([Pulse], [int]):
        """ extract the lead and head lengths from each of the segments defined in edges,
            buckets is the binarized image, the extent defines the inner and outer limits,
            each x co-ord is considered as a 'slice' across y (the radius), each edge is
            the x co-ord of a bit change, each slice is either a 'pulse' or empty (all black),
            all slices empty (or nearly empty) in an edge is a 'zero' (lead>0, head=0, tail=0),
            empty (or nearly empty) slices otherwise are ignored, (they are noise within a bit),
            the position of the head and tail of each slice is determined and their average is
            used to calculate pulse component lengths, any slice with a double head is ignored,
            the first and last slice of an edge are also ignored (to make sure we get no overlaps),
            a grey pixel in the lead is attributed half to the lead and half to the head,
            a grey pixel in the tail is attributed half to the head and half to the tail,
            returns a list of Scan.Pulse reflecting the lead, head and tail lengths found
            """

        # ToDo: a simple area is not good enough, think of a better way?
        #       cell edge detection not good enough - think again...

        if self.logging:
            header = 'find_pulses:'

        max_x, max_y = buckets.size()

        pulses = [None for _ in range(len(edges))]
        bad_slices = []  # diagnostic info

        for edge in range(len(edges)):
            lead_start = 0
            head_start = 0
            tail_start = 0
            tail_stop = 0
            start_x = edges[edge]
            stop_x = edges[(edge + 1) % len(edges)]
            if stop_x < start_x:
                # this one wraps
                stop_x += max_x
            head_slices = 0
            zero_slices = []
            for dx in range(start_x + 1, stop_x):  # NB: ignoring first and last slice
                x = dx % max_x
                lead = 0
                head = 0
                tail = 0
                lead_grey = 0
                tail_grey = 0
                ignore_slice = False
                for y in range(extent.inner[x], extent.outer[x]):
                    pixel = buckets.getpixel(x, y)
                    if pixel == MID_LUMINANCE:
                        # got a grey
                        if head == 0:
                            # we're in the lead area
                            lead_grey += 1
                        else:
                            # we're in the head or tail area
                            tail_grey += 1
                    elif pixel == MIN_LUMINANCE:
                        # got a black pixel
                        if head == 0:
                            # we're still in the lead area
                            lead += 1
                        elif tail == 0:
                            # we're entering the tail area
                            tail = 1
                        else:
                            # we're still on the tail area
                            tail += 1
                    else:
                        # got a white pixel
                        if head == 0:
                            # we're entering the head area
                            head = 1
                        elif tail == 0:
                            # we're still in the head area
                            head += 1
                        else:
                            # got a second pulse, this is a segment overlap due to neighbour bleeding
                            # ignore this slice
                            ignore_slice = True
                            bad_slices.append(x)
                            if self.logging:
                                if header is not None:
                                    self._log(header)
                                    header = None
                                self._log('    ignoring slice with double pulse at x:{} y:{}'.format(x, y))
                            break
                if ignore_slice:
                    continue
                # make the grey adjustments
                if head > 0:
                    lead += lead_grey * (1 - Scan.LEAD_GRAY_TO_HEAD)
                    head += lead_grey * Scan.LEAD_GRAY_TO_HEAD
                    head += tail_grey * Scan.TAIL_GRAY_TO_HEAD
                    tail += tail_grey * (1 - Scan.TAIL_GRAY_TO_HEAD)
                # check what we got
                if head < Scan.MIN_PULSE_HEAD:
                    # got an empty (or nearly empty) slice - note it for possible ignore
                    if self.logging and head > 0:
                        if header is not None:
                            self._log(header)
                            header = None
                        self._log('    treating slice with small head ({}) at x:{} as zero (limit is {}))'.
                                  format(head, x, Scan.MIN_PULSE_HEAD))
                    zero_slices.append(x)  # note for later
                else:
                    head_slices += 1
                    # note pulse part positions
                    lead_start_y = extent.inner[x]
                    head_start_y = lead_start_y + lead
                    tail_start_y = head_start_y + head
                    tail_stop_y = tail_start_y + tail
                    # update accumulators
                    lead_start += lead_start_y
                    head_start += head_start_y
                    tail_start += tail_start_y
                    tail_stop += tail_stop_y
            if head_slices > 0:
                # got a pulse for this edge
                lead_start /= head_slices
                head_start /= head_slices
                tail_start /= head_slices
                tail_stop /= head_slices
                lead = head_start - lead_start
                head = tail_start - head_start
                tail = tail_stop - tail_start
                pulses[edge] = Scan.Pulse(start=lead_start, stop=tail_stop,
                                          lead=lead, head=head, tail=tail,
                                          begin=start_x, end=stop_x % max_x)
                bad_slices += zero_slices
            elif len(zero_slices) > 0:
                # got a zero for this edge
                lead_start = 0
                tail_stop = 0
                for x in zero_slices:
                    lead_start += extent.inner[x]
                    tail_stop += extent.outer[x]
                lead_start /= len(zero_slices)
                tail_stop /= len(zero_slices)
                lead = tail_stop - lead_start
                pulses[edge] = Scan.Pulse(start=lead_start, stop=tail_stop,
                                          lead=lead, head=0, tail=0,
                                          begin=start_x, end=stop_x % max_x)

            if self.logging:
                if header is not None:
                    self._log(header)
                    header = None
                self._log('    {}: pulse {} in {} heads and {} zeroes'.
                          format(edge, pulses[edge], head_slices, len(zero_slices)))

        return pulses, bad_slices

    def _find_bits(self, pulses: [Pulse]) -> [Pulse]:
        """ extract the bit sequences and their errors from the given pulse list """

        def error(actual, ideal):
            """ calculate an error between the actual pulse part ratios and the ideal,
                we want a number that is in the range 0..N where 0 is no error and N
                is a huge error
                """
            if ideal > 0 and actual > 0:
                if ideal > actual:
                    err = actual / ideal  # range 1..0 (perfect..crap)
                else:
                    err = ideal /actual  # range 1..0 (perfect..crap)
            elif ideal == 0 and actual == 0:
                err = 1  # no error
            else:
                err = 0  # max poss error
            err = 1 - err  # range now 0..1
            err *= err  # range still 0..1
            err *= Scan.PULSE_ERROR_RANGE  # err now 0..N
            return err

        # get the most likely bits
        for x, pulse in enumerate(pulses):
            if pulse is None:
                continue
            actual = pulse.ratio()
            if actual is None:
                # not a valid pulse
                continue
            for idx, ideal in enumerate(Scan.RATIOS):
                lead_error = error(actual[0], ideal[0])
                head_error = error(actual[1], ideal[1])
                tail_error = error(actual[2], ideal[2])
                err = (lead_error + head_error + tail_error) / 3
                # ToDo: under new coding DIGITS[n] can be None
                pulse.bits.append(Scan.Bits(Scan.DIGITS[idx], err, actual=actual, ideal=ideal))
            pulse.bits.sort(key=lambda b: b.error())  # put into least error order
            if len(pulse.bits) > 0:
                # chuck out options where the error is too high relative to the best
                error_limit = pulse.bits[0].error() * Scan.MAX_CHOICE_ERROR_DIFF
                # ToDo: when MAX_CHOICE_ERROR_DIFF is 4 I get less choices than when it is 3, why?
                for bit in range(1, len(pulse.bits)):
                    if pulse.bits[bit].error() > error_limit:
                        # chuck this and the rest
                        pulse.bits = pulse.bits[:bit]
                        break
            # keep at most 3
            if len(pulse.bits) > 3:
                pulse.bits = pulse.bits[:2]

        if self.logging:
            self._log('find_bits: options:')
            for x, pulse in enumerate(pulses):
                if pulse is None:
                    self._log('    {}: None'.format(x))
                    continue
                msg = ''
                for bits in pulse.bits:
                    msg = '{}, {}'.format(msg, bits)
                self._log('    {}: {}'.format(x, msg[2:]))

        return pulses

    def _decode_bits(self, pulses: [Pulse], max_x, max_y):
        """ decode the pulse bits for the least doubt and return its corresponding number,
            in pulse bits we have a list of bit sequence choices across the data rings, i.e. bits x rings
            we need to rotate that to rings x bits to present it to our decoder,
            we present each combination to the decoder and pick the result with the least doubt
            """

        def build_choice(start_x, choice, choices):
            """ build a list of all combinations of the bit choices (up to some limit),
                each choice is a list of segments and the number of choices at that segment position
                """
            if len(choices) >= Scan.MAX_BIT_CHOICES:
                # got too many
                return choices, True
            for x in range(start_x, len(pulses)):
                pulse = pulses[x]
                if pulse is None:
                    continue
                bit_list = pulse.bits
                if len(bit_list) == 0:
                    continue
                if len(bit_list) > 1:
                    # got choices - recurse for the others
                    for dx in range(1, len(bit_list)):
                        bit_choice = choice.copy()
                        bit_choice[x] = (bit_list[dx], len(bit_list))
                        choices, overflow = build_choice(x+1, bit_choice, choices)
                        if overflow:
                            return choices, overflow
                choice[x] = (bit_list[0], len(bit_list))
            choices.append(choice)
            return choices, False

        # build all the choices
        choices, overflow = build_choice(0, [None for _ in range(len(pulses))], [])

        # try them all
        results = []
        max_bit_error = 0
        for choice in choices:
            code = [[None for _ in range(Scan.NUM_SEGMENTS)] for _ in range(Scan.NUM_DATA_RINGS)]
            bit_doubt = 0
            bit_error = 0
            for bit, (segment, options) in enumerate(choice):
                if segment is None:
                    # nothing recognized here
                    rings = None
                else:
                    rings = segment.bits
                    bit_error += segment.error()
                    if options > 1:
                        bit_doubt += options - 1
                if rings is not None:
                    for ring in range(len(rings)):
                        sample = rings[ring]
                        code[ring][bit] = sample
            if bit_error > max_bit_error:
                max_bit_error = bit_error
            bit_doubt = min(bit_doubt, 99)
            number, doubt, digits = self.decoder.unbuild(code)
            results.append(Scan.Result(number, doubt, digits, bit_doubt, bit_error, choice))

        # set the doubt for each result now we know the max bit error
        for result in results:
            result.doubt(max_bit_error)

        # put into least doubt order with numbers before None's
        results.sort(key=lambda r: (r.number is None, r.doubt(), r.count, r.number))  # NB: Relying on True > False

        # build list of unique results and count duplicates
        numbers = {}
        for result in results:
            if numbers.get(result.number) is None:
                # this is the first result for this number and is the best for that number
                numbers[result.number] = result
            else:
                numbers[result.number].count += 1  # increase the count for this number
        # move from a dictionary to a list so we can re-sort it
        results = [result for result in numbers.values()]
        results.sort(key=lambda r: (r.number is None, r.doubt(), r.count, r.number))

        # if we have multiple results with similar doubt we have ambiguity,
        # ambiguity can result in a false detection, which is not safe
        # we also have ambiguity if we overflowed building the choice list ('cos we don't know if
        # a better choice has not been explored
        ambiguous = None
        if len(results) > 1:
            # we have choices
            doubt_diff = results[1].digit_doubt - results[0].digit_doubt  # >=0 'cos results in doubt order
            if doubt_diff < Scan.CHOICE_DOUBT_DIFF_LIMIT:
                # its ambiguous
                ambiguous = doubt_diff

        # get the best result
        best = results[0]
        if ambiguous is not None or overflow:
            # mark result as ambiguous
            if best.number is not None:
                # make it negative as a signal that its ambiguous
                best.number = 0 - best.number

        if self.logging:
            if overflow:
                msg = ' with overflow (limit is {})'.format(Scan.MAX_BIT_CHOICES)
            elif ambiguous is not None:
                msg = ' with ambiguity (top 2 doubt diff is {}, limit is {})'.\
                      format(ambiguous, Scan.CHOICE_DOUBT_DIFF_LIMIT)
            else:
                msg = ''
            self._log('decode: {} results from {} choices{}:'.format(len(results), len(choices), msg))
            self._log('    best number={}, occurs={}, doubt={:.4f}, bits={}'.
                      format(best.number, best.count, best.doubt(), best.digits))
            self._log('        best result bits:')
            for x, (bits, _) in enumerate(best.choice):
                self._log('            {}: {}'.format(x, bits))
            for r in range(1, len(results)):
                result = results[r]
                self._log('    number={}, occurs={}, doubt={:.4f}, bits={}'.
                          format(result.number, result.count, result.doubt(), result.digits))

        if self.save_images:
            segments = []
            size = int(max_x / Scan.NUM_SEGMENTS)
            for x, (bits, _) in enumerate(best.choice):
                if bits is not None:
                    segments.append(Scan.Segment(x * size, bits.bits, size))
            grid = self._draw_segments(segments, max_x, max_y)
            self._unload(grid, '08-bits')

        # return best
        return best.number, best.doubt(), best.digits

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

            max_x, max_y, stretch_factor, buckets, slices, extent = result
            if extent.inner_fail is not None:
                # failed - this means we did not find its inner edge
                if self.save_images:
                    # add to reject list for labelling on the original image
                    reason = extent.inner_fail
                    rejects.append(Scan.Reject(self.centre_x, self.centre_y, blob_size, blob_size*4, reason))
                continue

            edges, reason = self._find_segment_edges(buckets, extent)
            if self.save_images and edges is not None and len(edges) > 0:
                plot = self._draw_extent(extent, buckets, bleed=0.8)
                lines = []
                for edge in edges:
                    lines.append((edge, 0, edge, max_y - 1,))
                if reason is not None:
                    colour = Scan.RED
                else:
                    colour = Scan.GREEN
                plot = self._draw_lines(plot, lines, colour=colour)
                self._unload(plot, '06-cells')
            if reason is not None:
                # we failed to find segment edges
                if self.save_images:
                    # add to reject list for labelling on the original image
                    rejects.append(Scan.Reject(self.centre_x, self.centre_y, blob_size, blob_size*4, reason))
                continue
            pulses, bad_slices = self._find_pulses(edges, buckets, extent)
            if self.save_images:
                if len(bad_slices) > 0:
                    # show bad slices as blue on our cells image
                    lines = []
                    for slice in bad_slices:
                        lines.append((slice, 0, slice, max_y - 1))
                    plot = self._draw_lines(plot, lines, colour=Scan.BLUE)
                # show pulse edges as green horizontal lines
                lines = []
                for pulse in pulses:
                    if pulse is None:
                        continue
                    lead_start = pulse.start  # there is always a start
                    if pulse.head is not None:
                        head_start = lead_start + pulse.lead  # there is always a lead
                        tail_start = head_start + pulse.head
                        tail_end = tail_start + pulse.tail  # there is always a tail if there is a head
                    else:
                        head_start = None
                        tail_start = None
                        tail_end = lead_start + pulse.lead  # there is always a lead
                    lines.append((pulse.begin, lead_start, pulse.end, lead_start))
                    lines.append((pulse.begin, tail_end, pulse.end, tail_end))
                    if head_start is not None:
                        lines.append((pulse.begin, head_start, pulse.end, head_start))
                        lines.append((pulse.begin, tail_start, pulse.end, tail_start))
                plot = self._draw_lines(plot, lines, colour=Scan.GREEN, h_wrap=True)
                self._unload(plot, '06-cells')  # overwrite the one we did earlier
            pulses = self._find_bits(pulses)
            outer_y = self.radial_steps  # y scale for drawing diagnostic images
            result = self._decode_bits(pulses, max_x, outer_y)
            target_size = self._measure(extent, stretch_factor)
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
            number, doubt, digits = target.result

            # add this result
            numbers.append(Scan.Detection(number, doubt, self.centre_x, self.centre_y, target_size, blob_size, digits))

            if self.save_images:
                number = numbers[-1]
                if number.number is None:
                    colour = Scan.PURPLE
                    label = 'invalid ({:.4f})'.format(number.doubt)
                elif number.number < 0:
                    colour = Scan.RED
                    label = 'ambiguous {} ({:.4f})'.format(0-number.number, number.doubt)
                else:
                    colour = Scan.GREEN
                    label = 'code is {} ({:.4f})'.format(number.number, number.doubt)
                # draw the detected blob in blue
                k = (number.centre_x, number.centre_y, number.blob_size)
                detections = self.transform.label(detections, k, Scan.BLUE)
                # draw the result
                k = (number.centre_x, number.centre_y, number.target_size)
                detections = self.transform.label(detections, k, colour, '{:.0f}x{:.0f}y {}'.
                                              format(number.centre_x, number.centre_y, label))

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

    def _draw_segments(self, segments: List[Segment], max_x, max_y):
        """ draw an image of the given segments """

        def draw_block(grid, start_x, end_x, start_y, ring_width, max_x, colour):
            """ draw a coloured block as directed """

            end_y = int(start_y + ring_width - 1)
            start_y = int(start_y)

            if end_x < start_x:
                # its a wrapper
                self.transform.fill(grid, (start_x, start_y), (max_x - 1, end_y), colour)
                self.transform.fill(grid, (0, start_y), (end_x, end_y), colour)
            else:
                self.transform.fill(grid, (start_x, start_y), (end_x, end_y), colour)

        def draw_segment(grid, segment, ring_width, max_x):
            """ draw the given segment """

            data_start = ring_width * 3  # 2 inner white + 1 inner black
            if segment.bits is None:
                if segment.choices is None:
                    bits = [None for _ in range(Scan.NUM_DATA_RINGS)]
                else:
                    bits = segment.choices[0].bits
                    one_colour = Scan.PALE_GREEN
                    zero_colour = Scan.PALE_BLUE
            else:
                bits = segment.bits
                if segment.choices is None:
                    one_colour = Scan.WHITE
                    zero_colour = Scan.BLACK
                else:
                    one_colour = Scan.GREEN
                    zero_colour = Scan.BLUE
            for ring, bit in enumerate(bits):
                if bit is None:
                    colour = Scan.PALE_RED
                elif bit == 1:
                    colour = one_colour
                else:
                    colour = zero_colour
                if colour is not None:
                    ring_start = data_start + (ring_width * ring)
                    draw_block(grid,
                               segment.start, (segment.start + segment.samples - 1) % max_x,
                               ring_start, ring_width,
                               max_x, colour)

        # make an empty (i.e. black) colour image to load our segments into
        ring_width = max_y / Scan.NUM_RINGS
        grid = self.original.instance()
        grid.new(max_x, max_y, MIN_LUMINANCE)
        grid.incolour()

        # draw the inner white rings (the 'bullseye')
        draw_block(grid, 0, max_x - 1, 0, ring_width * 2, max_x, Scan.WHITE)

        # draw the outer white ring
        draw_block(grid, 0, max_x - 1, max_y - ring_width, ring_width, max_x, Scan.WHITE)

        # fill data area with red so can see gaps
        draw_block(grid, 0, max_x - 1, ring_width * 3, ring_width * Scan.NUM_DATA_RINGS, max_x, Scan.RED)

        # draw the segments
        for segment in segments:
            draw_segment(grid, segment, ring_width, max_x)

        return grid

    def _draw_extent(self, extent: Extent, target, bleed):
        """ make the area outside the inner edge on the given target visible """

        max_x, max_y = target.size()
        inner = extent.inner

        inner_lines = []
        for x in range(max_x):
            if inner[x] is not None:
                inner_lines.append((x, 0, x, inner[x] - 1))  # inner edge is on the first black, -1 to get to white
        target = self._draw_lines(target, inner_lines, colour=Scan.RED, bleed=bleed)

        return target

    def _draw_pulses(self, pulses, extent, buckets):
        """ draw the given pulses on the given buckets image """

        # draw pulse lead/tail as green lines, head as a blue line, none as red
        # build lines
        max_x, max_y = buckets.size()
        lead_lines = []
        head_lines = []
        tail_lines = []
        none_lines = []
        for x, pulse in enumerate(pulses):
            if pulse is None:
                none_lines.append((x, 0, x, max_y - 1))
            else:
                lead_lines.append((x, pulse.start, x, pulse.start + pulse.lead - 1))
                head_lines.append((x, pulse.start + pulse.lead,
                                   x, pulse.start + pulse.lead + pulse.head - 1))
                if pulse.tail > 0:
                    # for half-pulses there is no tail
                    tail_lines.append((x, pulse.start + pulse.lead + pulse.head,
                                       x, pulse.start + pulse.lead + pulse.head + pulse.tail - 1))

        # mark the extent
        plot = self._draw_extent(extent, buckets, bleed=0.6)

        # draw lines on the bucketised image
        plot = self._draw_lines(plot, none_lines, colour=Scan.RED)
        plot = self._draw_lines(plot, lead_lines, colour=Scan.GREEN)
        plot = self._draw_lines(plot, head_lines, colour=Scan.BLUE)
        plot = self._draw_lines(plot, tail_lines, colour=Scan.GREEN)

        return plot

    def _draw_edges(self, edges, target: frame.Frame, extent: Extent=None, bleed=0.5):
        """ draw the edges and the inner extent on the given target image """

        falling_edges = edges[0]
        rising_edges = edges[1]

        if extent is None:
            plot = target
        else:
            # mark the image area outside the inner and outer extent
            plot = self._draw_extent(extent, target, bleed=0.8)

        # plot falling and rising edges
        x_falling_points = []
        x_rising_points = []
        y_falling_points = []
        y_rising_points = []
        for edge in falling_edges:
            if edge.direction == Scan.HORIZONTAL:
                x_falling_points.append((edge.where, edge.samples))
            else:
                y_falling_points.append((edge.where, edge.samples))
        for edge in rising_edges:
            if edge.direction == Scan.HORIZONTAL:
                x_rising_points.append((edge.where, edge.samples))
            else:
                y_rising_points.append((edge.where, edge.samples))
        plot = self._draw_plots(plot, plots_x=x_falling_points, colour=Scan.GREEN, bleed=bleed)
        plot = self._draw_plots(plot, plots_x=x_rising_points, colour=Scan.GREEN, bleed=bleed)
        plot = self._draw_plots(plot, plots_y=y_falling_points, colour=Scan.BLUE, bleed=bleed)
        plot = self._draw_plots(plot, plots_y=y_rising_points, colour=Scan.BLUE, bleed=bleed)

        return plot
    # endregion

