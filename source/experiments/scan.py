
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
    BULLSEYE_RINGS = ring.Ring.BULLSEYE_RINGS  # number of rings inside the inner edge
    NUM_DATA_RINGS = codec.Codec.RINGS_PER_DIGIT  # how many data rings in our codes
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
    # ToDo: de-tune blob area and radius if new scheme cannot get valid detections from ...-distant.jpg
    MIN_BLOB_AREA = 8  # min area of a blob we want (in pixels) (default 9)
    MIN_BLOB_RADIUS = 2  # min radius of a blob we want (in pixels) (default 2.5)
    BLOB_RADIUS_STRETCH = 1.1  # how much to stretch blob radius to ensure always cover everything when projecting
    MIN_CONTRAST = 0.5  # minimum luminance variation of a valid blob projection relative to the mean luminance
    THRESHOLD_WIDTH = 5  # the fraction of the projected image width to use as the integration area when binarizing
    THRESHOLD_HEIGHT = 1.5  # the fraction of the projected image height to use as the integration area (None=as width)
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
    MAX_EDGE_HEIGHT_JUMP = 2  # max jump in y, in pixels, along an edge before smoothing is triggered
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
            if len(self.samples) > 10:
                samples = ': {}..{}'.format(self.samples[:5], self.samples[-5:])
            else:
                samples = ': {}'.format(self.samples)
            from_x = self.where
            to_x = from_x + len(self.samples) - 1
            from_y = self.samples[0]
            to_y = self.samples[-1]
            return '({} at ({},{}) to ({},{}) for {}{})'.\
                format(self.type, from_x, from_y, to_x, to_y, len(self.samples), samples)

    class Extent:
        """ an Extent is the inner edge co-ordinates of a projected image along with
            the horizontal and vertical edge fragments it was built from """

        def __init__(self, inner=None, outer=None, inner_fail=None, outer_fail=None,
                     buckets=None, rising_edges=None, falling_edges=None):
            self.inner: [int] = inner  # list of y co-ords for the inner edge
            self.inner_fail = inner_fail  # reason if failed to find inner edge or None if OK
            self.outer: [int] = outer  # list of y co-ords for the outer edge
            self.outer_fail = outer_fail  # reason if failed to find outer edge or None if OK
            self.rising_edges: [Scan.Edge] = rising_edges  # rising edge list used to create this extent
            self.falling_edges: [Scan.Edge] = falling_edges  # falling edge list used to create this extent
            self.buckets = buckets  # the binarized image the extent was created from

    class Digit:
        """ a digit is a decode of a sequence of slice samples into the most likely digit """

        def __init__(self, digit, error, samples):
            self.digit = digit  # the most likely digit
            self.error = error  # the average error across its samples
            self.samples = samples  # the number of samples in this digit

        def __str__(self):
            return '({}, {:.2f}, {})'.format(self.digit, self.error, self.samples)

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
        params.min_area = Scan.MIN_BLOB_AREA
        params.min_radius = Scan.MIN_BLOB_RADIUS
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
            cX: float = centre_x + x
            cY: float = centre_y + y
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

        max_x, max_y = target.size()

        def make_binary(image: frame.Frame, width: float=8, height: float=None, black: float=15, white: float=None) -> frame.Frame:
            """ given a greyscale image return a binary image using an adaptive threshold.
                width, height, black, white - see description of overall method
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
                self._log('    {}: {}'.format(x, steps))

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
                        if step.type != step_type:
                            continue
                        if used[x][step.where]:
                            # already been here so not a candidate for another edge
                            continue
                        if step.type == Scan.FALLING:
                            if step.to_pixel == MIN_LUMINANCE:
                                # got a qualifying falling step
                                pass
                            else:
                                # ignore this step
                                continue
                        else:  # step.type == Scan.RISING
                            if step.from_pixel == MIN_LUMINANCE:
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

            candidate = [None for _ in range(max_x)]
            samples = 0
            for offset, increment in ((1, 1), (0, -1)):  # explore in both directions
                x = start_x - offset
                y = start_y
                for _ in range(max_x):
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
                # find the edge (==between None to not None and not None to None in candidate)
                start_x = None
                end_x = None
                for x in range(max_x):
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
                    # this means this edge goes all the way around
                    edge = Scan.Edge(0, edge_type, candidate)
                elif end_x is None:
                    # this means the edge ends at the end
                    edge = Scan.Edge(start_x, edge_type, candidate[start_x:max_x])
                elif end_x < start_x:  # NB: end cannot be < start unless wrapping is allowed
                    # this means the edge wraps
                    edge = Scan.Edge(start_x, edge_type, candidate[start_x:max_x] + candidate[0:end_x+1])
                else:
                    edge = Scan.Edge(start_x, edge_type, candidate[start_x:end_x+1])
            else:
                edge = None
            return edge

        # build the edges list
        edges = []
        for x, slice in enumerate(slices):
            for step in slice:
                if step.type != mode:
                    # not the step type we are looking for
                    continue
                if from_y is not None:
                    if step.where <= from_y[x]:
                        # too close to top
                        continue
                candidate = make_candidate(x, step.where, step.type)
                if candidate is not None:
                    edges.append(candidate)

        if self.logging:
            self._log('{}edges: {} edges'.format(context, len(edges)))
            for edge in edges:
                self._log('    {}'.format(edge))

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
                    log_edge(edge[2])

            if len(full_edges) == 0:
                # no edges detected!
                return [None for _ in range(max_x)], 'no edges'

            # extrapolate across any remaining gaps in the longest edge
            composed, reason = extrapolate(full_edges[0][2])
            if reason is not None:
                # failed
                return composed, reason

            # remove y 'steps'
            smoothed = composed  # ToDo: HACK-->smooth(composed, direction)

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

        direction = edges[0].type  # they must all be the same

        # make the edge
        edge, fail = compose(edges, direction)
        if self.logging:
            self._log('extent: {} (fail={})'.format(direction, fail))
            log_edge(edge)

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
            min_ring_size = max(Scan.MIN_PIXELS_PER_RING, self.cells[1])
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

        def make_extent(target, clean=False, context='') -> Scan.Extent:
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
        target, stretch_factor = self._project(self.centre_x, self.centre_y, blob_size)
        if target is None:
            # its been rejected
            return None

        # do the initial edge detection
        extent = make_extent(target, context='-warped')

        if extent.inner_fail is None and extent.outer_fail is None:
            flat, flat_stretch = self._flatten(target, extent)
            extent = make_extent(flat, clean=True, context='-flat')
            max_x, max_y = flat.size()
            return max_x, max_y, stretch_factor * flat_stretch, extent

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

    def _find_digits(self, extent: Extent) -> ([Digit], str):
        """ find the digits from an analysis of the given extent,
            returns a digit list and None if succeeded, or partial list and a reason if failed,
            """

        if self.logging:
            header = 'find_digits:'

        buckets = extent.buckets
        max_x, max_y = buckets.size()

        def show_options(options):
            """ produce a formatted string to describe the given digit options """
            msg = ''
            for option in options:
                msg = '{}, ({}, {})'.format(msg, option[0], vstr(option[1]))
            return msg[2:]

        # region generate likely digit choices for each x...
        inner = extent.inner
        outer = extent.outer
        slices = [[None, None] for _ in range(max_x)]
        for x in range(max_x):
            start_y = inner[x] + 1  # get past the white bullseye, so start at first black
            end_y   = outer[x]      # this is the first white after the inner black
            # scan down for lead/head/tail edges
            down_lead_at = None
            down_head_at = None
            down_tail_at = None
            for y in range(start_y, end_y):
                pixel = buckets.getpixel(x, y)
                if pixel == MID_LUMINANCE:
                    # treat like white everywhere
                    pixel = MIN_LUMINANCE  # ToDo: HACK-->MAX_LUMINANCE
                if pixel == MAX_LUMINANCE and down_lead_at is None:
                    # ignore until we see a black
                    continue
                if pixel == MAX_LUMINANCE and down_head_at is None:
                    # start of head
                    down_head_at = y
                    continue
                if pixel == MIN_LUMINANCE and down_lead_at is None:
                    # start of lead
                    down_lead_at = start_y  # use real start
                    continue
                if pixel == MIN_LUMINANCE and down_head_at is not None:
                    # end of head
                    down_tail_at = y
                    break
            if down_lead_at is None:
                # all white
                down_lead_at = start_y - 1
                down_head_at = start_y
                down_tail_at = end_y
            elif down_head_at is None:
                # all black
                down_head_at = end_y - 1
                down_tail_at = end_y
            elif down_tail_at is None:
                # white to the end
                down_tail_at = end_y
            # scan up for tail/head/lead edges
            up_head_at = None
            up_tail_at = None
            up_end_at = None
            for y in range(end_y - 1, start_y, -1):
                pixel = buckets.getpixel(x, y)
                if pixel == MID_LUMINANCE:
                    # treat like black everywhere
                    pixel = MIN_LUMINANCE
                if pixel == MAX_LUMINANCE and up_end_at is None:
                    # ignore until see a black
                    continue
                if pixel == MIN_LUMINANCE and up_end_at is None:
                    # end of tail
                    up_end_at = end_y - 1  # use real end
                    continue
                if pixel == MAX_LUMINANCE and up_tail_at is None:
                    # start of tail
                    up_tail_at = y + 1  # get back to the black
                    continue
                if pixel == MIN_LUMINANCE and up_tail_at is not None:
                    # end of head
                    up_head_at = y + 1  # get back to the white
                    break
            if up_end_at is None:
                # all white
                up_end_at = end_y
                up_tail_at = end_y - 1
                up_head_at = start_y
            elif up_tail_at is None:
                # all black
                up_tail_at = start_y + 1
                up_head_at = start_y
            elif up_head_at is None:
                # white to the end
                up_head_at = start_y
            # calculate ratios using average lead/head and head/tail edges
            head_at = (up_head_at + down_head_at) / 2
            tail_at = (up_tail_at + down_tail_at) / 2
            lead_length = head_at - down_lead_at
            head_length = tail_at - head_at
            tail_length = up_end_at - tail_at
            slices[x][0] = self.decoder.make_ratio(lead_length, head_length, tail_length)
            slices[x][1] = self.decoder.classify(slices[x][0])
        if self.logging:
            if header is not None:
                self._log(header)
                header = None
            self._log('    initial slices:')
            for x, (ratio, digit) in enumerate(slices):
                self._log('        {}: ratio={}, options={}'.format(x, ratio, show_options(digit)))
        # endregion

        # region repeat removing orphans, twins and triplets until nothing changes...
        orphans = 0
        twins = 0
        triplets = 0
        changes = True
        passes = 0
        if self.logging:
            sub_header = 'remove orphans, twins and triplets'
        while changes:
            passes += 1
            changes = False
            # adjust orphans by removing its first choice
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
                if this[0][0] != pred[0][0] and this[0][0] != succ[0][0]:
                    # got an orphan, remove top choice
                    if len(this) > 1:
                        # got choices, so remove one
                        removed.append(this[0])
                        del this[0]
                    else:
                        # no choices left, morph to one of our neighbours, which one?
                        removed.append(this[0])
                        this[0] = pred[0]
                if len(removed) > 0:
                    # we made a change
                    changes = True
                    orphans += 1
                    if self.logging:
                        if header is not None:
                            self._log(header)
                            header = None
                        if sub_header is not None:
                            self._log('    {}:'.format(sub_header))
                            sub_header = None
                        self._log('        pass {} at {}: (orphan) removed {} leaving {}'.
                                  format(passes, x, show_options(removed), show_options(this)))
            if changes:
                continue

            # no orphans left, adjust twins by making one of them into an orphan
            for x, (ratio, this) in enumerate(slices):
                if ratio is None:
                    continue
                ratio, pred = slices[(x - 1) % max_x]
                if ratio is None:
                    continue
                ratio, that = slices[(x + 1) % max_x]
                if ratio is None:
                    continue
                ratio, succ = slices[(x + 2) % max_x]
                if ratio is None:
                    continue
                removed = []
                while len(this) > 0 or len(that) > 0:
                    if this[0][0] == that[0][0] and this[0][0] != pred[0][0] and this[0][0] != succ[0][0]:
                        # got a twin, remove top choices
                        if len(this) > 1:
                            # there is a choice
                            removed.append(this[0])
                            del this[0]
                        elif len(that) > 1:
                            # there is a choice
                            removed.append(that[0])
                            del that[0]
                        else:
                            # no choices left, morph to one of our neighbours, which one?
                            removed.append(this[0])
                            this[0] = pred[0]
                    else:
                        # not or no longer a twin
                        break
                if len(removed) > 0:
                    # we made a change
                    changes = True
                    twins += 1
                    if self.logging:
                        if header is not None:
                            self._log(header)
                            header = None
                        self._log('        pass {} at {}: (twin) removed {} leaving {}'.
                                  format(passes, x, show_options(removed), show_options(this)))
            if changes:
                continue

            # no twins left, adjust triplets by making one of them into an orphan
            for x, (ratio, this) in enumerate(slices):
                if ratio is None:
                    continue
                ratio, far_left = slices[(x - 2) % max_x]
                if ratio is None:
                    continue
                ratio, left = slices[(x - 1) % max_x]
                if ratio is None:
                    continue
                ratio, right = slices[(x + 1) % max_x]
                if ratio is None:
                    continue
                ratio, far_right = slices[(x + 2) % max_x]
                if ratio is None:
                    continue
                removed = []
                while len(this) > 0 or len(left) > 0 or len(right) > 0:
                    if this[0][0] == left[0][0] and this[0][0] == right[0][0] \
                            and this[0][0] != far_left[0][0] and this[0][0] != far_right[0][0]:
                        # got an isolated triplet, remove top choice
                        if len(this) > 1:
                            # this has choices, remove top one
                            removed.append(this[0])
                            del this[0]
                        elif len(left) > 1:
                            # left has choices, remove top one
                            removed.append(left[0])
                            del left[0]
                        elif len(right) > 1:
                            # right has choices, remove top one
                            removed.append(right[0])
                            del right[0]
                        else:
                            # no choices left, morph to one of our neighbours, which one?
                            removed.append(this[0])
                            this[0] = far_left[0]
                    else:
                        # not or no longer a triplet
                        break
                if len(removed) > 0:
                    # we made a change
                    changes = True
                    triplets += 1
                    if self.logging:
                        if header is not None:
                            self._log(header)
                            header = None
                        self._log('        pass {} at {}: (triplet) removed {} leaving {}'.
                                  format(passes, x, show_options(removed), show_options(this)))

        if self.logging:
            if header is not None:
                self._log(header)
                header = None
            self._log('    {} passes, {} orphans adjusted, {} twins adjusted, {} triplets adjusted'.
                      format(passes, orphans, twins, triplets))
            self._log('    final slices:')
            for x, (ratio, digit) in enumerate(slices):
                self._log('        {}: ratio={}, options={}'.format(x, ratio, show_options(digit)))
        #endregion

        # region build digit list...
        digits = []
        last_digit = Scan.Digit(None, 0, 0)
        for x, (ratio, this) in enumerate(slices):
            if ratio is None:
                # this is junk - ignore it
                continue
            if last_digit.digit is None:
                # first digit
                last_digit.digit = this[0][0]
                last_digit.error = this[0][1][0]
                last_digit.samples = 1
            elif this[0][0] == last_digit.digit:
                # continue with this digit
                last_digit.error += this[0][1][0]  # accumulate error
                last_digit.samples += 1         # and sample count
            else:
                # save last digit
                last_digit.error /= last_digit.samples  # set average error
                digits.append(last_digit)
                # start a new digit
                last_digit = Scan.Digit(this[0][0], this[0][1][0], 1)
        # deal with last digit
        if len(digits) == 0:
            # its all the same digit - this must be junk
            last_digit.error /= last_digit.samples  # set average error
            digits = [last_digit]
        elif last_digit.digit == digits[0].digit:
            # its part of the first digit
            last_digit.error /= last_digit.samples  # set average error
            digits[0].error = (digits[0].error + last_digit.error) / 2
            digits[0].samples += last_digit.samples
        else:
            # its a separate digit
            last_digit.error /= last_digit.samples  # set average error
            digits.append(last_digit)

        if len(digits) < Scan.NUM_SEGMENTS:
            # ToDo: split large digits?
            reason = 'too few digits'
        elif len(digits) > Scan.NUM_SEGMENTS:
            # ToDo: combine small digits
            reason = 'too many digits'
        else:
            reason = None

        if self.logging:
            if header is not None:
                self._log(header)
                header = None
            self._log('    {} digits (fail reason {}):'.format(len(digits), reason))
            for x, digit in enumerate(digits):
                self._log('        {}: {}'.format(x, digit))
        # endregion

        if self.save_images:
            plot = self.transform.copy(buckets)
            lead_lines = []
            head_lines = []
            tail_lines = []
            for x, (_, slice) in enumerate(slices):
                ideal = self.decoder.to_ratio(slice[0][0])
                if ideal is None:
                    continue
                inner_average, outer_average = self._measure(extent, log=False)
                start_y = int(round(inner_average + 1))
                end_y = int(round(outer_average))
                span = end_y - start_y + 1
                lead_length = span * ideal.lead
                head_length = span * ideal.head
                tail_length = span * ideal.tail
                total = lead_length + head_length + tail_length  # this covers span and is in range 0..?
                scale = span / total  # so scale by this to get 0..span
                lead_length *= scale
                head_length *= scale
                tail_length *= scale
                lead_start = start_y
                head_start = int(round(start_y + lead_length))
                tail_start = int(round(head_start + head_length))
                tail_end = end_y
                lead_lines.append((x, lead_start, x, head_start - 1))
                head_lines.append((x, head_start, x, tail_start - 1))
                tail_lines.append((x, tail_start, x, tail_end - 1))
            plot = self._draw_lines(plot, lead_lines, Scan.GREEN)
            plot = self._draw_lines(plot, head_lines, Scan.RED)
            plot = self._draw_lines(plot, tail_lines, Scan.GREEN)
            self._unload(plot, '07-slices')

        return digits, reason

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

            digits, reason = self._find_digits(extent)
            if reason is not None:
                # we failed to find segment edges
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

    def _draw_extent(self, extent: Extent, target, bleed):
        """ make the area outside the inner and outer edges on the given target visible """

        max_x, max_y = target.size()
        inner = extent.inner
        outer = extent.outer

        inner_lines = []
        outer_lines = []
        for x in range(max_x):
            if inner is not None and inner[x] is not None:
                inner_lines.append((x, 0, x, inner[x] - 1))  # inner edge is on the first black, -1 to get to white
            if outer is not None and outer[x] is not None:
                outer_lines.append((x, outer[x], x, max_y - 1))  # outer edge is on first white
        target = self._draw_lines(target, inner_lines, colour=Scan.RED, bleed=bleed)
        target = self._draw_lines(target, outer_lines, colour=Scan.RED, bleed=bleed)

        return target

    def _draw_edges(self, edges, target: frame.Frame, extent: Extent=None, bleed=0.5):
        """ draw the edges and the extent on the given target image """

        falling_edges = edges[0]
        rising_edges = edges[1]

        if extent is None:
            plot = target
        else:
            # mark the image area outside the inner and outer extent
            plot = self._draw_extent(extent, target, bleed=0.8)

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

        return plot
    # endregion

