
import codec
import ring
import angle
import frame
import contours
import const
import structs
import utils
import cluster
import math
import os
from typing import List
import traceback
import time
import numpy as np

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

    # region target shape...
    NUM_RINGS = ring.Ring.NUM_RINGS  # total number of rings in the whole code (ring==cell in height)
    NUM_SEGMENTS = codec.Codec.DIGITS  # total number of segments in a ring (segment==cell in length)
    # endregion

    # region image 'segment' and 'ring' constraints...
    # a 'segment' is the angular division in a ring,
    # a 'ring' is a radial division,
    # a 'cell' is the intersection of a segment and a ring
    # these constraints set minimums that override the cells() property given to Scan
    MIN_PIXELS_PER_CELL = 4  # min pixels making up a cell length
    MIN_PIXELS_PER_RING = 4  # min pixels making up a ring width
    # endregion

    # region Tuning constants...
    BLOB_RADIUS_STRETCH = 1.2  # how much to stretch blob radius to ensure always cover everything when projecting
    MIN_CONTRAST = 0.35  # minimum luminance variation of a valid blob projection relative to the max luminance
    THRESHOLD_WIDTH = 8  # the fraction of the projected image width to use as the integration area when binarizing
    THRESHOLD_HEIGHT = 3.5  # the fraction of the projected image height to use as the integration area (None=as width)
    THRESHOLD_BLACK = 5  # the % below the average luminance in a projected image that is considered to be black
    THRESHOLD_WHITE = 5  # the % above the average luminance in a projected image that is considered to be white
    MIN_EDGE_SAMPLES = 3  # minimum samples in an edge to be considered a valid edge
    INNER_EDGE_GAP = 1.0  # fraction of inner edge y co-ord to add to inner edge when looking for the outer edge
    MAX_NEIGHBOUR_ANGLE_INNER = 0.4  # ~=22 degrees, tan of the max acceptable angle when joining inner edge fragments
    MAX_NEIGHBOUR_ANGLE_OUTER = 0.9  # ~=42 degrees, tan of the max acceptable angle when joining outer edge fragments
    MAX_NEIGHBOUR_HEIGHT_GAP = 1  # max x or y jump allowed when following an edge
    MAX_NEIGHBOUR_LENGTH_JUMP = 10  # max x jump, in pixels, between edge fragments when joining (arbitrary)
    MAX_NEIGHBOUR_HEIGHT_JUMP = 4  # max y jump, in pixels, between edge fragments when joining (arbitrary)
    MAX_NEIGHBOUR_OVERLAP = 4  # max edge overlap, in pixels, between edge fragments when joining (arbitrary)
    MAX_EDGE_GAP_SIZE = 4 / NUM_SEGMENTS  # max gap tolerated between edge fragments (as fraction of image width)
    SMOOTHING_WINDOW = 8  # samples in the moving average (we average the centre, so the full window is +/- this)
    # endregion

    # endregion

    def __init__(self, codec, frame, transform, cells=(MIN_PIXELS_PER_CELL, MIN_PIXELS_PER_RING),
                 video_mode=const.VIDEO_FHD, proximity=const.PROXIMITY_FAR,
                 debug=const.DEBUG_NONE, log=None):
        """ codec is the codec instance defining the code structure,
            frame is the frame instance containing the image to be scanned,
            transform is the class implmenting various image transforms (mostly for diagnostic purposes),
            cells is the angular/radial resolution to use,
            video_mode is the maximum resolution to work at, the image is downsized to this if required,
            proximity is the contour detection integration area parameter
            """

        # set debug options
        if debug == const.DEBUG_IMAGE:
            self.show_log = False
            self.save_images = True
        elif debug == const.DEBUG_VERBOSE:
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
        # NB: stretching radial steps by BLOB_RADIUS_STRETCH to ensure each ring achieves the min size we want
        self.radial_steps = int(round(Scan.NUM_RINGS * Scan.BLOB_RADIUS_STRETCH *
                                      max(self.cells[1], Scan.MIN_PIXELS_PER_RING)))

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

        # needed for finding digits (must be last)
        self.cluster = cluster.Cluster(self)

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
        params.black_threshold = const.BLACK_LEVEL[self.proximity]
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
                plot = self.transform.label(plot, blob, const.GREEN)
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
                pixel_xLyL = const.MIN_LUMINANCE
            if pixel_xLyH is None:
                pixel_xLyH = const.MIN_LUMINANCE
            if pixel_xHyL is None:
                pixel_xHyL = const.MIN_LUMINANCE
            if pixel_xHyH is None:
                pixel_xHyH = const.MIN_LUMINANCE
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
        code = self.original.instance().new(self.angle_steps, limit_radius, const.MIN_LUMINANCE)
        min_level = const.MAX_LUMINANCE
        max_level = const.MIN_LUMINANCE
        for radius in range(limit_radius):
            for angle in range(self.angle_steps):
                degrees = angle * angle_delta
                x, y = self.angle_xy(degrees, radius)
                if x is not None:
                    c = get_pixel(x, y)  # centre_x/y a in here
                    if c > const.MIN_LUMINANCE:
                        code.putpixel(angle, radius, c)
                        max_level = max(c, max_level)
                        min_level = min(c, min_level)

        # chuck out targets that do not have enough black/white contrast
        contrast = (max_level - min_level) / const.MAX_LUMINANCE
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
            blob = self.transform.label(blob, k, const.RED)
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
                                if left == const.MIN_LUMINANCE:
                                    h_black += 1
                                elif left == const.MAX_LUMINANCE:
                                    h_white += 1
                                else:
                                    h_grey += 1
                        elif False:  # ToDo: makes it unstable-->left != this and this != right and left != right and this != const.MID_LUMINANCE:
                            # all different and middle not grey, make middle grey
                            buckets.putpixel(x, y, const.MID_LUMINANCE)
                            pass_changes += 1
                            h_grey += 1
                        elif above == below:
                            # only look for vertical when there is no horizontal candidate, else it can oscillate
                            if this != above:
                                # got a vertical loner
                                # this condition is lower priority than above
                                buckets.putpixel(x, y, above)
                                pass_changes += 1
                                if above == const.MIN_LUMINANCE:
                                    v_black += 1
                                elif above == const.MAX_LUMINANCE:
                                    v_white += 1
                                else:
                                    v_grey += 1
                        elif above != this and this != below and above != below and this != const.MID_LUMINANCE:
                            # all different and middle not grey, make middle grey
                            buckets.putpixel(x, y, const.MID_LUMINANCE)
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

    def _slices(self, buckets: frame.Frame) -> List[List[structs.Step]]:
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
                    slices[x].append(structs.Step(y-1, const.FALLING, last_pixel, pixel))
                    transitions += 1
                elif pixel > last_pixel:
                    # rising step
                    slices[x].append(structs.Step(y, const.RISING, last_pixel, pixel))
                    transitions += 1
                last_pixel = pixel
            if transitions == 0:
                # this probably means a big pulse has merged with the inner and the outer edge,
                if last_pixel == const.MAX_LUMINANCE:
                    # its all white - not possible?
                    # create a rising step at 0 and a falling step at max_y
                    slices[x].append(structs.Step(0, const.RISING, const.MIN_LUMINANCE, const.MAX_LUMINANCE))
                    slices[x].append(structs.Step(max_y - 1, const.FALLING, const.MAX_LUMINANCE, const.MIN_LUMINANCE))
                    if self.logging:
                        self._log('slices: {}: all white and no transitions, creating {}, {}'.
                                  format(x, slices[x][-2], slices[x][-1]))
                elif last_pixel == const.MIN_LUMINANCE:
                    # its all black - not possible?
                    # create a falling step at 0 and a rising step at max_y
                    slices[x].append(structs.Step(0, const.FALLING, const.MAX_LUMINANCE, const.MIN_LUMINANCE))
                    slices[x].append(structs.Step(max_y - 1, const.RISING, const.MIN_LUMINANCE, const.MAX_LUMINANCE))
                    if self.logging:
                        self._log('slices: {}: all black and no transitions, creating {}, {}'.
                                  format(x, slices[x][-2], slices[x][-1]))
                else:
                    # its all grey - this means all pixels are nearly the same in the integration area
                    # treat as if all white
                    # create a rising step at 0 and a falling step at max_y
                    slices[x].append(structs.Step(0, const.RISING, const.MIN_LUMINANCE, const.MAX_LUMINANCE))
                    slices[x].append(structs.Step(max_y - 1, const.FALLING, const.MAX_LUMINANCE, const.MIN_LUMINANCE))
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

    def _edges(self, slices: List[List[structs.Step]], mode, max_y, from_y:[int]=None) -> [structs.Edge]:
        """ build a list of falling or rising edges of our target, mode is FALLING or RISING,
            iff from_y is given it is a list of the starting y co-ords for each x,
            FALLING is optimised for detecting the inner white->black edge and
            RISING is optimised for detecting the outer black->white edge,
            returns the falling or rising edges list in increasing co-ordinate order,
            an 'edge' here is a sequence of connected rising or falling Steps,
            """

        if mode == const.FALLING:
            # we want any fall to min, do not want a half fall to mid
            # so max to min or mid to min qualifies, but max to mid does not,
            # we achieve this by only considering falling edges where the 'to' pixel is min
            context = 'falling-'
        elif mode == const.RISING:
            # we want any rise, including a half rise
            # so min to mid or min to max qualifies, but mid to max does not,
            # we achieve this by only considering rising edges where the 'from' pixel is min
            context = 'rising-'
        else:
            raise Exception('_edges: mode must be {} or {} not {}'.format(const.FALLING, const.RISING, mode))

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
                        if step.type == const.FALLING:
                            to_pixel = step.to_pixel
                            if to_pixel == const.MID_LUMINANCE:
                                to_pixel = treat_grey_as
                            if to_pixel == const.MIN_LUMINANCE:
                                # got a qualifying falling step
                                pass
                            else:
                                # ignore this step
                                continue
                        else:  # step.type == const.RISING
                            to_pixel = step.to_pixel
                            if to_pixel == const.MID_LUMINANCE:
                                to_pixel = treat_grey_as
                            if to_pixel == const.MAX_LUMINANCE:
                                # got a qualifying rising step
                                pass
                            else:
                                # ignore this step
                                continue
                        gap = utils.wrapped_gap(y, step.where, max_y)
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
                    return structs.Edge(0, edge_type, sequence, treat_grey_as)
                elif end_x is None:
                    # this means the edge ends at the end
                    return structs.Edge(start_x, edge_type, sequence[start_x:max_x], treat_grey_as)
                elif end_x < start_x:  # NB: end cannot be < start unless wrapping is allowed
                    # this means the edge wraps
                    return structs.Edge(start_x, edge_type, sequence[start_x:max_x] + sequence[0:end_x+1], treat_grey_as)
                else:
                    # normal edge away from either extreme
                    return structs.Edge(start_x, edge_type, sequence[start_x:end_x+1], treat_grey_as)

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
                    tail_samples = 0
                    for x in range(max_x):
                        forward_sample = forwards[x]
                        backward_sample = backwards[x]
                        if forward_sample is not None and backward_sample is not None:
                            # got an overlap, add to tails
                            backward_tail[x] = backward_sample
                            forward_tail[x] = forward_sample
                            tail_samples += 1
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
                        if main_samples >= Scan.MIN_EDGE_SAMPLES:
                            edges.append(make_edge(main_sequence))
                        if tail_samples >= Scan.MIN_EDGE_SAMPLES:
                            edges.append(make_edge(forward_tail))
                            edges.append(make_edge(backward_tail))
                    elif tail_samples >= Scan.MIN_EDGE_SAMPLES:  # and main_samples == 0
                        # this means both forwards and backwards go right round, they must be the same in this case
                        edges.append(make_edge(forward_tail))
                    else:
                        # fragments too small - is this possible?
                        pass
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
        for treat_grey_as in [const.MIN_LUMINANCE, const.MAX_LUMINANCE]:
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

    def _extent(self, max_x, edges: [structs.Edge]) -> ([int], str):
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
            trimmed_edge = structs.Edge(edge.where, edge.type, edge.samples.copy(), edge.grey_as)
            trimmed_at = []
            for x, samples in enumerate(overlaps):
                if samples is None:
                    continue
                full_edge_y, edge_y = samples
                if edge_y > full_edge_y:
                    if direction == const.RISING:
                        # take out the full_edge sample - easy case
                        trimmed_full_edge[x] = None
                        trimmed_at.append(x)  # note where we changed it for clean-up later
                    else:
                        # take out the edge sample
                        if not remove_sample(x, trimmed_edge):
                            return None
                elif edge_y < full_edge_y:
                    if direction == const.FALLING:
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
                    returns a list of places (x co-ords) that was cleaned out
                    """
                if trimmed_full_edge[x % max_x] is not None:
                    # full edge has not been tweaked here, so nothing to clean
                    return []
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
                    return residue
                return []

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
                                reason = 'edge gap: {}+'.format(max_gap)
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

            if direction == const.FALLING:
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

        def make_extent(target, clean=True, context='') -> structs.Extent:
            """ binarize and inner/outer edge detect the given image """

            if self.logging:
                self._log('identify: phase{}'.format(context))

            max_x, max_y = target.size()

            buckets = self._binarize(target,
                                     width=Scan.THRESHOLD_WIDTH, height=Scan.THRESHOLD_HEIGHT,
                                     black=Scan.THRESHOLD_BLACK, white=Scan.THRESHOLD_WHITE,
                                     clean=clean, suffix=context)
            slices = self._slices(buckets)
            falling_edges = self._edges(slices, const.FALLING, max_y)
            inner, inner_fail = self._extent(max_x, falling_edges)
            if inner_fail is None:
                from_y = [y + (y * Scan.INNER_EDGE_GAP) for y in inner]
                rising_edges = self._edges(slices, const.RISING, max_y, from_y=from_y)
                outer, outer_fail = self._extent(max_x, rising_edges)
            else:
                rising_edges = None
                outer = None
                outer_fail = None

            extent = structs.Extent(inner=inner, inner_fail=inner_fail,
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

    def _measure(self, extent: structs.Extent, stretch_factor=1, log=True):
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

    def _find_codes(self) -> ([structs.Target], frame.Frame):
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
                    rejects.append(structs.Reject(self.centre_x, self.centre_y, blob_size, blob_size*4, reason))
                continue

            digits, reason = self.cluster.find_all_digits(extent)
            if reason is None:
                digits, reason = self.cluster.find_best_digits(digits, extent)
            if reason is not None:
                # we failed to find required digits
                if self.save_images:
                    # add to reject list for labelling on the original image
                    rejects.append(structs.Reject(self.centre_x, self.centre_y, blob_size, blob_size*4, reason))
                continue

            result = self.cluster.decode_digits(digits)
            target_size, _ = self._measure(extent, stretch_factor)
            targets.append(structs.Target(self.centre_x, self.centre_y, blob_size, target_size, result))

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
                labels = self.transform.label(labels, (x, y, blob_size), const.BLUE)
                # show reject reason
                labels = self.transform.label(labels, (x, y, target_size), const.RED,
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
            numbers.append(structs.Detection(result, self.centre_x, self.centre_y, target_size, blob_size))

            if self.save_images:
                detection = numbers[-1]
                result = detection.result
                if result.number is None:
                    colour = const.RED
                    label = 'invalid ({:.4f})'.format(result.doubt)
                else:
                    colour = const.GREEN
                    label = 'number is {} ({:.4f})'.format(result.number, result.doubt)
                # draw the detected blob in blue
                k = (detection.centre_x, detection.centre_y, detection.blob_size)
                detections = self.transform.label(detections, k, const.BLUE)
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
        image = self._draw_lines(image, lines, const.PINK)
        lines = []
        for y in range(10, max_y, 10):
            lines.append([0, y, 1, y])
            lines.append([max_x - 2, y, max_x - 1, y])
        image = self._draw_lines(image, lines, const.PINK)

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

    def _draw_blobs(self, source, blobs: List[tuple], colour=const.GREEN):
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

    def _draw_plots(self, source, plots_x=None, plots_y=None, colour=const.RED, bleed=0.5):
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

    def _draw_lines(self, source, lines, colour=const.RED, bleed=0.5, h_wrap=False, v_wrap=False):
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

    def _draw_extent(self, extent: structs.Extent, target, bleed=0.8):
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
        target = self._draw_lines(target, inner_lines, colour=const.RED, bleed=bleed)
        target = self._draw_lines(target, outer_lines, colour=const.RED, bleed=bleed)

        target = self._draw_plots(target, plots_x=[[0, inner]], colour=const.RED, bleed=bleed/2)
        target = self._draw_plots(target, plots_x=[[0, outer]], colour=const.RED, bleed=bleed/2)

        return target

    def _draw_edges(self, edges, target: frame.Frame, extent: structs.Extent=None, bleed=0.5):
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
        plot = self._draw_plots(plot, plots_x=falling_points, colour=const.GREEN, bleed=bleed)
        plot = self._draw_plots(plot, plots_x=rising_points, colour=const.BLUE, bleed=bleed)

        if extent is not None:
            # mark the image area outside the inner and outer extent
            plot = self._draw_extent(extent, plot, bleed=0.8)

        return plot

    # endregion
