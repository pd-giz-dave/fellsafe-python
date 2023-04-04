""" a module to detect corners in a grayscale image using the FAST algorithm,
    see https://en.wikipedia.org/wiki/Features_from_accelerated_segment_test
"""

# 4/4/23 DCN: An experiment to see if it useful - no its not in my context

import utils
import const
import shapes
import canvas

# neighbour brightness modes of a corner compared to its centre
DARKER   = -1
BRIGHTER = +1
SAME     = 0

class Params:
    file      = None   # file name of originating source image (for diagnostic naming purposes only)
    source    = None   # the source greyscale image buffer
    blurred   = None   # the blurred greyscale image buffer
    blur      = 3      # blur kernel size, 0 == do not blur
    threshold = 20     # minimum luminance variation required for a point to qualify as a corner
    radius    = 3      # radius of circle to use (must be 3 or more)
    window    = (0.7,  # minimum circle arc that must be contiguously darker or brighter than the centre,
                       # must be greater than a half and less than one
                0.8)   # maximum circle arc that can be contiguously darker or brighter than the centre,
                       # must be greater than min_arc and less than one
    distance  = 1.5    # minimum distance between corners in units of radius

def make_points(radius):
    """ make our circle points in the order required by the FAST algorithm (0..N clockwise from 12 o'clock) """
    # get the Bresenham points for the given radius
    points = shapes.circumference(0, 0, radius)
    # points is random from our pov, put into quadrant order
    quadrants = [[] for _ in range(4)]
    for point in points:
        if point[0] >= 0 and point[1] < 0:
            quadrants[0].append(point)
        elif point[0] > 0 and point[1] >= 0:
            quadrants[1].append(point)
        elif point[0] <= 0 and point[1] > 0:
            quadrants[2].append(point)
        else:
            quadrants[3].append(point)
    # put quadrants into order
    quadrants[0].sort(key=lambda k: (k[0], k[1]))
    quadrants[1].sort(key=lambda k: (-k[0], k[1]))
    quadrants[2].sort(key=lambda k: (-k[0], -k[1]))
    quadrants[3].sort(key=lambda k: (k[0], -k[1]))
    # put quadrants into our required order
    points = quadrants[0] + quadrants[1] + quadrants[2] + quadrants[3]
    return points

def find_corners(buffer, threshold=20, radius=3, window=(0.7, 0.8), logger=None):
    """ find corners in the given gray scale image buffer using the FAST algorithm,
        threshold is the luminance difference required of neighbours from the centre,
        radius is the radius of the neighbouring circle pixels (must be 3 or more),
        window is the min/max circle arc that must/can be contiguously darker or brighter (must be > 0.5 and < 1),
        returns a list of x,y co-ordinates, a neighbour mode (dark==-1 or bright==+1) and a strength,
        the strength is the sum of the neighbour luminance difference to the centre within the window,
        the returned list is in x then y co-ordinate order (ie. x=0..N, y=0..N within x),
        corners within radius of the image edge are not detected,
        see https://en.wikipedia.org/wiki/Features_from_accelerated_segment_test for the algorithm description
        """

    if radius < 3:
        raise Exception('radius must be at least 3, not {}'.format(radius))
    if window[0] < 0.5 or window[0] > 1.0 or window[1] < window[0] or window[1] > 1.0:
        raise Exception('window arcs must be between 0.5 and 1.0, not {}'.format(window))

    max_x, max_y = canvas.size(buffer)
    if max_x <= 2 * radius or max_y <= 2 * radius:
        # image too small to be meaningful
        return []

    def check_pixel(buffer, x, y, offset, limits):
        """ check if the pixel in buffer at x,y plus offset is within the limits low/high range,
            returns 0 if within, -1 if below or +1 if above, also returns the pixel to/from limit difference
            the pixel addressed must be within the buffer
            """
        pixel = buffer[y + offset[1], x + offset[0]]
        if pixel < limits[0]:
            return DARKER, limits[0] - pixel
        elif pixel > limits[1]:
            return BRIGHTER, pixel - limits[1]
        else:
            return SAME, 0

    # prepare context
    circle = make_points(radius)  # get points in a circle of the required radius
    points = len(circle)
    top_point = 0            # used to test 12 o'clock and 6 o'clock pixels for shortcut
    mid_point = points >> 1  # ..
    min_contiguous = int(points * window[0])  # need at least this (else it cannot be a corner)
    max_contiguous = int(points * window[1])  # but not more than this (else its a blob not a corner)

    # Loop through each pixel in the image (except a radius border)
    # NB: outer loop is x, inner is y, this is important to meet the sorting requirements of the returned list
    corners = []  # list of x,y co-ordinates that are corners
    for x in range(radius, max_x - radius):
        for y in range(radius, max_y - radius):
            pixel  = int(buffer[y, x])
            limits = (pixel - threshold, pixel + threshold)
            # check the test-points first (need 3 out of 4)
            pixel_first, _ = check_pixel(buffer, x, y, circle[0], limits)
            pixel_middle, _ = check_pixel(buffer, x, y, circle[mid_point], limits)
            if pixel_first == 0 and pixel_middle == 0:
                # neither distinguishable from centre pixel, so not a corner
                continue
            # got a possible candidate
            pixels = [None for _ in range(points)]
            brighter = 0
            darker = 0
            same = 0
            for p, point in enumerate(circle):
                pixel = check_pixel(buffer, x, y, point, limits)
                if pixel[0] == DARKER:
                    darker += 1
                elif pixel[0] == BRIGHTER:
                    brighter += 1
                else:
                    same += 1
                pixels[p] = pixel
            if darker < min_contiguous and brighter < min_contiguous:
                # not enough samples different, so not a candidate
                continue
            if darker > brighter:
                # looking for darker window
                require = DARKER
            elif brighter > darker:
                # looking for brighter window
                require = BRIGHTER
            else:
                # both the same, so ambiguous, discount it
                continue
            # now find the contiguous cells (==longest sequence between a leading edge and a trailing edge)
            best_span = 0
            best_strength = 0
            this_span = 0
            this_strength = 0
            first_span = None
            first_strength = 0
            # look for first leading edge
            previous_mode, _ = pixels[-1]  # pixels represent a loop, so previous to the first is the last
            for p in range(len(pixels)):
                mode, strength = pixels[p]
                if mode == require and previous_mode != require:
                    # found a leading edge, start counting
                    this_span = 1
                    this_strength = strength
                elif mode != require and previous_mode == require:
                    # found a trailing edge, is the leading->trailing span better than previous
                    if this_span > best_span:
                        # this is the first or better
                        best_span = this_span
                        best_strength = this_strength
                    # note first span for the wrapping case at the end
                    if first_span is None:
                        # not noted yet, note it now
                        first_span = this_span
                        first_strength = this_strength
                    # start again
                    this_span = 0
                    this_strength = 0
                elif mode == require and previous_mode == require:
                    # we're continuing in this span (or its the first sample and the last is also require, ie. wrapping)
                    this_span += 1
                    this_strength += strength
                else:  # mode != require and previous_mode != require
                    # we're continuing in a hole, do nothing
                    pass
                previous_mode = mode
            # handle edge case
            if this_span > 0:
                # this means we ended inside a span, there are two cases:
                #   1. it ends here (first pixel is not require)
                #   2. it wraps to the beginning (first pixel is require)
                if pixels[0][0] != require:
                    # case 1 - it ends here, easy, do nothing
                    pass
                else:
                    # case 2 - it wraps, append the first span (if there is one)
                    if first_span is not None:
                        this_span += first_span
                        this_strength += first_strength
                if this_span > best_span:
                    # this final one is the best so far
                    best_span = this_span
                    best_strength = this_strength
            if best_span < min_contiguous:
                # not enough contiguous points, so not a corner
                continue
            if best_span > max_contiguous:
                # too many contiguous points, so not a corner
                continue
            # found a corner
            corners.append(((x, y), require, int(best_strength / best_span)))

    # NB: corners are in x then y co-ordinate order, there can only be one corner per x,y pair
    return corners

def filter_corners(corners, distance=3):
    """ reduce corners within distance of each other into just the strongest one,
        the strongest is that with the biggest luminance difference between the centre and its perimeter,
        returns the reduced corner list,
        the provided corner list must be in x then y co-ordinate order
        """

    def add_best(best, next_corner, distance):
        """ if given next_corner is within distance of something already in best and its stronger
            than what's in best, then replace best with this, otherwise discard it,
            best is assumed to be ordered by x then y co-ordinates and the new corner must have x,y
            co-ordinates higher than anything already in best.
            NB: such a replacement may lead to the removal of several previous bests due to the new one
                falling between previously adequately separated ones
            """
        if len(best) == 0:
            # just add the first one
            best.append(corner)
            return best
        # we scan backwards examining all previous bests that are within distance of the new corner,
        # any not as good are dumped from best, if all are better we discard the new corner
        this_corner, this_mode, this_strength = next_corner
        distance *= distance  # squared 'cos that's what utils.distance returns, the square of the separation
        better = True  # gets set False if we're beaten by any best candidates
        for candidate in range(len(best)-1, -1, -1):
            best_corner, best_mode, best_strength = best[candidate]
            gap = utils.distance(best_corner, this_corner)
            if gap < distance:
                # this one too close
                if this_strength > best_strength:
                    # we're better than what's in best, chuck best
                    del best[candidate]
                else:
                    # we're no better than something that is there already, discard the new corner and stop looking
                    better = False
                    break
            else:
                # have we gone back far enough to ensure we've found all the close ones?
                x_gap = utils.distance((best_corner[0], 0), (this_corner[0], 0))
                if x_gap < distance:
                    # nope, carry on
                    continue
                break
        if better:
            # we're better than what's there already, so put us on the end
            best.append(next_corner)
        return best

    filtered = []
    for corner in corners:
        add_best(filtered, corner, distance)
    return filtered

def draw_corners(buffer, corners, radius, logger, file='corners', grid=False):
    """ draw a small circle around the detected corners in the given buffer """
    image = canvas.colourize(buffer)
    if grid:
        canvas.grid(image, 10, 10, const.BLUE)
    for corner, mode, _ in corners:
        if mode == DARKER:
            colour = const.RED
        else:
            colour = const.GREEN
        canvas.circle(image, corner, radius, colour)
    logger.draw(image, file=file)

def _test(src, params, size, logger):
    source = canvas.prepare(src, size, logger)
    if source is None:
        return
    params.source = source
    params.blurred = canvas.blur(source, params.blur)
    corners = find_corners(source, params.threshold, params.radius, params.window, logger=logger)
    logger.log('Found {} unfiltered corners with threshold={}, radius={}, window={}'.
               format(len(corners), params.threshold, params.radius, params.window))
    draw_corners(source, corners, params.radius, logger, file='corners-unfiltered')

    distance = params.distance * params.radius
    params.corners = filter_corners(corners, distance)
    logger.log('After filtering found {} corners with distance={}'.format(len(params.corners), distance))
    draw_corners(source, params.corners, params.radius, logger, 'corners')
    return params


if __name__ == "__main__":
    """ testing """
    #src = "/home/dave/precious/fellsafe/fellsafe-image/source/kilo-codes/codes/test-alt-bits.png"
    src = '/home/dave/precious/fellsafe/fellsafe-image/media/kilo-codes/kilo-codes-distant-150-257-263-380-436-647-688-710-777.jpg'

    logger = utils.Logger('corners.log', 'corners/{}'.format(utils.image_folder(src)))

    params = Params()
    _test(src, params, size=const.VIDEO_2K, logger=logger)
