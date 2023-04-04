""" wrapper around cv2 to hide it from everyone
    these functions manipulate a 2D array of luminance values,
    it uses the opencv library to read/write images and numpy arrays to modify images at the pixel level
    """

import numpy as np
import cv2
import const

""" WARNING
    cv2 and numpy's co-ordinates are backwards from our pov, the 'x' co-ordinate are columns and 'y' rows.
    The functions here use 'x' then 'y' parameters, swapping as required when dealing with cv2 and numpy arrays.
    NB: The final implementation will not use cv2 (its too big) so its use here is purely a diagnostic convenience.
        Significant algorithms are implemented DIY. 
    """

def copy(buffer):
    """ return a copy of the given buffer """
    return np.copy(buffer)

def new(width, height, luminance=const.MIN_LUMINANCE):
    """ prepare a new buffer of the given size and luminance """
    return np.full((height, width), luminance, dtype=np.uint8)  # NB: numpy arrays follow cv2 conventions

def size(buffer):
    """ return the x,y size of the given buffer """
    max_x = buffer.shape[1]  # NB: cv2 x, y are reversed
    max_y = buffer.shape[0]  # ..
    return max_x, max_y

def load(image_file):
    """ load a buffer from an image file as a grey scale image and return it """
    return cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

def unload(buffer, image_file):
    """ unload the given buffer to a PNG image file """
    cv2.imwrite(image_file, buffer)

def downsize(buffer, new_size):
    """ downsize the image such that either its width or height is at most that given,
        the aspect ratio is preserved, its a no-op if image already small enough,
        returns a new buffer of the new size,
        this is purely a diagnostic aid to simulate low-resolution cameras
        """
    width, height = size(buffer)
    if width <= new_size or height <= new_size:
        # its already small enough
        return buffer
    if width > height:
        # bring height down to new size
        new_height = new_size
        new_width = int(width / (height / new_size))
    else:
        # bring width down to new size
        new_width = new_size
        new_height = int(height / (width / new_size))
    return cv2.resize(buffer, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

def prepare(src, size, logger=None):
    """ load and downsize an image """
    if logger is not None:
        logger.log("Preparing image of size {} from {}".format(size, src))
    source = load(src)
    if source is None:
        if logger is not None:
            logger.log('Cannot load {}'.format(src))
        return None
    # Downsize it (to simulate low quality smartphone cameras)
    return downsize(source, size)

def integrate(buffer, box=None):
    """ generate the integral of the given box within the given image buffer """

    if box is None:
        box_min_x, box_min_y = (0, 0)
        box_max_x, box_max_y = size(buffer)
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
                acc = int(buffer[y, x])  # start a new row (# NB: need int 'cos source is uint8, i.e. unsigned)
            else:
                acc += int(buffer[y, x])  # extend existing row (# NB: need int 'cos source is uint8, i.e. unsigned)
            ix = x - box_min_x
            iy = y - box_min_y
            if iy == 0:
                integral[iy][ix] = acc  # start a new column
            else:
                integral[iy][ix] = acc + integral[iy - 1][ix]  # extend existing column

    return integral

def blur(buffer, kernel_size):
    """ return the blurred image as a mean blur over the given kernel size """
    # we do this by integrating then calculating the average via integral differences,
    # this means we only visit each pixel once irrespective of the kernel size
    if kernel_size is None or kernel_size < 2:
        return buffer  # pointless

    # integrate the image
    integral = integrate(buffer)

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

def get_box_corners(buffer, box=None):
    """ get the co-ordinates of a box in the buffer """
    buffer_min_x, buffer_min_y = (0, 0)
    buffer_max_x, buffer_max_y = size(buffer)
    if box is None:
        box_min_x = buffer_min_x
        box_max_x = buffer_max_x
        box_min_y = buffer_min_y
        box_max_y = buffer_max_y
    else:
        box_min_x = box[0][0]
        box_max_x = box[1][0] + 1
        box_min_y = box[0][1]
        box_max_y = box[1][1] + 1
    return (box_min_x, box_min_y), (box_max_x, box_max_y)

def binarize(buffer, box=None, width: float=8, height: float=None, black: float=15, white: float=None):
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
    (box_min_x, box_min_y), (box_max_x, box_max_y) = get_box_corners(buffer, box)
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
    integral = integrate(buffer, box)

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
            src = int(buffer[box_min_y + y, box_min_x + x]) * count  # NB: need int 'cos source is uint8 (unsigned)
            if src >= (acc * white_limit):
                binary[y, x] = const.MAX_LUMINANCE
            elif src <= (acc * black_limit):
                binary[y, x] = const.MIN_LUMINANCE
            else:
                binary[y, x] = const.MID_LUMINANCE
    # endregion

    return binary

def extract(image, box: ((int, int), (int, int))):
    """ extract a box from the given image """
    tl_x, tl_y = box[0]
    br_x, br_y = box[1]
    width  = br_x - tl_x + 1
    height = br_y - tl_y + 1
    buffer = np.zeros((height, width), np.uint8)
    for x in range(width):
        for y in range(height):
            buffer[y, x] = image[tl_y + y, tl_x + x]
    return buffer

def histogram(buffer, box=None):
    """ build a histogram of luminance values in the given buffer (or the box within it) """
    (tl_x, tl_y), (br_x, br_y) = get_box_corners(buffer, box)
    width  = br_x - tl_x
    height = br_y - tl_y
    samples = [0 for _ in range(const.MAX_LUMINANCE+1)]
    for x in range(width):
        for y in range(height):
            samples[buffer[tl_y + y, tl_x + x]] += 1
    return samples

def upsize(buffer, scale: float):
    """ return a buffer that is scale times bigger in width and height than that given,
        the contents of the given buffer are also scaled by interpolating neighbours (using makepixel),
        scale must be positive and greater then 1,
        each source pixel is considered to consist of scale * scale sub-pixels, each destination pixel
        is one sub-pixel and is constructed from an interpolation of a 1x1 source pixel area centred on
        the sub-pixel
        """
    if scale <= 1:
        return buffer
    max_x, max_y = size(buffer)
    width  = int(round(max_x * scale))
    height = int(round(max_y * scale))
    upsized = new(width, height)
    for dest_x in range(width):
        src_x = dest_x / scale
        for dest_y in range(height):
            src_y = dest_y / scale
            pixel = makepixel(buffer, src_x, src_y)
            upsized[dest_y, dest_x] = pixel
    return upsized

def show(buffer, title='title'):
    """ show the given buffer """
    cv2.imshow(title, buffer)
    cv2.waitKey(0)

def getpixel(buffer, x, y):
    """ get the pixel value at x,y
        nb: cv2 x,y is reversed from our pov
        """
    max_x, max_y = size(buffer)
    if x < 0 or x >= max_x or y < 0 or y >= max_y:
        # out of range pixel
        return None
    else:
        return buffer[y, x]  # NB: cv2 x, y are reversed

def putpixel(buffer, x, y, value):
    """ put the pixel of value at x,y
        value may be a greyscale value or a colour tuple
        """
    max_x, max_y = size(buffer)
    if x < 0 or x >= max_x or y < 0 or y >= max_y:
        return
    buffer[y, x] = value  # NB: cv2 x, y are reversed

def pixelparts(buffer, x: float, y: float) -> ():
    """ get the neighbour parts contributions for a pixel at x,y
        x,y are fractional so the pixel contributions is a mixture of the 4 pixels around x,y,
        the mixture is based on the ratio of the neighbours to include, the ratio of all 4 is 1,
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
        the pixel value is the sum of the overlap fractions of its neighbour pixel squares,
        P is the fractional pixel address in its pixel, 1, 2 and 3 are its neighbours,
        dotted area is contribution from neighbours:
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
    max_x, max_y = size(buffer)
    cX: float = x
    cY: float = y
    xL: int = int(cX)
    yL: int = int(cY)
    xH: int = xL + 1
    yH: int = yL + 1
    xU = min(xH, max_x - 1)
    yU = min(yH, max_y - 1)
    pixel_xLyL = buffer[yL][xL]
    pixel_xLyH = buffer[yU][xL]
    pixel_xHyL = buffer[yL][xU]
    pixel_xHyH = buffer[yU][xU]
    ratio_xLyL = (xH - cX) * (yH - cY)
    ratio_xHyL = (cX - xL) * (yH - cY)
    ratio_xLyH = (xH - cX) * (cY - yL)
    ratio_xHyH = (cX - xL) * (cY - yL)
    return (pixel_xLyL, ratio_xLyL), (pixel_xHyL, ratio_xHyL), (pixel_xLyH, ratio_xLyH), (pixel_xHyH, ratio_xHyH)

def makepixel(buffer, x: float, y: float) -> float:
    """ get the interpolated pixel value from buffer at x,y """
    LL, HL, LH, HH = pixelparts(buffer, x, y)
    part_xLyL = LL[0] * LL[1]
    part_xHyL = HL[0] * HL[1]
    part_xLyH = LH[0] * LH[1]
    part_xHyH = HH[0] * HH[1]
    return part_xLyL + part_xHyL + part_xLyH + part_xHyH

def inimage(buffer, x, y, r):
    """ determine if the points at radius R and centred at X, Y are within the image """
    max_x, max_y = size(buffer)
    if (x - r) < 0 or (x + r) >= max_x or (y - r) < 0 or (y + r) >= max_y:
        return False
    else:
        return True

def incolour(buffer, colour=None):
    """ turn buffer into a colour image if required and its not already """
    if colour is None or type(colour) == tuple:
        return colourize(buffer)
    return buffer

def colourize(buffer):
    """ make grey image into an RGB one,
        returns the image array with 3,
        its a no-op if we're not a grey image
        """
    if len(buffer.shape) == 2:
        image = cv2.merge([buffer, buffer, buffer])
    else:
        image = buffer
    return image

def settext(buffer, text, origin, size=0.5, colour=0):
    """ set a text string at x,y of given size and colour (greyscale or a colour tuple) """
    buffer = incolour(buffer, colour)  # colourize iff required
    return cv2.putText(buffer, text, make_int(origin), cv2.FONT_HERSHEY_SIMPLEX, size, colour, 1, cv2.LINE_AA)

def circle(buffer, origin, radius, colour=0, thickness=1):
    """ draw a circle as directed """
    buffer = incolour(buffer, colour)  # colourize iff required
    if radius < 1:
        # too small for a circle, do a point instead
        putpixel(buffer, make_int(origin[0]), make_int(origin[1]), colour)
        return buffer
    return cv2.circle(buffer, make_int(origin), make_int(radius), colour, thickness)

def rectangle(buffer, top_left, bottom_right, colour=0, thickness=1):
    """ draw a rectangle as directed """
    buffer = incolour(buffer, colour)  # colourize iff required
    return cv2.rectangle(buffer, make_int(top_left), make_int(bottom_right), colour, thickness)

def line(buffer, from_here, to_there, colour=0, thickness=1):
    """ draw a line as directed """
    buffer = incolour(buffer, colour)  # colourize iff required
    return cv2.line(buffer, make_int(from_here), make_int(to_there), colour, thickness)

def grid(buffer, x_spacing=None, y_spacing=None, colour=0, thickness=1, labels=None, label_spacing=1, text_size=0.5):
    """ draw grid lines as directed """
    max_x, max_y = size(buffer)
    if x_spacing is not None:
        for x in range(0, max_x, x_spacing):
            buffer = line(buffer, (x, 0), (x, max_y-1), colour=colour, thickness=thickness)
            if labels is not None:
                if (x % label_spacing) == 0:
                    buffer = settext(buffer, '{}'.format(x), (x, max_y-1), colour=labels, size=text_size)
    if y_spacing is not None:
        for y in range(0, max_y, y_spacing):
            buffer = line(buffer, (0, y), (max_x-1, y), colour=colour, thickness=thickness)
            if labels is not None:
                if (y % label_spacing) == 0:
                    buffer = settext(buffer, '{}'.format(y), (0, y), colour=labels, size=text_size)
    return buffer

def plot(buffer, points, colour):
    """ plot the given points in the given colour """
    buffer = incolour(buffer, colour)  # colourize iff required
    for x, y in points:
        putpixel(buffer, x, y, colour)
    return buffer

def make_int(this):
    """ make the given thing (a number or a tuple of numbers) into integers """
    if type(this) == tuple:
        # Stupid language, cannot create a tuple like this, it makes a 'generator', useless!
        return [int(round(x)) for x in this]
    else:
        return int(round(this))

def translate(point, radius, origin, scale):
    """ translate and scale a circle,
        if origin is +ve map from a full-image to a sub-image (an extraction from the full image),
        if origin is -ve map from a sub-image to a full-image,
        relative to (0,0) to be relative to origin and scale it by scale
        """
    if origin[0] < 0 and origin[1] < 0:
        # map from a sub-image to a full-image
        x = (point[0] / scale) - origin[0]
        y = (point[1] / scale) - origin[1]
        radius /= scale
    elif origin[0] >= 0 and origin[1] >= 0:
        # map from full-image to a sub-image
        x = (point[0] - origin[0]) * scale
        y = (point[1] - origin[1]) * scale
        radius *= scale
    else:
        raise Exception('Origin must be (-ve,-ve) or (+ve,+ve) not a mixture {}'.format(origin))
    return (x, y), radius
