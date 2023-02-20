""" Image transformation primitives """

import math
import const
import numpy as np

class Transform:
    """ various image transforming operations """

    identity_map = ((0, 0, 0),
                    (0, 1, 0),
                    (0, 0, 0))

    sharpen_map = (( 0, -1,  0),
                   (-1,  5, -1),
                   ( 0, -1,  0))

    #edge_map = ((-1, -1, -1),
    #            (-1,  8, -1),
    #            (-1, -1, -1))

    #edge_map = (( 0, -1,  0),
    #            (-1,  4, -1),
    #            ( 0, -1,  0))

    edge_map = (( 1,  0, -1),
                ( 0,  0,  0),
                (-1,  0,  1))

    sobel_x = ((-1, -2, -1),
               ( 0,  0,  0),
               ( 1,  2,  1))

    sobel_y = ((-1, 0, 1),
               (-2, 0, 2),
               (-1, 0, 1))

    def __init__(self, source):
        self.source = source
        self.max_x  = self.source.shape[1]  # NB: cv2 x, y are reversed
        self.max_y  = self.source.shape[0]  # ..

    @staticmethod
    def new(width, height, luminance=const.MIN_LUMINANCE):
        """ prepare a new buffer of the given size and luminance """
        return Transform(np.full((height, width), luminance, dtype=np.uint8))  # NB: numpy arrays follow cv2 conventions

    def get_size(self):
        """ return the max x, y in the given source buffer """
        return self.max_x, self.max_y

    def get_pixel(self, x, y):
        """ get the pixel at x,y in the given source buffer """
        if x < 0 or x >= self.max_x or y < 0 or y >= self.max_y:
            # out of range pixel
            return None
        else:
            return self.source[y, x]  # NB: cv2 x, y are reversed

    def put_pixel(self, x, y, pixel):
        """ put the pixel to x,y in source """
        if x < 0 or x >= self.max_x or y < 0 or y >= self.max_y:
            return
        self.source[y, x] = pixel  # NB: cv2 x, y are reversed

    @staticmethod
    def target_xy(x, y, step):
        """ calculate the equivalent target x,y from the given source x,y and step size """
        return int(x / step), int(y / step)

    def median_blur(self, kernel_size):
        """ perform a median blur with the given (square) kernel size """
        # calc the kernel addresses (based on the central element)
        mid_k = kernel_size >> 1
        max_k = kernel_size - 1
        k_off = [x - mid_k for x in range(max_k + 1)]
        # get a new target buffer
        target = Transform.new(self.max_x, self.max_y, const.MIN_LUMINANCE)
        # do the convolution
        for x in range(self.max_x):
            for y in range(self.max_y):
                pixels = []
                for dy in k_off:
                    for dx in k_off:
                        pixel = self.get_pixel(x + dx, y + dy)
                        if pixel is not None:
                            pixels.append(pixel)
                if len(pixels) > 0:
                    pixels.sort()
                    pixel = pixels[len(pixels) >> 1]
                else:
                    pixel = None
                target.put_pixel(x, y, pixel)
        return target

    def mean_blur(self, kernel_size):
        """ perform a mean blur with the given (square) kernel size """
        kernel = [[1 for _ in range(kernel_size)] for _ in range(kernel_size)]
        return self._convolve(kernel, step=1)

    @staticmethod
    def gaussian(sigma, cutoff):
        """ make a gaussian blur kernel with the given sigma,
            sigma is the standard deviation required and cutoff is ratio (0..1) at which to stop,
            bigger sigma means a bigger kernel, bigger cutoff means a smaller kernel, see
            https://patrickfuller.github.io/gaussian-blur-image-processing-for-scientists-and-engineers-part-4/
            for the inspiration for this function
            """
        if 0 <= cutoff >= 1:
            raise Exception('cutoff must be between 0 and 1, not {}'.format(cutoff))
        if 0 <= sigma >= 10:
            raise Exception('sigma must be between 0 and 10, not {}'.format(sigma))
        kernel_width = 1 + 2 * max(int(math.sqrt(-2*sigma*sigma*math.log(cutoff))),1)
        kernel = [[0 for _ in range(kernel_width)] for _ in range(kernel_width)]
        min_g = None
        for v in range(kernel_width):
            for u in range(kernel_width):
                uc = u - (kernel_width>>1)
                vc = v - (kernel_width>>1)
                g = math.exp(-(uc*uc+vc*vc)/(2*sigma*sigma))
                kernel[u][v] = g
                if min_g is None or g < min_g:
                    min_g = g
        for v in range(kernel_width):
            for u in range(kernel_width):
                kernel[u][v] = int(kernel[u][v]/min_g)
        return kernel

    def gaussian_blur(self, sigma=1, step=1):
        """ apply a gaussian blur to the given image"""
        kernel = self.gaussian(sigma, .1)
        return self._convolve(kernel, step)

    def edges(self, step=1):
        """ apply edge detection to the given image"""
        return self._convolve(self.edge_map, step)

    def sharpen(self, step=1):
        """ sharpen the given image"""
        return self._convolve(self.sharpen_map, step)

    def sobel(self, step=1):
        """ apply a sobel edge detecting filter to source """
        target = Transform.new(self.max_x, self.max_y, const.MIN_LUMINANCE)
        for x in range(1, self.max_x - 1):
            for y in range(1, self.max_y - 1):
                magx, magy = 0, 0
                for a in range(3):
                    for b in range(3):
                        xn = x + a - 1
                        yn = y + b - 1
                        pixel = self.getpixel(xn, yn)
                        magx += pixel * self.sobel_x[a][b]
                        magy += pixel * self.sobel_y[a][b]
                        colour = int(math.sqrt(magx ** 2 + magy ** 2))
                        target.putpixel(x, y, colour)
        return target

    def _convolve(self, kernel, step):
        """ apply the convolution defined by kernel to the source frame returning a target frame,
            the image edges are ignored (kernel/2+1 pixels all around),
            the kernel must be square and an odd number (so there is a middle) and consist of integers,
            step is the x, y increment to apply (e.g. 2 halves the width and height of the target,
             [ a b c ]   [ 1 2 3 ]
             [ d e f ] * [ 4 5 6 ] == i*1 + h*2 + g*3 + f*4 + e*5 + d*6 + c*7 + b*8 + a*9
             [ g h i ]   [ 7 8 9 ]
            """

        # validate
        if len(kernel) != len(kernel[0]):
            raise Exception('Only square kernels supported, not {}'.format(kernel))
        if len(kernel) & 1 == 0:
            raise Exception('Only odd number of kernel elements supported, not {}'.format(kernel))
        # calc scale of the kernel (we divide by this to keep pixels in the same range)
        scale = 0
        for line in kernel:
            for item in line:
                scale += item
        # calc the kernel addresses (based on the central element)
        mid_k_x = len(kernel[0]) >> 1
        mid_k_y = len(kernel) >> 1
        max_k_x = len(kernel[0]) - 1
        max_k_y = len(kernel) - 1
        kx = [x-mid_k_x for x in range(max_k_x+1)]
        ky = [y-mid_k_y for y in range(max_k_y+1)]
        # calc source/target co-ordinate ranges
        min_x = mid_k_x + 1
        min_y = mid_k_y + 1
        width, height = self.get_size()
        max_x = width - min_x
        max_y = height - min_y
        target_width, target_height = Transform.target_xy(width, height, step)
        # get a new target buffer
        target = Transform.new(target_width, target_height, const.MIN_LUMINANCE)
        # do the convolution
        for y in range(min_y, max_y, step):
            for x in range(min_x, max_x, step):
                pixel = 0
                for dy in ky:
                    for dx in kx:
                        pixel += (self.get_pixel(x+dx, y+dy) * kernel[mid_k_y-dy][mid_k_x-dx])
                if scale != 0:
                    pixel = int(pixel / scale)
                target_x, target_y = Transform.target_xy(x, y, step)
                target.put_pixel(target_x, target_y, pixel)
        return target

    def max(self):
        """ reduce each 3x3 segment to its max returning a new target buffer """
        width, height = self.get_size()
        min_x = 1
        min_y = 1
        max_x = width - min_x
        max_y = height - min_y
        target = Transform.new(max_x - min_x, max_y - min_y, const.MIN_LUMINANCE)
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                max_val = 0
                for dy in (-1, 0, +1):
                    for dx in (-1, 0, +1):
                        pixel = self.get_pixel(x+dx, y+dy)
                        if pixel > max_val:
                            max_val = pixel
                target.put_pixel(x-min_x, y-min_y, max_val)
        return target

    @staticmethod
    def diff(s1, s2):
        """ given 2 source images of the same size return an image of their difference
            the differences are scaled to span the whole luminance range (0==no diff==black,
            255==max diff==white)
            """
        s1_size = s1.get_size()
        s2_size = s2.get_size()
        if s1_size != s2_size:
            raise Exception('Source 1 and 2 must be the same size, not {} and {}'.format(s1_size, s2_size))
        width, height = s1_size
        target = Transform.new(width, height, const.MIN_LUMINANCE)
        min_d = const.MAX_LUMINANCE
        max_d = const.MIN_LUMINANCE
        for y in range(height):
            for x in range(width):
                p1 = s1.get_pixel(x, y)
                p2 = s2.get_pixel(x, y)
                if p1 < p2:
                    d = p2 - p1
                else:
                    d = p1 - p2
                if d < min_d:
                    min_d = d
                if d > max_d:
                    max_d = d
                target.put_pixel(x, y, d)
        scale = const.MAX_LUMINANCE / max(max_d - min_d, 1)
        for y in range(height):
            for x in range(width):
                pixel = int((target.get_pixel(x, y) - min_d) * scale)
                target.put_pixel(x, y, pixel)
        return target
