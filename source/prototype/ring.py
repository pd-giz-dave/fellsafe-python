
import codec
import math
import angle
import const

class Ring:
    """ this class knows how to draw the marker and data rings according to its constructor parameters
        see description in Codec for the overall target structure
        """

    BULLSEYE_RINGS = 2
    INNER_BLACK_RINGS = codec.Codec.INNER_BLACK_RINGS
    OUTER_BLACK_RINGS = codec.Codec.OUTER_BLACK_RINGS
    OUTER_WHITE_RINGS = 1
    EDGE_RINGS = INNER_BLACK_RINGS + OUTER_BLACK_RINGS  # rings used for inner/outer edge detection
    MARKER_RINGS = BULLSEYE_RINGS + EDGE_RINGS + OUTER_WHITE_RINGS  # all the non-data rings
    NUM_RINGS = codec.Codec.RINGS_PER_DIGIT + MARKER_RINGS  # total rings in our complete code
    DIGITS = codec.Codec.DIGITS  # how many bits in each ring
    BORDER_WIDTH = 0.2  # border width in units of rings
    OUTER_WHITE_WIDTH = 1 - BORDER_WIDTH  # outer white width only needs to be enough to ensure an edge is detected

    def __init__(self, centre_x, centre_y, width, frame):
        # set constant parameters
        self.w = width  # width of each ring in pixels
        self.f = frame  # where to draw it
        self.x = centre_x  # where the centre of the rings are
        self.y = centre_y  # ..

        # setup our angles look-up table such that get at least 2 pixels per step on outermost ring
        radius = width * Ring.NUM_RINGS
        self.scale = 2 * math.pi * radius * 2.1
        self.angle_xy = angle.Angle(self.scale, radius).polarToCart
        self.edge = 360 / Ring.DIGITS  # the angle at which a bit edge occurs (NB: not an int)

    def _point(self, x, y, bit):
        """ draw a point at offset x,y from our centre with the given bit (0 or 1) colour,
            bit can be 0 (draw 'black'), 1 (draw 'white'),
            any other value (inc None) will draw 'grey' (mid luminance),
            x,y can be fractional, in which case when drawing a white pixel we distribute
            the luminance across its neighbours such that their sum is white,
            """

        def put_pixel(cX: float, cY: float, pixel: int):
            """ put the interpolated pixel value to x,y (pixel value is +ve to increase luminance and -ve to decrease),
                x,y can be fractional so the pixel value is distributed to the 4 pixels around x,y
                the mixture is based on the ratio of the neighbours to include, the ratio of all 4 is 1
                see here: https://imagej.nih.gov/ij/plugins/download/Polar_Transformer.java for the inspiration
                explanation:
                x,y represent the top-left of a 1x1 pixel
                if x or y are not whole numbers the 1x1 pixel area overlaps its neighbours,
                the pixel value is distributed according to the overlap fractions of its neighbour
                pixel squares, P is the fractional pixel address in its pixel, 1, 2 and 3 are
                its neighbours, dotted area is distribution to neighbours:
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

            def update_pixel(x, y, value):
                """ update the pixel value at x,y by adding value (value may be -ve) """
                old_value = self.f.getpixel(x, y)
                new_value = int(max(min(old_value + value, const.MAX_LUMINANCE), const.MIN_LUMINANCE))
                self.f.putpixel(x, y, new_value, True)

            xL: int = int(cX)
            yL: int = int(cY)

            if (cX - xL) < (1/255) and (cY - yL) < (1/255):
                # short-cut when no/small neighbours involved ('cos Python is so slow!)
                self.f.putpixel(xL, yL, min(pixel, const.MIN_LUMINANCE), True)
                return

            xH: int = xL + 1
            yH: int = yL + 1
            ratio_xLyL = (xH - cX) * (yH - cY)
            ratio_xHyL = (cX - xL) * (yH - cY)
            ratio_xLyH = (xH - cX) * (cY - yL)
            ratio_xHyH = (cX - xL) * (cY - yL)
            part_xLyL = pixel * ratio_xLyL
            part_xHyL = pixel * ratio_xHyL
            part_xLyH = pixel * ratio_xLyH
            part_xHyH = pixel * ratio_xHyH

            update_pixel(xL, yL, part_xLyL)
            update_pixel(xL, yH, part_xLyH)
            update_pixel(xH, yL, part_xHyL)
            update_pixel(xH, yH, part_xHyH)

        if bit is None:
            colour = const.MID_LUMINANCE
        elif bit == 0:
            colour = const.MIN_LUMINANCE
        elif bit == 1:
            colour = const.MAX_LUMINANCE
        else:
            colour = const.MID_LUMINANCE

        if colour == const.MIN_LUMINANCE:
            # set to reduce the luminance
            colour = int(0 - const.MAX_LUMINANCE)

        put_pixel(self.x + x, self.y + y, colour)

    def _draw(self, radius, bits):
        """ draw a ring at radius of bits, a 1-bit is white, 0 black,
            if bits is None a solid grey ring is drawn,
            the bits are drawn big-endian and clockwise , i.e. MSB first (0 degrees), LSB last (360 degrees)
            """
        if radius <= 0:
            # special case - just draw a dot at x,y of the LSB colour of bits
            if bits is None:
                self._point(0, 0, None)
            else:
                self._point(0, 0, bits & 1)
        else:
            msb = 1 << (Ring.DIGITS - 1)
            for step in range(int(round(self.scale))):
                a = (step / self.scale) * 360
                x, y = self.angle_xy(a, radius)
                if a > 0:
                    segment = int(a / self.edge)
                else:
                    segment = 0
                mask = msb >> segment
                if bits is None:
                    self._point(x, y, None)
                elif bits & mask:
                    self._point(x, y, 1)
                else:
                    self._point(x, y, 0)

    def _draw_ring(self, ring_num, data_bits, ring_width):
        """ draw a data ring with the given data_bits and width,
            """
        start_ring = int(ring_num * self.w)
        end_ring = int(start_ring + ring_width * self.w)
        for radius in range(start_ring, end_ring):
            self._draw(radius, data_bits)

    def _border(self, ring_num, border_width):
        """ draw a border at the given ring_num,
            this is visual cue to stop people cutting into important parts of the code
            """

        self._draw_ring(ring_num, None, border_width)

    def _label(self, ring_num, number):
        """ draw a 3 digit number at given ring position in a size compatible with the ring width
            we do it DIY using a 4 x 5 grid
            """
        zero = [[1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1]]
        one = [[0, 1, 0, 0],
               [1, 1, 0, 0],
               [0, 1, 0, 0],
               [0, 1, 0, 0],
               [1, 1, 1, 0]]
        two = [[1, 1, 1, 1],
               [0, 0, 0, 1],
               [1, 1, 1, 1],
               [1, 0, 0, 0],
               [1, 1, 1, 1]]
        three = [[1, 1, 1, 1],
                 [0, 0, 0, 1],
                 [1, 1, 1, 1],
                 [0, 0, 0, 1],
                 [1, 1, 1, 1]]
        four = [[0, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 1, 1],
                [0, 0, 1, 0],
                [0, 0, 1, 0]]
        five = [[1, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 1],
                [1, 1, 1, 1]]
        six = [[1, 0, 0, 0],
               [1, 0, 0, 0],
               [1, 1, 1, 1],
               [1, 0, 0, 1],
               [1, 1, 1, 1]]
        seven = [[1, 1, 1, 1],
                 [0, 0, 0, 1],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0]]
        eight = [[1, 1, 1, 1],
                 [1, 0, 0, 1],
                 [1, 1, 1, 1],
                 [1, 0, 0, 1],
                 [1, 1, 1, 1]]
        nine = [[1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1]]
        digits = [zero, one, two, three, four, five, six, seven, eight, nine]
        point_size = max(int((self.w / len(zero[0])) * 1.0), 1)
        digit_size = point_size * len(zero[0])
        start_x = int(round(0 - min(ring_num * self.w, self.x)))
        start_y = start_x
        place = -1
        limits = [[None, None], [None, None]]  # min/max x/y used when drawing digits
        for power in [100, 10, 1]:
            place += 1
            x = start_x + ((digit_size + point_size) * place)
            y = start_y
            digit = digits[int(int(number / power) % 10)]
            for px in range(len(digit)):
                for py in range(len(digit[0])):
                    if digit[px][py] > 0:
                        # draw a black point here (0's are left at the background)
                        for dx in range(point_size):
                            for dy in range(point_size):
                                tx = x + dx
                                ty = y + dy
                                self._point(tx, ty, 0)    # draw at min luminance
                                if limits[0][0] is None or limits[0][0] > tx:
                                    limits[0][0] = tx
                                if limits[0][1] is None or limits[0][1] > ty:
                                    limits[0][1] = ty
                                if limits[1][0] is None or limits[1][0] < tx:
                                    limits[1][0] = tx
                                if limits[1][1] is None or limits[1][1] < ty:
                                    limits[1][1] = ty
                    x += point_size
                x -= digit_size
                y += point_size
        # draw a line under the number (so we know which way up 666 or 999, et al, should be)
        start_x = limits[0][0]
        end_x = limits[1][0]
        start_y = limits[1][1] + point_size
        end_y = start_y + point_size
        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                self._point(x, y, 0)    # draw at min luminance

    def code(self, number, rings):
        """ draw the complete code for the given number and code-words
            the code-words must match the number, this function is the
            only one that knows the overall target ring structure
            """

        # draw the bullseye and the inner white/black rings
        self._draw_ring(0.0, -1, Ring.BULLSEYE_RINGS)
        self._draw_ring(2.0, 0, Ring.INNER_BLACK_RINGS)
        draw_at = Ring.BULLSEYE_RINGS + Ring.INNER_BLACK_RINGS

        # draw the data rings
        for ring in rings:
            self._draw_ring(draw_at, ring, 1.0)
            draw_at += 1.0

        # draw the outer black and white
        self._draw_ring(draw_at, 0, Ring.OUTER_BLACK_RINGS)
        draw_at += Ring.OUTER_BLACK_RINGS
        self._draw_ring(draw_at, -1, Ring.OUTER_WHITE_WIDTH)
        draw_at += Ring.OUTER_WHITE_WIDTH

        # draw a border
        self._border(draw_at, Ring.BORDER_WIDTH)
        draw_at += Ring.BORDER_WIDTH

        if int(draw_at) != draw_at:
            raise Exception('number of target rings is not integral ({})'.format(draw_at))
        draw_at = int(draw_at)

        # safety check
        if draw_at != Ring.NUM_RINGS:
            raise Exception('number of rings exported ({}) is not {}'.format(Ring.NUM_RINGS, draw_at))

        # draw a human readable label
        self._label(draw_at, number)
