import os
import glob

import cv2
import numpy as np
import random
import math
import traceback

""" coding scheme

    This coding scheme is intended to be easy to detect and robust against noise, distortion and luminance artifacts.
    The code is designed for use as competitor numbers in club level fell races, as such the number range
    is limited to between 101 and 999 inclusive, i.e. always a 3 digit number and max 899 competitors.
    The actual implementation limit may be less depending on coding constraints chosen (parity, edges, code size).
    The code is circular (so orientation does not matter) and consists of:
        a solid circular centre of 'white' with a radius 2R,
        surrounded by a solid ring of 'black' and width 1R,
        surrounded by 3 concentric data rings of width R and divided into N equal segments,
        surrounded by a timing ring with edges such that there is an edge for all N segments of width R,
        enclosed by a solid 'black' ring of width 1R and, 
        finally, a solid 'white' ring of radius R. 
    Total radius is 9R.
    The 3 data rings are used as a triple redundant data bit copy, each consists of a start sync pattern (so bit
    0 can be distinguished in any sequence of bits), and payload bits, each ring is bit shifted right by N bits,
    (so that an image distortion does not hit the same bit in all copies), N is chosen such that the rings are
    evenly spaced around the circle, each data ring may be XOR'd with a bit pattern (used to invert the middle ring).
    A one-bit is white (i.e. high luminance) and a zero-bit is black (i.e. low luminance).
    The start sync pattern is 0110 (4 bits). The remaining bits are the payload (big-endian) and must not contain
    the start sync pattern and also must not end in 011 and must not start with 110 (else they look like 0110 when
    adjacent to the alignment marker).
    The payload is N data-bits and an optional parity bit and may be constrained to have at least N bit transitions.
    The inner white-to-black and outer black-to-white ring transitions are used detect the limits of the code in
    the image. The white ring preceding the inner white-to-black transition is used to set the white luminance
    threshold, the black ring succeeding the inner white-to-black transition is used to set the black luminance. 
    These are divided into bit segments to allow for luminance variation around the rings.
    The central 'bullseye' candidates are detected using a 'blob detector' (via opencv), then the area around
    that is polar to cartesian 'warped' (via opencv) into a rectangle. All further processing is on that rectangle.
    
    Note: The decoder can cope with several targets in the same image. It uses the relative size of
          the targets to present the results in distance order, nearest (i.e. biggest) first. This is
          important to determine finish order.
                
    """

# ToDo: create test background with random noise blobs and and random edges
# ToDo: add a blurring option when drawing test targets
# ToDo: add a rotation (vertically and horizontally) option when drawing test targets
# ToDo: add a curving (vertically and horizontally) option when drawing test targets
# ToDo: generate lots of (extreme) test targets using these options
# ToDo: draw the test targets on a relevant scene photo background

# colours
MAX_LUMINANCE = 255
MIN_LUMINANCE = 0
MID_LUMINANCE = (MAX_LUMINANCE - MIN_LUMINANCE) >> 1

# alpha channel
TRANSPARENT = MIN_LUMINANCE
OPAQUE = MAX_LUMINANCE

""" WARNING
    cv2's cor-ordinates are backwards from our pov, the 'x' co-ordinate is vertical and 'y' horizontal.
    The logic here uses 'x' horizontally and 'y' vertically, swapping as required when dealing with cv2
    """


def vstr(vector):
    """ given an array of numbers return a string representing them """
    result = ''
    for pt in vector:
        if pt is None:
            result += ', None'
        else:
            result += ', {:.2f}'.format(pt)
    return '[' + result[2:] + ']'


class Codec:
    """ Encode and decode a number or a bit or a blob
        a number is a payload, it can be encoded and decoded
        a bit is a raw bit decoded from 3 blobs
        a blob is decoded from 3 luminance level samples
        this class encapsulates all the encoding and decoding and their constants
        """

    BITS = 15  # total bits in the code
    MARKER_BITS = 4  # number of bits in our alignment marker

    # ring XOR masks (-1 just inverts the whole ring)
    INNER_MASK = 0
    MIDDLE_MASK = -1
    OUTER_MASK = 0

    # luminance level thresholds (white tends to bleed into black, so we make the black level bigger)
    WHITE_WIDTH = 0.25  # band width of white level within luminance range
    BLACK_WIDTH = 0.5  # band width of black level within luminance range

    # blob value categories
    BLACK = 0
    WHITE = 1
    GREY = 2

    # bit value categories
    IS_ZERO = 0
    IS_ONE = 1
    LIKELY_ZERO = 2
    LIKELY_ONE = 3
    MAYBE_ZERO = 4
    MAYBE_ONE = 5
    MAYBE_ONE_OR_ZERO = 6

    def __init__(self, min_num, max_num, parity, edges):
        """ create the valid code set for a number in the range min_num..max_num for code_size
            a valid code is one where there are no embedded start/stop bits bits but contains at least one 1 bit,
            we create two arrays - one maps a number to its code and the other maps a code to its number.
            """

        # params
        self.min_num = min_num  # minimum number we want to be able to encode
        self.max_num = max_num  # maximum number we want to be able to encode
        self.parity = parity  # None, 0 (even) or 1 (odd)
        self.edges = edges  # how many bit transitions we want per code
        self.skew = max(int(self.BITS / 3), 1)  # ring to ring skew in bits
        self.code_bits = self.BITS - self.MARKER_BITS  # code word bits is what's left
        self.marker = 6 << self.code_bits  # 0110 in MS 4 bits of code
        self.code_range = 1 << self.code_bits  # max code range before constraints applied
        self.bit_mask = (1 << self.BITS) - 1  # mask to isolate all bits

        # build code tables
        self.codes = [None for _ in range(self.code_range)]  # set all invalid initially
        self.nums = [None for _ in range(self.max_num + 1)]  # ..
        num = self.min_num - 1  # last number to be encoded (so next is our min)
        for code in range(1, self.code_range):  # start at 1 'cos 0 has no 1 bits
            if (code & 7) == 3:  # LS 3 bits are 011
                # can't use this 'cos it'll look like the marker when elided with it
                pass
            # NB: The leading 110 rule is implicit in our embedded 0110 check as the right shift introduces
            #     a leading 0 bit by the time it gets to the LS 4 bits
            else:
                check = code
                for _ in range(self.code_bits):
                    if (check & 15) == 6:  # LS 4 bits are 0110
                        # embedded alignment marker, so not a valid code
                        check = None
                        break
                    check >>= 1  # NB: introduces a leading 0 (required, see above)
                if check is not None:
                    # got a potential code, check its parity
                    one_bits = 0
                    edges = 0
                    check = code
                    for bit in range(self.code_bits):
                        if check & 1 == 1:
                            one_bits += 1
                        if (check & 3 == 1) or (check & 3 == 2):
                            edges += 1
                        check >>= 1
                    if (self.parity is None or ((one_bits & 1) == self.parity)) and (edges >= self.edges):
                        # got a good parity and edges, give it to next number
                        num += 1
                        if num > self.max_num:
                            # found enough, don't use this one
                            pass
                        else:
                            self.codes[code] = num  # decode code as num
                            self.nums[num] = code  # encode num as code
        self.num_limit = num

        # generate the bit limited version of our XOR masks
        self.inner_mask = Codec.INNER_MASK & self.bit_mask
        self.middle_mask = Codec.MIDDLE_MASK & self.bit_mask
        self.outer_mask = Codec.OUTER_MASK & self.bit_mask

    def encode(self, num):
        """ get the code for the given number, returns None if number not valid """
        if num > self.max_num or num < self.min_num:
            return None
        return self.nums[num]

    def decode(self, code):
        """ get the number for the given code, if not valid returns None """
        if code >= self.code_range or code < 0:
            return None
        return self.codes[code]

    def build(self, num):
        """ build the codes needed for the 3 rings
            returns the three integers required to build a 'target'
            """
        if num is None:
            return None
        code_word = self.encode(num)
        if code_word is None:
            return None
        return self.rings(code_word)

    def rings(self, code_word):
        """ build the data ring codes for the given code word,
            it returns 3 integers, the LS n bits are the code word and alignment marker, each word
            is rotated clockwise by n bits to give the required skew, each ring is XOR'd with a mask
            specific to that ring (see discussion above)
            """
        # the bit shift is relative to the MSB, this matters when 3 * shift is not the code size
        # for each shift the LS n bits must be moved to the MS n bits
        code_word &= (self.code_range - 1)  # make sure code is in range
        mask = (1 << self.skew) - 1  # mask to isolate the LS n bits
        shift = (self.BITS - (self.skew * 3)) + (self.skew * 2)
        # apply the XOR masks to each ring
        r1 = (self.marker + code_word) ^ (self.inner_mask & self.bit_mask)
        r2 = (self.marker + code_word) ^ (self.middle_mask & self.bit_mask)
        r3 = (self.marker + code_word) ^ (self.outer_mask & self.bit_mask)
        # skew r2 and r3
        r2 = (r2 >> self.skew) + ((r2 & mask) << shift)
        r3 = (r3 >> self.skew) + ((r3 & mask) << shift)  # one shift block
        r3 = (r3 >> self.skew) + ((r3 & mask) << shift)  # ..and again to do it twice
        # return result
        return r1, r2, r3

    def ring_bits_pos(self, n):
        """ given a bit index return a list of the indices of all the same bits from each ring """
        n1 = n
        n2 = int((n1 + self.skew) % self.BITS)
        n3 = int((n2 + self.skew) % self.BITS)
        return [n1, n2, n3]

    def marker_bits_pos(self, n):
        """ given a bit index return a list of the indices of all the bits that would make a marker """
        return [int(pos % self.BITS) for pos in range(n, n + self.MARKER_BITS)]

    def unbuild(self, samples, levels):
        """ given an array of 3 code-word rings with random alignment return the encoded number or None,
            each ring must be given as an array of blob values in bit number order,
            levels is a 2 x N array of luminance levels for each bit position that represent the white
            level and black level for that bit,
            returns the number (or None), the level of doubt and the bit classification for each bit,
            """

        # step 1 - decode the 3 rings bits
        bits = [None for _ in range(self.BITS)]
        for n in range(self.BITS):
            rings = self.ring_bits_pos(n)
            bit_mask = 1 << (self.BITS - (n + 1))
            s1_mask = (self.inner_mask & bit_mask)
            s2_mask = (self.middle_mask & bit_mask)
            s3_mask = (self.outer_mask & bit_mask)
            s1 = samples[0][rings[0]]
            s2 = samples[1][rings[1]]
            s3 = samples[2][rings[2]]
            w1 = levels[0][rings[0]]
            w2 = levels[0][rings[1]]
            w3 = levels[0][rings[2]]
            b1 = levels[1][rings[0]]
            b2 = levels[1][rings[1]]
            b3 = levels[1][rings[2]]
            bits[n] = self.blob(s1, w1, b1, s1_mask, s2, w2, b2, s2_mask, s3, w3, b3, s3_mask)
            continue

        # step 2 - find the alignment marker candidates
        maybe_at = [[] for _ in range(3 * 4)]  # 0..max maybe possibilities (3 per digit, 4 digits), a list for each
        for n in range(self.BITS):
            marker = self.is_marker(n, bits)
            if marker is None:
                continue  # no marker at this bit position, look at next
            # got a potential marker with marker maybe values, 0 == exact, 4 == all maybe
            maybe_at[marker].append(n)
        # maybe_at now contains a list of all possibilities for all maybe options

        # step 3 - extract all potential code words for each candidate alignment for each maybe level
        # any that yield more than one are crap and a give-up condition
        found = None
        for maybe in maybe_at:
            for n in maybe:
                # n is the next one we are going to try, demote all others
                word = [bit for bit in bits]  # make a copy
                code = self.extract_word(n, word)
                if code is not None:
                    if found is not None:
                        # got more than 1 - that's crap
                        return None, self._count_errors(bits), self._show_bits(bits)
                    found = code  # note the first one we find
            if found is not None:
                # only got 1 from this maybe level, go with it
                return found, self._count_errors(bits), self._show_bits(bits)

        # no candidates qualify
        return None, self._count_errors(bits), self._show_bits(bits)

    def _count_errors(self, bits):
        """ given a set of categorised bits, count how many ?'s it has, and how many digits have errors,
            the bigger this number there more doubt there is in the validity of the code,
            the number returned is in the form N.M where N is error digits and M is the total errors
            """
        total_doubt = 0
        digit_doubt = 0
        errors = [0,  # IS_ZERO
                  0,  # IS_ONE
                  1,  # LIKELY_ZERO
                  1,  # LIKELY_ONE
                  2,  # MAYBE_ZERO
                  2,  # MAYBE_ONE
                  3]  # IS_NEITHER
        for bit in bits:
            if errors[bit] > 1:
                digit_doubt += 1
            total_doubt += errors[bit]
        return '{}.{}'.format(digit_doubt, total_doubt)

    def _show_bits(self, bits):
        """ given an array of bit classifications return that in a readable CSV format
            """
        symbols = {self.IS_ONE: '111',
                   self.IS_ZERO: '000',
                   self.LIKELY_ONE: '11?',
                   self.LIKELY_ZERO: '00?',
                   self.MAYBE_ONE: '1??',
                   self.MAYBE_ZERO: '0??',
                   self.MAYBE_ONE_OR_ZERO: '???'}
        csv = ''
        for bit in bits:
            csv += ',' + symbols[bit]
        return csv[1:]

    def is_marker(self, n, bits):
        """ given a set of bits and a bit position check if an alignment marker is present there
            the function returns a doubt level (0==exact match, 12==no match), no match is returned as None
            """
        exact = 0
        maybe = 0
        likely = 0
        missing = 0
        i = self.marker_bits_pos(n)
        b1 = bits[i[0]]
        b2 = bits[i[1]]
        b3 = bits[i[2]]
        b4 = bits[i[3]]
        if b1 == self.IS_ZERO:
            exact += 1
        elif b1 == self.LIKELY_ZERO:
            likely += 1
        elif b1 == self.MAYBE_ZERO:
            maybe += 1
        elif b1 == self.MAYBE_ONE_OR_ZERO:
            missing += 1
        else:
            return None
        if b2 == self.IS_ONE:
            exact += 1
        elif b2 == self.LIKELY_ONE:
            likely += 1
        elif b2 == self.MAYBE_ONE:
            maybe += 1
        elif b2 == self.MAYBE_ONE_OR_ZERO:
            missing += 1
        else:
            return None
        if b3 == self.IS_ONE:
            exact += 1
        elif b3 == self.LIKELY_ONE:
            likely += 1
        elif b3 == self.MAYBE_ONE:
            maybe += 1
        elif b3 == self.MAYBE_ONE_OR_ZERO:
            missing += 1
        else:
            return None
        if b4 == self.IS_ZERO:
            exact += 1
        elif b4 == self.LIKELY_ZERO:
            likely += 1
        elif b4 == self.MAYBE_ZERO:
            maybe += 1
        elif b4 == self.MAYBE_ONE_OR_ZERO:
            missing += 1
        else:
            return None
        if exact == 4:
            return 0  # exact match
        if missing > 1:
            return None  # too ambiguous
        if missing > 0 and maybe > 2:
            return None  # also too ambiguous
        return (1 * likely) + (2 * maybe) + (3 * missing)

    def data_bits(self, n, bits):
        """ return an array of the data-bits from bits array starting at bit position n,
            this is effectively rotating the bits array and removing the marker bits such
            that the result is an array with [0] the first data bit and [n] the last
            """
        return [bits[int(pos % self.BITS)] for pos in range(n + self.MARKER_BITS, n + self.BITS)]

    def extract_word(self, n, bits):
        """ given an array of bit values with the alignment marker at position n
            extract the code word and decode it (via decode()), returns None if cannot
            """
        word = self.data_bits(n, bits)
        code = 0
        for bit in range(len(word)):
            code <<= 1  # make room for next bit
            val = word[bit]
            if (val == self.IS_ONE) or (val == self.LIKELY_ONE) or (val == self.MAYBE_ONE):
                code += 1  # got a one bit
            elif (val == self.IS_ZERO) or (val == self.LIKELY_ZERO) or (val == self.MAYBE_ZERO):
                pass  # got a zero bit
            else:
                return None  # got junk
        return self.decode(code)

    def bit(self, s1, m1, s2, m2, s3, m3):
        """ given 3 blob values and their inversion masks determine the most likely bit value,
            the blobs are designated as 'black', 'white' or 'grey'
            the return bit is one of 'is', 'likely' or 'maybe' one or zero, or is_neither
            """

        zeroes = 0
        ones = 0
        greys = 0

        def normal(sample):
            nonlocal zeroes, ones, greys
            if sample == self.GREY:
                greys += 1
            elif sample == self.BLACK:
                zeroes += 1
            elif sample == self.WHITE:
                ones += 1

        def inverted(sample):
            nonlocal zeroes, ones, greys
            if sample == self.GREY:
                greys += 1
            elif sample == self.BLACK:
                ones += 1
            elif sample == self.WHITE:
                zeroes += 1

        def count(sample, invert):
            if invert != 0:
                inverted(sample)
            else:
                normal(sample)

        # count states
        count(s1, m1)
        count(s2, m2)
        count(s3, m3)

        # test ideal cases
        if zeroes == 3:
            return self.IS_ZERO
        elif ones == 3:
            return self.IS_ONE

        # test likely cases
        if zeroes == 2:
            return self.LIKELY_ZERO
        elif ones == 2:
            return self.LIKELY_ONE

        # test maybe cases
        if zeroes == 1 and greys == 2:
            return self.MAYBE_ZERO
        elif ones == 1 and greys == 2:
            return self.MAYBE_ONE

        # the rest are junk
        return self.MAYBE_ONE_OR_ZERO

    def blob(self, s1, w1, b1, m1, s2, w2, b2, m2, s3, w3, b3, m3):
        """ given 3 luminance samples, their luminance levels and inversion masks determine the most
            likely blob value,
            each sample is checked against the luminance levels to determine if its black, grey or white
            then decoded as a bit
            sN is the sample level, wN is the white level for that sample and bN is its black level,
            """
        return self.bit(self.category(s1, w1, b1), m1,
                        self.category(s2, w2, b2), m2,
                        self.category(s3, w3, b3), m3)

    def category(self, sample_level, white_level, black_level):
        """ given a luminance level and its luminance range categorize it as black, white or grey,
            the white low threshold is the white width below the white level,
            the black high threshold is the black width above the black level,
            grey is below the white low threshold but above the black high threshold
            """
        if sample_level is None:
            # not given a sample, treat as grey
            return Codec.GREY
        if white_level is None or black_level is None:
            # we haven't been given the thresholds, treat as grey
            return Codec.GREY
        luminance_range = white_level - black_level
        white_range = luminance_range * Codec.WHITE_WIDTH
        black_range = luminance_range * Codec.BLACK_WIDTH
        black_max = int(round(black_level + black_range))
        white_min = int(round(white_level - white_range))
        if black_max > white_min:
            # not enough luminance variation, consider as grey
            return Codec.GREY
        if sample_level < black_max:
            return Codec.BLACK
        elif sample_level > white_min:
            return Codec.WHITE
        else:
            return Codec.GREY


class Angle:
    """ a fast mapping (i.e. uses lookup tables and not math functions) from angles to co-ordinates
        and co-ordinates to angles for a circle, also for the arc length of an ellipsis
        """

    def __init__(self, scale):
        """ build the lookup tables with the resolution required for a single octant, from this octant
            the entire circle can be calculated by rotation and reflection (see angle() and ratio()),
            scale defines the accuracy required, the bigger the more accurate, it must be a +ve integer
            """
        # NB: This code is only executed once so clarity over performance is preferred

        # generate polar to cartesian lookup table
        self.ratio_scale = int(round(scale))
        self.angles = [None for _ in range(self.ratio_scale + 1)]
        self.angles[0] = 0
        for step in range(1, len(self.angles)):
            # each step here represents 1/scale of an octant
            # the index is the ratio of x/y (0..1*scale), the result is the angle (in degrees)
            self.angles[step] = math.degrees(math.atan(step / self.ratio_scale))

        # generate cartesian to polar lookup table
        self.ratios = [[None, None] for _ in range(self.ratio_scale + 1)]
        self.step_angle = 45 / self.ratio_scale  # the angle represented by each step in the lookup table
        for step in range(len(self.ratios)):
            # each octant here consists of scale steps,
            # the index is an angle 0..45, the result is the x,y co-ordinates for circle of radius 1,
            # angle 0 is considered to be straight up and increase clockwise, the vertical axis is
            # considered to be -Y..0..+Y, and the horizontal -X..0..+X,
            # the lookup table contains 0..45 degrees, other octants are calculated by appropriate x,y
            # reversals and sign reversals
            self.ratios[step][0] = 0.0 + math.sin(math.radians(step * self.step_angle))  # NB: x,y reversed
            self.ratios[step][1] = 0.0 - math.cos(math.radians(step * self.step_angle))  # ..
        # Parameters for ratio() for each octant:
        #   edge angle, offset, 'a' multiplier', reverse x/y, x multiplier, y multiplier
        #                                            #                     -Y
        #                                        #                     -Y
        self.octants = [[45, 0, +1, 0, +1, +1],  # octant 0         \ 7 | 0 /
                        [90, +90, -1, 1, -1, -1],  # octant 1       6  \  |  /  1
                        [135, -90, +1, 1, -1, +1],  # octant 2           \ | /
                        [180, +180, -1, 0, +1, -1],  # octant 3    -X ------+------ +X
                        [225, -180, +1, 0, -1, -1],  # octant 4           / | \
                        [270, +270, -1, 1, +1, +1],  # octant 5       5  /  |  \  2
                        [315, -270, +1, 1, +1, -1],  # octant 6         / 4 | 3 \
                        [360, +360, -1, 0, -1, +1]]  # octant 7            +Y
        # these octants describe the parameters to this equation:
        #  x = ratios[offset + angle*signa][0+reversed] * signx * radius
        #  y = ratios[offset + angle*signa][1-reversed] * signy * radius
        # octant 0 is the native values from self.ratios
        # octant[0] is the angle threshold
        # octant[1] is the offset to rotate the octant
        # octant[2] is the sign of a
        # octant[3] is the reversed signal, 0 == not reversed, 1 == reversed
        # octant[4] is the sign of x
        # octant[5] is the sign of y

        # generate ellipsis arc length lookup table
        self.one_degree_as_radians = math.pi / 180

    def polarToCart(self, a, r):
        """ get the x,y co-ordinates on the circumference of a circle of radius 'r' for angle 'a'
            'a' is in degrees (0..360), 'r' is in pixels
            'a' of 0 is 12 o'clock and increases clockwise
            if a < 0 or > 360 it 'wraps'
            """
        if r == 0:
            return 0, 0
        while a < 0:
            a += 360
        while a > 360:
            a -= 360
        for octant in self.octants:
            if a <= octant[0]:
                ratio = self.ratios[int(round((octant[1] + (a * octant[2])) / self.step_angle))]
                x = ratio[0 + octant[3]] * octant[4] * r
                y = ratio[1 - octant[3]] * octant[5] * r
                return x, y
        return None, None

    def cartToPolar(self, x, y):
        """ get the angle and radius from these x,y co-ordinates around a circle,
            see diagram in __init__ for the octant mapping (its this):
                +x, -y, -y >  x -->   0..45    octant 0
                +x, -y, -y <  x -->  45..90    octant 1
                +x, +y,  y <  x -->  90..135   octant 2
                +x, +y,  y >  x --> 135..180   octant 3
                -x, +y,  y > -x --> 180..225   octant 4
                -x, +y,  y < -x --> 225..270   octant 5
                -x, -y, -y < -x --> 270..315   octant 6
                -x, -y, -y > -x --> 315..360   octant 7
            edge cases:
                x = 0, y = 0 --> None        edge 0
                x > 0, y = 0 --> 90          edge 1
                x = 0, y > 0 --> 180         edge 2
                x < 0, y = 0 --> 270         edge 3
                x = 0, y < 0 --> 0 (or 360)  edge 4
        """

        def _ratio2angle(offset, sign, ratio):
            """ do a lookup on the given ratio, changes its sign (+1 or -1) and add the offset (degrees)
                ratio is in the range 0..1 and is mapped via the LUT to 0..45 degrees,
                the sign may change that to -45..0,
                the offset rotates that by that many degrees,
                result is degrees in range 0..360, with 0 at 12 o'clock
                """
            return offset + (self.angles[int(round(ratio * self.ratio_scale))] * sign)

        def _xy2r(x, y):
            """" convert x, y to a radius """
            return math.sqrt(x * x + y * y)

        # edge cases
        if x == 0:
            if y == 0: return None, None  # edge 0
            if y > 0:
                return 180, +y  # edge 2
            else:
                return 0, -y  # edge 4
        elif y == 0:  # and x != 0
            if x > 0:
                return 90, +x  # edge 1
            else:
                return 270, -x  # edge 3
        # which octant?
        # NB: both x and y are not 0 to get here
        if x > 0:
            # octant 0, 1, 2, 3
            if y < 0:
                # octant 0, 1
                if -y > x:
                    # octant 0
                    return _ratio2angle(0, +1, x / -y), _xy2r(x, y)
                else:
                    # octant 1
                    return _ratio2angle(90, -1, -y / x), _xy2r(x, y)
            else:
                # octant 2, 3
                if y < x:
                    # octant 2
                    return _ratio2angle(90, +1, y / x), _xy2r(x, y)
                else:
                    # octant 3
                    return _ratio2angle(180, -1, x / y), _xy2r(x, y)
        else:  # x < 0
            # octant 4, 5, 6, 7
            if y > 0:
                # octant 4, 5
                if y > -x:
                    # octant 4
                    return _ratio2angle(180, +1, -x / y), _xy2r(x, y)
                else:
                    # octant 5
                    return _ratio2angle(270, -1, y / -x), _xy2r(x, y)
            else:  # y < 0
                # octant 6, 7
                if -y < -x:
                    # octant 6
                    return _ratio2angle(270, +1, -y / -x), _xy2r(x, y)
                else:
                    # octant 7
                    return _ratio2angle(360, -1, -x / -y), _xy2r(x, y)

    def arcLength(self, major_axis, minor_axis, angle):
        """ determine the approximate length for an arc of 1 degree at the given angle
            on an ellipse with major_axis and minor_axis
            """
        # do it the long way to see if it works, the long way is this (see https://en.wikipedia.org/wiki/Ellipse):
        #  co-ords of a point on the circumference of an ellipse with major-axis a and minor-axis b at angle t:
        #   x,y = a*cos(t), b*sin(t)
        #  and for t+dt:
        #   x',y' = a*cos(t+dt), b*sin(t+dt)
        #  if dt is small, the arc length can be considered a straight line, so the arc length d is:
        #   sqrt((x-x')**2 + (y-y')**2)
        #  which reduces to:
        #   sqrt((a*(sin(t)-sin(t+dt)))**2 + (b*(cos(t)-cos(t+dt)))**2)
        #  sin(t)-sin(t+dt) and cos(t)-cos(t+dt) can be looked up
        #  due to ellipse symmetry we only need consider angles 0..90
        #  angle 0 is x=major_axis, y=0, angle 90 is x=0, y=minor_axis, going anti-clockwise
        if angle < 90:
            # q1 - as is
            pass
        elif angle < 180:
            # q2 - reversed
            angle = 90 - (angle - 90)
        elif angle < 270:
            # q3 - offset by 180
            angle = angle - 180
        else:
            # q4 - reversed
            angle = 90 - (angle - 270)
        if angle < 45:
            delta = +1
        else:
            delta = -1
        t = math.radians(angle)
        dt = math.radians(angle + delta)
        sint = math.sin(t) - math.sin(dt)
        cost = math.cos(t) - math.cos(dt)
        a = minor_axis * sint
        b = major_axis * cost
        d = math.sqrt(a * a + b * b)
        return d


class Ring:
    """ this class knows how to draw the marker and data rings according to its constructor parameters
        see description above for the overall target structure
        """

    NUM_RINGS = 9  # total rings in our complete code

    def __init__(self, centre_x, centre_y, width, frame, contrast, offset):
        # set constant parameters
        self.b = Codec.BITS  # how many bits in each ring
        self.w = width  # width of each ring in pixels
        self.f = frame  # where to draw it
        self.x = centre_x  # where the centre of the rings are
        self.y = centre_y  # ..

        # setup black/white luminance levels for drawing pixels,
        # contrast specifies the luminance range between black and white, 1=full luminance range, 0.5=half, etc
        # offset specifies how much to bias away from the mid-point, -ve is below, +ve above, number is a
        # fraction of the max range
        # we subtract half the required range from the offset mid-point for black and add it for white
        level_range = MAX_LUMINANCE * contrast / 2  # range relative to mid point
        level_centre = MID_LUMINANCE + (MAX_LUMINANCE * offset)
        self.black_level = int(round(max(level_centre - level_range, MIN_LUMINANCE)))
        self.white_level = int(round(min(level_centre + level_range, MAX_LUMINANCE)))

        # setup our angles look-up table such that get 1 pixel resolution on outermost ring
        scale = 2 * math.pi * width * Ring.NUM_RINGS
        self.angle_xy = Angle(scale).polarToCart
        self.edge = 360 / self.b  # the angle at which a bit edge occurs (NB: not an int)

        # stuff needed for the timing ring
        self.msb = 1 << (self.b - 1)  # the MSB of a code word
        self.bit_mask = self.msb - 1  # mask to isolate the code word bits

    def _pixel(self, x, y, colour):
        """ draw a pixel at x,y from the image centre with the given luminance and opaque,
            to mitigate pixel gaps in the circle algorithm we draw several pixels near x,y
            """
        x += self.x
        y += self.y
        self.f.putpixel(x, y, colour, True)
        self.f.putpixel(x + 1, y, colour, True)
        self.f.putpixel(x, y + 1, colour, True)

    def _point(self, x, y, bit):
        """ draw a point at offset x,y from our centre with the given bit (0 or 1) colour (black or white)
            bit can be 0 (draw 'black'), 1 (draw 'white'), -1 (draw max luminance), -2 (draw min luminance),
            any other value (inc None) will draw 'grey' (mid luminance)
            """
        if bit == 0:
            colour = self.black_level
        elif bit == 1:
            colour = self.white_level
        elif bit == -1:  # lsb is 1
            colour = MAX_LUMINANCE
        elif bit == -2:  # lsb is 0
            colour = MIN_LUMINANCE
        else:
            colour = MID_LUMINANCE
        self._pixel(x, y, colour)

    def timing(self, rings):
        """ given a set of data rings build the timing ring for them, the timing is only present to
            ensure there is a bit transition for every bit in the code, this aids the target decode
            as it ensures every ring boundary and every bit boundary is represented
        """

        # find out where the missing transitions are
        edges = 0
        for ring in rings:
            mask1 = self.msb
            mask2 = mask1 >> 1
            while mask1 != 0:
                if mask2 == 0:
                    # doing last bit, wrap to the first
                    mask2 = self.msb
                if ((ring & mask1 == 0) and (ring & mask2) == 0) or ((ring & mask1 != 0) and (ring & mask2) != 0):
                    # there is no edge here, leave gasp bit as is
                    pass
                else:
                    # there is an edge here
                    edges |= mask2
                mask1 >>= 1
                mask2 >>= 1
            last_ring = ring

        # for every 0 bit in edges we need a bit transition
        # we do that by inserting a 0 if the preceding bit was 1 or 1 if it was 0

        bits = 0
        mask1 = self.msb
        mask2 = 1
        while mask1 != 0:
            if mask2 == 0:
                # we've wrapped
                mask2 = self.msb
            prev = bits & mask2
            if edges & mask1 == 0:
                # need an edge here, set this bit different to the preceding bit
                if prev == 0:
                    # was 0, so put a 1 in
                    bits |= mask1
                else:
                    # was 1, so need a 0, already there
                    pass
            elif mask1 == 1:
                # this is the last bit and we do not need an edge, but we must not compromise the initial edge
                # so set this the opposite of the first bit
                if bits & self.msb == 0:
                    # first bit was 0, so set a 1 here
                    bits |= mask1
                else:
                    # first bit was 1, so set a 0 here (already there)
                    pass
            else:
                # no edge here, set this bit opposite to the same bit in the last ring
                if last_ring & mask1 == 0:
                    # was 0 in last ring, so set a 1 here
                    bits |= mask1
                else:
                    # was 1 in last ring, so set a 0 here, already there
                    pass
            mask1 >>= 1
            mask2 >>= 1

        # return the required timing bits
        return bits

    def border(self, ring_num):
        """ draw a thin black/white broken border at the given ring_num,
            it must be broken to stop it being detected as an outer radius edge,
            this is only intended as a visual marker to stop people cutting into important parts of the code
            """

        radius = ring_num * self.w

        scale = 2 * math.pi * radius  # step the angle such that 1 pixel per increment
        interval = int(round(scale) / 16)
        bit = 0
        for step in range(int(round(scale))):
            a = (step / scale) * 360
            x, y = self.angle_xy(a, radius)
            x = int(round(x))
            y = int(round(y))
            if (step % interval) == 0:
                bit ^= 1
            self._point(x, y, bit)

    def _draw(self, radius, bits):
        """ draw a ring at radius of bits, a 1-bit is white, 0 black,
            the bits are drawn big-endian and clockwise , i.e. MSB first (0 degrees), LSB last (360 degrees)
            """
        if radius <= 0:
            # special case - just draw a dot at x,y of the LSB colour of bits
            self._point(0, 0, bits & 1)
        else:
            msb = 1 << (self.b - 1)
            scale = 2 * math.pi * radius  # step the angle such that 1 pixel per increment
            for step in range(int(round(scale))):
                a = (step / scale) * 360
                x, y = self.angle_xy(a, radius)
                x = int(round(x))
                y = int(round(y))
                if a > 0:
                    segment = int(a / self.edge)
                else:
                    segment = 0
                mask = msb >> segment
                if bits & mask:
                    self._point(x, y, 1)
                else:
                    self._point(x, y, 0)

    def draw(self, ring_num, data_bits, ring_width):
        """ draw a data ring with the given data_bits and width,
            """
        start_ring = int(ring_num * self.w)
        end_ring = int(start_ring + ring_width * self.w)
        for radius in range(start_ring, end_ring):
            self._draw(radius, data_bits)

    def label(self, number):
        """ draw a 3 digit number at the top left edge of the rings in a size compatible with the ring width
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
        start_x = -self.x + self.w
        start_y = start_x
        place = -1
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
                                self._point(x + dx, y + dy, -2)  # draw at min luminance (ie. true black)
                    x += point_size
                x -= digit_size
                y += point_size

    def code(self, number, rings):
        """ draw the complete code for the given number and code-words
            the code-words must match the number, this function is the
            only one that knows the overall target ring structure
            """

        # draw the bullseye and its enclosing ring
        self.draw(0.0, -1, 2.0)
        self.draw(2.0, 0, 1.0)
        draw_at = 3.0

        # draw the data rings
        for ring in rings:
            self.draw(draw_at, ring, 1.0)
            draw_at += 1.0

        # draw the timing ring
        self.draw(draw_at, self.timing(rings), 1.0)
        draw_at += 1.0

        # draw the outer black/white rings
        self.draw(draw_at, 0, 1.0)
        self.draw(draw_at + 1.0, -1, 1.0)
        draw_at += 2.0

        if int(draw_at) != draw_at:
            raise Exception('number of target rings is not integral ({})'.format(draw_at))
        draw_at = int(draw_at)

        # draw a border
        self.border(draw_at)

        # safety check in case I forgot to update the constant
        if draw_at != Ring.NUM_RINGS:
            raise Exception('number of rings exported ({}) is not {}'.format(Ring.NUM_RINGS, draw_at))

        # draw a human readable label
        self.label(number)


class Frame:
    """ image frame buffer as a 2D array of luminance values,
        it uses the opencv library to read/write and modify images at the pixel level,
        its the source of images for the Transform functions
        """

    def __init__(self, read='.', write='.'):
        self.read_from = read
        self.write_to = write
        self.source = None
        self.buffer = None
        self.alpha = None
        self.max_x = None
        self.max_y = None

    def instance(self):
        """ return a new instance of self """
        return Frame(self.read_from, self.write_to)

    def new(self, width, height, luminance):
        """ prepare a new buffer of the given size and luminance """
        self.buffer = np.full((height, width), luminance, dtype=np.uint8)  # NB: numpy arrays follow cv2 conventions
        self.alpha = None
        self.max_x = width
        self.max_y = height
        self.source = 'internal'
        return self

    def set(self, buffer, source='internal'):
        """ set the given buffer (assumed to be a cv2 image) as the current image """
        self.buffer = buffer
        self.source = source
        self.max_x = self.buffer.shape[1]  # NB: cv2 x, y are reversed
        self.max_y = self.buffer.shape[0]  # ..

    def get(self):
        """ get the current image buffer """
        return self.buffer

    def opacity(self):
        """ get the current image alpha mask (if there is one) """
        return self.alpha

    def size(self):
        """ return the x,y size of the current frame buffer """
        return self.max_x, self.max_y

    def load(self, image_file):
        """ load frame buffer from an image file as a grey scale image """
        self.buffer = cv2.imread('{}/{}'.format(self.read_from, image_file), cv2.IMREAD_GRAYSCALE)
        self.alpha = None
        self.max_x = self.buffer.shape[1]  # NB: cv2 x, y are reversed
        self.max_y = self.buffer.shape[0]  # ..
        self.source = image_file

    def unload(self, image_file, suffix=None):
        """ unload the frame buffer to a PNG image file
            returns the file name written
            """
        if len(self.buffer.shape) == 2:
            # its a grey scale image, convert to RGBA
            image = self.colourize()
        else:
            # assume its already colour
            image = self.buffer
        filename, ext = os.path.splitext(image_file)
        if suffix is not None:
            filename = '{}/_{}_{}.png'.format(self.write_to, filename, suffix)
        else:
            filename = '{}/{}.png'.format(self.write_to, filename)
        cv2.imwrite(filename, image)
        return filename

    def show(self, title='title'):
        """ show the current buffer """
        cv2.imshow(title, self.buffer)
        cv2.waitKey(0)

    def setpixel(self, x, y, value1, value2):
        """ like putpixel except if it is already value1 we put value2,
            only valid on a colour image
            """
        if x < 0 or x >= self.max_x or y < 0 or y >= self.max_y:
            return
        was = self.buffer[y, x]
        now = (was[0], was[1], was[2])
        if now == value1:
            self.buffer[y, x] = value2
        else:
            self.buffer[y, x] = value1

    def getpixel(self, x, y):
        """ get the pixel value at x,y
            nb: cv2 x,y is reversed from our pov
            """
        if x < 0 or x >= self.max_x or y < 0 or y >= self.max_y:
            return None
        else:
            return self.buffer[y, x]  # NB: cv2 x, y are reversed

    def putpixel(self, x, y, value, with_alpha=False):
        """ put the pixel of value at x,y
            value may be a greyscale value or a colour tuple
            """
        if x < 0 or x >= self.max_x or y < 0 or y >= self.max_y:
            return
        self.buffer[y, x] = value  # NB: cv2 x, y are reversed
        if with_alpha:
            if self.alpha is None:
                self.alpha = np.full((self.max_y, self.max_x), TRANSPARENT,  # default is fully transparent
                                     dtype=np.uint8)  # NB: numpy arrays follow cv2 conventions
            self.alpha[y, x] = OPAQUE  # want foreground only for our pixels

    def inimage(self, x, y, r):
        """ determine if the points radius R and centred at X, Y are within the image """
        if (x - r) < 0 or (x + r) >= self.max_x or (y - r) < 0 or (y + r) >= self.max_y:
            return False
        else:
            return True

    def incolour(self):
        """ turn self into a colour image if its not already """
        self.buffer = self.colourize()
        return self

    def colourize(self):
        """ make our grey image into an RGB one, or RGBA if there is an alpha channel,
            returns the image array with 3 or 4 channels,
            its a no-op if we're not a grey image
            """
        if len(self.buffer.shape) == 2:
            if self.alpha is not None:
                # got an alpha channel
                image = cv2.merge([self.buffer, self.buffer, self.buffer, self.alpha])
            else:
                image = cv2.merge([self.buffer, self.buffer, self.buffer])
        else:
            image = self.buffer
        return image


class Transform:
    """ this class provides a thin wrapper on various opencv functions """

    # mnemonics for annotate object types
    LINE = 0
    CIRCLE = 1
    RECTANGLE = 2
    PLOTX = 3
    PLOTY = 4
    POINTS = 5
    TEXT = 6

    def blur(self, source, size=3):
        """ apply a median blur to the given cv2 image with a kernel of the given size """
        target = source.instance()
        target.set(cv2.medianBlur(source.get(), size))
        return target

    def upheight(self, source, new_height):
        """ upsize the given image such that its height (y) is at least that given,
            the width is preserved,
            """
        width, height = source.size()
        if height >= new_height:
            # its already big enough
            return source
        target = source.instance()
        target.set(cv2.resize(source.get(), (width, new_height), interpolation=cv2.INTER_LINEAR))
        return target

    def downwidth(self, source, new_width):
        """ downsize the given image such that its width (x) is at least that given,
            the height is preserved,
            """
        width, height = source.size()
        if width <= new_width:
            # its already small enough
            return source
        target = source.instance()
        target.set(cv2.resize(source.get(), (new_width, height), interpolation=cv2.INTER_AREA))
        return target

    def downsize(self, source, new_size):
        """ downsize the given image such that either its width or height is at most that given,
            the aspect ratio is preserved,
            """
        width, height = source.size()
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
        target = source.instance()
        target.set(cv2.resize(source.get(), (new_width, new_height), interpolation=cv2.INTER_AREA))
        return target

    def crop(self, source, start_x, start_y, end_x, end_y):
        """ crop the given rectangle out of the given image,
            the returned rectangle contains pixels from start_x..end_x-1 and start_y..end_y-1,
            if the given rectangle overflows the image edge it is truncated to the edge,
            a rectangle completely outside the image returns an empty image
            """

        # validate/adjust given rectangle
        old_x, old_y = source.size()
        if start_x < 0:
            # shrink to image edge
            start_x = 0
        if start_x > old_x:
            # no can do
            return source.instance()
        if end_x < start_x:
            # no can do
            return source.instance()
        if end_x > old_x:
            # shrink to image edge
            end_x = old_x
        if start_y < 0:
            # shrink to image edge
            start_y = 0
        if start_y > old_y:
            return source.instance()
        if end_y < start_y:
            # no can do
            return source.instance()
        if end_y > old_y:
            # shrink to image edge
            end_y = old_y

        # crop the buffer
        old_buffer = source.get()
        new_buffer = old_buffer[start_y:end_y, start_x:end_x]

        # return new cropped image
        target = source.instance()
        target.set(new_buffer)
        return target

    def merge(self, image_1, image_2):
        """ returns an image that is the combination of the given images
            """
        target = image_1.instance()
        target.set(cv2.addWeighted(image_1.get(), 1.0, image_2.get(), 1.0, 0.0))
        return target

    def copy(self, source):
        """ make a copy of the given image """
        target = source.instance()
        target.set(source.get().copy())
        return target

    def fill(self, source, start, end, colour):
        """ draw a filled rectangle """

        image = source.get()
        image = cv2.rectangle(image, start, end, colour, -1)

    def blobs(self, source, threshold, circularity, convexity, inertia, area, gaps, colour):
        """ find bright blobs in the given image,
            parameters are those required by opencv.SimpleBlobDetector_create,
            see https://learnopencv.com/blob-detection-using-opencv-python-c/
            if None that parameter is not set,
            each is a multi-element tuple (colour),
            returns a keypoints array (which may be empty), each keypoint has:
                .pt[1] = x co-ord of the centre
                .pt[0] = y co-ord of centre
                .size = diameter of blob
            all floats
            """

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        if threshold is not None:
            if threshold[0] is not None:
                params.minThreshold = threshold[0]  # default is 10
            if threshold[1] is not None:
                params.maxThreshold = threshold[1]  # default is 220
            if threshold[2] is not None:
                params.thresholdStep = threshold[2]  # default is 10
        if circularity is not None:
            params.filterByCircularity = True
            if circularity[0] is not None:
                params.filterByCircularity = True
                params.minCircularity = circularity[0]
            if circularity[1] is not None:
                params.filterByCircularity = True
                params.maxCircularity = circularity[1]  # default 3.4
        if convexity is not None:
            if convexity[0] is not None:
                params.filterByConvexity = True
                params.minConvexity = convexity[0]
            if convexity[1] is not None:
                params.filterByConvexity = True
                params.maxConvexity = convexity[1]  # default 3.4
        if inertia is not None:
            if inertia[0] is not None:
                params.filterByInertia = True
                params.minInertiaRatio = inertia[0]
            if inertia[1] is not None:
                params.filterByInertia = True
                params.maxInertiaRatio = inertia[1]  # default 3.4
        if area is not None:
            if area[0] is not None:
                params.filterByArea = True
                params.minArea = area[0]
            if area[1] is not None:
                params.filterByArea = True
                params.maxArea = area[1]
        if gaps is not None:
            if gaps[0] is not None:
                params.minDistBetweenBlobs = gaps[0]  # default is 10
            if gaps[1] is not None:
                params.minRepeatability = gaps[1]  # default is 2
        if colour is not None:
            params.filterByColor = True
            params.blobColor = colour

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs
        return detector.detect(source.get())

    def drawKeypoints(self, source, blobs, colour=(0, 0, 255)):
        """ draw cv2 'keypoints' of source and return a new image showing them,
            keypoints is a structure created by some cv2 operation (e.g. simple blob detector)
            """
        target = source.instance()
        target.set(cv2.drawKeypoints(source.get(), blobs, np.array([]), colour,
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
        return target

    def warpPolar(self, source, centre_x, centre_y, radius, width, height):
        """ warp the disc defined by centre_x/y and radius in the source image
            into a rectangular image of width (angle) x height (radius)
            """
        polar = (cv2.warpPolar(source.get(), (height, width),
                               (centre_x, centre_y), radius,
                               cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR))

        # cv2 produces an image of width=radius, height=angle,
        # so we rotate it 90 degrees clockwise to get width=angle, height=radius
        # ToDo: this could be optimised out by tweaking the rest of my algorithms to work radius x angle
        #       the reason I work angle x radius is I only discovered warpPolar after I'd done it all DIY!
        rotated = cv2.transpose(polar)

        target = source.instance()
        target.set(rotated)

        return target

    def canny(self, source, min_val, max_val, size=3):
        """ perform a 'canny' edge detection on given source,
            min-val and max-val are the hysteresis thresholding values (see openCV documentation),
            size is the gaussian filter size,
            returns an image of the edges
            """
        target = source.instance()
        target.set(cv2.Canny(source.get(), min_val, max_val, size))
        return target

    def edges(self, source, xorder, yorder, size=3, inverted=True):
        """ perform an edge detection filter on the given image and return a new image of the result,
            if inverted is True we find white to black edges, else black to white
            xorder=1, yorder=0 will detect horizontal edges,
            xorder=0, yorder=1 will detect vertical edges,
            xorder=1, yorder=1 will detect both
            """
        target = source.instance()
        if inverted:
            target.set(cv2.Sobel(255 - source.get(), -1, xorder, yorder, size))
        else:
            target.set(cv2.Sobel(source.get(), -1, xorder, yorder, size))
        return target

    def contours(self, source):
        """ get the contours in the given binary image,
            it returns a list of contours as an array of vectors of points,
            it does not return or seek a hierarchy
            """
        found, _ = cv2.findContours(source.get(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return found

    def threshold(self, source, thresh=None, size=None):
        """ turn source image into a binary image,
            returns the binarised image,
            if thresh is not None an absolute threshold is applied,
            if size is not None an adaptive threshold with that block size is applied
            otherwise an automatic Otsu threshold is applied
            """
        target = source.instance()
        if thresh is not None:
            _, buffer = cv2.threshold(source.get(), thresh, MAX_LUMINANCE, cv2.THRESH_BINARY)
        elif size is not None:
            buffer = cv2.adaptiveThreshold(source.get(), MAX_LUMINANCE,
                                           cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, size, 0.0)
        else:
            _, buffer = cv2.threshold(source.get(), MIN_LUMINANCE, MAX_LUMINANCE,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        target.set(buffer)
        return target

    def label(self, source, keypoint, colour=(0, 0, 255), title=None):
        """ return an image with a coloured ring around the given key point in the given image
            and a textual title at the key point centre if one is given,
            keypoint is a list of centre x,y and circle radius
            """
        objects = []
        org = (int(round(keypoint[0])), int(round(keypoint[1])))
        objects.append({"colour": colour,
                        "type": self.CIRCLE,
                        "centre": org,
                        "radius": int(round(keypoint[2]))})
        if title is not None:
            objects.append({"colour": colour,
                            "type": self.TEXT,
                            "start": org,
                            "size": 0.5,
                            "text": title})
        return self.annotate(source, objects)

    def annotate(self, source, objects):
        """ annotate an image with 'objects' (an array), each object is a dictionary of:
                colour (rgb),
                type, - line, circle, rectangle, plot-x, plot-y, text
                start or centre position (x,y) or first plot x,
                end position (x,y) or radius or plot-x y points or plot-y x points or font-size,
                text string for the text type
            wherever point arithmetic is involved, out of range values are wrapped
            returns the source updated, it'll be converted to colour if required,
            """

        max_x, max_y = source.size()

        def _x(fx):
            if fx is None:
                return None
            return int(round(fx) % max_x)

        def _y(fy):
            if fy is None:
                return None
            return int(round(fy) % max_y)

        def _int(pt):
            return _x(pt[0]), _y(pt[1])

        def alt_colour(colour):
            return colour[0] | 128, colour[1] | 128, colour[2] | 128

        source.incolour()
        image = source.get()

        for obj in objects:
            if obj["type"] == self.LINE:
                start = obj["start"]
                end = obj["end"]
                if int(round(start[0])) > int(round(end[0])):
                    # wraps in x, split it: start to max_x is part 1, 0 to end is part 2
                    start1 = [start[0], start[1]]
                    end1 = [max_x - 1, end[1]]
                    start[0] = 0
                    image = cv2.line(image, _int(start1), _int(end1), obj["colour"], 1)
                if int(round(start[1])) > int(round(end[1])):
                    # wraps in y, split it: start to max_y is part 1, 0 to end is part 2
                    start1 = [start[0], start[1]]
                    end1 = [end[0], max_y - 1]
                    start[1] = 0
                    image = cv2.line(image, _int(start1), _int(end1), obj["colour"], 1)
                image = cv2.line(image, _int(start), _int(end), obj["colour"], 1)
            elif obj["type"] == self.CIRCLE:
                image = cv2.circle(image, _int(obj["centre"]), int(round(obj["radius"])), obj["colour"], 1)
            elif obj["type"] == self.RECTANGLE:
                image = cv2.rectangle(image, _int(obj["start"]), _int(obj["end"]), obj["colour"], 1)
            elif obj["type"] == self.PLOTX:
                colour = obj["colour"]
                points = obj["points"]
                if points is not None:
                    x = _x(obj["start"])
                    for pt in range(len(points)):
                        y = _y(points[pt])
                        if y is not None:
                            source.setpixel(_x(x + pt), _y(y), colour, alt_colour(colour))
            elif obj["type"] == self.PLOTY:
                colour = obj["colour"]
                points = obj["points"]
                if points is not None:
                    y = _y(obj["start"])
                    for pt in range(len(points)):
                        x = _x(points[pt])
                        if x is not None:
                            source.setpixel(_x(x), _y(y + pt), colour, alt_colour(colour))
            elif obj["type"] == self.POINTS:
                colour = obj["colour"]
                points = obj["points"]
                for pt in points:
                    source.setpixel(_x(pt[0]), _y(pt[1]), colour, alt_colour(colour))
            elif obj["type"] == self.TEXT:
                image = cv2.putText(image, obj["text"], _int(obj["start"]),
                                    cv2.FONT_HERSHEY_SIMPLEX, obj["size"], obj["colour"], 1, cv2.LINE_AA)
            else:
                raise Exception('Unknown object type {}'.format(obj["type"]))
        source.set(image)
        return source


class Target:
    """ struct to hold info about a Scan detected target """

    def __init__(self, number, doubt, centre_x, centre_y, target_size, blob_size):
        self.centre_x = centre_x
        self.centre_y = centre_y
        self.blob_size = blob_size
        self.number = number
        self.doubt = doubt
        self.target_size = target_size


class Scan:
    """ this class provides functions to scan an image and extract any codes in it,
        algorithm summary:
            1. blob detect our bullseye
            2. warp polar around its centre to create a cartesian co-ordinate rectangle
            3. find the inner and outer edges in that
            4. 'flatten' the image such that the inner edge is straight
            5. edge detect all the ring and bit boundaries
            6. extract the cells for each bit
            7. decode those bits
        the algorithm has been developed more by experimentation than theory!
        """

    # tuning constants
    MIN_BLOB_SEPARATION = 5  # smallest blob within this distance of each other are dropped
    BLOB_RADIUS_STRETCH = 2.5  # how much to stretch blob radius to ensure always cover the whole lot
    MIN_PIXELS_PER_RING = 4  # upsize flattened image such that this constraint is met
    MIN_PIXELS_PER_BIT = 4  # stretch angles param such that this constraint is met
    BRIGHTER_THRESHOLD = 0.7  # the threshold at which a bright pixel is considered brighter
    SAMPLE_WIDTH_FACTOR = 0.4  # fraction of a bit width that is probed for a luminance value
    SAMPLE_HEIGHT_FACTOR = 0.3  # fraction of a ring height that is probed for a luminance value

    # luminance threshold when getting the mean luminance of an image, pixels below this are ignored
    MIN_EDGE_THRESHOLD = int(MAX_LUMINANCE * 0.1)

    # minimum luminance relative to the mean for an edge pixel to qualify as an edge when following it
    EDGE_THRESHOLD = 1.2

    # direction sensitive tuning params (each is a tuple of ring(0),bit(1) tuning options)

    # max drift in y of ring edge or x in bit edges as approx tan of the angle (1.0==45 degrees, 0.12==7 degrees)
    # edges that exceed this are split into segments that do not, the assumption is that blurry edges in small
    # images have run into each other, making two edges appear as one
    EDGE_DRIFT_LIMIT = (0.2, 0.12)
    MIN_DRIFT_EDGE_LENGTH = 4  # do not check for drift until an edge is at least this long (in pixels)

    # edges shorter than this are discarded as noise
    # min length of a ring/bit edge as a fraction of the nominal bit/ring width
    MIN_EDGE_LENGTH = (0.3, 0.3)
    MIN_EDGE_LENGTH_PIXELS = 3  # minimum length in pixels if the above is less than this

    # overlapping edges closer together than this are merged
    # min width of a ring/bit edge as a fraction of the nominal bit/ring width
    MIN_EDGE_WIDTH = (0.2, 0.1)
    MIN_EDGE_WIDTH_PIXELS = 2  # minimum width in pixels if the above is less than this

    # when merging edges, only those below this length are merged and the merged length cannot exceed it either
    # as a fraction of the image width (for rings) or height (for bits)
    MERGE_LENGTH_LIMIT = (0.6, 0.6)

    # video modes image height
    VIDEO_SD = 480
    VIDEO_HD = 720
    VIDEO_FHD = 1080
    VIDEO_2K = 1152
    VIDEO_4K = 2160

    # debug options
    DEBUG_NONE = 0  # no debug output
    DEBUG_IMAGE = 1  # just write debug annotated image files
    DEBUG_VERBOSE = 2  # do everything - generates a *lot* of output

    # scanning directions, these are bit masks, so they can be added to identify multiple directions
    TOP_DOWN = 1  # top-down y scan direction      (looking for radius edges or ring edges)
    BOTTOM_UP = 2  # bottom-up y scan direction     (..)
    LEFT_TO_RIGHT = 4  # left-to-right x scan direction (looking for bit edges)
    RIGHT_TO_LEFT = 8  # right-to-left x scan direction (..)
    NUM_DIRECTIONS = 16  # for use if creating direction arrays

    # useful multi-directions
    UP_AND_DOWN = TOP_DOWN + BOTTOM_UP
    LEFT_AND_RIGHT = LEFT_TO_RIGHT + RIGHT_TO_LEFT

    # our target shape
    NUM_RINGS = Ring.NUM_RINGS  # total number of rings in the whole code
    NUM_BITS = Codec.BITS  # total number of bits in a ring

    # context type identifiers
    CONTEXT_RING = 0
    CONTEXT_BIT = 1

    # edge type identifiers
    LEADING_EDGE = 1  # black-to-white transition
    TRAILING_EDGE = 0  # white-to-black transition

    # ring numbers of flattened image
    INNER_POINT = 0  # central bullseye
    INNER_WHITE = 1  # used for white level detection
    INNER_BLACK = 2  # used for black level detection (option 1)
    DATA_RING_1 = 3
    DATA_RING_2 = 4
    DATA_RING_3 = 5
    TIMING_RING = 6
    OUTER_BLACK = 7  # used for black level detection (option 2)
    OUTER_WHITE = 8
    OUTER_LIMIT = 9  # end of target area

    class Stepper:
        """ this class is used to step through an image in a direction,
            use of this class unifies any scanning loops in any direction and handles all wrapping issues
            """

        x = None  # current x coordinate, these wrap at image edge
        y = None  # current y coordinate, these do not wrap at image edge
        x_multiplier = None  # the x coord multiplier for the current direction
        y_multiplier = None  # the y coord multiplier for the current direction
        max_x = None  # the image size in x
        max_y = None  # the image size in y
        step = None  # how much the co-ordinates are stepping by for each increment
        steps = None  # the maximum number of steps to do
        iterations = None  # how many steps have been done so far

        def __init__(self, direction, max_x, max_y, step=1, steps=None):
            self.max_x = max_x
            self.max_y = max_y
            self.step = step
            self.iterations = 0
            self.steps = steps
            self.x = 0
            self.y = 0
            self.x_multiplier = 0
            self.y_multiplier = 0
            if (direction & Scan.TOP_DOWN) != 0:
                if self.steps is None:
                    self.steps = int(round(self.max_y / self.step))
                self.y_multiplier = +1
            elif (direction & Scan.BOTTOM_UP) != 0:
                if self.steps is None:
                    self.steps = int(round(self.max_y / self.step))
                self.y = max_y - 1
                self.y_multiplier = -1
            elif (direction & Scan.LEFT_TO_RIGHT) != 0:
                if self.steps is None:
                    self.steps = int(round(self.max_x / self.step))
                self.x_multiplier = +1
            elif (direction & Scan.RIGHT_TO_LEFT) != 0:
                if self.steps is None:
                    self.steps = int(round(self.max_x / self.step))
                self.x = max_x - 1
                self.x_multiplier = -1
            else:
                raise Exception('illegal direction {}'.format(direction))

        def cycles(self):
            """ return the number of steps left to do """
            return self.steps - self.iterations

        def reset(self, x=None, y=None, step=None, steps=None):
            """ reset parameters, None means leave as is,
                whatever is given here is returned as the first subsequent .next() then
                steps resume from there, the number of iterations is always reset to 0
                """
            if x is not None:
                self.x = x
            if y is not None:
                self.y = y
            if step is not None:
                self.step = step
            if steps is not None:
                self.steps = steps
            self.iterations = 0

        def next(self):
            """ get the (first or) next x,y co-ord pair,
                returns an x,y tuple or None if no more
                """

            # check we have something to return
            if self.y is None:
                # we've gone off the top or bottom of the image
                return None
            if self.iterations >= self.steps:
                # we've done enough steps
                return None

            # note result for caller
            xy = [self.x, self.y]

            # set the next co-ords to return
            self.x = int(round((self.x + (self.x_multiplier * self.step)) % self.max_x))
            self.y = int(round(self.y + (self.y_multiplier * self.step)))
            if self.y < 0 or self.y >= self.max_y:
                # we've gone off the top or bottom of the image
                # so next step must report
                self.y = None

            # note we've done another step
            self.iterations += 1

            return xy

    class Edge:
        """ structure to hold detected edge information """

        def __init__(self, position=0, length=0, span=0, start=0, width=None, where=None, edge=None):
            self.position = position  # where its midpoint is (an X or Y co-ord)
            self.length = length  # how long it is (in pixels)
            self.span = span  # how wide it is in its bright band (in pixels)
            self.start = start  # where it starts
            self.width = width  # how wide it is in total (in pixels)
            self.where = where  # where it was found (an X or Y co-ord actually probed)
            self.edge = edge  # either leading (back-to-white) or trailing (white-to-black) or None
            if self.width is None:
                self.width = self.span
            if self.where is None:
                self.where = int(round(self.position))

        def __str__(self):
            return '(at {} as {:.2f} from {:.2f} for {}, s:{:.2f}, w:{:.2f}, e:{})'. \
                format(self.where, self.position, self.start, self.length, self.span, self.width, self.edge)

    class Target:
        """ structure to hold detected target information """

        def __init__(self, centre_x, centre_y, size, scale, flattened, slices):
            self.centre_x = centre_x  # x co-ord of target in original image
            self.centre_y = centre_y  # y co-ord of target in original image
            self.size = size  # blob size originally detected by the blob detector
            self.scale = scale  # scale of target in original image
            self.image = flattened  # the flattened image of the target
            self.slices = slices  # list of ring edges inside each bit

    class Reject:
        """ struct to hold info about rejected targets """

        def __init__(self, centre_x, centre_y, blob_size, target_size, reason):
            self.centre_x = centre_x
            self.centre_y = centre_y
            self.blob_size = blob_size
            self.target_size = target_size
            self.reason = reason

    class EdgePoint:
        """ struct to hold info about an edge point """

        def __init__(self, midpoint=0, first=0, last=0, bright_first=0, bright_last=0, where=0):
            # NB: the co-ordinates given here may be out of range if they have wrapped,
            #     its up the user of this object to deal with it
            self.midpoint = midpoint  # midpoint co-ord (X or Y)
            self.first = first  # min co-ord in the edge width
            self.last = last  # max co-ord in the edge width
            self.bright_first = bright_first  # min co-ord of the 'bright' part
            self.bright_last = bright_last  # max co-ord of the 'bright' part
            self.where = where  # the other co-ord of the point (Y or X)

        def __str__(self):
            return '(w:{}, m:{}, f:{}, l:{}, bf:{}, bl:{})'. \
                format(self.where, self.midpoint, self.first, self.last, self.bright_first, self.bright_last)

    class Kernel:
        """ an iterator that returns a series of x,y co-ordinates for a 'kernel' in a given direction,
            direction is an 'angle' with 'left-to-right' being the natural angle for the kernel, and
            other directions rotating clockwise in 90 degree steps, through top-down, right-to-left
            and bottom-up
            """

        def __init__(self, kernel, direction):

            self.kernel = kernel

            if direction == Scan.LEFT_TO_RIGHT:
                self.delta_x = 1
                self.delta_y = 1
                self.swapped = False
            elif direction == Scan.RIGHT_TO_LEFT:
                self.delta_x = -1
                self.delta_y = 1
                self.swapped = False
            elif direction == Scan.BOTTOM_UP:
                self.delta_x = 1
                self.delta_y = -1
                self.swapped = True
            elif direction == Scan.TOP_DOWN:
                self.delta_x = 1
                self.delta_y = 1
                self.swapped = True
            else:
                raise Exception('illegal direction: {}'.format(direction))

        def __iter__(self):
            # init and return iterator
            self.next = 0
            return self

        def __next__(self):
            # return next tuple or raise StopIteration
            if self.next < len(self.kernel):
                kx = self.kernel[self.next][0]
                ky = self.kernel[self.next][1]
                self.next += 1
                if self.swapped:
                    dx = ky
                    dy = kx
                else:
                    dx = kx
                    dy = ky
                tx = dx * self.delta_x
                ty = dy * self.delta_y
                return tx, ty
            else:
                raise StopIteration

    class Context:
        """ this struct holds all the context that is dependant on direction and the image size being processed,
            its created once for each target, it creates a parameter set for a direction (or direction pair),
            the properties set here allow the processing logic to be direction agnostic,
            they only need to evaluate expressions using the properties set here
            """

        # prefix for log messages
        prefix = None
        type = None  # descriptive type of the edge (ring or bit) for log messages and image names
        context = None  # type of context (CONTEXT_RING or CONTEXT_BIT)

        # image limits
        max_x = None
        max_y = None

        # co-ordinate limits, scan-direction is the main direction, cross-direction is 90 degrees to it
        # these are either max_x-1 or max_y-1
        max_scan_coord = None
        max_cross_coord = None

        # in an ideal world the length of a ring/bit edge
        nominal_length = None

        # in an ideal world the gap between ring/bit edges
        nominal_width = None

        # indices into a [x,y] tuple for the scanning co-ord and the cross co-ord
        # these are offsets into an [x, y] array/tuple to access the in-direction co-ordinate
        # or the cross-direction co-ordinate, as they are 0 or 1 and mutually exclusive they can also
        # be used as multipliers for x and y in direction agnostic loops, e.g.
        # x += (something * cross_coord) will only update x if x is the scanning direction, and
        # x += (something * cross_coord * scan_multiplier) will only update it in the scanning direction
        scan_coord = None
        cross_coord = None

        # multiplier for scanning loop coordinates, either +1 or -1, use as:
        #   xy[scan_coord] += something * scan_multiplier
        scan_multiplier = None

        # tuning parameters
        min_length = None  # edges shorter than this are ignored (fraction of nominal width)
        min_width = None  # edges closer than this are combined (fraction of nominal width)
        drift_limit = None  # edges that drift 'sideways' by more than this (relative to width) are split
        length_limit = None  # only combine edges below this limit, as a fraction of image height/width

        # edge detector params (for 0,1 horizontal or 1,0 vertical edges, or 1,1 for both)
        x_order = None
        y_order = None

        # wrapping constraints
        allow_edge_wrap = None  # edges are allowed to wrap around image limits (e.g. rings)
        allow_scan_wrap = None  # scanning is allowed to wrap around image limits (e.g. bits)

        # kernel matrix to use to detect connected neighbours when following edges
        # this is set in processing logic, not here as it is dependant on more than just the direction
        kernel = None

        # which edge transitions are required
        b2w = None  # True for black-to-white
        w2b = None  # True for white-to-black

        # alternative scan directions relative to the main (scan) direction
        scan_direction = None  # 0 degress
        opposite_direction = None  # 180 degrees
        backward_direction = None  # -90 degrees
        forward_direction = None  # +90 degrees

        def __init__(self, target, direction):

            self.max_x, self.max_y = target.size()

            if (direction & Scan.UP_AND_DOWN) != 0:
                # stuff common to TOP_DOWN and BOTTOM_UP scanning (ring and radius edges)

                self.context = Scan.CONTEXT_RING
                self.type = 'ring'

                # coordinate limits
                self.max_scan_coord = self.max_y - 1
                self.max_cross_coord = self.max_x - 1

                # in an ideal world the length of a ring edge (we see these across the tops/bottoms of bits)
                self.nominal_length = self.max_x / Scan.NUM_BITS

                # in an ideal world the gap between ring edges
                self.nominal_width = self.max_y / Scan.NUM_RINGS

                # indices into a [x,y] tuple for the scanning co-ord and the cross co-ord
                self.scan_coord = 1  # ==y
                self.cross_coord = 0  # ==x

                # tuning parameters
                self.min_length = max(self.nominal_length * Scan.MIN_EDGE_LENGTH[0], Scan.MIN_EDGE_LENGTH_PIXELS)
                self.min_width = max(self.nominal_width * Scan.MIN_EDGE_WIDTH[0], Scan.MIN_EDGE_WIDTH_PIXELS)
                self.drift_limit = Scan.EDGE_DRIFT_LIMIT[0]
                self.length_limit = self.max_x * Scan.MERGE_LENGTH_LIMIT[0]

                # edge detector params (for horizontal edges)
                self.x_order = 0
                self.y_order = 1

                # wrapping constraints
                self.allow_edge_wrap = True
                self.allow_scan_wrap = False

                # cross directions
                self.backward_direction = Scan.RIGHT_TO_LEFT
                self.forward_direction = Scan.LEFT_TO_RIGHT

                # which edge transitions are required (one or the other or both)
                if (direction & Scan.TOP_DOWN) != 0:
                    self.b2w = True
                else:
                    self.b2w = False
                if (direction & Scan.BOTTOM_UP) != 0:
                    self.w2b = True
                else:
                    self.w2b = False

            elif (direction & Scan.LEFT_AND_RIGHT) != 0:
                # stuff common to left-to-right and right-to-left scanning (bit edges)

                self.context = Scan.CONTEXT_BIT
                self.type = 'bit'

                # coordinate limits
                self.max_scan_coord = self.max_x - 1
                self.max_cross_coord = self.max_y - 1

                # in an ideal world the length of a bit edge
                self.nominal_length = self.max_y / Scan.NUM_RINGS

                # in an ideal world the gap between bit edges
                self.nominal_width = self.max_x / Scan.NUM_BITS

                # indices into a [x,y] tuple for the scanning co-ord and the cross co-ord
                self.scan_coord = 0  # ==x
                self.cross_coord = 1  # ==y

                # tuning parameters
                self.min_length = max(self.nominal_length * Scan.MIN_EDGE_LENGTH[1], Scan.MIN_EDGE_LENGTH_PIXELS)
                self.min_width = max(self.nominal_width * Scan.MIN_EDGE_WIDTH[1], Scan.MIN_EDGE_WIDTH_PIXELS)
                self.drift_limit = Scan.EDGE_DRIFT_LIMIT[1]
                self.length_limit = self.max_y * Scan.MERGE_LENGTH_LIMIT[1]

                # edge detector params (for vertical edges)
                self.x_order = 1
                self.y_order = 0

                # wrappng constraints
                self.allow_edge_wrap = False
                self.allow_scan_wrap = True

                # same for left and right
                self.backward_direction = Scan.BOTTOM_UP
                self.forward_direction = Scan.TOP_DOWN

                # which edge transitions are required (one or the other or both)
                if (direction & Scan.LEFT_TO_RIGHT) != 0:
                    self.b2w = True
                else:
                    self.b2w = False
                if (direction & Scan.RIGHT_TO_LEFT) != 0:
                    self.w2b = True
                else:
                    self.w2b = False

            else:
                raise Exception('illegal direction {}'.format(direction))

            if (direction & Scan.TOP_DOWN) != 0:
                # NB: This takes priority if doing up-and-down
                # context is looking for ring or radius edges top-down
                self.prefix = 'ring-down'
                # alternative directions relative to this direction
                self.scan_direction = Scan.TOP_DOWN
                self.opposite_direction = Scan.BOTTOM_UP
                # multiplier when scanning
                self.scan_multiplier = 1

            elif (direction & Scan.BOTTOM_UP) != 0:
                # NB: This is ignored if doing up-and-down
                # context is looking for ring or radius edges bottom-up
                self.prefix = 'ring-up'
                # alternative directions relative to this direction
                self.scan_direction = Scan.BOTTOM_UP
                self.opposite_direction = Scan.TOP_DOWN
                # multiplier when scanning
                self.scan_multiplier = -1

            elif (direction & Scan.LEFT_TO_RIGHT) != 0:
                # NB: This takes priority if doing left-and-right
                # context is looking for bit edges left-to-right
                self.prefix = 'bit-left'
                # alternative directions relative to this direction
                self.scan_direction = Scan.LEFT_TO_RIGHT
                self.opposite_direction = Scan.RIGHT_TO_LEFT
                # multiplier when scanning
                self.scan_multiplier = 1

            elif (direction & Scan.RIGHT_TO_LEFT) != 0:
                # NB: This is ignored if doing left-and-right
                # context is looking for bit edges right-to-left
                self.prefix = 'bit-right'
                # alternative directions relative to this direction
                self.scan_direction = Scan.RIGHT_TO_LEFT
                self.opposite_direction = Scan.LEFT_TO_RIGHT
                # multiplier when scanning
                self.scan_multiplier = -1

            else:
                # can't get here!
                raise Exception('unreachable code reached!')

    def __init__(self, code, frame, angles=360, video_mode=VIDEO_FHD, debug=DEBUG_NONE, log=None):
        """ code is the code instance defining the code structure,
            frame is the frame instance containing the image to be scanned,
            angles is the angular resolution to use
            """

        # params
        self.angle_steps = angles  # angular resolution when 'projecting'
        self.video_mode = video_mode  # actually the downsized image height
        self.original = frame
        self.coder = code  # class to decode what we find

        # stretch angle steps such that each bit width is an odd number of pixels (so there is always a middle)
        self.angle_steps = max(self.angle_steps, Scan.NUM_BITS * Scan.MIN_PIXELS_PER_BIT)

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

        # context
        self.transform = Transform()  # make a new transform instance
        self.angle_xy = None  # see _find_blobs

        # samples probed when detecting edge widths (not lengths), see _find_best_neighbour,
        # these co-ords are rotated in steps of 45 degrees depending on the direction being
        # followed and the direction the edge is heading in,
        # the reference set here is for left-to-right and is considered as 0 degrees rotation,
        # the co-ordinates should be defined in 'best' first order (so the preferred direction
        # is followed when there are choices)

        # x,y pairs for neighbours when looking for inner/outer radius edges,
        # in this context the edges can be very 'wavy' this means they change rapidly vertically
        # so we probe quite a long way 'sideways' to ensure we continue to pick it up
        self.radius_kernel = [[0, 0], [1, 0], [2, 0],  # straight line
                              [0, -1], [0, 1],  # near diagonals
                              [1, -1], [1, 1],  # near neighbours
                              [2, -2], [2, 2],  # far diagonals
                              [0, -2], [0, 2],  # far neighbours
                              [0, -3], [0, 3]]  # distant neighbours
        self.radius_kernel_width = 3  # max offset of the pixels scanned by the radius_kernel

        # x,y pairs for neighbours when looking for ring and bit edges
        # in this context the edges are fairly straight so we do not tolerate a gap
        self.edge_kernel = [[0, 0], [1, 0],  # straight line
                            [0, -1], [0, 1]]  # diagonal
        self.edge_kernel_width = 1  # max offset of the pixels scanned by the edge_kernel

        # edge vector smoothing kernel, pairs of offset and scale factor (see _smooth_edge)
        self.edge_smoothing_kernel = [[-3, 0.5], [-2, 1], [-1, 1.5], [0, 2], [+1, 1.5], [+2, 1], [+3, 0.5]]

        # decoding context (used for logging and image saving)
        self.centre_x = 0
        self.centre_y = 0

        # logging context
        self._log_file = None
        self._log_folder = log
        self._log_prefix = '{}: Blob'.format(self.original.source)

        self.logging = self.show_log or (self._log_folder is not None)

    def __del__(self):
        """ close our log file """
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _find_blobs(self):
        """ find the target blobs in our image,
            this must be the first function called to process our image,
            creates a blob list each of which is a 'keypoint' tuple of:
                .pt[1] = x co-ord of the centre
                .pt[0] = y co-ord of centre
                .size = diameter of blob
            returns a list of unique blobs found
            also builds a polar-to-cartesian co-ordinate translation table of sufficient accuracy
            to cope with the largest blob detected
            """

        # prepare image
        blurred = self.transform.blur(self.original)  # de-noise
        self.image = self.transform.downsize(blurred, self.video_mode)  # re-size to given video mode

        # set filter parameters
        # ToDo: these need tuning to hell and back to optimise for our context
        threshold = (MIN_LUMINANCE, MAX_LUMINANCE, 8)  # min, max luminance, luminance step
        circularity = (0.75, None)  # min, max 'corners' in blob edge or None
        convexity = (0.5, None)  # min, max 'gaps' in blob edge or None
        inertia = (0.4, None)  # min, max 'squashed-ness' or None
        area = (30, 250000)  # min, max area in pixels, or None for no area filter
        gaps = (None, None)  # how close blobs have to be to be merged and min number
        colour = MAX_LUMINANCE  # we want bright blobs, use MIN_LUMINANCE for dark blobs

        # find the blobs
        blobs = self.transform.blobs(self.image, threshold, circularity, convexity, inertia, area, gaps, colour)

        # filter out blobs that are co-incident (opencv can get over enthusiastic!)
        dup_blobs = []
        dropped = True
        while dropped:
            dropped = False
            for blob1 in range(len(blobs)):
                x1 = int(round(blobs[blob1].pt[0]))
                y1 = int(round(blobs[blob1].pt[1]))
                size1 = blobs[blob1].size
                for blob2 in range(len(blobs)):
                    if blob2 == blob1:
                        # do not check self
                        continue
                    x2 = int(round(blobs[blob2].pt[0]))
                    y2 = int(round(blobs[blob2].pt[1]))
                    size2 = blobs[blob2].size
                    if math.fabs(x1 - x2) < Scan.MIN_BLOB_SEPARATION \
                            and math.fabs(y1 - y2) < Scan.MIN_BLOB_SEPARATION:
                        # these are too close, drop the worst one
                        if x1 == x2 and y1 == y2:
                            # got same centre, this happens sometimes, keep the smallest
                            if size1 < size2:
                                # drop #2
                                dup_blobs.append(blobs.pop(blob2))
                            else:
                                # drop #1
                                dup_blobs.append(blobs.pop(blob1))
                        else:
                            # they are not co-centric, keep the biggest
                            # we can tolerate blobs being slightly off-centre,
                            # it just gives the flattened image wavy inner/outer edges which we can deal with
                            if size1 < size2:
                                # drop #1
                                dup_blobs.append(blobs.pop(blob1))
                            else:
                                # drop #2
                                dup_blobs.append(blobs.pop(blob2))
                        dropped = True
                        break
                if dropped:
                    break

        # build the polar to cartesian translation table
        max_size = 3  # do one for at least 3 pixels
        for blob in blobs:
            if blob.size > max_size:
                max_size = blob.size

        max_radius = (max_size / 4) * Scan.NUM_RINGS * Scan.BLOB_RADIUS_STRETCH
        max_circumference = min(2 * math.pi * max_radius, 3600)  # good enough for 0.1 degree resolution
        angle = Angle(int(round(max_circumference)))
        self.angle_xy = angle.polarToCart

        if self.logging:
            self._log('blob-detect: found {} blobs'.format(len(blobs) + len(dup_blobs)), 0, 0)
            for blob in blobs:
                self._log('    centred at {}x{}y, size {:.2f}  (kept)'.
                          format(int(round(blob.pt[0])), int(round(blob.pt[1])), blob.size))
            for blob in dup_blobs:
                self._log('    centred at {}x{}y, size {:.2f}  (dropped as a duplicate)'.
                          format(int(round(blob.pt[0])), int(round(blob.pt[1])), blob.size))
        if self.save_images:
            grid = self.image
            if len(blobs) > 0:
                grid = self.transform.drawKeypoints(grid, blobs, (0, 255, 0))
            if len(dup_blobs) > 0:
                grid = self.transform.drawKeypoints(grid, dup_blobs, (0, 0, 255))
            self._unload(grid, 'blobs', 0, 0)

        return blobs

    def _project(self, centre_x, centre_y, blob_size):
        """ 'project' a potential target at centre_x/y from its circular shape to a rectangle
            of radius (y) by angle (x), blob_size is used as a guide to limit the radius projected,
            we assume the blob-size is (roughly) the diameter of the inner two white rings
            but err on the side of going too big, we use the opencv warpPolar function,
            returns the projected image and the diameter extracted from the original image
            """

        # calculate the maximum radius to go out to
        max_x, max_y = self.image.size()
        edge_top = centre_y
        edge_left = centre_x
        edge_bottom = max_y - centre_y
        edge_right = max_x - centre_x
        limit_radius = int(round(max(min((edge_top, edge_bottom, edge_left, edge_right)), 1)))
        blob_radius = int(round(((blob_size / 4) * Scan.NUM_RINGS) * Scan.BLOB_RADIUS_STRETCH))
        if blob_radius < limit_radius:
            # max possible size is less than the image edge, so use the blob size
            limit_radius = blob_radius

        image_height = limit_radius  # one pixel per radius
        image_width = int(round(2 * math.pi * limit_radius))  # one pixel per circumference

        # do the projection
        code = self.transform.warpPolar(self.image, centre_x, centre_y, limit_radius, image_width, image_height)

        # downsize to required angle resolution
        code = self.transform.downwidth(code, self.angle_steps)

        return code, limit_radius

    def _flatten(self, target, orig_radius):
        """ remove perspective distortions from the given 'projected' image,
            a circle when not viewed straight on appears as an ellipse, when that is projected into a rectangle
            the radius edges becomes 'wavy' (a sine wave), this function straightens those wavy edges, other
            distortions can arise if the target is curved (e.g. if it is wrapped around someones leg), in
            this case the the outer rings are even more 'wavy', for the purposes of this function the distortion
            is assumed to be proportional to the variance in the inner and outer edge positions, we know between
            the inner and outer edges there are 6 rings, we apply a stretching factor to each ring that is a
            function of the inner and outer edge variance,
            the returned image is just enough to contain the (reduced) image pixels of all the target rings
            (essentially number of bits wide by number of rings high),
            orig_radius is the target radius in the original image, its used to calculate the target scale,
            returns the flattened image, its scale and reject reason
            """

        # get the edge limits we need
        max_x, projected_y = target.size()

        # find our marker edges in the radius
        probe_centre = 0  # ToDo: for reasons unknown this screws the flattening-->int(round(max_x / 2))
        ring_inner_edge, ring_outer_edge, reason = self._find_extent(target, probe_centre)
        if reason is not None:
            return None, None, reason

        # ref point for inner is image edge in height, i.e. 0
        max_inner_edge = 0
        # ref point for outer is the corresponding inner
        max_outer_edge = 0
        max_outer_inner_edge = 0  # inner edge at max outer
        for x in range(max_x):
            inner_edge = ring_inner_edge[x]
            outer_edge = ring_outer_edge[x]
            if inner_edge is None:
                return None, None, 'inner edge has a gap at {}'.format(x)
            if outer_edge is None:
                return None, None, 'outer edge has a gap at {}'.format(x)
            if inner_edge > max_inner_edge:
                max_inner_edge = inner_edge
            distance = outer_edge - inner_edge
            if distance > max_outer_inner_edge:
                max_outer_edge = outer_edge
                max_outer_inner_edge = inner_edge

        if self.logging:
            self._log('max inner edge {}, outer edge at max distance from inner {}, inner edge at max outer edge {}'.
                      format(max_inner_edge, max_outer_edge, max_outer_inner_edge))

        stretched_size = self._get_estimated_ring_sizes(0, max_inner_edge,
                                                        max_outer_inner_edge, max_outer_edge,
                                                        stretch=True)

        if self.logging:
            self._log('stretched_size with ({}, {}) is {}'.
                      format(max_outer_inner_edge, max_outer_edge, stretched_size))

        # create a new image to flatten into
        flat_y = int(round(sum(stretched_size)))
        code = self.original.instance().new(max_x, flat_y, MID_LUMINANCE)

        # build flat image
        # NB: ...edge[0] in the vector is really ...edge[probe_centre] in the image
        truncated_y = 0
        for x in range(max_x):

            actual_size = self._get_estimated_ring_sizes(0, ring_inner_edge[x],
                                                         ring_inner_edge[x], ring_outer_edge[x])
            if self.logging:
                self._log('actual_size at {}x with ({:.2f}, {:.2f}) is {}'.
                          format(x, ring_inner_edge[x], ring_outer_edge[x], vstr(actual_size)))

            in_y = 0
            out_y = 0
            in_ring_start = 0
            out_ring_start = 0
            for ring in range(Scan.NUM_RINGS):
                # change each ring from its size now to its stretched size
                in_ring_end = in_ring_start + actual_size[ring]  # these may be fractional
                in_pixels = [None for _ in range(max(int(round(in_ring_end - in_y)), 1))]
                for dy in range(len(in_pixels)):
                    in_pixels[dy] = target.getpixel(x, in_y)
                    in_y += 1
                out_ring_end = out_ring_start + stretched_size[ring]  # these may be fractional
                out_pixels = self._stretch(in_pixels, max(int(round(out_ring_end - out_y)), 1))
                # the out ring sizes have been arranged to be whole pixel widths, so no fancy footwork here
                for dy in range(len(out_pixels)):
                    pixel = out_pixels[dy]
                    code.putpixel(x, out_y, pixel)
                    out_y += 1
                if self.logging:
                    self._log('    ring {}, in_y {}, out_y {}: in_pixels {:.2f}..{:.2f}, out_pixels {:.2f}..{:.2f}'.
                              format(ring, in_y - len(in_pixels), out_y - len(out_pixels),
                                     in_ring_start, in_ring_end, out_ring_start, out_ring_end))
                in_ring_start += actual_size[ring]
                out_ring_start += stretched_size[ring]
            if in_y > truncated_y:
                truncated_y = in_y
            continue  # statement here as a debug hook

        # upsize such that each ring width is sufficient to always find a middle
        new_y = Scan.NUM_RINGS * Scan.MIN_PIXELS_PER_RING
        code = self.transform.upheight(code, new_y)

        target_x, target_y = code.size()

        # calculate the flattened image scale relative to the original
        # orig_radius is what was extracted from the original image, it then
        # became projected_y, then it became flat_y and finally target_y
        # orig to projected may have been upscaled
        # projected to flat is truncating, truncated_y is the truncation boundary
        # flat to target may have been upscaled
        scale1 = orig_radius / projected_y
        scale2 = truncated_y / flat_y
        scale3 = flat_y / target_y
        scale = scale1 * scale2 * scale3

        if self.save_images:
            # draw blue ticks and start and end of all nominal ring edges
            nominal_width = target_y / Scan.NUM_RINGS
            grid = code
            for ring in range(Scan.NUM_RINGS):
                grid = self._draw_lines(grid, [[0, nominal_width * ring, 1, nominal_width * ring]], (255, 0, 0))
                grid = self._draw_lines(grid,
                                        [[target_x - 2, nominal_width * ring, target_x - 1, nominal_width * ring]],
                                        (255, 0, 0))
            self._unload(grid, 'flat')

        # return flattened image
        return code, scale, None

    def _find_extent(self, target, probe_centre):
        """ find the inner and outer edges of the given target,
            probe_centre is the x co-ordinate to scan at,
            the inner edge is the first white to black transition that goes all the way around,
            the outer edge is the first black to white transition that goes all the way around,
            returns two vectors, y co-ord for every angle of the edges, or a reason if one or both not found
            """

        max_x, _ = target.size()

        # look for the outer edge
        b2w_edges, threshold = self._get_transitions(target, 0, 1, False)
        ring_outer_edge, best_outer_partial = self._find_radius_edge(b2w_edges, probe_centre,
                                                                     Scan.BOTTOM_UP, threshold, 'radius-outer')
        if ring_outer_edge is None:
            # don't bother looking for the inner if there is no outer
            ring_inner_edge = None
            w2b_edges = None
        else:
            # look for the inner edge
            w2b_edges, threshold = self._get_transitions(target, 0, 1, True)
            ring_inner_edge, best_inner_partial = self._find_radius_edge(w2b_edges, probe_centre,
                                                                         Scan.TOP_DOWN, threshold, 'radius-inner')
        if ring_outer_edge is None:
            reason = 'no outer edge'
        elif ring_inner_edge is None:
            reason = 'no inner edge'
        elif ring_outer_edge <= ring_inner_edge:
            reason = 'outer edge before inner edge'
        else:
            reason = None

        if self.save_images:
            if w2b_edges is not None:
                plot = self._draw_below(w2b_edges, threshold, (0, 0, 255))
                if ring_inner_edge is not None:
                    points = [[probe_centre, ring_inner_edge]]
                    plot = self._draw_plots(plot, points, None, (0, 255, 0))
                if best_inner_partial is not None:
                    points = [[probe_centre, best_inner_partial]]
                    plot = self._draw_plots(plot, points, None, (0, 255, 0))
                self._unload(plot, 'inner')

            if b2w_edges is not None:
                plot = self._draw_below(b2w_edges, threshold, (0, 0, 255))
                if ring_outer_edge is not None:
                    points = [[probe_centre, ring_outer_edge]]
                    plot = self._draw_plots(plot, points, None, (255, 0, 0))
                if best_outer_partial is not None:
                    points = [[probe_centre, best_outer_partial]]
                    plot = self._draw_plots(plot, points, None, (255, 0, 0))
                self._unload(plot, 'outer')

            plot = target
            if ring_inner_edge is not None:
                plot = self._draw_plots(plot, [[probe_centre, ring_inner_edge]], None, (0, 255, 0))
            if ring_outer_edge is not None:
                plot = self._draw_plots(plot, [[probe_centre, ring_outer_edge]], None, (0, 0, 255))
            self._unload(plot, 'wavy')

        return ring_inner_edge, ring_outer_edge, reason

    def _find_radius_edge(self, edges, probe_centre, direction, threshold, prefix):
        """ look for a continuous edge in the given edges image either top-down (inner) or bottom-up (outer),
            probe_centre is the x co-ord to scan up/down at,
            prefix is purely for debug log messages,
            to qualify as an 'edge' it must be continuous across all angles (i.e. no dis-connected jumps),
            the returned edge is smoothed,
            returns a tuple of either:
                smoothed edge, None if succeeded
                None, best partial edge if failed
            """

        context = self.Context(edges, direction)  # direction assumed to be top-down or bottom-up
        context.kernel = self.radius_kernel
        context.prefix = prefix  # overwrite standard prefix with the one we are given

        max_x = context.max_x
        max_y = context.max_y

        if self.save_images:
            target = self.transform.copy(edges)  # following edges destroys pixels so make a copy here
        else:
            target = edges  # when not showing images, pixel destruction is irrelevant

        best_edge = None

        stepper = Scan.Stepper(direction, max_x, max_y, step=self.radius_kernel_width)
        stepper.reset(x=probe_centre)
        while True:
            xy = stepper.next()
            if xy is None:
                break
            edge = self._follow_edge(target, xy[0], xy[1], context, Scan.LEFT_TO_RIGHT, threshold, clear=False)
            if len(edge) == 0:
                # nothing here, move on
                # NB: if nothing going forwards there will also be nothing going backwards
                continue
            if len(edge) == max_x:
                # found an edge that goes all the way, so that's it
                if self.logging:
                    self._log('{}: full radius edge length: {} from {},{:.2f} (as 0 backwards + {} forwards)'.
                              format(prefix, len(edge), probe_centre, edge[0].midpoint, len(edge)))
                edge_points = [edge[x].midpoint for x in range(max_x)]
                return self._smooth_edge(edge_points, True), (None, None)

            if best_edge is None:
                best_edge = edge
            elif len(edge) > len(best_edge):
                best_edge = edge

            # forward edge too short, see if a backward edge will do it
            # ToDo: is this necessary anymore? its a crude attempt to cope with forking edges,
            #       better to deal with that properly, how?
            #       be sloppy - we know the first significant edge is the one we want (inner or outer)
            #       so find all segments that are reasonably 'close' to one another and just join them up
            #       by interpolating across the ends?
            edge_extension = self._follow_edge(target, xy[0], xy[1], context, Scan.RIGHT_TO_LEFT, threshold,
                                               clear=False)
            if len(edge_extension) == 0:
                if self.logging:
                    self._log('{}: partial radius edge length: {} from 0,{:.2f} (as 0 backwards + {} forwards)'.
                              format(prefix, len(edge), edge[0].midpoint, len(edge)))
                continue
            if len(edge_extension) == max_x:
                # the extension went all the way around, use that
                if self.logging:
                    self._log('{}: full radius edge length: {} from 0,{:.2f} (as {} backwards + 0 forwards)'.
                              format(prefix, len(edge_extension), edge_extension[0].midpoint, len(edge_extension)))
                edge_points = [edge_extension[x].midpoint for x in range(max_x)]
                return self._smooth_edge(edge_points, True)

            # backward edge too short as well, give up
            if self.logging:
                self._log('{}: partial radius edge length: {} from {},{:.2f} (as {} backwards + {} forwards)'.
                          format(prefix, len(edge) + len(edge_extension), probe_centre, edge[0].midpoint,
                                 len(edge_extension), len(edge)))
            continue

        # we did not find one
        if best_edge is not None:
            best_edge_points = [best_edge[x].midpoint for x in range(len(best_edge))]
            return None, best_edge_points
        else:
            return None, None

    def _midpoint(self, start_at, end_at):
        """ given two *inclusive* co-ordinates (either two x's or two y's) return their mid-point,
            ie. that co-ordinate that is central between the two (can be fractional),
            the co-ordinates can be given either way round (start < end or end < start),
            if either is None it just returns the other
            """
        if end_at is None:
            return start_at
        elif start_at is None:
            return end_at
        elif end_at > start_at:
            return start_at + ((end_at - start_at) / 2)
        else:
            return start_at - ((start_at - end_at) / 2)

    def _smooth_edge(self, in_edge, wraps=False):
        """ smooth the given edge vector by doing a mean across N pixels,
            if wraps is True the edge wraps end-to-end,
            return the smoothed vector
            """
        # ToDo: is smoothing radius edges necessary?
        extent = len(in_edge)
        out_edge = [None for _ in range(extent)]
        for x in range(extent):
            v = 0  # value accumulator
            d = 0  # divisor accumulator
            for dx, f in self.edge_smoothing_kernel:
                if wraps:
                    sample = in_edge[(x + dx) % extent]
                elif x + dx >= extent:
                    # gone off end of edge
                    continue
                else:
                    sample = in_edge[x + dx]
                if sample is None:
                    # what does this mean?
                    continue
                v += (sample * f)
                d += f
            if d > 0:
                out_edge[x] = v / d
        return out_edge

    def _get_within_threshold(self, target, x, y, context, threshold, reversed=False):
        """ get a list of pixels from x,y in the given scan direction that are over the given threshold,
            x is wrapped at image edge, y is not, if given an excessive y None is returned,
            context direction defines the scanning direction (x or y),
            reversed is True if backwards in the scanning direction or False if not,
            threshold is used to find the extremes of the edge,
            returns a list of pixels or an empty list if given x,y is not over the threshold,
            NB: This function is called as part of the find edges system and is expected to be called
                exactly once for any particular edge. The image given is expected to be an 'edges' image.
                Once a pixel has been read it is cleared in the image to the minimum value. This is to
                prevent the same edge being detected more than once.
            """

        max_x = context.max_x
        max_y = context.max_y
        allow_wrap = context.allow_scan_wrap
        scan_limit = context.max_scan_coord + 1
        scan_coord = context.scan_coord
        if reversed:
            scan_inc = 0 - context.scan_multiplier
        else:
            scan_inc = context.scan_multiplier

        x = int(round(x % max_x))
        y = int(round(y))
        if y >= max_y:
            return []

        pixels = []
        xy = [x, y]
        while True:
            pixel = target.getpixel(xy[0], xy[1])  # NB: returns None when reach image edge
            if pixel is None or pixel < threshold:
                break
            pixels.append(pixel)  # add it to our list
            xy[scan_coord] += scan_inc  # move on
            if xy[scan_coord] >= scan_limit:
                if allow_wrap:
                    xy[scan_coord] %= scan_limit  # carry on around
                else:
                    break  # that's it

        return pixels

    def _get_midpoint(self, pixels):
        """ given a list of pixels find their midpoint,
            the midpoint is the median of the 'bright' pixels,
            returns midpoint and the bright limits
            """

        brightest = 0
        brightest_first = None
        brightest_last = None
        for x in range(len(pixels)):
            pixel = pixels[x]
            if pixel > brightest:
                # the bright threshold changes depending on what we discover
                # if pixel is above the brightest by enough the brightest rises
                if pixel * Scan.BRIGHTER_THRESHOLD > brightest:
                    # its changed by enough to justify pushing the threshold up
                    brightest = pixel * Scan.BRIGHTER_THRESHOLD
                    brightest_first = x
                    brightest_last = brightest_first
                else:
                    # leave brightest where it is, but note the brightest is spreading
                    if brightest_first is None:
                        # this means we've found the first bright thing
                        # but its not bright enough to change the threshold
                        brightest_first = x
                    brightest_last = x

        if brightest_first is None:
            # this means the caller gave us an empty list
            midpoint = None
        else:
            midpoint = self._midpoint(brightest_first, brightest_last)

        return midpoint, brightest_first, brightest_last

    def _set_image_pixels(self, target, from_xy, to_xy, luminance=MIN_LUMINANCE):
        """ set pixels in the given target to the luminance given in the rectangle defined by from_xy to to_xy,
            from/to_xy is an x,y tuple defining the top-left corner and bottom-right corner to be cleared,
            """
        self.transform.fill(target, from_xy, to_xy, luminance)

    def _find_width_midpoint(self, target, x, y, context, threshold, clear):
        """ from x,y in the target image find the width midpoint x or y that is over the given threshold,
            if clear is True the pixels over the threshold are cleared (so they cannot be considered again),
            the pixel at x,y must be over else None is returned,
            the context is for the scanning direction,
            this function is used when scanning for edges and an edge start has been found,
            it then finds the width extent of that edge (not its length),
            returns a fully populated EdgePoint instance or None if no midpoint
            """

        scan_coord = context.scan_coord
        cross_coord = context.cross_coord
        scan_multiplier = context.scan_multiplier

        xy = [int(round(x)), int(round(y))]  # must int+round to ensure we get a correct midpoint

        pixels_up = self._get_within_threshold(target, xy[0], xy[1], context, threshold, reversed=True)
        if len(pixels_up) == 0:
            return None

        centre = xy[scan_coord]  # note for limits calc later

        xy[scan_coord] += (scan_multiplier * 1)
        pixels_down = self._get_within_threshold(target, xy[0], xy[1], context, threshold, reversed=False)

        if scan_multiplier < 0:
            # up is natural, down is reversed
            #  lowest co-ord-->highest co-ord
            #  [dddddd] x [uuuuuu] x is the reference
            first = centre - len(pixels_down)
            last = centre + (len(pixels_up) - 1)
            # join the two sequences in the correct order
            pixels_down.reverse()
            pixels = pixels_down + pixels_up
        else:
            # down is natural, up is reversed
            #  lowest co-ord-->highest co-ord
            #  [uuuuuu] x [dddddd] x is the reference
            first = centre - (len(pixels_up) - 1)
            last = centre + len(pixels_down)
            # join the two sequences in the correct order
            pixels_up.reverse()
            pixels = pixels_up + pixels_down

        # find their midpoint
        midpoint, bright_first, bright_last = self._get_midpoint(pixels)
        if midpoint is None:
            return None

        if clear:
            # we're done with these pixels now, so zap them from our edges image (this stops us finding duplicates)
            from_xy = [xy[0], xy[1]]
            to_xy = [xy[0], xy[1]]
            from_xy[scan_coord] = first
            to_xy[scan_coord] = last
            self._set_image_pixels(target, from_xy, to_xy, MIN_LUMINANCE)

        bright_first += first
        bright_last += first

        return self.EdgePoint(first + midpoint,
                              first, last,
                              bright_first, bright_last, xy[cross_coord])

    def _find_best_neighbour(self, target, x, y, context, direction, threshold, clear):
        """ find best neighbour in target from x,y in given direction using the given threshold,
            direction affects the kernel orientation,
            if clear is True the pixels involed are cleared (so the neighbour cannot be found again),
            context.kernel is the matrix to use to determine that a neighbour is 'connected',
            returns a fully populated EdgePoint instance,
            returns None if there is not one (ie. no neighbours are connected),
            it finds the best ongoing x or y co-ordinate to follow,
            see _find_width_midpoint for tht definition of 'best',
            the x,y co-ords given here may be fractional, this can lead to position ambiguities
            when they are converted to int(), to make sure we do not miss any matrix addresses
            we execute the kernel from floor(x) to ceiling(x) and floor(y) to ceiling(y), in the
            worst case this could be four times
            """

        kernel = context.kernel

        max_x = context.max_x
        max_y = context.max_y

        x_min = int(math.floor(x))
        x_max = int(math.ceil(x))
        y_min = int(math.floor(y))
        y_max = int(math.ceil(y))

        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                for dx, dy in self.Kernel(kernel, direction):
                    dy += y
                    if dy < 0 or dy >= max_y:
                        continue  # y never wraps
                    dx = (dx + x) % max_x  # x always wraps
                    best = self._find_width_midpoint(target, dx, dy, context, threshold, clear)
                    if best is not None:
                        return best

        return None

    def _is_direction(self, direction, wanted):
        """ given a direction return True if it contains wanted
            """
        return (direction & wanted) == wanted

    def _drift_exceeded(self, edge, edge_point, drift_limit, max_coord, prefix):
        """ check if the given edge has drifted too far sideways,
            edge is the edge detected so far,
            edge_point is the new edge point about to be added,
            start is the midpoint of the first edge point, now is the midpoint of the given edge point,
            if the distance between these is beyond the given limit then return True, otherwise return False
            """

        if drift_limit is None:
            # there is no limit
            return False

        if len(edge) < Scan.MIN_DRIFT_EDGE_LENGTH:
            # too soon to tell
            return False

        start_midpoint = edge[0].midpoint
        now_midpoint = edge_point.midpoint
        drift = math.fabs(start_midpoint - now_midpoint)
        if drift > (max_coord / 2):  # we're assuming the drift is never more than half the image
            # the midpoint has wrapped, so true drift is in two parts from either image edge
            drift = (max_coord + 1) - max(start_midpoint, now_midpoint)
            drift += min(start_midpoint, now_midpoint)
        if drift > 1 and (drift / len(edge)) > drift_limit:
            # we've drifted too far sideways, so chop it here
            if self.logging:
                self._log('{}: edge drifted too far ({:.2f}..{:.2f} across {}={:.2f}), limit is {:.2f}'.
                          format(prefix, start_midpoint, now_midpoint, len(edge), drift / len(edge), drift_limit))
            return True

        return False

    def _follow_edge(self, target, x, y, context, direction, threshold, continuation=None, clear=True):
        """ follow the edge at x,y if there is one there, until come to its end,
            if clear is True the pixels in the edge are cleared (this prevents us finding the same edge again),
            if continuation is given this is a continuation from a previous call to here in the opposite
            direction and the continuation param is the result of the previous call,
            in this case the function behaves as if the inner loop is being resumed at x,y but in the
            other direction, context is for the scanning direction (which is different to the direction here),
            direction param is for the following direction (i.e. along the edge length discovered),
            this direction is left-to-right or right-to-left for following edges in x (rings),
            or direction is top_down or bottom_up for following edges in y (bits),
            NB: the 'following' direction is different to the scanning direction in the context,
            threshold is the threshold to apply to determine if a pixel is a candidate for an edge,
            its followed for at most for the image width/height,
            x co-ordinates wrap, y does not,
            returns a vector of EdgePoints, or an empty vector if none,
            the pixel at x,y must be over the threshold else an empty vector is returned
            """

        max_x = context.max_x
        max_y = context.max_y
        scan_direction = context.scan_direction
        best_coord = context.scan_coord  # NB: 90 degree rotated from scan direction

        stepper = Scan.Stepper(direction, max_x, max_y)
        stepper.reset(x=x, y=y)

        if continuation is not None and len(continuation) > 0:
            # just set best to the first entry found last time
            # and reduce the number of steps to whats left
            best = continuation[0]
            stepper.reset(steps=stepper.cycles() - len(continuation))
            last_edge = best
            edge = []
        else:
            # not a continuation, find a starting point
            xy = stepper.next()
            # NB: the direction here is for the kernel and is the scan direction not the following direction
            best = self._find_best_neighbour(target, xy[0], xy[1], context, scan_direction, threshold, clear)
            if best is None:
                # there is no edge (reachable) from here
                return []
            last_edge = best
            edge = [best]

        while True:
            xy = stepper.next()
            if xy is None:
                break
            xy[best_coord] = best.midpoint
            best = self._find_best_neighbour(target, xy[0], xy[1], context, direction, threshold, clear)
            if best is None:
                # did not find one at x,y
                # move x,y within the bounds of the previous edge point first/last and try those
                first = last_edge.first
                last = last_edge.last
                midpoint = last_edge.midpoint
                # scan either side of the midpoint until gone past both first and last
                delta = 0
                more = (first != last)  # there are only more if the width is > 1
                while more:
                    more = False
                    delta += 1
                    if midpoint + delta <= last:
                        xy[best_coord] = midpoint + delta
                        best = self._find_best_neighbour(target, xy[0], xy[1], context, direction, threshold, clear)
                        if best is not None:
                            # found one, go with it
                            break
                        more = True
                    if midpoint - delta >= first:
                        xy[best_coord] = midpoint - delta
                        best = self._find_best_neighbour(target, xy[0], xy[1], context, direction, threshold, clear)
                        if best is not None:
                            # found one, go with it
                            break
                        more = True
                if best is None:
                    # no more qualifying neighbors
                    break
            # it qualifies
            edge.append(best)
            last_edge = best

        return edge

    def _get_edge_threshold(self, target):
        """ find the edge threshold to apply in the given target,
            target is assumed to be an edge detected image,
            this function finds the mean luminance (ignoring near black pixels)
            and returns that (turned into an int in the range MIN_LUMINANCE..MAX_LUMINANCE)
            """

        # turn image into a flat vector
        buffer = target.get().reshape(-1)

        # chuck out near black pixels
        filtered = buffer[np.where(buffer > Scan.MIN_EDGE_THRESHOLD)]
        if len(filtered) == 0:
            # nothing qualifies
            return 0.0

        # give caller the mean of what is left
        return np.mean(filtered)

    def _get_transitions(self, target, x_order, y_order, inverted):
        """ get black-to-white or white-to-black edges from target
            returns an image of the edges and a threshold to detect edges
            """

        edges = self.transform.edges(target, x_order, y_order, 3, inverted)
        image_mean = self._get_edge_threshold(edges)
        threshold = max(min(int(round(image_mean * Scan.EDGE_THRESHOLD)), MAX_LUMINANCE), 1)

        return edges, threshold

    def _stretch(self, pixels, size):
        """ given a vector of pixels stretch them such that the resulting pixels is size long,
            consecutive pixel values are interpolated as necessary when stretched into the new size,
            this is a helper function for _flatten, in that context we are stretching pixels in the
            'y' direction, the names used here reflect that but this logic is totally generic
            """

        if len(pixels) == 0:
            print('eh?')
        dest = [None for _ in range(size)]
        # do the initial stretch
        y_delta = size / len(pixels)
        y = 0
        for pixel in pixels:
            dest[int(y)] = pixel
            y += y_delta
        # we've now got a vector with gaps that need interpolating across
        start_gap = None
        end_gap = None
        for y in range(size):
            if dest[y] is None:
                if start_gap is None:
                    # start of a gap
                    start_gap = y
                end_gap = y
            elif start_gap is not None:
                # we're coming out of a gap, interpolate across it
                end_pixel = dest[y]
                if start_gap > 0:
                    start_pixel = dest[start_gap - 1]
                else:
                    start_pixel = end_pixel
                span = end_gap - start_gap + 1
                pixel_delta = (int(end_pixel) - int(start_pixel)) / (span + 1)  # NB: use int() to stop numpy minus
                gap_pixel = start_pixel
                for dy in range(span):
                    gap_pixel += pixel_delta
                    dest[start_gap + dy] = np.uint8(max(min(round(gap_pixel), MAX_LUMINANCE), MIN_LUMINANCE))
                start_gap = None
                end_gap = None
        if start_gap is not None and start_gap > 0:
            # do the final gap by just propagating the last pixel
            span = size - start_gap
            gap_pixel = dest[start_gap - 1]
            for dy in range(span):
                dest[start_gap + dy] = gap_pixel
        # that's it
        return dest

    def _get_estimated_ring_sizes(self, inner_start, inner_end, outer_start, outer_end, stretch=False):
        """ given an inner and outer ring position, calculate the estimated width of each ring,
            returns a vector of ring sizes,
            when stretch is True all rings are set to the same size
            """

        # The distance between the inner white/black edge and the outer black/white edge spans all but the two
        # inner white rings and one outer white rings (i.e. -3)
        ring_span = Scan.NUM_RINGS - 3  # number of rings between inner and outer edge
        outer_ring_size = (outer_end - outer_start) / ring_span  # average outer ring size (i.e. beyond inner edge)
        inner_ring_size = (inner_end - inner_start) / 2  # average inner ring size (i.e. inside inner edge)

        ring_sizes = [None for _ in range(Scan.NUM_RINGS)]

        if stretch:
            # we want all rings the same size here and as big as the max
            outer_ring_size = max(outer_ring_size, inner_ring_size)
            inner_ring_size = outer_ring_size

        ring_sizes[self.INNER_POINT] = inner_ring_size
        ring_sizes[self.INNER_WHITE] = inner_ring_size
        # inner edge is here
        ring_sizes[self.INNER_BLACK] = outer_ring_size
        ring_sizes[self.DATA_RING_1] = outer_ring_size
        ring_sizes[self.DATA_RING_2] = outer_ring_size
        ring_sizes[self.DATA_RING_3] = outer_ring_size
        ring_sizes[self.TIMING_RING] = outer_ring_size
        ring_sizes[self.OUTER_BLACK] = outer_ring_size
        # outer edge is here
        ring_sizes[self.OUTER_WHITE] = outer_ring_size

        return ring_sizes

    def _get_start(self, edge, max_length):
        """ work out the edge start from a consideration of the .where co-ordinates in the given edge,
            that edge must be a vector of EdgePoints,
            max_length is the length limit and used to detect wrapping,
            returns the start co-ordinate or -1 if the edge is empty
            """

        if len(edge) == 0:
            edge_start = -1
        elif len(edge) < 2:
            # only a single sample, so that must be the start
            edge_start = edge[0].where
        else:
            if edge[0].where == 0 and edge[1].where == (max_length - 1):
                # this means we wrapped while scanning backwards
                edge_start = edge[-1].where
            elif edge[1].where < edge[0].where:
                # co-ords decreasing, this means we're scanning backwards
                edge_start = edge[-1].where
            elif edge[1].where > edge[0].where:
                # co-ords increasing, this means we're scanning forwards
                edge_start = edge[0].where
            else:
                # two consecutive samples cannot have the same 'where'
                raise Exception('edge[0] and [1] have same .where ({})!'.format(edge[0].where))

        return edge_start

    def _measure_edge(self, edge_where, down_edge, up_edge, max_length, edge_type, history=None):
        """ this is a helper function for _get_edges,
            it analyses the given edge points and returns one or more instances of Edge that characterises them,
            the down_edge and up_edge must be the results of scans going 'forwards' then 'backwards'
            around a pivot point and both consist of a vector of EdgePoints,
            max_length is used to detect wrapping,
            edge_where and edge_type are just deposited into the Edge instance created,
            history is the vector of previous Edge instances and is used for smoothing the midpoints,
            if history is not given no smoothing is performed,
            down_edge or up_edge may be empty,
            returns a list of Edges which may be empty
            """

        # ToDo: re-arrange this function to join the up and down pixels and split edges that drift too far
        #       the function returns a list of edges not a single one
        # if self._drift_exceeded(edge, best, drift_limit, max_cross_coord, prefix):
        #    # we've drifted too far sideways
        #    break

        # NB: the co-ordinates in an EdgePoint may be out of range if the edge width wrapped the image edge,
        #     however that does not affect the logic here as they are only used to calculate widths and they
        #     are valid wrapping or not.

        # set edge start
        down_start = self._get_start(down_edge, max_length)
        up_start = self._get_start(up_edge, max_length)
        if down_start < 0 and up_start < 0:
            # both edges are empty
            return []
        elif down_start < 0:
            # down empty, up not
            edge_start = up_start
        elif up_start < 0:
            # up empty, down not
            edge_start = down_start
        elif up_start + len(up_edge) > max_length:
            # up edge wraps, they can't both wrap, so up must be the start
            edge_start = up_start
        elif down_start + len(down_edge) > max_length:
            # down edge wraps, they can't both wrap, so down must be the start
            edge_start = down_start
        else:
            # got both and neither wrap, so start is the min of down and up
            edge_start = min(down_start, up_start)

        max_edge = 0
        min_edge = 31 * 1024  # arbitrary large number bigger than any legit co-ord
        max_bright_edge = max_edge
        min_bright_edge = min_edge
        edge_length = 0
        midpoints = 0

        def update_edge(edge_point):
            nonlocal midpoints, min_bright_edge, max_bright_edge, min_edge, max_edge, edge_length
            midpoints += edge_point.midpoint
            if edge_point.bright_first < min_bright_edge:
                min_bright_edge = edge_point.bright_first
            if edge_point.bright_last > max_bright_edge:
                max_bright_edge = edge_point.bright_last
            if edge_point.first < min_edge:
                min_edge = edge_point.first
            if edge_point.last > max_edge:
                max_edge = edge_point.last
            edge_length += 1

        # find edge limits
        for edge_point in up_edge:
            update_edge(edge_point)
        for edge_point in down_edge:
            update_edge(edge_point)

        if edge_length == max_length:
            # consider max length edges to always start at 0
            edge_start = 0

        # set midpoint
        midpoint = midpoints / edge_length

        # set widths
        edge_bright_span = max_bright_edge - min_bright_edge + 1
        edge_span = max_edge - min_edge + 1

        # ToDo: apply smoothing to the midpoint within the edge_span
        edge = self.Edge(midpoint, edge_length, edge_bright_span, edge_start, edge_span,
                         where=edge_where, edge=edge_type)

        return [edge]

    def _get_edges(self, target, context, centres, threshold, edge_type):
        """ return a vector of the edges in the given target within the centres given,
            direction in context is top-down or bottom-up if looking for ring edges,
            direction in context is left-to-right or right-to-left if looking for bit edges,
            centres provides a list of 'cross' co-ordinates that should be probed,
            edge_type is just deposited into the Edge instances created as the edge property,
            the target must consist of either white-to-black edges or black-to-white edges or both,
            each edge in the returned vector consists of an Edge instance,
            the result vector is sorted by position then start then length,
            returns the result and an image with detections drawn on it (None if not in debug mode),
            NB: ring edges wrap and bit edges do not
            """

        context.kernel = self.edge_kernel  # set the kernel to use for detecting connected-ness

        prefix = context.prefix
        max_length = context.max_cross_coord + 1
        min_length = context.min_length
        drift_limit = context.drift_limit
        down = context.forward_direction
        up = context.backward_direction
        scan_coord = context.scan_coord
        probe_coord = context.cross_coord
        min_scan_coord = 0
        max_scan_coord = context.max_scan_coord
        scan_direction = context.scan_multiplier
        if scan_direction > 0:
            # forwards, 0..max
            start_scan_coord = 0
        else:
            # backwards, max..0
            start_scan_coord = max_scan_coord

        if self.logging:
            self._log('{}: probe centres are: {}'.format(prefix, centres))
        if self.save_images:
            # highlight pixels that are below our threshold
            # NB: this makes a copy of the image, which is required as the following procedure destroys the target
            grid = self._draw_below(target, threshold, (0, 0, 255))
            # draw the centres we're probing in dark green
            if len(centres) > (max_length / 4):
                # don't bother, they're too dense
                pass
            else:
                centre_plots = [None, None]
                centre_plots[scan_coord] = centres
                grid = self._draw_grid(grid, centre_plots[0], centre_plots[1], (0, 128, 0))
        else:
            grid = None

        edges = []
        xy = [0, 0]
        for probe in centres:
            scan_inc = 0  # no increment for the first sample
            xy[scan_coord] = start_scan_coord
            skip_edge = None
            while True:
                if skip_edge is not None:
                    # this means we found an edge and now want to carry on past it
                    scan_inc = math.ceil(skip_edge.width / 2)
                    skip_edge = None
                xy[scan_coord] += int((scan_direction * scan_inc))
                scan_inc = self.edge_kernel_width  # reset to the default increment
                if xy[scan_coord] >= max_scan_coord or xy[scan_coord] < min_scan_coord:
                    break
                xy[probe_coord] = probe
                px = xy[0]
                py = xy[1]
                down_edge = self._follow_edge(target, px, py, context, down, threshold)
                if len(down_edge) == 0:
                    # found nothing going down, this means we'll also find nothing going up
                    # 'cos they overlap in the initial pixel address
                    continue
                elif len(down_edge) < max_length:
                    # try the other way when we don't find a complete edge
                    up_edge = self._follow_edge(target, px, py, context, up, threshold, continuation=down_edge)
                else:
                    # gone right round, so don't bother going the other way
                    up_edge = []
                edge_pieces = self._measure_edge(xy[scan_coord], down_edge, up_edge, max_length, edge_type, edges)
                for edge in edge_pieces:
                    if edge.length < min_length:
                        # too small ignore it
                        if self.logging:
                            self._log('{}: ignoring short edge of {} at {},{} (up-len:{}, down-len:{}), limit is {}'.
                                      format(prefix, edge, px, py, len(up_edge), len(down_edge), min_length))
                        continue
                    if len(edges) > 0 and edge == edges[-1]:
                        # got a duplicates (this should not happen!)
                        if self.logging:
                            self._log('{}: ignoring duplicate edge of {} at {},{} (up-len:{}, down-len:{})'.
                                      format(prefix, edge, px, py, len(up_edge), len(down_edge)))
                        continue

                    # the edge qualifies, add it to our list
                    edges.append(edge)
                    if self.logging:
                        self._log('{}: adding edge {} at {},{} (up-len:{}, down-len:{})'.
                                  format(prefix, edges[-1], px, py, len(up_edge), len(down_edge)))
                    if self.save_images:
                        # draw the edge we detected in green and its ends in yellow
                        # NB: we're doing this on our image copy as follow edges destroyed pixels in the scanned one
                        down_plots = []
                        down_start = self._get_start(down_edge, max_length)
                        for plot in range(len(down_edge)):
                            down_plots.append(down_edge[plot].midpoint)
                        up_plots = []
                        up_start = self._get_start(up_edge, max_length)
                        for plot in range(len(up_edge)):
                            up_plots.append(up_edge[plot].midpoint)
                        if scan_direction < 0:
                            # we scanned backwards, so reverse down plot
                            down_plots.reverse()
                        else:
                            # reverse the up plot
                            up_plots.reverse()
                        xy_plots = [None, None]
                        xy_plots[probe_coord] = [[down_start, down_plots],
                                                 [up_start, up_plots]]
                        grid = self._draw_plots(grid, xy_plots[0], xy_plots[1], (0, 255, 0))
                        if len(up_edge) > 0:
                            up_val = up_plots[0]
                        else:
                            up_start = down_start
                            up_val = down_plots[0]
                        if len(down_edge) > 0:
                            down_end = down_start + len(down_edge) - 1
                            down_val = down_plots[-1]
                        else:
                            down_end = up_start + len(up_edge) - 1
                            down_val = up_plots[-1]
                        xy_plots = [None, None]
                        xy_plots[probe_coord] = [[up_start, [up_val]],
                                                 [down_end, [down_val]]]
                        grid = self._draw_plots(grid, xy_plots[0], xy_plots[1], (0, 255, 255))
                # move scan past this edge
                # ToDo: don't bother with this? dups are no longer possible due to pixel clearing
                skip_edge = edge
                continue

        edges.sort(key=lambda e: (int(round(e.position)), int(round(e.start)), int(round(e.length))))

        return edges, grid

    def _measure_gap(self, this_position, that_position, same_limit, position_limit=None):
        """ given two positions, determine if 'this' is above, below or the same as 'that',
            when wrapping the given positions must be less than (position_limit / 2) apart to be unambiguous,
            same_limit is the distance (in pixels) within which the positions are considered the same,
            position_limit is used for wrap detection, None means no wrapping,
            returns -n if 'this' is below 'that' by n, 0 if the same or +n if above by n
            """

        if position_limit is not None:
            this_position = int(round(this_position % position_limit))
            that_position = int(round(that_position % position_limit))
            offset = position_limit
        else:
            this_position = int(round(this_position))
            that_position = int(round(that_position))
            offset = 0                   # make wrapping irrelevant

        gap_1 = math.fabs(this_position - that_position)
        gap_2 = math.fabs((offset + this_position) - that_position)
        min_gap = min(gap_1, gap_2)

        if min_gap > same_limit:
            # they are not the same position, work out what the gap is
            # nb: we know this_position and that_position are not the same to get here
            if gap_1 > gap_2:
                # this means we've wrapped (or are not wrapping)
                return (offset + this_position) - that_position
            else:
                # not wrapping/wrapped, just return the difference
                return this_position - that_position
        else:
            # they are in the same position within the given limit
            return 0

    def _get_combined_edges(self, target, context, centres):
        """ find all the edges (black to white (up) and white to black (down)) in the given target,
            direction is top-down or bottom-up or both when looking for ring edges,
            and left-to-right or right-to-left or both when looking for bit edges,
            top-down or left-to-right specify using black-to-white edges,
            bottom-up or right-to-left specify using white-to-black edges,
            when both, both edge transitions are used,
            centres is just passed on to the edge detector,
            this function handles the logistics of finding edges of both types and merging them into
            a single vector for returning to the caller, an empty vector is returned if no edges found,
            the returned vector consists of Edge instances in ascending position order
            """

        prefix = context.prefix
        min_length = context.min_length
        max_length = context.max_cross_coord + 1
        nominal_length = context.nominal_length
        min_width = context.min_width
        max_width = context.max_scan_coord + 1
        nominal_width = context.nominal_width
        edge_type = context.type
        x_order = context.x_order
        y_order = context.y_order
        b2w = context.b2w
        w2b = context.w2b

        if b2w:
            # get black-to-white edges
            b2w_edges, threshold = self._get_transitions(target, x_order, y_order, False)
            edges_up, b2w_grid = self._get_edges(b2w_edges, context, centres, threshold, Scan.LEADING_EDGE)
            if self.save_images:
                if b2w_grid is not None:
                    self._unload(b2w_grid, '{}-b2w-edges'.format(edge_type))
        else:
            edges_up = []

        if w2b:
            # get white_to_black edges
            w2b_edges, threshold = self._get_transitions(target, x_order, y_order, True)
            edges_down, w2b_grid = self._get_edges(w2b_edges, context, centres, threshold, Scan.TRAILING_EDGE)
            if self.save_images:
                if w2b_grid is not None:
                    self._unload(w2b_grid, '{}-w2b-edges'.format(edge_type))
        else:
            edges_down = []

        # merge the two sets of edges so that got a single list in ascending position order,
        # each is separately in order now, we get them separately and merge rather than do both
        # at once 'cos they can get very close in small images
        edges_up_down = edges_up + edges_down
        edges_up_down.sort(key=lambda e: (int(round(e.position)), int(round(e.start)), int(round(e.length))))

        if self.logging:
            self._log('{}: found {} total edges, {} leading and {} trailing'.
                      format(prefix, len(edges_up_down), len(edges_up), len(edges_down)))
            for edge in edges_up_down:
                if edge.edge == Scan.LEADING_EDGE:
                    self._log('    leading: {}'.format(edge))
                else:
                    self._log('    trailing: {}'.format(edge))
            self._log('{}: min/nominal/max length: {:.2f}, {:.2f}, {:.2f}, width: {:.2f}, {:.2f}, {:.2f}'.
                      format(prefix, min_length, nominal_length, max_length,
                                     min_width, nominal_width, max_width))

        # merge near neighbour edges, this can happen if we get a 'saddle' in a very blurred edge
        merged_edges = self._merge_neighbours(context, edges_up_down)

        return merged_edges

    def _merge_neighbours(self, context, edges):
        """ given a list of edges in a suitable sort order, merge those that are 'close',
            returns the merged list
            """

        min_width = context.min_width
        max_width = context.max_scan_coord + 1
        max_useful_length = context.length_limit

        merged_edges = []
        done_edges = []
        for this in range(len(edges)):
            if this in done_edges:
                # already been seen
                continue
            this_edge = edges[this]
            if this_edge.length > max_useful_length:
                # ignore near full length edges, they are not interesting and/or joined up noise
                if self.logging:
                    self._log('{}: ignoring near full edge {}, limit is {}'.
                              format(context.prefix, this_edge, max_useful_length))
                continue
            # ToDo: is there a wrapping issue here?
            for that in range(this + 1, len(edges)):
                if that in done_edges:
                    # already been seen
                    continue
                that_edge = edges[that]
                if that_edge.length > max_useful_length:
                    # ignore near full length edges, they are not interesting and/or joined up noise
                    if self.logging:
                        self._log('{}: ignoring near full edge {}, limit is {}'.
                                  format(context.prefix, that_edge, max_useful_length))
                    continue
                if self._measure_gap(this_edge.position, that_edge.position, min_width, max_width) != 0:
                    # gone far enough to have found all our close neighbours
                    break
                joined_edge = self._join_edges(context, this_edge, that_edge)
                if joined_edge is not None:
                    if joined_edge.length > max_useful_length:
                        # joined is too long to be believed, ignore it
                        if self.logging:
                            self._log('{}: dropping joined edge {}, exceeds length limit {}'.
                                      format(context.prefix, joined_edge, max_useful_length))
                    else:
                        this_edge = joined_edge
                    # always mark it as done
                    done_edges.append(that)
            merged_edges.append(this_edge)
            done_edges.append(this)

        return merged_edges

    def _join_edges(self, context, this, that):
        """ given this and that edges join them together and return the joined edge if they can be merged,
            otherwise return None, to be merged they must be close to each other and overlap and be the same type,
            this is a helper for _get_combined_edges
            """

        if this.edge != that.edge:
            # cannot merge leading to trailing edges or vice-versa
            return None

        prefix = context.prefix
        max_length = context.max_cross_coord + 1
        min_length = 1  # we want some precision here

        gap1 = self._measure_gap(this.start + this.length, that.start, min_length, max_length)
        if gap1 < 0:
            # this ends before that start
            return None

        gap2 = self._measure_gap(this.start, that.start + that.length, min_length, max_length)
        if gap2 > 0:
            # this starts after that end
            return None

        # they overlap
        gap3 = self._measure_gap(this.start, that.start, min_length, max_length)
        gap4 = self._measure_gap(this.start + this.length, that.start + that.length, min_length, max_length)

        edge_length = this.length
        edge_start = this.start
        if gap3 > 0:
            # this means that started earlier than this
            edge_length += gap3          # extend the length by the gap
            edge_start = that.start      # ..and set the earlier start
        if gap4 < 0:
            # this means that ended later than this
            edge_length -= gap4          # extend the length by the (-ve) gap

        edge_position = (this.position + that.position) / 2
        edge_span = max(this.span, that.span)
        edge_width = max(this.width, that.width)

        new_edge = Scan.Edge(edge_position, edge_length, edge_span, edge_start, edge_width, edge=this.edge)

        if self.logging:
            self._log('{}: merging this and that: {}, {} into {}'.
                      format(prefix, this, that, new_edge))

        return new_edge

    def _finalise_rings(self, context, edges, edge_type, position_limit):
        """ given a vector of ring Edge instance references for every x 'finalise' them,
            edges contains either a list of initial edges or final edges in y,
            edges is a list of *references* to Edge instances, this is important 'cos we tweak them,
            edge_type specifies which edge type to consider, either LEADING_EDGE or TRAILING_EDGE,
            when LEADING_EDGE what we're doing is either dropping TRAILING_EDGE types in edges or adding
            a missing LEADING_EDGE depending on how close it is the position_limit, position_limit in this
            case is the inner edge, when TRAILING_EDGE its similar except we add a missing TRAILING_EDGE
            and position_limit is the outer edge,
            the function returns a list of Edge instances that must be added to the source list and has
            marked Edge instances that should be removed by setting their start property to None,
            the edges list contents are destroyed by this function
            """

        max_x = context.max_x
        min_width = context.min_width
        min_length = context.min_length

        new_rings = []
        for x in range(len(edges)):
            ring = edges[x]
            if ring is None:
                # we've already dealt with it
                continue
            if ring.edge == edge_type:
                # this edge type is not relevant here (this is what we think is missing and its not, so no action)
                continue
            if ring.where is None:
                # already due to dump this one
                continue

            if self._measure_gap(position_limit, ring.position, min_width * 2) == 0:
                # this is too close to the limit, drop it,
                # nb: we looked two min_widths away 'cos we know there should be a black ring around
                # a limit edge, this ensure we have room to insert a new edge
                if self.logging:
                    if edge_type == Scan.TRAILING_EDGE:
                        msg = 'leading'
                    else:
                        msg = 'trailing'
                    self._log('cells: dropping {} ring edge {} too close to outer edge {}, limit is {:.2f}'.
                              format(msg, ring, position_limit, min_width * 2))
                # mark it for demolition
                ring.where = None
                continue

            # got a dangling edge, terminate it near the position_limit by adding an appropriate edge
            # we know there is at least a min_width * 2 gap from the position_limit, so putting the
            # new edge a min_width away is not going to overlap it
            # find out how long the edge needs to be (by iterating edges over the length of this edge)
            new_start = x
            new_length = 0
            for dx in range(x, x + ring.length):
                if edges[dx % max_x] == ring:
                    edges[dx % max_x] = None
                    new_length += 1
                else:
                    break
            if new_length < min_length:
                # too small, its just a noise gap, so ignore it
                continue
            if edge_type == Scan.TRAILING_EDGE:
                # just before the outer edge
                new_position = position_limit - min_width
            else:
                # just after the inner edge
                new_position = position_limit + min_width
            new_edge = Scan.Edge(new_position, new_length, start=new_start, edge=edge_type)
            if self.logging:
                if edge_type == Scan.TRAILING_EDGE:
                    msg = 'trailing'
                else:
                    msg = 'leading'
                self._log('cells: adding missing {} edge {}'.format(msg, new_edge))

            new_rings.append(new_edge)

        return new_rings

    def _extract_rings(self, context, ring_edges, inner_position, outer_position):
        """ given a set of ring edges as detected, extract the useful ones """

        max_x = context.max_x
        max_useful_length = context.length_limit
        min_width = context.min_width
        min_length = context.min_length

        # we are only interested in rings between the inner edge and the outer edge, chuck the rest
        # trailing edges after the inner edge (white noise creeping down) and leading
        # edges before the outer edge (white noise creeping up) are discarded
        rings = []
        first_edge = [None for _ in range(max_x)]  # set to the first ring seen for every x
        last_edge = [None for _ in range(max_x)]  # set to the last ring seen for every x
        for ring in ring_edges:
            if self._measure_gap(ring.position, inner_position, 1) <= 0:
                # do not want to consider rings inside the inner edge, nor the inner edge
                if self.logging:
                    self._log('cells: ignoring ring edge {}, inside inner-edge at {:.2f}'.
                              format(ring, inner_position))
                continue
            if self._measure_gap(ring.position, outer_position, 1) >= 0:
                # do not want to consider rings outside the outer edge, nor the outer edge
                if self.logging:
                    self._log('cells: ignoring ring edges beyond {}, outside outer-edge at {:.2f}'.
                              format(ring, outer_position))
                # not interested in anything beyond here
                break
            if ring.length > max_useful_length:
                # these are marker rings and/or noise and not useful
                if self.logging:
                    self._log('cells: ignoring long ring edge {}, limit is {}'.
                              format(ring, max_useful_length))
                continue

            # got a keeper (for now)
            rings.append(ring)
            if self.logging:
                self._log('cells: adding ring edge {}'.format(ring))

            # note first edge for possible termination/dumping later
            for dx in range(ring.length):
                if first_edge[(dx + ring.start) % max_x] is None:
                    first_edge[(dx + ring.start) % max_x] = ring

            # note the last edge for possible termination/dumping later
            for dx in range(ring.length):
                last_edge[(dx + ring.start) % max_x] = ring

        # now tidy up the borderline edges
        new_trailing_rings = self._finalise_rings(context, last_edge, Scan.TRAILING_EDGE, outer_position)
        new_leading_rings = self._finalise_rings(context, first_edge, Scan.LEADING_EDGE, inner_position)

        # now actually chuck the duds by scanning backwards so index stays valid as we delete stuff
        for ring in range(len(rings) - 1, -1, -1):
            if rings[ring].where is None:
                del rings[ring]

        # add the missing edges
        rings += (new_trailing_rings + new_leading_rings)

        return rings

    def _find_cells(self):
        """ find target cells within each blob in our image,
            for each blob we determine a grid of all ring and bit edges,
            returns a list of a grid per blob where one was detected,
            there are numerous validation constraints that may result in a target being rejected,
            returns a list of target candidates found which may be empty if none found (as instances of Target)
            and when debugging an image with all rejects labelled with why rejected (None if not debugging)
            """

        # find the blobs in the image
        blobs = self._find_blobs()
        if len(blobs) == 0:
            # no blobs here
            return [], None

        targets = []
        rejects = []
        for blob in blobs:
            self.centre_x = blob.pt[0]
            self.centre_y = blob.pt[1]
            blob_size = blob.size / 2  # change diameter to radius

            # do the polar to cartesian projection
            projected, orig_radius = self._project(self.centre_x, self.centre_y, blob.size)  # this does not fail

            # do the perspective correction
            flattened, scale, reason = self._flatten(projected, orig_radius)
            if reason is not None:
                # failed - this means some constraint was not met
                if self.logging:
                    self._log('cells: {}'.format(reason))
                if self.save_images:
                    rejects.append(self.Reject(self.centre_x, self.centre_y, blob_size, None, reason))
                continue
            max_x, max_y = flattened.size()

            ring_context = self.Context(flattened, Scan.UP_AND_DOWN)
            probe_width = ring_context.min_length
            ring_width = ring_context.nominal_width
            min_length = ring_context.min_length
            max_useful_length = ring_context.length_limit
            inner_position = int(math.floor(ring_width * Scan.INNER_BLACK))
            outer_position = int(math.ceil(ring_width * Scan.OUTER_WHITE))

            # get the ring edges
            # probe such that guaranteed to hit any edge over the min length
            probe_centres = [x for x in range(probe_width, max_x - probe_width + 1, probe_width)]
            ring_edges = self._get_combined_edges(flattened, ring_context, probe_centres)
            if len(ring_edges) == 0:
                # we did not find any edges
                if self.logging:
                    self._log('no ring edges found')
                if self.save_images:
                    rejects.append(self.Reject(self.centre_x, self.centre_y,
                                               blob_size, None, 'no ring edges found'))
                continue
            if self.save_images:
                # draw edge segments as detected in red
                lines = []
                for edge in ring_edges:
                    lines.append([edge.start, edge.position,
                                  (edge.start + edge.length - 1) % max_x, edge.position])
                grid = self._draw_lines(flattened, lines, (0, 0, 255))
                # draw inner and outer edges
                lines = [inner_position, outer_position]
                grid = self._draw_grid(grid, lines, colours=(128, 0, 128))

            # the targets are constructed such that there is an edge for every bit across all the rings,
            # edge ends represent bit edges, so given the list we have in ring_edges we can compute
            # all the bit cell boundaries

            # extract the useful rings for bit edge detection
            rings = self._extract_rings(ring_context, ring_edges, inner_position, outer_position)

            # if no ring edges left, we're looking at junk
            if len(rings) == 0:
                if self.logging:
                    self._log('cells: no ring edges inside inner ring ({:.2f}) and outer ring ({:.2f})'.
                              format(inner_position, outer_position))
                if self.save_images:
                    rejects.append(self.Reject(self.centre_x, self.centre_y,
                                               blob_size, None, 'no data ring edges found'))
                continue

            if self.save_images:
                # re-draw edges we're gonna use in blue
                lines = []
                for edge in rings:
                    lines.append([edge.start, edge.position,
                                  (edge.start + edge.length - 1) % max_x, edge.position])
                grid = self._draw_lines(grid, lines, (255, 0, 0))

            # the ring edge ends represent bit edges, ring edges are either leading or trailing,
            # overlapping leading edges with no intervening trailing edge are merged, as are overlapping
            # trailing edges (they represent 'saddles', a false floor, in the luminance slopes)
            # ToDo: the above

            # get all bit edges (as either end of a ring edge) in ascending order of x
            bit_edges = []
            for ring in rings:
                bit_edges.append(ring.start)
                bit_edges.append((ring.start + ring.length) % max_x)
            bit_edges.sort()             # put into ascending x

            # merge duplicates (we set the mean for all edges that are 'close')
            bit_context = self.Context(flattened, Scan.LEFT_AND_RIGHT)
            min_width = bit_context.min_width
            max_width = bit_context.max_scan_coord + 1
            bits = []
            done_edges = []
            for this in range(len(bit_edges)):
                if this in done_edges:
                    # already been seen
                    continue
                this_edge = bit_edges[this]
                last_edge = this_edge
                matched = this_edge
                matches = 1
                # ToDo: is there a wrapping issue here?
                for that in range(this + 1, len(bit_edges)):
                    if that in done_edges:
                        # already been seen
                        continue
                    that_edge = bit_edges[that]
                    if self._measure_gap(this_edge, that_edge, min_width, max_width) != 0:
                        # gone far enough to have found all our close neighbours
                        break
                    if self.logging:
                        self._log('cells: bit edges at {} and {} will be merged'.format(last_edge, that_edge))
                    last_edge = that_edge
                    matched += that_edge
                    matches += 1
                    done_edges.append(that)
                merged_edge = matched / matches
                if self.logging:
                    if matches > 1:
                        self._log('cells: adding bit edge at {:.2f} (merge of {} bit edges between {} and {})'.
                                  format(merged_edge, matches, this_edge, last_edge))
                    else:
                        self._log('cells: adding bit edge at {:.2f}'.format(merged_edge))
                bits.append(merged_edge)
                done_edges.append(this)

            # build ring edge list for each unique bit edge
            slices = []
            for bit in range(len(bits)):
                slice = [inner_position]               # all slices have the inner edge marker
                bit_position = bits[bit] + min_length  # nominal bit position for comparison purposes
                for ring in rings:
                    ring_end = ring.start + ring.length - 1
                    if ring_end >= max_x:
                        # this ring edge wraps
                        ring_end %= max_x
                        if bit_position > ring.start:
                            # our bit is inside the start, so its part of the slice
                            pass
                        elif bit_position > ring_end:
                            # this ring edge not part of our bit
                            continue
                    else:
                        # this ring edge does not wrap
                        if bit_position < ring.start:
                            # our bit is before this ring, so not part of the slice
                            continue
                        if bit_position > ring_end:
                            # our bit is after the end of this ring edge, so not part of the slice
                            continue
                        # our bit is inside this ring edge, so part of the slice
                    slice.append(int(round(ring.position)))
                slice.append(outer_position)           # all slices have the outer edge marker
                bit_start = bits[bit]
                bit_end = bits[(bit + 1) % len(bits)]
                if bit_end < bit_position:
                    # we've wrapped
                    bit_end += max_x
                bit_width = bit_end - bit_start
                slices.append(((bit_start, bit_width), slice))

            if self.logging:
                self._log('cells: {} bit edges: {}'.format(len(bits), vstr(bits)))
                for slice in slices:
                    self._log('    bit at {:.2f} for {:.2f}, edges: {}'.format(slice[0][0], slice[0][1], slice[1]))
            if self.save_images:
                # draw our detected bit edges
                grid = self._draw_grid(grid, None, bits, colours=(0, 128, 0))
                self._unload(grid, 'cells')

            if len(bits) != Scan.NUM_BITS:
                if self.logging:
                    self._log('cells: need {} bit edges, found {}'.
                              format(Scan.NUM_BITS, len(bits)))
                if self.save_images:
                    rejects.append(self.Reject(self.centre_x, self.centre_y,
                                               blob_size, None, 'need {} bit edges, found {}'.
                                                                format(Scan.NUM_BITS, len(bits))))
                continue

            targets.append(self.Target(self.centre_x, self.centre_y, blob_size, scale, flattened, slices))

        if self.save_images:
            # label all the blobs we processed that were rejected
            labels = self.transform.copy(self.image)
            for reject in rejects:
                x = reject.centre_x
                y = reject.centre_y
                blob_size = reject.blob_size / 2  # assume blob detected is just inner two white rings
                ring = reject.target_size
                reason = reject.reason
                if ring is None:
                    # it got rejected before its inner/outer ring was detected
                    ring = blob_size * 4
                # show blob detected
                colour = (255, 0, 0)  # blue
                labels = self.transform.label(labels, (x, y, blob_size), colour)
                # show reject reason
                colour = (0, 0, 255)  # red
                labels = self.transform.label(labels, (x, y, ring), colour, '{:.0f}x{:.0f}y {}'.format(x, y, reason))
        else:
            labels = None

        return targets, labels

    def decode_targets(self):
        """ find and decode the targets in the source image,
            returns a list of x,y blob co-ordinates, the encoded number there (or None) and the level of doubt
            """

        targets, labels = self._find_cells()
        if len(targets) == 0:
            if self.logging:
                self._log('image {} does not contain any target candidates'.format(self.original.source))
            if self.save_images:
                if labels is not None:
                    self._unload(labels, 'targets', 0, 0)
                else:
                    self._unload(self.image, 'targets', 0, 0)
            return []

        numbers = []
        for target in targets:
            self.centre_x = target.centre_x
            self.centre_y = target.centre_y
            blob_size = target.size
            target_scale = target.scale
            image = target.image
            slices = target.slices

            # what we have now is a list of potential targets, each target consists of a list of bit edges
            # (a slice) and the ring edges within those edges, the first edge is the inner ring and the last
            # the outer, in between are the black-to-white and white-to-black transitions, we know the first
            # is black (the inner black ring) and the last is black (the outer black ring), in between there
            # are 4 rings, 3 data rings and a timing ring, the nominal width of each ring is outer edge minus
            # inner edge over six, the nominal ring before the inner is used to get the white level, and the
            # nominal ring before the outer edge is used to get the black level, we sample the centre of the
            # 3 data rings for decoding, to get here we know we have found the correct number of bit edges

            # get all the samples
            white_level = []
            black_level = []
            data_ring_1 = []
            data_ring_2 = []
            data_ring_3 = []
            for slice in slices:
                slice_start = slice[0][0]      # x co-ord
                slice_width = slice[0][1]      # pixels
                slice_centre = slice_start + (slice_width / 2)
                slice_edges = slice[1]         # list of y co-ords (at least 2 - inner and outer edge)
                ring_width = (slice_edges[-1] - slice_edges[0]) / (Scan.NUM_RINGS - 3)
                white_cell = (slice_edges[0] - ring_width) + (ring_width / 2)
                black_cell = (slice_edges[-1] - ring_width) + (ring_width / 2)
                # ToDo: use the slice edges to get a better ring position estimate
                data_cell_1 = (slice_edges[0] + ring_width) + (ring_width / 2)
                data_cell_2 = data_cell_1 + ring_width
                data_cell_3 = data_cell_2 + ring_width
                sample_height = ring_width * Scan.SAMPLE_HEIGHT_FACTOR
                sample_width = slice_width * Scan.SAMPLE_WIDTH_FACTOR
                white_level.append(self._get_sample(image, slice_centre, white_cell, sample_width, sample_height))
                black_level.append(self._get_sample(image, slice_centre, black_cell, sample_width, sample_height))
                data_ring_1.append(self._get_sample(image, slice_centre, data_cell_1, sample_width, sample_height))
                data_ring_2.append(self._get_sample(image, slice_centre, data_cell_2, sample_width, sample_height))
                data_ring_3.append(self._get_sample(image, slice_centre, data_cell_3, sample_width, sample_height))

            # decode our bits
            number, doubt, bits = self.coder.unbuild([data_ring_1, data_ring_2, data_ring_3],
                                                     [white_level, black_level])

            # calculate the target size relative to the original image (as the largest outer edge)
            target_size = 0
            for slice in slices:
                if slice[1][-1] > target_size:
                    target_size = slice[1][-1]
            target_size *= target_scale  # scale to size in original image

            # add this result
            numbers.append(Target(number, doubt, self.centre_x, self.centre_y, target_size, blob_size))

            if self.logging:
                number = numbers[-1]
                self._log('number:{}, bits:{}'.format(number.number, bits), number.centre_x, number.centre_y)
                if number.number is None:
                    self._log('--- white samples:{}'.format(vstr(white_level)))
                    self._log('--- black samples:{}'.format(vstr(black_level)))
                    self._log('--- ring1 samples:{}'.format(vstr(data_ring_1)))
                    self._log('--- ring2 samples:{}'.format(vstr(data_ring_2)))
                    self._log('--- ring3 samples:{}'.format(vstr(data_ring_3)))
            if self.save_images:
                number = numbers[-1]
                if number.number is None:
                    colour = (255, 0, 255, 0)  # purple
                    label = 'code is invalid ({})'.format(number.doubt)
                else:
                    colour = (0, 255, 0)  # green
                    label = 'code is {} ({})'.format(number.number, number.doubt)
                if labels is not None:
                    # draw the detected blob in blue
                    k = (number.centre_x, number.centre_y, number.blob_size)
                    labels = self.transform.label(labels, k, (255, 0, 0))
                    # draw the result
                    k = (number.centre_x, number.centre_y, number.target_size)
                    labels = self.transform.label(labels, k, colour, '{:.0f}x{:.0f}y {}'.
                                                  format(number.centre_x, number.centre_y, label))
                    self._unload(labels, 'targets', 0, 0)

        return numbers

    def _get_sample(self, target, centre_x, centre_y, sample_width, sample_height):
        """ given a perspective corrected image, a bit number, data and a box, get the luminance sample,
            centre_x specifies the centre of the sample box in the x direction in the target,
            centre_y specifies the centre of the sample box in the y direction in the target,
            sample_width and sample_height is the width (x) and height (y) of the area to sample,
            returns the average luminance level in the sample box
            """

        max_x, _ = target.size()

        x_width = sample_width / 2
        y_width = sample_height / 2

        start_x = math.ceil(centre_x - x_width)
        stop_x = math.floor(centre_x + x_width)

        start_y = math.ceil(centre_y - y_width)
        stop_y = math.floor(centre_y + y_width)

        luminance_accumulator = 0
        pixels_found = 0
        for x in range(start_x, stop_x + 1):
            for y in range(start_y, stop_y + 1):
                pixel = target.getpixel((x % max_x), y)
                if pixel is not None:
                    luminance_accumulator += pixel
                    pixels_found += 1

        if pixels_found > 0:
            return luminance_accumulator / pixels_found
        else:
            return MIN_LUMINANCE

    def _log(self, message, centre_x=None, centre_y=None, fatal=False):
        """ print a debug log message
            centre_x/y are the co-ordinates of the centre of the associated blob, if None use decoding context
            centre_x/y of 0,0 means no x/y identification in the log
            iff fatal is True an exception is raised, else the message is just printed
            """
        if centre_x is None:
            centre_x = self.centre_x
        if centre_y is None:
            centre_y = self.centre_y
        if centre_x > 0 and centre_y > 0:
            message = '{} {:.0f}x{:.0f}y - {}'.format(self._log_prefix, centre_x, centre_y, message)
        else:
            message = '{} {}'.format(self._log_prefix, message)
        if self._log_folder:
            # we're logging to a file
            if self._log_file is None:
                filename, ext = os.path.splitext(self.original.source)
                self._log_file = open('{}/{}.log'.format(self._log_folder, filename), 'w')
            self._log_file.write('{}\n'.format(message))
        if fatal:
            raise Exception(message)
        elif self.show_log:
            # we're logging to the console
            print(message)

    def _unload(self, image, suffix, centre_x=None, centre_y=None):
        """ unload the given image with a name that indicates its source and context,
            suffix is the file name suffix (to indicate context),
            centre_x/y identify the blob the image represents, if None use decoding context,
            centre_x/y of 0,0 means no x/y identification on the image,
            as a diagnostic aid to find co-ordinates, small tick marks are added along the edges every 10 pixels
            """

        # add tick marks
        max_x, max_y = image.size()
        lines = []
        for x in range(10, max_x, 10):
            lines.append([x, 0, x, 1])
            lines.append([x, max_y - 2, x, max_y - 1])
        image = self._draw_lines(image, lines, (128, 0, 128))
        lines = []
        for y in range(10, max_y, 10):
            lines.append([0, y, 1, y])
            lines.append([max_x - 2, y, max_x - 1, y])
        image = self._draw_lines(image, lines, (128, 0, 128))

        # construct the file name
        if centre_x is None:
            centre_x = self.centre_x
        if centre_y is None:
            centre_y = self.centre_y
        if centre_x > 0 and centre_y > 0:
            name = '{:.0f}x{:.0f}y-'.format(centre_x, centre_y)
        else:
            name = ''

        # save the image
        filename = image.unload(self.original.source, '{}{}'.format(name, suffix))
        if self.logging:
            self._log('{}: image saved as: {}'.format(suffix, filename), centre_x, centre_y)

    def _draw_contour(self, source, contour, colour=(0, 255, 0)):
        """ draw a 'contour' in the given colour,
            a 'contour' is an array of arrays of x,y points (as produced by Transform.contour)
            """
        objects = []
        for points in contour:
            objects.append({"colour": colour,
                            "type": self.transform.POINTS,
                            "points": points})
        target = self.transform.copy(source)
        return self.transform.annotate(target, objects)

    def _draw_plots(self, source, plots_x=None, plots_y=None, colour=(0, 0, 255)):
        """ draw plots in the given colour, each plot is a set of points and a start x or y,
            returns a new colour image of the result
            """
        objects = []
        if plots_x is not None:
            for plot in plots_x:
                objects.append({"colour": colour,
                                "type": self.transform.PLOTX,
                                "start": plot[0],
                                "points": plot[1]})
        if plots_y is not None:
            for plot in plots_y:
                objects.append({"colour": colour,
                                "type": self.transform.PLOTY,
                                "start": plot[0],
                                "points": plot[1]})
        target = self.transform.copy(source)
        return self.transform.annotate(target, objects)

    def _draw_lines(self, source, lines, colour=(0, 0, 255)):
        """ draw lines in given colour,
            lines param is an array of start-x,start-y,end-x,end-y tuples
            """
        objects = []
        for line in lines:
            objects.append({"colour": colour,
                            "type": self.transform.LINE,
                            "start": [line[0], line[1]],
                            "end": [line[2], line[3]]})
        target = self.transform.copy(source)
        return self.transform.annotate(target, objects)

    def _draw_boxes(self, source, boxes, colour=(0, 0, 255)):
        """ draw boxes in the image of the given colour,
            boxes is an array of rectangle corners: top-left and bottom-right as (x,y,x,y) tuples
            """
        objects = []
        for box in boxes:
            if box is not None:
                objects.append({"colour": colour,
                                "type": self.transform.RECTANGLE,
                                "start": (box[0], box[1]),
                                "end": (box[2], box[3])})
        target = self.transform.copy(source)
        return self.transform.annotate(target, objects)

    def _draw_grid(self, source, horizontal=None, vertical=None, colours=None, radius=None, box=None):
        """ draw horizontal and vertical lines across the given source image,
            if a radius is given: circles of that radius are drawn at each horizontal/vertical intersection,
            if a box is given: rectangles of that size are drawn centred at each horizontal/vertical intersection,
            colours is a tuple of up to 3 colours - horizontal line, vertical line, intersection colours,
            if colours is None it defaults to dark red, dark blue, dark green,
            if only one colour is provided the same is used for all objects,
            returns a new colour image of the result
            """

        if colours is None:
            colours = ((0, 0, 128), (128, 0, 0), (0, 128, 0))
        if type(colours[0]) != tuple:
            # assume only been given a single colour
            colours = (colours, colours, colours)

        max_x, max_y = source.size()
        objects = []
        if horizontal is not None:
            for h in horizontal:
                objects.append({"colour": colours[0],
                                "type": self.transform.LINE,
                                "start": (0, h),
                                "end": (max_x - 1, h)})

        if vertical is not None:
            for v in vertical:
                objects.append({"colour": colours[1],
                                "type": self.transform.LINE,
                                "start": (v, 0),
                                "end": (v, max_y - 1)})

        if horizontal is not None and vertical is not None:
            if radius is not None:
                for h in horizontal:
                    for v in vertical:
                        objects.append({"colour": colours[2],
                                        "type": self.transform.CIRCLE,
                                        "centre": (h, v),
                                        "radius": radius})
            if box is not None:
                width = box[0]
                height = box[1]
                for h in horizontal:
                    for v in vertical:
                        start = (int(round((v - width / 2))), int(round((h - height / 2))))
                        end = (int(round((v + width / 2))), int(round((h + height / 2))))
                        objects.append({"colour": colours[2],
                                        "type": self.transform.RECTANGLE,
                                        "start": start,
                                        "end": end})

        target = self.transform.copy(source)
        return self.transform.annotate(target, objects)

    def _draw_below(self, source, threshold, colour=(0, 0, 255)):
        """ re-draw every pixel below the given threshold in the given colour,
            the given source must be a greyscale image,
            the colour is interpreted as a scaling factor, 0=none, 255=all,
            the intensity of the re-drawn pixel is the same as that already there in some other colour,
            returns a colour image
            """
        scale = (colour[0] / 255, colour[1] / 255, colour[2] / 255)
        target = self.transform.copy(source)
        target.incolour()
        max_x, max_y = target.size()
        for x in range(max_x):
            for y in range(max_y):
                grey_pixel = source.getpixel(x, y)
                if grey_pixel < threshold:
                    colour_pixel = (grey_pixel * scale[0], grey_pixel * scale[1], grey_pixel * scale[2])
                else:
                    colour_pixel = (grey_pixel, grey_pixel, grey_pixel)
                target.putpixel(x, y, colour_pixel)
        return target


class Test:
    # exit codes from scan
    EXIT_OK = 0  # found what was expected
    EXIT_FAILED = 1  # did not find what was expected
    EXIT_EXCEPTION = 2  # an exception was raised

    def __init__(self, log=None):
        self.min_num = None
        self.coder = None
        self.max_num = None
        self.frame = None
        self.num_rings = Ring.NUM_RINGS
        self.bits = Codec.BITS
        self.marker_bits = Codec.MARKER_BITS
        self.angles = None
        self.video_mode = None
        self.contrast = None
        self.offset = None
        self.debug_mode = None
        self.log_folder = log
        self.log_file = None
        self._log('')
        self._log('******************')
        self._log('Rings: {}, Total bits: {}, Marker bits: {}'.format(self.num_rings, self.bits, self.marker_bits))

    def __del__(self):
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def _log(self, message):
        """ print a log message and maybe write it to a file too """
        print(message)
        if self.log_folder is not None:
            if self.log_file is None:
                self.log_file = open('{}/test.log'.format(self.log_folder), 'w')
            self.log_file.write('{}\n'.format(message))

    def encoder(self, min_num, max_num, parity, edges):
        """ create the coder/decoder and set its parameters """
        self.min_num = min_num
        self.codec = Codec(min_num, max_num, parity, edges)
        self.max_num = min(max_num, self.codec.num_limit)
        self._log('Coder: {} parity, {} edges, available numbers are {}..{}'.
                  format(parity, edges, min_num, self.max_num))

    def folders(self, read=None, write=None):
        """ create an image frame and set the folders to read input media and write output media """
        self.frame = Frame(read, write)
        self._log('Frame: media in: {}, media out: {}'.format(read, write))

    def options(self, angles=None, mode=None, contrast=None, offset=None, debug=None, log=None):
        """ set processing options, only given options are changed """
        if angles is not None:
            self.angles = angles
        if mode is not None:
            self.video_mode = mode
        if contrast is not None:
            self.contrast = contrast
        if offset is not None:
            self.offset = offset
        if debug is not None:
            self.debug_mode = debug
        if log is not None:
            self.log_folder = log
        self._log('Options: angles {}, video mode {}, contrast {}, offset {}, debug {}, log {}'.
                  format(self.angles, self.video_mode, self.contrast, self.offset, self.debug_mode, self.log_folder))

    def coding(self):
        """ test for encode uniqueness and encode/decode symmetry """
        self._log('')
        self._log('******************')
        self._log('Check encode/decode from {} to {}'.format(self.min_num, self.max_num))

        def check(num):
            """ check encode/decode is symmetric
                returns None if check fails or the coded number if OK
                """
            encoded = self.codec.encode(num)
            if encoded is None:
                self._log('{} encodes as None'.format(num))
                return None
            decoded = self.codec.decode(encoded)
            if decoded != num:
                self._log('{} encodes to {} but decodes as {}'.format(num, encoded, decoded))
                return None
            return encoded

        try:
            good = 0
            bad = 0
            for n in range(self.min_num, self.max_num + 1):
                if check(n) is None:
                    bad += 1
                else:
                    good += 1
            self._log('{} good, {} bad'.format(good, bad))
        except:
            traceback.print_exc()
        self._log('******************')

    def decoding(self, black=MIN_LUMINANCE, white=MAX_LUMINANCE, noise=0):
        """ test build/unbuild symmetry
            black is the luminance level to use for 'black', and white for 'white',
            noise is how much noise to add to the luminance samples created, when
            set luminance samples have a random number between 0 and noise added or
            subtracted, this is intended to test the 'maybe' logic, the grey thresholds
            are set such that the middle 1/3rd of the luminance range is considered 'grey'
            """
        self._log('')
        self._log('******************')
        self._log('Check build/unbuild from {} to {} with black={}, white={} and noise {}'.
                  format(self.min_num, self.max_num, black, white, noise))
        try:
            colours = [black, white]
            good = 0
            doubted = 0
            fail = 0
            bad = 0
            levels = [[None for _ in range(self.bits)] for _ in range(2)]
            for bit in range(self.bits):
                levels[0][bit] = white
                levels[1][bit] = black
            for n in range(self.min_num, self.max_num + 1):
                rings = self.codec.build(n)
                samples = [[] for _ in range(len(rings))]
                for ring in range(len(rings)):
                    word = rings[ring]
                    for bit in range(self.bits):
                        # NB: Being encoded big-endian (MSB first)
                        samples[ring].insert(0, max(min(
                            colours[word & 1] + (random.randrange(0, noise + 1) - (noise >> 1)),
                            MAX_LUMINANCE), MIN_LUMINANCE))
                        word >>= 1
                m, doubt, bits = self.codec.unbuild(samples, levels)
                if m is None:
                    # failed to decode
                    fail += 1
                    if noise == 0:
                        self._log('    FAIL: {:03}-->{}, build={}, doubt={}, bits={}, samples={}'.
                                  format(n, m, rings, doubt, bits, samples))
                elif m != n:
                    # incorrect decode
                    bad += 1
                    self._log('****BAD!: {:03}-->{} , build={}, doubt={}, bits={}, samples={}'.
                              format(n, m, rings, doubt, bits, samples))
                elif doubt != '0.0':
                    # got doubt
                    doubted += 1
                    self._log('****DOUBT!: {:03}-->{} , build={}, doubt={}, bits={}, samples={}'.
                              format(n, m, rings, doubt, bits, samples))
                else:
                    good += 1
            self._log('{} good, {} doubted, {} bad, {} fail'.format(good, doubted, bad, fail))
        except:
            traceback.print_exc()
        self._log('******************')

    def test_set(self, size):
        """ make a set of test codes,
            the test codes consist of the minimum and maximum numbers plus those with the most
            1's and the most 0's and alternating 1's and 0's and N random numbers to make the
            set size up to that given
            """
        max_ones = -1
        max_zeroes = -1
        num_set = [self.min_num, self.max_num]
        max_ones_num = None
        max_zeroes_num = None
        num_bits = self.bits - self.marker_bits
        all_bits_mask = (1 << num_bits) - 1
        alt_ones = 0x55555555 & all_bits_mask
        alt_zeroes = 0xAAAAAAAA & all_bits_mask
        for num in range(self.min_num + 1, self.max_num - 1):
            code = self.codec.encode(num)
            if code is None:
                continue
            if code == alt_ones:
                num_set.append(num)
            if code == alt_zeroes:
                num_set.append(num)
            ones = 0
            zeroes = 0
            for bit in range(num_bits):
                mask = 1 << bit
                if (mask & code) != 0:
                    ones += 1
                else:
                    zeroes += 1
                if ones > max_ones:
                    max_ones = ones
                    max_ones_num = num
                if zeroes > max_zeroes:
                    max_zeroes = zeroes
                    max_zeroes_num = num
        if max_ones_num is not None:
            num_set.append(max_ones_num)
        if max_zeroes_num is not None:
            num_set.append(max_zeroes_num)
        while len(num_set) < size:
            num = random.randrange(self.min_num + 1, self.max_num - 1)
            if num in num_set:
                # don't want a duplicate
                continue
            num_set.append(num)
        num_set.sort()
        return num_set

    def code_words(self, numbers):
        """ test code-word rotation with given set plus the extremes (visual) """
        self._log('')
        self._log('******************')
        self._log('Check code-words (visual)')
        bin = '{:0' + str(self.bits) + 'b}'
        frm_ok = '{}(' + bin + ')=(' + bin + ', ' + bin + ', ' + bin + ')'
        frm_bad = '{}(' + bin + ')=(None)'
        try:
            for n in numbers:
                if n is None:
                    # this means a test code pattern is not available
                    continue
                rings = self.codec.build(n)
                if rings is None:
                    self._log(frm_bad.format(n, n))
                else:
                    self._log(frm_ok.format(n, n, rings[0], rings[1], rings[2]))
        except:
            traceback.print_exc()
        self._log('******************')

    def circles(self):
        """ test accuracy of co-ordinate conversions - polar to/from cartesian,
            also check polarToCart goes clockwise
            """
        self._log('')
        self._log('******************************************')
        self._log('Check co-ordinate conversions (radius 100)')
        # check clockwise direction by checking sign and relative size as we go round each octant
        #          angle, x-sign, y-sign, xy-sign
        octants = [[45, +1, -1, -1],
                   [90, +1, -1, +1],
                   [135, +1, +1, +1],
                   [180, +1, +1, -1],
                   [225, -1, +1, -1],
                   [270, -1, +1, +1],
                   [315, -1, -1, +1],
                   [360, -1, -1, -1]]
        # checks are:
        #  x * x-sign > 0
        #  y * y-sign > 0
        #  (x-y) * xy-sign > 0
        try:
            good = 0
            bad = 0
            radius = 100
            scale = 360 * 10  # 0.1 degrees
            angle = Angle(scale)
            for step in range(int(round(scale))):
                a = (step / scale) * 360
                cx, cy = angle.polarToCart(a, radius)
                ca, cr = angle.cartToPolar(cx, cy)
                rotation_err = 'angle out of range!'
                for octant in octants:
                    if a <= octant[0]:
                        # found the octant we are in
                        xsign = cx * octant[1]
                        ysign = cy * octant[2]
                        xysign = (xsign - ysign) * octant[3]
                        if xsign >= 0:
                            if ysign >= 0:
                                if xysign >= 0:
                                    # rotation correct
                                    rotation_err = None
                                else:
                                    rotation_err = 'x bigger than y'
                            else:
                                rotation_err = 'y sign'
                        else:
                            rotation_err = 'x sign'
                        break
                rerr = math.fabs(cr - radius)
                aerr = math.fabs(ca - a)
                if rerr > 0.01 or aerr > 0.3 or rotation_err is not None:
                    bad += 1
                    self._log(
                        '{:.3f} degrees --> {:.3f}x, {:.3f}y --> {:.3f} degrees, {:.3f} radius: aerr={:.3f}, rerr={:.3f}, rotation={}'.
                        format(a, cx, cy, ca, cr, aerr, rerr, rotation_err))
                else:
                    good += 1
            self._log('{} good, {} bad'.format(good, bad))
        except:
            traceback.print_exc()
        self._log('******************************************')

    def rings(self, folder, width):
        """ draw angle test rings in given folder (visual) """
        self._log('')
        self._log('******************')
        self._log('Draw an angle test ring (visual)')
        self.folders(write=folder)
        try:
            image_width = width * (self.num_rings + 1) * 2
            self.frame.new(image_width, image_width, MID_LUMINANCE)
            x, y = self.frame.size()
            ring = Ring(x >> 1, y >> 1, width, self.frame, self.contrast, self.offset)
            ring.code(000, (0x5555, 0xAAAA, 0x5555))
            self.frame.unload('test-code-000')
        except:
            traceback.print_exc()
        self._log('******************')

    def codes(self, folder, numbers, width):
        """ draw test codes for the given test_set in the given folder """
        self._log('')
        self._log('******************')
        self._log('Draw test codes (visual)')
        self.folders(write=folder)
        self._remove_test_codes(folder, 'test-code-')
        try:
            for n in numbers:
                if n is None:
                    # this means a test code pattern is not available
                    continue
                rings = self.codec.build(n)
                if rings is None:
                    self._log('{}: failed to generate the code rings'.format(n))
                else:
                    image_width = width * (self.num_rings + 1) * 2
                    self.frame.new(image_width, image_width, MID_LUMINANCE)
                    x, y = self.frame.size()
                    ring = Ring(x >> 1, y >> 1, width, self.frame, self.contrast, self.offset)
                    ring.code(n, rings)
                    self.frame.unload('test-code-{}'.format(n))
        except:
            traceback.print_exc()
        self._log('******************')

    def scan_codes(self, folder):
        """ find all the test codes in the given folder and scan them,
            these are all 'perfect' images saved as drawn, each file name
            is assumed to include the code number of the image
            """

        filelist = glob.glob('{}/test-code-*.*'.format(folder))
        filelist.sort()
        for f in filelist:
            f = os.path.basename(f)
            num = ''.join([s for s in f if s.isdigit()])
            if num == '':
                num = 0
            else:
                num = int(num)
            self.scan(folder, [num], f)

    def scan_media(self, folder):
        """ find all the media in the given folder and scan them,
            these are photos of codes in various states of distortion,
            each file name is assumed to include the code number in the image
            """

        filelist = glob.glob('{}/*.jpg'.format(folder))
        filelist.sort()
        for f in filelist:
            f = os.path.basename(f)
            num = ''.join([s for s in f if s.isdigit()])
            if num == '':
                num = 0
            else:
                num = int(num)
            self.scan(folder, [num], f)

    def scan(self, folder, numbers, image):
        """ do a scan for the code set in image in the given folder and expect the number given,
            returns an exit code to indicate what happened
            """
        self._log('')
        self._log('******************')
        self._log('Scan image {} for codes {}'.format(image, numbers))
        debug_folder = 'debug_images'
        self.folders(read=folder, write=debug_folder)
        exit_code = self.EXIT_OK  # be optimistic
        try:
            self._remove_derivatives(debug_folder, image)
            self.frame.load(image)
            scan = Scan(self.codec, self.frame, self.angles, self.video_mode,
                        debug=self.debug_mode, log=self.log_folder)
            results = scan.decode_targets()
            # analyse the results
            found = [False for _ in range(len(numbers))]
            analysis = []
            for result in results:
                centre_x = result.centre_x
                centre_y = result.centre_y
                num = result.number
                doubt = result.doubt
                size = result.target_size
                expected = None
                found_num = None
                for n in range(len(numbers)):
                    if numbers[n] == num:
                        # found another expected number
                        found[n] = True
                        found_num = num
                        expected = '{:b}'.format(self.codec.encode(num))
                        break
                analysis.append([found_num, centre_x, centre_y, num, doubt, size, expected])
            # create dummy result for those not found
            for n in range(len(numbers)):
                if not found[n]:
                    # this one is missing
                    num = numbers[n]
                    expected = self.codec.encode(num)
                    if expected is None:
                        # not a legal code
                        expected = 'not-valid'
                    else:
                        expected = '{:b}'.format(expected)
                    analysis.append([None, 0, 0, numbers[n], 0, 0, expected])
            # print the results
            for loop in range(3):
                for result in analysis:
                    found = result[0]
                    centre_x = result[1]
                    centre_y = result[2]
                    num = result[3]
                    doubt = result[4]
                    size = result[5]
                    expected = result[6]
                    if found is not None:
                        if loop != 0:
                            # don't want these in this loop
                            continue
                        # got what we are looking for
                        self._log('Found {} ({}) at {:.0f}x, {:.0f}y size {:.2f}, doubt level {}'.
                                  format(num, expected, centre_x, centre_y, size, doubt))
                        continue
                    if expected is not None:
                        if loop != 1:
                            # don't want these in this loop
                            continue
                        # did not get what we are looking for
                        if num == 0:
                            # this is used when we don't expect to find anything
                            continue
                        else:
                            prefix = '**** '
                        self._log('{}Failed to find {} ({})'.format(prefix, num, expected))
                        exit_code = self.EXIT_FAILED
                        continue
                    if True:
                        if loop != 2:
                            # don't want these in this loop
                            continue
                        # found something we did not expect
                        if num is None:
                            actual_code = 'not-valid'
                            prefix = ''
                        else:
                            actual_code = self.codec.encode(num)
                            if actual_code is None:
                                actual_code = 'not-valid'
                                prefix = ''
                            else:
                                actual_code = '{} ({:b})'.format(num, actual_code)
                                prefix = '**** UNEXPECTED **** ---> '
                        self._log('{}Found {} ({}) at {:.0f}x, {:.0f}y size {:.2f}, doubt level {}'.
                                  format(prefix, num, actual_code, centre_x, centre_y, size, doubt))
                        continue
        except:
            traceback.print_exc()
            exit_code = self.EXIT_EXCEPTION
        finally:
            del (scan)  # needed to close log files
        self._log('Scan image {} for codes {}'.format(image, numbers))
        self._log('******************')
        return exit_code

    def _remove_test_codes(self, folder, pattern):
        """ remove all the test code images with file names containing the given pattern in the given folder
            """
        filelist = glob.glob('{}/*{}*.*'.format(folder, pattern))
        for f in filelist:
            try:
                os.remove(f)
            except:
                self._log('Could not remove {}'.format(f))

    def _remove_derivatives(self, folder, filename):
        """ remove all the diagnostic image derivatives of the given file name in the given folder,
            a derivative is that file name prefixed by '_', suffixed by anything with any file extension
            """
        filename, _ = os.path.splitext(filename)
        filelist = glob.glob('{}/_{}_*.*'.format(folder, filename))
        for f in filelist:
            try:
                os.remove(f)
            except:
                self._log('Could not remove {}'.format(f))


def verify():
    # parameters
    min_num = 101  # min number we want
    max_num = 999  # max number we want (may not be achievable)
    parity = None  # code word parity to apply (None, 0=even, 1=odd)
    edges = 5  # how many bit transitions we want per code word
    contrast = 1.0  # reduce dynamic luminance range when drawing to minimise 'bleed' effects
    offset = 0.0  # offset luminance range from the mid-point, -ve=below, +ve=above

    test_codes_folder = 'codes'
    test_media_folder = 'media'
    test_log_folder = 'logs'
    test_ring_width = 32
    test_black = MIN_LUMINANCE + 64  # + 32
    test_white = MAX_LUMINANCE - 64  # - 32
    test_noise = MID_LUMINANCE >> 1
    test_scan_angle_steps = 90
    test_scan_video_mode = Scan.VIDEO_4K

    test_debug_mode = Scan.DEBUG_IMAGE
    #test_debug_mode = Scan.DEBUG_VERBOSE

    # setup test params
    test = Test(log=test_log_folder)
    test.encoder(min_num, max_num, parity, edges)
    test.options(angles=test_scan_angle_steps,
                 mode=test_scan_video_mode,
                 contrast=contrast,
                 offset=offset,
                 debug=test_debug_mode)

    # build a test code set
    test_num_set = test.test_set(10)

    test.coding()
    test.decoding(test_black, test_white, test_noise)
    test.circles()
    test.code_words(test_num_set)
    test.codes(test_codes_folder, test_num_set, test_ring_width)
    # test.rings(test_codes_folder, test_ring_width)

    # test.scan_codes(test_codes_folder)
    # test.scan_media(test_media_folder)

    # test.scan(test_media_folder, [000], 'photo-angle-test-flat.jpg')
    # test.scan(test_media_folder, [000], 'photo-angle-test-curved-flat.jpg')

    #test.scan(test_codes_folder, [101], 'test-code-101.png')

    # test.scan(test_media_folder, [365], 'photo-658-crumbled-very-distant.jpg')
    # test.scan(test_media_folder, [102], 'photo-102.jpg')
    # test.scan(test_media_folder, [365], 'photo-365.jpg')
    # test.scan(test_media_folder, [640], 'photo-640.jpg')
    # test.scan(test_media_folder, [658], 'photo-658.jpg')
    # test.scan(test_media_folder, [828], 'photo-828.jpg')
    # test.scan(test_media_folder, [102], 'photo-102-distant.jpg')
    # test.scan(test_media_folder, [365], 'photo-365-oblique.jpg')
    # test.scan(test_media_folder, [365], 'photo-365-blurred.jpg')
    # test.scan(test_media_folder, [658], 'photo-658-small.jpg')
    # test.scan(test_media_folder, [658], 'photo-658-crumbled-bright.jpg')
    # test.scan(test_media_folder, [658], 'photo-658-crumbled-dim.jpg')
    # test.scan(test_media_folder, [658], 'photo-658-crumbled-close.jpg')
    # test.scan(test_media_folder, [658], 'photo-658-crumbled-very-distant.jpg')
    # test.scan(test_media_folder, [658], 'photo-658-crumbled-blurred.jpg')
    # test.scan(test_media_folder, [658], 'photo-658-crumbled-dark.jpg')
    # test.scan(test_media_folder, [101, 102, 365, 640, 658, 828], 'photo-all-test-set.jpg')

    del (test)  # needed to close the log file(s)


if __name__ == "__main__":
    verify()
