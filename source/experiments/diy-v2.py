import os
import glob
import pathlib
import shutil

import contours
import numpy as np
import cv2
# import matplotlib.pyplot as plt
import rand
import math
import traceback
import time
from typing import List
from typing import Optional
import copy

""" coding scheme

    This coding scheme is intended to be easy to detect and robust against noise, distortion and luminance artifacts.
    The code is designed for use as competitor numbers in club level fell races, as such the number range
    is limited to between 101 and 999 inclusive, i.e. always a 3 digit number and max 899 competitors.
    The actual implementation limit may be less depending on coding constraints chosen.
    The code is circular (so orientation does not matter) and consists of:
        a solid circular centre of 'white' with a radius 2R (a 'blob'),
        surrounded by a solid ring of 'black' and width 1R,
        surrounded by 3 concentric data rings of width R and divided into N equal segments,
        enclosed by a solid 'black' ring of width 1R,
        enclosed by a solid 'white' ring of width 1R. 
    Total radius is 8R.
    The code consists of a 5 digit number in base 6, the lead digit is always 0 and the remaining digits are
    in the range 1..6 (i.e. do not use 0), these 5 digits are repeated 3 times yielding a 15 digit number
    with triple redundancy. The six numbers are mapped across the 3 data rings to produce a 3-bit code. Of
    the 8 possibilities for that code only those that yield a single 'pulse' are used (i.e. the bits are
    encoded such that there is a series of 0 or more leading 0's, then a series of 1's, then 0 or more
    trailing 0's), these 3 components are referred to as the 'lead', 'head' and 'tail' of the pulse.
    A one-bit is white (i.e. high luminance) and a zero-bit is black (i.e. low luminance).
    
    Note: The decoder can cope with several targets in the same image. It uses the relative size of
          the targets to present the results in distance order, nearest (i.e. biggest) first. This is
          important to determine finish order.
          
    This Python implementation is just a proof-of-concept. In particular, it does not represent good
    coding practice, not does it utilise the Python language to its fullest extent.
                
    """

# ToDo: create test background with random noise blobs and random edges
# ToDo: add a blurring option when drawing test targets
# ToDo: add a rotation (vertically and horizontally) option when drawing test targets
# ToDo: add a curving (vertically and horizontally) option when drawing test targets
# ToDo: generate lots of (extreme) test targets using these options
# ToDo: draw the test targets on a relevant scene photo background
# ToDo: try frame grabbing from a movie
# ToDo: in the final product speak the number detected (just need 10 sound snips - spoken 0 to 9)
# ToDo: generate some test images in heavy rain (and snow?)

# colours
MAX_LUMINANCE = 255
MIN_LUMINANCE = 0
MID_LUMINANCE = (MAX_LUMINANCE - MIN_LUMINANCE) >> 1

# alpha channel
TRANSPARENT = MIN_LUMINANCE
OPAQUE = MAX_LUMINANCE

""" WARNING
    cv2's co-ordinates are backwards from our pov, the 'x' co-ordinate is vertical and 'y' horizontal.
    The logic here uses 'x' horizontally and 'y' vertically, swapping as required when dealing with cv2
    """


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


def count_bits(mask):
    """ count the number of one bits in the given integer """
    bits = 0
    while mask != 0:
        if (mask & 1) != 0:
            bits += 1
        mask >>= 1
    return bits


class Codec:
    """ Encode and decode a code-word or a code-block,
        a code-word is a number with specific properties,
        a code-block is an encoding of a code-word with specific properties,
        a code-word is N copies of an N digit number encoded base N as digits 1..N,
        this number is preceded by a 0 digit as a start marker giving N+1 digits in total,
        this sequence is repeated N times in a code-word, giving multiple redundancy with
        each symbol maximally spread out,
        a code-block is drawn as N rings with the digits 0 and 1..N drawn in such a
        way that across the rings they can be interpreted as a single pulse with
        differing ratios between the leading low, the central high and the trailing low lengths,
        this class encapsulates all the encoding and decoding and their constants
        """

    # region constants...
    DIGITS_PER_NUM = 4  # how many digits per encoded number
    COPIES = 3  # number of copies in a code-word (thus 'bits' around a ring is DIGITS_PER_NUM * COPIES), must be odd
    EDGES = 6  # min number of edges per rendered ring (to stop accidental solid rings looking like a blob)
    # the ratio is referring to the across rings pulse shape (lead:head:tail relative lengths)
    # base 6 encoding in 4 rings (yields 520 usable codes)
    ENCODING_4 = [[0, 0, 0, 0],   # ratio 6:-:- digit 0 (the sync digit, must be first)
                  [0, 0, 0, 1],   # ratio 4:1:1 digit 1
                  [0, 0, 1, 0],   # ratio 3:1:2 digit 2
                  #0, 0, 1, 1     # ratio 3:2:1
                  [0, 1, 0, 0],   # ratio 2:1:3 digit 3
                  #0, 1, 0, 1     # ratio 2:1:3 digit 3 (double pulse is truncated to first, so same as 0100)
                  #0, 1, 1, 0     # ratio 2:2:2 digit 5
                  #0, 1, 1, 1     # ratio 2:3:1
                  [1, 0, 0, 0],   # ratio 1:1:4 digit 4
                  #1, 0, 0, 1     # ratio 1:1:4 digit 4 (double pulse is truncated to first, so same as 1000)
                  #1, 0, 1, 0     # ratio 1:1:4 digit 4 (double pulse is truncated to first, so same as 1000)
                  #1, 0, 1, 1     # ratio 3:2:1         (double pulse is truncated to second, so same as 0011)
                  #1, 1, 0, 0     # ratio 1:2:3
                  #1, 1, 0, 1     # ratio 1:2:3         (double pulse is truncated to first, so same as 1100)
                  #1, 1, 1, 0     # ratio 1:3:2
                  [0, 1, 1, 0],   # ratio 2:2:2 digit 5
                  [1, 1, 1, 1]]   # ratio 1:4:1 digit 6
    # encoding as a number (just used as a diagnostic visual aid) - maps digit to ring bit pattern
    BITS_4 = [0, 1, 2, 4, 8, 6, 15]
    # these are the ratios for the above (1:1 index correspondence), each digit's ratios must sum to SPAN
    # its defined here to be close to the encoding table but its only used externally
    RATIOS_4 = [[6, 0, 0],  # digit 0
                [4, 1, 1],  # digit 1
                [3, 1, 2],  # digit 2
                [2, 1, 3],  # digit 3
                [1, 1, 4],  # digit 4
                [2, 2, 2],  # digit 5
                [1, 4, 1]]  # digit 6
    # base 6 encoding in 3 rings (yields 700 usable codes)
    ENCODING_3 = [[0, 0, 0],  # ratio 5:-:- digit 0 (the sync digit, must be first)
                  [0, 0, 1],  # ratio 3:1:1 digit 1
                  [0, 1, 0],  # ratio 2:1:2 digit 2
                  [0, 1, 1],  # ratio 2:2:1 digit 3
                  [1, 0, 0],  # ratio 1:1:3 digit 4
                  #1, 0, 1    # ratio 1:1:3 digit 4 (double pulse is truncated to first, so same as 0100)
                  [1, 1, 0],  # ratio 1:2:2 digit 5
                  [1, 1, 1]]  # ratio 1:3:1 digit 6
    # encoding as a number (just used as a diagnostic visual aid) - maps digit to ring bit pattern
    BITS_3 = [0, 1, 2, 3, 4, 6, 7]
    # these are the ratios for the above (1:1 index correspondence), each digit's ratios must sum to SPAN
    # its defined here to be close to the encoding table but its only used externally
    RATIOS_3 = [[5, 0, 0],  # digit 0  bits 0
                [3, 1, 1],  # digit 1  bits 1
                [2, 1, 2],  # digit 2  bits 2
                [2, 2, 1],  # digit 3  bits 3
                [1, 1, 3],  # digit 4  bits 4
                [1, 2, 2],  # digit 5  bits 6
                [1, 3, 1]]  # digit 6  bits 7
    ENCODING = ENCODING_3
    RATIOS = RATIOS_3
    BITS = BITS_3
    BASE = len(ENCODING) - 1  # number base of each digit (-1 to exclude the '0')
    RINGS = len(ENCODING[0])  # number of data rings to encode each digit in a code-block
    SPAN = RINGS + 2  # total span of the code including its margin black rings
    WORD = DIGITS_PER_NUM + 1  # number of digits in a 'word' (+1 for the '0')
    DIGITS = WORD * COPIES  # number of digits in a full code-word
    DOUBT_LIMIT = int(COPIES / 2)  # we want the majority to agree, so doubt must not exceed half
    # endregion

    def __init__(self, min_num, max_num):
        """ create the valid code set for a number in the range min_num..max_num,
            the min_num cannot be zero,
            we create two arrays - one maps a number to its code and the other maps a code to its number.
            """

        self.code_range = int(math.pow(Codec.BASE, Codec.DIGITS_PER_NUM))

        # params
        self.min_num = max(min_num, 1)  # minimum number we want to be able to encode (0 not allowed)
        self.max_num = max(max_num, self.min_num)  # maximum number we want to be able to encode

        # build code tables
        self.codes = [None for _ in range(self.code_range)]  # set all invalid initially
        self.nums = [None for _ in range(self.max_num + 1)]  # ..
        self.min_edges = Codec.DIGITS  # minimum edges around the ring of any valid code block
        self.max_edges = 0           # maximum edges around the ring of any valid code block
        num = self.min_num - 1  # last number to be encoded (so next is our min)
        for code in range(self.code_range):
            edges = self._allowable(code)
            if edges is not None:
                # meets requirements, give this code to the next number
                num += 1
                if num > self.max_num:
                    # found enough
                    break
                else:
                    self.codes[code] = num  # decode code as num
                    self.nums[num] = code  # encode num as code
                    min_edges = min(edges)
                    max_edges = max(edges)
                    if max_edges > self.max_edges:
                        self.max_edges = max_edges
                    if min_edges < self.min_edges:
                        self.min_edges = min_edges
        if num < self.min_num:
            # nothing meets criteria!
            self.num_limit = None
        else:
            self.num_limit = num

    def encode(self, num):
        """ get the code for the given number, returns None if number not valid """
        if num is None or num > self.max_num or num < self.min_num:
            return None
        return self.nums[num]

    def decode(self, code):
        """ get the number for the given code, if not valid returns None """
        if code is None or code >= self.code_range or code < 0:
            return None
        return self.codes[code]

    def build(self, num):
        """ build the codes needed for the data rings
            returns the N integers required to build a 'target'
            """
        if num is None:
            return None
        code = self.encode(num)
        if code is None:
            return None
        code_word = self._make_code_word(code)
        return self._rings(code_word)

    def unbuild(self, slices):
        """ given an array of N code-word rings with random alignment return the encoded number or None,
            each ring must be given as an array of bit values in bit number order,
            returns the number (or None), the level of doubt and the decoded slice digits (as bit patterns),
            """

        # step 1 - decode slice digits (and bits)
        digits = [[None for _ in range(Codec.COPIES)] for _ in range(Codec.WORD)]
        bits = [[None for _ in range(Codec.COPIES)] for _ in range(Codec.WORD)]
        for digit in range(Codec.DIGITS):
            word = int(digit / Codec.WORD)
            bit = digit % Codec.WORD
            code_slice = [None for _ in range(Codec.RINGS)]
            for slice in range(Codec.RINGS):
                code_slice[slice] = slices[slice][digit]
            for idx, seq in enumerate(Codec.ENCODING):
                if seq == code_slice:
                    digits[bit][word] = idx
                    bits[bit][word] = Codec.BITS[idx]
                    break

        # step 2 - amalgamate digit copies into most likely with a doubt
        merged = [[None, None] for _ in range(Codec.WORD)]  # number and doubt for each digit in a word
        for digit in range(len(digits)):
            # the counts structure contains the digit and a count of how many copies of that digit exist
            counts = [[0, idx] for idx in range(Codec.BASE + 1)]  # +1 for the 0
            for element in digits[digit]:
                if element is None:
                    # no count for this
                    continue
                counts[element][0] += 1
            # pick digit with the biggest count (i.e. first in sorted counts)
            counts.sort(key=lambda c: (c[0], c[1]), reverse=True)
            doubt = Codec.COPIES - counts[0][0]
            # possible doubt values are: 0==all copies the same, 1==1 different, 2+==2+ different
            merged[digit] = (counts[0][1], doubt)

        # step 3 - verify presence of a single 0, calculate overall doubt and build bits to return to caller
        doubt = 0
        zeroes = 0
        bad_doubt = False
        for digit in merged:
            doubt += digit[1]
            if digit[1] > Codec.DOUBT_LIMIT:
                # too much ambiguity
                bad_doubt = True
            if digit[0] == 0:
                zeroes += 1
        if zeroes > 1:
            # too many zeroes - this is a show stopper
            doubt += zeroes * Codec.COPIES
            bad_doubt = True
        elif zeroes < 1:
            # no zero - this is a show stopper
            doubt += Codec.DIGITS_PER_NUM + 1
            bad_doubt = True
        if bad_doubt:
            # too much ambiguity - this is a show stopper
            return None, doubt, bits

        # step 4 - look for the '0' and extract the code
        code = None
        zero_at = None
        for idx, digit in enumerate(merged):
            if digit[0] == 0:
                zero_at = idx
                code = 0
                for _ in range(Codec.DIGITS_PER_NUM):
                    idx = (idx - 1) % Codec.WORD
                    digit = merged[idx]
                    code *= Codec.BASE
                    code += digit[0] - 1  # digit is in range 1..base, we want 0..base-1
        if zero_at is None:
            # this should not be possible due to step 3 loop above
            return None, doubt, bits
        else:
            # re-align the bits so the 0 is first (this is just a visual aid)
            aligned_bits = []
            idx = zero_at
            for _ in range(Codec.DIGITS_PER_NUM + 1):
                aligned_bits.append(bits[idx])
                idx = (idx + 1) % Codec.WORD
            bits = aligned_bits

        # step 5 - lookup number
        number = self.decode(code)
        if number is None:
            if doubt == 0:
                doubt = Codec.DIGITS_PER_NUM * Codec.COPIES

        # that's it
        return number, doubt, bits

    def bits(self, code):
        """ given a code return its bit pattern across the rings """
        digits = self.digits(code)
        bits = []
        for digit in digits:
            bits.append(Codec.BITS[digit])
        return bits

    def digits(self, code):
        """ given a code return the digits for that code """
        partial = [0]  # the 'sync' digit
        for digit in range(Codec.DIGITS_PER_NUM):
            partial.append((code % Codec.BASE) + 1)
            code = int(code / Codec.BASE)
        return partial

    def _rings(self, code_word):
        """ build the data ring codes for the given code-word,
            code_word must be a list of digits in the range 0..6,
            each digit is encoded into 4 bits, one for each ring,
            returns a list of integers representing each ring
            """

        # build code block
        code_block = [0 for _ in range(Codec.RINGS)]
        data = 1 << (Codec.DIGITS - 1)  # start at the msb
        for digit in code_word:
            coding = Codec.ENCODING[digit]
            for ring, bit in enumerate(coding):
                if bit == 1:
                    # want a 1 bit in this position in this ring
                    code_block[ring] += data
            data >>= 1  # move to next bit position

        return code_block

    def _make_code_word(self, code):
        """ make the code-word from the given code,
            a code-word is the given code encoded as base N with a leading 0 and copied N times,
            the encoding is little-endian (LS digit first),
            it is returned as a list of digits
            """
        partial = self.digits(code)
        code_word = []
        for _ in range(Codec.COPIES):
            code_word += partial
        return code_word

    def _allowable(self, candidate):
        """ given a code word candidate return True if its allowable as the basis for a code-word,
            code-words are not allowed if they do not meet our edge requirements around the rings,
            the requirement is that at least one edge must exist for every digit within the rings,
            also that there be at least N edges in every ring and every ring is different,
            returns None if the code word is not allowed or a list of edge counts (one per ring)
            """

        code_word = self._make_code_word(candidate)
        msb = 1 << (Codec.DIGITS - 1)  # 1 in the MSB of a code word
        all_ones = (1 << Codec.DIGITS) - 1  # all 1's in the relevant bits
        # check meets edges requirements around the rings (across all rings must be an edge for each bit)
        rings = self._rings(code_word)
        all_edges = 0
        ring_edges = [0 for _ in range(len(rings))]
        for idx, ring in enumerate(rings):
            mask1 = msb
            mask2 = mask1 >> 1
            while mask1 != 0:
                if mask2 == 0:
                    # doing last bit, wrap to the first
                    mask2 = msb
                if ((ring & mask1 == 0) and (ring & mask2) == 0) or (
                        (ring & mask1 != 0) and (ring & mask2) != 0):
                    # there is no edge here, leave edges mask as is
                    pass
                else:
                    # there is an edge here
                    all_edges |= mask2
                    ring_edges[idx] += 1
                mask1 >>= 1
                mask2 >>= 1
        missing = (all_edges ^ all_ones)
        if missing != 0:
            # does not meet edges requirement around the rings
            return None
        # check meet min edges requirement for each ring
        for edges in ring_edges:
            if edges < Codec.EDGES:
                # does not meet requirement for this ring
                return None
        # # check every ring is different (drops too many for little purpose)
        # data = []
        # for ring in rings:
        #     if ring in data:
        #         # got a dup
        #         return None
        #     data.append(ring)
        # got a goody
        return ring_edges


class Angle:
    """ a fast mapping (i.e. uses lookup tables and not math functions) from angles to co-ordinates
        and co-ordinates to angles for a circle, also for the arc length of an ellipsis
        """

    def __init__(self, scale, radius):
        """ build the lookup tables with the resolution required for a single octant, from this octant
            the entire circle can be calculated by rotation and reflection (see angle() and ratio()),
            scale defines the angle accuracy required, the bigger the more accurate, it must be a +ve integer,
            radius defines the maximum radius that cartToPolar has to cater for
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
            # the index is an angle 0..45, the result is the x,y co-ordinates for a circle of radius 1,
            # angle 0 is considered to be straight up and increase clockwise, the vertical axis is
            # considered to be -Y..0..+Y, and the horizontal -X..0..+X,
            # the lookup table contains 0..45 degrees, other octants are calculated by appropriate x,y
            # reversals and sign reversals
            self.ratios[step][0] = 0.0 + math.sin(math.radians(step * self.step_angle))  # NB: x,y reversed
            self.ratios[step][1] = 0.0 - math.cos(math.radians(step * self.step_angle))  # ..
        # Parameters for ratio() for each octant:
        #   edge angle, offset, 'a' multiplier', reverse x/y, x multiplier, y multiplier
        #                                            #                     -Y
        self.octants = [[45, 0, +1, 0, +1, +1],      # octant 0         \ 7 | 0 /
                        [90, +90, -1, 1, -1, -1],    # octant 1       6  \  |  /  1
                        [135, -90, +1, 1, -1, +1],   # octant 2           \ | /
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

        # generate radius look-up table (r = sqrt(x*x + y*y) for x,y in range 0..255)
        self.max_radius = int(radius + 1.5)
        self.radii = [[0 for _ in range(self.max_radius)] for _ in range(self.max_radius)]
        for x in range(self.max_radius):
            for y in range(self.max_radius):
                self.radii[x][y] = math.sqrt(x * x + y * y)


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
            """" convert x, y to a radius (by Pythagoras) """
            if y < 0:
                y = 0 - y
            if x < 0:
                x = 0 - x
            return self.radii[int(round(x))][int(round(y))]

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

    NUM_RINGS = Codec.RINGS + 5  # total rings in our complete code
    DIGITS = Codec.DIGITS  # how many bits in each ring
    BORDER_WIDTH = 0.5  # border width in units of rings

    def __init__(self, centre_x, centre_y, width, frame, contrast, offset):
        # set constant parameters
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
        radius = width * Ring.NUM_RINGS
        scale = 2 * math.pi * radius
        self.angle_xy = Angle(scale, radius).polarToCart
        self.edge = 360 / Ring.DIGITS  # the angle at which a bit edge occurs (NB: not an int)

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
        if bit is None:
            colour = MID_LUMINANCE
        elif bit == 0:
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

    def _border(self, ring_num):
        """ draw a grey border at the given ring_num,
            this is visual cue to stop people cutting into important parts of the code
            and to stop the blob detector seeing the outer white ring to background as a blob
            """

        self._draw_ring(ring_num, None, Ring.BORDER_WIDTH)

        # radius = ring_num * self.w
        #
        # scale = 2 * math.pi * radius  # step the angle such that 1 pixel per increment
        # interval = int(round(scale) / 16)
        # bit = 0
        # for step in range(int(round(scale))):
        #     a = (step / scale) * 360
        #     x, y = self.angle_xy(a, radius)
        #     x = int(round(x))
        #     y = int(round(y))
        #     if (step % interval) == 0:
        #         bit ^= 1
        #     self._point(x, y, bit)

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
        start_x = 0 - min(ring_num * self.w, self.x)
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
                                self._point(tx, ty, -2)    # draw at min luminance (ie. true black)
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
                self._point(x, y, -2)    # draw at min luminance (ie. true black)

    def code(self, number, rings):
        """ draw the complete code for the given number and code-words
            the code-words must match the number, this function is the
            only one that knows the overall target ring structure
            """

        # draw the bullseye and the inner white/black rings
        self._draw_ring(0.0, -1, 2.0)
        self._draw_ring(2.0, 0, 1.0)
        draw_at = 3.0

        # draw the data rings
        for ring in rings:
            self._draw_ring(draw_at, ring, 1.0)
            draw_at += 1.0

        # draw the outer black and white rings
        self._draw_ring(draw_at, 0, 1.0)
        self._draw_ring(draw_at + 1, -1, 1.0)
        draw_at += 2.0

        if int(draw_at) != draw_at:
            raise Exception('number of target rings is not integral ({})'.format(draw_at))
        draw_at = int(draw_at)

        # safety check
        if draw_at != Ring.NUM_RINGS:
            raise Exception('number of rings exported ({}) is not {}'.format(Ring.NUM_RINGS, draw_at))

        # draw a border
        self._border(draw_at)

        # draw a human readable label
        self._label(draw_at, number)


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

    def unload(self, image_file, suffix=None, folder=None):
        """ unload the frame buffer to a PNG image file
            returns the file name written
            """
        if len(self.buffer.shape) == 2:
            # its a grey scale image, convert to RGBA
            image = self.colourize()
        else:
            # assume its already colour
            image = self.buffer
        filename, _ = os.path.splitext(image_file)
        # make sure the write-to folder exists
        pathlib.Path('{}'.format(self.write_to)).mkdir(parents=True, exist_ok=True)
        if folder is not None:
            # being written into a sub-folder, create that as required
            pathlib.Path('{}/{}'.format(self.write_to, folder)).mkdir(parents=True, exist_ok=True)
            folder = '/{}'.format(folder)
        else:
            folder = ''
        if suffix is not None:
            filename = '{}{}/_{}_{}.png'.format(self.write_to, folder, filename, suffix)
        else:
            filename = '{}{}/{}.png'.format(self.write_to, folder, filename)
        cv2.imwrite(filename, image)
        return filename

    def show(self, title='title'):
        """ show the current buffer """
        cv2.imshow(title, self.buffer)
        cv2.waitKey(0)

    def setpixel(self, x, y, value, bleed):
        """ like putpixel except value is scaled by the intensity of what is there already,
            bleed is a BGR tuple with each in the range 0..1,
            where 0=none (i.e use all of value), 1=full (i.e. use all of existing),
            only valid on a colour image
            """
        if x < 0 or x >= self.max_x or y < 0 or y >= self.max_y:
            return
        was = self.buffer[y, x]
        now_part = (was[0] * bleed[0], was[1] * bleed[1], was[2] * bleed[2])
        val_part = (value[0] * (1 - bleed[0]), value[1] * (1 - bleed[1]), value[2] * (1 - bleed[2]))
        pixel = (now_part[0] + val_part[0], now_part[1] + val_part[1], now_part[2] + val_part[2])
        self.buffer[y, x] = pixel

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

    def downheight(self, source, new_height):
        """ downsize the given image such that its height (y) is at most that given,
            the width is preserved,
            """
        width, height = source.size()
        if height <= new_height:
            # its already small enough
            return source
        target = source.instance()
        target.set(cv2.resize(source.get(), (width, new_height), interpolation=cv2.INTER_LINEAR))
        return target

    def downwidth(self, source, new_width):
        """ downsize the given image such that its width (x) is at most that given,
            the height is preserved,
            """
        width, height = source.size()
        if width <= new_width:
            # its already small enough
            return source
        target = source.instance()
        target.set(cv2.resize(source.get(), (new_width, height), interpolation=cv2.INTER_LINEAR))
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
            xorder=0, yorder=1 will detect horizontal edges,
            xorder=1, yorder=0 will detect vertical edges,
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

    def threshold(self, source, thresh=None, size=None, offset=2):
        """ turn source image into a binary image,
            returns the binarised image,
            if thresh is not None an absolute threshold is applied,
            if size is not None an adaptive threshold with that block size is applied
            when adaptive offset is the 'C' constant subtracted from the mean (see OpenCV)
            otherwise an automatic Otsu threshold is applied
            """
        target = source.instance()
        if thresh is not None:
            _, buffer = cv2.threshold(source.get(), thresh, MAX_LUMINANCE, cv2.THRESH_BINARY)
        elif size is not None:
            buffer = cv2.adaptiveThreshold(source.get(), MAX_LUMINANCE,
                                           cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, size, offset)
        else:
            _, buffer = cv2.threshold(source.get(), MIN_LUMINANCE, MAX_LUMINANCE,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        target.set(buffer)
        return target

    def equalise(self, source):
        """ given a source image perform a histogram equalisation to improve the contrast,
            returns the contrast enhanced image
            """
        target = source.instance()
        target.set(cv2.equalizeHist(source.get()))
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
                bleed = obj["bleed"]
                if points is not None:
                    x = _x(obj["start"])
                    for pt in range(len(points)):
                        y = _y(points[pt])
                        if y is not None:
                            source.setpixel(_x(x + pt), _y(y), colour, bleed)
            elif obj["type"] == self.PLOTY:
                colour = obj["colour"]
                points = obj["points"]
                bleed = obj["bleed"]
                if points is not None:
                    y = _y(obj["start"])
                    for pt in range(len(points)):
                        x = _x(points[pt])
                        if x is not None:
                            source.setpixel(_x(x), _y(y + pt), colour, bleed)
            elif obj["type"] == self.POINTS:
                colour = obj["colour"]
                points = obj["points"]
                bleed = obj["bleed"]
                for pt in points:
                    source.setpixel(_x(pt[0]), _y(pt[1]), colour, bleed)
            elif obj["type"] == self.TEXT:
                image = cv2.putText(image, obj["text"], _int(obj["start"]),
                                    cv2.FONT_HERSHEY_SIMPLEX, obj["size"], obj["colour"], 1, cv2.LINE_AA)
            else:
                raise Exception('Unknown object type {}'.format(obj["type"]))
        source.set(image)
        return source


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
    NUM_RINGS = Ring.NUM_RINGS  # total number of rings in the whole code (ring==cell in height)
    NUM_DATA_RINGS = Codec.RINGS  # how many data rings in our codes
    NUM_SEGMENTS = Codec.DIGITS  # total number of segments in a ring (segment==cell in length)
    DIGITS_PER_NUM = Codec.DIGITS_PER_NUM  # how many digits per encoded number
    COPIES = Codec.COPIES  # number of copies in a code-word

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
    MIN_BLOB_ROUNDNESS = 0.5  # min circularity of a blob we want (default 0.84)
    BLOB_RADIUS_STRETCH = 1.1  # how much to stretch blob radius to ensure always cover everything when projecting
    MIN_CONTRAST = 0.5  # minimum luminance variation of a valid blob projection relative to the mean luminance
    THRESHOLD_SIZE = 2.5  # the fraction of the projected image size to use as the integration area when binarizing
    THRESHOLD_BLACK = 2  # the % below the average luminance in a projected image that is considered to be black
    THRESHOLD_WHITE = 3  # the % above the average luminance in a projected image that is considered to be white
    MIN_EDGE_SAMPLES = 2  # minimum samples in an edge to be considered a valid edge
    MAX_NEIGHBOUR_ANGLE_INNER = 0.4  # ~=22 degrees, tan of the max acceptable angle when joining inner edge fragments
    MAX_NEIGHBOUR_ANGLE_OUTER = 0.6  # ~=31 degrees, tan of the max acceptable angle when joining outer edge fragments
    MAX_NEIGHBOUR_HEIGHT_GAP = 1  # max y jump allowed when following an edge
    MAX_NEIGHBOUR_HEIGHT_GAP_SQUARED = MAX_NEIGHBOUR_HEIGHT_GAP * MAX_NEIGHBOUR_HEIGHT_GAP
    MAX_NEIGHBOUR_LENGTH_JUMP = 10  # max x jump, in pixels, between edge fragments when joining
    MAX_NEIGHBOUR_HEIGHT_JUMP = 3  # max y jump, in pixels, between edge fragments when joining
    MAX_NEIGHBOUR_OVERLAP = 4  # max edge overlap, in pixels, between edge fragments when joining
    MAX_EDGE_GAP_SIZE = 3 / NUM_SEGMENTS  # max gap tolerated between edge fragments (as fraction of image width)
    MAX_EDGE_HEIGHT_JUMP = 2  # max jump in y, in pixels, along an edge before smoothing is triggered
    INNER_OFFSET = 0  # move inner edge by this many pixels (to reduce effect of very narrow black ring)
    OUTER_OFFSET = 0  # move outer edge by this many pixels (to reduce effect of very narrow black ring)
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

    # region Step/Edge types...
    RISING = 'rising'
    FALLING = 'falling'
    # endregion

    # region Ring numbers of flattened image...
    INNER_POINT = 0
    INNER_WHITE = 1
    INNER_BLACK = 2
    DATA_RING_1 = 3
    DATA_RING_2 = 4
    DATA_RING_3 = 5
    OUTER_BLACK = 6
    OUTER_WHITE = 7
    OUTER_LIMIT = 8  # end of target area
    # endregion

    # region Pulse classifications...
    # the RATIOS table is the lead/head/tail relative lengths across a 'span' (inner to outer guard rings)
    # the order in the RATIOS table is the same as DIGITS which is used to look-up the bit sequence
    RATIOS = Codec.RATIOS
    DIGITS = Codec.ENCODING
    SPAN = Codec.SPAN  # total span of the target rings
    # endregion

    # region Pulse size/position 'nudges'...
    # when measuring pulses we try several 'nudge' possibility (+/- 1 on lead, head, tail sizes)
    # and pick the best, this mitigates against low resolution effects where true edges are between
    # pixels or the (noisy/blurry) image is in the process of migrating from one pulse to another
    # a 'nudge' is a lead+head+tail offset to apply when calculating pulse ratios
    # a nudge of [0,0,0] is auto included
    # PULSE_NUDGES = [[-1, 0, +1],  # move head 'up' 1 pixel
    #                 [+1, 0, -1],  # move head 'down' 1 pixel
    #                 [-1, +1, 0],  # make head fatter in up direction
    #                 [0, +1, -1],  # make head fatter in down direction
    #                 [0, -1, +1],  # make head thinner in up direction
    #                 [+1, -1, 0]]  # make head thinner in down direction
    PULSE_NUDGES = [[+1, -1, 0]]  # make head thinner in down direction (assuming slow rising edges)
    # endregion

    # region Segment transition types...
    NONE_TO_LEFT = 'none-left'
    NONE_TO_RIGHT = 'none-right'
    EDGE_TO_LEFT = 'edge-left'
    EDGE_TO_RIGHT = 'edge-right'
    TWO_CHOICES = 'two-choices'
    MATCH_TO_LEFT = 'match-left'
    MATCH_TO_RIGHT = 'match-right'
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
        """ a Step is a description of a luminance change in a radius """

        def __init__(self, where, type, pixel):
            self.where = where  # the y co-ord of the step
            self.type = type  # the step type, rising or falling
            self.pixel = pixel  # the pixel level falling from or rising to

        def __str__(self):
            if self.type == Scan.RISING:
                direction = 'to'
            else:
                direction = 'from'
            return '({} at {} {} {})'.format(self.type, self.where, direction, self.pixel)

    class Edge:
        """ an Edge is a sequence of joined steps """

        def __init__(self, where, type, samples=None):
            self.where = where  # the x co-ord of the start of this edge
            self.type = type  # the type of the edge, falling or rising
            self.samples = samples  # the list of connected y's making up this edge

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
            return '({} at {},{} for {}{})'.format(self.type, self.where, y, len(self.samples), samples)

    class Extent:
        """ an Extent is the inner/outer edge co-ordinates of a projected image """

        def __init__(self, inner, outer, inner_fail=None, outer_fail=None):
            self.inner = inner  # list of y co-ords for the inner edge
            self.outer = outer  # list of y co-ords for the outer edge
            self.inner_fail = inner_fail  # reason if failed to find inner edge or None if OK
            self.outer_fail = outer_fail  # reason if failed to find outer edge or None if OK

    class Pulse:
        """ a Pulse describes the pixels in a radius """

        def __init__(self, start=None, stop=None, lead=0, head=0, tail=None, begin=None, end=None):
            self.start = start  # y co-ord where the pulse starts in the radius
            self.stop = stop  # y co-ord where the pulse ends in the radius
            self.lead = lead  # length of the lead '0' pixels (including the inner black)
            self.head = head  # length of the '1' pixels
            self.tail = tail  # length of the tail '0' pixels (including the outer black), None==a half-pulse
            self.begin = begin  # x-coord of start of the pulse samples
            self.end = end  # x-coord of end of the pulse samples
            self.bits = []  # list of bits and their error in least error order

        def adjust(self, part, delta):
            # apply deltas unless they make us too small
            # the pulse parts are floats,
            # for -ve deltas the floor is used,
            # for +ve deltas the ceiling is used
            # for a 0 delta the raw value is used
            if delta > 0:
                adjusted = math.ceil(part)
            elif delta < 0:
                adjusted = math.floor(part)
            else:
                adjusted = part
            if adjusted < 1:
                # can't go that small
                return None
            else:
                return adjusted

        def ratio(self, lead_delta=0, head_delta=0, tail_delta=0):
            """ return the ratios that represent this pulse,
                the ..._delta parameters are used to adjust the pulse parts in a particular direction,
                this allows for the determination of a ratio with 'jitter' on the pulse edges
                """
            # deal with special cases
            if self.tail is None:
                # this is a half-pulse, ratios have no meaning
                return None
            if self.lead == 0:
                # this should not be possible
                return None
            if self.head == 0 or self.tail == 0:
                # this can only be 5:0:0 (i.e. zero)
                return [Scan.SPAN, 0, 0]
            # apply deltas unless they make us too small
            lead = self.adjust(self.lead, lead_delta)
            if lead is None:
                # can't go that small
                return None
            head = self.adjust(self.head, head_delta)
            if head is None:
                # can't go that small
                return None
            tail = self.adjust(self.tail, tail_delta)
            if tail is None:
                # can't go that small
                return None
            # calculate the ratios
            span = self.lead + self.head + self.tail  # this represents the entire pulse span, it cannot be 0
            span /= Scan.SPAN  # scale so units of result are the same as Scan.RATIOS
            lead /= span
            head /= span
            tail /= span
            return [lead, head, tail]

        def __str__(self):
            bits = ''
            if len(self.bits) > 0:
                for bit in self.bits:
                    bits = '{}, {}'.format(bits, bit)
                bits = ', bits={}'.format(bits[2:])
            ratio = vstr(self.ratio())
            return '(x={}..{}, y={}..{}, lead={}, head={}, tail={}, ratios={}{})'.\
                   format(nstr(self.begin), nstr(self.end), nstr(self.start), nstr(self.stop),
                          nstr(self.lead), nstr(self.head), nstr(self.tail),
                          ratio, bits)

    class Bits:
        """ this encapsulates a bit sequence for a digit and its error """

        def __init__(self, bits, error, samples=1, actual=None, ideal=None):
            self.bits = bits  # the bits across the data rings
            self.errors = error  # the error accumulator for these samples
            self.samples = samples  # how many of these there are before a change
            self.actual = actual  # actual pulse head, top, tail measured
            self.ideal = ideal  # the ideal head, top, tail for these bits

        def extend(self, samples, error):
            """ extend the bits by the given number of samples with the given error """
            self.samples += samples
            self.errors += error

        def error(self):
            """ get the geometric mean error for the bits """
            return self.errors / self.samples

        def format(self, short=False):
            if short or self.actual is None:
                actual = ''
            else:
                actual = ' = actual:{}'.format(vstr(self.actual))
            if short or self.ideal is None:
                ideal = ''
            else:
                ideal = ', ideal:{}'.format(vstr(self.ideal))
            if self.samples > 1:
                samples = '*{}'.format(self.samples)
            else:
                samples = ''
            return '({}{}, {:.2f}{}{})'.format(self.bits, samples, self.error(), actual, ideal)

        def __str__(self):
            return self.format()

    class Segment:
        """ a Segment describes a contiguous, in angle, sequence of Bits and its choices """

        def __init__(self, start, bits, samples=1, error=0, choices=None, ideal=None):
            self.start = start  # the start x of this sequence
            self.bits = bits  # the bit pattern for this sequence
            self.samples = samples  # how many of them we have
            self.errors = error  # the error accumulator for the given samples
            self.choices = choices  # if there are bits choices, these are they
            self.ideal = ideal  # used to calculate the relative size of the segment, <1==small, >1==big

        def size(self):
            """ return the relative size of this segment """
            if self.ideal is None:
                return None
            else:
                return self.samples / self.ideal

        def extend(self, samples, error):
            """ extend the segment by the given number of samples with the given error """
            self.samples += samples
            self.errors += error
            # if self.choices is not None:
            #     for choice in self.choices:
            #         choice.extend(samples, error)

        def replace(self, bits=None, samples=None, error=None):
            """ replace the given properties if they are not None """
            if bits is not None:
                self.bits = bits
            if samples is not None:
                # NB: if samples are replaced, error must be too
                self.samples = samples
            if error is not None:
                self.errors = error

        def error(self):
            """ get the geometric mean error for the segment """
            return self.errors / self.samples

        def __str__(self):
            if self.choices is None:
                choices = ''
            else:
                choices = ''
                for choice in self.choices:
                    choices = '{}, {}'.format(choices, choice.format(short=True))
                choices = ', choices={}'.format(choices[2:])
            if self.ideal is None:
                size = ''
            else:
                size = ', size={:.2f}'.format(self.size())
            return '(at {} bits={}*{}, error={:.2f}{}{})'.\
                   format(self.start, self.bits, self.samples, self.error(), size, choices)

    class Result:
        """ a result is the result of a number decode and its associated error/confidence level """

        def __init__(self, number, digit_doubt, digits, bit_doubt, bit_error, choice, count=1):
            self.number = number  # the code found
            self.digit_doubt = digit_doubt  # sum of error digits (i.e. where not all three copies agree)
            self.digits = digits  # the digit pattern used for the result
            self.bit_doubt = bit_doubt  # sum of choices across bit segments with more than one choice
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

    def __init__(self, code, frame, transform, cells=(8, 4), video_mode=VIDEO_FHD, debug=DEBUG_NONE, log=None):
        """ code is the code instance defining the code structure,
            frame is the frame instance containing the image to be scanned,
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
        self.decoder = code  # class to decode what we find

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
        max_circumference = min(2 * math.pi * max_radius, 3600)  # good enough for 0.1 degree resolution
        angle = Angle(max_circumference, max_radius)
        self.angle_xy = angle.polarToCart

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
        #params.min_roundness = Scan.MIN_BLOB_ROUNDNESS
        blobs, binary = contours.get_targets(self.image.buffer, params=params, logger=logger)
        self.binary = self.image.instance()
        self.binary.set(binary)

        blobs.sort(key=lambda e: e[0])      # just so the processing order is deterministic (helps debugging)

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

    def _project(self, centre_x, centre_y, blob_size) -> Frame:
        """ 'project' a potential target from its circular shape to a rectangle of radius (y) by angle (x),
            limit_radius is used to limit the radius projected
            returns the projection or None if its not worth pursuing further
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
                if x or y are not whole numbers the 1x1 pixel area overlaps its neighbours
                the returned pixel value is the sum of the overlap fractions of its neighbours
                pixel squares, P is the fractional pixel address in its pixel, 1, 2 and 3 are
                its neighbours, dotted area is contribution from neighbours:
                    +-----+-----+
                    |  P  |  1  |
                    |   ..|..   |
                    +-----+-----+
                    |   ..|..   |
                    |  3  |  2  |
                    +-----+-----+
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
        angle_delta = 360 / self.angle_steps
        code = self.original.instance().new(self.angle_steps, limit_radius, MIN_LUMINANCE)
        for radius in range(limit_radius):
            for angle in range(self.angle_steps):
                degrees = angle * angle_delta
                x, y = self.angle_xy(degrees, radius)
                if x is not None:
                    c = get_pixel(x, y)
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

    def _binarize(self, target: Frame, s: float=2, black: float=-1, white: float=None, clean=True, suffix='') -> Frame:
        """ binarize the given projected image,
            s is the fraction of the image size to use as the integration area (square) in pixels,
            t is the % below the average that is considered to be black,
            also 'tidy' it by changing pixel sequences of BWB or WBW sequences to BBB or WWW
            """

        max_x, max_y = target.size()

        def make_binary(image: Frame, s: float=8, black: float=15, white: float=None) -> Frame:
            """ given a greyscale image return a binary image using an adaptive threshold.
                s is the fraction of the image size to use as the integration area (square) in pixels,
                t is the % below the average that is considered to be black,
                anything above this is considered to be white.
                See the adaptive-threshold-algorithm.pdf paper for algorithm details.
                """

            thresholded = contours.make_binary(image.buffer, s, black, white)
            binary: Frame = image.instance()
            binary.set(thresholded)

            if self.save_images:
                self._unload(binary, '03-binary{}'.format(suffix))

            return binary

        buckets = make_binary(target, s, black, white)

        if clean:
            passes = 0
            total_to_black_changes = 0
            total_to_white_changes = 0
            pass_changes = 1
            while pass_changes > 0:
                passes += 1
                pass_changes = 0
                # clean the pixels - BWB or WBW sequences are changed to BBB or WWW
                # pixels wrap in the x direction but not in the y direction
                for x in range(max_x):
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

    def _slices(self, buckets: Frame) -> List[List[Step]]:
        """ detect radial luminance steps in the given binary/tertiary image,
            returns the slice list for the image,
            it is guaranteed that at least one step is created for every x co-ord in the image
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
                    slices[x].append(Scan.Step(y, Scan.FALLING, last_pixel))
                    transitions += 1
                elif pixel > last_pixel:
                    # rising step
                    slices[x].append(Scan.Step(y, Scan.RISING, pixel))
                    transitions += 1
                last_pixel = pixel
            if transitions == 0:
                # this probably means a big pulse has merged with the inner and the outer edge,
                # if the last_pixel is >0 treat as if got a single black pixel at the image edges
                # otherwise create just a falling edge at 0
                if last_pixel == MAX_LUMINANCE:
                    # its all white - not possible?
                    # create a rising step at 0 and a falling step at max_y
                    slices[x].append(Scan.Step(0, Scan.RISING, MAX_LUMINANCE))
                    slices[x].append(Scan.Step(max_y - 1, Scan.FALLING, MAX_LUMINANCE))
                    if self.logging:
                        self._log('slices: {}: all white and no transitions, creating {}, {}'.
                                  format(x, slices[x][-2], slices[x][-1]))
                elif last_pixel == MIN_LUMINANCE:
                    # its all black - not possible?
                    # create a falling step at 0 and a rising step at max_y
                    slices[x].append(Scan.Step(0, Scan.FALLING, MAX_LUMINANCE))
                    slices[x].append(Scan.Step(max_y - 1, Scan.RISING, MAX_LUMINANCE))
                    if self.logging:
                        self._log('slices: {}: all black and no transitions, creating {}, {}'.
                                  format(x, slices[x][-2], slices[x][-1]))
                else:
                    # its all grey - this means all pixels are nearly the same in the integration area
                    # create a rising step at 0 and a falling step at max_y
                    slices[x].append(Scan.Step(0, Scan.RISING, MAX_LUMINANCE))
                    slices[x].append(Scan.Step(max_y - 1, Scan.FALLING, MAX_LUMINANCE))
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

    def _edges(self, slices: List[List[Step]], max_x, max_y) -> ([Edge], [Edge]):
        """ build a list of falling and rising edges of our target,
            returns the falling and rising edges list in length order,
            an 'edge' here is a sequence of connected rising or falling Steps,
            white to not-white is a falling edge,
            not-white to white is a rising edge
            """

        used = [[False for _ in range(max_y)] for _ in range(max_x)]

        def make_candidate(start_x, start_y, edge_type):
            """ make a candidate edge from the step from start_x,
                pixel pairs at x and x +/- 1 and x and x +/- 2 are considered,
                the next y is the closest of those 2 pairs,
                returns an instance of Edge or None
                """

            def get_nearest_y(x, y, step_type):
                """ find the y with the minimum acceptable gap to the given y at x
                    """

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
                        gap = y - step.where
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
                    x = (x + increment) % max_x
                    this_y = get_nearest_y(x, y, edge_type)
                    if this_y is None:
                        # direct neighbour not connected, see if indirect one is
                        # this allows us to skip a slice (it may be noise)
                        next_y = get_nearest_y((x + increment) % max_x, y, edge_type)
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
                # find the edge (==between None to not None and not None to None)
                start_x = None
                end_x = None
                for x in range(max_x):
                    prev_x = (x - 1) % max_x
                    this_y = candidate[x]
                    prev_y = candidate[prev_x]
                    if prev_y is None and this_y is not None:
                        start_x = x
                    elif prev_y is not None and this_y is None:
                        end_x = prev_x
                    if start_x is not None and end_x is not None:
                        # there can only be 1 sequence, so we're done when got both ends
                        break
                # make the Edge instance
                if start_x is None:
                    # this means this edge goes all the way around
                    edge = Scan.Edge(0, edge_type, candidate)
                elif end_x < start_x:
                    # this means the edge wraps
                    edge = Scan.Edge(start_x, edge_type, candidate[start_x:max_x] + candidate[0:end_x+1])
                else:
                    edge = Scan.Edge(start_x, edge_type, candidate[start_x:end_x+1])
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
            self._log('edges: {} falling edges (inner edge candidates)'.format(len(falling_edges)))
            for edge in falling_edges:
                self._log('    {}'.format(edge))
            self._log('edges: {} rising edges (outer edge candidates)'.format(len(rising_edges)))
            for edge in rising_edges:
                self._log('    {}'.format(edge))

        return falling_edges, rising_edges

    def _extent(self, edges, max_x, max_y) -> Extent:
        """ determine the target inner and outer edges,
            there should be a consistent set of falling edges for the inner black ring
            and another set of rising edges for the outer white ring,
            edges that are within a few pixels of each other going right round is what we want,
            returns the inner, outer edges and their fail reasons or None, None if not failed,
            the inner edge is the y where y-1=white and y=black,
            the outer edge is the y where y-1=black and y=white
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

        def compose(edges, direction=None, offset=0):
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

            if offset != 0:
                for x in range(len(smoothed)):
                    smoothed[x] += offset

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

        falling_edges, rising_edges = edges

        # make the inner edge first, this tends to be more reliably detected
        inner, inner_fail = compose(falling_edges, Scan.FALLING, Scan.INNER_OFFSET)
        if self.logging:
            self._log('extent: inner (offset={}) (fail={})'.format(Scan.INNER_OFFSET, inner_fail))
            log_edge(inner)

        # remove rising edges that come before or too close to the inner
        # if self.logging:
        #     header = 'extent: remove outer candidates too close to inner:'
        # for e in range(len(rising_edges)-1, -1, -1):
        #     edge = rising_edges[e]
        #     inside_samples = 0
        #     for dx, y in enumerate(edge.samples):
        #         x = (edge.where + dx) % max_x
        #         inner_y = inner[x]
        #         if inner_y is None:
        #             continue
        #         min_outer_y = int(round(inner_y * Scan.INNER_OUTER_MARGIN))
        #         if y < min_outer_y:
        #             # this is too close, note it
        #             inside_samples += 1
        #     if inside_samples / len(edge.samples) > Scan.MAX_INNER_OVERLAP:
        #         # this is too close to the inner, chuck it
        #         if self.logging:
        #             if header is not None:
        #                 self._log(header)
        #                 header = None
        #             self._log('    {} has {} samples to close to the inner edge, limit is {:.2f}, margin is {:.2f}'.
        #                       format(edge, inside_samples,
        #                              len(edge.samples) * Scan.MAX_INNER_OVERLAP, Scan.INNER_OUTER_MARGIN))
        #         del rising_edges[e]

        if len(rising_edges) == 0:
            # everything is too close!
            outer = [max_y - 1 for _ in range(max_x)]
            outer_fail = 'no outer'
        else:
            # make the outer edge from what is left and smooth it (it typically jumps around)
            outer, outer_fail = compose(rising_edges, Scan.RISING, Scan.OUTER_OFFSET)

        if self.logging:
            self._log('extent: outer (offset={}) (fail={})'.format(Scan.OUTER_OFFSET, outer_fail))
            log_edge(outer)

        return Scan.Extent(inner, outer, inner_fail, outer_fail)

    def _find_segment_edges(self, buckets: Frame, extent: Extent) -> ([int], str):
        """ find the segment edges from an analysis of the given buckets and extent,
            returns an edge list and None if succeeded, or partial list and a reason if failed,
            we look for zero sequences, we know there should be Scan.COPIES of these,
            between the start and end of these zero areas we know there should be Scan.DIGITS_PER_NUM segments,
            we scan between the zeroes looking for significant changes,
            if we find too many edges, the pair with the smallest change are merged,
            if we do not find enough edges, the largest span segment is split,
            using zeroes as sync markers mitigates digit stretching in the x direction which happens due
            to various distortions (perspective, bib crumpling, et al),
            the buckets image is considered to consist of vertical radial slices, one per x,
            a zero is defined as all black pixels between the inner and outer extent,
            a significant change is when the number of black->white and white->black transitions across
            three slices is above a threshold
            """

        max_x, max_y = buckets.size()
        reason = None

        if self.logging:
            header = 'find_segment_edges:'

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

    def _find_pulses(self, edges: [int], buckets: Frame, extent: Extent) -> ([Pulse], [int]):
        """ extract the lead, head and tail lengths from each of the segments defined in edges,
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

    def _identify(self, blob_size):
        """ identify the target in the current image with the given blob size (its radius),
            """

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
        edges = self._edges(slices, max_x, max_y)
        extent = self._extent(edges, max_x, max_y)

        # ToDo: is this a good idea? its removing information
        # make sure the inner and outer guard areas are not white
        for x in range(max_x):
            inner_at = extent.inner[x]
            if inner_at is not None:
                # inner_at is the start of the inner black ring
                for dy in range(Scan.INNER_GUARD):
                    pixel = buckets.getpixel(x, inner_at + dy)
                    if pixel == MAX_LUMINANCE:
                        buckets.putpixel(x, inner_at + dy, MID_LUMINANCE)
            outer_at = extent.outer[x]
            if outer_at is not None:
                # outer_at is the start of the outer white ring
                outer_at -= 1  # move to the end of the outer black ring
                for dy in range(Scan.OUTER_GUARD):
                    pixel = buckets.getpixel(x, outer_at - dy)
                    if pixel == MAX_LUMINANCE:
                        buckets.putpixel(x, outer_at - dy, MID_LUMINANCE)

        if self.save_images:
            plot = self._draw_edges(edges, extent, target)
            self._unload(plot, '05-edges')

        return max_x, max_y, stretch_factor, buckets, slices, extent

    def _measure(self, extent: Extent, stretch_factor):
        """ get a measure of the target size by examining the extent,
            stretch_factor is how much the image height was stretched during projection,
            its used to re-scale the target size such that all are consistent wrt the original image
            """

        max_x = len(extent.inner)
        target_size = 0  # set as the most average of the pulse ends
        for x in range(max_x):
            inner_x = extent.inner[x]
            outer_x = extent.outer[x]
            target_size += (outer_x - inner_x)
        target_size /= max_x
        target_size /= stretch_factor

        if self.logging:
            self._log('measure: target size is {:.2f} (with stretch compensation of {:.2f})'.
                      format(target_size, stretch_factor))

        return target_size

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

    def _find_codes(self) -> ([Target], Frame):
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
            if extent.inner_fail is not None or extent.outer_fail is not None:
                # failed - this means we did not find its inner and/or outer edge
                if self.save_images:
                    # add to reject list for labelling on the original image
                    if extent.inner_fail is not None:
                        reason = extent.inner_fail
                    else:
                        reason = extent.outer_fail
                    rejects.append(Scan.Reject(self.centre_x, self.centre_y, blob_size, blob_size*4, reason))
                continue

            # ToDo: HACK try detecting the vertical segment edges directly
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
                self._log('image {} does not contain any target candidates'.format(self.original.source),
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
        parent = '_{}/'.format(filename)

        # construct the file name
        if centre_x is None:
            centre_x = self.centre_x
        if centre_y is None:
            centre_y = self.centre_y
        if centre_x > 0 and centre_y > 0:
            # use a sub-folder for this blob
            folder = '{}_{}-{:.0f}x{:.0f}y'.format(parent, filename, centre_x, centre_y)
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
        """ make the area outside the inner and outer edges on the given target visible """

        max_x, max_y = target.size()
        inner = extent.inner
        outer = extent.outer

        inner_lines = []
        outer_lines = []
        for x in range(max_x):
            if inner[x] is not None:
                inner_lines.append((x, 0, x, inner[x] - 1))  # inner edge is on the first black, -1 to get to white
            if outer[x] is not None:
                outer_lines.append((x, outer[x], x, max_y - 1))  # outer edge is on first white
        target = self._draw_lines(target, inner_lines, colour=Scan.RED, bleed=bleed)
        target = self._draw_lines(target, outer_lines, colour=Scan.RED, bleed=bleed)

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

    def _draw_edges(self, edges, extent: Extent, target: Frame):
        """ draw the edges and the inner and outer extent on the given target image """

        falling_edges = edges[0]
        rising_edges = edges[1]

        # mark the image area outside the inner and outer extent
        plot = self._draw_extent(extent, target, bleed=0.8)

        # plot falling and rising edges
        falling_points = []
        rising_points = []
        for edge in falling_edges:
            falling_points.append((edge.where, edge.samples))
        for edge in rising_edges:
            rising_points.append((edge.where, edge.samples))
        plot = self._draw_plots(plot, falling_points, colour=Scan.GREEN, bleed=0.7)
        plot = self._draw_plots(plot, rising_points, colour=Scan.BLUE, bleed=0.7)

        return plot
    # endregion


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
        self.transform = Transform()     # our opencv wrapper
        self.angles = None
        self.video_mode = None
        self.contrast = None
        self.offset = None
        self.debug_mode = None
        self.log_folder = log
        self.log_file = None
        self._log('')
        self._log('******************')
        self._log('Rings: {}, Digits: {}'.format(Ring.NUM_RINGS, Codec.DIGITS))

    def __del__(self):
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def encoder(self, min_num, max_num):
        """ create the coder/decoder and set its parameters """
        self.min_num = min_num
        self.codec = Codec(min_num, max_num)
        if self.codec.num_limit is None:
            self.max_num = None
            self._log('Codec: {} bits, available numbers are None!'.format(Codec.BITS))
        else:
            self.max_num = min(max_num, self.codec.num_limit)
            self._log('Codec: {} bits, available numbers are {}..{}, edges per ring: min {}, max {}'.
                      format(Codec.DIGITS, min_num, self.max_num, self.codec.min_edges, self.codec.max_edges))

    def folders(self, read=None, write=None):
        """ create an image frame and set the folders to read input media and write output media """
        self.frame = Frame(read, write)
        self._log('Frame: media in: {}, media out: {}'.format(read, write))

    def options(self, cells=None, mode=None, contrast=None, offset=None, debug=None, log=None):
        """ set processing options, only given options are changed """
        if cells is not None:
            self.cells = cells
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
        self._log('Options: cells {}, video mode {}, contrast {}, offset {}, debug {}, log {}'.
                  format(self.cells, self.video_mode, self.contrast, self.offset, self.debug_mode, self.log_folder))

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
            if self.max_num is not None:
                for n in range(self.min_num, self.max_num + 1):
                    if check(n) is None:
                        bad += 1
                    else:
                        good += 1
            self._log('{} good, {} bad'.format(good, bad))
        except:
            traceback.print_exc()
        self._log('******************')

    def decoding(self):
        """ test build/unbuild symmetry
            black is the luminance level to use for 'black', and white for 'white',
            noise is how much noise to add to the luminance samples created, when
            set luminance samples have a random number between 0 and noise added or
            subtracted, this is intended to test the 'maybe' logic, the grey thresholds
            are set such that the middle 1/3rd of the luminance range is considered 'grey'
            """

        def rotate(word, shift, bits):
            """ rotate the given word by shift right within a length of bits """
            msb = 1 << (bits - 1)
            for bit in range(shift):
                if word & 1 == 1:
                    word >>= 1
                    word += msb
                else:
                    word >>= 1
            return word

        self._log('')
        self._log('******************')
        self._log('Check build/unbuild from {} to {} with random rotations'.format(self.min_num, self.max_num))
        try:
            good = 0
            doubted = 0
            fail = 0
            bad = 0
            if self.max_num is not None:
                for n in range(self.min_num, self.max_num + 1):
                    rings = self.codec.build(n)
                    # do a random rotation to test it gets re-aligned correctly
                    rotation = random.randrange(0, Codec.DIGITS - 1)
                    for ring in range(len(rings)):
                        rings[ring] = rotate(rings[ring], rotation, Codec.DIGITS)
                        # ToDo: add random errors to test doubt feedback
                    samples = [[] for _ in range(len(rings))]
                    for ring in range(len(rings)):
                        word = rings[ring]
                        for bit in range(Codec.DIGITS):
                            samples[ring].insert(0, word & 1)
                            word >>= 1
                    m, doubt, bits = self.codec.unbuild(samples)
                    if m is None:
                        # failed to decode
                        fail += 1
                        self._log('****FAIL: {:03}-->{}, build={}, doubt={}, bits={}, samples={}'.
                                  format(n, m, rings, doubt, bits, samples))
                    elif m != n:
                        # incorrect decode
                        bad += 1
                        self._log('****BAD!: {:03}-->{} , build={}, doubt={}, bits={}, samples={}'.
                                  format(n, m, rings, doubt, bits, samples))
                    elif doubt > 0:
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

    def test_set(self, size, presets=None):
        """ make a set of test codes,
            the test codes consist of the given presets plus the minimum and maximum numbers
            and N random numbers to make the set size up to that given
            """
        if self.max_num is None:
            return []
        if size < 2:
            size = 2
        if self.max_num - self.min_num <= size:
            return [num for num in range(self.min_num, self.max_num + 1)]
        num_set = [self.min_num, self.max_num]
        if presets is not None:
            num_set += presets
        while len(num_set) < size:
            num = random.randrange(self.min_num + 1, self.max_num - 1)
            if num in num_set:
                # don't want a duplicate
                continue
            num_set.append(num)
        num_set.sort()
        return num_set

    def code_words(self, numbers):
        """ test code-word generation with given set (visual) """
        self._log('')
        self._log('******************')
        self._log('Check code-words (visual)')
        frm_bin = '{:0' + str(Codec.DIGITS) + 'b}'
        frm_prefix = '{}(' + frm_bin + ')=('
        suffix = ')'
        try:
            for n in numbers:
                if n is None:
                    # this means a test code pattern is not available
                    continue
                prefix = frm_prefix.format(n, n)
                rings = self.codec.build(n)
                if rings is None:
                    infix = 'None'
                else:
                    infix = ''
                    for ring in rings:
                        bits = frm_bin.format(ring)
                        infix = '{}, {}'.format(infix, bits)
                    infix = infix[2:]  # remove leading comma space
                self._log('{}{}{}'.format(prefix, infix, suffix))
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
            radius = 256
            scale = 360 * 10  # 0.1 degrees
            angle = Angle(scale, radius)
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
                if rerr > 0.7 or aerr > 0.3 or rotation_err is not None:
                    bad += 1
                    self._log(
                        '{:.3f} degrees, {:.3f} radius --> {:.3f}x, {:.3f}y --> {:.3f} degrees, {:.3f} radius: aerr={:.3f}, rerr={:.3f}, rotation={}'.
                        format(a, radius, cx, cy, ca, cr, aerr, rerr, rotation_err))
                else:
                    good += 1
            self._log('{} good, {} bad'.format(good, bad))
        except:
            traceback.print_exc()
        self._log('******************************************')

    def rings(self, folder, width):
        """ draw pulse test rings in given folder (visual) """
        self._log('')
        self._log('******************')
        self._log('Draw pulse test rings (to test scanner edge detection)')
        self.folders(write=folder)
        try:
            x, y = self._make_image_buffer(width)
            ring = Ring(x >> 1, y >> 1, width, self.frame, self.contrast, self.offset)
            bits = Codec.ENCODING
            block = [0 for _ in range(Codec.RINGS)]
            for slice in range(Codec.DIGITS):
                bit = bits[int(slice % len(bits))]
                slice_mask = 1 << slice
                for r in range(Codec.RINGS):
                    if bit[r] == 1:
                        block[r] += slice_mask
            ring.code(000, block)
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
                    x, y = self._make_image_buffer(width)
                    ring = Ring(x >> 1, y >> 1, width, self.frame, self.contrast, self.offset)
                    ring.code(n, rings)
                    self.frame.unload('test-code-{}'.format(n))
        except:
            traceback.print_exc()
        self._log('******************')

    def scan_codes(self, folder):
        """ find all the test codes in the given folder and scan them,
            these are all 'perfect' images saved as drawn, each file name
            is assumed to include the code number of the image as 3 digits
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
            each file name is assumed to include the code number in the image,
            code numbers must be 3 digits, if an image contains more than one code include
            all numbers separated by a non-digit
            """

        filelist = glob.glob('{}/*.jpg'.format(folder))
        filelist.sort()
        for f in filelist:
            f = os.path.basename(f)
            digits = ''.join([s for s in f if s.isdigit()])
            # every 3 digits is assumed to be a code number
            codes = []
            while len(digits) >= 3:
                num = int(digits[0:3])
                if num < self.min_num or num > self.max_num:
                    num = 0
                codes.append(num)
                digits = digits[3:]
            if len(codes) == 0:
                codes = [0]
            self.scan(folder, codes, f)

    def scan(self, folder, numbers, image):
        """ do a scan for the code set in image in the given folder and expect the number given,
            returns an exit code to indicate what happened
            """
        self._log('')
        self._log('******************')
        self._log('Scan image {} for codes {}'.format(image, numbers))
        if not os.path.isfile('{}/{}'.format(folder, image)):
            self._log('Image {} does not exist in {}'.format(image, folder))
            exit_code = self.EXIT_FAILED
        else:
            debug_folder = 'debug_images'
            self.folders(read=folder, write=debug_folder)
            exit_code = self.EXIT_OK  # be optimistic
            scan = None
            try:
                self._remove_debug_images(debug_folder, image)
                self.frame.load(image)
                scan = Scan(self.codec, self.frame, self.transform, self.cells, self.video_mode,
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
                    bits = result.digits
                    expected = None
                    found_num = None
                    for n in range(len(numbers)):
                        if numbers[n] == num:
                            # found another expected number
                            found[n] = True
                            found_num = num
                            code = self.codec.encode(num)
                            expected = 'code={}, bits={}'.format(code, self.codec.bits(code))
                            break
                    analysis.append([found_num, centre_x, centre_y, num, doubt, size, expected, bits])
                # create dummy result for those not found
                for n in range(len(numbers)):
                    if not found[n]:
                        # this one is missing
                        num = numbers[n]
                        code = self.codec.encode(num)
                        if code is None:
                            # not a legal code
                            expected = 'not-valid'
                        else:
                            expected = 'code={}, bits={}'.format(code, self.codec.bits(code))
                        analysis.append([None, 0, 0, numbers[n], 0, 0, expected, None])
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
                        bits = result[7]
                        bits = ', bits {}'.format(bits)
                        if found is not None:
                            if loop != 0:
                                # don't want these in this loop
                                continue
                            # got what we are looking for
                            self._log('Found {} ({}) at {:.0f}x, {:.0f}y size {:.2f}, doubt {:.4f}{}'.
                                      format(num, expected, centre_x, centre_y, size, doubt, bits))
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
                            elif num < 0:
                                num = 0 - num
                                actual_code = self.codec.encode(num)
                                if actual_code is None:
                                    actual_code = 'not-valid'
                                else:
                                    actual_code = 'code={}, bits={}'. \
                                                  format(actual_code, self.codec.bits(actual_code))
                                prefix = '**** AMBIGUOUS **** --> '
                            else:
                                actual_code = self.codec.encode(num)
                                if actual_code is None:
                                    actual_code = 'not-valid'
                                    prefix = ''
                                else:
                                    actual_code = 'code={}, bits={}'.\
                                                  format(actual_code, self.codec.bits(actual_code))
                                    prefix = '**** UNEXPECTED **** ---> '
                            self._log('{}Found {} ({}) at {:.0f}x, {:.0f}y size {:.2f}, doubt {:.4f}{}'.
                                      format(prefix, num, actual_code, centre_x, centre_y, size, doubt, bits))
                            continue
                if len(results) == 0:
                    self._log('**** FOUND NOTHING ****')
            except:
                traceback.print_exc()
                exit_code = self.EXIT_EXCEPTION
            finally:
                if scan is not None:
                    del (scan)  # needed to close log files
        self._log('Scan image {} for codes {}'.format(image, numbers))
        self._log('******************')
        return exit_code

    def _log(self, message):
        """ print a log message and maybe write it to a file too """
        print(message)
        if self.log_folder is not None:
            if self.log_file is None:
                log_file = '{}/test.log'.format(self.log_folder)
                self._remove_file(log_file, silent=True)
                self.log_file = open(log_file, 'w')
            self.log_file.write('{}\n'.format(message))

    def _make_image_buffer(self, width):
        """ make an image buffer suitable for drawing our test images within,
            width is the width to allow for each ring,
            returns the buffer x, y size
            """

        image_width = width * (Ring.NUM_RINGS + 1) * 2  # rings +1 for the border
        self.frame.new(image_width, image_width, MID_LUMINANCE)
        x, y = self.frame.size()

        return x, y

    def _remove_file(self, f, silent=False):
        try:
            os.remove(f)
        except:
            if silent:
                pass
            else:
                traceback.print_exc()
                self._log('Could not remove {}'.format(f))

    def _remove_test_codes(self, folder, pattern):
        """ remove all the test code images with file names containing the given pattern in the given folder
            """
        filelist = glob.glob('{}/*{}*.*'.format(folder, pattern))
        for f in filelist:
            self._remove_file(f)

    def _remove_debug_images(self, folder, filename):
        """ remove all the diagnostic images of the given file name in the given folder,
            diagnostic images are in a folder with that base file name prefixed by '_'
            """
        if os.path.isdir(folder):
            dir_list = os.listdir('{}/'.format(folder))
            filename, _ = os.path.splitext(filename)
            exists = '_{}'.format(filename) in dir_list
            if exists:
                debug_folder = '{}/_{}'.format(folder, filename)
                try:
                    shutil.rmtree(debug_folder)
                except:
                    traceback.print_exc()
                    self._log('could not remove {}'.format(debug_folder))


def verify():
    # parameters
    min_num = 101  # min number we want
    max_num = 999  # max number we want (may not be achievable)
    contrast = 1.0  # reduce dynamic luminance range when drawing to minimise 'bleed' effects
    offset = 0.0  # offset luminance range from the mid-point, -ve=below, +ve=above

    test_codes_folder = 'codes'
    test_media_folder = 'media'
    test_log_folder = 'logs'
    test_ring_width = 30  # this makes a target that fits on A5

    # cell size is critical,
    # going small in length creates edges that are steep vertically, going more takes too long
    # going small in height creates edges that are too small and easily confused with noise
    test_scan_cells = (7, 5)

    # reducing the resolution means targets have to be closer to be detected,
    # increasing it takes longer to process, most modern smartphones can do 4K at 30fps, 2K is good enough
    test_scan_video_mode = Scan.VIDEO_2K

    # test_debug_mode = Scan.DEBUG_IMAGE
    test_debug_mode = Scan.DEBUG_VERBOSE

    try:
        test = None
        # setup test params
        test = Test(log=test_log_folder)
        test.encoder(min_num, max_num)
        test.options(cells=test_scan_cells,
                     mode=test_scan_video_mode,
                     contrast=contrast,
                     offset=offset,
                     debug=test_debug_mode)

        # build a test code set
        test_num_set = test.test_set(20, [111, 222, 333, 444, 555])

        # test.coding()
        # test.decoding()
        # test.circles()
        # test.code_words(test_num_set)
        # test.codes(test_codes_folder, test_num_set, test_ring_width)
        # test.rings(test_codes_folder, test_ring_width)  # must be after test.codes (else it gets deleted)

        # test.scan_codes(test_codes_folder)
        # test.scan_media(test_media_folder)

        # test.scan(test_codes_folder, [000], 'test-code-000.png')
        # test.scan(test_codes_folder, [332], 'test-code-332.png')
        # test.scan(test_codes_folder, [222], 'test-code-222.png')
        # test.scan(test_codes_folder, [555], 'test-code-555.png')
        # test.scan(test_codes_folder, [800], 'test-code-800.png')
        # test.scan(test_codes_folder, [574], 'test-code-574.png')
        # test.scan(test_codes_folder, [371], 'test-code-371.png')
        # test.scan(test_codes_folder, [757], 'test-code-757.png')
        # test.scan(test_codes_folder, [611], 'test-code-611.png')
        # test.scan(test_codes_folder, [620], 'test-code-620.png')
        # test.scan(test_codes_folder, [132], 'test-code-132.png')

        # test.scan(test_media_folder, [301], 'photo-301.jpg')
        # test.scan(test_media_folder, [775, 592, 184, 111, 101, 285, 612, 655, 333, 444], 'photo-775-592-184-111-101-285-612-655-333-444.jpg')
        test.scan(test_media_folder, [332, 222, 555, 800, 574, 371, 757, 611, 620, 132], 'photo-332-222-555-800-574-371-757-611-620-132-mid.jpg')
        # test.scan(test_media_folder, [332, 222, 555, 800, 574, 371, 757, 611, 620, 132], 'photo-332-222-555-800-574-371-757-611-620-132-distant.jpg')
        # test.scan(test_media_folder, [332, 222, 555, 800, 574, 371, 757, 611, 620, 132], 'photo-332-222-555-800-574-371-757-611-620-132.jpg')
        # test.scan(test_media_folder, [332, 222, 555, 800, 574, 371, 757, 611, 620, 132], 'photo-332-222-555-800-574-371-757-611-620-132-near.jpg')

    except:
        traceback.print_exc()

    finally:
        if test is not None:
            del (test)  # needed to close the log file(s)


if __name__ == "__main__":
    verify()
