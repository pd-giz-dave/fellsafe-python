import os
import glob
import pathlib
import shutil

import cv2
import numpy as np
import random
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
          
    This Python implementation is just a proof-of-concept.
                
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


def vstr(vector, fmt='.2f', open='[', close=']'):
    """ given a list of numbers return a string representing them """
    result = ''
    fmt = ',{:'+fmt+'}'
    for pt in vector:
        if pt is None:
            result += ',None'
        else:
            result += fmt.format(pt)
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
    DIGITS_PER_NUM = 4 # how many digits per encoded number
    COPIES = 3  # number of copies in a code-word (thus 'bits' around a ring is DIGITS_PER_NUM * COPIES)
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
    # these are the ratios for the above (1:1 index correspondence), each digit's ratios must sum to SPAN
    # its defined here to be close to the encoding table but its only used externally
    RATIOS_3 = [[5, 0, 0],  # digit 0
                [3, 1, 1],  # digit 1
                [2, 1, 2],  # digit 2
                [2, 2, 1],  # digit 3
                [1, 1, 3],  # digit 4
                [1, 2, 2],  # digit 5
                [1, 3, 1]]  # digit 6
    ENCODING = ENCODING_3
    RATIOS = RATIOS_3
    BASE = len(ENCODING) - 1  # number base of each digit (-1 to exclude the '0')
    RINGS = len(ENCODING[0])  # number of data rings to encode each digit in a code-block
    SPAN = RINGS + 2  # total span of the code including its margin black rings
    WORD = DIGITS_PER_NUM + 1  # number of digits in a 'word' (+1 for the '0')
    DIGITS = WORD * COPIES  # number of digits in a full code-word
    # endregion

    def __init__(self, min_num, max_num):
        """ create the valid code set for a number in the range min_num..max_num,
            we create two arrays - one maps a number to its code and the other maps a code to its number.
            """

        self.code_range = int(math.pow(Codec.BASE, Codec.DIGITS_PER_NUM))

        # params
        self.min_num = min_num  # minimum number we want to be able to encode
        self.max_num = max_num  # maximum number we want to be able to encode

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
            returns the number (or None), the level of doubt and the decoded slice digits,
            """

        # step 1 - decode slice digits
        digits = [[None for _ in range(Codec.COPIES)] for _ in range(Codec.WORD)]
        for digit in range(Codec.DIGITS):
            word = int(digit / Codec.WORD)
            bit = digit % Codec.WORD
            code_slice = [0 for _ in range(Codec.RINGS)]
            for slice in range(Codec.RINGS):
                code_slice[slice] = slices[slice][digit]
            for idx, seq in enumerate(Codec.ENCODING):
                if seq == code_slice:
                    digits[bit][word] = idx
                    break

        # step 2 - amalgamate digit copies into most likely with a doubt
        merged = [[None, None] for _ in range(Codec.WORD)]  # number and doubt for each digit in a word
        for digit in range(len(digits)):
            # the counts structure contains the digit and a count of how many copies of that digit
            # exist or its neighbours, the rendered pulses are such that a stretch or shrink in either
            # the low lead-in or high lead-out (due to image distortion) result in the digit migrating
            # by +1 or -1, so we count the +/- 1 neighbours as well as self, for the digit the count
            # is increased by N', for its neighbours its 1, so N direct hits results in N'*N and a single
            # indirect hit results in 1 (N' is Codec.RINGS, N is Codec.COPIES), increasing a direct hit
            # by N' ensures it will always beat N'-1 indirect hits,
            # the doubt on a digit is then just count minus N**2
            counts = [[0, idx] for idx in range(Codec.BASE + 1)]  # +1 for the 0
            for bit in digits[digit]:
                if bit is None:
                    # no count for this
                    continue
                counts[bit][0] += Codec.RINGS
                if bit == 0 or counts[bit - 1] is None:
                    # no previous for this
                    pass
                else:
                    counts[bit - 1][0] += 1
                if bit == Codec.BASE or counts[bit + 1] is None:
                    # no next for this
                    pass
                else:
                    counts[bit + 1][0] += 1
            # pick digit with the biggest count (i.e. first in sorted counts)
            counts.sort(key=lambda c: c[0], reverse=True)
            merged[digit] = (counts[0][1], Codec.RINGS * Codec.COPIES - counts[0][0])

        # step 3 - look for the '0' and extract the code
        code = None
        doubt = None
        for idx, digit in enumerate(merged):
            if digit[0] == 0:
                for _ in range(Codec.DIGITS_PER_NUM):
                    idx = (idx - 1) % Codec.WORD
                    digit = merged[idx]
                    if code is None:
                        code = 0
                        doubt = 0
                    code *= Codec.BASE
                    code += digit[0] - 1  # digit is in range 1..base, we want 0..base-1
                    doubt += digit[1]
                break
        if doubt is None:
            # this means we did not find a 0 - so the doubt is huge
            doubt = Codec.RINGS * Codec.COPIES * Codec.DIGITS_PER_NUM

        # step 4 - lookup number
        number = self.decode(code)

        # that's it
        return number, doubt, digits

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
        partial = [0]
        for digit in range(Codec.DIGITS_PER_NUM):
            partial.append((code % Codec.BASE) + 1)
            code = int(code / Codec.BASE)
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
        scale = 2 * math.pi * width * Ring.NUM_RINGS
        self.angle_xy = Angle(scale).polarToCart
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
        target.set(cv2.resize(source.get(), (width, new_height), interpolation=cv2.INTER_CUBIC))
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
        #       working the other way also means 'slice processing' is all in the same Y, so cache local and faster
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

    # image 'segment' and 'ring' constraints,
    # a 'segment' is the angular division in a ring,
    # a 'ring' is a radial division,
    # a 'cell' is the intersection of a segment and a ring
    # these constraints set minimums that override the cells() property given to Scan
    MIN_PIXELS_PER_CELL = 4  # min pixels making up a cell length
    MIN_PIXELS_PER_RING = 4  # min pixels making up a ring width

    # region Tuning constants...
    MIN_BLOB_AREA_RADIUS = 2  # min blob radius we want (any smaller and get too many false targets)
    MAX_BLOB_AREA_RADIUS = 80  # max blob radius we want (any bigger means they are too close to the camera!)
    MIN_BLOB_SIZE = 6  # minimum blob size, in pixels, we want - NB: this is the 'diameter' of the detected blob
    MIN_BLOB_SEPARATION = 5  # smallest blob within this distance (in pixels) of each other are dropped
    BLOB_RADIUS_STRETCH = 1.3  # how much to stretch blob radius to ensure always cover everything
    THRESHOLD_START = 1/8  # where to start threshold calculation within image height
    THRESHOLD_END = 6/8  # where to end threshold calculation within image height
    THRESHOLD_END_MIN = 0.8  # minimum threshold end as fraction of min image height (used if above is too small)
    THRESHOLD_RANGE = [-1, +1]  # x offsets to include when calculating the threshold at x
    THRESHOLD_OFFSET = 1.0  # fiddle with threshold for heuristic reasons
    MAX_NEIGHBOUR_ANGLE_TAN = 0.6  # ~=30 degrees, tan of the max acceptable angle when joining edge fragments
    MAX_NEIGHBOUR_HEIGHT_GAP = 1  # max y jump allowed when following an edge
    MAX_NEIGHBOUR_HEIGHT_GAP_SQUARED = MAX_NEIGHBOUR_HEIGHT_GAP * MAX_NEIGHBOUR_HEIGHT_GAP
    MAX_NEIGHBOUR_LENGTH_JUMP = 6  # max x jump, in pixels, between edge fragments when joining
    MAX_NEIGHBOUR_HEIGHT_JUMP = 3  # max y jump, in pixels, between edge fragments when joining
    MAX_NEIGHBOUR_OVERLAP = 4  # max edge overlap, in pixels, between edge fragments when joining
    MAX_EDGE_HEIGHT_JUMP = 2  # max jump in y, in pixels, along an edge before smoothing is triggered
    INNER_OUTER_MARGIN = 1.7  # minimum margin between the inner edge and the outer edge
    MAX_INNER_OVERLAP = 0.2  # max num of samples of outer edge fragment allowed to be inside the inner edge
    INNER_OFFSET = -1  # move inner edge by this many pixels (to reduce effect of very narrow black ring)
    OUTER_OFFSET = +1  # move outer edge by this many pixels (to reduce effect of very narrow black ring)
    MIN_PULSE_LEAD = 2  # minimum pixels for a valid pulse lead (or tail) period, pulse ignored if less than this
    MIN_PULSE_HEAD = 2  # minimum pixels for a valid pulse head period, pulse ignored if less than this
    MIN_SEGMENT_LENGTH = 0.3  # min (relative) segment length, shorter segments are merged/dropped
    MAX_SEGMENT_LENGTH = 1.3  # max (relative) segment length, longer segments are split
    MIN_0_SEGMENT_LENGTH = 0.2  # 000's segments shorter than this are candidates for merging/dropping
    DOMINANT_SEGMENT_RATIO = 3  # a segment this much bigger than its neighbour dominates it so its properties prevail
    RATIO_QUANTA = 99  # number of quanta in a ratio error, errors are in the range 1..RATIO_QUANTA+1
    NO_CHOICE_ERROR = 1  # a choice error of this is no error, i.e. a dead cert
    MAX_CHOICE_ERR_DIFF = int(0.1 * RATIO_QUANTA)  # 2nd choices ignored if err diff more than this
    MAX_CHOICE_ERR_DIFF_SQUARED = MAX_CHOICE_ERR_DIFF * MAX_CHOICE_ERR_DIFF
    MAX_BIT_CHOICES = 1024  # max bit choices to explore when decoding bits to the associated number
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

    # region Pulse classifications...
    # for the RATIOS table, the head length==1, the table entry is the corresponding lead/tail length
    # the order in the RATIOS table is the same as DIGITS which is used to look-up the bit sequence
    RATIOS = Codec.RATIOS
    DIGITS = Codec.ENCODING
    SPAN = Codec.SPAN  # total span of the target rings
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

        def __init__(self, where, type):
            self.where = where  # the y co-ord of the step
            self.type = type  # the step type, rising or falling

        def __str__(self):
            return '({} at {})'.format(self.type, self.where)

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

    class Pulse:
        """ a Pulse describes the pixels in a radius """

        def __init__(self, start, lead=0, head=0, tail=None):
            self.start = start  # y co-ord where the pulse starts in the radius
            self.lead = lead  # length of the lead '0' pixels (including the inner black)
            self.head = head  # length of the '1' pixels
            self.tail = tail  # length of the tail '0' pixels (including the outer black), None==a half-pulse
            self.bits = []  # list of bits and their error in least error order

        def ratio(self, lead_delta=0, head_delta=0, tail_delta=0):
            """ return the ratios that represent this pulse,
                the ..._delta parameters added to the relevant component first,
                this allows for the determination of a ratio with 'jitter' on the pulse edges
                """
            # deal with special cases
            if self.tail is None:
                # this is a half-pulse, ratios have no meaning
                return []
            if self.lead == 0:
                # this should not be possible
                return []
            if self.head == 0 or self.tail == 0:
                # this can only be 5:0:0
                return [Scan.SPAN, 0, 0]
            # apply deltas unless they make us too small
            lead = self.lead + lead_delta
            if lead < 2:
                lead -= lead_delta
                lead_delta = 0
            head = self.head + head_delta
            if head < 2:
                head -= head_delta
                head_delta = 0
            tail = self.tail + tail_delta
            if tail < 2:
                tail -= tail_delta
                tail_delta = 0
            # calculate the ratios
            span = self.lead + self.head + self.tail  # this represents the entire pulse span, it cannot be 0
            span += lead_delta + head_delta + tail_delta
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
            return '(start={}, lead={}, head={}, tail={}, ratios={}{})'.\
                   format(self.start, self.lead, self.head, self.tail, ratio, bits)

    class Bits:
        """ this encapsulates a bit sequence for a digit and its error,
            NB: error() returns a harmonic mean calculated as samples/errors
            where 'errors' is an accumulator of samples/error-value,
            the relations error==samples/errors and errors==samples/error always hold
            (apart from quantisation issues)
            """

        def __init__(self, bits, error, samples=1, actual=None, ideal=None):
            self.bits = bits  # the bits across the data rings
            self.errors = samples / error  # the harmonic mean error accumulator
            self.samples = samples  # how many of these there are before a change
            self.actual = actual  # actual pulse head, top, tail measured
            self.ideal = ideal  # the ideal head, top, tail for these bits

        def extend(self, samples, error):
            """ extend the bits by the given number of samples with the given error """
            self.samples += samples
            self.errors += samples / error  # the error given is assumed to be the inverse of some measure

        def error(self):
            """ get the harmonic mean error for the bits """
            if self.errors > 0:
                return int(round(self.samples / self.errors))  # get harmonic mean as the error
            else:
                return 0

        def format(self, short=False):
            if short or self.actual is None:
                actual = ''
            else:
                actual = ' = actual:{}'.format(vstr(self.actual))
            if short or self.ideal is None:
                ideal = ''
            else:
                ideal = ', ideal:{}'.format(vstr(self.ideal))
            return '({}*{}, {:.2f}{}{})'.format(self.bits, self.samples, self.error(), actual, ideal)

        def __str__(self):
            return self.format()

    class Segment:
        """ a Segment describes a contiguous, in angle, sequence of Pulses,
            NB: error() returns a harmonic mean calculated as samples/errors
            where 'errors' is an accumulator of samples/error-value,
            the relations error==samples/errors and errors==samples/error always hold
            (apart from quantisation issues)
            """

        def __init__(self, start, bits, samples=1, error=1, choices=None, ideal=None):
            self.start = start  # the start x of this sequence
            self.bits = bits  # the bit pattern for this sequence
            self.samples = samples  # how many of them we have
            self.errors = samples / error  # the harmonic mean error accumulator
            self.choices = choices  # if there are bits choices, these are they
            self.ideal = ideal  # used to calculate the relative size of the segment, <1==small, >1==big

        def size(self):
            """ return the relative size of this segment """
            if self.ideal is None:
                return None
            else:
                return self.samples / self.ideal

        def extend(self, samples, error):
            """ extend the segment and its choices by the given number of samples with the given error """
            self.samples += samples
            self.errors += samples / error
            if self.choices is not None:
                for choice in self.choices:
                    choice.extend(samples, error)

        def replace(self, bits=None, samples=None, error=None):
            """ replace the given properties if they are not None """
            if bits is not None:
                self.bits = bits
            if samples is not None:
                # NB: if samples are replaced, error must be too
                self.samples = samples
            if error is not None:
                self.errors = self.samples / error

        def error(self):
            """ get the harmonic mean error for the segment """
            if self.errors > 0:
                err = max(int(round(self.samples / self.errors)), 1)  # get harmonic mean as the error
            else:
                err = 0
            return err

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

    class Target:
        """ structure to hold detected target information """

        def __init__(self, centre_x, centre_y, blob_size, target_size, image, result):
            self.centre_x = centre_x  # x co-ord of target in original image
            self.centre_y = centre_y  # y co-ord of target in original image
            self.blob_size = blob_size  # blob size originally detected by the blob detector
            self.target_size = target_size  # target size scaled to the original image (==outer edge average Y)
            self.image = image  # the image of the target(s)
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

        # set warped image width
        self.angle_steps = Scan.NUM_SEGMENTS * max(self.cells[0], Scan.MIN_PIXELS_PER_CELL)

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

    def __del__(self):
        """ close our log file """
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _blobs(self):
        """ find the target blobs in our image,
            this must be the first function called to process our image,
            creates a blob list each of which is a 'keypoint' tuple of:
                .pt[1] = x co-ord of the centre
                .pt[0] = y co-ord of centre
                .size = diameter of blob
            returns a list of unique blobs found
            """

        # prepare image
        blurred = self.transform.blur(self.original)  # de-noise
        self.image = self.transform.downsize(blurred, self.video_mode)  # re-size to given video mode

        # set filter parameters
        threshold = (MIN_LUMINANCE, MAX_LUMINANCE, 8)  # min, max luminance, luminance step
        circularity = (0.75, None)  # min, max 'corners' in blob edge or None
        convexity = (0.5, None)  # min, max 'gaps' in blob edge or None
        inertia = (0.4, None)  # min, max 'squashed-ness' or None
        min_area = 2 * math.pi * (Scan.MIN_BLOB_AREA_RADIUS * Scan.MIN_BLOB_AREA_RADIUS)
        max_area = 2 * math.pi * (Scan.MAX_BLOB_AREA_RADIUS * Scan.MAX_BLOB_AREA_RADIUS)
        area = (min_area, max_area)  # min, max area in pixels, or None
        gaps = (None, None)  # how close blobs have to be to be merged and min number
        colour = MAX_LUMINANCE  # we want bright blobs, use MIN_LUMINANCE for dark blobs

        # find the blobs
        blobs = self.transform.blobs(self.image, threshold, circularity, convexity, inertia, area, gaps, colour)
        blobs = list(blobs)

        # filter out blobs that are too small (opencv min area does not translate to blob diameter)
        small_blobs = []
        dropped = True
        while dropped:
            dropped = False
            for blob in range(len(blobs)):
                if blobs[blob].size < Scan.MIN_BLOB_SIZE:
                    # too small
                    small_blobs.append(blobs.pop(blob))
                    dropped = True
                    break

        # filter out blobs that are co-incident (opencv can get over enthusiastic!)
        dup_blobs = []
        dropped = True
        min_separation = Scan.MIN_BLOB_SEPARATION * Scan.MIN_BLOB_SEPARATION   # square it so sign irrelevant
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
                    gap = self._get_gap((x1, y1), (x2, y2))
                    if gap < min_separation:
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

        blobs.sort(key=lambda e: e.pt[0])      # just so the processing order is deterministic (helps debugging)

        if self.logging:
            self._log('blob-detect: found {} blobs'.format(len(blobs) + len(dup_blobs)), 0, 0)
            for blob in blobs:
                self._log('    centred at {}x{}y, size {:.2f}  (kept)'.
                          format(int(round(blob.pt[0])), int(round(blob.pt[1])), blob.size))
            for blob in small_blobs:
                self._log('    centred at {}x{}y, size {:.2f}  (dropped as too small)'.
                          format(int(round(blob.pt[0])), int(round(blob.pt[1])), blob.size))
            for blob in dup_blobs:
                self._log('    centred at {}x{}y, size {:.2f}  (dropped as a duplicate)'.
                          format(int(round(blob.pt[0])), int(round(blob.pt[1])), blob.size))

        if self.save_images:
            grid = self.image
            if len(blobs) > 0:
                grid = self.transform.drawKeypoints(grid, blobs, Scan.GREEN)
            if len(dup_blobs) > 0:
                grid = self.transform.drawKeypoints(grid, dup_blobs, Scan.BLUE)
            if len(small_blobs) > 0:
                grid = self.transform.drawKeypoints(grid, small_blobs, Scan.RED)
            self._unload(grid, 'blobs', 0, 0)

        return blobs

    def _radius(self, centre_x, centre_y, blob_size):
        """ determine the image radius to extract around the given blob position and size
            blob_size is used as a guide to limit the radius projected,
            we assume the blob-size is (roughly) the diameter of the inner two white rings
            but err on the side of going too big
            """

        max_x, max_y = self.image.size()
        edge_top = centre_y
        edge_left = centre_x
        edge_bottom = max_y - centre_y
        edge_right = max_x - centre_x
        ring_width = blob_size / 4
        limit_radius = int(round(max(min(edge_top, edge_bottom, edge_left, edge_right), 1)))
        blob_radius = int(ring_width * Scan.NUM_RINGS * Scan.BLOB_RADIUS_STRETCH)
        if blob_radius < limit_radius:
            # max possible size is less than the image edge, so use the blob size
            limit_radius = blob_radius

        if self.logging:
            self._log('radius: limit radius {}'.format(limit_radius))

        if self.save_images:
            # draw cropped source image of just this blob
            start_x = max(int(centre_x - limit_radius), 0)
            end_x = min(int(centre_x + limit_radius), max_x)
            start_y = max(int(centre_y - limit_radius), 0)
            end_y = min(int(centre_y + limit_radius), max_y)
            blob = self.transform.crop(self.image, start_x, start_y, end_x, end_y)
            # mark the blob and its co-ords from the original image
            # draw the detected blob in blue and its co-ords and size
            k = (limit_radius, limit_radius, blob_size / 2)
            blob = self.transform.label(blob, k, Scan.RED, '{:.1f} ({:.0f}x{:.0f}y)'.
                                        format(blob_size, centre_x, centre_y))
            self._unload(blob, '01-target', centre_x, centre_y)

        return limit_radius

    def _project(self, centre_x, centre_y, limit_radius) -> Frame:
        """ 'project' a potential target at centre_x/y from its circular shape to a rectangle
            of radius (y) by angle (x), limit_radius is for the warp,
            returns the projected image
            """

        image_height = limit_radius  # one pixel per radius
        image_width = self.angle_steps

        # do the projection
        code = self.transform.warpPolar(self.image, centre_x, centre_y, limit_radius, image_width, image_height)

        max_x, max_y = code.size()
        min_y = int(round(Scan.MIN_PIXELS_PER_RING * Scan.NUM_RINGS))
        if max_y < min_y:
            # increase image height to meet our min pixel per ring requirement
            # ToDo: do DIY by power of 2 and replicate (darkest) pixels rather than interpolate?
            code = self.transform.upheight(code, min_y)
            stretch_factor = min_y / max_y
        else:
            stretch_factor = 1

        if self.logging:
            self._log('project: projected image size {}x {}y (stretch factor {:.2f})'.
                      format(max_x, max_y, stretch_factor))

        if self.save_images:
            # draw projected image
            self._unload(code, '02-projected')

        return code, stretch_factor

    def _threshold(self, target: Frame) -> Frame:
        """ threshold the given image into binary """

        max_x, max_y = target.size()

        def make_binary(range_x, range_y):
            """ make a binary image from a threshold constructed across the given ranges,
                range_x is a series of offsets on the x being considered, the last must be 0,
                range_y is the image portion, starting at the top, to include in the threshold
                """

            # make an empty (i.e. black) image
            buckets: Frame = target.instance()
            buckets.new(max_x, max_y, MIN_LUMINANCE)

            # build the binary image
            for x in range(max_x):
                # get the pixels
                grey = 0
                samples = 0
                for dx in range_x:  # NB: the last element must be 0
                    slice_pixels = [None for _ in range(max_y)]  # pixels of our slice
                    for y in range(max_y):
                        pixel = target.getpixel((x + dx) % max_x, y)
                        slice_pixels[y] = pixel
                    # set threshold as the harmonic mean of our pixel slice
                    for y in range_y:
                        grey += 1 / max(slice_pixels[y], 1)
                        samples += 1
                grey = samples / grey
                grey *= Scan.THRESHOLD_OFFSET  # bias to consider more as black
                # 2 levels: black, white
                for y in range(max_y):
                    if slice_pixels[y] > grey:
                        pixel = MAX_LUMINANCE
                    else:
                        pixel = MIN_LUMINANCE
                    buckets.putpixel(x, y, pixel)

            return buckets

        # ToDo: do threshold span on slice-by-slice basis and not simple y-span for all
        start_y = int(max_y * Scan.THRESHOLD_START)
        span_y = max_y * Scan.THRESHOLD_END
        min_span = int(Scan.MIN_PIXELS_PER_RING * Scan.NUM_RINGS * Scan.THRESHOLD_END_MIN)
        if span_y < min_span:
            span_y = min_span
        span_y = min(int(round(span_y)), max_y - 1)
        range_y = range(start_y, span_y + 1)
        range_x = Scan.THRESHOLD_RANGE + [0]  # last element must be 0
        buckets = make_binary(range_x, range_y)

        # clean the pixels - BWB or WBW sequences are changed to BBB or WWW
        for x in range(max_x):
            for y in range(max_y):
                left = buckets.getpixel((x - 1) % max_x, y)
                this = buckets.getpixel(x, y)
                right = buckets.getpixel((x + 1) % max_x, y)
                if left == right and this != left:
                    buckets.putpixel(x, y, left)

        if self.logging:
            self._log('threshold: y span is (0, {}) of (0, {})'.format(span_y, max_y))

        if self.save_images:
            threshold_lines = []
            for x in range(max_x):
                threshold_lines.append((x, start_y, x, span_y))
            plot = self._draw_lines(target, threshold_lines, colour=Scan.GREEN, bleed=0.8)
            self._unload(plot, '03-threshold')
            self._unload(buckets, '04-buckets')

        return buckets

    def _slices(self, buckets: Frame) -> List[List[Step]]:
        """ detect radial luminance steps in the given binary image,
            returns the slice list for the image
            """

        max_x, max_y = buckets.size()

        # build list of transitions
        slices = [[] for _ in range(max_x)]
        for x in range(max_x):
            last_pixel = buckets.getpixel(x, 0)
            for y in range(1, max_y):
                pixel = buckets.getpixel(x, y)
                if pixel < last_pixel:
                    # falling step
                    slices[x].append(Scan.Step(y, Scan.FALLING))
                elif pixel > last_pixel:
                    # rising step
                    slices[x].append(Scan.Step(y, Scan.RISING))
                last_pixel = pixel

        if self.logging:
            self._log('slices: {}'.format(len(slices)))
            for x, slice in enumerate(slices):
                steps = ''
                for step in slice:
                    steps += ', {}'.format(step)
                steps = steps[2:]
                self._log('    {}: {}'.format(x, steps))

        return slices

    def _transitions(self, slices: List[List[Step]]) -> List[List[Pulse]]:
        """ get list of transitions from the given slices,
            a transition is a 'half-pulse', i.e. the lead and head parts only of a full-pulse
            """

        max_x = len(slices)

        if self.logging:
            header = 'transitions: building half-pulses:'
        half_pulses = [None for _ in range(max_x)]
        for x, slice in enumerate(slices):
            if slice is None:
                continue
            # build half-pulse list for this slice
            slice_pulses = []
            pulse = None
            for idx, step in enumerate(slice):
                if step.type == Scan.RISING:
                    if pulse is None:
                        # ignore rising before first falling
                        if self.logging:
                            if header is not None:
                                self._log(header)
                            header = None
                            self._log('    {} ignoring early rising step {} (not falling step yet)'.format(x, step))
                    else:
                        # end of 0 run, start of 1
                        pulse.lead = step.where - pulse.start
                else:  # step.type == Scan.FALLING
                    if pulse is None:
                        # start of initial lead, no previous end for this case
                        pass
                    else:
                        # end of half-pulse
                        pulse.head = step.where - pulse.start - pulse.lead
                    # start a new half-pulse
                    pulse = Scan.Pulse(step.where)
                    slice_pulses.append(pulse)
            if len(slice_pulses) == 0:
                # nothing here at all
                slice_pulses = None
            else:
                if slice_pulses[-1].lead == 0:
                    # got a falling step with no rising step,
                    # dump this pulse and remove the head of predecessor
                    del slice_pulses[-1]
                    if len(slice_pulses) > 0:
                        # no predecessor so this is really the outer edge
                        slice_pulses[-1].head = 0
                if len(slice_pulses) == 0:
                    # nothing left here
                    slice_pulses = None
            half_pulses[x] = slice_pulses

        return half_pulses

    def _edges(self, slices: List[List[Step]], max_x, max_y):
        """ build a list of falling and rising edges of our target,
            returns the falling and rising edges list in length order,
            an 'edge' here is a sequence of connected rising or falling Steps
            """

        used = [[False for _ in range(max_y)] for _ in range(max_x)]

        def make_candidate(start_x, start_y, edge_type):
            """ make a candidate edge from transitions from start_x/y,
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

            if samples > 0:
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
            if slice is None:
                continue
            for step in slice:
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

    def _extent(self, edges, max_x, max_y):
        """ determine the target inner and outer edges,
            there should be a consistent set of falling edges for the inner black ring
            and another set of rising edges for the outer white ring,
            edges that are within a few pixels of each other going right round is what we want
            """

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

        def angle_OK(this_xy, that_xy):
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
            if angle > Scan.MAX_NEIGHBOUR_ANGLE_TAN:
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

        def can_merge(full_edge, edge, max_distance):
            """ check if edge can be merged into full_edge,
                full_edge is a list of y's or None's for every x, edge is the candidate to check,
                max_distance is the distance beyond which a merge is not allowed,
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
            if angle_OK(nearest_back_xy, our_start_xy) and distance_OK(nearest_back_xy, our_start_xy, max_distance):
                # angle and distance OK from our start to end of full
                return True
            if angle_OK(our_end_xy, nearest_next_xy) and distance_OK(our_end_xy, nearest_next_xy, max_distance):
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
                a 'nipple' is a y value with both its neighbours having the same but different y value,
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
            """ extrapolate across the gaps in the given edge """

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

            # we need to start from a known position, so find the first non gap
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
                        continue
                    if gap_start is not None:
                        # we're coming out of a gap, extrapolate across it
                        edge = fill_gap(edge, gap_size, gap_start, x, start_y, y)
                    # no longer in a gap
                    gap_start = None
                    gap_size = 0
                    start_y = y

            return edge

        def compose(edges, direction=None, offset=0):
            """ we attempt to compose a complete edge by starting at every edge, then adding
                near neighbours until no more merge, then pick the longest and extrapolate
                across any remaining gaps
                """

            distance_span = range(2, Scan.MAX_NEIGHBOUR_LENGTH_JUMP + 1)

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
                            if can_merge(trimmed[0], trimmed[1], distance):
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

            # extrapolate across any remaining gaps in the longest edge
            composed = extrapolate(full_edges[0][1])

            # remove y 'steps'
            smoothed = smooth(composed, direction)

            if offset != 0:
                for x in range(len(smoothed)):
                    smoothed[x] += offset

            return smoothed

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

        falling_edges, rising_edges = edges

        # make the inner edge first, this tends to be more reliably detected
        inner = compose(falling_edges, Scan.FALLING, Scan.INNER_OFFSET)  # smoothing the inner edge is not so critical
        if self.logging:
            self._log('extent: inner (offset={})'.format(Scan.INNER_OFFSET))
            log_edge(inner)

        # remove rising edges that come before or too close to the inner
        if self.logging:
            header = 'extent: remove outer candidates too close to inner:'
        for e in range(len(rising_edges)-1, -1, -1):
            edge = rising_edges[e]
            inside_samples = 0
            for dx, y in enumerate(edge.samples):
                x = (edge.where + dx) % max_x
                inner_y = inner[x]
                min_outer_y = int(round(inner_y * Scan.INNER_OUTER_MARGIN))
                if y < min_outer_y:
                    # this is too close, note it
                    inside_samples += 1
            if inside_samples / len(edge.samples) > Scan.MAX_INNER_OVERLAP:
                # this is too close to the inner, chuck it
                if self.logging:
                    if header is not None:
                        self._log(header)
                        header = None
                    self._log('    {} has {} samples of {} inside the inner edge, limit is {}'.
                              format(edge, inside_samples, len(edge.samples),
                                     int(len(edge.samples) * Scan.MAX_INNER_OVERLAP)))
                del rising_edges[e]

        if len(rising_edges) == 0:
            # everything is too close!
            outer = [max_y - 1 for _ in range(max_x)]
        else:
            # make the outer edge from what is left and smooth it (it typically jumps around)
            outer = compose(rising_edges, Scan.RISING, Scan.OUTER_OFFSET)  # smoothing the outer edge is important

        if self.logging:
            self._log('extent: outer (offset={})'.format(Scan.OUTER_OFFSET))
            log_edge(outer)

        return inner, outer

    def _constrain(self, half_pulses: List[List[Pulse]], edges) -> List[List[Pulse]]:
        """ remove half-pulses that are not within the given (inner and outer) edges,
            in an ideal world all half-pulses will have their lead start on the inner edge,
            or have their head start on the outer edge or between the two
            """

        inner, outer = edges

        def adjust(x, pulse):
            """ adjust pulse if its parts are within the inner/outer offset """
            inner_min = min(inner[x], inner[x] - Scan.INNER_OFFSET)
            inner_max = max(inner[x], inner[x] - Scan.INNER_OFFSET)
            if inner_min <= pulse.start <= inner_max:
                # start is in the offset region, move to the inner and adjust the lead
                change = pulse.start - inner[x]
                pulse.start = inner[x]
                pulse.lead += change
            pulse_tail_start = pulse.start + pulse.lead + pulse.head
            if inner_min <= pulse_tail_start <= inner_max:
                # whole thing is inside inner
                change = pulse_tail_start - inner[x]
                if pulse.head > change:
                    pulse.head -= change
                elif pulse.lead > change:
                    pulse.lead -= change
                else:
                    pulse.start -= change
            pulse_head_start = pulse.start + pulse.lead
            outer_min = min(outer[x], outer[x] - Scan.OUTER_OFFSET)
            outer_max = max(outer[x], outer[x] - Scan.OUTER_OFFSET)
            if outer_min <= pulse_head_start <= outer_max:
                # head start is in the offset region, move to the outer and adjust the tail
                change = pulse_head_start - outer[x]
                pulse.lead -= change
                if pulse.tail is not None and pulse.tail > 0:
                    pulse.tail -= change
            return pulse

        # NB: we may have moved the inner/outer edges by an offset, anything that has only changed
        #     by that is just silently snapped to the new inner/outer
        if self.logging:
            header = 'constrain: dropping/adjusting half-pulses outside/spanning inner/outer edges'
        pulse_count = 0
        for x, slice_pulses in enumerate(half_pulses):
            if slice_pulses is None:
                continue
            good_pulses = []
            for idx, pulse in enumerate(slice_pulses):
                pulse = adjust(x, pulse)
                if pulse.start < inner[x]:
                    if (pulse.start + pulse.lead + pulse.head) <= inner[x]:
                        # its wholly inside, drop it
                        if self.logging:
                            if header is not None:
                                self._log(header)
                                header = None
                            self._log('    {}: {} wholly inside inner of {} (dropping)'.format(x, pulse, inner[x]))
                        continue
                    elif (pulse.start + pulse.lead) <= inner[x]:
                        # head starts at inner, move it to +1
                        old_start = pulse.start + pulse.lead
                        new_start = inner[x] + 1
                        new_head = pulse.head - (new_start - old_start)
                        if self.logging:
                            if header is not None:
                                self._log(header)
                                header = None
                            self._log('    {}: {} lead finishes at {} inside inner of {}, '
                                      'moving to {} (changing head from {} to {})'.
                                      format(x, pulse, old_start, inner[x], new_start, pulse.head, new_head))
                        pulse.start = inner[x]
                        pulse.lead = 1
                        pulse.head = new_head
                    else:
                        # runs past the inner, move to start at the inner
                        new_lead = pulse.lead - (inner[x] - pulse.start)
                        if self.logging:
                            if header is not None:
                                self._log(header)
                                header = None
                            self._log('    {}: {} spans inner of {} (moving to inner, new lead length is {})'.
                                      format(x, pulse, inner[x], new_lead))
                        pulse.lead = new_lead
                        pulse.start = inner[x]
                elif idx == 0 and pulse.start > inner[x]:
                    # first half-pulse starts late
                    # this is either just 'jitter' or the pulse has merged into the inner white,
                    spare_space = pulse.start - inner[x]
                    if spare_space > 1:
                        # we've got room to insert another pulse, so assume we've merged with the inner white
                        # we assume this happens when the black ring is very small,
                        # so insert a pulse before with a lead length of 1 and a head length of the overshoot
                        good_pulses.append(Scan.Pulse(inner[x], 1, spare_space - 1))
                        if self.logging:
                            if header is not None:
                                self._log(header)
                                header = None
                            self._log('    {}: {} starts late, inserting {}'.format(x, pulse, good_pulses[-1]))
                    else:
                        # its just jitter, move start to inner and increase lead
                        if self.logging:
                            if header is not None:
                                self._log(header)
                                header = None
                            self._log('    {}: {} starts late, moving to inner {} with lead {}'.
                                      format(x, pulse, inner[x], pulse.lead + spare_space))
                        pulse.start = inner[x]
                        pulse.lead += spare_space
                head_start = pulse.start + pulse.lead
                if head_start == outer[x]:
                    # this is really the final pulse, so add that as a final pulse and then stop
                    pulse.head = 0
                    good_pulses.append(pulse)
                    break
                if head_start > outer[x]:
                    if pulse.start >= outer[x]:
                        # its wholly outside, this means the head of the preceding pulse extends too far,
                        # we assume this is due to a data pulse merging into the outer edge, so re-arrange
                        # the preceding pulse to have a shorter head and this pulse to have a small lead
                        # such that its lead ends at the outer and no head (it becomes the outer)
                        if len(good_pulses) > 0:
                            # action options:
                            #  1. reduce head length of preceding pulse such that it ends at outer-1,
                            #     set this pulse to start at outer-1 with a lead of 1 and a head of 0
                            #  2. increase length of preceding pulse lead such that it ends at outer,
                            #     drop this pulse
                            # action 2 is done is action 1 is not possible (because the preceding pulse
                            # start is too close to outer)
                            prior_pulse = good_pulses[-1]
                            prior_head_start = prior_pulse.start + prior_pulse.lead
                            head_end = outer[x] - 1
                            if prior_head_start < head_end:
                                # we've got room to do action 1
                                prior_pulse.head = head_end - prior_head_start
                                pulse.start = head_end
                                pulse.lead = outer[x] - head_end
                                if self.logging:
                                    if header is not None:
                                        self._log(header)
                                        header = None
                                    self._log('    {}: {} wholly outside outer of {} (sharing with {})'.
                                              format(x, pulse, outer[x], prior_pulse))
                                good_pulses.append(pulse)  # keep this adjusted pulse
                            else:
                                # no room for action 1, do action 2 instead
                                prior_pulse.lead = max(outer[x] - prior_pulse.start, 1)
                                prior_pulse.head = 0
                                if self.logging:
                                    if header is not None:
                                        self._log(header)
                                        header = None
                                    self._log('    {}: {} wholly outside outer of {} (merging in to {})'.
                                              format(x, pulse, outer[x], prior_pulse))
                        else:
                            # there is no preceding pulse, this means nothing here, just drop it
                            if self.logging:
                                if header is not None:
                                    self._log(header)
                                    header = None
                                self._log('    {}: {} wholly outside outer of {} (dropping)'.
                                          format(x, pulse, outer[x]))
                    else:
                        # lead spans the outer, truncate lead at the outer
                        head_start = pulse.start + pulse.lead
                        new_lead = pulse.lead - (head_start - outer[x])
                        if self.logging:
                            if header is not None:
                                self._log(header)
                                header = None
                            self._log('    {}: {} lead spans outer of {} (moving to outer, new lead length is {})'.
                                      format(x, pulse, outer[x], new_lead))
                        pulse.lead = new_lead
                        pulse.head = 0
                        good_pulses.append(pulse)  # keep this adjusted pulse
                    break  # we don't want anything beyond the outer
                if idx == (len(slice_pulses) - 1) and (pulse.start + pulse.lead) < outer[x]:
                    # final half-pulse starts early (NB: head is 0 in the last half-pulse)
                    # there are two cases: a data pulse has merged into the outer white or
                    # edge smoothing has moved us inside the outer edge slightly, if we are
                    # inside by enough to make a new pulse we assume the former, else the latter
                    spare_space = outer[x] - (pulse.start + pulse.lead)
                    if spare_space > 1:
                        # got enough room for a new pulse
                        # so split it such that pulse goes to outer edge and insert a new outer half-pulse after it
                        head_space = int(spare_space / 2)  # will be at least 1
                        lead_space = spare_space - head_space  # will also be at least 1
                        pulse.head = head_space
                        good_pulses.append(pulse)
                        pulse = Scan.Pulse(outer[x] - lead_space, lead_space)
                        if self.logging:
                            if header is not None:
                                self._log(header)
                                header = None
                            self._log('    {}: {} head starts early, splitting {}'.format(x, good_pulses[-1], pulse))
                    else:
                        # no room for a new pulse, just stretch the lead of the existing one
                        if self.logging:
                            if header is not None:
                                self._log(header)
                                header = None
                            self._log('    {}: {} head starts early, stretching lead to reach outer {} by {}'.
                                      format(x, pulse, outer[x], spare_space))
                        pulse.lead += spare_space
                # it survived, keep it
                good_pulses.append(pulse)
            if len(good_pulses) > 0:
                half_pulses[x] = good_pulses
                pulse_count += 1
            else:
                half_pulses[x] = None

        if self.logging:
            self._log('constrain: {} qualifying half-pulses:'.format(pulse_count))
            for x, slice_pulses in enumerate(half_pulses):
                if slice_pulses is None:
                    continue
                pulses = ''
                for pulse in slice_pulses:
                    pulses += ', {}'.format(pulse)
                pulses = pulses[2:]
                self._log('    {}: {}'.format(x, pulses))

        return half_pulses

    def _pulses(self, half_pulses: List[List[Pulse]]) -> List[Pulse]:
        """ find the radial pulses in the given half_pulses,
            returns a full pulse list
            """

        def find_predecessor(idx, slice_pulses):
            """ find the predecessor index of the given pulse index,
                it iterates backwards looking for the first non-None pulse
                """
            if idx > 0:
                for p in range(idx - 1, -1, -1):
                    if slice_pulses[p] is not None:
                        return p
            return None

        max_x = len(half_pulses)

        # merge short half-pulses (this is filtering noise)
        if self.logging:
            header = 'pulses: merge short half-pulses:'
        for x, slice_pulses in enumerate(half_pulses):
            if slice_pulses is None:
                continue
            for idx, pulse in enumerate(slice_pulses):
                if pulse is None:
                    continue
                if pulse.head > 0:
                    # NB: this means this is not the outer edge as they have a head len of zero
                    # if there is a following pulse with a longer head, dump this one
                    if idx < len(slice_pulses) - 1:
                        successor = slice_pulses[idx + 1]
                        if successor.head > pulse.head:
                            # assume this one is noise, merge it into the lead of its successor
                            slice_pulses[idx] = None
                            successor.start = pulse.start
                            successor.lead += (pulse.lead + pulse.head)
                            if self.logging:
                                if header is not None:
                                    self._log(header)
                                    header = None
                                self._log('    {}: merge {} into successor lead {}'.format(x, pulse, successor))
                            continue
                if 0 < pulse.head < Scan.MIN_PULSE_HEAD:
                    # too small, merge it into the lead of its successor
                    slice_pulses[idx] = None
                    if (idx + 1) < len(slice_pulses):
                        successor = slice_pulses[idx + 1]
                        successor.start = pulse.start
                        successor.lead += (pulse.lead + pulse.head)
                        if self.logging:
                            if header is not None:
                                self._log(header)
                                header = None
                            self._log('    {}: merge {} into successor lead {}'.format(x, pulse, successor))
                    else:
                        if self.logging:
                            if header is not None:
                                self._log(header)
                                header = None
                            self._log('    {}: dropping {}, too small with no successor'.format(x, pulse))
                    continue
                if pulse.head > 0 and pulse.lead < Scan.MIN_PULSE_LEAD:
                    # too small, merge it into the head of its predecessor
                    slice_pulses[idx] = None
                    pred = find_predecessor(idx, slice_pulses)
                    if pred is None:
                        # no predecessor, move self up instead
                        pulse.start -= pulse.lead
                        pulse.lead += pulse.lead
                        if self.logging:
                            if header is not None:
                                self._log(header)
                                header = None
                            self._log('    {}: moving self to {} from {}'.format(x, pulse, slice_pulses[idx]))
                        slice_pulses[idx] = pulse
                    else:
                        slice_pulses[pred].head += (pulse.lead + pulse.head)
                        if self.logging:
                            if header is not None:
                                self._log(header)
                                header = None
                            self._log('    {}: merge {} into predecessor head {}'.format(x, pulse, slice_pulses[pred]))
                    continue

        # merge extra half-pulses
        # due to the action of _constrain we know the first half-pulse starts at the inner edge
        # and the last half-pulse head starts at the outer edge, we expect there to be no other
        # half-pulses in between, if there are they are noise and the smallest is merged into
        # one of its neighbours (extra lead in its predecessor or extra head in its successor).
        if self.logging:
            header = 'pulses: merge extra half-pulses:'
        for x, slice_pulses in enumerate(half_pulses):
            if slice_pulses is None:
                continue
            # drop None pulses
            for idx in range(len(slice_pulses)-1, -1, -1):
                if slice_pulses[idx] is None:
                    del slice_pulses[idx]
            if len(slice_pulses) == 0:
                half_pulses[x] = None
                continue
            while len(slice_pulses) > 2:
                for idx in range(1, len(slice_pulses)-1):
                    pulse = slice_pulses[idx]
                    if pulse is None:
                        continue
                    # merge this extra with the most appropriate neighbour
                    pred = slice_pulses[idx - 1]
                    if pred.head > pulse.head:
                        # predecessor head is bigger - merge us into the lead of our successor
                        succ = slice_pulses[idx + 1]
                        if self.logging:
                            if header is not None:
                                self._log(header)
                                header = None
                            self._log('    {}: merge extra pulse {} into successor lead {}'
                                      ' (predecessor {} has bigger head)'.
                                      format(x, pulse, succ, pred))
                        succ.start = pulse.start
                        succ.lead += pulse.lead + pulse.head
                        del slice_pulses[idx]
                        break
                    else:  # pred.head <= pulse.head:
                        # we're bigger, merge predecessor into our lead
                        if self.logging:
                            if header is not None:
                                self._log(header)
                                header = None
                            self._log('    {}: merge extra pulse {} into successor lead {}'
                                      ' (successor {} has bigger or same head)'.
                                      format(x, pred, pulse, pulse))
                        pulse.start = pred.start
                        pulse.lead += pred.lead + pred.head
                        del slice_pulses[idx - 1]
                        break

        # combine two half-pulses into a full pulse
        # when we get here there are either zero, one or two half-pulses per slice, never more
        if self.logging:
            header = 'pulses: combining two half-pulses into full-pulses:'
        pulses = [None for _ in range(max_x)]
        for x, slice_pulses in enumerate(half_pulses):
            if slice_pulses is None:
                continue
            full_pulse = None
            for idx, pulse in enumerate(slice_pulses):
                if pulse is None:
                    continue
                # two half pulses make a full pulse,
                # we have a lead and head in the first and a tail from the lead in the second
                # we know each of these parts are of an acceptable length (from previous loop)
                if full_pulse is None:
                    # found the first one, we'll fill in the tail of this from the next half-pulse
                    full_pulse = pulse
                    full_pulse.tail = 0
                elif full_pulse.tail > 0:
                    # this means we've found an extra half-pulse, which implies the previous loop failed
                    self._log('pulses: extra half-pulse at {} of {}'.format(x, pulse), fatal=True)
                else:
                    # found the second one, its lead is our full pulse tail
                    full_pulse.tail = pulse.lead
                prev_pulse = pulse
            pulses[x] = full_pulse

        if self.logging:
            qualifiers = 0
            for pulse in pulses:
                if pulse is not None:
                    qualifiers += 1
            self._log('pulses: {} qualifying full pulses:'.format(qualifiers))
            for x, pulse in enumerate(pulses):
                if pulse is None:
                    continue
                self._log('    {}: {}'.format(x, pulse))

        return pulses

    def _clean(self, pulses: List[Pulse], max_x, max_y) -> List[Pulse]:
        """ clean the given full-pulse list of 'artifacts',
            'cleaning' means removing nipples, slopes and corners,
            artifacts are caused by luminance 'bleeding' in low resolution images
            """

        # ToDo: move these constants to Scan

        EXACT = 'exact'
        AT_LEAST = 'at-least'
        INHERIT_LEFT = 'inherit-left'
        INHERIT_RIGHT = 'inherit-right'
        UP_NIPPLE = 'up-nipple'
        DOWN_NIPPLE = 'down-nipple'
        FALLING_SLOPE = 'falling-slope'
        RISING_SLOPE = 'rising-slope'
        TOP_RIGHT_CORNER = 'top-right-corner'
        TOP_LEFT_CORNER = 'top-left-corner'
        BOTTOM_LEFT_CORNER = 'bottom-left-corner'
        BOTTOM_RIGHT_CORNER = 'bottom-right-corner'
        APPLY_AT_START = 'apply-at-start'
        APPLY_AT_END = 'apply-at-end'

        # an artifact to remove is defined by left and right neighbour offsets, exact or at-least,
        # inherit right or left and a context, offset is what must be added to self to meet the neighbour
        ARTIFACTS =((-1, EXACT   , +2, AT_LEAST, INHERIT_LEFT , TOP_RIGHT_CORNER   ),
                    (+2, AT_LEAST, -1, EXACT   , INHERIT_RIGHT, TOP_LEFT_CORNER    ),
                    (-2, AT_LEAST, +1, EXACT   , INHERIT_RIGHT, BOTTOM_LEFT_CORNER ),
                    (+1, EXACT   , -2, AT_LEAST, INHERIT_LEFT , BOTTOM_RIGHT_CORNER),
                    (-1, EXACT   , -1, EXACT   , INHERIT_LEFT , DOWN_NIPPLE        ),
                    (+1, EXACT   , +1, EXACT   , INHERIT_LEFT , UP_NIPPLE          ),
                    (-1, EXACT   , +1, EXACT   , INHERIT_LEFT , FALLING_SLOPE      ),
                    (+1, EXACT   , -1, EXACT   , INHERIT_LEFT , RISING_SLOPE       ))

        def move_head_start(x, pulse, from_start, to_start, context):
            """ move the head start of pulse as directed """
            nonlocal header
            change = from_start - to_start
            if self.logging:
                if header is not None:
                    self._log(header)
                    header = None
                self._log('    {}: ({}) moving head start from {} to {} in {} (change is {})'.
                          format(x, context, from_start, to_start, pulse, change))
            pulse.lead -= change
            pulse.head += change
            return pulse

        def move_head_end(x, pulse, from_end, to_end, context):
            """ move the head end of pulse as directed """
            nonlocal header
            change = from_end - to_end
            if self.logging:
                if header is not None:
                    self._log(header)
                    header = None
                self._log('    {}: ({}) moving head end from {} to {} in {} (change is {})'.
                          format(x, context, from_end, to_end, pulse, change))
            pulse.head -= change
            pulse.tail += change
            return pulse

        def get_start_end(pulse):
            """ get the head start/end for the given pulse """
            if pulse.tail == 0 or pulse.head == 0:
                head_start = pulse.start + pulse.lead
                head_end = pulse.start
            else:
                head_start = pulse.start + pulse.lead
                head_end = head_start + pulse.head
            return head_start, head_end

        def adjust_start_end(other_head_start, this_head_start, other_head_end, this_head_end):
            """ adjust other start/end if they do not overlap """
            if other_head_end < this_head_start:
                other_start = max_y
            else:
                other_start = other_head_start
            if other_head_start > this_head_end:
                other_end = 0
            else:
                other_end = other_head_end
            return other_start, other_end

        def apply(x, pulse, artifact, apply_at, left_at, this_at, right_at):
            """ apply the artifact rule """
            left_offset, left_mode, right_offset, right_mode, inherit_side, context = artifact
            if left_mode == EXACT:
                left_match = left_at == (this_at + left_offset)
            elif left_offset < 0:
                left_match = left_at <= (this_at + left_offset)
            else:
                left_match = left_at >= (this_at + left_offset)
            if right_mode == EXACT:
                right_match = right_at == (this_at + right_offset)
            elif right_offset < 0:
                right_match = right_at <= (this_at + right_offset)
            else:
                right_match = right_at >= (this_at + right_offset)
            if left_match and right_match:
                if inherit_side == INHERIT_LEFT:
                    inherit = left_at
                else:
                    inherit = right_at
                if apply_at == APPLY_AT_START:
                    pulse = move_head_start(x, pulse, this_at, inherit, context)
                else:
                    pulse = move_head_end(x, pulse, this_at, inherit, context)
            return pulse

        if self.logging:
            header = 'clean: remove artifacts'
        for x in range(len(pulses)):
            this_pulse = pulses[x]
            if this_pulse is None:
                continue
            left_x = (x - 1) % max_x
            left_pulse = pulses[left_x]
            if left_pulse is None:
                continue
            right_x = (x + 1) % max_x
            right_pulse = pulses[right_x]
            if right_pulse is None:
                continue
            left_head_start, left_head_end = get_start_end(left_pulse)
            right_head_start, right_head_end = get_start_end(right_pulse)
            for artifact in ARTIFACTS:
                this_head_start, this_head_end = get_start_end(this_pulse)
                left_start, left_end = adjust_start_end(left_head_start, this_head_start,
                                                        left_head_end, this_head_end)
                right_start, right_end = adjust_start_end(right_head_start, this_head_start,
                                                          right_head_end, this_head_end)

                this_pulse = apply(x, this_pulse, artifact, APPLY_AT_START,
                                   left_start, this_head_start, right_start)

                this_pulse = apply(x, this_pulse, artifact, APPLY_AT_END,
                                   left_end, this_head_end, right_end)

        return pulses

    def _extract(self, pulses: List[Pulse]) -> List[Pulse]:
        """ extract the bit sequences and their errors from the given full-pulse list """

        def error(actual, ideal):
            """ calculate an error between the actual pulse part ratios and the ideal,
                we want a number that is in the range 1..N where 1 is no error and N
                is a huge error, and quantised such that changes in the error mean the
                target has changed significantly
                """
            if ideal > actual:
                diff = ideal - actual
            else:
                diff = actual - ideal
            if ideal > 0:
                err = diff / ideal
            else:
                # actual needs to be 0 too, anything else is a big error
                if actual > 0:
                    err = 1
                else:
                    err = 0
            # err is now in range 0..1
            err *= Scan.RATIO_QUANTA  # 0..N
            err = int(err+1)  # quantised to 1..N
            return err

        # get the most likely bits
        for x, pulse in enumerate(pulses):
            if pulse is None:
                continue
            pulse.bits = []
            done = False
            for idx, ideal in enumerate(Scan.RATIOS):
                # we try every 'nudge' possibility (+/- 1 on lead, head, tail sizes) and pick the best
                # this mitigates against low resolution effects where true edges are between pixels or
                # the (noisy/blurry) image is in the process of migrating from one pulse to another
                options = []
                for lead_delta in (0, -1, +1):
                    for head_delta in (0, -1, +1):
                        for tail_delta in (0, -1, +1):
                            actual = pulse.ratio(lead_delta, head_delta, tail_delta)
                            err = 0
                            for part in range(len(ideal)):
                                err += error(actual[part], ideal[part])
                            err /= len(ideal)
                            options.append((int(err), actual))
                            if int(err) == 1:  # NB: 1 is our error offset, 1==no error
                                # got an exact match, so no point looking any further
                                done = True
                                break
                        if done:
                            break
                    if done:
                        break
                if len(options) > 1:
                    options.sort(key=lambda o: o[0])  # put into least error order
                pulse.bits.append(Scan.Bits(Scan.DIGITS[idx], options[0][0], actual=options[0][1], ideal=ideal))
                if done:
                    break
            pulse.bits.sort(key=lambda b: b.error())  # put into last error order
            if pulse.bits[0].error() == Scan.NO_CHOICE_ERROR:
                # got an exact match, so chuck the rest
                pulse.bits = [pulse.bits[0]]
        if self.logging:
            self._log('extract: bits and their errors:')
            for x, pulse in enumerate(pulses):
                if pulse is None:
                    continue
                elif len(pulse.bits) <= 1:
                    self._log('    {}: {}'.format(x, pulse.bits[0]))
                else:
                    self._log('    {}: {} options:'.format(x, len(pulse.bits)))
                    for b, bits in enumerate(pulse.bits):
                        self._log('        {}: {}'.format(b, bits))

        return pulses

    def _resolve(self, pulses: List[List[Pulse]], max_x) -> List[Bits]:
        """ resolve full-pulses into a list of bit choices """

        def choices(pulse):
            """ get the choices for the given pulse based on the error differential """
            if pulse is None:
                return []
            options = []
            for choice in pulse.bits:
                if len(options) == 0:
                    # this is the first choice
                    options.append(choice)
                    if choice.error() == Scan.NO_CHOICE_ERROR:
                        # this is a dead cert, so do not add other lesser choices (NB: this will always be first)
                        break
                else:
                    diff = choice.error() - options[0].error()
                    diff *= diff
                    if diff > Scan.MAX_CHOICE_ERR_DIFF_SQUARED:
                        # too big an error gap to consider as a choice
                        break
                    options.append(choice)
            return options

        def choose(my_choices, neighbour_choices):
            """ return a list of choices that are in both sets """
            chosen = []
            for this_choice in my_choices:
                for neighbour_choice in neighbour_choices:
                    if this_choice.bits == neighbour_choice.bits:
                        chosen.append(this_choice)
                        break
            return chosen

        # resolve choices
        chosen = [None for _ in range(max_x)]
        resolutions = 0
        for x, this_pulse in enumerate(pulses):
            this_choices = choices(this_pulse)
            if len(this_choices) > 1:
                # we have choices, so make a choice based on our neighbours
                # we choose the one that matches either neighbour,
                # if it matches neither we make no choice and keep them all
                left_choices = choices(pulses[(x - 1) % max_x])
                left_chosen = choose(this_choices, left_choices)
                right_choices = choices(pulses[(x + 1) % max_x])
                right_chosen = choose(this_choices, right_choices)
                if len(left_chosen) == 0 and len(right_chosen) == 0:
                    # no match with either neighbour
                    chosen[x] = this_choices
                else:
                    # get the union of left_chosen and right_chosen
                    for r in right_chosen:
                        in_l = False
                        for l in left_chosen:
                            if l == r:
                                in_l = True
                                break
                        if not in_l:
                            left_chosen.append(r)
                    # put back into least error order
                    left_chosen.sort(key=lambda b: b.error())  # ToDo: is this necessary?
                    chosen[x] = left_chosen
                resolutions += len(this_pulse.bits) - len(chosen[x])
            elif len(this_choices) > 0:
                # only one choice
                chosen[x] = this_choices
            else:
                # nothing here
                continue

        if self.logging:
            qualifiers = 0
            for choice in chosen:
                if choice is not None:
                    qualifiers += 1
            self._log('resolve: {} resolutions, chosen {}:'.format(resolutions, qualifiers))
            for x, choice in enumerate(chosen):
                if choice is None:
                    continue
                else:
                    msg = ''
                    for bits in choice:
                        msg = '{}, {}'.format(msg, bits.format(short=True))
                    self._log('    {}: {}'.format(x, msg[2:]))

        return chosen

    def _segments(self, chosen: List[Bits], max_x) -> List[Segment]:
        """ extract the segments from the given bit choice list """

        def same_choices(this_bits, that_bits, this_choices, that_choices, this_error, that_error):
            """ see if this and that bits and choices are the same,
                to be the same the bits must match and also for no-error choices, the error,
                the error check ensures we get a distinction between no-error and some-error,
                no-error cases never have choices beyond themselves (enforced by _resolve)
                """
            if this_choices is None:
                this_set = [this_bits]
            else:
                this_set = [this_choices[x].bits for x in range(len(this_choices))]
                this_set.append(this_bits)
            if that_choices is None:
                that_set = [that_bits]
            else:
                that_set = [that_choices[x].bits for x in range(len(that_choices))]
                that_set.append(that_bits)
            if len(this_set) != len(that_set):
                return False
            for this_choice in this_set:
                if this_choice not in that_set:
                    return False
            if this_error == Scan.NO_CHOICE_ERROR and that_error != Scan.NO_CHOICE_ERROR:
                return False
            if this_error != Scan.NO_CHOICE_ERROR and that_error == Scan.NO_CHOICE_ERROR:
                return False
            return True

        # amalgamate like bits (chosen is now a list of bits not pulses)
        ideal_length = int(round(max_x / Scan.NUM_SEGMENTS))
        segments = []
        for x, bits_list in enumerate(chosen):
            if bits_list is None:
                pulse_bits = None  # this is the nothing here signal
                pulse_error = 1  # error irrelevant when no bits but must not be 0
                pulse_choices = None
            elif len(bits_list) > 1:
                # got choices
                pulse_bits = bits_list[0].bits
                pulse_error = bits_list[0].error()
                pulse_choices = bits_list[1:]  # this is the choices signal
            else:
                pulse_bits = bits_list[0].bits
                pulse_error = bits_list[0].error()
                pulse_choices = None
            if len(segments) == 0:
                # first one, start a sequence
                segments.append(Scan.Segment(x, pulse_bits,
                                             error=pulse_error, choices=pulse_choices, ideal=ideal_length))
            elif same_choices(pulse_bits, segments[-1].bits,
                              pulse_choices, segments[-1].choices,
                              pulse_error, segments[-1].error()):
                # got another the same
                segments[-1].extend(1, pulse_error)
            else:
                # start of a new sequence
                segments.append(Scan.Segment(x, pulse_bits,
                                             error=pulse_error, choices=pulse_choices, ideal=ideal_length))
        if len(segments) > 1:
            if same_choices(segments[0].bits, segments[-1].bits,
                            segments[0].choices, segments[-1].choices,
                            segments[0].error(), segments[-1].error()):
                # got a wrapping segment
                segments[-1].extend(segments[0].samples, segments[0].error())
                del segments[0]

        if self.logging:
            self._log('segments: {} segments (ideal segment length={}):'.format(len(segments), ideal_length))
            for segment in segments:
                self._log('    {}'.format(segment))

        return segments

    def _combine(self, segments: List[Segment], max_x) -> List[Segment]:
        """ combine short segments:
              merge segment pairs that are below the max limit when combined
                merge rules: neighbour has a matching choice (all bits the same)
                             or has most common 1 bits
                             or abutts 000's
                             or anything
                if matched by choice, bring that choice to the fore
            """

        def neighbour(start_x, increment):
            """ scan segments from start_x for the first non-None one in direction implied by increment """
            x = start_x
            for _ in range(len(segments)-1):
                x = (x + increment) % len(segments)
                if segments[x] is None:
                    continue
                return x
            # everything is None if get here
            return None

        def merge(segment, mergee, bits, offset=0):
            """ merge mergee into segment and set bits as bits,
                if offset is 0 segment must immediately precede mergee,
                if offset is >0 segment must immediately succeed mergee,
                existing bits are added as a choice if different to bits,
                choices are merged too
                """
            nonlocal header
            if self.logging:
                if header is not None:
                    self._log(header)
                first = min(segment.start, mergee.start)
                last = max(segment.start, mergee.start)
                header = '    merging at {} and {}'.format(first, last)
            if bits != segment.bits:
                # add a choice of what it used to be
                choice = Scan.Bits(segment.bits, segment.error(), segment.samples)
                if segment.choices is None:
                    segment.choices = [choice]
                else:
                    segment.choices.append(choice)
                    segment.choices.sort(key=lambda c: c.error())
            segment.bits = bits
            segment.start = (segment.start - offset) % max_x
            merge_choices(segment, mergee, bits)
            if self.logging:
                self._log('{} into {}'.format(header, segment))
                header = None

        def merge_choices(segment, mergee, bits):
            """ merge the choices from mergee into segment and select the bits choice,
                choices in mergee that are in segment are extended,
                duplicates in segment are removed,
                choices that are the same as the current selection are removed
                """

            samples = segment.samples + mergee.samples  # final samples must span the entire space

            if bits is None:
                # this is a 'killing' action
                segment.replace(bits=bits, samples=samples, error=Scan.NO_CHOICE_ERROR)
                segment.choices = None
                return

            new_choices = extend_choices(mergee.choices,
                                         Scan.Bits(mergee.bits, mergee.error(), mergee.samples))
            old_choices = extend_choices(segment.choices,
                                         Scan.Bits(segment.bits, segment.error(), segment.samples))
            for new_choice in new_choices:
                is_dup = False
                for old_choice in old_choices:
                    if new_choice.bits == old_choice.bits:
                        old_choice.extend(new_choice.samples, new_choice.error())
                        is_dup = True
                        break
                if is_dup:
                    continue
                # ToDo: HACK causes too much identity change-->old_choices.append(new_choice)
            # pick the given bits choice
            for x, choice in enumerate(old_choices):
                if choice.bits == bits:
                    error = choice.error()  # ToDo: this is a distortion but do we care here?
                    segment.replace(bits=bits, samples=samples, error=error)
                    del old_choices[x]
                    break
            if len(old_choices) == 0:
                segment.choices = None
            else:
                segment.choices = old_choices
                segment.choices.sort(key=lambda c: c.error())  # put the rest back into least error order

        def extend_choices(choices, choice):
            """ create a new choice set with that given extended by the given extra,
                the returned choices will not have any duplicates
                """
            extended = [choice]
            if choices is None:
                return extended
            for old_choice in choices:
                is_dup = False
                for new_choice in extended:
                    if new_choice.bits == old_choice.bits:
                        is_dup = True
                        break
                if is_dup:
                    continue
                extended.append(old_choice)
            return extended

        def merge_pair(this_x, that_x, join_bits):
            """ merge the segment pair at this_x and that_x via join_bits,
                this is done because there are common bits between the pair (the join_bits)
                """
            this_segment = segments[this_x]
            that_segment = segments[that_x]
            if this_segment.choices is None and that_segment.choices is None:
                # neither have choices - so makes no difference
                this = this_x
                that = that_x
                offset = 0
            elif this_segment.choices is None:  # and that_segment.choices is not None:
                this = that_x
                that = this_x
                offset = this_segment.samples
            elif that_segment.choices is None:  # and this_segment.choices is not None:
                this = this_x
                that = that_x
                offset = 0
            elif len(this_segment.choices) < len(that_segment.choices):
                # right has more choices, stick with that
                this = that_x
                that = this_x
                offset = this_segment.samples
            else:
                # segment has more or same choices, extend that
                this = this_x
                that = that_x
                offset = 0
            merge(segments[this], segments[that], join_bits, offset)
            segments[that] = None

        def can_merge(this: Scan.Segment, that: Scan.Segment):
            """ determine if the given segments are allowed to be merged,
                to be allowed, both must be less than the ideal size,
                or at least one must be less than the min size limit,
                and None and non-None cannot merge,
                in both cases the combination must not exceed the max size limit (except for None's)
                """
            if this.bits is None and that.bits is None:
                # no restriction on merging these
                return True
            elif this.bits is None or that.bits is None:
                # cannot merge a mix of None and non-None
                return False
            elif (segment.size() + right.size()) < Scan.MAX_SEGMENT_LENGTH:
                return True
            elif this.size() < Scan.MIN_SEGMENT_LENGTH and that.size() < Scan.MIN_SEGMENT_LENGTH:
                return True
            else:
                return False

        def swap_bits(segment):
            """ swap the segment bits to its first other choice (but preserve samples),
                returns True iff a swap was mode, False iff a swap is not available
                """
            nonlocal header
            if segment.choices is None:
                # no choices available
                return False
            if segment.choices[0].bits == segment.bits:
                # duplicate choice - chuck it - this eats choices as we go
                del segment.choices[0]
            if len(segment.choices) == 0:
                # nothing left - down stream will have to sort it
                segment.choices = None
                return False
            if self.logging:
                if header is not None:
                    self._log(header)
                    header = None
                self._log('    swapping bits from {} to {} in {}'.
                          format(segment.bits, segment.choices[0], segment))
            samples = segment.samples  # keep for restore after replace
            segment.replace(bits=segment.choices[0].bits,
                            samples=segment.choices[0].samples,
                            error=segment.choices[0].error())
            segment.samples = samples  # keep the orig span
            return True

        def log(header):
            nonlocal updates
            if updates == 0:
                return
            self._log('combine: {} {} ({} updates):'.format(count(segments), header, updates))
            for segment in segments:
                if segment is None:
                    continue
                self._log('    {}'.format(segment))

        def count(segments):
            """ count non-none segments """
            count = 0
            for segment in segments:
                if segment is None:
                    continue
                count += 1
            return count

        # combine same bits
        if self.logging:
            header = 'combine: combine same bits:'
        updates = 0
        changes = True
        while changes:
            changes = False
            for x, segment in enumerate(segments):
                if segment is None:
                    continue
                if segment.choices is not None:
                    # leave these for later
                    continue
                right_x = neighbour(x, +1)
                if right_x is None:
                    continue
                right = segments[right_x]
                if right.choices is not None:
                    # leave these for later
                    continue
                if segment.bits == right.bits:
                    # same bits - merge them
                    merge_pair(x, right_x, segment.bits)
                    changes = True
                    updates += 1
                    break
        if self.logging and updates > 0:
            log('combined same bits')

        # combine matching choices
        if self.logging:
            header = 'combine: combine matching choices:'
        updates = 0
        changes = True
        while changes:
            changes = False
            for x, segment in enumerate(segments):
                if segment is None:
                    continue
                right_x = neighbour(x, +1)
                if right_x is None:
                    continue
                right = segments[right_x]
                if not can_merge(segment, right):
                    continue
                join_bits = self._overlaps(segment, right, exact=True)
                if join_bits is None:
                    # always allow None's to merge
                    if segment.bits is not None or right.bits is not None:
                        continue
                merge_pair(x, right_x, join_bits)
                changes = True
                updates += 1
                break
        if self.logging and updates > 0:
            log('combined matching choices')

        # # combine most common 1's choices
        # if self.logging:
        #     header = 'combine: combine most common 1\'s choices:'
        # updates = 0
        # changes = True
        # while changes:
        #     changes = False
        #     for x, segment in enumerate(segments):
        #         if segment is None:
        #             continue
        #         right_x = neighbour(x, +1)
        #         if right_x is None:
        #             continue
        #         right = segments[right_x]
        #         if not can_merge(segment, right):
        #             continue
        #         join_bits = self._overlaps(segment, right)
        #         if join_bits is None:
        #             continue
        #         # merge this pair
        #         # if half the pair is much bigger than the other, we use its bits rather than the common ones
        #         if segment.samples / right.samples > Scan.DOMINANT_SEGMENT_RATIO:
        #             join_bits = segment.bits
        #         elif right.samples / segment.samples > Scan.DOMINANT_SEGMENT_RATIO:
        #             join_bits = right.bits
        #         merge_pair(x, right_x, join_bits)
        #         changes = True
        #         updates += 1
        #         break
        # if self.logging:
        #     log('combined most common 1\'s choices')

        # combine 000 abutting choices
        if self.logging:
            header = 'combine: combine 000 abutting choices:'
        updates = 0
        changes = True
        while changes:
            changes = False
            for x, segment in enumerate(segments):
                if segment is None:
                    continue
                right_x = neighbour(x, +1)
                if right_x is None:
                    continue
                right = segments[right_x]
                if not can_merge(segment, right):
                    continue
                if segment.bits == Scan.DIGITS[0]:
                    if segment.size() > right.size() and right.size() < Scan.MIN_SEGMENT_LENGTH:
                        # merge small right segment into this segment
                        merge(segment, right, Scan.DIGITS[0])
                        segments[right_x] = None
                        changes = True
                        if self.logging:
                            self._log('        dropping {}'.format(right))
                elif right.bits == Scan.DIGITS[0]:
                    if right.size() > segment.size() and segment.size() < Scan.MIN_SEGMENT_LENGTH:
                        # merge small segment into the bigger right segment
                        merge(right, segment, Scan.DIGITS[0], segment.samples)
                        segments[x] = None
                        changes = True
                        if self.logging:
                            self._log('        dropping {}'.format(segment))
                if changes:
                    updates += 1
                    break
        if self.logging:
            log('combined 000 abutting choices')

        # # combine anything that is short
        # if self.logging:
        #     header = 'combine: combine short segments:'
        # updates = 0
        # changes = True
        # while changes:
        #     changes = False
        #     for x, segment in enumerate(segments):
        #         if segment is None:
        #             continue
        #         right_x = neighbour(x, +1)
        #         if right_x is None:
        #             continue
        #         right = segments[right_x]
        #         if not can_merge(segment, right):
        #             continue
        #         if segment.bits == Scan.DIGITS[0] and right.bits != Scan.DIGITS[0]:
        #             # cannot merge 000's into non-000's unless the 000's are very short
        #             if segment.size() > Scan.MIN_0_SEGMENT_LENGTH:
        #                 continue
        #         elif segment.bits != Scan.DIGITS[0] and right.bits == Scan.DIGITS[0]:
        #             # cannot merge 000's into non-000's unless the 000's are very short
        #             if segment.size() > Scan.MIN_0_SEGMENT_LENGTH:
        #                 continue
        #         if segment.size() > right.size():
        #             # merge right into segment
        #             merge(segment, right, segment.bits)
        #             segments[right_x] = None
        #             changes = True
        #         elif right.size() > segment.size():
        #             # merge segment into right
        #             merge(right, segment, right.bits, segment.samples)
        #             segments[x] = None
        #             changes = True
        #         else:
        #             # both same size, merge right into segment
        #             merge(segment, right, segment.bits)
        #             segments[right_x] = None
        #             changes = True
        #         if changes:
        #             updates += 1
        #             break
        # if self.logging:
        #     log('combined short segments')

        # # finally: change choice if got consecutive segments of the same bit
        # if self.logging:
        #     header = 'combine: swapping consecutive segments with same bits:'
        # updates = 0
        # changes = True
        # while changes:
        #     changes = False
        #     for x, segment in enumerate(segments):
        #         if segment is None:
        #             continue
        #         right_x = neighbour(x, +1)
        #         if right_x is None:
        #             continue
        #         right = segments[right_x]
        #         if segment.bits == right.bits:
        #             # got two the same - if there is a different choice use that
        #             if segment.choices is None:
        #                 # cannot change this, but maybe can the other
        #                 if swap_bits(right):
        #                     changes = True
        #                     updates += 1
        #                     break
        #                 continue
        #             elif right.choices is None:
        #                 # cannot change this, but maybe can the other
        #                 if swap_bits(segment):
        #                     changes = True
        #                     updates += 1
        #                     break
        #                 continue
        #             # swap the one with the biggest error
        #             if segment.error() > right.error():
        #                 if swap_bits(segment):
        #                     changes = True
        #                     updates += 1
        #                     break
        #             if swap_bits(right):
        #                 changes = True
        #                 updates += 1
        #                 break
        # if self.logging:
        #     log('swapped duplicate segments')

        # remove our dropped segments
        for x in range(len(segments)-1, -1, -1):
            if segments[x] is None:
                del segments[x]

        return segments

    def _separate(self, segments: List[Segment], max_x) -> List[Segment]:
        """ remove overlaps:
              an 'overlap' is e.g. (A)011->(B)111->(C)110, i.e. A&B<>0 and B&C<>0 and
                                                                A(len)>B(len) and C(len)>B(len) and
                                                                B(len)<some limit (the short limit)
              share B len with A and C, shortest first, drop B,
              overlaps occur due to pixel bleeding in low resolution images
            remove 000 incursions:
              an 'incursion' is when a short segment abutts a non-short 000's segment
            """

        def neighbour(start_x, increment):
            """ scan segments from start_x for the first non-None one in direction implied by increment """
            x = start_x
            for _ in range(len(segments) - 1):
                x = (x + increment) % len(segments)
                if segments[x] is None:
                    continue
                return x
            # everything is None if get here
            return None

        if self.logging:
            header = 'separate: removing overlaps:'
        for x, segment in enumerate(segments):
            if segment is None:
                continue
            if segment.size() < Scan.MIN_SEGMENT_LENGTH:
                left_x = neighbour(x, -1)
                if left_x is None:
                    continue
                right_x = neighbour(x, +1)
                if right_x is None:
                    continue
                left = segments[left_x]
                right = segments[right_x]
                if left.samples < segment.samples:
                    continue
                if right.samples < segment.samples:
                    continue
                if self._overlaps(left, segment) is None:
                    continue
                if self._overlaps(right, segment) is None:
                    continue
                # got an overlap
                if self.logging:
                    if header is not None:
                        self._log(header)
                        header = None
                    self._log('    dropping {}'.format(segment))
                segments[x] = None

        # remove our dropped segments
        dropped = 0
        for x in range(len(segments) - 1, -1, -1):
            if segments[x] is None:
                del segments[x]
                dropped += 1

        if self.logging and dropped > 0:
            self._log('separate: {} separated segments ({} dropped):'.format(len(segments), dropped))
            for segment in segments:
                self._log('    {}'.format(segment))

        return segments

    def _analyse(self, segments: List[Segment], max_x, max_y):
        """ analyse the segments to extract the bit sequences for each segment,
            each segment's bits consists of one or more of the encoding BITS or None,
            this is the function that is 'expected number of bits aware', all preceding
            functions are (mostly) number of bits agnostic,
            returns a list of bit choices for each segment
            """

        num_choices = 0
        num_gaps = 0
        for segment in segments:
            if segment.bits is None:
                num_gaps += 1
                continue
            if segment.choices is not None:
                num_choices += 1
                continue
        orig_segments = len(segments)

        reason = None

        if self.logging:
            header = 'analyse: too many segments: got {}, only want {}, dropping duplicates'.\
                     format(len(segments), Scan.NUM_SEGMENTS)
        while len(segments) > Scan.NUM_SEGMENTS:
            # got too many - chuck duplicates with no choices
            drop_x = None
            for x in range(len(segments)):
                segment = segments[x]
                neighbour_x = (x + 1) % len(segments)
                neighbour = segments[neighbour_x]
                if segment.bits == neighbour.bits:
                    # two in a row - drop the one with the least choices
                    if segment.choices is None and neighbour.choices is not None:
                        drop_x = x
                    elif segment.choices is not None and neighbour.choices is None:
                        drop_x = neighbour_x
                    elif segment.choices is None and neighbour.choices is None:
                        if segment.samples < neighbour.samples:
                            drop_x = x
                        else:
                            drop_x = neighbour_x
                    elif len(segment.choices) > len(neighbour.choices):
                        drop_x = neighbour_x
                    elif len(segment.choices) < len(neighbour.choices):
                        drop_x = x
                    elif segment.samples < neighbour.samples:
                        drop_x = x
                    else:
                        drop_x = neighbour_x
                    break
            if drop_x is None:
                break
            if self.logging:
                if header is not None:
                    self._log(header)
                    header = None
                self._log('    dropping {}'.format(segments[drop_x]))
            del segments[drop_x]

        if self.logging:
            header = 'analyse: too many segments: got {}, only want {}, dropping shortest'.format(len(segments), Scan.NUM_SEGMENTS)
        while len(segments) > Scan.NUM_SEGMENTS:
            # got too many - chuck shortest with biggest error
            shortest = None
            for x, segment in enumerate(segments):
                if segment.bits == Scan.DIGITS[0]:
                    # don't drop 000's
                    continue
                if shortest is None:
                    shortest = x
                    continue
                if segment.bits is None and segments[shortest].bits is not None:
                    # drop None's in preference to non-None's
                    shortest = x
                    continue
                if segment.bits is not None and segments[shortest].bits is None:
                    # don't promote from None to non-None
                    continue
                if segment.samples < segments[shortest].samples:
                    shortest = x
                    continue
                if segment.samples == segments[shortest].samples and segment.error() > segments[shortest].error():
                    shortest = x
                    continue
            if shortest is None:
                # this means everything is 000's
                if self.logging:
                    self._log('analyse: too many segments: got {}, need {}'.
                              format(len(segments), Scan.NUM_SEGMENTS))
                reason = 'got {}, need {} bits'.format(len(segments), Scan.NUM_SEGMENTS)
                break
            if self.logging:
                if header is not None:
                    self._log(header)
                    header = None
                self._log('    dropping {}'.format(segments[shortest]))
            del segments[shortest]

        if self.logging:
            header = 'analyse: not enough segments: got {}, need {}'.format(len(segments), Scan.NUM_SEGMENTS)
        while len(segments) < Scan.NUM_SEGMENTS:
            # haven't got enough - split biggest with biggest error
            biggest = None
            for x, segment in enumerate(segments):
                if segment.size() < (2 * Scan.MIN_SEGMENT_LENGTH):
                    # not big enough to split
                    continue
                if biggest is None:
                    biggest = x
                    continue
                if segment.bits is None and segments[biggest].bits is not None:
                    # split None's in preference to non-None's
                    biggest = x
                    continue
                if segment.bits is not None and segments[biggest].bits is None:
                    # don't promote from None to non-None
                    continue
                if segment.samples > segments[biggest].samples:
                    biggest = x
                    continue
                if segment.samples == segments[biggest] and segment.error() > segments[biggest].error():
                    biggest = x
                    continue
            if biggest is None:
                # this means nothing is splittable
                if self.logging:
                    self._log('analyse: not enough segments: got {}, need {}'.
                              format(len(segments), Scan.NUM_SEGMENTS))
                reason = 'got {}, need {} bits'.format(len(segments), Scan.NUM_SEGMENTS)
                break
            segment = segments[biggest]
            split_samples = int(segment.ideal * Scan.MIN_SEGMENT_LENGTH)
            if self.logging:
                if header is not None:
                    self._log(header)
                    header = None
                self._log('    splitting {} samples from {}'.format(split_samples, segment))
            segment.samples -= split_samples
            segments.insert(biggest, Scan.Segment(segment.start + split_samples,
                                                  segment.bits, split_samples,
                                                  segment.error(), ideal=segment.ideal))

        # return the bit choices from the final segment mix
        bits = [[] for _ in range(len(segments))]
        for x, segment in enumerate(segments):
            if segment.bits is None:
                bits[x].append([None for _ in range(Scan.NUM_DATA_RINGS)])
            elif segment.choices is None:
                bits[x].append(segment.bits)
            else:
                bits[x].append(segment.bits)
                for choice in segment.choices:
                    bits[x].append(choice.bits)

        if self.logging:
            self._log('analyse: {} bits from {} segments, {} with choices and {} as gaps:'.
                      format(len(bits), orig_segments, num_choices, num_gaps))
            for x, bit_list in enumerate(bits):
                msg = ''
                for bit in bit_list:
                    msg = '{}, {}'.format(msg, bit)
                self._log('    {}: {}'.format(x, msg[2:]))

        if self.save_images:
            grid = self._draw_segments(segments, max_x, max_y)
            self._unload(grid, '07-segments')

        return bits, reason

    def _measure(self, pulses: List[List[Pulse]], stretch_factor):
        """ get a measure of the target size by examining the slices,
            stretch_factor is how much the image height was stretched during projection,
            its used to re-scale the target size such that all are consistent wrt the original image
            """

        target_size = 0  # set as the most average of the pulse ends
        samples = 0
        for pulse in pulses:
            if pulse is not None:
                target_size += pulse.start + pulse.lead + pulse.head + pulse.tail
                samples += 1
        target_size /= samples
        target_size /= stretch_factor

        if self.logging:
            self._log('measure: target size is {:.2f} (with stretch compensation of {:.2f})'.
                      format(target_size, stretch_factor))

        return target_size

    def _decode_bits(self, bits):
        """ decode the bits for the least doubt and return its corresponding number,
            in bits we have a list of bit sequence choices across the data rings, i.e. bits x rings
            we need to rotate that to rings x bits to present it to our decoder,
            we present each combination to the decoder and pick the result with the least doubt
            """

        def build_choice(start_x, choice, choices):
            for x in range(start_x, Scan.NUM_SEGMENTS):
                bit_list = bits[x]
                if len(bit_list) > 1 and len(choices) < Scan.MAX_BIT_CHOICES:
                    # got choices - recurse for the others
                    for dx in range(1, len(bit_list)):
                        bit_choice = choice.copy()
                        bit_choice[x] = bit_list[dx]
                        choices = build_choice(x+1, bit_choice, choices)
                choice[x] = bit_list[0]
            choices.append(choice)
            return choices

        # build all the choices
        choices = build_choice(0, [None for _ in range(Scan.NUM_SEGMENTS)], [])

        # try them all
        results = []
        for choice in choices:
            code = [[None for _ in range(Scan.NUM_SEGMENTS)] for _ in range(Scan.NUM_DATA_RINGS)]
            for bit in range(Scan.NUM_SEGMENTS):
                rings = choice[bit]
                if rings is not None:
                    for ring in range(len(rings)):
                        sample = rings[ring]
                        code[ring][bit] = sample
            number, doubt, digits = self.decoder.unbuild(code)
            results.append((number, doubt, digits, choice))

        # put into least doubt order with numbers before None's
        # ToDo: or pick the one that has most solutions?
        results.sort(key=lambda r: (r[0] is None, r[1]))

        number, doubt, digits, choice = results[0]

        if self.logging:
            self._log('decode: {} results (limit is {}), best is: number={}, doubt={}, digits={}, bits:'.
                      format(len(results), Scan.MAX_BIT_CHOICES, number, doubt, digits))
            for x, bits in enumerate(choice):
                self._log('    {}: {}'.format(x, bits))

        # return best
        return number, doubt, digits

    def _find_codes(self):
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
            self.centre_x = int(round(blob.pt[0]))
            self.centre_y = int(round(blob.pt[1]))
            blob_size = blob.size / 2  # change diameter to radius

            if self.logging:
                self._log('***************************')
                self._log('processing candidate target')

            # do the polar to cartesian projection
            limit_radius = self._radius(self.centre_x, self.centre_y, blob.size)
            target, stretch_factor = self._project(self.centre_x, self.centre_y, limit_radius)
            max_x, max_y = target.size()

            # do the edge detection
            buckets = self._threshold(target)
            slices = self._slices(buckets)
            edges = self._edges(slices, max_x, max_y)
            extent = self._extent(edges, max_x, max_y)

            if self.save_images:
                plot = self._draw_edges(edges, extent, target)
                self._unload(plot, '05-edges')

            # do the pulse detection
            half_pulses = self._transitions(slices)
            half_pulses = self._constrain(half_pulses, extent)
            full_pulses = self._pulses(half_pulses)
            pulses = self._clean(full_pulses, max_x, max_y)

            if self.save_images:
                plot = self._draw_pulses(pulses, extent, buckets)
                self._unload(plot, '06-pulses')

            # get target size relative to original image (for relative range judgement)
            target_size = self._measure(pulses, stretch_factor)

            # do the segment extraction
            pulses = self._extract(pulses)
            chosen = self._resolve(pulses, max_x)
            segments = self._segments(chosen, max_x)
            segments = self._combine(segments, max_x)
            segments = self._separate(segments, max_x)

            # analyse segments to get the most likely bit sequences
            bits, reason = self._analyse(segments, max_x, max_y)
            if reason is not None:
                # failed - this means some constraint was not met (its already been logged)
                if self.save_images:
                    # add to reject list for labelling on the original image
                    rejects.append(Scan.Reject(self.centre_x, self.centre_y, blob_size, target_size, reason))
                continue

            # decode the bits for the best result
            result = self._decode_bits(bits)

            targets.append(Scan.Target(self.centre_x, self.centre_y, blob_size, target_size, target, result))

        if self.save_images:
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

        numbers = []
        for target in targets:
            self.centre_x = target.centre_x    # for logging and labelling
            self.centre_y = target.centre_y    # ..
            blob_size = target.blob_size
            target_size = target.target_size
            image = target.image
            number, doubt, digits = target.result

            # add this result
            numbers.append(Scan.Detection(number, doubt, self.centre_x, self.centre_y, target_size, blob_size, digits))

            if self.save_images:
                number = numbers[-1]
                if number.number is None:
                    colour = Scan.PURPLE
                    label = 'code is invalid ({})'.format(number.doubt)
                else:
                    colour = Scan.GREEN
                    label = 'code is {} ({})'.format(number.number, number.doubt)
                if labels is not None:
                    # draw the detected blob in blue
                    k = (number.centre_x, number.centre_y, number.blob_size)
                    labels = self.transform.label(labels, k, Scan.BLUE)
                    # draw the result
                    k = (number.centre_x, number.centre_y, number.target_size)
                    labels = self.transform.label(labels, k, colour, '{:.0f}x{:.0f}y {}'.
                                                  format(number.centre_x, number.centre_y, label))

        if self.save_images:
            self._unload(labels, 'targets', 0, 0)

        return numbers

    # region Helpers...
    def _overlaps(self, this_segment, that_segment, exact=False):
        """ see if this and that segments have overlapping 1 bits or have a common choice,
            if exact is True want same bits in both, else just most common one bits,
            returns the matching bit set from this_segment or None if no match
            """
        if this_segment.choices is None:
            this_set = [this_segment.bits]
        else:
            this_set = [this_segment.choices[x].bits for x in range(len(this_segment.choices))]
            this_set.append(this_segment.bits)
        if that_segment.choices is None:
            that_set = [that_segment.bits]
        else:
            that_set = [that_segment.choices[x].bits for x in range(len(that_segment.choices))]
            that_set.append(that_segment.bits)
        best_match = None
        best_ones = 0
        best_zeroes = 0
        for this_choice in this_set:
            if this_choice is None:
                continue
            for that_choice in that_set:
                if that_choice is None:
                    continue
                one_count, zero_count, diff_count = self._common_bits(this_choice, that_choice)
                if exact:
                    if diff_count == 0:
                        return this_choice
                    else:
                        continue
                if one_count > 0:
                    if one_count > best_ones:
                        best_ones = one_count
                        best_zeroes = zero_count
                        best_match = this_choice
                    elif one_count == best_ones:
                        # when got an equal 1's choice pick the one with the most 0's
                        if zero_count > best_zeroes:
                            best_ones = one_count
                            best_zeroes = zero_count
                            best_match = this_choice

        return best_match

    def _common_bits(self, this_bits: List[int], that_bits: List[int]):
        """ given two sets of bits return a count the common 1 bits and 0 zero bits and different bits,
            e.g. 111 and 111 is 3, 011 and 111 is 2, 011 and 110 is 1, 100 and 010 is 0
            """
        if this_bits is None or that_bits is None:
            return 0
        one_count = 0
        zero_count = 0
        diff_count = 0
        for bit in range(len(this_bits)):
            if this_bits[bit] == that_bits[bit]:
                if this_bits[bit] == 1:
                    one_count += 1
                else:
                    zero_count += 1
            else:
                diff_count += 1

        return one_count, zero_count, diff_count

    def _get_gap(self, first, second, max_x=0):
        """ compute distance between first and second pixel (by Pythagoras without the square root),
            if wrapping possible first must be 'before' second but second may have wrapped in x at max_x,
            NB: when wrapping first == second is considered as wrapped
            """

        if second[0] <= first[0]:
            # its wrapped, adjust to compensate
            second_x = second[0] + max_x
        else:
            second_x = second[0]

        x_gap = (second_x - first[0])
        x_gap *= x_gap
        y_gap = (second[1] - first[1])
        y_gap *= y_gap

        return x_gap + y_gap

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

    def _draw_lines(self, source, lines, colour=RED, bleed=0.5):
        """ draw lines in given colour,
            lines param is an array of start-x,start-y,end-x,end-y tuples,
            for horizontal lines start-y and end-y are the same,
            for vertical lines start-x and end-x are the same,
            for horizontal and vertical lines bleed defines how much background bleeds through
            """
        dlines = []
        vlines = []
        hlines = []
        for line in lines:
            if line[0] == line[2]:
                # vertical line
                vlines.append([line[1], [line[0] for _ in range(line[1], line[3]+1)]])
            elif line[1] == line[3]:
                # horizontal line
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

    def _draw_extent(self, extent, target, bleed):
        """ make the area outside the inner and outer edges on the given target """

        max_x, max_y = target.size()
        inner, outer = extent

        inner_lines = []
        outer_lines = []
        for x in range(max_x):
            inner_lines.append((x, 0, x, inner[x]))
            outer_lines.append((x, outer[x], x, max_y - 1))
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
                head_lines.append((x, pulse.start + pulse.lead, x,
                                   pulse.start + pulse.lead + pulse.head - 1))
                if pulse.tail > 0:
                    # for half-pulses there is no tail
                    tail_lines.append((x, pulse.start + pulse.lead + pulse.head, x,
                                       pulse.start + pulse.lead + pulse.head + pulse.tail - 1))

        # mark the extent
        plot = self._draw_extent(extent, buckets, bleed=0.6)

        # draw lines on the bucketised image
        plot = self._draw_lines(plot, none_lines, colour=Scan.RED)
        plot = self._draw_lines(plot, lead_lines, colour=Scan.GREEN)
        plot = self._draw_lines(plot, head_lines, colour=Scan.BLUE)
        plot = self._draw_lines(plot, tail_lines, colour=Scan.GREEN)

        return plot

    def _draw_edges(self, edges, extent, target):
        """ draw the edges and the inner and outer extent on the given target image """

        falling_edges, rising_edges = edges

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

        # ToDo: add random rotate digits and error rate params

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
                            expected = '{:b}'.format(self.codec.encode(num))
                            break
                    analysis.append([found_num, centre_x, centre_y, num, doubt, size, expected, bits])
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
                        if doubt == 0:
                            bits = ''
                        else:
                            bits = ', bits {}'.format(bits)
                        if found is not None:
                            if loop != 0:
                                # don't want these in this loop
                                continue
                            # got what we are looking for
                            self._log('Found {} ({}) at {:.0f}x, {:.0f}y size {:.2f}, doubt {}{}'.
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
                            else:
                                actual_code = self.codec.encode(num)
                                if actual_code is None:
                                    actual_code = 'not-valid'
                                    prefix = ''
                                else:
                                    actual_code = '{} ({:b})'.format(num, actual_code)
                                    prefix = '**** UNEXPECTED **** ---> '
                            self._log('{}Found {} ({}) at {:.0f}x, {:.0f}y size {:.2f}, doubt {}{}'.
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
    test_scan_cells = (8, 4)

    # reducing the resolution means targets have to be closer to be detected,
    # increasing it takes longer to process, most modern smartphones can do 4K at 30fps, 2K is good enough
    test_scan_video_mode = Scan.VIDEO_2K

    test_debug_mode = Scan.DEBUG_IMAGE
    #test_debug_mode = Scan.DEBUG_VERBOSE

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

        test.scan_codes(test_codes_folder)
        test.scan_media(test_media_folder)

        # test.scan(test_codes_folder, [000], 'test-code-000.png')
        # test.scan(test_codes_folder, [444], 'test-code-444.png')

        # test.scan(test_media_folder, [101], 'photo-101.jpg')
        # test.scan(test_media_folder, [101, 102, 182, 247, 301, 424, 448, 500, 537, 565], 'photo-101-102-182-247-301-424-448-500-537-565-v1.jpg')

    except:
        traceback.print_exc()

    finally:
        if test is not None:
            del (test)  # needed to close the log file(s)


if __name__ == "__main__":
    verify()
