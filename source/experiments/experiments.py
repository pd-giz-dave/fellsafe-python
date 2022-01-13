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
        surrounded by 4 concentric data rings of width R and divided into N equal segments,
        enclosed by a solid 'black' ring of width 1R,
        enclosed by a solid 'white' ring of width 1R. 
    Total radius is 9R.
    The code consists of a 5 digit number in base 6, the lead digit is always 0 and the remaining digits are
    in the range 1..5 (i.e. do not use 0), these 5 digits are repeated 3 times yielding a 15 digit number
    with triple redundancy. The six numbers are mapped across the 4 data rings to produce a 4-bit code, of
    the 16 possibilities for that code only those that yield a single 'pulse' are used (i.e. the bits are
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
    cv2's cor-ordinates are backwards from our pov, the 'x' co-ordinate is vertical and 'y' horizontal.
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
        differing ratios between the low length and the high length,
        this class encapsulates all the encoding and decoding and their constants
        """

    # region constants...
    DIGITS_PER_NUM = 4 # how many digits per encoded number
    COPIES = 3  # number of copies in a code-word (thus 'bits' around a ring is DIGITS_PER_NUM * COPIES)
    EDGES = 6  # min number of edges per rendered ring (to stop accidental solid rings looking like a blob)
    # these encodings MUST be arranged in descending ratio order (e.g. 1:4-->1:1/4)
    # the ratio is referring to the across rings pulse shape, there are choices for some ratios
    # where there are choices those that tend to an overall balance in 1's and 0's are chosen
    # the ordering means when there is doubt the most likely other candidate is +/- 1
    # base 6 encoding in 4 rings (yields 618 usable codes)
    ENCODING = [[1, 1, 1, 1],   # ratio 1:4   digit 0 (the sync digit)
                [1, 1, 1, 0],   # ratio 1:3   digit 1
                [1, 1, 0, 0],   # ratio 1:2   digit 2
                [0, 1, 1, 1],   # ratio 1:3/2 digit 3
                [1, 0, 0, 0],   # ratio 1:1   digit 4 (can be 1:1 -> 0, 1, 1, 0 or 1:1 -> 1, 0, 0, 0)
                [0, 1, 0, 0],   # ratio 1:1/2 digit 5 (can be 1:2/3 -> 0, 0, 1, 1 or 1:1/2 -> 0, 1, 0, 0)
                [0, 0, 0, 1]]   # ratio 1:1/4 digit 6 (can be 1:1/4 -> 0, 0, 0, 1 or 1:1/3 -> 0, 0, 1, 0)
    NOTHING = [0, 0, 0, 0]  # for external use to mark a 'nothing' bit sequence
    RATIOS = (4, 3, 2, 3 / 2, 1, 1 / 2, 1 / 4)  # see ENCODING table, for external use
    BASE = len(ENCODING) - 1  # number base of each digit (-1 to exclude the '0')
    RINGS = len(ENCODING[0])  # number of rings to encode each digit in a code-block
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

    NUM_RINGS = Codec.RINGS + 4  # total rings in our complete code
    DIGITS = Codec.DIGITS  # how many bits in each ring

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
        """ draw a half width black ring and a half width grey ring,
            the black is intended to reduce the possibility of the outer black being lost in distortion,
            the grey is intended as a visual marker to stop people cutting into important parts
            of the code and as a predictable outer luminance environment
            """

        # draw the border
        self._draw_ring(ring_num, 0, 0.5)  # half width black ring
        self._draw_ring(ring_num + 0.5, None, 0.5)  # half width grey ring

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
        start_x = (self.w >> 1) - min(ring_num * self.w, self.x)
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

        # draw the outer black ring
        self._draw_ring(draw_at, 0, 1.0)
        draw_at += 1.0

        if int(draw_at) != draw_at:
            raise Exception('number of target rings is not integral ({})'.format(draw_at))
        draw_at = int(draw_at)

        # safety check
        if draw_at != Ring.NUM_RINGS:
            raise Exception('number of rings exported ({}) is not {}'.format(Ring.NUM_RINGS, draw_at))

        # draw a border (this serves two purposes: a visual cue and a predictable luminance outline)
        self._border(draw_at)
        draw_at += 1

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
    NUM_RINGS = Ring.NUM_RINGS  # total number of rings in the whole code
    NUM_DATA_RINGS = Codec.RINGS  # how many data rings in our codes
    NUM_BITS = Codec.DIGITS  # total number of bits in a ring (bits==segments==cells)

    # image 'cell' constraints (a 'cell' is the angular division in a ring)
    # these constraints set minimums that override the cells() property given to Scan
    MIN_PIXELS_PER_CELL = 4  # min pixels making up a cell length
    MIN_PIXELS_PER_RING = 4  # min pixels making up a ring width

    # region Tuning constants...
    MIN_BLOB_SEPARATION = 5  # smallest blob within this distance of each other are dropped
    BLOB_RADIUS_STRETCH = 1.3  # how much to stretch blob radius to ensure always cover everything
    MIN_INNER_EDGE = 3  # minimum y co-ord of the inner edge (first white-->black transition), earlier==noise
    THRESHOLD_TOP_BORDER = 1 / NUM_RINGS  # fraction of top of image to ignore when calculating thresholds
    THRESHOLD_TOP_BORDER_PIXELS = 2  # border pixels if the above is less than this
    THRESHOLD_BOTTOM_BORDER = 3 / NUM_RINGS  # fraction of bottom of image to ignore when calculating thresholds
    THRESHOLD_BOTTOM_BORDER_PIXELS = 6  # border pixels if the above is less than this
    MIN_PULSE_LOW = 2  # minimum pixels for a valid pulse low period, pulse ignored if less than this
    MIN_PULSE_HIGH = 2  # minimum pixels for a valid pulse high period, pulse ignored if less than this
    MAX_START_DIFF = 0.5  # how far from average a pulse is allowed to start, further away and its dropped
    MAX_END_DIFF = 0.5  # how far from average a pulse is allowed to end, further away and its dropped
    MIN_SEGMENT_LENGTH = 3  # min length of a segment, shorter segments are merged with their neighbours
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

    # region Edge types...
    RISING = 'rising'
    FALLING = 'falling'
    # endregion

    # region Pulse classifications...
    # for the RATIOS table, the low length==1, the table entry is the corresponding high length
    # the order in the RATIOS table is the same as BITS which is used to look-up the bit sequence
    RATIOS = Codec.RATIOS
    BITS = Codec.ENCODING
    NO_BITS = Codec.NOTHING
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
    class Edge:
        """ an Edge is a description of a luminance change in a radius """

        def __init__(self, where, type):
            self.where = where  # the y co-ord of the edge
            self.type = type  # the edge type, rising or falling

        def __str__(self):
            return '({} at {})'.format(self.type, self.where)

    class Pulse:
        """ a Pulse describes the pixels in a radius """

        def __init__(self, start, low=0, high=0):
            self.start = start  # y co-ord where the pulse starts in the radius
            self.low = low  # length of the '0' pixels (including the inner black)
            self.high = high  # length of the '1' pixels
            self.bits = []  # list of bits and their error for this ratio in least error order

        def ratio(self):
            """ return the ratio best representing this pulse """
            if self.low > 0.0:
                # ToDo: do the mean of (H+/-1) / (L+/-1), i.e. of all nine variants
                #       to mitigate effect of very short pulses
                return self.high / self.low
            else:
                return 0.0

        def __str__(self):
            bits = ''
            if len(self.bits) > 0:
                for bit in self.bits:
                    bits = '{}, {} ({:.2f})'.format(bits, bit[0], bit[1])
                bits = ', bits={}'.format(bits[2:])
            return '(start={}, low={}, high={}, ratio={:.2f}{})'.\
                   format(self.start, self.low, self.high, self.ratio(), bits)

    class Segment:
        """ a Segment describes a contiguous sequence of Pulses """

        def __init__(self, start, bits, samples=1, error=None):
            self.start = start  # the start x of this sequence
            self.bits = bits  # the bit pattern for this sequence
            self.samples = samples  # how many of them we have
            self.error = error  # the harmonic mean error of the samples of this segment * 1000

        def __str__(self):
            return '(at {} for {} bits={}, error={})'.format(self.start, self.samples, self.bits, self.error)

    class Target:
        """ structure to hold detected target information """

        def __init__(self, centre_x, centre_y, blob_size, target_size, image, bits):
            self.centre_x = centre_x  # x co-ord of target in original image
            self.centre_y = centre_y  # y co-ord of target in original image
            self.blob_size = blob_size  # blob size originally detected by the blob detector
            self.target_size = target_size  # target size scaled to the original image (==outer edge average Y)
            self.image = image  # the image of the target(s)
            self.bits = bits  # list of bit sequences across the data rings

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
            angles is the angular resolution to use
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
        self.cells = cells  # (bit length x ring height) size of bit cells to use when decoding
        self.video_mode = video_mode  # actually the downsized image height
        self.original = frame
        self.decoder = code  # class to decode what we find

        # set warped image width
        self.angle_steps = Scan.NUM_BITS * max(self.cells[0], Scan.MIN_PIXELS_PER_CELL)

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

        def get_gap(first, second, max_x=0):
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

        # prepare image
        blurred = self.transform.blur(self.original)  # de-noise
        self.image = self.transform.downsize(blurred, self.video_mode)  # re-size to given video mode

        # set filter parameters
        threshold = (MIN_LUMINANCE, MAX_LUMINANCE, 8)  # min, max luminance, luminance step
        circularity = (0.75, None)  # min, max 'corners' in blob edge or None
        convexity = (0.5, None)  # min, max 'gaps' in blob edge or None
        inertia = (0.4, None)  # min, max 'squashed-ness' or None
        area = (30, 250000)  # min, max area in pixels, or None for no area filter
        gaps = (None, None)  # how close blobs have to be to be merged and min number
        colour = MAX_LUMINANCE  # we want bright blobs, use MIN_LUMINANCE for dark blobs

        # find the blobs
        blobs = self.transform.blobs(self.image, threshold, circularity, convexity, inertia, area, gaps, colour)
        blobs = list(blobs)

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
                    gap = get_gap((x1, y1), (x2, y2))
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
            for blob in dup_blobs:
                self._log('    centred at {}x{}y, size {:.2f}  (dropped as a duplicate)'.
                          format(int(round(blob.pt[0])), int(round(blob.pt[1])), blob.size))
        if self.save_images:
            grid = self.image
            if len(blobs) > 0:
                grid = self.transform.drawKeypoints(grid, blobs, Scan.GREEN)
            if len(dup_blobs) > 0:
                grid = self.transform.drawKeypoints(grid, dup_blobs, Scan.RED)
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
            blob = self.transform.label(blob, k, Scan.RED, '{:.0f}x{:.0f}y ({:.1f})'.
                                        format(centre_x, centre_y, blob_size))
            self._unload(blob, '01-target', centre_x, centre_y)

        if self.logging:
            self._log('radius: limit radius {}'.format(limit_radius))

        return limit_radius

    def _project(self, centre_x, centre_y, limit_radius) -> Frame:
        """ 'project' a potential target at centre_x/y from its circular shape to a rectangle
            of radius (y) by angle (x), limit_radius is far the warp,
            returns the projected image
            """

        image_height = limit_radius  # one pixel per radius
        image_width = self.angle_steps

        # do the projection
        code = self.transform.warpPolar(self.image, centre_x, centre_y, limit_radius, image_width, image_height)

        if self.save_images:
            # draw projected image
            self._unload(code, '02-projected')
        if self.logging:
            max_x, max_y = code.size()
            self._log('project: projected image size {}x {}y'.format(max_x, max_y))

        return code

    def _threshold(self, target: Frame) -> Frame:
        """ threshold the given image into binary """

        # make an empty (i.e. black) image
        max_x, max_y = target.size()
        buckets: Frame = target.instance()
        buckets.new(max_x, max_y, MIN_LUMINANCE)

        # build bucketised image
        for x in range(max_x):
            # get the pixels
            slice_pixels = [None for _ in range(max_y)]  # pixels of our slice
            for y in range(max_y):
                pixel = target.getpixel(x, y)
                slice_pixels[y] = pixel
            # set threshold as the mean of our pixel slice
            # we ignore the first few and last few pixels (they're typically distorted and bias the threshold)
            top_border = int(min(max(max_y * Scan.THRESHOLD_TOP_BORDER, Scan.THRESHOLD_TOP_BORDER_PIXELS),
                                 max_y / 3))
            bottom_border = int(min(max(max_y * Scan.THRESHOLD_BOTTOM_BORDER, Scan.THRESHOLD_BOTTOM_BORDER_PIXELS),
                                    max_y / 3))
            y_threshold_span = range(top_border, max_y - bottom_border)
            grey = 0
            samples = 0
            for y in y_threshold_span:
                grey += slice_pixels[y]
                samples += 1
            grey /= samples
            # 2 levels: black, white
            for y in range(max_y):
                if slice_pixels[y] > grey:
                    pixel = MAX_LUMINANCE
                else:
                    pixel = MIN_LUMINANCE
                buckets.putpixel(x, y, pixel)

        # ToDo: do an initial threshold, then find edges, then adjust threshold y span to do it better

        if self.save_images:
            self._unload(buckets, '03-buckets')

        return buckets

    def _slice(self, buckets: Frame) -> List[Edge]:
        """ detect radial edges in the given binary image,
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
                    # falling edge
                    slices[x].append(Scan.Edge(y, Scan.FALLING))
                elif pixel > last_pixel:
                    # rising edge
                    slices[x].append(Scan.Edge(y, Scan.RISING))
                last_pixel = pixel

        # trim duplicate edges (falling followed by falling, rising followed by rising)
        if self.logging:
            header = 'slice: drop/merge edges:'
        for x, slice in enumerate(slices):
            last_edge = None
            for idx, edge in enumerate(slice):
                if last_edge is None and edge.type == Scan.RISING:
                    # the first edge we want is falling, so mark this rising one as dead
                    if self.logging:
                        if header is not None:
                            self._log(header)
                            header = None
                        self._log('    {}: dropping edge {} (no falling edge yet)'.format(x, edge))
                    slice[idx] = None
                    continue
                if last_edge is None and edge.where < Scan.MIN_INNER_EDGE:
                    # too close to the centre - ignore it as noise
                    if self.logging:
                        if header is not None:
                            self._log(header)
                            header = None
                        self._log('    {}: dropping {} (above min y of {})'.format(x, edge, Scan.MIN_INNER_EDGE))
                    slice[idx] = None
                    continue
                if last_edge is not None and last_edge.type == edge.type:
                    # consecutive edge
                    slice[idx - 1] = None  # mark predecessor as dead
                last_edge = edge
            # remove dead edges
            for idx in range(len(slice) - 1, -1, -1):
                if slice[idx] is None:
                    del slice[idx]
            if len(slice) == 0:
                # no edges here
                slices[x] = None
            elif slice[-1].type == Scan.RISING:
                # remove trailing rising edge - means ran off outer black ring
                if self.logging:
                    if header is not None:
                        self._log(header)
                        header = None
                    self._log('    {}: dropping trailing edge {}'.format(x, slice[-1]))
                del slice[-1]

        return slices

    def _pulsify(self, slices) -> List[Pulse]:
        """ find the radial pulses in the given slices,
            returns a pulse list
            """

        def has_start_neighbour(x, pulses: List[List[Scan.Pulse]], pulse: Scan.Pulse):
            """ return True iff a left or right neighbour of this pulse has the same start """
            left = pulses[(x - 1) % len(pulses)]
            if left is not None:
                for slice_pulse in left:
                    if slice_pulse is None:
                        continue
                    if slice_pulse.start == pulse.start:
                        # its got a left neighbour
                        return True
            right = pulses[(x + 1) % len(pulses)]
            if right is not None:
                for slice_pulse in right:
                    if slice_pulse is None:
                        continue
                    if slice_pulse.start == pulse.start:
                        # its got a right neighbour
                        return True
            return False

        def has_high_neighbour(x, pulses: List[List[Scan.Pulse]], pulse: Scan.Pulse):
            """ return True iff a left or right neighbour of this pulse has the same rising edge """
            pulse_middle = pulse.start + pulse.low
            left = pulses[(x - 1) % len(pulses)]
            if left is not None:
                for slice_pulse in left:
                    if slice_pulse is None:
                        continue
                    left_middle = slice_pulse.start + slice_pulse.low
                    if left_middle == pulse_middle:
                        # its got a left neighbour
                        return True
            right = pulses[(x + 1) % len(pulses)]
            if right is not None:
                for slice_pulse in right:
                    if slice_pulse is None:
                        continue
                    right_middle = slice_pulse.start + slice_pulse.low
                    if right_middle == pulse_middle:
                        # its got a right neighbour
                        return True
            return False

        def has_low_neighbour(x, pulses: List[List[Scan.Pulse]], pulse: Scan.Pulse):
            """ return True iff a left or right neighbour of this pulse has the same falling edge """
            pulse_end = pulse.start + pulse.low + pulse.high
            left = pulses[(x - 1) % len(pulses)]
            if left is not None:
                for slice_pulse in left:
                    if slice_pulse is None:
                        continue
                    left_end = slice_pulse.start + slice_pulse.low + slice_pulse.high
                    if left_end == pulse_end:
                        # its got a left neighbour
                        return True
            right = pulses[(x + 1) % len(pulses)]
            if right is not None:
                for slice_pulse in right:
                    if slice_pulse is None:
                        continue
                    right_end = slice_pulse.start + slice_pulse.low + slice_pulse.high
                    if right_end == pulse_end:
                        # its got a right neighbour
                        return True
            return False

        # build pulse lists
        pulses = [None for _ in range(len(slices))]
        for x, slice in enumerate(slices):
            if slice is None:
                continue
            # build pulse list for this slice
            slice_pulses = []
            for idx, edge in enumerate(slice):
                if idx == 0:
                    # first edge is start of a 0 run
                    pulse = Scan.Pulse(edge.where)
                    continue
                if edge.type == Scan.RISING:
                    # end of 0 run, start of 1
                    pulse.low = edge.where - pulse.start
                else:
                    # end of 1 run, start a new pulse
                    pulse.high = edge.where - pulse.start - pulse.low
                    slice_pulses.append(pulse)
                    pulse = Scan.Pulse(edge.where)
            pulses[x] = slice_pulses

        # merge short pulses
        for x, slice_pulses in enumerate(pulses):
            if slice_pulses is None:
                continue
            for idx, pulse in enumerate(slice_pulses):
                if pulse is None:
                    continue
                if pulse.high < Scan.MIN_PULSE_HIGH:
                    if has_high_neighbour(x, pulses, pulse):
                        # leave alone if it has a neighbour high in the same place
                        pass
                    elif has_low_neighbour(x, pulses, pulse):
                        # leave alone if it has a neighbour low in the same place
                        pass
                    else:
                        # too small, merge it into the low of its successor
                        slice_pulses[idx] = None
                        if (idx + 1) < len(slice_pulses):
                            successor = slice_pulses[idx + 1]
                            successor.start = pulse.start
                            successor.low += (pulse.low + pulse.high)
                        continue
                if pulse.low < Scan.MIN_PULSE_LOW:
                    if has_start_neighbour(x, pulses, pulse):
                        # leave alone if it has a neighbour in the same place
                        pass
                    else:
                        # too small, merge it into the high of its predecessor
                        slice_pulses[idx] = None
                        pred = None
                        if idx > 0:
                            # find predecessor
                            for p in range(idx-1, -1, -1):
                                if slice_pulses[p] is not None:
                                    pred = p
                                    slice_pulses[p].high += (pulse.low + pulse.high)
                                    break
                        if pred is None:
                            # no predecessor, move self up instead
                            pulse.start -= 1
                            pulse.low += 1
                            slice_pulses[idx] = pulse
                        continue

        # keep only the first pulse
        for x, slice_pulses in enumerate(pulses):
            if slice_pulses is None:
                continue
            pulses[x] = None
            if len(slice_pulses) > 0:
                for pulse in slice_pulses:
                    if pulse is not None:
                        pulses[x] = pulse
                        break

        # remove 'nipples', a 'nipple' is a pulse start, middle or end that is +/- 1 on its neighbours
        for x, pulse in enumerate(pulses):
            if pulse is None:
                continue
            left = pulses[(x - 1) % len(pulses)]
            right = pulses[(x + 1) % len(pulses)]
            if left is None or right is None:
                continue
            if left.start == right.start:
                # neighbour starts are the same
                diff = pulse.start - left.start
                if diff == 1:
                    # got a 'down' start nipple --_--
                    pulse.start = left.start
                    pulse.low += 1
                elif diff == -1:
                    # got an 'up' start nipple  __-__
                    pulse.start = left.start
                    pulse.low -= 1
            if (left.start + left.low) == (right.start + right.low):
                # neighbour rising edges are the same
                diff = (pulse.start + pulse.low) - (left.start + left.low)
                if diff == 1:
                    # got a 'down' rising nipple --_--
                    pulse.low -= 1
                    pulse.high += 1
                elif diff == -1:
                    # got an 'up' rising nipple __-__
                    pulse.low += 1
                    pulse.high -= 1
            if (left.start + left.low + left.high) == (right.start + right.low + right.high):
                # neighbour falling edges are the same
                diff = (pulse.start + pulse.low + pulse.high) - (left.start + left.low + left.high)
                if diff == 1:
                    # got a 'down' falling nipple --_--
                    pulse.high -= 1
                elif diff == -1:
                    # got an 'up' falling nipple __-__
                    pulse.high += 1

        # trim anomalous pulses (i.e. big jumps in y)
        # get average y
        y_start = 0
        y_end = 0
        y_count = 0
        for pulse in pulses:
            if pulse is not None:
                y_start += pulse.start
                y_end += (pulse.start + pulse.low + pulse.high)
                y_count += 1
        if y_count > 0:
            y_start /= y_count
            y_end /= y_count
        # drop pulses starting/ending too far from the average
        y_start_limit = y_start * Scan.MAX_START_DIFF
        y_start_limit *= y_start_limit  # square it to remove sign consideration
        y_end_limit = y_end * Scan.MAX_END_DIFF
        y_end_limit *= y_end_limit  # square it to remove sign consideration
        if self.logging:
            header = 'pulsify: dropping pulses too far from {:.2f} or {:.2f} (limits are {:.2f}, {:.2f})'.\
                     format(y_start, y_end, math.sqrt(y_start_limit), math.sqrt(y_end_limit))
        pulse_count = 0
        for x, pulse in enumerate(pulses):
            if pulse is not None:
                y_diff = pulse.start - y_start
                y_diff *= y_diff  # square it to remove sign consideration
                if y_diff > y_start_limit:
                    # this too far away from average, so drop it
                    if self.logging:
                        if header is not None:
                            self._log(header)
                            header = None
                        self._log('    {}: start of {} too far from {:.2f}: {}'.
                                  format(x, pulse.start, y_start, pulse))
                    pulses[x] = None
                    continue
                y_diff = (pulse.start + pulse.low + pulse.high) - y_end
                y_diff *= y_diff  # square it to remove sign consideration
                if y_diff > y_end_limit:
                    # this too far away from average, so drop it
                    if self.logging:
                        if header is not None:
                            self._log(header)
                            header = None
                        self._log('    {}: end of {} too far from {:.2f}: {}'.
                                  format(x, pulse.start + pulse.low + pulse.high, y_end, pulse))
                    pulses[x] = None
                    continue
                else:
                    pulse_count += 1

        if self.logging:
            self._log('pulsify: {} qualifying pulses:'.format(pulse_count))
            for x, pulse in enumerate(pulses):
                if pulse is not None:
                    self._log('    {}: {}'.format(x, pulse))

        return pulses

    def _extract(self, pulses: List[Pulse], max_x) -> List[Segment]:
        """ extract the segments from the given pulse list """

        # step 1 - get the most likely bits
        for x, pulse in enumerate(pulses):
            if pulse is None:
                # change it to an 'empty' pulse
                pulse = Scan.Pulse(0)
                pulse.bits = [[Scan.NO_BITS, 9999]]
                pulses[x] = pulse
                continue
            actual = pulse.ratio()
            pulse.bits = []
            for idx, ideal in enumerate(Scan.RATIOS):
                error = (actual - ideal) / ideal  # 0 == no error
                error *= error  # make sure its always positive
                error += 1  # make sure its >0
                error = int(error * 1000)  # only want 3 dp
                pulse.bits.append([Scan.BITS[idx], error])
            pulse.bits.sort(key=lambda b: b[1])
        if self.logging:
            self._log('extract: bits and their errors:')
            for x, pulse in enumerate(pulses):
                msg = ''
                for bits in pulse.bits:
                    msg += ', ({}, {})'.format(bits[0], bits[1])
                if msg != '':
                    msg = msg[2:]
                self._log('    {}: {}'.format(x, msg))

        # ToDo: when combining and get a bit change, if 2nd choice is nearly the same is it a change?

        # step 2 - amalgamate like bits
        segments = []
        for x, pulse in enumerate(pulses):
            if len(segments) == 0:
                # first one, start a sequence
                segments.append(Scan.Segment(x, pulse.bits[0][0], error=1 / pulse.bits[0][1]))
            elif pulse.bits[0][0] == segments[-1].bits:
                # got another the same
                segments[-1].samples += 1
                segments[-1].error += 1 / pulse.bits[0][1]
            else:
                # start of a new sequence
                segments.append(Scan.Segment(x, pulse.bits[0][0], error=1 / pulse.bits[0][1]))
        if len(segments) > 1 and segments[0].bits == segments[-1].bits:
            # got a wrapping segment
            segments[-1].samples += segments[0].samples
            segments[-1].error += segments[0].error
            del segments[0]
        for segment in segments:
            segment.error = int(round(segment.samples / segment.error))  # set harmonic mean as the error
        if self.logging:
            self._log('extract: {} segments:'.format(len(segments)))
            for segment in segments:
                self._log('    {}'.format(segment))

        # ToDo: update the error when merge/extend segments - how?

        # step 3 - drop short samples (they are noise)
        if self.logging:
            header = 'extract: dropping short segments'
        for idx, segment in enumerate(segments):
            if segment is None:
                continue
            if segment.samples < Scan.MIN_SEGMENT_LENGTH:
                segments[idx] = None  # kill this short segment
                if self.logging:
                    if header is not None:
                        self._log(header)
                        header = None
                next_segment = None
                for dx in range(1, len(segments)):
                    next_at = (idx + dx) % len(segments)
                    next_segment = segments[next_at]
                    if next_segment is not None:
                        break
                prev_segment = None
                for dx in range(1, len(segments)):
                    prev_at = (idx - dx) % len(segments)
                    prev_segment = segments[prev_at]
                    if prev_segment is not None:
                        break
                if next_segment is None or prev_segment is None:
                    # eh? this implies the segment list is only 1 long
                    if self.logging:
                        self._log('    {}: no neighbours, prev {}, next {}'.
                                  format(segment, prev_segment, next_segment))
                elif next_segment.bits == prev_segment.bits:
                    # both neighbours are same - merge them
                    prev_segment.samples += next_segment.samples + segment.samples
                    if self.logging:
                        self._log('    {}: merged {} into {}, dropping {}'.
                                  format(segment, next_segment, prev_segment, next_segment))
                    segments[next_at] = None  # kill the one we merged with
                elif next_segment.samples > 2 and prev_segment.samples > 2:
                    # neither neighbour is noise
                    if prev_segment.bits == Scan.NO_BITS:
                        # previous is empty, so merge with next
                        next_segment.samples += segment.samples
                        next_segment.start = (next_segment.start - segment.samples) % max_x
                        if self.logging:
                            self._log('    {}: added to {}'.format(segment, next_segment))
                    elif next_segment.bits == Scan.NO_BITS:
                        # next is empty, so merge with previous
                        prev_segment.samples += segment.samples
                        if self.logging:
                            self._log('    {}: added to {}'.format(segment, prev_segment))
                    elif next_segment.samples < prev_segment.samples:
                        # next is shorter, so merge with that
                        next_segment.samples += segment.samples
                        next_segment.start = (next_segment.start - segment.samples) % max_x
                        if self.logging:
                            self._log('    {}: added to {}'.format(segment, next_segment))
                    elif prev_segment.samples < next_segment.samples:
                        # previous is shorter, so merge with that
                        prev_segment.samples += segment.samples
                        if self.logging:
                            self._log('    {}: added to {}'.format(segment, prev_segment))
                    else:
                        # both same length, lengthen both
                        prev_inc = int(segment.samples / 2)
                        next_inc = segment.samples - prev_inc
                        next_segment.samples += next_inc
                        next_segment.start = (next_segment.start - next_inc) % max_x
                        prev_segment.samples += prev_inc
                        if self.logging:
                            self._log('    {}: added to {} and {}'.format(segment, prev_segment, next_segment))
                elif prev_segment.samples > 2:
                    # next is noise, lengthen previous neighbour
                    prev_segment.samples += segment.samples
                    if self.logging:
                        self._log('    {}: added to {}'.format(segment, prev_segment))
                else:
                    # both neighbours are noise, this means we have 3 in a row (at least)
                    # we know the neighbours are different to get here
                    segments[idx] = Scan.Segment(segment.start, Scan.NO_BITS)
                    if self.logging:
                        self._log('    {}: both neighbours are noise, prev {}, next {}'.
                                  format(segment, prev_segment, next_segment))
                    continue

        # step 4 - remove dropped segments
        for segment in range(len(segments) - 1, -1, -1):
            if segments[segment] is None:
                del segments[segment]

        if self.logging:
            self._log('extract: {} qualifying segments:'.format(len(segments)))
            for segment in segments:
                self._log('    {}'.format(segment))

        # ToDo: create alternatives from second choices - how?

        return segments

    def _analyse(self, segments: List[Segment], max_x):
        """ analyse the segments to extract the bit sequences for each segment,
            each segment's bits consists of one of the encoding BITS or NO_BITS
            """
        reason = None
        while len(segments) > Scan.NUM_BITS:
            # got too many - chuck something that increases 3 copies
            # ToDo: chuck something that increases 3 copies - most error?
            if self.logging:
                self._log('analyse: too many segments: got {}, only want {}'.
                          format(len(segments), Scan.NUM_BITS))
            reason = 'got {}, need {} bits'.format(len(segments), Scan.NUM_BITS)
            break
        while len(segments) < Scan.NUM_BITS:
            # haven't got enough - what we want is some second choices - how?
            # ToDo: split long sequences - exploit should be 3 copies, split NO_BITS, 2nd choices? - how?
            if self.logging:
                self._log('analyse: not enough segments: got {}, need {}'.
                          format(len(segments), Scan.NUM_BITS))
            reason = 'got {}, need {} bits'.format(len(segments), Scan.NUM_BITS)
            break

        # return the bits from the final segment mix
        bits = []
        for segment in segments:
            bits.append(segment.bits)

        if self.save_images:
            grid = self._draw_segments(segments, max_x)
            self._unload(grid, '05-segments')

        return bits, reason

    def _measure(self, pulses):
        """ get a measure of the target size by examining the slices """

        target_size = 0  # set as the most distant falling edge (i.e. end of the high)
        for pulse in pulses:
            if pulse is not None:
                outer_edge = pulse.start + pulse.low + pulse.high
                if outer_edge > target_size:
                    target_size = outer_edge

        if self.logging:
            self._log('measure: target size is {}'.format(target_size))

        return target_size

    def _find_codes(self):
        """ find the codes within each blob in our image,
            returns a list of potential targets
            """

        # find the blobs in the image
        blobs = self._find_blobs()
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
            target = self._project(self.centre_x, self.centre_y, limit_radius)
            max_x, max_y = target.size()

            # do the pulse detection
            buckets = self._threshold(target)
            slices = self._slice(buckets)
            pulses = self._pulsify(slices)

            if self.save_images:
                plot = self._draw_pulses(pulses, buckets)
                self._unload(plot, '04-pulses')

            # get target size (for relative range judgement)
            target_size = self._measure(pulses)

            # do the segment extraction
            segments = self._extract(pulses, max_x)

            # analyse segments to get the most likely bit sequences
            bits, reason = self._analyse(segments, max_x)
            if reason is not None:
                # failed - this means some constraint was not met (its already been logged)
                if self.save_images:
                    # add to reject list for labelling on the original image
                    rejects.append(Scan.Reject(self.centre_x, self.centre_y, blob_size, target_size, reason))
                continue

            targets.append(Scan.Target(self.centre_x, self.centre_y, blob_size, target_size, target, bits))

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

    def _decode_bits(self, bits):
        # in bits we have a list of bit sequences across the data rings, i.e. bits x rings
        # we need to rotate that to rings x bits to present it to our decoder
        code = [[None for _ in range(Scan.NUM_BITS)] for _ in range(Scan.NUM_DATA_RINGS)]
        for bit in range(Scan.NUM_BITS):
            rings = bits[bit]
            for ring in range(len(rings)):
                sample = rings[ring]
                code[ring][bit] = sample
        return code

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
            bits = target.bits

            code = self._decode_bits(bits)
            number, doubt, digits = self.decoder.unbuild(code)

            # add this result
            numbers.append(Scan.Detection(number, doubt, self.centre_x, self.centre_y, target_size, blob_size, digits))

            if self.logging:
                number = numbers[-1]
                self._log('decode: number {}, digits {}'.format(number.number, number.digits), number.centre_x, number.centre_y)
                # for ring in range(len(code)):
                #     self._log('    {}'.format(code[ring]), number.centre_x, number.centre_y)
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
                self._log_file = open('{}/{}.log'.format(self._log_folder, filename), 'w')
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

    def _draw_segments(self, segments: List[Segment], max_x):
        """ draw an image of the given segments """

        def draw_block(grid, start_x, end_x, start_y, ring_width, max_x, colour):
            """ draw a coloured block as directed """

            if end_x < start_x:
                # its a wrapper
                self.transform.fill(grid,
                                    (start_x, start_y),
                                    (max_x - 1, start_y + ring_width - 1),
                                    colour)
                self.transform.fill(grid,
                                    (0, start_y),
                                    (end_x, start_y + ring_width - 1),
                                    colour)
            else:
                self.transform.fill(grid,
                                    (start_x, start_y),
                                    (end_x, start_y + ring_width - 1),
                                    colour)

        def draw_segment(grid, segment, ring_width, max_x):
            """ draw the given segment """

            data_start = ring_width * 3  # 2 inner white + 1 inner black
            for ring, bit in enumerate(segment.bits):
                if bit == 1:
                    ring_start = data_start + (ring_width * ring)
                    draw_block(grid,
                               segment.start, (segment.start + segment.samples - 1) % max_x,
                               ring_start, ring_width,
                               max_x, Scan.WHITE)

        # make an empty (i.e. black) colour image to load our segments into
        ring_width = int(round((max_x / Scan.NUM_BITS)))
        max_y = ring_width * Scan.NUM_RINGS
        grid = self.original.instance()
        grid.new(max_x, max_y, MIN_LUMINANCE)
        grid.incolour()

        # draw the inner white rings (the 'bullseye')
        draw_block(grid, 0, max_x - 1, 0, ring_width * 2, max_x, Scan.WHITE)

        # draw the segments
        for segment in segments:
            draw_segment(grid, segment, ring_width, max_x)

        return grid

    def _draw_pulses(self, pulses, buckets):
        """ draw the given pulses on the given buckets image """

        # draw pulse low as a green line, high as a blue line, none as red
        # build lines
        max_x, max_y = buckets.size()
        low_lines = []
        high_lines = []
        none_lines = []
        for x, pulse in enumerate(pulses):
            if pulse is None:
                none_lines.append((x, 0, x, max_y - 1))
            else:
                low_lines.append((x, pulse.start, x, pulse.start + pulse.low - 1))
                high_lines.append((x, pulse.start + pulse.low, x, pulse.start + pulse.low + pulse.high - 1))
        # draw lines on the bucketised image
        plot = buckets
        plot = self._draw_lines(plot, none_lines, colour=Scan.RED)
        plot = self._draw_lines(plot, low_lines, colour=Scan.GREEN)
        plot = self._draw_lines(plot, high_lines, colour=Scan.BLUE)

        return plot


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
        self.num_rings = Ring.NUM_RINGS
        self.bits = Codec.DIGITS
        self.angles = None
        self.video_mode = None
        self.contrast = None
        self.offset = None
        self.debug_mode = None
        self.log_folder = log
        self.log_file = None
        self._log('')
        self._log('******************')
        self._log('Rings: {}, Total bits: {}'.format(self.num_rings, self.bits))

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
                    rotation = random.randrange(0, self.bits - 1)
                    for ring in range(len(rings)):
                        rings[ring] = rotate(rings[ring], rotation, self.bits)
                    samples = [[] for _ in range(len(rings))]
                    for ring in range(len(rings)):
                        word = rings[ring]
                        for bit in range(self.bits):
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
        frm_bin = '{:0' + str(self.bits) + 'b}'
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
            image_width = width * (self.num_rings + 1) * 2  # rings +1 for the border
            self.frame.new(image_width, image_width, MID_LUMINANCE)
            x, y = self.frame.size()
            ring = Ring(x >> 1, y >> 1, width, self.frame, self.contrast, self.offset)
            bits = Codec.ENCODING + [Codec.NOTHING]
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
                    image_width = width * (self.num_rings + 1) * 2  # rings +1 for the border
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

    def _remove_test_codes(self, folder, pattern):
        """ remove all the test code images with file names containing the given pattern in the given folder
            """
        filelist = glob.glob('{}/*{}*.*'.format(folder, pattern))
        for f in filelist:
            try:
                os.remove(f)
            except:
                traceback.print_exc()
                self._log('Could not remove {}'.format(f))

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
    test_ring_width = 28  # this makes a target that fits on A5

    # cell size is critical,
    # going small in length creates edges that are steep vertically, going more takes too long
    # going small in height creates edges that are too small and easily confused with noise
    test_scan_cells = (8, 4)

    # reducing the resolution means targets have to be closer to be detected,
    # increasing it takes longer to process, most modern smartphones can do 4K at 30fps, 2K is good enough
    test_scan_video_mode = Scan.VIDEO_2K

    test_debug_mode = Scan.DEBUG_IMAGE
    #test_debug_mode = Scan.DEBUG_VERBOSE

    # setup test params
    test = Test(log=test_log_folder)
    test.encoder(min_num, max_num)
    test.options(cells=test_scan_cells,
                 mode=test_scan_video_mode,
                 contrast=contrast,
                 offset=offset,
                 debug=test_debug_mode)

    # build a test code set
    test_num_set = test.test_set(20, [111, 222, 333, 444, 555, 666])

    # test.coding()
    # test.decoding()
    # test.circles()
    # test.code_words(test_num_set)
    # test.codes(test_codes_folder, test_num_set, test_ring_width)
    # test.rings(test_codes_folder, test_ring_width)  # must be after test.codes (else it gets deleted)

    # test.scan_codes(test_codes_folder)
    # test.scan_media(test_media_folder)

    # test.scan(test_codes_folder, [000], 'test-code-000.png')
    # test.scan(test_codes_folder, [444], 'test-code-444.png')

    # test.scan(test_media_folder, [182], 'photo-182.jpg')
    test.scan(test_media_folder, [101, 102, 182, 247, 301, 424, 448, 500, 537, 565], 'photo-101-102-182-247-301-424-448-500-537-565-v1.jpg')

    del (test)  # needed to close the log file(s)


if __name__ == "__main__":
    verify()
