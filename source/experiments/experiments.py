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
        enclosed by a solid 'black' ring of width 1R and, 
        finally, a solid 'white' ring of radius 1R. 
    Total radius is 8R.
    The 3 data rings are used as a triple redundant data bit copy, each consists of a start sync pattern (so bit
    0 can be distinguished in any sequence of bits), and payload bits, each ring is bit shifted right by N bits,
    and the code word also is constrained such that there is a bit transition for every bit across the data rings,
    this in turn ensures bit boundaries can always be detected for every bit, each data ring may be XOR'd with a
    bit pattern (used to invert the middle ring).
    A one-bit is white (i.e. high luminance) and a zero-bit is black (i.e. low luminance).
    The start sync pattern is 0110 (4 bits). The remaining bits are the payload (big-endian) and must not contain
    the start sync pattern and also must not end in 011 and must not start with 110 (else they look like 0110 when
    adjacent to the alignment marker).
    The inner white-to-black ring transition and the outer black-to-white ring transition are used to detect the
    limits of the code in the image. 
    The central 'bullseye' candidates are detected using a 'blob detector' (via opencv), then the area around
    that is polar to cartesian 'warped' (via opencv) into a rectangle. All further processing is on that rectangle.
    
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
    """ Encode and decode a number or a bit or a blob
        a number is a payload, it can be encoded and decoded
        a bit is a raw bit decoded from 3 blobs
        a blob is decoded from 3 luminance level samples
        this class encapsulates all the encoding and decoding and their constants
        """

    BITS = 15  # total bits in the code
    MARKER_BITS = 4  # number of bits in our alignment marker
    MARKER_PATTERN = 0b_0110  # the bit 0 sync pattern
    MIN_ACROSS_RING_EDGES = 5  # min bit transitions across the rings

    # ring XOR masks (-1 just inverts the whole ring)
    INNER_MASK = 0
    MIDDLE_MASK = -1
    OUTER_MASK = 0

    # bit value categories
    BLACK = 0
    WHITE = 1
    AMBIGUOUS = None

    # bit value categories
    IS_ZERO = 0
    IS_ONE = 1
    LIKELY_ZERO = 2
    LIKELY_ONE = 3
    MAYBE_EITHER = 4
    UNKNOWN = 5

    def __init__(self, min_num, max_num):
        """ create the valid code set for a number in the range min_num..max_num for code_size
            a valid code is one where there are no embedded start/stop bits bits but contains at least one 1 bit,
            we create two arrays - one maps a number to its code and the other maps a code to its number.
            """

        # setup, and verify, code characteristics
        self.skew = int(Codec.BITS / 3)  # ring to ring skew in bits
        self.code_bits = Codec.BITS - Codec.MARKER_BITS  # code word bits is what's left
        if self.code_bits < 1:
            raise Exception('BITS must be greater than {}, {} is not'.format(Codec.MARKER_BITS, Codec.BITS))

        self.marker = Codec.MARKER_PATTERN << self.code_bits  # marker in MS n bits of code
        self.code_range = 1 << self.code_bits  # max code range before constraints applied
        self.bit_mask = (1 << Codec.BITS) - 1  # mask to isolate all bits

        # generate the bit limited version of our XOR masks
        self.inner_mask = Codec.INNER_MASK & self.bit_mask
        self.middle_mask = Codec.MIDDLE_MASK & self.bit_mask
        self.outer_mask = Codec.OUTER_MASK & self.bit_mask

        # params
        self.min_num = min_num  # minimum number we want to be able to encode
        self.max_num = max_num  # maximum number we want to be able to encode

        # build code tables
        self.codes = [None for _ in range(self.code_range)]  # set all invalid initially
        self.nums = [None for _ in range(self.max_num + 1)]  # ..
        self.min_edges = Codec.BITS  # minimum edges around the ring of any valid code word
        self.max_edges = 0           # maximum edges around the ring of any valid code word
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
                    # got a potential code, check if it meets our edges requirement
                    edges = self.allowable(code)
                    if edges > 0:
                        # got enough edges, give it to next number
                        num += 1
                        if num > self.max_num:
                            # found enough, don't use this one
                            pass
                        else:
                            self.codes[code] = num  # decode code as num
                            self.nums[num] = code  # encode num as code
                            if edges > self.max_edges:
                                self.max_edges = edges
                            if edges < self.max_edges:
                                self.min_edges = edges
        if num < self.min_num:
            # nothing meets criteria!
            self.num_limit = None
        else:
            self.num_limit = num

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
        shift = (Codec.BITS - (self.skew * 3)) + (self.skew * 2)
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

    def allowable(self, code_word):
        """ given a code word return True if its allowable as a code word,
            codes are not allowed if they do not meet our bit transition requirements around or across the rings,
            the requirement is that at least one bit transition must exist for every bit within the rings and
            at least 5 bit transition across the rings, this ensures all bit transitions can be detected,
            returns 0 if the code word is not allowed or the number of bit transitions if it is
            """

        msb = 1 << (Codec.BITS - 1)  # 1 in the MSB of a code word
        all_ones = (1 << Codec.BITS) - 1  # all 1's in the relevant bits
        rings = self.rings(code_word)
        # check meets edges requirements around the rings (across all rings must be an edge for each bit)
        all_edges = 0
        for ring in rings:
            ring_edges = 0
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
                    ring_edges += 1
                mask1 >>= 1
                mask2 >>= 1
        missing = (all_edges ^ all_ones)
        if missing != 0:
            # does not meet edges requirement around the rings
            return 0

        # check meets edge requirements across the rings
        if count_bits(rings[0] ^ rings[1]) < Codec.MIN_ACROSS_RING_EDGES:
            # does not meet edges requirement across the rings
            return 0
        if count_bits(rings[1] ^ rings[2]) < Codec.MIN_ACROSS_RING_EDGES:
            # does not meet edges requirement across the rings
            return 0

        return max(ring_edges, 1)

    def ring_bits_pos(self, n):
        """ given a bit index return a list of the indices of all the same bits from each ring """
        n1 = n
        n2 = int((n1 + self.skew) % Codec.BITS)
        n3 = int((n2 + self.skew) % Codec.BITS)
        return [n1, n2, n3]

    def marker_bits_pos(self, n):
        """ given a bit index return a list of the indices of all the bits that would make a marker """
        return [int(pos % Codec.BITS) for pos in range(n, n + Codec.MARKER_BITS)]

    def unbuild(self, samples):
        """ given an array of 3 code-word rings with random alignment return the encoded number or None,
            each ring must be given as an array of bit values in bit number order,
            returns the number (or None), the level of doubt and the bit classification for each bit,
            """

        # step 1 - decode the 3 rings bits
        bits = [None for _ in range(Codec.BITS)]
        for n in range(Codec.BITS):
            rings = self.ring_bits_pos(n)
            bit_mask = 1 << (Codec.BITS - (n + 1))
            s1_mask = (self.inner_mask & bit_mask)
            s2_mask = (self.middle_mask & bit_mask)
            s3_mask = (self.outer_mask & bit_mask)
            s1 = samples[0][rings[0]]
            s2 = samples[1][rings[1]]
            s3 = samples[2][rings[2]]
            bits[n] = self.bit(s1, s1_mask, s2, s2_mask, s3, s3_mask)

        # step 2 - find the alignment marker candidates
        maybe_at = [[] for _ in range(4*5)]  # 0..max maybe possibilities, worst case is 4 missing (at 5 each)
        for n in range(Codec.BITS):
            marker = self.is_marker(n, bits)
            if marker is None:
                continue  # no marker at this bit position, look at next
            # got a potential marker with marker maybe values, 0 == exact, 4 == all maybe
            maybe_at[marker].append(n)
        # maybe_at now contains a list of all possibilities for all maybe options
        # the array is organised such that the best choice is first

        # step 3 - extract all potential code words for each candidate alignment for each maybe level
        # any that yield more than one are crap and a give-up condition
        found = None
        for maybe in maybe_at:           # NB: must consider in array index order (best choice first)
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
        """ given a set of categorised bits, count how many digits have errors,
            the bigger this number the more doubt there is in the validity of the code,
            returns the total doubt for all digits
            """
        doubt = 0
        errors = [0,  # IS_ZERO
                  0,  # IS_ONE
                  1,  # LIKELY_ZERO
                  1,  # LIKELY_ONE
                  2,  # MAYBE_EITHER
                  4]  # UNKNOWN
        for bit in bits:
            doubt += errors[bit]
        return doubt

    def _show_bits(self, bits):
        """ given an array of bit classifications return that in a readable CSV format
            """
        symbols = {self.IS_ONE: '111',
                   self.IS_ZERO: '000',
                   self.LIKELY_ONE: '11?',
                   self.LIKELY_ZERO: '00?',
                   self.MAYBE_EITHER: '0?1',
                   self.UNKNOWN: '???'}
        csv = ''
        for bit in bits:
            csv += ',' + symbols[bit]
        return csv[1:]

    def is_marker(self, n, bits):
        """ given a set of bits and a bit position check if an alignment marker is present there
            the function returns a doubt level (0==exact match, >0 increasing levels of doubt, None=no match)
            """
        exact = 0
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
        else:
            missing += 1
        if b2 == self.IS_ONE:
            exact += 1
        elif b2 == self.LIKELY_ONE:
            likely += 1
        else:
            missing += 1
        if b3 == self.IS_ONE:
            exact += 1
        elif b3 == self.LIKELY_ONE:
            likely += 1
        else:
            missing += 1
        if b4 == self.IS_ZERO:
            exact += 1
        elif b4 == self.LIKELY_ZERO:
            likely += 1
        else:
            missing += 1
        if missing > 1:
            # too much to tolerate
            return None
        # NB: if exact is 4 then likely and missing are both 0
        return likely + (5 * missing)    # 1 missing is worse than 4 likely

    def data_bits(self, n, bits):
        """ return an array of the data-bits from bits array starting at bit position n,
            this is effectively rotating the bits array and removing the marker bits such
            that the result is an array with [0] the first data bit and [n] the last
            """
        return [bits[int(pos % Codec.BITS)] for pos in range(n + Codec.MARKER_BITS, n + Codec.BITS)]

    def extract_word(self, n, bits):
        """ given an array of bit values with the alignment marker at position n
            extract the code word and decode it (via decode()), returns None if cannot
            """
        word = self.data_bits(n, bits)
        code = 0
        for bit in range(len(word)):
            code <<= 1  # make room for next bit
            val = word[bit]
            if (val == self.IS_ONE) or (val == self.LIKELY_ONE):
                code += 1  # got a one bit
            elif (val == self.IS_ZERO) or (val == self.LIKELY_ZERO):
                pass  # got a zero bit
            else:
                return None  # got junk
        return self.decode(code)

    def classify(self, sample, invert):
        """ given a bit value and its inversion mask determine its classification,
            the returned bit classification is one of IS_ZERO, IS_ONE, MAYBE_EITHER or UNKNOWN
            """

        if invert != 0:
            if sample == Codec.BLACK:
                return Codec.IS_ONE
            elif sample == Codec.WHITE:
                return Codec.IS_ZERO
            elif sample == Codec.AMBIGUOUS:
                return Codec.MAYBE_EITHER
        else:
            if sample == Codec.BLACK:
                return Codec.IS_ZERO
            elif sample == Codec.WHITE:
                return Codec.IS_ONE
            elif sample == Codec.AMBIGUOUS:
                return Codec.MAYBE_EITHER

        # anything else is junk (means caller gave us samples that were not 0 or 1 or None)
        return self.UNKNOWN

    def bit(self, s1, m1, s2, m2, s3, m3):
        """ given 3 bit values and their inversion masks determine the most likely overall bit value,
            bit values must be 'black' or 'white',
            the return bit is one of 'is' or 'likely' one or zero or neither
            """

        zeroes = 0
        ones = 0
        either = 0

        c1 = self.classify(s1, m1)
        c2 = self.classify(s2, m2)
        c3 = self.classify(s3, m3)

        if c1 == Codec.IS_ZERO:
            zeroes += 1
        elif c1 == Codec.IS_ONE:
            ones += 1
        elif c1 == Codec.MAYBE_EITHER:
            either += 1

        if c2 == Codec.IS_ZERO:
            zeroes += 1
        elif c2 == Codec.IS_ONE:
            ones += 1
        elif c2 == Codec.MAYBE_EITHER:
            either += 1

        if c3 == Codec.IS_ZERO:
            zeroes += 1
        elif c3 == Codec.IS_ONE:
            ones += 1
        elif c3 == Codec.MAYBE_EITHER:
            either += 1

        # test ideal cases
        if zeroes == 3:
            return self.IS_ZERO
        elif ones == 3:
            return self.IS_ONE

        # if zeroes == 2 and either == 1:
        #     return self.IS_ZERO
        # elif ones == 2 and either == 1:
        #     return self.IS_ONE

        # test likely cases
        if zeroes == 2:
            return self.LIKELY_ZERO
        elif ones == 2:
            return self.LIKELY_ONE

        # # test maybe cases
        # if zeroes == 1 and either == 2:
        #     return self.LIKELY_ZERO
        # elif ones == 1 and either == 2:
        #     return self.LIKELY_ONE

        # the rest are junk
        return self.UNKNOWN


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

    NUM_RINGS = 8  # total rings in our complete code

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

        # draw the bullseye and its enclosing ring
        self.draw(0.0, -1, 2.0)
        self.draw(2.0, 0, 1.0)
        draw_at = 3.0

        # draw the data rings
        for ring in rings:
            self.draw(draw_at, ring, 1.0)
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
            3. find the inner and outer edges in that
            4. 'flatten' the image such that the inner edge is straight
            5. edge detect all the ring and bit boundaries
            6. extract the cells for each bit
            7. decode those bits
        the algorithm has been developed more by experimentation than theory!
        """

    # region Constants...
    # our target shape
    NUM_RINGS = Ring.NUM_RINGS  # total number of rings in the whole code
    NUM_DATA_RINGS = 3  # how many data rings in our codes
    NUM_BITS = Codec.BITS  # total number of bits in a ring

    # image 'cell' constraints (a 'cell' is the area of the flattened image that contains a single bit)
    # these constraints set minimums that override the cells() property given to Scan
    MIN_PIXELS_PER_BIT = 4  # pixels making up a bit length
    MIN_PIXELS_PER_RING = 4  # pixels making up a ring height

    # region Tuning constants...
    MIN_BLOB_SEPARATION = 5  # smallest blob within this distance of each other are dropped
    BLOB_RADIUS_STRETCH = 4  # how much to stretch blob radius to ensure always cover the whole lot
    RING_RADIUS_STRETCH = 2.0  # how much to stretch nominal ring width to ensure encompass the outer edge
    LARGE_IMAGE_LIMIT = 4  # ratio over expected projected height to consider as a 'large' image
    MAX_NEIGHBOUR_GAP = 1  # maximum pixel gap in y between successive x's for two y's to be considered as neighbours
    RADIUS_PROBE_MIN_LENGTH = 1/15  # min length of an edge fragment as a fraction of the max
    RADIUS_PROBE_MIN_PIXELS = 3  # if the above is less than this pixels, use this min length in pixels
    MAX_NEIGHBOURS = 2  # max numbers of neighbours to probe from fragment ends
    MAX_RADIUS_EDGE_GAP = 1/8  # max gap/distance between radius edge fragments in x as a fraction of the max
    MAX_FULL_EDGES = 8  # stop looking for fragment joins when found this many full length ones
    MIN_PIXELS_PER_RING_PROJECTED = 4  # project an image such that at least this many pixels per ring at the outer edge
    MIN_PIXELS_PER_RING_MEASURED = 1  # min number of pixels per ring after measuring target extent
    LUMINANCE_SLOPE_THRESHOLD = 0.9  # change to qualify a slope change as a fraction of the threshold, must be < 1
    MIN_EDGE_TO_EDGE_SPAN = 0.3  # min length of a ring edge-to-edge span as a fraction of the nominal ring width
    MIN_EDGE_TO_EDGE_PIXELS = 3  # minimum pixels if the above is below this
    MIN_BIT_TO_BIT_SPAN = 0.6  # min length of a bit edge-to-edge span as a fraction of the nominal bit length
    MIN_BIT_TO_BIT_PIXELS = 3  # minimum pixels if the above is below this
    MAX_BIT_SEQUENCE_LENGTH = 2.0  # max length of a bit sequence as a multiple of the nominal bit length
    MIN_BIT_SEQUENCE_LENGTH = 1/3  # min length of a bit sequence as a multiple of the nominal bit length
    BLACK_THRESHOLD = 0.3  # max white samples for bit to be considered as black as a ratio of samples
    WHITE_THRESHOLD = 0.7  # min white samples for bit to be considered as white as a ratio of samples

    # region CONNECTED_KERNEL
    # x,y pairs for neighbours when looking for inner/outer radius edges,
    # the first x,y pair *must* be 0,0 (this constraint is required for edge split detection)
    # the connected kernel controls the Y spacing for a pixel neighbour to be considered as connected,
    # this should be small (i.e. +/- 1) because we want Y jumps to terminate an edge as it probably
    # represents a choice point for joining fragments, the kernel height controls what is considered
    # to be 'close' when joining fragments, this should be bigger so that more fragments are joined and
    # thereby reduce the search space when joining (long) fragments into edges
    CONNECTED_KERNEL = [[0, 0], [0, 1], [0, -1]]
    CONNECTED_KERNEL_HEIGHT = 2  # the max Y span of the kernel from edge-to-edge
    # endregion

    # region ISLAND_KERNEL
    # this is used to detect single pixel islands
    # an island is a pixel surrounded by pixels all the same colour
    ISLAND_KERNEL = [(-1, -1), (0, -1), (1, -1),
                     (-1, 0), (1, 0),
                     (-1, 1), (0, 1), (1, 1)]
    # endregion

    # edge vector smoothing kernel, pairs of offset and scale factor (see _smooth_edge)
    SMOOTHING_KERNEL = [[-4, 1.25], [-3, 1], [-2, 0.75], [-1, 0.5], [0, 0], [+1, 0.5], [+2, 0.75], [+3, 1], [+4, 1.25]]

    # min width of a ring as a fraction of the nominal ring width
    MIN_EDGE_WIDTH = 0.1
    MIN_EDGE_WIDTH_PIXELS = 2  # minimum width in pixels if the above is less than this
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

    # region Scanning directions...
    # these are bit masks, so they can be added to identify multiple directions
    TOP_DOWN = 1  # top-down y scan direction      (looking for radius edges or ring edges)
    BOTTOM_UP = 2  # bottom-up y scan direction     (..)
    LEFT_TO_RIGHT = 4  # left-to-right x scan direction (looking for bit edges)
    RIGHT_TO_LEFT = 8  # right-to-left x scan direction (..)

    # useful multi-directions
    UP_AND_DOWN = TOP_DOWN + BOTTOM_UP
    LEFT_AND_RIGHT = LEFT_TO_RIGHT + RIGHT_TO_LEFT
    # endregion

    # context type identifiers
    CONTEXT_RING = 0
    CONTEXT_BIT = 1

    # edge type identifiers
    LEADING_EDGE = 'leading'  # black-to-white transition
    TRAILING_EDGE = 'trailing'  # white-to-black transition

    # region Luminance level bands...
    # they must sum to less than 1
    # entries in here represent the 'buckets' the target image luminance levels are translated into
    # as a fraction of the luminance range of an image slice,
    # the first bucket is the black threshold, the last is the white threshold,
    # the white band is implied as 1-(sum of the rest)

    # these are the levels used when detecting inner/outer edges
    # the objective when detecting inner/outer edges is to find a continous contour through all X's
    # these thresholds divide the image into N buckets, every luminance bucket change is a potential
    # edge, so more buckets is more edges and more opportunities to find a contour, every combination
    # is explored, so more buckets increases the search space (so takes longer), conventional edge
    # detection does not work as well as this bucket approach and has its own complexity issues...
    # these levels are not linear as we want to control the bucket sizes to discriminate small changes
    # near the extremes
    EDGE_THRESHOLD_LEVELS = (0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125)

    # these are the levels used when translating an image into 'buckets'
    # it creates 4 buckets which are interpreted as black, maybe-black, maybe-white or white in _compress
    BUCKET_THRESHOLD_LEVELS = (0.25, 0.25, 0.25)
    # endregion

    # region Fragment end reasons...
    WRAPPED = 'wrapped'
    MERGED = 'merged'
    SPLIT = 'split'
    SPLITEND = 'split-end'
    JUMPED = 'jumped'
    JOINED = 'joined'
    DEAD = 'dead'
    ENDED = 'ended'
    # endregion

    # region Edge state codes...
    HEADJOIN = 'head-join'
    TAILJOIN = 'tail-join'
    COMPLETE = 'complete'
    CONSIDERED = 'considered'
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
    # head to top to tail relative sizes
    THREE_ONE_ONE = '3:1:1'
    TWO_ONE_TWO = '2:1:2'
    TWO_TWO_ONE = '2:2:1'
    ONE_ONE_ONE = '1:1:1'
    ONE_ONE_THREE = '1:1:3'
    ONE_TWO_TWO = '1:2:2'
    ONE_THREE_ONE = '1:3:1'

    # pulse ratios for the above (see diagram in _measure_pulses)
    THREE_ONE_ONE_RATIOS = ((3, 1, 1),)  # (3, 1, 0.5))
    TWO_ONE_TWO_RATIOS = ((2, 1, 2),)  # (2, 1, 1.5))
    TWO_TWO_ONE_RATIOS = ((2, 2, 1),)  # (2, 2, 0.5))
    ONE_ONE_THREE_RATIOS = ((1, 1, 3),)  # (1, 1, 2.5))
    ONE_ONE_ONE_RATIOS = ((1, 1, 1),)  # (1, 1, 0.5))
    ONE_TWO_TWO_RATIOS = ((1, 2, 2),)  # (1, 2, 1.5))
    ONE_THREE_ONE_RATIOS = ((1, 3, 1),)  # (1, 3, 0.5))
    # endregion

    # slice dead reasons
    SINGLE_SAMPLE = 'singleton'

    # region Slice bit dead reasons...
    NOISE_BIT = 'noise'
    TOO_SHORT = 'too-short'
    TOO_LONG = 'too-long'
    WORSE_ERROR = 'worse-error'
    SHORTER = 'shorter'
    ONLY_CHOICE = 'only-choice'
    SHORTEST = 'shortest'
    WORST_ERROR = 'worst-error'
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
    class Stepper:
        """ this class is used to step through an image in a direction between min/max limits,
            use of this class unifies any scanning loops in any direction and handles all wrapping
            and cropping issues, x wraps, y does not
            """

        x = None  # current x coordinate, these wrap at image edge
        y = None  # current y coordinate, these do not wrap at image edge
        x_multiplier = None  # the x coord multiplier for the current direction
        y_multiplier = None  # the y coord multiplier for the current direction
        min_x = None  # the image scan start in x
        max_x = None  # the image scan end in x
        min_y = None  # the image scan start in y
        max_y = None  # the image scan end in y
        step = None  # how much the co-ordinates are stepping by for each increment
        steps = None  # the maximum number of steps to do
        iterations = None  # how many steps have been done so far

        def __init__(self, direction, max_x, max_y, step=1, steps=None, min_x=0, min_y=0):
            self.min_x = min_x
            self.max_x = max_x
            self.min_y = min_y
            self.max_y = max_y
            self.step = step
            self.iterations = 0
            self.steps = steps
            self.x = self.min_x
            self.y = self.min_y
            self.x_multiplier = 0
            self.y_multiplier = 0
            if (direction & Scan.TOP_DOWN) != 0:
                if self.steps is None:
                    self.steps = int(round((self.max_y - self.min_y) / self.step))
                self.y_multiplier = +1
            elif (direction & Scan.BOTTOM_UP) != 0:
                if self.steps is None:
                    self.steps = int(round((self.max_y - self.min_y) / self.step))
                self.y = self.max_y - 1
                self.y_multiplier = -1
            elif (direction & Scan.LEFT_TO_RIGHT) != 0:
                if self.steps is None:
                    self.steps = int(round((self.max_x - self.min_x) / self.step))
                self.x_multiplier = +1
            elif (direction & Scan.RIGHT_TO_LEFT) != 0:
                if self.steps is None:
                    self.steps = int(round((self.max_x - self.min_x) / self.step))
                self.x = self.max_x - 1
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

        def cross_to(self, dest):
            """ move the non-iterating co-ordinate to that given,
                it will not allow the co-ord to exceed the min/max limits,
                returns an xy tuple to reflect the updated position
                """

            if self.x_multiplier != 0:
                # non-iterating co-ord is y
                if dest >= self.min_y and dest <= self.max_y:
                    self.y = dest

            elif self.y_multiplier != 0:
                # non-iterating co-ord is x
                if dest >= self.min_x and dest <= self.max_x:
                    self.x = dest

            return [self.x, self.y]

        def skip_to(self, dest):
            """ skip steps until the next step will yield that given or None,
                it returns the x,y tuple reached or None if beyond the end,
                """

            xy = [self.x, self.y]

            if self.x_multiplier != 0:
                while self.x != dest:
                    xy = self.next()
                    if xy is None:
                        break
            elif self.y_multiplier != 0 and self.y is not None:
                while self.y != dest:
                    xy = self.next()
                    if xy is None:
                        break

            return xy

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
            self.x = int(round(self.x + (self.x_multiplier * self.step)))
            if self.x >= self.max_x:
                # gone off the right edge, re-start on the left edge
                self.x = self.min_x
            elif self.x < self.min_x:
                # gone off the left edge, re-start on the right edge
                self.x = self.max_x - 1
            self.y = int(round(self.y + (self.y_multiplier * self.step)))
            if self.y < self.min_y or self.y >= self.max_y:
                # we've gone off the top or bottom of the image
                # so next step must report done
                self.y = None

            # note we've done another step
            self.iterations += 1

            return xy

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

    class EdgePoint:
        """ struct to hold info about an edge point """

        def __init__(self, midpoint=0, where=0, first=0, last=0, pixel=None):
            # NB: the co-ordinates given here may be out of range if they have wrapped,
            #     it is up to the user of this object to deal with it
            self.midpoint = midpoint  # midpoint co-ord (Y)
            self.where = where  # the other co-ord of the point (X)
            self.first = first  # first pixel co-ord over the threshold in this edge
            self.last = last  # last pixel co-ord over the threshold in this edge
            self.pixel = pixel  # the pixel value at the midpoint

        def __str__(self):
            return '(at {},{} span={}..{} pixel={})'.\
                   format(self.where, self.midpoint, self.first, self.last, self.pixel)

        def __eq__(self, other):
            return self.midpoint == other.midpoint and self.where == other.where

    class Kernel:
        """ an iterator that returns a series of x,y co-ordinates for a 'kernel' in a given direction,
            direction is an 'angle' with 'left-to-right' being the natural angle for the kernel, and
            other directions rotating clockwise in 90 degree steps, through top-down, right-to-left
            and bottom-up
            """

        def __init__(self, kernel, direction):

            self.kernel = kernel

            if direction == Scan.LEFT_TO_RIGHT:  # 0 degrees
                self.delta_x = 1
                self.delta_y = 1
                self.swapped = False
            elif direction == Scan.TOP_DOWN:  # 90 degrees
                self.delta_x = -1
                self.delta_y = 1
                self.swapped = True
            elif direction == Scan.RIGHT_TO_LEFT:  # 180 degrees
                self.delta_x = -1
                self.delta_y = -1
                self.swapped = False
            elif direction == Scan.BOTTOM_UP:  # 270 degrees
                self.delta_x = 1
                self.delta_y = -1
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

        # image and its limits
        target = None
        max_x = None
        max_y = None
        mergers = None  # see _get_mergee

        # co-ordinate limits, scan-direction is the main direction, cross-direction is 90 degrees to it
        # these are either max_x-1 or max_y-1
        max_scan_coord = None
        max_cross_coord = None

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

        # wrapping constraints
        allow_scan_wrap = None  # scanning is allowed to wrap around image limits (e.g. bits)

        # kernel matrix to use to detect connected neighbours when following edges
        # this is set in processing logic, not here as it is dependant on more than just the direction
        kernel = None

        # scan direction
        scan_direction = None

        # used when detecting/joining fragments
        min_length = None  # the minimum useful length of a fragment, those below this are discarded
        max_gap = None  # when joining *fragments* the square of the max gap to be considered as a direct connection

        max_edge_x_gap = None  # the maximum linear X gap allowed between fragments when constructing edges
        max_edge_gap = None  # square of the max distance allowed between fragments when constructing edges

        def __init__(self, target, direction):

            self.target = target
            self.max_x, self.max_y = target.size()

            self.min_length = int(max(round(self.max_x * Scan.RADIUS_PROBE_MIN_LENGTH), Scan.RADIUS_PROBE_MIN_PIXELS))
            self.max_gap = Scan.CONNECTED_KERNEL_HEIGHT
            self.max_gap *= self.max_gap
            self.max_gap += 1  # this is to allow for the get_gap function treating X to X+1 as a length of 1

            self.max_edge_x_gap = int(self.max_x * Scan.MAX_RADIUS_EDGE_GAP)
            self.max_edge_gap = self.max_edge_x_gap * self.max_edge_x_gap

            if (direction & Scan.UP_AND_DOWN) != 0:
                # stuff common to TOP_DOWN and BOTTOM_UP scanning (ring and radius edges)

                self.context = Scan.CONTEXT_RING
                self.type = 'ring'

                # coordinate limits
                self.max_scan_coord = self.max_y - 1
                self.max_cross_coord = self.max_x - 1

                # indices into a [x,y] tuple for the scanning co-ord and the cross co-ord
                self.scan_coord = 1  # ==y
                self.cross_coord = 0  # ==x

                # wrapping constraints
                self.allow_scan_wrap = False

            elif (direction & Scan.LEFT_AND_RIGHT) != 0:
                # stuff common to left-to-right and right-to-left scanning (bit edges)

                self.context = Scan.CONTEXT_BIT
                self.type = 'bit'

                # coordinate limits
                self.max_scan_coord = self.max_x - 1
                self.max_cross_coord = self.max_y - 1

                # indices into a [x,y] tuple for the scanning co-ord and the cross co-ord
                self.scan_coord = 0  # ==x
                self.cross_coord = 1  # ==y

                # wrapping constraints
                self.allow_scan_wrap = True

            else:
                raise Exception('illegal direction {}'.format(direction))

            if (direction & Scan.TOP_DOWN) != 0:
                # NB: This takes priority if doing up-and-down
                # context is looking for ring or radius edges top-down
                self.prefix = 'ring-down'
                self.scan_direction = Scan.TOP_DOWN
                # multiplier when scanning
                self.scan_multiplier = 1

            elif (direction & Scan.BOTTOM_UP) != 0:
                # NB: This is ignored if doing up-and-down
                # context is looking for ring or radius edges bottom-up
                self.prefix = 'ring-up'
                self.scan_direction = Scan.BOTTOM_UP
                # multiplier when scanning
                self.scan_multiplier = -1

            elif (direction & Scan.LEFT_TO_RIGHT) != 0:
                # NB: This takes priority if doing left-and-right
                # context is looking for bit edges left-to-right
                self.prefix = 'bit-left'
                self.scan_direction = Scan.LEFT_TO_RIGHT
                # multiplier when scanning
                self.scan_multiplier = 1

            elif (direction & Scan.RIGHT_TO_LEFT) != 0:
                # NB: This is ignored if doing left-and-right
                # context is looking for bit edges right-to-left
                self.prefix = 'bit-right'
                self.scan_direction = Scan.RIGHT_TO_LEFT
                # multiplier when scanning
                self.scan_multiplier = -1

            else:
                # can't get here!
                raise Exception('unreachable code reached!')

    class Boundary:
        """ a boundary is a horizontal or vertical line that represents a ring or bit transition,
            these lines have a 'thickness' - a start and an end coord, these will be the same unless
            an edge has been merged, for merged edges they are the inner and outer coord limits of
            the merge, this area is a no-mans-land when the bit pixels are analysed
            """

        def __init__(self, where, leading_edges, trailing_edges, extent=None):
            self.where = where  # the start x (or y) co-ord of the boundary
            self.extent = self.where if extent is None else extent
            self.leading_edges = leading_edges  # the count of leading edges of this boundary
            self.trailing_edges = trailing_edges  # the count of trailing edges of this boundary

        def __str__(self):
            return '(at {}..{}: leading={}, trailing={})'.\
                    format(self.where, self.extent, self.leading_edges, self.trailing_edges)

    class Slice:
        """ a slice is a vertical (or horizontal) slice through the image 1 pixel wide consisting of SlicePoints
            """

        def __init__(self):
            self.points: List[Scan.SlicePoint] = []  # points in this slice
            self.bits = []  # the bit sequence represented by this slice
            self.pulses = []  # how many pulses in the slice
            self.error = 0  # our confidence factor (0..1) in the bits decoded (0=total, 1=none)
            self.dead = None  # when not None the reason this slice was killed

        def __str__(self):
            if len(self.points) > 0:
                where = '{}'.format(self.points[0])
            else:
                where = 'nowhere'
            return '(at {}: bits={}, error={:.2f}, points={}, pulses={}, dead={})'.\
                    format(where, self.bits, self.error, len(self.points), len(self.pulses), self.dead)

    class SlicePoint:
        """ a slice point is a pixel that is part of a detected edge """

        def __init__(self, where, type=None):
            self.where = where  # pixel x,y co-ords
            self.type = type    # either leading or trailing
            self.dead = False   # set True if point should be ignored

        def __str__(self):
            return '(at {}: type={}, dead={})'.\
                    format(self.where, self.type, self.dead)

    class SliceBits:
        """ this is the bit pattern across a slice or along a strip """

        last_id = -1

        def __init__(self, bits=None, length=1, error=0, where=None):
            self.where = where  # the x co-ord where it starts
            self.bits: List[int] = [] if bits is None else bits  # an array of three bits, 0 or 1 or None
            self.length = length  # how many consecutive slices this pattern covers
            self.error = error  # the average slice error associated with this pattern
            self.dead = None  # set to a reason if this bit sequence is rejected
            Scan.SliceBits.last_id += 1
            self.id = Scan.SliceBits.last_id

        def __str__(self):
            return '(#{} at {},*: bits={}, length={}, error={:.2f}, dead={})'.\
                   format(self.id, self.where, self.bits, self.length, self.error, self.dead)

        @staticmethod
        def reset():
            Scan.SliceBits.last_id = -1

    class Threshold:
        def __init__(self, levels, limit):
            self.levels = [None for _ in range(len(levels))]
            self.limit = limit  # the scan limit that yielded this threshold

    class Extent:
        """ the result of the _measure function """
        def __init__(self, target=None, reason=None, inner_edge=None, outer_edge=None, size=None):
            self.target = target           # the cropped image containing the 'projected' target
            self.inner_edge = inner_edge   # list of y's for the inner edge
            self.outer_edge = outer_edge   # list of y's for the outer edge
            self.size = size               # orig radius or target size scaled to the original image
            self.reason = reason           # if edge detection failed, the reason

    class Fragment:
        """ info about a detection radius edge fragment """

        last_id = -1

        def __init__(self, start, end, points, reason, mergee=None):
            self.start = start  # the x co-ord of the start of the fragment
            self.end = end  # the x co-ord of the end of the fragment
            self.points = points  # list of y co-ords for this fragment
            self.reason = reason  # the reason the fragment ended
            self.mergee: Optional[Scan.Fragment] = mergee  # the fragment this merges with or None if it doesn't
            self.collisions: List[Scan.Fragment] = []  # list of fragments that collide with this one
            Scan.Fragment.last_id += 1
            self.id = Scan.Fragment.last_id  # unique id for this Fragment, <0 means its due for deletion
            self.pred: List[Scan.Link] = []  # list of links to fragments that can reach this one
            self.succ: List[Scan.Link] = []  # list of links to fragments that this one can reach

        def __str__(self):
            if len(self.points) > 4:
                points = '{}..{}'.format(vstr(self.points[:3], 'n', close=''), vstr(self.points[-3:], 'n', open=''))
            else:
                points = '{}'.format(vstr(self.points, 'n'))
            return '(#{}({}): {},{}..{},{} points({}): {}, collisions:{}, #pred:{}, #succ:{})'.\
                format(self.id, self.reason, self.start, self.points[0], self.end, self.points[-1],
                       len(self.points), points, len(self.collisions), len(self.pred), len(self.succ))

        @staticmethod
        def reset():
            Scan.Fragment.last_id = -1

    class Link:
        """ info about a link between two fragments """

        def __init__(self, head, tail, gap, jump):
            self.head: Scan.Fragment = head  # fragment 1
            self.tail: Scan.Fragment = tail  # fragment 2
            self.gap = gap  # the distance between these two fragments
            self.jump = jump  # the X 'jump' between these two fragments

        def __str__(self):
            return 'gap:{}, jump:{}, head:{}, tail:{}'.format(self.gap, self.jump, self.head, self.tail)

    class Edge:
        """ info about continuous edges being constructed from edge fragments """

        last_id = -1

        def __init__(self, fragments=None, length=None):
            self.fragments: List[Scan.Fragment] = fragments  # list of Fragment instances making up this edge
            self.length = length  # total length in X (None==not known yet)
            self.state = None
            Scan.Edge.last_id += 1
            self.id = Scan.Edge.last_id

        def __str__(self):
            if len(self.fragments) > 1:
                fragments = ', first={}, last={}'.format(self.fragments[0], self.fragments[-1])
            elif len(self.fragments) > 0:
                fragments = ', first={}'.format(self.fragments[0])
            else:
                fragments = ''
            return '(#{}: state:{}, length:{}, #fragments:{}{})'.\
                   format(self.id, self.state, self.length, len(self.fragments), fragments)

        @staticmethod
        def reset():
            Scan.Edge.last_id = -1

    class EdgeMetric:
        """ result of measuring inner and outer edge candidates """

        def __init__(self, where, slope, strength, gaps, jumps):
            self.where = where  # tha average Y position of the edge across all X's
            self.slope = slope  # count of number of slope changes in the edge (less is better)
            self.strength = strength  # average pixel value along the edge
            self.gaps = gaps  # a tuple of gap count, total gaps, biggest gap
            self.jumps = jumps  # a tuple of jump count, total jumps, biggest jump

        def __str__(self):
            return '(y={:.2f}, slope={}, strength={:.2f}, gaps:[#{}, sum={}, max={}], jumps:[#{}, sum={:.2f}, max={:.2f}])'.\
                   format(self.where, self.slope, self.strength,
                          self.gaps[0], self.gaps[1], self.gaps[2],
                          self.jumps[0], math.sqrt(self.jumps[1]), math.sqrt(self.jumps[2]))

    class Pulse:
        """ a Pulse is a low (head), high (top), low (tail) span across a slice,
            the tail of pulse N is the head of pulse N+1
            """

        def __init__(self, head, top=0, tail=0):
            self.head = head  # the relative length of the low period up to the leading edge
            self.top = top  # the relative length of the high period up to the trailing edge
            self.tail = tail  # the relative length of the low period after the trailing edge
            self.type = None  # the classification according to the relative lengths of the head, top and tail
            self.error = 0  # the deviation (squared) from the ideal relative lengths, 0=no deviation
            self.type2 = None  # next best type
            self.error2 = 0  # and its deviation

        def __str__(self):
            return '({:.2f}-{:.2f}-{:.2f} type={} error={:.2f}, type2={} error2={:.2f})'.\
                    format(self.head, self.top, self.tail, self.type, self.error, self.type2, self.error2)

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

    def __init__(self, code, frame, transform, cells=(8, 8), video_mode=VIDEO_FHD, debug=DEBUG_NONE, log=None):
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
        self.angle_steps = Scan.NUM_BITS * max(self.cells[0], Scan.MIN_PIXELS_PER_BIT)

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
        limit_radius = int(round(max(min(edge_top, edge_bottom, edge_left, edge_right), 1)))
        blob_radius = int((max(blob_size / 4, Scan.MIN_PIXELS_PER_RING_PROJECTED) * Scan.NUM_RINGS)
                          * Scan.BLOB_RADIUS_STRETCH)
        if blob_radius < limit_radius:
            # max possible size is less than the image edge, so use the blob size
            limit_radius = blob_radius

        image_height = limit_radius  # one pixel per radius
        image_width = self.angle_steps

        # do the projection
        code = self.transform.warpPolar(self.image, centre_x, centre_y, limit_radius, image_width, image_height)

        if self.save_images:
            # crop the source image to just this blob
            start_x = max(int(centre_x - limit_radius), 0)
            end_x = min(int(centre_x + limit_radius), max_x)
            start_y = max(int(centre_y - limit_radius), 0)
            end_y = min(int(centre_y + limit_radius), max_y)
            blob = self.transform.crop(self.image, start_x, start_y, end_x, end_y)
            # mark the blob and its co-ords from the original image
            # draw the detected blob in blue and its co-ords and size
            k = (limit_radius, limit_radius, blob_size)
            blob = self.transform.label(blob, k, Scan.RED, '{:.0f}x{:.0f}y ({:.1f})'.
                                        format(centre_x, centre_y, blob_size))
            # save it
            self._unload(blob, '01-target', centre_x, centre_y)
        if self.logging:
            max_x, max_y = code.size()
            self._log('project: projected image size {}x {}y'.format(max_x, max_y))

        return code, limit_radius

    def _measure(self, target: Frame, orig_radius):
        """ find the inner and outer radius edges in the given target,
            returns an instance of Extent with the detected edges,
            the inner edge is the first white to black transition that goes all the way around,
            the outer edge is last black to white transition that goes all the way around,
            """

        def make_edge(transitions):
            """ make the best continuous edge from the given transitions """
            # for an X
            #  for an edge
            #    connect closest for full circle - save with its metrics
            #  repeat for next edge
            # repeat for next X
            # pick best

            def make_candidate(start_x, start_y):
                """ make a candidate edge from transitions from start_x/y,
                    edge pairs at x and x+1 and x and x+2 are considered,
                    the next y is the closest of those 2 pairs
                    """

                def get_nearest_y(x, y):
                    """ find the y with the minimum gap and strongest edge to the given y at slice x,
                        for any x,y there are 3 immediate neighbours: (x+1,y-1),(x+1,y),(x+1,y+1),
                        if more than one of these is the nearest neighbour we choose the one with the
                        'strongest' edge, 'strongest' is that with the pixel value closest to black
                        """
                    neighbours = []
                    min_gap = max_y * max_y
                    min_y = None
                    for t, transition in enumerate(transitions[x]):
                        gap = y - transition[0]
                        gap *= gap
                        if gap < min_gap:
                            min_gap = gap
                            min_y = transition[0]
                        if gap > Scan.MAX_NEIGHBOUR_GAP:
                            # not a neighbour
                            continue
                        neighbours.append((transition[0], transition[1], gap))
                    if len(neighbours) > 1:
                        # got a choice, pick the best
                        best = neighbours[0]
                        for n, neighbour in enumerate(neighbours):
                            if neighbour[1] < best[1]:
                                best = neighbour
                        return best[0], best[2]
                    else:
                        return min_y, min_gap

                candidate = [None for _ in range(max_x)]
                x = start_x - 1
                y = start_y
                for _ in range(max_x):
                    x = (x + 1) % max_x
                    this_y, this_gap = get_nearest_y(x, y)
                    if this_gap > Scan.MAX_NEIGHBOUR_GAP:
                        # direct neighbour not connected, see if indirect one is nearer
                        next_y, next_gap = get_nearest_y((x + 1) % max_x, y)
                        if this_gap <= next_gap:
                            # nope
                            y = this_y
                        else:
                            # yep
                            y = next_y
                    else:
                        # direct neighbour connected, use that
                        y = this_y
                    candidate[x] = y

                return candidate

            # This makes lots of duplicates, but it is easier to deal with them than suppress them!
            edges = []
            for x in range(max_x):
                for transition in transitions[x]:
                    edges.append(make_candidate(x, transition[0]))

            # pick best
            best_edge = None
            best_jumps = None
            for edge in edges:
                # count y jumps
                jumps = 0
                for x in range(max_x):
                    last_y = edge[(x - 1) % max_x]
                    this_y = edge[x]
                    gap = last_y - this_y
                    gap *= gap
                    if gap > Scan.MAX_NEIGHBOUR_GAP:
                        # not connected
                        jumps += gap
                if best_edge is None or jumps < best_jumps:
                    best_edge = edge
                    best_jumps = jumps

            return best_edge

        # make an image to detect our edges within
        max_x, max_y = target.size()
        edges: Frame = target.instance()
        edges.new(max_x, max_y, MIN_LUMINANCE)

        # build bucketised image (reduce to the 4 luminance levels described above)
        bucket_range = int(round((MAX_LUMINANCE - MIN_LUMINANCE) / len(Scan.EDGE_THRESHOLD_LEVELS)))
        thresholds: List[Scan.Threshold] = [None for _ in range(max_x)]
        for x in range(max_x):
            threshold = self._get_slice_threshold(target, x, max_x, 0, max_y, Scan.EDGE_THRESHOLD_LEVELS)
            for y in range(threshold.limit):
                pixel = self._get_threshold_pixel(target, x, y, threshold, bucket=bucket_range)
                edges.putpixel(x, y, pixel)
            thresholds[x] = threshold
        # remove islands - pixels completely surrounded by the same colour become that colour
        for x in range(max_x):
            for y in range(thresholds[x].limit):
                is_island = False
                colour = None
                for dx, dy in Scan.ISLAND_KERNEL:
                    pixel = edges.getpixel((x + dx) % max_x, y + dy)
                    if pixel is None:
                        # ignore pixels off the edge
                        continue
                    if colour is None:
                        colour = pixel
                        is_island = True
                    elif pixel != colour:
                        is_island = False
                        break
                if is_island:
                    edges.putpixel(x, y, colour)

        # build list of transitions to black (the first is probably the inner edge)
        to_black = [[] for _ in range(max_x)]  # falling edges
        for x in range(max_x):
            threshold = thresholds[x]
            last_pixel = edges.getpixel(x, 0)
            for y in range(1, threshold.limit):
                pixel = edges.getpixel(x, y)
                if pixel < last_pixel:
                    # falling edge, want y of highest luminance (i.e. previous y)
                    to_black[x].append((y - 1, pixel))
                last_pixel = pixel
            if len(to_black[x]) == 0:
                # this means the whole slice is the same intensity
                to_black[x] = [(0, 0)]
        inner = make_edge(to_black)

        # adjust threshold limit to reflect the inner edge we discovered
        for x in range(max_x):
            # inner y is assumed to represent the two inner white rings,
            # so the limit for the outer is y/2 * num-rings * a fudge factor
            thresholds[x].limit = int(min((inner[x] / 2) * Scan.NUM_RINGS * Scan.RING_RADIUS_STRETCH, max_y))

        # build list of transitions to white (the last is probably the outer edge)
        to_white = [[] for _ in range(max_x)]  # rising edges
        for x in range(max_x):
            threshold = thresholds[x]
            last_pixel = edges.getpixel(x, 0)
            for y in range(inner[x] + 1, threshold.limit):
                pixel = edges.getpixel(x, y)
                if pixel > last_pixel:
                    # rising edge, want y of highest luminance (i.e. this y)
                    to_white[x].append((y, last_pixel))
                last_pixel = pixel
            if len(to_white[x]) == 0:
                # this means the whole slice is black
                to_white[x] = [(max(threshold.limit - 1, 0), 0)]
        outer = make_edge(to_white)

        # smooth the edges
        inner_edge = self._smooth_edge(inner)
        outer_edge = self._smooth_edge(outer)

        if self.save_images:
            y_limit = [max(thresholds[x].limit - 1, 0) for x in range(max_x)]
            plot = edges
            plot = self._draw_plots(plot, [[0, y_limit]], colour=Scan.RED)
            plot = self._draw_plots(plot, [[0, inner]], colour=Scan.GREEN)
            plot = self._draw_plots(plot, [[0, outer]], colour=Scan.BLUE)
            self._unload(plot, '03-edges')

            plot = target
            plot = self._draw_plots(plot, [[0, inner_edge]], None, Scan.GREEN)
            plot = self._draw_plots(plot, [[0, outer_edge]], None, Scan.GREEN)
            self._unload(plot, '04-wavy')

        return Scan.Extent(target=target, inner_edge=inner_edge, outer_edge=outer_edge, size=orig_radius)

    def _get_estimated_ring_sizes(self, inner_start, inner_end, outer_start, outer_end, stretch=False):
        """ given an inner and outer ring position, calculate the estimated width of each ring,
            returns a vector of ring sizes,
            when stretch is True all rings are set to the same size and made at least
            """

        # The distance between the inner white/black edge and the outer black/white edge spans all but the two
        # inner white rings and one outer white rings (i.e. -3)
        ring_span = Scan.NUM_RINGS - 3  # number of rings between inner and outer edge
        outer_ring_size = (outer_end - outer_start) / ring_span  # average outer ring size (i.e. beyond inner edge)
        inner_ring_size = (inner_end - inner_start) / 2  # average inner ring size (i.e. inside inner edge)

        ring_sizes = [None for _ in range(Scan.NUM_RINGS)]

        if stretch:
            # we want all rings the same size here and as big as the max
            outer_ring_size = max(outer_ring_size, inner_ring_size, Scan.MIN_PIXELS_PER_RING, self.cells[1])
            inner_ring_size = outer_ring_size

        ring_sizes[self.INNER_POINT] = inner_ring_size
        ring_sizes[self.INNER_WHITE] = inner_ring_size
        # inner edge is here
        ring_sizes[self.INNER_BLACK] = outer_ring_size
        ring_sizes[self.DATA_RING_1] = outer_ring_size
        ring_sizes[self.DATA_RING_2] = outer_ring_size
        ring_sizes[self.DATA_RING_3] = outer_ring_size
        ring_sizes[self.OUTER_BLACK] = outer_ring_size
        # outer edge is here
        ring_sizes[self.OUTER_WHITE] = outer_ring_size

        return ring_sizes

    def _stretch(self, pixels, size, offset_1_in, offset_2_in):
        """ given a vector of pixels stretch them such that the resulting pixels is size long,
            offset_1/2 are the offsets in the input for which we want the equivalent offsets in the output,
            consecutive pixel values are interpolated as necessary when stretched into the new size,
            this is a helper function for _flatten, in that context we are stretching pixels in the
            'y' direction, the names used here reflect that but this logic is totally generic
            """

        dest = [None for _ in range(size)]
        # do the initial stretch
        y_delta = size / len(pixels)
        y = 0
        offset_1_out = None
        offset_2_out = None
        for pixel in range(len(pixels)):
            dest[int(y)] = pixels[pixel]
            if offset_1_in is not None and pixel == offset_1_in:
                offset_1_out = int(y)
            if offset_2_in is not None and pixel == offset_2_in:
                offset_2_out = int(y)
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
                    dest[start_gap + dy] = int(gap_pixel)
                start_gap = None
                end_gap = None
        if start_gap is not None and start_gap > 0:
            # do the final gap by just propagating the last pixel
            span = size - start_gap
            gap_pixel = dest[start_gap - 1]
            for dy in range(span):
                dest[start_gap + dy] = gap_pixel
        # that's it
        return dest, offset_1_out, offset_2_out

    def _flatten(self, measured):
        """ measured is the output from the _measure function, it can be passed in here naked, if it was
            aborted (reason not None) we'll just abort out of here too,
            remove perspective distortions from the given 'measured' image and its inner/outer edges,
            this helps to mitigate ring width distortions which is significant when we analyse pulse,
            a circle when not viewed straight on appears as an ellipse, when that is projected into a rectangle
            the radius edges becomes 'wavy' (a sine wave), this function straightens those wavy edges, other
            distortions can arise if the target is curved (e.g. if it is wrapped around someones leg), in
            this case the the outer rings are even more 'wavy', for the purposes of this function the distortion
            is assumed to be proportional to the variance in the inner and outer edge positions, we know between
            the inner and outer edges there are 5 rings, we apply a stretching factor to each ring that is a
            function of the inner and outer edge variance,
            the returned image is just enough to contain the (reduced) image pixels of all the target rings
            (essentially number of bits wide by number of rings high),
            orig_radius is the target radius in the original image, its used to calculate the target scale,
            returns the flattened image, its inner and outer edges, its scale and reject reason (as an Extent),
            NB: the inner and outer edges returned are expected to be flat but this is not verified here
            """

        if measured.reason is not None:
            # propagate what _measured said
            return measured

        target = measured.target
        ring_inner_edge = measured.inner_edge
        ring_outer_edge = measured.outer_edge
        orig_radius = measured.size

        # get the edge limits we need
        max_x, projected_y = target.size()

        # ref point for inner is image edge in height, i.e. 0
        max_inner_edge = 0
        # ref point for outer is the corresponding inner
        max_distance = 0  # max distance between inner and outer edge
        max_outer_edge = 0  # outer edge at max distance
        max_outer_inner_edge = 0  # inner edge at max distance
        for x in range(max_x):
            inner_edge = ring_inner_edge[x]
            outer_edge = ring_outer_edge[x]
            if inner_edge > max_inner_edge:
                max_inner_edge = inner_edge
            distance = outer_edge - inner_edge
            if distance > max_distance:
                max_outer_edge = outer_edge
                max_outer_inner_edge = inner_edge
                max_distance = distance

        if self.logging:
            self._log('flatten: max inner edge {}, outer edge at max distance from inner {}, '
                      'inner edge at max outer edge {}'.
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

        # build flat image and flat inner/outer edge
        inner_edge = [None for _ in range(max_x)]
        outer_edge = [None for _ in range(max_x)]
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
                inner_dy = None
                outer_dy = None
                for dy in range(len(in_pixels)):
                    in_pixels[dy] = target.getpixel(x, in_y)
                    if in_y == ring_inner_edge[x]:
                        inner_dy = dy
                    if in_y == ring_outer_edge[x]:
                        outer_dy = dy
                    in_y += 1
                out_ring_end = out_ring_start + stretched_size[ring]  # these may be fractional
                out_pixels, inner_dy, outer_dy = self._stretch(in_pixels,
                                                               max(int(round(out_ring_end - out_y)), 1),
                                                               inner_dy, outer_dy)
                # the out ring sizes have been arranged to be whole pixel widths, so no fancy footwork here
                for dy in range(len(out_pixels)):
                    pixel = out_pixels[dy]
                    code.putpixel(x, out_y, pixel)
                    if inner_dy is not None and inner_dy == dy:
                        inner_edge[x] = out_y
                    if outer_dy is not None and outer_dy == dy:
                        outer_edge[x] = out_y
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

        # calculate the flattened image scale relative to the original
        # orig_radius is what was extracted from the original image, it then
        # became projected_y, then it became flat_y and finally target_y
        # orig to projected may have been up-scaled
        # projected to flat is truncating, truncated_y is the truncation boundary
        scale1 = orig_radius / projected_y
        scale2 = truncated_y / flat_y
        scale = scale1 * scale2

        # calculate the target size relative to the original image (as the average outer edge)
        target_size = sum(outer_edge) / len(outer_edge)
        target_size *= scale  # scale to size in original image

        if self.save_images:
            plot = self._draw_plots(code, [[0, inner_edge]], None, Scan.GREEN)
            plot = self._draw_plots(plot, [[0, outer_edge]], None, Scan.GREEN)
            self._unload(plot, '05-flat')
        if self.logging:
            max_x, max_y = code.size()
            inner_average = 0
            outer_average = 0
            for x in range(max_x):
                inner_average += inner_edge[x]
                outer_average += outer_edge[x]
            self._log('flatten: flattened image size {}x {}y, '
                      'inner edge from {} to {}, average={:.2f}, '
                      'outer edge from {} to {}, average={:.2f})'.
                      format(max_x, max_y,
                             inner_edge[0], inner_edge[-1], inner_average / max_x,
                             outer_edge[0], outer_edge[-1], outer_average / max_x))

        # return flattened image
        return Scan.Extent(target=code, inner_edge=inner_edge, outer_edge=outer_edge, size=target_size)

    def _smooth_edge(self, in_edge, no_round=False):
        """ smooth the given edge vector by doing a mean across N pixels,
            the result is rounded as an integer unless no_round is given,
            the edge may wrap,
            return the smoothed vector
            """
        extent = len(in_edge)
        out_edge = [None for _ in range(extent)]
        for x in range(extent):
            v = 0  # value accumulator
            d = 0  # divisor accumulator
            for dx, f in Scan.SMOOTHING_KERNEL:
                sample = in_edge[(x + dx) % extent]
                if sample is None:
                    # what does this mean?
                    continue
                v += (sample * f)
                d += f
            if d > 0:
                if no_round:
                    out_edge[x] = v / d
                else:
                    out_edge[x] = int(round(v / d))
        return out_edge

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

    def _get_slice_threshold(self, target, x, max_x, start_y, max_y, levels) -> Threshold:
        """ determine a suitable threshold to apply to the given image slice to segregate it into levels,
            x is the slice origin, start_y/end_y is the slice range,
            levels defines the threshold levels to use as a ratio of the luminance range,
            the slice ia assumed to extend beyond the area of interest, we want the threshold for the
            area of interest only, the area of interest is assumed to consist of the eight rings of our
            target which is composed of black and white areas, with at most 4 'white' bands and 3 'black'
            bands, this implies a luminance level change for each data ring, we first estimate where these
            end with an overall threshold, then re-do the threshold across this estimate, the objective is
            to maximise contrast across the area of interest, this is only done if start_y is 0, otherwise
            we assume the caller has already determined the "area of interest"
            """

        def get_end_y(x):
            """ determine the likely end_y for the given x """
            end_y = max_y
            up_changes = 0
            down_changes = 0
            previous_pixel = None
            thresh1 = self._get_threshold(target, x, start_y, max_y, initial_levels)
            for y in range(start_y, max_y):
                pixel = self._get_threshold_pixel(target, x, y, thresh1, bucket=bucket_range)
                if pixel is None:
                    break
                if previous_pixel is None:
                    previous_pixel = pixel
                    continue
                if pixel > previous_pixel:
                    up_changes += 1
                elif pixel < previous_pixel:
                    down_changes += 1
                previous_pixel = pixel
                if up_changes >= changes_limit or down_changes >= changes_limit:
                    # found enough changes
                    end_y = min(y + 1, max_y)
                    break
            return end_y

        end_y = max_y
        if start_y == 0:
            # look at the whole slice to get an initial estimate
            if len(levels) < len(Scan.BUCKET_THRESHOLD_LEVELS):
                # not enough levels to reliably detect ring edges (where they are very 'smudged')
                initial_levels = Scan.BUCKET_THRESHOLD_LEVELS
            else:
                initial_levels = levels
            # NB: because we have N levels there could be N up_changes per band, hence * below
            changes_limit = Scan.NUM_DATA_RINGS * len(initial_levels)
            # set the range of each luminance bucket
            bucket_range = int(round((MAX_LUMINANCE - MIN_LUMINANCE) / len(initial_levels)))
            end_y = get_end_y(x)
            # if max_y > (Scan.NUM_RINGS * self.cells[1]) * Scan.LARGE_IMAGE_LIMIT:
            #     # we've got a very large image, in this context there can be a significant 'wobble'
            #     # on edges, so we check our neighbour too and use the larger range
            #     end_y2 = get_end_y((x + 1) % max_x)
            #     end_y = max(end_y, end_y2)
            # now get a better threshold over the reduced slice up to end_y
        else:
            # caller is forcing our hand
            pass
        thresh2 = self._get_threshold(target, x, start_y, end_y, levels)
        return thresh2

    def _get_threshold(self, target, x, start_y, end_y, levels) -> Threshold:
        """ get the threshold levels for the given target slice,
            x is the slice origin, start_y/end_y is the slice range,
            levels defines the threshold levels to use as a ratio of the luminance range
            """

        # # ToDo: HACK START - worse than linear (below)
        # # do a histogram based threshold
        # threshold2 = Scan.Threshold(levels, end_y)
        # counts = [0 for _ in range(MAX_LUMINANCE)]
        # for y in range(start_y, end_y):
        #     pixel = target.getpixel(x, y)
        #     if pixel is None or pixel == MIN_LUMINANCE or pixel == MAX_LUMINANCE:
        #         # ignore the extremes (they distort the range too much)
        #         continue
        #     counts[pixel] += 1
        # pixel_total = sum(counts)
        # level_count = 0
        # pixel_count = 0
        # luminance = 0
        # count_per_level = pixel_total / (len(levels) + 1)
        # for level in range(len(levels)):
        #     level_count += count_per_level
        #     while luminance < (MAX_LUMINANCE - 1):
        #         if (pixel_count + counts[luminance + 1]) < level_count:
        #             luminance += 1
        #             pixel_count += counts[luminance]
        #         else:
        #             threshold2.levels[level] = luminance
        #             break
        # return threshold2
        # # ToDo: HACK END
        threshold = Scan.Threshold(levels, end_y)
        white_level = MIN_LUMINANCE
        black_level = MAX_LUMINANCE
        for y in range(start_y, end_y):
            pixel = target.getpixel(x, y)
            if pixel == MIN_LUMINANCE or pixel == MAX_LUMINANCE:
                # ignore the extremes (they distort the range too much)
                continue
            if pixel > white_level:
                white_level = pixel
            if pixel < black_level:
                black_level = pixel
        if black_level == MAX_LUMINANCE:
            # this means the image is purely black and white
            black_level = MIN_LUMINANCE
        if white_level == MIN_LUMINANCE:
            # ditto
            white_level = MAX_LUMINANCE
        if white_level <= black_level:
            # this means the image is only 1 colour
            luminance_range = MAX_LUMINANCE - MIN_LUMINANCE
        else:
            luminance_range = white_level - black_level

        # set the threshold for each level based on the luminance range we've measured
        # NB: the luminance range is always > 0 and this means any threshold is always > 0
        previous_threshold = black_level
        for level in range(len(levels)):
            this_threshold = previous_threshold + luminance_range * levels[level]
            threshold.levels[level] = this_threshold
            previous_threshold = this_threshold

        return threshold

    def _get_slices_thresholds(self, target, levels, inner_edge=None, outer_edge=None, offset=1) -> List[Threshold]:
        """ find the thresholds to apply in the given target for every x,
            levels defines the threshold levels to use,
            if inner_edge is given it specifies the minimum y to probe, else we probe from the image edge,
            if outer_edge is given it specifies the maximum y to probe, else we probe to the image edge,
            offset is applied to the inner and outer edge and specifies how far to probe within these edges,
            all the y's associated with an x are referred to as a 'slice',
            this function generates a number of thresholds for every slice,
            these can be used to categorise a pixel into several 'buckets',
            returns a vector of Threshold instances (NB: all thresholds are guaranteed to be > 0),
            this is the *only* function that creates Threshold instances
            """

        max_x, max_y = target.size()

        thresholds = [None for _ in range(max_x)]
        for x in range(max_x):
            # find the luminance range
            if inner_edge is None:
                start_y = 0
            else:
                start_y = max(int(round(inner_edge[x] - offset)), 0)
            if outer_edge is None:
                end_y = max_y
            else:
                end_y = min(int(round(outer_edge[x] + offset)), max_y)
            thresholds[x] = self._get_slice_threshold(target, x, max_x, start_y, end_y, levels)

        return thresholds

    def _get_threshold_level(self, thresholds, x, level=0):
        """ given a list thresholds, an x and a level return the corresponding threshold level """
        return thresholds[x].levels[level]

    def _get_threshold_pixel(self, target, x, y, threshold: Threshold, level=0, bucket=None):
        """ get the pixel at x,y if it is over the 'black' threshold,
            level is the threshold level to apply, 0=black, -1=white, ignored if bucket given,
            if bucket is given it represents the luminance range of each level and the function
            returns a pixel value of its threshold level scaled by the bucket range,
            otherwise the actual image pixel is returned if it os over the given level,
            this is the *only* function that is aware of the structure of the threshold,
            returning None means x,y is off the image edge,
            returning > 0 means the pixel is over the threshold (or in one of the buckets),
            returning <= 0 means the pixel is below the threshold (0-return is the pixel value)
            """

        pixel = target.getpixel(x, y)  # NB: returns None when reach image edge
        if pixel is None:
            return None

        if bucket is not None:
            # consider each level as a bucket of 1/nth the max and return that as a pixel value
            # the last level is considered to be the 'white' threshold, the first the 'black'
            if pixel >= threshold.levels[-1]:
                # we're over the last, so its white
                return MAX_LUMINANCE
            for level in range(len(threshold.levels)-2, -1, -1):   # NB: -2 'cos already done the last
                if pixel >= threshold.levels[level]:
                    # found the bucket
                    return (level+1) * bucket
            # if get here the pixel is below the black threshold, so its black
            return MIN_LUMINANCE

        if pixel >= threshold.levels[level]:
            return int(pixel)
        else:
            return 0 - int(pixel)

    def _get_within_threshold(self, target, context, x, y, threshold, reversed=False):
        """ get a list of pixels from x,y in the given scan direction that are over the given threshold,
            NB: target here is *not* the same as the context target,
            x is wrapped at image edge, y is not, if given an excessive y None is returned,
            context direction defines the scanning direction (x or y),
            reversed is True if want result backwards relative to the scanning direction or False if not,
            returns a list of pixels or an empty list if the given x,y is not over the threshold,
            the order of the pixels returned is the same as the scan direction
            """

        max_y = context.max_y
        scan_limit = context.max_scan_coord + 1
        scan_coord = context.scan_coord
        if reversed:
            scan_inc = 0 - context.scan_multiplier
        else:
            scan_inc = context.scan_multiplier

        if y >= max_y:
            return []

        pixels = []
        xy = [x, y]
        while True:
            edge_pixel = self._get_threshold_pixel(target, xy[0], xy[1], threshold[x])
            if edge_pixel is None or edge_pixel <= 0:
                break
            pixels.append(edge_pixel)    # add it to our list
            xy[scan_coord] += scan_inc  # move on
            if xy[scan_coord] >= scan_limit:
                break  # that's it

        return pixels

    def _compress(self, target, inner_edge, outer_edge):
        """ compress the context target pixel luminance range into just black and white,
            inner/outer_edge is a list of y's for the target inner and outer edges,
            returns the modified image such that pixels before the inner or after the outer edge are white,
            this is important to ensure the first edge is the trailing edge of the inner target extent and
            the last edge is the leading edge of the outer target extent,
            """

        def is_artifact(target, x, y, kernel, neighbours, direction, max_x, max_y):
            """ determine if we have an 'artifact' at x,y in the given target image in the given direction,
                the given target image must be a binary one (i.e. all pixels black or white)
                kernel defines the pixels that must all be black or all white,
                neighbours defines the pixels that must be all the other colour,
                an artifact is when both the above criteria are met,
                returns MAX_LUMINANCE for a black artifact, MIN_LUMINANCE for a white one, or None if no artifact
                """

            def count_bw(target, x, y, kernel, direction, max_x, max_y):
                """ count the number of black and white pixels in the given binary target around the given kernel """

                whites = 0
                blacks = 0
                for kx, ky in Scan.Kernel(kernel, direction):
                    dy = y + ky
                    if dy < 0 or dy > (max_y - 1):
                        continue
                    dx = (x + kx) % max_x
                    pixel = target.getpixel(dx, dy)
                    if pixel == MAX_LUMINANCE:
                        whites += 1
                    elif pixel == MIN_LUMINANCE:
                        blacks += 1
                return blacks, whites

            blacks, whites = count_bw(target, x, y, kernel, direction, max_x, max_y)
            if whites == len(kernel):
                # found a potential white artifact
                blacks, whites = count_bw(target, x, y, neighbours, direction, max_x, max_y)
                if blacks == len(neighbours):
                    # got a white artifact
                    return MIN_LUMINANCE

            elif blacks == len(kernel):
                # found a potential black artifact
                blacks, whites = count_bw(target, x, y, neighbours, direction, max_x, max_y)
                if whites == len(neighbours):
                    # got a black artifact
                    return MAX_LUMINANCE

            return None

        def count_neighbours(target, x, y, inner_edge, outer_edge, buckets):
            """ count how many neighbours of x,y are in each luminance bucket or border an edge,
                inner/outer_edge define where the borders are,
                buckets defines the luminance levels of each of our buckets,
                returns a tuple of the counts in ascending luminance order plus the edges count,
                an 'edge' is a pixel that neighbours the inner or outer edge,
                """

            nonlocal max_x, max_y

            NEIGHBOURS = [[-1, -1], [ 0, -1], [+1, -1],
                          [-1,  0],           [+1,  0],
                          [-1, +1], [ 0, +1], [+1, +1]]

            counts = [0 for _ in range(len(buckets))]
            edge = 0
            for xy in NEIGHBOURS:
                dx = (xy[0] + x) % max_x
                dy = xy[1] + y
                if dy >= max_y or dy <= inner_edge[dx] or dy >= outer_edge[dx]:
                    edge += 1
                else:
                    pixel = target.getpixel(dx, dy)
                    for bucket in range(len(buckets)):
                        if pixel == buckets[bucket]:
                            counts[bucket] += 1
                            break

            return counts, edge

        def pixel_level(pixel, buckets):
            """ given a pixel and bucket levels, determine which bucket the pixel is within,
                returns the bucket number or None if its none of them
                """

            for bucket in range(len(buckets)):
                if pixel == buckets[bucket]:
                    return bucket
            return None

        def bounded_by(level, counts):
            """ given a level and set of neighbour counts determine the neighbours above and below that level """

            if level is None:
                return 0, 0

            high_neighbours = 0
            low_neighbours = 0
            for count in range(len(counts)):
                if count < level:
                    low_neighbours += counts[count]
                elif count > level:
                    high_neighbours += counts[count]

            return low_neighbours, high_neighbours

        max_x, max_y = target.size()

        # calc the minimum width of a ring in pixels (used in thresholding and enforcing inner/outer black rings)
        min_width = max((max_y / Scan.NUM_RINGS) * Scan.MIN_EDGE_WIDTH, Scan.MIN_EDGE_WIDTH_PIXELS)

        # get the thresholds to apply
        threshold = self._get_slices_thresholds(target, Scan.BUCKET_THRESHOLD_LEVELS,
                                                inner_edge, outer_edge, min_width)

        # set the range of each luminance bucket
        bucket_range = int(round((MAX_LUMINANCE - MIN_LUMINANCE) / len(Scan.BUCKET_THRESHOLD_LEVELS)))

        # set the luminance level for every bucket (NB: this is one more than the threshold count)
        buckets = [x * bucket_range for x in range(len(Scan.BUCKET_THRESHOLD_LEVELS) + 1)]

        # set meaningful names for some of our bucket indices
        middle = int(round((len(buckets) + 1) / 2))

        # make an empty image to load our buckets into
        compressed = target.instance()
        compressed.new(max_x, max_y, MIN_LUMINANCE)

        # make the image (this yields pixels of one of our bucket levels)
        for x in range(max_x):
            for y in range(max_y):
                pixel = self._get_threshold_pixel(target, x, y, threshold[x], bucket=bucket_range)
                compressed.putpixel(x, y, pixel)

        if self.save_images:
            self._unload(compressed, '06-buckets')

        # tidy the image by a consideration of pixel neighbours and the inner/outer edges
        # this is effectively a "Morphological Operation" on the image
        # there are several operations performed:
        #  1: enforce white before inner and after outer
        #  2: enforce black near inner and outer
        #  3: bright cells surrounded by mostly darker migrate towards darker
        #  4: dark cells surrounded by mostly brighter migrate towards brighter
        # the term 'mostly' above means 6 or 7 or 8, e.g.:
        #   x x x   or  x x x  or  x x x  or  x x x  or  x x x  or  x x x  or  x x x  and their rotations
        #   x . x       x . x      x . x      x . x      - . -      x . X      x . -
        #   x x x       x - x      - x x      x - -      x x x      - x -      - x x
        # these operations are heuristics developed by observation of lots of noisy code examples
        # the objective is to end up with a binary image (either black+grey or black+white or grey+white)
        passes = []
        while True:
            migrated = 0
            for x in range(max_x):
                inner = inner_edge[x]
                outer = outer_edge[x]
                for y in range(max_y):
                    pixel = compressed.getpixel(x, y)
                    if y < inner or y > outer:
                        # rule 1: these must all be white
                        if pixel != MAX_LUMINANCE:
                            compressed.putpixel(x, y, MAX_LUMINANCE)
                            migrated += 1
                        continue
                    if (y - inner) < min_width or (outer - y) < min_width:
                        # rule 2: these must all be black
                        if pixel != MIN_LUMINANCE:
                            compressed.putpixel(x, y, MIN_LUMINANCE)
                            migrated += 1
                        continue
                    counts, edge = count_neighbours(compressed, x, y, inner_edge, outer_edge, buckets)
                    if edge > 0:
                        # rule 2: anything near an edge must either be black or migrate to black
                        if pixel == MIN_LUMINANCE:
                            migrate = None
                        else:
                            migrate = MIN_LUMINANCE
                    else:
                        # if a pixel is mostly surrounded by more than itself, go up a level
                        # if a pixel is mostly surrounded by less than itself, go down a level
                        # bias is to go up (so faint areas get highlighted)
                        level = pixel_level(pixel, buckets)
                        low_neighbours, high_neighbours = bounded_by(level, counts)
                        if low_neighbours >= 7:
                            # rule 3: migrate towards black
                            migrate = buckets[level - 1]
                        elif high_neighbours >= 7:
                            # rule 4: migrate towards white
                            migrate = buckets[level + 1]
                        else:
                            migrate = None
                    if migrate is not None:
                        compressed.putpixel(x, y, migrate)
                        migrated += 1
            if migrated == 0:
                break
            passes.append(migrated)

        # migrate all below middle to black and all above middle to white
        migrated = 0
        for x in range(max_x):
            for y in range(max_y):
                pixel = compressed.getpixel(x, y)
                if pixel < buckets[middle]:
                    if pixel != MIN_LUMINANCE:
                        compressed.putpixel(x, y, MIN_LUMINANCE)
                        migrated += 1
                else:
                    if pixel != MAX_LUMINANCE:
                        compressed.putpixel(x, y, MAX_LUMINANCE)
                        migrated += 1
        if migrated > 0:
            passes.append(migrated)

        # suppress 'nipples' and 'corners'
        nipple_kernel = ((0, 0), )  # ToDo: HACK-->(0, 1))
        nipple_neighbours = ((-1, 0), (1, 0), )  # ToDo: HACK-->(-1, -1), (0, -1), (1, -1), )
        corner_kernel = ((0, 0), (-1, 1), (1, -1), )
        corner_neighbours = ((-1, 0), (-1, -1), (0, -1), )
        directions = (Scan.LEFT_TO_RIGHT, Scan.TOP_DOWN, Scan.RIGHT_TO_LEFT, Scan.BOTTOM_UP)
        artifacts = ((nipple_kernel, nipple_neighbours), )  # ToDo: HACK-->(corner_kernel, corner_neighbours), )
        migrated = 0
        for x in range(max_x):
            for y in range(max_y):
                changed = 0
                for direction in directions:
                    for artifact in artifacts:
                        pixel = is_artifact(compressed, x, y, artifact[0], artifact[1], direction, max_x, max_y)
                        if pixel is None:
                            continue
                        else:
                            # got an artifact - change its colour
                            compressed.putpixel(x, y, pixel)
                            changed += 1
                            break
                    if changed > 0:
                        break
                migrated += changed
        if migrated > 0:
            passes.append(migrated)

        if self.logging:
            self._log('pixel migrations in {} passes: {}'.format(len(passes), passes))
        if self.save_images:
            self._unload(compressed, '07-binary')

        return compressed

    def _make_slices(self, target, direction) -> List[Slice]:
        """ make slices from the given binary image in the given direction,
            direction is TOP_DOWN
            direction is LEFT_TO_RIGHT to make slices for every y counting edges horizontally,
            a slice is a list of leading and trailing edges of white areas,
            TOP_DOWN there will be at least the trailing and leading edge of the white/black/white rings,
            LEFT_TO_RIGHT there will be at least leading and trailing edge of the marker sync bits,
            edge to edge spans that are too small are discarded (they're noise),
            empty slices are ignored (these will be the inner and outer white and black rings)
            """

        max_x, max_y = target.size()
        if direction == Scan.TOP_DOWN:
            min_span = max((max_y / Scan.NUM_RINGS) * Scan.MIN_EDGE_TO_EDGE_SPAN, Scan.MIN_EDGE_TO_EDGE_PIXELS)
            scan_coord = 1
            sum_coord = 0
            max_scan_coord = max_y
            max_sum_coord = max_x
        else:
            min_span = max((max_x / Scan.NUM_BITS) * Scan.MIN_BIT_TO_BIT_SPAN, Scan.MIN_BIT_TO_BIT_PIXELS)
            scan_coord = 0
            sum_coord = 1
            max_scan_coord = max_x
            max_sum_coord = max_y
        min_span *= min_span

        if self.logging:
            header = 'slices: {} ignoring short edges:'.format(direction)
        slices = [Scan.Slice() for _ in range(max_sum_coord)]
        for x in range(max_x):
            for y in range(max_y):
                xy = [x, y]
                slice = slices[xy[sum_coord]]
                prev_xy = [xy[0], xy[1]]
                prev_xy[scan_coord] = (prev_xy[scan_coord] - 1) % max_scan_coord
                prev = target.getpixel(prev_xy[0], prev_xy[1])
                curr = target.getpixel(xy[0], xy[1])
                trailing = prev > MID_LUMINANCE > curr
                leading = prev < MID_LUMINANCE < curr
                if (leading or trailing) and len(slice.points) > 0:
                    # got an edge, check if it is too small
                    where = slice.points[-1].where[scan_coord]
                    span = xy[scan_coord] - where
                    span *= span
                    if span < min_span:
                        # too small, ignore it and chuck its other edge
                        if self.logging:
                            if header is not None:
                                self._log(header)
                                header = None
                            start_xy = [0, 0]
                            start_xy[scan_coord] = where
                            start_xy[sum_coord] = xy[sum_coord]
                            self._log('    {},{} .. {},{} (span {}, min is {:.2f})'.
                                      format(start_xy[0], start_xy[1], xy[0], xy[1],
                                             math.sqrt(span), math.sqrt(min_span)))
                        del slice.points[-1]
                        continue
                if trailing:
                    # for a trailing edge we want the end of the bright area, not the start of the dark
                    xy[scan_coord] = (xy[scan_coord] - 1) % max_scan_coord
                    slice.points.append(Scan.SlicePoint((xy[0], xy[1], ), type=Scan.TRAILING_EDGE))
                elif leading:
                    slice.points.append(Scan.SlicePoint((xy[0], xy[1], ), type=Scan.LEADING_EDGE))

        # remove empty slices (these will be the inner and outer white and black marker rings)
        for s in range(len(slices)-1, -1, -1):
            if len(slices[s].points) == 0:
                del slices[s]

        return slices

    def _make_patches(self, target: Frame, strips: List[Slice], slices: List[Slice]):
        """ make a binary image with white patches that correspond to the areas between leading/trailing
            bit (strips) and ring (slices) edges
            """

        max_x, max_y = target.size()

        # make an empty image to load our patches into
        patches: Frame = target.instance()
        patches.new(max_x, max_y, MIN_LUMINANCE)

        # create horizontal white bars between all leading/trailing bit edges
        for strip in strips:
            for p, this_point in enumerate(strip.points):
                if this_point.type == Scan.LEADING_EDGE:
                    next_point = strip.points[(p + 1) % len(strip.points)]
                    if next_point.type != Scan.TRAILING_EDGE:
                        self._log('strip: leading point is not followed by trailing point!', fatal=True)
                    x = this_point.where[0]
                    end_x = (next_point.where[0] + 1) % max_x
                    while x != end_x:
                        y = this_point.where[1]
                        patches.putpixel(x, y, MAX_LUMINANCE)
                        x = (x + 1) % max_x

        # clear white areas that are not within a leading/trailing ring edge
        for slice in slices:
            for p, this_point in enumerate(slice.points):
                if this_point.type == Scan.TRAILING_EDGE:
                    next_point = slice.points[(p + 1) % len(slice.points)]
                    if next_point.type != Scan.LEADING_EDGE:
                        self._log('slice: trailing point is not followed by leading point!', fatal=True)
                    for y in range(this_point.where[1] + 1, next_point.where[1]):
                        x = this_point.where[0]
                        patches.putpixel(x, y, MIN_LUMINANCE)

        return patches

    def _make_pulses(self, slice: Slice) -> Slice:
        """ make the pulses for the given slice,
            creates a list of pulses as the lengths of the head, top and tail parts relative to the total width,
            the head is the leading low period, the top is the high period and the tail is the trailing low,
            the lengths are relative to ring width, the ring width is the inner to outer distance over the five
            rings that span that distance in our target structures,
            the points of the slice are assumed to have been 'cleaned' in that leading/trailing
            edges alternate across the slice (i.e. no consecutive edges of the same type) and the
            first edge is the inner edge and is a trailing edge and
            the last edge is the outer edge and is a is leading edge,
            populates slice.pulses and returns the modified slice
            """

        inner = slice.points[0].where[1]
        outer = slice.points[-1].where[1]
        rings_width = outer - inner
        slice.pulses = []
        pulse = None
        for point in slice.points:
            if point.dead:
                continue
            if point.type == Scan.LEADING_EDGE:
                if pulse is None:
                    self._log('first edge is a leading edge!', fatal=True)
                # this is the end of the low for the current pulse
                pulse.head = (point.where[1] - pulse.head) / rings_width
                pulse.top = point.where[1]
            elif pulse is None:
                # this is the inner trailing edge, it marks the start of the first pulse
                pulse = Scan.Pulse(point.where[1])
                slice.pulses.append(pulse)
            else:
                # this is the end of the high for the current pulse and the start of the low for the next
                pulse.top = (point.where[1] - pulse.top) / rings_width
                pulse = Scan.Pulse(point.where[1])
                slice.pulses.append(pulse)
        # fill in the tails of each pulse from the head of the next
        # NB: the last pulse is the incomplete one from the last trailing edge to the final outer edge,
        #     we only want the head from that and its there, so OK
        for p in range(len(slice.pulses)-1):
            this_pulse = slice.pulses[p]
            next_pulse = slice.pulses[p+1]
            this_pulse.tail = next_pulse.head

        # the last (partial) pulse is to the outer edge and not wanted
        del slice.pulses[-1]

        return slice

    def _pulse_error(self, pulse, ideals):
        """ calculate the pulse error betw`een actual and ideal pulse component lengths,
            each ideal consists of a list of ratios and the classification to assign if those ratios match,
            this is the *only* function that knows the structure of the ideals presented here
            """

        # determine error based on sizes relative to one ring width,
        # the pulse component lengths are currently relative to the whole target width
        # which is 5 rings (inner black, data 1, data 2, data 2, outer black), so each
        # component relative to 1 ring is that measured multiplied by 5
        rings = Scan.NUM_RINGS - 3             # using this to show our dependence on it
        head_length = (pulse.head * rings)
        top_length = pulse.top * rings
        tail_length = (pulse.tail * rings)

        # Note: various error functions have been tried,
        #       this very simple average of the square of the differences works best
        best_error = None
        for ideal in ideals:
            head_error = ideal[0] - head_length
            head_error *= head_error

            top_error = ideal[1] - top_length
            top_error *= top_error

            tail_error = ideal[2] - tail_length
            tail_error *= tail_error

            error = (head_error + top_error + tail_error) / 3  # use average error of the components
            if best_error is None or error < best_error:
                best_error = error

        return best_error

    def _measure_pulses(self, slice):
        """ measure the pulses in the given slice,
            pulse component lengths range between 1 and 3 in an 'ideal' image,
            variations from this ideal are an error measure (calculated later),
            an empty list indicates no pulses,
            ideal pulse configurations are:

                 inner edge----+                                  +----outer edge
                               | Head                        Tail |
                               V<---->...                ...<---->V
            [0, 0, 0]    ------+                                  +------
                               |                                  |           zero pulses is unambiguous
                               +------+------+------+------+------+
                                                       Top
                                                    +<---->+
            [0, 0, 1]    ------+                    +------+      +------     pulse, head/tail, head/top, top/tail
                               |                    |      |      |           3:1:1     3:1   3:1   1:1
                               +------+------+------+      +------+

            [0, 1, 0]    ------+             +------+             +------
                               |             |      |             |           2:1:2     1:1   2:1   1:2
                               +------+------+      +------+------+

            [0, 1, 1]    ------+             +------+------+      +------
                               |             |             |      |           2:2:1     2:1   1:1   2:1
                               +------+------+             +------+

            [1, 0, 0]    ------+      +------+                    +------
                               |      |      |                    |           1:1:3     1:3   1:1   1:3
                               +------+      +------+------+------+

            [1, 0, 1]    ------+      +------+      +------+      +------
                               |      |      |      |      |      |           1:1:1 * 2 1:1   1:1   1:1
                               +------+      +------+      +------+

            [1, 1, 0]    ------+      +------+------+             +------
                               |      |             |             |           1:2:2     1:2   1:2   1:1
                               +------+             +------+------+

                         ------+      +------+-----+-------+      +------
            [1, 1, 1]          |      |                    |      |           1:3:1     1:1   1:3   3:1
                               +------+      +     +       +------+
            we test every pulse against every possibility and pick the one with the
            least deviation from these ideals,
            populates pulse.type and pulse.error and returns the number of pulses
            """

        ratios = [(Scan.THREE_ONE_ONE_RATIOS, Scan.THREE_ONE_ONE),
                  (Scan.TWO_ONE_TWO_RATIOS, Scan.TWO_ONE_TWO),
                  (Scan.TWO_TWO_ONE_RATIOS, Scan.TWO_TWO_ONE),
                  (Scan.ONE_ONE_THREE_RATIOS, Scan.ONE_ONE_THREE),
                  (Scan.ONE_ONE_ONE_RATIOS, Scan.ONE_ONE_ONE),
                  (Scan.ONE_TWO_TWO_RATIOS, Scan.ONE_TWO_TWO),
                  (Scan.ONE_THREE_ONE_RATIOS, Scan.ONE_THREE_ONE)]

        # find the best match
        for pulse in slice.pulses:
            errors = []
            for ratio in range(len(ratios)):
                errors.append((self._pulse_error(pulse, ratios[ratio][0]), ratios[ratio][1]))
            errors.sort(key=lambda e: e[0])

            # record best 2 options
            pulse.error = errors[0][0]
            pulse.type = errors[0][1]

            pulse.error2 = errors[1][0]
            pulse.type2 = errors[1][1]

        return len(slice.pulses)

    def _find_slice_boundaries(self, target, slices: List[Slice], direction) -> List[Boundary]:
        """ find all the boundaries in the given set of slices in the requested direction,
            direction is TOP_DOWN if want ring boundaries, or LEFT_TO_RIGHT if want bit boundaries,
            boundaries are co-ords of the most likely transitions for each of the expected edges
            (black->white and white->black)
            """

        # region prepare...
        max_x, max_y = target.size()
        if direction == Scan.TOP_DOWN:
            nominal_span = max_y / Scan.NUM_RINGS
            num_edges = Scan.NUM_RINGS - 2  # we will not see the inner and outer white rings
            max_coord = max_y
            point_coord = 1
            prefix = 'rings'
        else:
            nominal_span = max_x / Scan.NUM_BITS
            num_edges = Scan.NUM_BITS
            max_coord = max_x
            point_coord = 0
            prefix = 'bits'
        nominal_gap = nominal_span * nominal_span
        # endregion

        # region count edges for every coord...
        leading_edges = [0 for _ in range(max_coord)]
        trailing_edges = [0 for _ in range(max_coord)]
        for slice in slices:
            for point in slice.points:
                if point.type == Scan.LEADING_EDGE:
                    leading_edges[point.where[point_coord]] += 1
                else:
                    trailing_edges[point.where[point_coord]] += 1
        # endregion
        # region determine bits boundaries...
        edges = []
        for x in range(max_coord):
            if leading_edges[x] > 0 or trailing_edges[x] > 0:
                edges.append(Scan.Boundary(x, leading_edges[x], trailing_edges[x]))
        if self.logging:
            self._log('{}: found {} edges:'.format(prefix, len(edges)))
            for edge in edges:
                self._log('    {}'.format(edge))
        if len(edges) > num_edges:
            # region go too many edges, knock out single edge samples...
            if self.logging:
                header = '{}: got {} edges, want {}, dropping singletons:'.format(prefix, len(edges), num_edges)
            for e in range(len(edges) - 1, -1, -1):
                edge = edges[e]
                if edge.leading_edges + edge.trailing_edges == 1:
                    if self.logging:
                        if header is not None:
                            self._log(header)
                            header = None
                        self._log('    {}'.format(edge))
                    del edges[e]
            # endregion
        if self.logging:
            header = '{}: got {} edges, want {}, merging:'.format(prefix, len(edges), num_edges)
        while len(edges) > num_edges:
            # region got too many edges, merge two closest edges, repeat until not got too many...
            min_gap = 0
            min_at = None
            last_at = len(edges) - 1
            last_edge = edges[last_at]
            for e in range(len(edges)):
                next_edge = edges[e]
                gap = self._get_gap((last_edge.where, 0), (next_edge.where, 0), max_coord)
                if min_at is None or gap < min_gap:
                    min_at = [last_at, e]
                    min_gap = gap
                last_at = e
                last_edge = edges[last_at]
            last_edge = edges[min_at[0]]
            next_edge = edges[min_at[1]]
            # we merge with a bias towards the highest edge count
            last_count = last_edge.leading_edges + last_edge.trailing_edges
            next_count = next_edge.leading_edges + next_edge.trailing_edges
            all_counts = last_count + next_count
            gap = math.sqrt(min_gap)
            offset = (next_count / all_counts) * gap
            merged_where = (last_edge.where + offset) % max_coord
            merged_edge = Scan.Boundary(merged_where,
                                        last_edge.leading_edges + next_edge.leading_edges,
                                        last_edge.trailing_edges + next_edge.trailing_edges)
            edges[min_at[0]] = merged_edge
            del edges[min_at[1]]
            if self.logging:
                if header is not None:
                    self._log(header)
                    header = None
                self._log('    ({})-({}) gap is {:.2f}, merging as {}'.
                          format(last_edge, next_edge, math.sqrt(min_gap), merged_edge))
            # endregion
        if self.logging:
            header = '{}: got {} edges, want {}, inserting:'.format(prefix, len(edges), num_edges)
        while len(edges) < num_edges:
            # region got too few edges, insert an edge in the biggest gap, the inserted width is the nominal span...
            if len(edges) > 0:
                max_gap = 0
                max_at = None
                last_edge = edges[-1]
                for e in range(len(edges)):
                    next_edge = edges[e]
                    gap = self._get_gap((last_edge.where, 0), (next_edge.where, 0), max_coord)
                    if max_at is None or gap > max_gap:
                        max_at = [e, next_edge.where]
                        max_gap = gap
                    last_edge = next_edge
            else:
                # we've got an empty image
                max_at = [0, max_x - 1]
                max_gap = (max_x - 1) * (max_x - 1)
            if max_gap > (nominal_gap * nominal_gap):
                # big gap, add a nominal span
                length = nominal_span
            else:
                # gap small, just halve it
                length = math.sqrt(max_gap) / 2
            edge = Scan.Boundary(int((max_at[1] - length) % max_coord), 0, 0)
            edges.insert(max_at[0], edge)
            if self.logging:
                if header is not None:
                    self._log(header)
                    header = None
                self._log('    ({})-({}) gap is {:.2f}, inserting {} at {}'.
                          format(edges[max_at[0]], max_at[1], math.sqrt(max_gap), edge, max_at[0]))
            # endregion
        # endregion

        # normalise co-ords
        for edge in edges:
            edge.where = int(round(edge.where)) % max_coord
            edge.extent = int(round(edge.extent)) % max_coord

        if self.logging:
            self._log('{}: final {} edges:'.format(prefix, len(edges)))
            for edge in edges:
                self._log('    {}'.format(edge))

        return edges

    def _make_bits_from_boundaries(self, target,
                                   bit_boundaries: List[Boundary], ring_boundaries: List[Boundary]) -> List[SliceBits]:
        """ given bit and ring boundaries decode the three data ring bits therein """

        Scan.SliceBits.reset()                     # reset bit counter

        max_x, max_y = target.size()
        bit_length = int(max_x / Scan.NUM_BITS)

        bits = []
        for bit in range(Scan.NUM_BITS):
            start_x = bit_boundaries[(bit - 1) % Scan.NUM_BITS].where
            end_x = bit_boundaries[bit].where
            if start_x != end_x:
                # add a leading border
                start_x = (start_x + 1) % max_x
                if start_x != end_x:
                    # add a trailing border
                    end_x = (end_x - 1) % max_x
            slice_bits = Scan.SliceBits(where=bit * bit_length, length=bit_length)  # where and length only used for diagnostic image
            for ring in range(1, Scan.NUM_DATA_RINGS + 1):
                whites = 0
                blacks = 0
                x = start_x
                start_y = ring_boundaries[ring].where
                end_y = ring_boundaries[ring + 1].where
                if start_y != end_y:
                    # add a leading border
                    start_y += 1
                    if start_y != end_y:
                        # add a trailing border
                        end_y -= 1
                while True:
                    y = start_y
                    while True:
                        pixel = target.getpixel(x, y)
                        if pixel is None:
                            continue
                        if pixel > MID_LUMINANCE:
                            whites += 1
                        else:
                            blacks += 1
                        if y == end_y:
                            break
                        y += 1
                    if x == end_x:
                        break
                    x = (x + 1) % max_x
                total = whites + blacks
                if whites / total > 0.6:  # ToDo: make 0.6 a tuning constant
                    # got a one bit
                    slice_bits.bits.append(1)
                    slice_bits.error += (blacks / total)
                elif whites / total < 0.3:  # ToDo: make 0.3 a tuning constant
                    # got a zero bit
                    slice_bits.bits.append(0)
                    slice_bits.error += (whites / total)
                else:
                    # its ambiguous
                    slice_bits.bits.append(None)
                    slice_bits.error += 0.5
            slice_bits.error /= Scan.NUM_DATA_RINGS
            bits.append(slice_bits)

        return bits

    def _decode_slices(self, slices: List[Slice]) -> List[Slice]:
        """ determine the bits represented by each slice by an analysis of pulses,
            the relevant part of a slice extends from the inner edge to the outer edge,
            in between there can be zero, one or two pulses, zero pulses can only be 000,
            two pulses can only 101, one pulse can be the other six possibilities,
            returns the modified slices with the bits and pulses properties set,
            """

        # for every slice determine the bits represented
        for slice in slices:
            pulses = self._measure_pulses(self._make_pulses(slice))
            if pulses == 0:
                slice.bits = [0, 0, 0]
                slice.error = 0
                continue
            if pulses > 2:
                # at least one of these is junk, drop the ones with the biggest error
                if self.logging:
                    self._log('slices: dropping excess pulses in slice {}'.format(slice))
                    for pulse in slice.pulses:
                        self._log('    {}'.format(pulse))
                while pulses > 2:
                    worst_error = 0
                    worst_at = None
                    for pulse in range(pulses):
                        if slice.pulses[pulse].error > worst_error:
                            worst_error = slice.pulses[pulse].error
                            worst_at = pulse
                    if self.logging:
                        self._log('        dropping {}'.format(slice.pulses[worst_at]))
                    del slice.pulses[worst_at]
                    pulses -= 1
            if pulses == 2:
                # filter out junk, to be a true 2 pulse both must be 1:1:1
                if slice.pulses[0].type == Scan.ONE_ONE_ONE and slice.pulses[1].type == Scan.ONE_ONE_ONE:
                    # its a goody
                    slice.bits = [1, 0, 1]
                    slice.error = slice.pulses[0].error + slice.pulses[1].error
                    continue
                # one or both are not 1:1:1, drop the one with biggest error and proceed as 1 pulse
                if self.logging:
                    self._log('slices: dropping bad pulse in slice {}'.format(slice))
                    for pulse in slice.pulses:
                        self._log('    {}'.format(pulse))
                if slice.pulses[0].error > slice.pulses[1].error:
                    if self.logging:
                        self._log('        dropping {}'.format(slice.pulses[0]))
                    del slice.pulses[0]
                else:
                    if self.logging:
                        self._log('        dropping {}'.format(slice.pulses[1]))
                    del slice.pulses[1]
                pulses -= 1
            # pulses == 1
            pulse = slice.pulses[0]
            slice.error = pulse.error
            if pulse.type == Scan.THREE_ONE_ONE:
                slice.bits = [0, 0, 1]
            elif pulse.type == Scan.TWO_ONE_TWO:
                slice.bits = [0, 1, 0]
            elif pulse.type == Scan.TWO_TWO_ONE:
                slice.bits = [0, 1, 1]
            elif pulse.type == Scan.ONE_ONE_THREE:
                slice.bits = [1, 0, 0]
            elif pulse.type == Scan.ONE_TWO_TWO:
                slice.bits = [1, 1, 0]
            elif pulse.type == Scan.ONE_THREE_ONE:
                slice.bits = [1, 1, 1]
            elif pulse.type == Scan.ONE_ONE_ONE:
                # this is only legal when we've got two pulses, treat like 2:1:2 with an error to match
                if self.logging:
                    self._log('slices: bad single pulse in slice {}'.format(slice))
                    self._log('    interpreting 1:1:1 as 2:1:2 {}'.format(pulse))
                slice.bits = [0, 1, 0]
                pulse.error = self._pulse_error(pulse, Scan.TWO_ONE_TWO_RATIOS)  # set new pulse error
                slice.error = pulse.error                                        # put the new error in
            else:
                raise Exception('unknown pulse type: {}'.format(pulse.type))

        if self.logging:
            self._log('slices: ({})'.format(len(slices)))
            for slice in slices:
                self._log('    {}'.format(slice))
                if len(slice.pulses) > 0:
                    for pulse in slice.pulses:
                        self._log('        {}'.format(pulse))

        return slices

    def _filter_slices(self, target, slices):
        """ given a set of slices filter out single sample sequences,
            this is removing 'noise' before we start accumulating similar bits,
            the updates slices list is returned,
            the target and direction parameters are purely for diagnostic image drawing,
            """

        if self.logging:
            header = 'slices: killing single sample slices'

        for slice in range(len(slices)):
            pred = slices[(slice - 1) % len(slices)]
            succ = slices[(slice + 1) % len(slices)]
            me = slices[slice % len(slices)]
            if me.bits != pred.bits and me.bits != succ.bits:
                # got a single sample, kill it
                me.dead = Scan.SINGLE_SAMPLE
                if self.logging:
                    if header is not None:
                        self._log(header)
                        header = None
                    self._log('    {}'.format(me))

        if self.save_images:
            grid = self._draw_slices(target, slices)
            self._unload(grid, '08-slices')

        return slices

    def _make_bits_from_slices(self, slices) -> List[SliceBits]:
        """ make the bit sequence across all slices
            from a consideration of the bits in each slice generate the bit sequence across the target,
            from the construction of the target we know there is a bit transition for every bit, thus
            a bit edge causes the bits of the ring to change, we detect this change and generate a bit
            list with the length of each sequence,
            returns the list of bits
            """

        Scan.SliceBits.reset()                     # reset bit counter

        # isolate and count similar bits (nb: must take note of x wrapping)
        bits = []
        for slice in slices:
            if slice.dead is not None:
                # this one has been killed, ignore it
                continue
            where = slice.points[0].where[0]
            if len(bits) == 0:
                # this is the first one
                bits.append(Scan.SliceBits(slice.bits, 1, slice.error, where))  # init a new bit sequence
            elif bits[-1].bits == slice.bits:
                # got another one the same
                bits[-1].length += 1                 # up the sequence length of this one
                bits[-1].error += slice.error
            else:
                # got a new one, finish the previous and start a new
                bits[-1].error /= bits[-1].length
                bits.append(Scan.SliceBits(slice.bits, 1, slice.error, where))  # init a new bit sequence
        # set error of the last one
        bits[-1].error /= bits[-1].length

        # see if last and first are the same (means the sequence wrapped)
        if len(bits) > 1 and bits[0].bits == bits[-1].bits:
            # they wrap, combine first with last and dump the first (so .where does not change)
            bits[-1].length += bits[0].length
            bits[-1].error = (bits[0].error + bits[-1].error) / 2
            del bits[0]

        if self.logging:
            self._log('bits: ({})'.format(len(bits)))
            for bit in bits:
                self._log('    {}'.format(bit))

        return bits

    def _filter_slice_bits(self, target, bits: List[SliceBits]):
        """ given the vector of bits decoded from the slices, filter out the junk and split merges,
            in an ideal target the length of the bits array given here will be the same as the number of
            bits in the code, but noise and distortion can lead to less or more, where two 'corners' meet
            there can be a gap or an overlap that looks like a different bit sequence, e.g.:
               #######.......      #########
               ##(1)##..(0)..      ##(1)####
               #######.......      #########
               .........#######    .......########
               ..(0)....##(1)##    ..(0)..###(1)##
               .........#######    .......########
                      ^^                   ^^
                      ||                   ||
                     gap                   overlap
            gaps look like short sequences of 0,0 and overlaps look like short sequences of 1,1,
            we analyse the sequence lengths here and chuck out excessively short ones,
            returns the list with the rejected bits marked as dead (it is up to the caller to remove them)
            """

        def set_dead(bad_bit, good_bit, reason):
            """ set the given bit sequence in the list of such sequences as dead """

            nonlocal max_x, killed, header

            bad_bit.dead = reason
            killed += 1

            if good_bit is not None:
                # merge error and length and adjust where
                # errors are an average across the sequence length
                bad_err = bad_bit.error * bad_bit.length
                good_err = good_bit.error * good_bit.length
                if ((good_bit.where + good_bit.length) % max_x) == bad_bit.where:
                    # leave good_bit where it is
                    pass
                elif ((bad_bit.where + bad_bit.length) % max_x) == good_bit.where:
                    good_bit.where = bad_bit.where
                else:
                    # what to do with where?
                    pass
                good_bit.length += bad_bit.length
                good_bit.error = (bad_err + good_err) / good_bit.length

            if self.logging:
                if header is not None:
                    self._log(header)
                    header = None
                self._log('    killing {}, merging with {}'.format(bad_bit, good_bit))

        max_x, _ = target.size()
        max_length = int(round((max_x / Scan.NUM_BITS) * Scan.MAX_BIT_SEQUENCE_LENGTH))
        min_length = int(round((max_x / Scan.NUM_BITS) * Scan.MIN_BIT_SEQUENCE_LENGTH))

        # ToDo: re-jig to build ring based bit sequences from all the slices, then mess with those,
        #       including speculative decodes and pick best
        if self.logging:
            header = 'bits: got {} when only want {}, killing short sequences:'.format(len(bits), Scan.NUM_BITS)
        killed = 0
        while (len(bits) - killed) > Scan.NUM_BITS:
            worst = None
            worst_at = None
            reason = None
            for bit_num in range(len(bits)):
                bit = bits[bit_num]
                if bit.dead is not None:
                    # already marked as dead, ignore it
                    continue
                if worst is None:
                    # set first none dead we find as the worst
                    worst = bit
                    worst_at = bit_num
                    reason = Scan.ONLY_CHOICE
                elif bit.length < worst.length:
                    # found a shorter one, note it
                    worst = bit
                    worst_at = bit_num
                    reason = Scan.SHORTEST
                elif bit.length > worst.length:
                    # this is longer, keep it
                    pass
                elif bit.error > worst.error:
                    # same length but worse error, note this one
                    worst = bit
                    worst_at = bit_num
                    reason = Scan.WORST_ERROR
            # merge this about to be killed one with its shortest neighbour
            # find nearest non-dead predecessor
            predecessor = worst_at
            for bit_num in range(len(bits)-1, -1, -1):
                predecessor = (predecessor - 1) % len(bits)
                if bits[predecessor].dead is None:
                    break
            # find nearest non-dead successor
            successor = worst_at
            for bit_num in range(len(bits)):
                successor = (successor + 1) % len(bits)
                if bits[successor].dead is None:
                    break
            # merge with the most common neighbour
            if bits[predecessor].dead is not None:
                if bits[successor].dead is not None:
                    # both neighbours dead (means everything is dead!)
                    neighbour = None
                else:
                    # got a successor but no predecessor
                    neighbour = bits[successor]
            elif bits[successor].dead is not None:
                # got a predecessor but no successor
                neighbour = bits[predecessor]
            else:
                # got both, merge with the shortest
                if bits[predecessor].length < bits[successor].length:
                    neighbour = bits[predecessor]
                else:
                    neighbour = bits[successor]
            set_dead(worst, neighbour, reason)
            continue

        if self.logging:
            header = 'bits: got {} when need {}, splitting long sequences:'.format(len(bits), Scan.NUM_BITS)
        added = 0
        while len(bits) < Scan.NUM_BITS:
            # find longest sequence
            longest = None
            longest_at = None
            for bit_num in range(len(bits)):
                bit = bits[bit_num]
                if bit.dead is not None:
                    # marked as dead, ignore it
                    continue
                if longest is None or bit.length > longest:
                    longest = bit.length
                    longest_at = bit_num
            if longest is None or longest < (2 * min_length):
                # this means the bits list is empty or no longer splittable
                break
            # we split a minimum sized sequence off this long one
            split_size = min_length
            long_bit = bits[longest_at]
            if self.save_images:
                # add a dummy dead bit at the join so we can see it in the bits image
                dummy_bit = Scan.SliceBits(bits=(0, 0, 0), where=(long_bit.where + split_size) % max_x)
                dummy_bit.dead = Scan.TOO_LONG
                bits.insert(longest_at + 1, dummy_bit)
                longest_at += 1
            long_bit.length -= split_size
            new_bit = Scan.SliceBits(long_bit.bits, split_size, long_bit.error,
                                     (long_bit.where + split_size) % max_x)
            bits.insert(longest_at + 1, new_bit)
            added += 1
            if self.logging:
                if header is not None:
                    self._log(header)
                    header = None
                self._log('    splitting {}, split off {}'.format(long_bit, new_bit))

        # check bit lengths are reasonable (after merging)
        good_bits = []
        for bit in bits:
            if bit.dead is not None:
                # already rejected
                continue
            if bit.length > max_length:
                # NB: this now takes us under the number of bits we want, so the whole target
                #     will be rejected - good, it helps to prevent false positives later
                bit.dead = Scan.TOO_LONG
                killed += 1
                if self.logging:
                    self._log('bits: bit too long ({}, limit is {}), killing {}'.
                              format(bit.length, max_length, bit))
            elif bit.length < min_length:
                # NB: as above wrt rejecting
                bit.dead = Scan.TOO_SHORT
                killed += 1
                if self.logging:
                    self._log('bits: bit too short ({}, limit is {}), killing {}'.
                              format(bit.length, min_length, bit))
            else:
                good_bits.append(bit)

        if self.logging:
            if killed > 0 or added > 0:
                self._log('bits: (killed {}, added {} in {}, leaving {}):'.
                                     format(killed, added, len(bits),                           len(bits)-killed+added))
                for bit in bits:
                    if bit.dead is None:
                        self._log('    {}'.format(bit))

        if self.save_images:
            # draw an image of the bits showing which were rejected/added
            grid = self._draw_bits(target, bits)
            # show the image
            self._unload(grid, '09-bits')

        if len(good_bits) != Scan.NUM_BITS:
            if self.logging:
                self._log('bits: need {} bits, found {}'. format(Scan.NUM_BITS, len(good_bits)))
            reason = 'only {} bits'.format(len(good_bits))
        else:
            reason = None

        return good_bits, reason

    def _get_target_bits(self, target, inner_edge, outer_edge):
        """ get all the bits in the context target,
            inner/outer edge define the limits of the code within the target,
            returns a vector of SliceBits
            """

        # ToDo: consider looking at the binary image horizontally and vertically?
        #       the problem with the horizontal approach is that a missed bit shifts all
        #       subsequent bits by 1 which renders the whole code un-readable, try finding
        #       edges in both directions to make bit squares then count pixels in those

        compressed = self._compress(target, inner_edge, outer_edge)
        slices = self._make_slices(compressed, Scan.TOP_DOWN)
        # ToDo: HACK START
        strips = self._make_slices(compressed, Scan.LEFT_TO_RIGHT)
        patches = self._make_patches(compressed, strips, slices)
        bit_boundaries = self._find_slice_boundaries(patches, strips, direction=Scan.LEFT_TO_RIGHT)
        ring_boundaries = self._find_slice_boundaries(patches, slices, direction=Scan.TOP_DOWN)
        grid = self._draw_boundaries(patches, bit_boundaries, ring_boundaries)
        self._unload(grid, '08-edges-2')
        grid = self._draw_slices(compressed, strips)
        self._unload(grid, '08-strips')
        grid = self._draw_slices(patches, strips)
        self._unload(grid, '08-strips-2')
        bits = self._make_bits_from_boundaries(patches, bit_boundaries, ring_boundaries)
        grid = self._draw_bits(patches, bits)
        self._unload(grid, '09-bits-2')
        code = self._decode_bits(bits)
        number, doubt, digits = self.decoder.unbuild(code)
        self._log('number={}, doubt={}, digits={}'.format(number, doubt, digits), console=True)
        # ToDo: HACK END
        slices = self._decode_slices(slices)
        slices = self._filter_slices(compressed, slices)
        bits = self._make_bits_from_slices(slices)
        good_bits = self._filter_slice_bits(compressed, bits)

        return good_bits

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
            projected, orig_radius = self._project(self.centre_x, self.centre_y, blob.size)  # this does not fail

            # do the inner/outer edge detection
            measured = self._measure(projected, orig_radius)
            flattened = self._flatten(measured)
            reason = flattened.reason
            if reason is not None:
                # failed - this means some constraint was not met (its already been logged)
                if self.save_images:
                    # add to reject list for labelling on the original image later
                    rejects.append(Scan.Reject(self.centre_x, self.centre_y, blob_size, None, reason))
                continue
            target = flattened.target
            inner_edge = flattened.inner_edge
            outer_edge = flattened.outer_edge
            target_size = flattened.size

            # get the slice bits
            bits, reason = self._get_target_bits(target, inner_edge, outer_edge)
            if reason is not None:
                if self.save_images:
                    rejects.append(Scan.Reject(self.centre_x, self.centre_y, blob_size, None, reason))
                continue

            targets.append(Scan.Target(self.centre_x, self.centre_y, blob_size, target_size, target, bits))

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
                colour = Scan.BLUE  # blue
                labels = self.transform.label(labels, (x, y, blob_size), colour)
                # show reject reason
                colour = Scan.RED  # red
                labels = self.transform.label(labels, (x, y, ring), colour, '{:.0f}x{:.0f}y {}'.format(x, y, reason))
        else:
            labels = None

        return targets, labels

    def _decode_bits(self, bits):
        # in bits we have a list of bit sequences across the data rings, i.e. 15 x 3
        # we need to rotate that to 3 x 15 to present it to our decoder
        code = [[None for _ in range(Scan.NUM_BITS)] for _ in range(3)]
        for bit in range(Scan.NUM_BITS):
            rings = bits[bit].bits
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
                self._log('image {} does not contain any target candidates'.format(self.original.source))
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

            if self.save_images:
                grid = image

            code = self._decode_bits(bits)
            number, doubt, digits = self.decoder.unbuild(code)

            # add this result
            numbers.append(Scan.Detection(number, doubt, self.centre_x, self.centre_y, target_size, blob_size, digits))

            if self.logging:
                number = numbers[-1]
                self._log('number:{}, bits:{}'.format(number.number, number.digits), number.centre_x, number.centre_y)
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

    def _log(self, message, centre_x=None, centre_y=None, fatal=False, console=False):
        """ print a debug log message
            centre_x/y are the co-ordinates of the centre of the associated blob, if None use decoding context
            centre_x/y of 0,0 means no x/y identification in the log
            iff fatal is True an exception is raised, else the message is just printed
            iff console is True the message is logged to console regardless of other settings
            """
        if centre_x is None:
            centre_x = self.centre_x
        if centre_y is None:
            centre_y = self.centre_y
        if centre_x > 0 and centre_y > 0:
            message = '{} {:.0f}x{:.0f}y - {}'.format(self._log_prefix, centre_x, centre_y, message)
        else:
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

    def _draw_lines(self, source, lines, colour=RED):
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

    def _draw_below(self, source, threshold, colour=RED):
        """ re-draw every pixel below the black threshold in the given colour,
            the given source must be a greyscale edges image (as returned by _get_transition),
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
                grey_pixel = self._get_threshold_pixel(source, x, y, threshold[x])
                if grey_pixel is None:
                    continue
                if grey_pixel <= 0:
                    image_pixel = 0 - grey_pixel
                    colour_pixel = (image_pixel * scale[0], image_pixel * scale[1], image_pixel * scale[2])
                else:
                    colour_pixel = (grey_pixel, grey_pixel, grey_pixel)
                target.putpixel(x, y, colour_pixel)
        return target

    def _draw_slices(self, target, slices):
        """ draw slices debug image,
            green for leading edge, blue for trailing, red-line for dead slices
            """
        max_x, max_y = target.size()
        grid = target
        leading_points = []
        trailing_points = []
        dead_slices = []
        for slice in slices:
            if slice.dead is not None:
                where = slice.points[0].where[0]  # NB: a dead slice has at least 1 point
                dead_slices.append([where, 0, where, max_y - 1])
            for point in slice.points:
                if point.type == Scan.LEADING_EDGE:
                    leading_points.append([point.where[0], [point.where[1]]])
                else:
                    trailing_points.append([point.where[0], [point.where[1]]])
        grid = self._draw_lines(grid, dead_slices, Scan.PALE_RED)
        grid = self._draw_plots(grid, trailing_points, None, Scan.BLUE)
        grid = self._draw_plots(grid, leading_points, None, Scan.GREEN)
        return grid

    def _draw_bits(self, target, bits):
        """ draw an image of the bits showing which were rejected/added """

        max_x, _ = target.size()

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

        def draw_bits(grid, bits, ring_width, max_x, dead=None):
            """ draw the given dead or non-dead bits """

            for bit in bits:
                start_x = bit.where
                end_x = (start_x + bit.length - 1) % max_x
                start_y = ring_width * 3
                if bit.dead is not None:
                    if dead is None or dead:
                        one_colour = Scan.PALE_RED
                        zero_colour = Scan.GREY
                    else:
                        # do not want dead ones
                        continue
                else:
                    if dead is None or not dead:
                        one_colour = Scan.WHITE
                        zero_colour = Scan.BLACK
                    else:
                        # do not want non-dead ones
                        continue
                for sample in bit.bits:
                    if sample is None:
                        # draw a grey block here
                        draw_block(grid, start_x, end_x, start_y, ring_width, max_x, Scan.GREY)
                    elif sample > 0:
                        # draw a white block here
                        draw_block(grid, start_x, end_x, start_y, ring_width, max_x, one_colour)
                    else:
                        # draw a black block here
                        draw_block(grid, start_x, end_x, start_y, ring_width, max_x, zero_colour)
                    start_y += ring_width

        # make an empty (i.e. black) colour image to load our bits into
        ring_width = int(round((max_x / Scan.NUM_BITS)))  # this makes the ideal bits visually square
        max_y = ring_width * Scan.NUM_RINGS
        grid = target.instance()
        grid.new(max_x, max_y, MIN_LUMINANCE)
        grid.incolour()

        # draw the inner and outer white rings
        draw_block(grid, 0, max_x - 1, 0, ring_width * 2, max_x, Scan.WHITE)
        draw_block(grid, 0, max_x - 1, ring_width * 7, ring_width, max_x, Scan.WHITE)

        # draw the too-long lines (to show what was added)
        for bit in bits:
            if bit.dead == Scan.TOO_LONG:
                draw_block(grid, bit.where, bit.where + bit.length - 1, ring_width, int(ring_width / 2),
                           max_x, Scan.PALE_RED)

        # draw data rings initially in blue so can see gaps where rejected slices where
        # ('cos they will not be over drawn in black or white or red)
        draw_block(grid, 0, max_x - 1, ring_width * 3, ring_width * 3, max_x, Scan.BLUE)

        # draw the good bits first, then the dead (so overlaps show the dead areas)
        draw_bits(grid, bits, ring_width, max_x, dead=False)
        draw_bits(grid, bits, ring_width, max_x, dead=True)

        return grid

    def _draw_boundaries(self, target, bit_boundaries, ring_boundaries):
        """ draw the boundary detections debug image """

        max_x, max_y = target.size()
        grid = target
        if bit_boundaries is not None:
            bit_lines = []
            for bit_boundary in bit_boundaries:
                x = bit_boundary.where
                while True:
                    bit_lines.append([x, 0, x, max_y - 1])
                    if x == bit_boundary.extent:
                        break
                    x = (x + 1) % max_x
            grid = self._draw_lines(grid, bit_lines, Scan.PALE_RED)
        if ring_boundaries is not None:
            ring_lines = []
            for ring_boundary in ring_boundaries:
                y = ring_boundary.where
                while True:
                    ring_lines.append([0, y, max_x - 1, y])
                    if y == ring_boundary.extent:
                        break
                    y += 1
            grid = self._draw_lines(grid, ring_lines, Scan.PALE_RED)
        return grid


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
                      format(Codec.BITS, min_num, self.max_num, self.codec.min_edges, self.codec.max_edges))

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
        self._log('')
        self._log('******************')
        self._log('Check build/unbuild from {} to {}'.format(self.min_num, self.max_num))
        try:
            good = 0
            doubted = 0
            fail = 0
            bad = 0
            if self.max_num is not None:
                for n in range(self.min_num, self.max_num + 1):
                    rings = self.codec.build(n)
                    samples = [[] for _ in range(len(rings))]
                    for ring in range(len(rings)):
                        word = rings[ring]
                        for bit in range(self.bits):
                            # NB: Being encoded big-endian (MSB first)
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
            plus those with the most 1's and the most 0's and alternating 1's and 0's and
            N random numbers to make the set size up to that given
            """
        if self.max_num is None:
            return []
        if size < 2:
            size = 2
        if self.max_num - self.min_num <= size:
            return [num for num in range(self.min_num, self.max_num + 1)]
        max_ones = -1
        max_zeroes = -1
        num_set = [self.min_num, self.max_num]
        if presets is not None:
            num_set += presets
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
    test_ring_width = 32

    # cell size is critical,
    # going small in length creates edges that are steep vertically, going more takes too long
    # going small in height creates edges that are too small and easily confused with noise
    test_scan_cells = (8, 8)

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
    test_num_set = test.test_set(10, [161, 191])

    # test.coding()
    # test.decoding()
    # test.circles()
    # test.code_words(test_num_set)
    # test.codes(test_codes_folder, test_num_set, test_ring_width)
    # test.rings(test_codes_folder, test_ring_width)

    # test.scan_codes(test_codes_folder)
    # test.scan_media(test_media_folder)

    # test.scan(test_codes_folder, [101], 'test-code-101.png')

    # test.scan(test_media_folder, [101], 'photo-101.jpg')
    test.scan(test_media_folder, [101, 102, 182, 247, 301, 424, 448, 500, 537, 565], 'photo-101-102-182-247-301-424-448-500-537-565-v4.jpg')

    del (test)  # needed to close the log file(s)


if __name__ == "__main__":
    verify()
