import os
import glob

import cv2
import numpy as np
import random
import math
import traceback

# ToDo:
#   1. use opencv threshold instead of DIY and tweak 'followers' to follow mid-point not brightest
#   2. create a new set of test photos with no timing ring

""" coding scheme
    
    This coding scheme is intended to be easy to detect and robust against noise, distortion and luminance artifacts.
    The code is designed for use as competitor numbers in club level fell races, as such the number range
    is limited to between 101 and 999 inclusive, i.e. always a 3 digit number and max 899 competitors.
    The actual implementation limit may be less depending on coding constraints chosen (parity, edges, code size).
    The code is circular (so orientation does not matter) and consists of:
        a solid circular centre of 'white' with a radius 2R,
        surrounded by a solid ring of 'black' and width R,
        surrounded by 3 concentric data rings of width R and divided into N (typically 14..16) equal segments,
        enclosed by a solid 'black' ring of width R and, finally, a solid 'white' ring of radius R. 
    Total radius is 8R.
    The radial segments include 3 data rings, which are used as a triple redundant data bit copy, the middle
    data ring is inverted (to spectrum spread the luminance).
    Segments around the ring provide for N bits. Each ring is skewed clockwise by n bits (a kind of interleave).
    A one-bit is white (i.e. high luminance) and a zero-bit is black (i.e. low luminance).
    An alignment marker of 0110 (4 bits) is used to identify the start/stop of the code word encoded in a ring.
    The remaining bits are the payload (big-endian) and must not contain the alignment marker and also must
    not end in 011 and must not start with 110 (else they look like 0110 when adjacent to the alignment marker).
    The payload is N data-bits and an optional parity bit and may be constrained to have at least N bit transitions.
    The 3 payload rings are skewed round by n bits so that the alignment marker is not aligned radially. This 
    evens out the luminance levels and also has the effect of interleaving the redundant bits.
    The central 'bullseye' candidates are detected using a 'blob detector' (via opencv).
    The bits are extracted by thresholding. There are two thresholds, min-grey and max-grey, which are determined
    by dividing the luminance range detected across the radius for each bit by three.  The result of the
    thresholding is 3 levels: black (0), white (1), grey (?).  The bit skew between rings is known, so all 
    three rings can be decoded into these three levels. They are then decoded as follows:
        three 0's             = 0
        three 1's             = 1
        two zeroes + one grey = probably 0
        two ones + one grey   = probably 1
        one zero + two grey   = maybe 0
        one one + two grey    = maybe 1
        anything else         = junk (a give-up condition)
    This results in 7 states for each bit: 000, 00?, 0??, 111, 11?, 1??, ???.
    The more ?'s the less confident the result, three ?'s is junk.
    Alignment markers are sought in most confident to least confident order on any bit set extracted.
    
    Note: The decoder can cope with several targets in the same image. It uses the relative size of
          the targets to present the results in distance order, nearest (i.e. biggest) first. This is
          important to determine finish order.
                
    """

# colours
max_luminance = 255
min_luminance = 0
mid_luminance = (max_luminance - min_luminance) >> 1

# alpha channel
transparent = min_luminance
opaque = max_luminance

""" WARNING
    cv2's cor-ordinates are backwards IMHO, the 'x' co-ordinate is vertical and 'y' horizontal.
    Bonkers!
    The logic here uses 'x' horizontally and 'y' vertically, swapping as required when dealing with cv2
    """

class Codes:
    """ Encode and decode a number or a bit or a blob
        a number is a payload, it can be encoded and decoded
        a bit is a raw bit decoded from 3 blobs
        a blob is decoded from 3 luminance level samples
        this class encapsulates all the encoding and decoding and their constants
        """

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
    IS_NEITHER = 6

    def __init__(self, size, min_num, max_num, parity, edges):
        """ create the valid code set for a number in the range min_num..max_num for code_size
            a valid code is one where there are no embedded start/stop bits bits but contains at least one 1 bit,
            we create two arrays - one maps a number to its code and the other maps a code to its number.
            """

        # params
        self.size = size                                       # total ring code size in bits
        self.min_num = min_num                                 # minimum number we want to be able to encode
        self.max_num = max_num                                 # maximum number we want to be able to encode
        self.parity = parity                                   # None, 0 (even) or 1 (odd)
        self.edges = edges                                     # how many bit transitions we want per code
        self.skew = max(int(self.size / 3), 1)                 # ring to ring skew in bits
        self.marker_bits = 4                                   # number of bits in our alignment marker
        self.code_bits = self.size - self.marker_bits          # code word bits is what's left
        self.marker = 6 << self.code_bits                      # 0110 in MS 4 bits of code
        self.code_range = 1 << self.code_bits

        # build code tables
        self.codes = [None for _ in range(self.code_range)]    # set all invalid initially
        self.nums = [None for _ in range(self.max_num + 1)]    # ..
        num = self.min_num - 1                                 # last number to be encoded (so next is our min)
        for code in range(1, self.code_range):                 # start at 1 'cos 0 has no 1 bits
            if (code & 7) == 3:                                # LS 3 bits are 011
                # can't use this 'cos it'll look like the marker when elided with it
                pass
            # NB: The leading 110 rule is implicit in our embedded 0110 check as the right shift introduces
            #     a leading 0 bit by the time it gets to the LS 4 bits
            else:
                check = code
                for _ in range(self.code_bits):
                    if (check & 15) == 6:                      # LS 4 bits are 0110
                        # embedded alignment marker, so not a valid code
                        check = None
                        break
                    check >>= 1                                # NB: introduces a leading 0 (required, see above)
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
                            self.codes[code] = num                 # decode code as num
                            self.nums[num] = code                  # encode num as code
        self.num_limit = num

        # luminance level thresholds (white tends to bleed into black, so we make the black level bigger)
        self.white_width = 0.25                                # band width of white level within luminance range
        self.black_width = 0.5                                 # band width of black level within luminance range

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
            it returns 3 integers, the LS n bits are the code word and alignment marker, each word
            is rotated clockwise by n bits to give the required skew (see discussion above), ring 1
            is considered to be the outer ring, ring 3 the inner (just means unbuild() must be given
            rings in the same order), ring 2 is inverted (to spectrum spread the luminance).
            """
        code_word = self.encode(num)
        if code_word is None:
            return None
        # the bit shift is relative to the MSB, this matters when 3 * shift is not the code size
        # for each shift the LS n bits must be moved to the MS n bits
        mask = (1 << self.skew) - 1      # make to isolate the LS n bits
        shift = (self.size - (self.skew * 3)) + (self.skew * 2)
        r1 = self.marker + code_word
        r2 = (r1 >> self.skew) + ((r1 & mask) << shift)
        r3 = (r2 >> self.skew) + ((r2 & mask) << shift)
        return r1, (r2 ^ (int(-1))), r3        # middle ring is inverted (to spread the luminance)

    def ring_bits_pos(self, n):
        """ given a bit index return a list of the indices of all the same bits from each ring """
        n1 = n
        n2 = int((n1 + self.skew) % self.size)
        n3 = int((n2 + self.skew) % self.size)
        return [n1, n2, n3]

    def marker_bits_pos(self, n):
        """ given a bit index return a list of the indices of all the bits that would make a marker """
        return [int(pos % self.size) for pos in range(n, n+self.marker_bits)]

    def unbuild(self, samples, levels):
        """ given an array of 3 code-word rings with random alignment return the encoded number or None,
            each ring must be given as an array of blob values in bit number order,
            levels is a 2 x N array of luminance levels for each bit position that represent the white
            level and black level for that bit,
            returns the number (or None), the level of doubt and the bit classification for each bit
            """

        # step 1 - decode the 3 rings bits
        bits = [None for _ in range(self.size)]
        for n in range(self.size):
            rings = self.ring_bits_pos(n)
            bits[n] = self.blob(samples[0][rings[0]], levels[0][rings[0]], levels[1][rings[0]],
                                samples[1][rings[1]], levels[0][rings[1]], levels[1][rings[1]],
                                samples[2][rings[2]], levels[0][rings[2]], levels[1][rings[2]])

        # step 2 - find the alignment marker candidates (see discussion above)
        maybe_at = [[], [], [], [], []]  # 0..4 maybe possibilities, a list for each
        for n in range(self.size):
            marker = self.is_marker(n, bits)
            if marker is None:
                continue                 # no marker at this bit position, look at next
            # got a potential marker with marker maybe values, 0 == exact, 4 == all maybe
            maybe_at[marker].append(n)
        # maybe_at now contains a list of all possibilities for all maybe options

        # step 3 - extract all potential code words for each candidate alignment for each maybe level
        # any that yield more than one are crap and a give-up condition
        found = None
        for maybe in maybe_at:
            for n in maybe:
                # n is the next one we are going to try, demote all others
                word = [bit for bit in bits]   # make a copy
                code = self.extract_word(n, word)
                if code is not None:
                    if found is not None:
                        # got more than 1 - that's crap
                        return None, self._count_errors(bits), self._show_bits(bits)
                    found = code         # note the first one we find
            if found is not None:
                # only got 1 from this maybe level, go with it
                return found, self._count_errors(bits), self._show_bits(bits)

        # no candidates qualify
        return None, self._count_errors(bits), self._show_bits(bits)

    def _count_errors(self, bits):
        """ give a set of categorised bits, count how many ?'s it has,
            the bigger this number there more doubt there is in the validity of the code
            """
        doubt = 0
        errors = [0,  # IS_ZERO
                  0,  # IS_ONE
                  1,  # LIKELY_ZERO
                  1,  # LIKELY_ONE
                  2,  # MAYBE_ZERO
                  2,  # MAYBE_ONE
                  3]  # IS_NEITHER
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
                   self.MAYBE_ONE: '1??',
                   self.MAYBE_ZERO: '0??',
                   self.IS_NEITHER: '???'}
        csv = ''
        for bit in bits:
            csv += ',' + symbols[bit]
        return csv[1:]

    def is_marker(self, n, bits):
        """ given a set of bits and a bit position check if an alignment marker is present there
            the function returns the number of maybe bits (0..4) if a marker is found or None if not
            """
        maybes = 0
        i = self.marker_bits_pos(n)
        b1 = bits[i[0]]
        b2 = bits[i[1]]
        b3 = bits[i[2]]
        b4 = bits[i[3]]
        if b1 == self.IS_ZERO:
            pass                         # exact match
        elif b1 == self.LIKELY_ZERO:
            pass                         # probable match
        elif b1 == self.MAYBE_ZERO:
            maybes += 1                  # maybe match
        else:
            return None                  # not a marker
        if b2 == self.IS_ONE:
            pass                         # exact match
        elif b2 == self.LIKELY_ONE:
            pass                         # probable match
        elif b2 == self.MAYBE_ONE:
            maybes += 1                  # maybe match
        else:
            return None                  # not a match
        if b3 == self.IS_ONE:
            pass                         # exact match
        elif b3 == self.LIKELY_ONE:
            pass                         # probable match
        elif b3 == self.MAYBE_ONE:
            maybes += 1                  # maybe match
        else:
            return None                  # not a match
        if b4 == self.IS_ZERO:
            pass                         # exact match
        elif b4 == self.LIKELY_ZERO:
            pass                         # probable match
        elif b4 == self.MAYBE_ZERO:
            maybes += 1                  # maybe match
        else:
            return None                  # not a match
        return maybes

    def data_bits(self, n, bits):
        """ return an array of the data-bits from bits array starting at bit position n,
            this is effectively rotating the bits array and removing the marker bits such
            that the result is an array with [0] the first data bit and [n] the last
            """
        return [bits[int(pos % self.size)] for pos in range(n+self.marker_bits, n+self.size)]

    def extract_word(self, n, bits):
        """ given an array of bit values with the alignment marker at position n
            extract the code word and decode it (via decode()), returns None if cannot
            """
        word = self.data_bits(n, bits)
        code = 0
        for bit in range(len(word)):
            code <<= 1                   # make room for next bit
            val = word[bit]
            if (val == self.IS_ONE) or (val == self.LIKELY_ONE) or (val == self.MAYBE_ONE):
                code += 1                # got a one bit
            elif (val == self.IS_ZERO) or (val == self.LIKELY_ZERO) or (val == self.MAYBE_ZERO):
                pass                     # got a zero bit
            else:
                return None              # got junk
        return self.decode(code)

    def bit(self, s1, s2, s3):
        """ given 3 blob values determine the most likely bit value
            the blobs are designated as 'black', 'white' or 'grey'
            the return bit is one of 'is', 'likely' or 'maybe' one or zero, or is_neither
            the middle sample (s2) is expected to be inverted (i.e. black is considered as white and visa versa)
            """
        zeroes = 0
        ones   = 0
        greys  = 0

        # count states
        if s1 == self.GREY:
            greys += 1
        elif s1 == self.BLACK:
            zeroes += 1
        elif s1 == self.WHITE:
            ones += 1
        if s2 == self.GREY:
            greys += 1
        elif s2 == self.BLACK:
            ones += 1                    # s2 is inverted
        elif s2 == self.WHITE:
            zeroes += 1                  # s2 is inverted
        if s3 == self.GREY:
            greys += 1
        elif s3 == self.BLACK:
            zeroes += 1
        elif s3 == self.WHITE:
            ones += 1

        # test definite cases
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
        return self.IS_NEITHER

    def blob(self, s1, w1, b1, s2, w2, b2, s3, w3, b3):
        """ given 3 luminance samples and their luminance levels determine the most likely blob value,
            each sample is checked against the luminance levels to determine if its black, grey or white
            then decoded as a bit
            sN is the sample level, wN is the white level for that sample and bN is its black level
            """
        return self.bit(self.category(s1, w1, b1),
                        self.category(s2, w2, b2),
                        self.category(s3, w3, b3))

    def category(self, sample_level, white_level, black_level):
        """ given a luminance level and its luminance range categorize it as black, white or grey,
            the white low threshold is the white width below the white level,
            the black high threshold is the black width above the black level,
            grey is below the white low threshold but above the black high threshold
            """
        if sample_level is None:
            # not given a sample, treat as grey
            return self.GREY
        if white_level is None or black_level is None:
            # we haven't been given the thresholds, treat as grey
            return self.GREY
        luminance_range = white_level - black_level
        white_range = luminance_range * self.white_width
        black_range = luminance_range * self.black_width
        black_max = int(round(black_level + black_range))
        white_min = int(round(white_level - white_range))
        if black_max > white_min:
            # not enough luminance variation, consider as grey
            return self.GREY
        if sample_level < black_max:
            return self.BLACK
        elif sample_level > white_min:
            return self.WHITE
        else:
            return self.GREY

    def check(self, num):
        """ check encode/decode is symmetric
            returns None if check fails or the given number if OK
            """
        encoded = self.encode(num)
        if encoded is None:
            print('{} encodes as None'.format(num))
            return None
        decoded = self.decode(encoded)
        if decoded != num:
            print('{} encodes to {} but decodes as {}'.format(num, encoded, decoded))
            return None
        return num

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
        self.angles = [None for _ in range(self.ratio_scale+1)]
        self.angles[0] = 0
        for step in range(1,len(self.angles)):
            # each step here represents 1/scale of an octant
            # the index is the ratio of x/y (0..1*scale), the result is the angle (in degrees)
            self.angles[step] = math.degrees(math.atan(step / self.ratio_scale))

        # generate cartesian to polar lookup table
        self.ratios = [[None, None] for _ in range(self.ratio_scale+1)]
        self.step_angle = 45 / self.ratio_scale  # the angle represented by each step in the lookup table
        for step in range(len(self.ratios)):
            # each octant here consists of scale steps,
            # the index is an angle 0..45, the result is the x,y co-ordinates for circle of radius 1,
            # angle 0 is considered to be straight up and increase clockwise, the vertical axis is
            # considered to be -Y..0..+Y, and the horizontal -X..0..+X,
            # the lookup table contains 0..45 degrees, other octants are calculated by appropriate x,y
            # reversals and sign reversals
            self.ratios[step][0] = 0.0 + math.sin(math.radians(step * self.step_angle))  # NB: x,y reversed
            self.ratios[step][1] = 0.0 - math.cos(math.radians(step * self.step_angle))  #     ..
        # Parameters for ratio() for each octant:
        #   edge angle, offset, 'a' multiplier', reverse x/y, x multiplier, y multiplier
        #                                            #                     -Y
        #                                        #                     -Y
        self.octants = [[45 ,   0,+1,0,+1,+1],   # octant 0         \ 7 | 0 /
                        [90 , +90,-1,1,-1,-1],   # octant 1       6  \  |  /  1
                        [135, -90,+1,1,-1,+1],   # octant 2           \ | /
                        [180,+180,-1,0,+1,-1],   # octant 3    -X ------+------ +X
                        [225,-180,+1,0,-1,-1],   # octant 4           / | \
                        [270,+270,-1,1,+1,+1],   # octant 5       5  /  |  \  2
                        [315,-270,+1,1,+1,-1],   # octant 6         / 4 | 3 \
                        [360,+360,-1,0,-1,+1]]   # octant 7            +Y
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
                x = ratio[0+octant[3]] * octant[4] * r
                y = ratio[1-octant[3]] * octant[5] * r
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
            return math.sqrt(x*x + y*y)

        # edge cases
        if x == 0:
            if y == 0: return None, None       # edge 0
            if y  > 0: return 180, +y          # edge 2
            else:      return   0, -y          # edge 4
        elif y == 0:  # and x != 0
            if x  > 0: return  90, +x          # edge 1
            else:      return 270, -x          # edge 3
        # which octant?
        # NB: both x and y are not 0 to get here
        if x > 0:
            # octant 0, 1, 2, 3
            if y < 0:
                # octant 0, 1
                if -y > x:
                    # octant 0
                    return _ratio2angle(  0, +1,  x/-y), _xy2r(x, y)
                else:
                    # octant 1
                    return _ratio2angle( 90, -1, -y/ x), _xy2r(x, y)
            else:
                # octant 2, 3
                if y < x:
                    # octant 2
                    return _ratio2angle( 90, +1,  y/ x), _xy2r(x, y)
                else:
                    # octant 3
                    return _ratio2angle(180, -1,  x/ y), _xy2r(x, y)
        else:  # x < 0
            # octant 4, 5, 6, 7
            if y > 0:
                # octant 4, 5
                if y > -x:
                    # octant 4
                    return _ratio2angle(180, +1, -x/ y), _xy2r(x, y)
                else:
                    # octant 5
                    return _ratio2angle(270, -1,  y/-x), _xy2r(x, y)
            else:  # y < 0
                # octant 6, 7
                if -y < -x:
                    # octant 6
                    return _ratio2angle(270, +1, -y/-x), _xy2r(x, y)
                else:
                    # octant 7
                    return _ratio2angle(360, -1, -x/-y), _xy2r(x, y)

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
        d = math.sqrt(a*a + b*b)
        return d

class Ring:
    """ this class knows how to draw the marker and data rings according to its constructor parameters
        """

    def __init__(self, centre_x, centre_y, segments, width, frame, contrast, offset):
        # set constant parameters
        self.s = segments          # how many bits in each ring
        self.w = width             # width of each ring
        self.c = frame             # where to draw it
        self.x = centre_x          # where the centre of the rings are
        self.y = centre_y          # ..

        # setup black/white luminance levels for drawing pixels,
        # contrast specifies the luminance range between black and white, 1=full luminance range, 0.5=half, etc
        # offset specifies how much to bias away from the mid-point, -ve is below, +ve above, number is a
        # fraction of the max range
        # we subtract half the required range from the offset mid-point for black and add it for white
        level_range = max_luminance * contrast / 2             # range relative to mid point
        level_centre = mid_luminance + (max_luminance * offset)
        self.black_level = int(round(max(level_centre - level_range, min_luminance)))
        self.white_level = int(round(min(level_centre + level_range, max_luminance)))

        # setup our angles look-up table
        scale = 2 * math.pi * width * 7
        self.angle_xy = Angle(scale).polarToCart
        self.edge = 360 / self.s   # the angle at which a bit edge occurs (NB: not an int)

    def _pixel(self, x, y, colour):
        """ draw a pixel at x,y from the image centre with the given luminance and opaque,
            to mitigate pixel gaps in the circle algorithm we draw several pixels near x,y
            """
        x += self.x
        y += self.y
        self.c.putpixel(x  , y  , colour, True)
        self.c.putpixel(x+1, y  , colour, True)
        self.c.putpixel(x  , y+1, colour, True)

    def _point(self, x, y, bit):
        """ draw a point at offset x,y from our centre with the given bit (0 or 1) colour (black or white)
            bit can be 0 (draw 'black'), 1 (draw 'white'), -1 (draw max luminance), -2 (draw min luminance),
            any other value (inc None) will draw 'grey' (mid luminance)
            """
        if bit == 0:
            colour = self.black_level
        elif bit == 1:
            colour = self.white_level
        elif bit == -1:                  # lsb is 1
            colour = max_luminance
        elif bit == -2:                  # lsb is 0
            colour = min_luminance
        else:
            colour = mid_luminance
        self._pixel(x, y, colour)

    def _draw(self, radius, bits):
        """ draw a ring at radius of bits, a 1-bit is white, 0 black, if bits is None the bit timing ring is drawn
            the bits are drawn big-endian and clockwise , i.e. MSB first (0 degrees), LSB last (360 degrees)
            """
        if radius <= 0:
            # special case - just draw a dot at x,y of the LSB colour of bits
            self._point(0, 0, bits & 1)
        else:
            msb = 1 << (self.s-1)
            scale = 2 * math.pi * radius       # step the angle such that 1 pixel per increment
            for step in range(int(round(scale))):
                a = (step / scale) * 360
                x, y = self.angle_xy(a, radius)
                x = int(round(x))
                y = int(round(y))
                if a > 0:
                    if bits is None:
                        segment = int(a / (self.edge / 2))     # want half-bit segments for the timing ring
                    else:
                        segment = int(a / self.edge)           # full-bit segments for data (or one colour) rings
                else:
                    segment = 0
                mask = msb >> segment
                if bits is None:
                    if (segment & 1) == 0:
                        # white for even half bits
                        self._point(x, y, 1)
                    else:
                        # black for odd half bits
                        self._point(x, y, 0)
                elif bits & mask:
                    self._point(x, y, 1)
                else:
                    self._point(x, y, 0)

    def draw(self, ring_num, data_bits):
        """ draw a data ring with the given data_bits,
            if data_bits is None a timing ring is drawn (this is a diagnostic aid, see _draw)
            """
        for radius in range(ring_num*self.w, (ring_num+1)*self.w):
            self._draw(radius, data_bits)

    def label(self, number):
        """ draw a 3 digit number at the top left edge of the rings in a size compatible with the ring width
            we do it DIY using a 4 x 5 grid
            """
        zero   = [[1, 1, 1, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 1, 1, 1]]
        one    = [[0, 1, 0, 0],
                  [1, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [1, 1, 1, 0]]
        two    = [[1, 1, 1, 1],
                  [0, 0, 0, 1],
                  [1, 1, 1, 1],
                  [1, 0, 0, 0],
                  [1, 1, 1, 1]]
        three  = [[1, 1, 1, 1],
                  [0, 0, 0, 1],
                  [1, 1, 1, 1],
                  [0, 0, 0, 1],
                  [1, 1, 1, 1]]
        four   = [[0, 1, 1, 0],
                  [1, 0, 1, 0],
                  [1, 1, 1, 1],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0]]
        five   = [[1, 1, 1, 1],
                  [1, 0, 0, 0],
                  [1, 1, 1, 1],
                  [0, 0, 0, 1],
                  [1, 1, 1, 1]]
        six    = [[1, 0, 0, 0],
                  [1, 0, 0, 0],
                  [1, 1, 1, 1],
                  [1, 0, 0, 1],
                  [1, 1, 1, 1]]
        seven  = [[1, 1, 1, 1],
                  [0, 0, 0, 1],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0]]
        eight  = [[1, 1, 1, 1],
                  [1, 0, 0, 1],
                  [1, 1, 1, 1],
                  [1, 0, 0, 1],
                  [1, 1, 1, 1]]
        nine   = [[1, 1, 1, 1],
                  [1, 0, 0, 1],
                  [1, 1, 1, 1],
                  [0, 0, 0, 1],
                  [0, 0, 0, 1]]
        digits = [zero, one, two, three, four, five, six, seven, eight, nine]
        point_size = max(int((self.w / len(zero[0])) * 0.6), 1)
        digit_size = point_size * len(zero[0])
        start_x = -self.x + (self.w >> 1)
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
                                self._point(x+dx, y+dy, -2)    # draw at min luminance (ie. true black)
                    x += point_size
                x -= digit_size
                y += point_size

    def code(self, number, rings):
        """ draw the complete code for the given number and code-words
            the code-words must match the number
            """
        # draw the bullseye and its enclosing ring
        self.draw(0, -1)
        self.draw(1, -1)
        self.draw(2,  0)
        # draw the data rings
        self.draw(3, rings[0])
        self.draw(4, rings[1])
        self.draw(5, rings[2])
        # draw the outer black/white rings
        self.draw(6,  0)
        self.draw(7,  0)
        self.draw(8, -1)
        # draw a human readable label
        self.label(number)

class Frame:
    """ image frame buffer as a 2D array of luminance values
        it uses the opencv library to read/write and modify images at the pixel level
        """

    def __init__(self):
        self.source = None
        self.buffer = None
        self.alpha  = None
        self.max_x  = None
        self.max_y  = None

    def instance(self):
        """ return a new instance of self """
        return Frame()

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
        self.max_x = self.buffer.shape[1]      # NB: cv2 x, y are reversed
        self.max_y = self.buffer.shape[0]      # ..

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
        self.buffer = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        self.alpha = None
        self.max_x = self.buffer.shape[1]      # NB: cv2 x, y are reversed
        self.max_y = self.buffer.shape[0]      # ..
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
            filename = '_{}_{}.png'.format(filename, suffix)
        else:
            filename = '{}.png'.format(filename)
        cv2.imwrite(filename, image)
        return filename

    def show(self, title='title'):
        """ show the current buffer """
        cv2.imshow(title, self.buffer)
        cv2.waitKey(0)

    def getpixel(self, x, y):
        """ get the pixel value at x,y
            nb: cv2 x,y is reversed from our pov
            """
        if x < 0 or x >= self.max_x or y < 0 or y >= self.max_y:
            return None
        else:
            return self.buffer[y, x]     # NB: cv2 x, y are reversed

    def putpixel(self, x, y, value, with_alpha=False):
        """ put the pixel of value at x,y """
        if x < 0 or x >= self.max_x or y < 0 or y >= self.max_y:
            pass
        else:
            self.buffer[y, x] = min(max(value, min_luminance), max_luminance)  # NB: cv2 x, y are reversed
            if with_alpha:
                if self.alpha is None:
                    self.alpha = np.full((self.max_y, self.max_x), transparent,  # default is fully transparent
                                          dtype=np.uint8)  # NB: numpy arrays follow cv2 conventions
                self.alpha[y, x] = opaque                                        # want foreground only for our pixels

    def inimage(self, x, y, r):
        """ determine if the points radius R and centred at X, Y are within the image """
        if (x-r) < 0 or (x+r) >= self.max_x or (y-r) < 0 or (y+r) >= self.max_y:
            return False
        else:
            return True

    def colourize(self):
        """ make our grey image into an RGB one, or RGBA if there is an alpha channel,
            returns the image array with 3 or 4 channels
            """
        if self.alpha is not None:
            # got an alpha channel
            image = cv2.merge([self.buffer, self.buffer, self.buffer, self.alpha])
        else:
            image = cv2.merge([self.buffer, self.buffer, self.buffer])
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

    def upsize(self, source, new_height):
        """ upsize the given image such that its height (y) is at least that given,
            the width is preserved,
            """
        width, height = source.size()
        if height >= new_height:
            # its already big enough
            return source
        target = source.instance()
        target.set(cv2.resize(source.get(), (width, new_height), interpolation=cv2.INTER_NEAREST))
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
        target.set(cv2.resize(source.get(), (new_width, new_height), interpolation=cv2.INTER_NEAREST))
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
                params.minThreshold = threshold[0]         # default is 10
            if threshold[1] is not None:
                params.maxThreshold = threshold[1]         # default is 220
            if threshold[2] is not None:
                params.thresholdStep = threshold[2]        # default is 10
        if circularity is not None:
            params.filterByCircularity = True
            if circularity[0] is not None:
                params.filterByCircularity = True
                params.minCircularity = circularity[0]
            if circularity[1] is not None:
                params.filterByCircularity = True
                params.maxCircularity = circularity[1]     # default 3.4
        if convexity is not None:
            if convexity[0] is not None:
                params.filterByConvexity = True
                params.minConvexity = convexity[0]
            if convexity[1] is not None:
                params.filterByConvexity = True
                params.maxConvexity = convexity[1]         # default 3.4
        if inertia is not None:
            if inertia[0] is not None:
                params.filterByInertia = True
                params.minInertiaRatio = inertia[0]
            if inertia[1] is not None:
                params.filterByInertia = True
                params.maxInertiaRatio = inertia[1]        # default 3.4
        if area is not None:
            if area[0] is not None:
                params.filterByArea = True
                params.minArea = area[0]
            if area[1] is not None:
                params.filterByArea = True
                params.maxArea = area[1]
        if gaps is not None:
            if gaps[0] is not None:
                params.minDistBetweenBlobs = gaps[0]       # default is 10
            if gaps[1] is not None:
                params.minRepeatability = gaps[1]          # default is 2
        if colour is not None:
            params.filterByColor = True
            params.blobColor = colour

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs
        return detector.detect(source.get())

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
            target.set(cv2.Sobel(255-source.get(), -1, xorder, yorder, size))
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
            _, buffer = cv2.threshold(source.get(), thresh, max_luminance, cv2.THRESH_BINARY)
        elif size is not None:
            buffer = cv2.adaptiveThreshold(source.get(), max_luminance,
                                           cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, size, 0.0)
        else:
            _, buffer = cv2.threshold(source.get(), min_luminance, max_luminance,
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
        image = source.get()
        if len(image.shape) == 2:
            # its a grey scale image, convert to RGB (actually BGR)
            image = cv2.merge([image, image, image])
        max_x = image.shape[1]
        max_y = image.shape[0]
        for obj in objects:
            if obj["type"] == self.LINE:
                image = cv2.line(image, obj["start"], obj["end"], obj["colour"], 1)
            elif obj["type"] == self.CIRCLE:
                image = cv2.circle(image, obj["centre"], obj["radius"], obj["colour"], 1)
            elif obj["type"] == self.RECTANGLE:
                image = cv2.rectangle(image, obj["start"], obj["end"], obj["colour"], 1)
            elif obj["type"] == self.PLOTX:
                colour = obj["colour"]
                points = obj["points"]
                if points is not None:
                    x = obj["start"]
                    for pt in range(len(points)):
                        y = points[pt]
                        if y is not None:
                            image[y, (x + pt) % max_x] = colour
            elif obj["type"] == self.PLOTY:
                colour = obj["colour"]
                points = obj["points"]
                if points is not None:
                    y = obj["start"]
                    for pt in range(len(points)):
                        x = points[pt]
                        if x is not None:
                            image[(y + pt) % max_y, x] = colour
            elif obj["type"] == self.POINTS:
                colour = obj["colour"]
                points = obj["points"]
                for pt in points:
                    x = pt[0]
                    y = pt[1]
                    image[y, x] = colour
            elif obj["type"] == self.TEXT:
                image = cv2.putText(image, obj["text"], obj["start"], cv2.FONT_HERSHEY_SIMPLEX, obj["size"], obj["colour"], 1, cv2.LINE_AA)
            else:
                raise Exception('Unknown object type {}'.format(obj["type"]))
        source.set(image)
        return source

class Scan:
    """ scan an image looking for codes """

    # debug options
    DEBUG_NONE = 0             # no debug output
    DEBUG_IMAGE = 1            # just write debug annotated image files
    DEBUG_VERBOSE = 2          # do everything - generates a *lot* of output

    # directions
    TOP_DOWN = 1               # top-down y scan direction
    BOTTOM_UP = 2              # bottom-up y scan direction
    LEFT_TO_RIGHT = 3          # left-to-right x scan direction
    RIGHT_TO_LEFT = 4          # right-to-left x scan direction

    # ring numbers of flattened image
    DODGY_WHITE = 0
    INNER_WHITE = 1            # used for white level detection
    INNER_BLACK = 2
    DATA_RING_1 = 3
    DATA_RING_2 = 4
    DATA_RING_3 = 5
    DODGY_BLACK = 6
    OUTER_BLACK = 7            # used for black level detection
    OUTER_WHITE = 8

    # total number of rings in the whole code
    NUM_RINGS = 9

    class Edge:
        """ structure to hold detected edge information,
            this is as close as you can get to a 'struct' in Python
            """
        @staticmethod
        def show(position=0, length=0, span=0, matches=0):
            return 'p:{}, l:{}, s:{}, m:{}'.format(position, length, span, matches)

        def __init__(self, position, length, span):
            self.position = position     # where it is (an X or Y co-ord)
            self.length = length         # how long it is (in pixels)
            self.span = span             # how wide it is (in pixels)
            self.matches = 0             # how many actual edges match expectations
            self.ideal = None            # ideal edges starting from position
            self.actual = None           # actual matches starting from position

        def __str__(self):
            return Scan.Edge.show(self.position, self.length, self.span, self.matches)

    class Kernel:
        """ an iterator that returns a series of x,y co-ordinates for a 'kernel' in a given direction
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

    def __init__(self, code, frame, angles=360, debug=DEBUG_NONE):
        """ code is the code instance defining the code structure,
            frame is the frame instance containing the image to be scanned,
            angles is the angular resolution to use
            """

        # threshold constants
        self.min_blob_separation = 3         # smallest blob within this distance of each other are dropped
        self.min_inner_outer_span = self.NUM_RINGS   # min tolerable gap between inner and outer target edge
        self.blob_radius_stretch = 1.6       # how much to stretch blob radius to ensure always cover the whole lot
        self.min_ring_edge_length = 0.3      # min length of a ring edge as a fraction of the nominal bit width
        self.ring_edge_span_limit = 0.2      # maximum span in y of ring edge as fraction of nominal width
        self.ring_edge_tolerance = 0.3       # fraction of actual ring width considered 'close-enough' to ideal
        self.thin_bit_edge_width = 2         # bit edge width to distinguish a thin or thick edge
        self.thin_bit_edge_length = 0.7      # min length of a bit edge in a data ring as a fraction of the ring width
        self.thick_bit_edge_length = 0.5     # min length of a bit edge in a data ring as a fraction of the ring width
        self.bit_edge_tolerance = 0.6        # fraction of actual bit width considered 'close-enough' to ideal
        self.max_bit_edge_overspill = 0.5    # fraction of a bit edge allowed to overspill out of data rings
        self.sample_width_factor = 0.5       # fraction of a bit width that is probed for a luminance value
        self.sample_height_factor = 0.3      # fraction of a ring height that is probed for a luminance value
        self.min_target_radius = 6 * self.NUM_RINGS  # upsize target radius to at least this (6 pixels per ring)

        # luminance threshold when getting the mean luminance of an image, pixels below this are ignored
        self.min_edge_threshold = int(max_luminance * 0.1)

        # minimum luminance for an edge pixel to qualify as an edge when following in x relative to the mean
        self.edge_threshold_x = 0.8

        # minimum luminance for an edge pixel to qualify as an edge when following in y relative to the mean
        self.edge_threshold_y = 0.8

        # params
        self.angle_steps = angles                          # angular resolution when 'projecting'
        self.original = frame
        self.c = code
        self.size = code.size                              # total ring code size in bits

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
        self.transform = Transform()                       # make a new frame instance
        self.angle_xy = None                               # see _find_blobs

        # samples probed (x,y offsets from some reference) for detecting connected neighbours,
        # some reference x,y is 'connected' to a neighbour if one of these is over a threshold,
        # when doing this 'self' is over the threshold so is not considered,
        # these co-ords are rotated by 0, 90, 180 or 270 degrees depending on the direction being followed,
        # the reference set here is for left-to-right and is considered as 0 degrees rotation

        # x,y pairs for neighbours where a small gap is tolerated
        self.edge_kernel_loose = [[1,  0], [2, 0], [3, 0],     # straight line
                                  [1, -1], [1, 1],             # diagonal
                                  [2, -1], [2, 1]]                                 # acute diagonal
        self.edge_kernel_loose_width = 4                               # radius of pixels scanned
        self.max_radius_deviation = 2 * self.edge_kernel_loose_width   # limit of gap when joining radius edges

        # x,y pairs for neighbours where no gap is tolerated
        self.edge_kernel_tight = [[1, 0],                      # straight line
                                  [1, -1], [1, 1]]             # diagonal
        self.edge_kernel_tight_width = 2                       # radius of pixels scanned

        # edge vector smoothing kernel, pairs of offset and scale factor (see _smooth_edge)
        # NB: the smoothing is done in-place, so only look ahead here
        self.edge_smoothing_kernel = [[0, 1], [+1, 1], [+2, 1], [+3, 1]]

        # logging context
        self._log_last_x = 0
        self._log_last_y = 0
        self._log_prefix = '{}: Blob'.format(self.original.source)

        # image unloading context
        self.unload_last_x = 0
        self.unload_last_y = 0

    def _log(self, message, centre_x=None, centre_y=None, fatal=False):
        """ print a debug log message
            message is what to log, use None to just note centre_x/y
            centre_x/y are the co-ordinates of the centre of the associated blob, if None use last time
            centre_x/y of 0,0 means no x/y identification in the log
            iff fatal is True an exception is raised, else the message is just printed
            """
        if centre_x is None:
            centre_x = self._log_last_x
        elif centre_x > 0:
            self._log_last_x = centre_x
        if centre_y is None:
            centre_y = self._log_last_y
        elif centre_y > 0:
            self._log_last_y = centre_y
        if message is not None:
            if centre_x > 0 and centre_y > 0:
                message = '{} {:.0f}x{:.0f}y - {}'.format(self._log_prefix, centre_x, centre_y, message)
            else:
                message = '{} {}'.format(self._log_prefix, message)
            if fatal:
                raise Exception(message)
            else:
                print(message)

    def _unload(self, image, suffix, centre_x=None, centre_y=None):
        """ unload the given image with a name that indicates its source and context,
            suffix is the file name suffix (to indicate context),
            centre_x/y identify the blob the image represents, if None use same as last time,
            if image is None just note the centre_x/y,
            centre_x/y of 0,0 means no x/y identification on the image
            """
        if centre_x is None:
            centre_x = self.unload_last_x
        elif centre_x > 0:
            self.unload_last_x = centre_x
        if centre_y is None:
            centre_y = self.unload_last_y
        elif centre_y > 0:
            self.unload_last_y = centre_y
        if image is not None:
            if centre_x > 0 and centre_y > 0:
                name = '{:.0f}x{:.0f}y-{}'.format(centre_x, centre_y, suffix)
            else:
                name = suffix
            filename = image.unload(self.original.source, name)
            if self.show_log:
                self._log('Saved image: {}'.format(filename), centre_x, centre_y)

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
        blurred = self.transform.blur(self.original)           # de-noise
        self.image = self.transform.downsize(blurred, 1080)      # re-size (to HD video convention)

        # set filter parameters
        # ToDo: these need tuning to hell and back to optimise for our context
        threshold = (min_luminance, max_luminance, 8)  # min, max luminance, luminance step
        circularity = (0.75, None)             # min, max 'corners' in blob edge or None
        convexity = (0.5, None)                # min, max 'gaps' in blob edge or None
        inertia = (0.4, None)                  # min, max 'squashed-ness' or None
        area = (30, 250000)                    # min, max area in pixels, or None for no area filter
        gaps = (None, None)                    # how close blobs have to be to be merged and min number
        colour = max_luminance                 # we want bright blobs, use min_luminance for dark blobs

        # find the blobs
        blobs = self.transform.blobs(self.image, threshold, circularity, convexity, inertia, area, gaps, colour)

        # filter out blobs that are co-incident (opencv can get over enthusiastic!)
        dropped = True
        while dropped:
            dropped = False
            for blob1 in range(len(blobs)):
                x1 = blobs[blob1].pt[0]
                y1 = blobs[blob1].pt[1]
                size1 = blobs[blob1].size
                for blob2 in range(len(blobs)):
                    if blob2 == blob1:
                        # do not check self
                        continue
                    x2 = blobs[blob2].pt[0]
                    y2 = blobs[blob2].pt[1]
                    size2 = blobs[blob2].size
                    if math.fabs(x1 - x2) < self.min_blob_separation and math.fabs(y1 - y2) < self.min_blob_separation:
                        # these are too close, drop the smallest one
                        if size1 < size2:
                            # drop #1
                            blobs.pop(blob1)
                            if self.show_log:
                                self._log('duplicate blob dropped', x1, y1)
                        else:
                            # drop #2
                            blobs.pop(blob2)
                            if self.show_log:
                                self._log('duplicate blob dropped', x2, y2)
                        dropped = True
                        break
                if dropped:
                    break

        # build the polar to cartesian translation table
        max_size = 3                                           # do one for at least 3 pixels
        for blob in blobs:
            if blob.size > max_size:
                max_size = blob.size

        max_radius = (max_size / 4) * self.NUM_RINGS * self.blob_radius_stretch
        max_circumference = min(2 * math.pi * max_radius, 3600)  # good enough for 0.1 degree resolution
        angle = Angle(int(round(max_circumference)))
        self.angle_xy = angle.polarToCart

        return blobs

    def _project(self, centre_x, centre_y, blob_size):
        """ 'project' a potential target at centre_x/y from its circular shape to a rectangle
            of radius (y) by angle (x), blob_size is used as a guide to limit the radius projected,
            we assume the blob-size is (roughly) the diameter of the inner two white rings
            but err on the side of going too big, we use the opencv warpPolar function,
            if the target is very small we upsize it, this helps later when calculating pixel
            addresses (e.g. mid-points can be more accurate)
            """

        # calculate the maximum radius to go out to
        max_x, max_y = self.image.size()
        edge_top = centre_y
        edge_left = centre_x
        edge_bottom = max_y - centre_y
        edge_right = max_x - centre_x
        limit_radius = int(round(max(min((edge_top, edge_bottom, edge_left, edge_right)), 1)))
        blob_radius = int(round(((blob_size / 4) * self.NUM_RINGS) * self.blob_radius_stretch))
        if blob_radius < limit_radius:
            # max possible size is less than the image edge, so use the blob size
            limit_radius = blob_radius

        # do the projection
        code = self.transform.warpPolar(self.image, centre_x, centre_y, limit_radius, self.angle_steps, limit_radius)

        # upsize small targets
        code = self.transform.upsize(code, self.min_target_radius)

        return code

    def _smooth_edge(self, edge, centre_x, centre_y):
        """ smooth the given edge vector by doing a mean across N pixels,
            centre_x/y params are purely for diagnostic reporting
            """
        extent = len(edge)
        for x in range(extent):
            v = 0  # value accumulator
            d = 0  # divisor accumulator
            for dx, f in self.edge_smoothing_kernel:
                sample = edge[(x + dx) % extent]
                if sample is None:
                    # eh?
                    if self.show_log:
                        self._log('edge[{}] is None! - info'.format((x + dx) % extent), centre_x, centre_y)
                else:
                    v += int(round(sample * f))
                    d += f
            if d > 0:
                edge[x] = int(round(v / d))
        return edge

    def _midpoint(self, start_at, end_at):
        """ given two co-ordinates (either two x's or two y's) return their mid-point,
            ie. that co-ordinate that is central between the two (truncated),
            the co-ordinates can bne given either way round (start < end or end < start)
            """
        if end_at > start_at:
            midpoint = start_at + ((end_at - start_at) / 2)
        else:
            midpoint = start_at - ((start_at - end_at) / 2)
        return int(round(midpoint))

    def _find_midpoint_in_y(self, target, x, y, threshold_y):
        """ from x,y in the target image find the midpoint y that is over the given threshold,
            the pixel at x,y must also be over else None is returned,
            returns the y co-ord of the midpoint that is over our threshold or None if isn't one,
            this is used to find ring edges,
            the mid-point is between the first and last y that is over our threshold,
            """
        _, max_y = target.size()
        # find start y
        start_at = None
        for dy in range(y, -1, -1):
            pixel = target.getpixel(x, dy)
            if pixel < threshold_y:
                break
            start_at = dy
        # find end y
        end_at = None
        for dy in range(y, max_y):
            pixel = target.getpixel(x, dy)
            if pixel < threshold_y:
                break
            end_at = dy
        if start_at is None or end_at is None:
            return None
        else:
            return min(self._midpoint(start_at, end_at), max_y - 1)

    def _find_midpoint_in_x(self, target, x, y, threshold_x):
        """ from x,y in the target image find the midpoint x that is over our threshold,
            the pixel at x,y must also be over else None is returned,
            returns the x co-ord of the midpoint that is over our threshold or None if isn't one,
            this is used to find data ring bit edges,
            the mid-point is between the first and last x that is over our threshold,
            NB: angles wrap 360..0 in target
            """
        max_x, _ = target.size()
        # find start
        start_at = None
        for _ in range(max_x):
            x = (x - 1) % max_x
            pixel = target.getpixel(x, y)
            if pixel < threshold_x:
                break
            start_at = x
        # find end
        end_at = None
        for _ in range(max_x):
            x = (x + 1) % max_x
            pixel = target.getpixel(x, y)
            if pixel < threshold_x:
                break
            end_at = x
        if start_at is None or end_at is None:
            return None
        if end_at < start_at:
            # we wrapped
            end_at += max_x
        return min(self._midpoint(start_at, end_at), max_x - 1)

    def _is_connected(self, target, x, y, direction, threshold, kernel):
        """ determine if the given x,y is 'connected' to a neighbour in the given direction,
            to be connected at least one of the pixels addressed by the given kernel must be over the threshold,
            the direction can be top-down or bottom up in y, or left-to-right or right-to-left in x,
            the kernel should define a series of x,y offsets in the left-to-right direction, they are rotated
            appropriately for the other directions, the offsets typically define a semi-circle radiating
            from the reference x,y,
            co-ordinates going over the x limit are wrapped,
            returns True iff got a connected neighbour
            """
        max_x, max_y = target.size()
        for dx, dy in self.Kernel(kernel, direction):
            tx = (x + dx) % max_x        # wrap in x
            ty = y + dy                  # no wrap in y, out of range returns None from getpixel
            pixel = target.getpixel(tx, ty)
            if pixel is not None and pixel >= threshold:
                # found an ongoing neighbour also in range, so its connected
                return True
        # if we get here there are no connected neighbours
        return False

    def _find_best_y_neighbour(self, target, x, y, direction_in_x, threshold_x, kernel):
        """ find best y neighbour in target from x, y,
            kernel is the matrix to use to determine that a neighbour is 'connected',
            returns the y co-ord of the best neighbour that is over our edge threshold,
            returns None if there is not one (ie. no neighbours are connected),
            this function is intended to be used when following edges in the x direction,
            direction_in_x is either LEFT_TO_RIGHT or RIGHT_TO_LEFT (affects kernel orientation),
            it finds the best ongoing y co-ordinate to follow,
            the 'best' y is the mid-point between the first and last y that is connected to a neighbour
            """

        # find the start y
        start_at = None
        dy = y + 1
        while dy > 0:
            dy -= 1
            if self._is_connected(target, x, dy, direction_in_x, threshold_x, kernel):
                # still connected
                start_at = dy
            else:
                # no longer connected
                break
        # find the end y
        end_at = None
        _, max_y = target.size()
        dy = y - 1
        while dy < (max_y -1):
            dy += 1
            if self._is_connected(target, x, dy, direction_in_x, threshold_x, kernel):
                # still connected
                end_at = dy
            else:
                # no longer connected
                break
        if start_at is None or end_at is None:
            return None
        else:
            return min(self._midpoint(start_at, end_at), max_y - 1)

    def _find_best_x_neighbour(self, target, x, y, direction_in_y, threshold_y, kernel):
        """ find best x neighbour in target from x, y,
            kernel is the matrix to use to determine that a neighbour is 'connected',
            returns the x co-ord of the best neighbour that is over our edge threshold,
            returns None if there is not one (ie. no neighbours are connected),
            this function is intended to be used when following edges in the y direction,
            direction_in_y is either TOP_DOWN or BOTTOM_UP (affects kernel orientation),
            it finds the best ongoing x co-ordinate to follow,
            the 'best' x is the mid-point between the first and last x that is connected to a neighbour,
            NB: x co-ords wrap, so we probe the whole 360 degrees no matter what initial x is given
            """

        max_x, _ = target.size()

        # find start x
        start_at = None
        dx = x + 1
        for _ in range(max_x):
            dx = (dx - 1) % max_x
            if self._is_connected(target, dx, y, direction_in_y, threshold_y, kernel):
                # still connected
                start_at = dx
            else:
                # no longer connected
                break
        # find end x
        end_at = None
        dx = x - 1
        for _ in range(max_x):
            dx = (dx + 1) % max_x
            if self._is_connected(target, dx, y, direction_in_y, threshold_y, kernel):
                # still connected
                end_at = dx
            else:
                # no longer connected
                break
        if start_at is None or end_at is None:
            return None
        if start_at > end_at:
            # we wrapped
            end_at += max_x
        return min(self._midpoint(start_at, end_at), max_x - 1)

    def _follow_edge_in_x(self, target, x, y,
                          direction_in_x,
                          threshold_x, threshold_y,
                          kernel,
                          span_limit=None):
        """ follow the edge at x,y until come to its end in x or its y span becomes too large,
            kernel specifies the matrix to use to detect connected neighbours,
            its followed for up to a full revolution (360 deg) and may wrap around the image edge,
            return a vector of its y co-ords, or an empty vector if none,
            the pixel at x,y must be over else None is returned,
            x direction is left-to-right or right-to-left,
            span_limit is the maximum y span to follow, None is no limit,
            """
        edge_y = self._find_midpoint_in_y(target, x, y, threshold_y)
        if edge_y is None:
            return []
        start_y = edge_y
        edge = [edge_y]
        if direction_in_x == self.LEFT_TO_RIGHT:
            increment = +1
        elif direction_in_x == self.RIGHT_TO_LEFT:
            increment = -1
        else:
            raise Exception('illegal direction_in_x: {}'.format(direction_in_x))
        max_x, max_y = target.size()
        if span_limit is None:
            span_limit = max_y
        for _ in range(max_x - 1):
            x = (x + increment) % max_x
            best_dy = self._find_best_y_neighbour(target, x, edge_y, direction_in_x, threshold_x, kernel)
            if best_dy is None:
                # no more qualifying neighbors
                break
            elif math.fabs(start_y - best_dy) > span_limit:
                # span reached its limit, stop here
                # this is limiting how far the edge can drift left or right
                break
            else:
                edge_y = best_dy
                edge.append(edge_y)
        return edge

    def _follow_edge_in_y(self, target, x, y,
                          direction_in_y,
                          threshold_x, threshold_y,
                          kernel,
                          span_limit=None):
        """ follow the edge at x,y until come to its end in y (its followed until it hits the image edge),
            return a vector of its x co-ords, or an empty vector if none,
            the pixel at x,y must be over the brightness threshold else None is returned,
            y direction is top-down or bottom-up,
            span_limit is the maximum x span to follow, None is no limit
            """

        edge_x = self._find_midpoint_in_x(target, x, y, threshold_x)
        if edge_x is None:
            return []
        start_x = edge_x
        max_x, max_y = target.size()
        if span_limit is None:
            span_limit = max_x
        edge = [edge_x]
        if direction_in_y == self.TOP_DOWN:
            radius = (dy for dy in range(y+1, max_y))
            increment = +1
        elif direction_in_y == self.BOTTOM_UP:
            radius = (dy for dy in range(y-1, -1, -1))
            increment = -1
        else:
            raise Exception('illegal direction_in_y: {}'.format(direction_in_y))
        for _ in radius:
            y += increment
            best_dx = self._find_best_x_neighbour(target, edge_x, y, direction_in_y, threshold_y, kernel)
            if best_dx is None:
                # no more qualifying neighbors
                break
            elif math.fabs(start_x - best_dx) > span_limit:
                # reached limit of span allowed, stop here
                # this is limiting how far the edge can drift left or right
                break
            else:
                edge_x = best_dx
                edge.append(edge_x)
        return edge

    def _find_radius_edge(self, target, direction, threshold_x, threshold_y, centre_x, centre_y):
        """ look for a continuous edge in the given target either top-down (inner) or bottom-up (outer),
            direction is top-down or bottom-up, an 'edge' is vertical, i.e along the radius,
            centre_x/y are purely for diagnostic reporting,
            to qualify as an 'edge' it must be continuous across all angles (i.e. no dis-connected jumps)
            to find the edge we scan the radius in both directions, if we get full 360 without a break we've found it
            """
        max_x, max_y = target.size()
        if direction == self.TOP_DOWN:
            radius = (y for y in range(max_y))
        else:
            radius = (y for y in range(max_y-1, -1, -1))
        for y in radius:
            edge = self._follow_edge_in_x(target, 0, y,
                                          self.LEFT_TO_RIGHT,
                                          threshold_x, threshold_y,
                                          self.edge_kernel_loose)
            if len(edge) == 0:
                # nothing here, move on
                # NB: if nothing going forwards there will also be nothing going backwards
                continue
            if self.show_log:
                if direction == self.TOP_DOWN:
                    self._log('TOP_DOWN+LEFT_TO_RIGHT radius edge length: {} at {}'.
                              format(len(edge), y), centre_x, centre_y)
                else:
                    self._log('BOTTOM_UP+LEFT_TO_RIGHT radius edge length: {} at {}'.
                              format(len(edge), y), centre_x, centre_y)
            if len(edge) == max_x:
                # found an edge that goes all the way, so that's it, just smooth it and go
                return self._smooth_edge(edge, centre_x, centre_y)
            # this one too short, see if can extend it by going the other way
            edge_extension = self._follow_edge_in_x(target, 0, y,
                                                    self.RIGHT_TO_LEFT,
                                                    threshold_x, threshold_y,
                                                    self.edge_kernel_loose)
            if len(edge_extension) == 0:
                # nothing more here, move on
                continue
            if self.show_log:
                if direction == self.TOP_DOWN:
                    self._log('TOP_DOWN+RIGHT_TO_LEFT radius edge extension length: {} at {}'.
                              format(len(edge_extension), y), centre_x, centre_y)
                else:
                    self._log('BOTTOM_UP+RIGHT_TO_LEFT radius edge extension length: {} at {}'.
                              format(len(edge_extension), y), centre_x, centre_y)
            if len(edge) + len(edge_extension) + self.edge_kernel_loose_width > max_x:
                # gone right round now, merge the two edges
                # they may overlap, in which case we use the median (NB: the first sample is the same in both)
                # we interpolate across a small ending gap (it will be at most edge_kernel_loose_width)
                edge_combined = [y for y in edge]              # copy our edge
                edge_combined.extend([None] * max_x)           # stretch it with None's
                edge_combined = edge_combined[:max_x]          # make it the right size
                dropped = False
                for x in range(1, len(edge_extension)):        # skip first sample, we know that's the same
                    slot = max_x - x                           # where this goes in edge
                    if edge_combined[slot] is None:
                        edge_combined[slot] = edge_extension[x]
                    else:
                        # got an overlap, if they deviate too much abandon it
                        deviation = math.fabs(edge_combined[slot] - edge_extension[x])
                        if deviation > self.max_radius_deviation:
                            # they've diverged too far to join up, drop it
                            if self.show_log:
                                self._log('edge[{}] diverges too far from edge_extension[{}], limit is {} - dropping'.
                                          format(edge_combined[slot], edge_extension[x], self.max_radius_deviation),
                                          centre_x, centre_y)
                            dropped = True
                            break
                        # they are close enough to combine
                        edge_combined[slot] = min(edge_combined[slot] + int(round(deviation / 2)), max_y - 1)
                if dropped:
                    continue
                # any None's left at the join need interpolating across
                if len(edge) + len(edge_extension) > max_x:    # NB: edge and edge_extension overlap by 1
                    # no gap, so that's it
                    return self._smooth_edge(edge_combined, centre_x, centre_y)
                # gap is at len(edge)..(max_x-len(edge_extension))
                gap = (max_x - len(edge_extension)) - len(edge) + 1  # +1 'cos edge_extension and edge overlap
                start_y = edge_combined[len(edge) - 1]
                stop_y = edge_combined[max_x - len(edge_extension) + 1]
                delta_y = (start_y - stop_y) / gap
                next_y = start_y
                for x in range(len(edge), max_x - len(edge_extension)):
                    next_y += delta_y
                    if edge_combined[x] is None:
                        edge_combined[x] = int(round(next_y))
                    else:
                        # eh? got me sums wrong sumplace
                        raise Exception('got me sums wrong sumplace')
                # all done now
                return self._smooth_edge(edge_combined, centre_x, centre_y)

        # we did not find one
        return None

    def _get_edge_threshold(self, target):
        """ find the edge threshold to apply in the given target,
            target is assumed to be an edge detected image,
            this function finds the mean luminance (ignoring near black pixels)
            and returns that (turned into an int in the range min_luminance..max_luminance)
            """

        # turn image into a flat vector
        buffer = target.get().reshape(-1)
        buffer_pixels = len(buffer)

        # chuck out near black pixels
        filtered = buffer[np.where(buffer > self.min_edge_threshold)]
        if len(filtered) == 0:
            # nothing qualifies
            return 0.0

        # give caller the mean of what is left
        return np.mean(filtered)

    def _get_transitions(self, target, x_order, y_order, inverted):
        """ get black-to-white or white-to-black edges from target
            returns an image of the edges and a threshold to use for x and y
            """

        edges = self.transform.edges(target, x_order, y_order, 3, inverted)
        image_mean = self._get_edge_threshold(edges)
        threshold_x = int(round(image_mean * self.edge_threshold_x))
        threshold_y = int(round(image_mean * self.edge_threshold_y))

        return edges, threshold_x, threshold_y

    def _find_extent(self, target, centre_x, centre_y):
        """ find the inner and outer edges of the given target,
            the inner edge is the first white to black transition that goes all the way around,
            the outer edge is the first black to white transition that goes all the way around,
            the centre_x and centre_y params are only present for naming the debug image files created,
            returns two vectors, y co-ord for every angle of the edges, or a reason if one or both not found,
            we look for the outer first as that will fail when looking at junk whereas the inner may not
            """

        # create a vector of the outer edge radius for every angle (this can be distorted by curvature)
        b2w_edges, threshold_x, threshold_y = self._get_transitions(target, 0, 1, False)
        ring_outer_edge = self._find_radius_edge(b2w_edges, self.TOP_DOWN, threshold_x, threshold_y,
                                                 centre_x, centre_y)

        if ring_outer_edge is None:
            # don't bother looking for the inner if there is no outer
            ring_inner_edge = None
            w2b_edges = None
        else:
            # found an outer, so worth looking for an inner
            # create a vector of the inner edge radius for every angle (this is usually accurate)
            w2b_edges, threshold_x, threshold_y = self._get_transitions(target, 0, 1, True)
            ring_inner_edge = self._find_radius_edge(w2b_edges, self.TOP_DOWN, threshold_x, threshold_y,
                                                     centre_x, centre_y)

        if ring_outer_edge is None:
            reason = 'no outer edge'
        elif ring_inner_edge is None:
            reason = 'no inner edge'
        else:
            reason = None

        if self.save_images:
            if w2b_edges is not None:
                if ring_inner_edge is not None:
                    points = [[0, ring_inner_edge]]
                else:
                    points = None
                plot = self._draw_plots(w2b_edges, points, None, (0, 255, 0))
                self._unload(plot, 'inner', centre_x, centre_y)

            if b2w_edges is not None:
                if ring_outer_edge is not None:
                    points = [[0, ring_outer_edge]]
                else:
                    points = None
                plot = self._draw_plots(b2w_edges, points, None, (0, 0, 255))
                self._unload(plot, 'outer', centre_x, centre_y)

            plot = target
            if ring_inner_edge is not None:
                plot = self._draw_plots(plot, [[0, ring_inner_edge]], None, (0, 255, 0))
            if ring_outer_edge is not None:
                plot = self._draw_plots(plot, [[0, ring_outer_edge]], None, (0, 0, 255))
            self._unload(plot, 'wavy', centre_x, centre_y)

        return ring_inner_edge, ring_outer_edge, reason

    def _flatten(self, target, centre_x, centre_y):
        """ remove perspective distortions from the given 'projected' image,
            the centre_x and centre_y params are only present for naming the debug image files created,
            a circle when not viewed straight on appears as an ellipse, when that is projected into a rectangle
            the radius edges becomes 'wavy' (a sine wave), this function straightens those wavy edges, other
            distortions can arise if the target is curved (e.g. if it is wrapped around someones leg), in
            this case the outer black ring can appear narrow in some parts of the circle (this does not happen
            with the inner ring because it is inside the code and curvature there would need to be vast to
            make any significant difference),
            the 'straightening' is just a matter of scaling such that all angles span the same range from
            the inner edge to the outer edge,
            the returned image is just enough to contain the (reduced) image pixels of all the target rings,
            """

        # find our marker edges in the radius
        ring_inner_edge, ring_outer_edge, reason = self._find_extent(target, centre_x, centre_y)
        if reason is not None:
            return None, reason

        # get the edge and distance limits we need
        max_x, max_y = target.size()
        min_inner_edge = max_y
        max_inner_edge = 0
        max_outer_edge = 0
        max_distance = 0
        for x in range(max_x):
            inner_edge = ring_inner_edge[x]
            outer_edge = ring_outer_edge[x]
            if inner_edge is None:
                return None, 'inner edge has a gap at {}'.format(x)
            if outer_edge is None:
                return None, 'outer edge has a gap at {}'.format(x)
            distance = outer_edge - inner_edge
            if distance < self.min_inner_outer_span:
                # we're looking at junk
                return None, 'inner to outer edge too small {}, limit {}'.format(distance, self.min_inner_outer_span)
            if distance > max_distance:
                max_distance = distance
            if inner_edge < min_inner_edge:
                min_inner_edge = inner_edge
            if inner_edge > max_inner_edge:
                max_inner_edge = inner_edge
            if outer_edge > max_outer_edge:
                max_outer_edge = outer_edge

        # calculate estimated ring width as the scaled distance divided by number of rings in that distance
        # max_distance is that between the inner white/black edge and the outer black/white edge, that spans
        # all but the two inner white rings and one outer white rings (i.e. -3)
        ring_span = self.NUM_RINGS - 3

        # create a new image scaled as appropriate
        code = self.original.instance().new(max_x, max_y, mid_luminance)
        max_new_y = 0                    # will be max radius of the image after perspective corrections
        min_new_y = max_y                # will be min radius of the image after perspective corrections
        for x in range(max_x):
            # the objective here is to create an image where the inner and outer edges are straight
            # the adjusted inner edge is positioned at max_inner_edge
            # pixels inside the inner edge are scaled such that the last is positioned on the inner edge
            # pixels outside the inner edge are scaled and placed after the inner edge
            start_y = ring_inner_edge[x]
            stop_y = ring_outer_edge[x]
            span = stop_y - start_y
            one_ring = span / ring_span
            y_scale = max_distance / span
            ref_y = max_inner_edge                       # the reference point for scaling
            # do the inside of ref point first
            last_y = ref_y
            last_pixel = mid_luminance
            for y in range(start_y-1, -1, -1):
                span = start_y - y
                new_y = int(round(ref_y - (span * y_scale)))
                if new_y < 0:
                    # this means there is not enough room for all our pixels, just truncate to here
                    # but first do the last pixel
                    while last_y > 0:
                        last_y -= 1
                        code.putpixel(x, last_y, last_pixel)
                    break
                pixel = target.getpixel(x, y)
                if pixel is None:
                    # this means we've gone off the top
                    # to prevent a false edge detection we propagate the last pixel
                    pixel = last_pixel
                last_pixel = pixel
                while last_y > new_y:
                    last_y -= 1
                    code.putpixel(x, last_y, pixel)
                if new_y < min_new_y:
                    min_new_y = new_y
                if new_y > max_new_y:
                    max_new_y = new_y
            # to prevent false edges being detected inside the bullseye we propagate the last pixel
            # backwards to our image edge, new_y is how far we got, last_pixel is the last pixel value
            if new_y < 0:
                # we went to the edge anyway
                pass
            else:
                # there is a residue to be filled
                for y in range(new_y, -1, -1):
                    code.putpixel(x, y, last_pixel)
            # now do the outside of ref point pixels
            stop_y += one_ring                           # go forward over our outer white ring
            last_y = ref_y - 1
            last_pixel = mid_luminance
            for y in range(int(round(start_y)), int(round(stop_y+1))):
                span = y - start_y
                new_y = int(round((span * y_scale) + ref_y))
                pixel = target.getpixel(x, y)
                if pixel is None:
                    # this means we've gone off the bottom
                    # to prevent a false edge detection we propagate the last pixel
                    pixel = last_pixel
                last_pixel = pixel
                while last_y < new_y:
                    last_y += 1
                    code.putpixel(x, last_y, pixel)
                if new_y < min_new_y:
                    min_new_y = new_y
                if new_y > max_new_y:
                    max_new_y = new_y
            # to prevent false edges being detected outside the outer white we propagate the last pixel
            # to our image bottom edge, new_y is how far we got, last_pixel is the last pixel value
            if new_y == (max_y - 1):
                # we went to the edge anyway
                pass
            else:
                # there is a residue to be filled
                for y in range(new_y, max_y):
                    code.putpixel(x, y, last_pixel)
                continue

        # crop the image to the max limits we actually filled
        old_buffer = code.get()
        new_buffer = old_buffer[min_new_y:max_new_y+1, 0:max_x]
        code.set(new_buffer)

        # upsize small targets
        code = self.transform.upsize(code, self.min_target_radius)

        # return flattened image
        return code, None

    def _get_ring_edges(self, target, _, centre_x, centre_y):
        """ return a vector of the ring edges in the given target,
            the target must consist of either white-to-black edges or black-to-white edges or both,
            centre_x/y are purely for diagnostic messages,
            this function is called generically, we do not use the '_' parameter here,
            returns the result and an image with detections drawn on it (None if not in debug mode)
            """

        max_x, max_y = target.size()

        # initial estimate of ring widths
        nominal_width = max_y / self.NUM_RINGS

        # margins within which we do not want edges (first and last edge must be outside these margins)
        low_margin = int(nominal_width * (1 - self.ring_edge_tolerance)) + 1
        high_margin = int(max(max_y - low_margin, 0))

        edge_mean = self._get_edge_threshold(target)
        threshold_x = int(round(edge_mean * self.edge_threshold_x))
        threshold_y = int(round(edge_mean * self.edge_threshold_y))

        if self.save_images:
            grid = target
        else:
            grid = None

        # an edge has to be a minimum length to qualify (half a bit width)
        # so we probe x in increments of that, anything missed is by definition too small
        # we probe every y looking for a pixel over our edge threshold, then follow it left and right to its end
        min_length = int(max((max_x / self.size) * self.min_ring_edge_length, 1))
        span_limit = int(max(nominal_width * self.ring_edge_span_limit, 1))
        ring_edges = []
        y = low_margin - 1
        while y < high_margin:
            y += 1
            x = 0
            next_y = y
            while x < (max_x - 1):
                x += 1
                left_edge = self._follow_edge_in_x(target, x, y,
                                                   self.RIGHT_TO_LEFT,
                                                   threshold_x, threshold_y,
                                                   self.edge_kernel_tight,
                                                   span_limit)
                if len(left_edge) == 0:
                    # found nothing going left, this means we'll also find nothing going right
                    continue
                elif len(left_edge) == max_x:
                    # we've found one that goes right round, so right_edge must be empty
                    right_edge = []
                else:
                    right_edge = self._follow_edge_in_x(target, x, y,
                                                        self.LEFT_TO_RIGHT,
                                                        threshold_x, threshold_y,
                                                        self.edge_kernel_tight,
                                                        span_limit)
                    if len(right_edge) == max_x:
                        # we've found one that goes right round, so dump left_edge
                        left_edge = []
                edge_length = len(left_edge) + len(right_edge) - 1   # -1 'cos they overlap at x,y
                if edge_length < min_length:      # ring edge must be at least this long
                    # too small ignore it
                    if self.show_log:
                        self._log('ignoring short ring edge {} at {},{}, limit is {}'.
                                  format(self.Edge.show(y, edge_length), x, y, min_length), centre_x, centre_y)
                    x += len(right_edge)          # move x past this edge
                    continue
                if edge_length > max_x:
                    # we've wrapped in a spiral!! what to do?
                    raise Exception('Blob {:.0f}x{:.0f}Y - ring edge greater than one revolution! Got {} when limit is {}'.
                                    format(centre_x, centre_y, edge_length, max_x))
                # it qualifies, note the median y as the edge
                min_edge_y = max_y
                max_edge_y = y                    # start here as do not want y to go backwards
                edge_y = 0
                for dy in left_edge:
                    if dy < min_edge_y:
                        min_edge_y = dy
                    if dy > max_edge_y:
                        max_edge_y = dy
                    edge_y += dy
                for dy in right_edge:
                    if dy < min_edge_y:
                        min_edge_y = dy
                    if dy > max_edge_y:
                        max_edge_y = dy
                    edge_y += dy
                edge_span = max_edge_y - min_edge_y + 1
                edge_y = int(round(edge_y / (len(left_edge) + len(right_edge))))
                ring_edges.append(self.Edge(edge_y, edge_length, edge_span))
                if self.show_log:
                    self._log('adding ring edge {} at x:{} y:{}'.
                              format(self.Edge.show(edge_y, edge_length, edge_span), x, y), centre_x, centre_y)
                if self.save_images:
                    if grid is not None:
                        # draw the edge we detected in green
                        left_edge.reverse()
                        grid = self._draw_plots(grid, [[x, right_edge], [x-(len(left_edge)-1), left_edge]], None, (0, 255, 0))
                # we move on from this edge
                # coincident ring edges in x do not add any new information, so once we've found one, move on in y
                next_y = max_edge_y               # move the y loop on
                x += len(right_edge)              # move x past this edge
                continue
            y = next_y

        return ring_edges, grid

    def _get_bit_edges(self, target, rings, centre_x, centre_y):
        """ return a vector of the bit edges in the given target within the data rings given,
            rings is an array of the detected ring edges and their widths,
            the target must consist of either white-to-black edges or black-to-white edges or both
            each edge consists of:
                edge_x,
                edge_y,
                edge_length (in y),
                edge_span (in x),
                None, None, None - space for stuff to come later
            centre_x/y are purely for diagnostic messages,
            returns the result and an image with detections drawn on it (None if not in debug mode)
            """

        max_x, max_y = target.size()

        edge_mean = self._get_edge_threshold(target)
        threshold_x = int(round(edge_mean * self.edge_threshold_x))
        threshold_y = int(round(edge_mean * self.edge_threshold_y))

        # determine the data ring centres and width limits
        data_rings = [int(round(rings[self.DATA_RING_1][0] + (rings[self.DATA_RING_1][1] / 2))),
                      int(round(rings[self.DATA_RING_2][0] + (rings[self.DATA_RING_2][1] / 2))),
                      int(round(rings[self.DATA_RING_3][0] + (rings[self.DATA_RING_3][1] / 2)))]
        # bit edge must be at least this long
        thin_lengths = [int(rings[self.DATA_RING_1][1] * self.thin_bit_edge_length),
                        int(rings[self.DATA_RING_2][1] * self.thin_bit_edge_length),
                        int(rings[self.DATA_RING_3][1] * self.thin_bit_edge_length)]
        thick_lengths = [int(rings[self.DATA_RING_1][1] * self.thick_bit_edge_length),
                         int(rings[self.DATA_RING_2][1] * self.thick_bit_edge_length),
                         int(rings[self.DATA_RING_3][1] * self.thick_bit_edge_length)]
        # set y limits to aid detection of edges that are mostly out of range
        lowest_y = int(round(rings[self.DATA_RING_1][0]))
        highest_y = int(round(rings[self.DATA_RING_3][0] + rings[self.DATA_RING_3][1]))

        if self.save_images:
            grid = target
        else:
            grid = None

        # get a vector of the edges for the data rings (NB: the ring wraps at end back to start)
        bit_edges = []
        x = -1
        while x < max_x:
            x += 1
            next_x = x
            for ring in range(len(data_rings)):
                y = data_rings[ring]
                thin_length = thin_lengths[ring]
                thick_length = thick_lengths[ring]
                down_edge = self._follow_edge_in_y(target, x, y,
                                                   self.TOP_DOWN,
                                                   threshold_x, threshold_y,
                                                   self.edge_kernel_tight)
                if len(down_edge) == 0:
                    # found nothing going down, this means we'll also find nothing going up
                    # NB: This assumes we do not probe off the image edge, OK as ring edges cannot be there
                    continue
                else:
                    up_edge = self._follow_edge_in_y(target, x, y,
                                                     self.BOTTOM_UP,
                                                     threshold_x, threshold_y,
                                                     self.edge_kernel_tight)
                edge_length = len(down_edge) + len(up_edge) - 1    # -1 'cos they overlap at x,y
                # suppress edges that are mostly above ring the first ring or below the last one
                # these are probably noise
                overspill = (y + len(down_edge)) - highest_y
                if (overspill / edge_length) > self.max_bit_edge_overspill:
                    if self.show_log:
                        self._log('ignoring overspill bit edge ({}) at {},{} in ring {}, overspill is {}'.
                                  format(self.Edge.show(x, edge_length), x, y, ring, overspill),
                                  centre_x, centre_y)
                    continue
                overspill = lowest_y - (y - len(up_edge))
                if (overspill / edge_length) > self.max_bit_edge_overspill:
                    if self.show_log:
                        self._log('ignoring overspill bit edge ({}) at {},{} in ring {}, overspill is {}'.
                                  format(self.Edge.show(x, edge_length), x, y, ring, overspill),
                                  centre_x, centre_y)
                    continue
                # it qualifies, note the median x as the edge
                min_edge_x = max_x
                max_edge_x = x                 # start here as do not want x to go backwards
                for dx in down_edge:
                    if dx < min_edge_x:
                        min_edge_x = dx
                    if dx > max_edge_x:
                        max_edge_x = dx
                for dx in up_edge:
                    if dx < min_edge_x:
                        min_edge_x = dx
                    if dx > max_edge_x:
                        max_edge_x = dx
                edge_span = max_edge_x - min_edge_x
                if edge_span > (self.angle_steps / 2):
                    # this edge wraps
                    edge_span = min_edge_x + self.angle_steps - max_edge_x
                if edge_span < self.thin_bit_edge_width and edge_length < thin_length:
                    # too thin and short ignore it
                    if self.show_log:
                        self._log('ignoring thin short bit edge ({}) at {},{} in ring {}, limit is {}'.
                                  format(self.Edge.show(x, edge_length, edge_span), x, y, ring, thin_length),
                                  centre_x, centre_y)
                    continue
                elif edge_span >= self.thin_bit_edge_width and edge_length < thick_length:
                    # too thin and short ignore it
                    if self.show_log:
                        self._log('ignoring thick short bit edge ({}) at {},{} in ring {}, limit is {}'.
                                  format(self.Edge.show(x, edge_length, edge_span), x, y, ring, thick_length),
                                  centre_x, centre_y)
                    continue
                edge_span += 1
                edge_x = int(round(min_edge_x + (edge_span / 2)))  # set as middle of span
                bit_edges.append(self.Edge(edge_x, edge_length, edge_span))
                if self.show_log:
                    self._log('adding bit edge {} at x:{} y:{} ring:{} '.
                              format(bit_edges[-1], x, y, ring), centre_x, centre_y)
                if self.save_images:
                    if grid is not None:
                        # draw the edge we detected in green
                        up_edge.reverse()
                        grid = self._draw_plots(grid, None, [[y, down_edge], [y - (len(up_edge) - 1), up_edge]], (0, 255, 0))
                # we move on from this edge
                # coincident bit edges in y do not add any new information, so once we've found one, move on in x
                next_x = max_edge_x                 # move the x loop on
            x = next_x

        return bit_edges, grid

    def _get_edges(self, target, direction, extra_data, centre_x, centre_y):
        """ find all the edges (black to white (up) and white to black (down)) in the given target,
            direction is either top-down (looking for ring edges) or left-to-right (looking for bit edges),
            extra_data is just passed on to the specific edge detector,
            this function handles the logistics of finding both 'up' and 'down' edges and merging them into
            a single vector for returning to the caller, an empty vector is returned if no edges found,
            centre_x/y are purely for diagnostic image naming purposes
            """

        if direction == self.TOP_DOWN:
            # looking for ring edges in y
            x_order = 0
            y_order = 1
            edge_finder = self._get_ring_edges
            context = 'ring'
        elif direction == self.LEFT_TO_RIGHT:
            # looking for bit edges in x
            x_order = 1
            y_order = 0
            edge_finder = self._get_bit_edges
            context = 'bit'
        else:
            raise Exception('illegal direction {}'.format(direction))

        # get black-to-white edges
        b2w_edges = self.transform.edges(target, x_order, y_order, 3, False)
        edges_up, grid = edge_finder(b2w_edges, extra_data, centre_x, centre_y)
        if self.save_images:
            if grid is None:
                grid = b2w_edges
            self._unload(grid, '{}-b2w-edges'.format(context), centre_x, centre_y)

        # get white_to_black edges
        w2b_edges = self.transform.edges(target, x_order, y_order, 3, True)
        edges_down, grid = edge_finder(w2b_edges, extra_data, centre_x, centre_y)
        if self.save_images:
            if grid is None:
                grid = w2b_edges
            self._unload(grid, '{}-w2b-edges'.format(context), centre_x, centre_y)

        # merge the two sets of edges so that got a single list in ascending order, each is separately in order now,
        # we get them separately and merge rather than do both at once 'cos they can get very close in small images
        # we also drop overlapping edges (the 'best' is kept, 'best' is thinner or longer)
        edges = []
        edge_up = None
        edge_down = None
        while True:

            # get the next pair to consider
            if edge_up is None and len(edges_up) > 0:
                edge_up = edges_up.pop(0)
            if edge_down is None and len(edges_down) > 0:
                edge_down = edges_down.pop(0)

            # determine which one to add next
            if edge_up is not None and edge_down is not None:
                # got both, add lowest
                if edge_up.position < edge_down.position:
                    this_edge = edge_up
                    edge_up = None
                else:
                    this_edge = edge_down
                    edge_down = None
            elif edge_up is not None:
                # only got an up
                this_edge = edge_up
                edge_up = None
            elif edge_down is not None:
                # only got a down
                this_edge = edge_down
                edge_down = None
            else:
                # run out, so we're done
                break

            # add this edge if does not overlap with what is already there
            if len(edges) == 0:
                # nothing to compare, just add it
                if self.show_log:
                    self._log('merging {} edge {}'.format(context, this_edge), centre_x, centre_y)
                edges.append(this_edge)
                continue
            that_edge = edges[-1]              # get what is already there

            # if they overlap keep the 'best', if that is the one we are holding then replace end of list
            # else just ignore the one we are holding
            # the one we are holding is referred to as 'this' and the one already there as 'that'
            # we know 'that' is <= 'this', thus an overlap occurs if end 'that' >= start 'this'
            start_this_span = this_edge.position - (this_edge.span / 2)
            end_this_span = start_this_span + this_edge.span
            start_that_span = that_edge.position - (that_edge.span / 2)
            end_that_span = start_that_span + that_edge.span
            if end_that_span < start_this_span:
                # no overlap, add it
                edges.append(this_edge)
                continue
            # overlap, drop the biggest span one
            if this_edge.span > that_edge.span:
                # drop this_edge
                if self.show_log:
                    self._log('dropping overlapping wide {} edge {} (overlaps with {})'.
                              format(context, this_edge, that_edge), centre_x, centre_y)
                # just ignore this one
                continue
            elif this_edge.span < that_edge.span:
                # drop that_edge
                if self.show_log:
                    self._log('dropping overlapping wide {} edge {} (overlaps with {})'.
                              format(context, that_edge, this_edge), centre_x, centre_y)
                # replace end of list with this one
                edges[-1] = this_edge
                continue
            # same span, drop shortest
            if this_edge.length > that_edge.length:
                if self.show_log:
                    self._log('dropping overlapping short {} edge {} (overlaps with {})'.
                              format(context, that_edge, this_edge), centre_x, centre_y)
                # replace end of list with this one
                edges[-1] = this_edge
                continue
            elif this_edge.length < that_edge.length:
                if self.show_log:
                    self._log('dropping overlapping short {} edge {} (overlaps with {})'.
                              format(context, this_edge, that_edge), centre_x, centre_y)
                # just ignore this one
                continue
            # same span and same length, arbitrarily drop the this edge
            if self.show_log:
                self._log('dropping overlapping similar {} edge {} (overlaps with {})'.
                          format(context, this_edge, that_edge), centre_x, centre_y)
            # just ignore this one
            continue

        if self.show_log:
            self._log('Found {} {} edges'.format(len(edges), context), centre_x, centre_y)
            for edge in edges:
                self._log('    {} edge: {}'.format(context, edge))

        return edges

    def _match_edges(self, target, edges, direction):
        """ given a list of actual edge detections, for each build a list of probable matches to expectations,
            target is present purely to find image limits,
            direction is TOP_DOWN (matching ring edges) or LEFT_TO_RIGHT (matching bit edges),
            returns the updated edges list with an array of actual matches and their deviation from ideal,
            if there is no (close enough) match for an edge expectation None is set
            """

        max_x, max_y = target.size()
        if direction == self.TOP_DOWN:
            distance_range = max_y
            error_tolerance = self.ring_edge_tolerance
            wanted_edges = self.NUM_RINGS
            nominal_width = max_y / self.NUM_RINGS
        elif direction == self.LEFT_TO_RIGHT:
            distance_range = max_x
            error_tolerance = self.bit_edge_tolerance
            wanted_edges = self.size
            nominal_width = max_x / self.size
        else:
            raise Exception('illegal direction {}'.format(direction))

        # build vectors of best fit actual edges for every edge we found
        # for every actual edge we iterate all its ideal edges looking for the closest match in all actual edges
        # ToDo: this is a crude O(N*N) algorithm, find a better way
        for this in range(len(edges)):
            this_edge = edges[this]
            actual_edges = [None for _ in range(wanted_edges)]
            ideal_edges = [None for _ in range(wanted_edges)]
            ideal_edge = this_edge.position - nominal_width    # first one is at the actual
            for edge in range(wanted_edges):
                ideal_edge = (ideal_edge + nominal_width) % distance_range   # move on to next expected edge
                ideal_edges[edge] = ideal_edge                 # save for later error analysis
                best_distance = distance_range
                best_edge = None
                for that in range(len(edges)):
                    that_edge = edges[that]
                    # if this ideal is close enough to that actual use it
                    distance = math.fabs(ideal_edge - that_edge.position) % distance_range
                    if ((distance / nominal_width) < error_tolerance) and distance < best_distance:
                        best_distance = distance
                        best_edge = that_edge
                if best_edge is not None:
                    actual_edges[edge] = best_edge.position
                    ideal_edge = best_edge.position            # re-sync ideal to the edge we picked

            this_edge.actual = actual_edges                    # note what we're using
            this_edge.ideal = ideal_edges                      # ..and the ideal
            # count matches for best picker
            matches = 0
            for edge in actual_edges:
                if edge is not None:
                    matches += 1
            this_edge.matches = matches

        return edges

    def _pick_best_edge(self, edges, centre_x, centre_y, context=''):
        """ given a vector of edges pick the best one,
            edges may be ring edges or bit edges,
            returns the best match from the set given,
            centre_x/y and context are only present for debug messages
            """
        best = edges[0]
        if self.show_log:
            self._log('{}: picking {} as initial best candidate'.
                      format(context, best.actual), centre_x, centre_y)
        for candidate in range(1, len(edges)):
            edge = edges[candidate]
            if edge.matches > best.matches:
                # this one has more matches
                if self.show_log:
                    self._log('{}: best now {} (more matches than {})'.
                              format(context, edge.actual, best.actual))
                best = edge
            elif edge.matches < best.matches:
                # worse match, ignore it
                if self.show_log:
                    self._log('{}: best staying {} (more matches than {})'.
                              format(context, best.actual, edge.actual))
                pass
            else:
                # same, so pick the better of the two:
                #   least edge error to the ideal
                #   longer is better
                #   thinner is better
                # calculate the match error (accumulative diff of ideal to actual
                edge_err = 0
                best_err = 0
                for sample in range(len(edge.actual)):
                    ideal_edge = edge.ideal[sample]
                    ideal_best = best.ideal[sample]
                    actual_edge = edge.actual[sample]
                    actual_best = best.actual[sample]
                    if actual_edge is not None and ideal_edge is not None:
                        edge_err += math.fabs(actual_edge - ideal_edge)
                    if actual_best is not None and ideal_best is not None:
                        best_err += math.fabs(actual_best - ideal_best)
                if edge_err < best_err:
                    # got a closer match
                    if self.show_log:
                        self._log('{}: best now {} (smaller error to ideal than {})'.
                                  format(context, edge.actual, best.actual))
                    best = edge
                elif edge_err > best_err:
                    # this one worse, ignore it
                    if self.show_log:
                        self._log('{}: best staying {} (smaller error to ideal than {})'.
                                  format(context, best.actual, edge.actual))
                    pass
                elif edge.length > best.length:
                    # longer is better
                    if self.show_log:
                        self._log('{}: best now {} (longer edge than {})'.
                                  format(context, edge.actual, best.actual))
                    best = edge
                elif edge.length == best.length and edge.span < best.span:
                    # thinner is better
                    if self.show_log:
                        self._log('{}: best now {} (thinner edge than {})'.
                                  format(context, edge.actual, best.actual))
                    best = edge
                elif edge.length == best.length and edge.span == best.span:
                    # run out of criteria!
                    # keep the first one we found
                    if self.show_log:
                        self._log('identical {} edge candidates at {} and {}'.
                                  format(context, edge.position, best.position),
                                  centre_x, centre_y)
                        self._log('     ideal(edge): {}'.format(edge.ideal))
                        self._log('    actual(edge): {}'.format(edge.actual))
                        self._log('     ideal(best): {}'.format(best.ideal))
                        self._log('    actual(best): {}'.format(best.actual))
                else:
                    # this one is worse, keep the one we got
                    if self.show_log:
                        self._log('{}: best staying {} (worse edge {})'.
                                  format(context, edge.actual, best.actual))
                    pass

        if self.show_log:
            self._log('{}: picking {} as final best candidate'.
                      format(context, best.actual), centre_x, centre_y)

        return best

    def _fill_edge_gaps(self, target, actual_edges, direction, centre_x, centre_y):
        """ given an edge list interpolate across gaps to fill with an expectation,
            a 'gap' is a None value, the expectation is direction dependent,
            direction is TOP_DOWN (for rings) or LEFT_TO_RIGHT (for bits),
            target is the image the edges were detected in, its used to get dimension limits
            and to add a highlight of inserted edges,
            centre_x/y are only present for debug purposes,
            returns an edge list with no None values and in ascending order
            along with the modified target image when in debug mode (else None)
            """

        max_x, max_y = target.size()
        if direction == self.TOP_DOWN:
            extent = self.NUM_RINGS
            edge_limit = max_y
            context = 'ring'
        elif direction == self.LEFT_TO_RIGHT:
            extent = self.size
            edge_limit = max_x
            context = 'bit'
        else:
            raise Exception('given illegal direction {}'.format(direction))

        if self.save_images:
            grid = target
        else:
            grid = None

        start_at = None
        gap = 0
        for edge in range(extent + 1):         # go one extra so we get the 'wrapped' case
            if actual_edges[edge % extent] is None:
                # got a gap (NB: we know the first edge is not None as that's the 'anchor' we started with)
                if start_at is None:
                    # start of a new gap, note it
                    start_at = actual_edges[(edge - 1) % extent]
                    gap = edge
                else:
                    # continuing in same gap
                    pass
                continue
            if self.save_images:
                # highlight the edge we are using in blue
                if direction == self.TOP_DOWN:
                    plot_x = [actual_edges[edge % extent]]
                    plot_y = None
                else:
                    plot_x = None
                    plot_y = [actual_edges[edge % extent]]
                grid = self._draw_grid(grid, plot_x, plot_y, (255, 0, 0))
            if start_at is not None:
                # end of gap, fill in the missing edges
                stop_at = actual_edges[edge % extent]
                if stop_at < start_at:
                    # we've wrapped
                    span = (stop_at + edge_limit) - start_at
                else:
                    span = stop_at - start_at
                gaps = edge - gap
                width = span / (gaps + 1)
                for extra in range(gaps):
                    new_edge = int((start_at + ((extra + 1) * width)) % edge_limit)
                    if self.save_images:
                        # draw the edge we're inserting in red
                        if direction == self.TOP_DOWN:
                            plot_x = [new_edge]
                            plot_y = None
                        else:
                            plot_x = None
                            plot_y = [new_edge]
                        grid = self._draw_grid(grid, plot_x, plot_y, (0, 0, 255))
                    if self.show_log:
                        self._log('inserting {} edge at {}'.format(context, new_edge), centre_x, centre_y)
                    actual_edges[(gap + extra) % extent] = new_edge
            start_at = None
            gap = 0

        actual_edges.sort()

        if self.show_log:
            self._log('{}: final actual edges: {}'.
                      format(context, actual_edges), centre_x, centre_y)

        return actual_edges, grid

    def _find_ring_edges(self, target, centre_x, centre_y):
        """ find the ring edges in the given target,
            the target must have been perspective corrected and cropped,
            the centre_x, and centre_y params are only present for naming the debug image files created,
            this deduces the ring edges by finding all horizontal edges, both white to black and black to white,
            in the image given, the image given is almost flat so there should be distinct edge radii that fall
            on the ring edges, the data coding is such that there will be at least one edge per ring, we know
            how many rings we expect so from all the edges we find a best fit to our expectation, we go to this
            trouble because image distortion can squash and stretch the rings (in particular white tends to bleed
            into black which shrinks mostly black rings and expands mostly white rings),
            returns a vector of y co-ords for the ring edges or a reason if failed
            """

        # get the ring edge candidates
        ring_edges = self._get_edges(target, self.TOP_DOWN, None, centre_x, centre_y)
        if len(ring_edges) == 0:
            return None, 'found no ring edges'

        if self.save_images:
            # draw all detected edges in green on our target
            detections = []
            for edge in ring_edges:
                detections.append(edge.position)
            grid = self._draw_grid(target, detections, None, (0, 255, 0))
        else:
            grid = target

        _, max_y = target.size()

        # match actual edges to expectations
        ring_edges = self._match_edges(target, ring_edges, self.TOP_DOWN)

        # pick the actual edge set with the best matches
        best = self._pick_best_edge(ring_edges, centre_x, centre_y, 'ring')

        # fill in the gaps in the set we've chosen
        actual_edges, grid = self._fill_edge_gaps(grid, best.actual, self.TOP_DOWN, centre_x, centre_y)

        if self.save_images:
            self._unload(grid, 'rings', centre_x, centre_y)

        # now got ring edges in ascending order in actual_edges,
        # actual edges will include *all* our code rings including the inner two white rings
        # caller wants start+width for each ring

        # ring width is start ring(N+1) - ring(N)
        rings = []
        for ring in range(len(actual_edges) - 1):
            prev_edge = actual_edges[ring]
            next_edge = actual_edges[ring + 1]
            if prev_edge is None or next_edge is None:
                raise Exception('Blob {:.0f}x{:.0f}y - ring {} is {} and ring {} is {}, None not allowed'.
                                format(centre_x, centre_y, ring, prev_edge, ring + 1, next_edge))
            rings.append((prev_edge, next_edge - prev_edge))
        # final ring is from last edge to image edge
        rings.append((actual_edges[-1], max_y - actual_edges[-1]))

        if len(rings) != self.NUM_RINGS:
            # oops - sumink wrong sumplace!
            raise Exception('Blob {:.0f}x{:.0f}y - expected {} rings, but constructed {}'.
                            format(centre_x, centre_y, self.NUM_RINGS, len(rings)))

        return rings, None

    def _find_bit_edges(self, target, rings, centre_x, centre_y):
        """ find the bit edges in the given target within the data rings defined in rings,
            the target must have been perspective corrected,
            rings is an array of the detected ring edges and their width,
            the centre_x, and centre_y params are only present for naming the debug image files created,
            this function looks for vertical edges in the x direction, both black to white and white to black,
            within the data rings given, edges are bit edges, we need at least one bit edge to calculate where
            all the other bit edges are, the data ring coding guarantees at least two for each ring, this means
            we should find at least six across the three data rings, which is sufficient to compensate for
            possible stretching and shrinking around the ring caused by image distortion (e.g. due to it being
            screwed up and then flattened then pinned around a leg),
            we know the nominal spacing of the bits, so the more edges we detect the more accurate the bit
            missing bit edge calculation will be,
            returns a vector of the x co-ord of all the bit edges
            """

        # get all the candidate edges
        bit_edges = self._get_edges(target, self.LEFT_TO_RIGHT, rings, centre_x, centre_y)
        if len(bit_edges) == 0:
            # no edges detected at all, we need at least one to find bit centres
            return None, 'found no bit edges'

        if self.save_images:
            # draw all detected edges in green on our target
            detections = []
            for edge in bit_edges:
                detections.append(edge.position)
            grid = self._draw_grid(target, None, detections, (0, 255, 0))
        else:
            grid = target

        max_x, max_y = target.size()
        nominal_width = max_x / self.size

        # match actual edges to expectations
        bit_edges = self._match_edges(target, bit_edges, self.LEFT_TO_RIGHT)

        # pick the actual edge set with the best matches
        best = self._pick_best_edge(bit_edges, centre_x, centre_y, 'bit')

        # fill in the gaps in the set we've chosen
        actual_edges, grid = self._fill_edge_gaps(grid, best.actual, self.LEFT_TO_RIGHT, centre_x, centre_y)

        if self.save_images:
            self._unload(grid, 'bits', centre_x, centre_y)

        # we've done it
        return actual_edges, None

    def _find_targets(self):
        """ find targets within our image,
            the detection consists of several steps:
                0. find all the image 'blobs'
                1. detect the central bullseye (as an opencv 'blob', done in __init__)
                2. project the circular target into a rectangle of radius x angle
                3. adjust for perspective distortions (a circle can look like an ellipsis)
                4. adjust for luminance variations
                5. classify the data ring bits (black, white or grey)
                6. decode the data bits
            there are validation constraints in most steps that may result in a target being rejected,
            the objective is to achieve 100% confidence in the result or reject it,
            returns a list of target images found which may be empty if none found and a labelled image when debugging
            and when debugging an image with all rejects labelled with why rejected (None if not debugging)
            """

        blobs = self._find_blobs()
        if len(blobs) == 0:
            # no blobs here
            return [], None

        targets = []
        status = []
        for blob in blobs:
            centre_x = blob.pt[0]
            centre_y = blob.pt[1]
            size = blob.size / 2               # change diameter to radius

            # do the polar to cartesian projection
            projected = self._project(centre_x, centre_y, blob.size)   # this does not fail

            # do the perspective correction
            flattened, reason = self._flatten(projected, centre_x, centre_y)
            if reason is not None:
                # failed - this means some constraint was not met
                if self.show_log:
                    self._log('{} - rejecting'.format(reason), centre_x, centre_y)
                if self.save_images:
                    status.append([centre_x, centre_y, size, None, reason])
                continue

            # got a corrected image, get the ring edges
            rings, reason = self._find_ring_edges(flattened, centre_x, centre_y)
            if reason is not None:
                # failed - this means some constraint was not met
                if self.show_log:
                    self._log('{} - rejecting'.format(reason), centre_x, centre_y)
                if self.save_images:
                    status.append([centre_x, centre_y, size, None, reason])
                continue

            if self.save_images:
                self._unload(flattened, 'flat', centre_x, centre_y)

            # find the bit boundaries in our data rings
            bits, reason = self._find_bit_edges(flattened, rings, centre_x, centre_y)
            if reason is not None:
                # failed - this means some constraint was not met
                if self.show_log:
                    self._log('{} - rejecting'.format(reason), centre_x, centre_y)
                if self.save_images:
                    status.append([centre_x, centre_y, size, rings[self.OUTER_BLACK][0], reason])
                continue

            # find the centres of the bit, these are half way between bit edges
            max_x, _ = flattened.size()
            centres = [None for _ in range(len(bits))]
            for bit in range(len(bits)):
                here = bits[bit]
                there = bits[(bit + 1) % len(bits)]
                if there < here:
                    # we've wrapped
                    width = (there + max_x) - here
                else:
                    width = there - here
                centres[bit] = (int((here + (width / 2)) % max_x), width)

            targets.append([centre_x, centre_y, size, flattened, rings, centres])

        if self.save_images:
            # label all the blobs we processed that were rejected
            labels = self.transform.copy(self.image)
            for reject in status:
                x = reject[0]
                y = reject[1]
                size = reject[2] / 2     # assume blob detected is just inner two white rings
                ring = reject[3]
                reason = reject[4]
                if ring is None:
                    # it got rejected before its inner/outer ring was detected
                    ring = size * 4
                # show blob detected
                colour = (255, 0, 0)     # blue
                labels = self.transform.label(labels, (x, y, size), colour)
                # show reject reason
                colour = (0, 0, 255)     # red
                labels = self.transform.label(labels, (x, y, ring), colour, '{:.0f}x{:.0f}y {}'.format(x, y, reason))
        else:
            labels = None

        return targets, labels

    def _get_sample(self, target, bit, ring, sample_size):
        """ given a perspective corrected image, a bit number, data and a box, get the luminance sample,
            bit specifies the centre of the sample box in the x direction in the target,
            ring specifies the centre of the sample box in the y direction in the target,
            sample_size is the width (x) and height (y) of the area to sample,
            returns the average luminance level in the sample box
            """
        start_x = int(round(bit - (sample_size[0] / 2)))
        stop_x = int(round(bit + (sample_size[0] / 2)))
        start_y = int(round(ring - (sample_size[1] / 2)))
        stop_y = int(round(ring + (sample_size[1] / 2)))
        luminance_accumulator = 0
        pixels_found = 0
        for x in range(start_x, stop_x+1):
            for y in range(start_y, stop_y+1):
                pixel = target.getpixel(x, y)
                if pixel is not None:
                    luminance_accumulator += pixel
                    pixels_found += 1
        if pixels_found > 0:
            return luminance_accumulator / pixels_found
        else:
            return min_luminance

    def decode_targets(self):
        """ find and decode the targets in the source image,
            the targets found are in self.targets, each consists of:
                x,y co-ordinates of the central blob,
                perspective corrected rectangular image of the target,
                the y co-ordinate of the centre of each ring - white level, black level, data rings 1, 2 and 3,
                the x co-ordinate of the centre of each data bit,
                the width, height of the luminance sample area
            returns a list of x,y blob co-ordinates, the encoded number there (or None) and the level of doubt
            """

        targets, labels = self._find_targets()
        if len(targets) == 0:
            if self.show_log:
                self._log('image {} does not contain any target candidates'.format(self.original.source))
            if self.save_images:
                if labels is not None:
                    self._unload(labels, 'targets', 0, 0)
                else:
                    self._unload(self.image, 'targets', 0, 0)
            return []

        numbers = []
        for target in targets:
            centre_x = target[0]
            centre_y = target[1]
            size = target[2]
            image = target[3]
            rings = target[4]
            bits = target[5]

            # calculate ring centre and sample size for each bit in each ring
            max_x, _ = image.size()
            ring_centres = [None for _ in range(len(rings))]
            sample_size = [[None for _ in range(len(bits))] for _ in range(len(rings))]
            for ring in range(len(rings)):
                ring_width = rings[ring][1]
                ring_centres[ring] = rings[ring][0] + int(round(ring_width / 2))
                sample_height = int(round(ring_width * self.sample_height_factor))
                for bit in range(len(bits)):
                    sample_width = int(bits[bit][1] * self.sample_width_factor)
                    sample_size[ring][bit] = (sample_width, sample_height)

            if self.save_images:
                grid = image
                for bit in range(len(bits)):
                    points = [bits[bit][0]]
                    grid = self._draw_grid(grid, [ring_centres[self.INNER_WHITE]], points, box=sample_size[self.INNER_WHITE][bit])
                    grid = self._draw_grid(grid, [ring_centres[self.DATA_RING_1]], points, box=sample_size[self.DATA_RING_1][bit])
                    grid = self._draw_grid(grid, [ring_centres[self.DATA_RING_2]], points, box=sample_size[self.DATA_RING_2][bit])
                    grid = self._draw_grid(grid, [ring_centres[self.DATA_RING_3]], points, box=sample_size[self.DATA_RING_3][bit])
                    grid = self._draw_grid(grid, [ring_centres[self.OUTER_BLACK]], points, box=sample_size[self.OUTER_BLACK][bit])
                self._unload(grid, 'grid', centre_x, centre_y)

            # the sample_size box around the intersection of each ring and each bit is what we look at
            white_level = [None for _ in range(len(bits))]
            black_level = [None for _ in range(len(bits))]
            data_ring_1 = [None for _ in range(len(bits))]
            data_ring_2 = [None for _ in range(len(bits))]
            data_ring_3 = [None for _ in range(len(bits))]
            for bit in range(len(bits)):
                x = bits[bit][0]
                white_level[bit] = self._get_sample(image, x, ring_centres[self.INNER_WHITE], sample_size[self.INNER_WHITE][bit])
                data_ring_1[bit] = self._get_sample(image, x, ring_centres[self.DATA_RING_1], sample_size[self.DATA_RING_1][bit])
                data_ring_2[bit] = self._get_sample(image, x, ring_centres[self.DATA_RING_2], sample_size[self.DATA_RING_2][bit])
                data_ring_3[bit] = self._get_sample(image, x, ring_centres[self.DATA_RING_3], sample_size[self.DATA_RING_3][bit])
                black_level[bit] = self._get_sample(image, x, ring_centres[self.OUTER_BLACK], sample_size[self.OUTER_BLACK][bit])

            # now decode what we got
            number, doubt, bits = self.c.unbuild([data_ring_1, data_ring_2, data_ring_3], [white_level, black_level])
            numbers.append([centre_x, centre_y, size, number, doubt, rings[self.OUTER_BLACK][0] + rings[self.OUTER_BLACK][1]])

            if self.show_log:
                number = numbers[-1]
                self._log('number:{}, bits:{}'.format(number[3], bits), number[0], number[1])
                if number[3] is None:
                    self._log('--- white samples:{}'.format(white_level))
                    self._log('--- black samples:{}'.format(black_level))
                    self._log('--- ring1 samples:{}'.format(data_ring_1))
                    self._log('--- ring2 samples:{}'.format(data_ring_2))
                    self._log('--- ring3 samples:{}'.format(data_ring_3))
            if self.save_images:
                number = numbers[-1]
                if number[3] is None:
                    colour = (0, 0, 255, 0)  # red
                    label = 'invalid code'
                else:
                    colour = (0, 255, 0)     # green
                    label = 'code is {}'.format(number[3])
                if labels is not None:
                    # draw the detected blob in blue
                    k = (number[0], number[1], number[2])
                    labels = self.transform.label(labels, k, (255, 0, 0))
                    # draw the result
                    k = (number[0], number[1], number[5])
                    labels = self.transform.label(labels, k, colour, '{:.0f}x{:.0f}y {}'.
                                                  format(number[0], number[1], label))
                    self._unload(labels, 'targets', 0, 0)

        return numbers

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
                            "start": (line[0], line[1]),
                            "end": (line[2], line[3])})
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

    def _draw_grid(self, source, horizontal=None, vertical=None, colour=(0, 0, 255), radius=None, box=None):
        """ draw horizontal and vertical lines across the given source image,
            if a radius is given: circles of that radius are drawn at each horizontal/vertical intersection,
            if a box is given: rectangles of that size are drawn centred at each horizontal/vertical intersection,
            lines are drawn in the given colour,
            circles and boxes are drawn in a pale version of the given colour,
            returns a new colour image of the result
            """
        max_x, max_y = source.size()
        objects = []
        if horizontal is not None:
            for h in horizontal:
                objects.append({"colour": colour,
                                "type": self.transform.LINE,
                                "start": (0, h),
                                "end": (max_x-1, h)})

        if vertical is not None:
            for v in vertical:
                objects.append({"colour": colour,
                                "type": self.transform.LINE,
                                "start": (v, 0),
                                "end": (v, max_y-1)})

        if horizontal is not None and vertical is not None:
            pale_colour = (int(colour[0]/2), int(colour[1]/2), int(colour[2]/2))
            if radius is not None:
                for h in horizontal:
                    for v in vertical:
                        objects.append({"colour": pale_colour,
                                        "type": self.transform.CIRCLE,
                                        "centre": (h, v),
                                        "radius": radius})
            if box is not None:
                width  = box[0]
                height = box[1]
                for h in horizontal:
                    for v in vertical:
                        start = (int(round((v - width/2))), int(round((h - height/2))))
                        end   = (int(round((v + width/2))), int(round((h + height/2))))
                        objects.append({"colour": pale_colour,
                                        "type": self.transform.RECTANGLE,
                                        "start": start,
                                        "end": end})

        target = self.transform.copy(source)
        return self.transform.annotate(target, objects)


class Test:

    # exit codes from scan
    EXIT_OK = 0                # found what was expected
    EXIT_FAILED = 1            # did not find what was expected
    EXIT_EXCEPTION = 2         # an exception was raised

    def __init__(self, code_bits, min_num, max_num, parity, edges, rings, contrast, offset):
        self.code_bits = code_bits
        self.min_num = min_num
        self.c = Codes(self.code_bits, min_num, max_num, parity, edges)
        self.frame = Frame()
        self.max_num = min(max_num, self.c.num_limit)
        self.num_rings = rings
        self.contrast = contrast
        self.offset = offset
        print('With {} code bits, {} parity, {} edges available numbers are {}..{}'.format(code_bits,
                                                                                           parity,
                                                                                           edges,
                                                                                           self.min_num,
                                                                                           self.max_num))

    def coding(self):
        """ test for encode/decode symmetry """
        print('')
        print('******************')
        print('Check encode/decode from {} to {}'.format(self.min_num, self.max_num))
        try:
            good = 0
            bad = 0
            for n in range(self.min_num, self.max_num + 1):
                if self.c.check(n) is None:
                    bad += 1
                else:
                    good += 1
            print('{} good, {} bad'.format(good, bad))
        except:
            traceback.print_exc()
        print('******************')

    def decoding(self, black=min_luminance, white=max_luminance, noise=0):
        """ test build/unbuild symmetry
            black is the luminance level to use for 'black', and white for 'white',
            noise is how much noise to add to the luminance samples created, when
            set luminance samples have a random number between 0 and noise added or
            subtracted, this is intended to test the 'maybe' logic, the grey thresholds
            are set such that the middle 1/3rd of the luminance range is considered 'grey'
            """
        print('')
        print('******************')
        print('Check build/unbuild from {} to {} with black={}, white={} and noise {}'.
              format(self.min_num, self.max_num, black, white, noise))
        try:
            colours = [black, white]
            good = 0
            fail = 0
            bad = 0
            levels = [[None for _ in range(self.code_bits)] for _ in range(2)]
            for bit in range(self.code_bits):
                levels[0][bit] = white
                levels[1][bit] = black
            for n in range(self.min_num, self.max_num + 1):
                rings = self.c.build(n)
                samples = [[] for _ in range(len(rings))]
                for ring in range(len(rings)):
                    word = rings[ring]
                    for bit in range(self.code_bits):
                        # NB: Being encoded big-endian (MSB first)
                        samples[ring].insert(0, max(min(colours[word & 1] + (random.randrange(0, noise+1) - (noise >> 1)),
                                                        max_luminance), min_luminance))
                        word >>= 1
                m, doubt, bits = self.c.unbuild(samples, levels)
                if m is None:
                    # failed to decode
                    fail += 1
                    if noise == 0:
                        print('    FAIL: {:03}-->{}, build={}, bits={}, samples={}'.format(n, m, rings, bits, samples))
                elif m != n:
                    # incorrect decode
                    bad += 1
                    print('****BAD!: {:03}-->{} , build={}, bits={}, samples={}'.format(n, m, rings, bits, samples))
                else:
                    good += 1
            print('{} good, {} bad, {} fail'.format(good, bad, fail))
        except:
            traceback.print_exc()
        print('******************')

    def test_set(self, size):
        """ make a set of test codes,
            the test codes consist of the minimum and maximum numbers plus those with the most
            1's and the most 0's and alternating 1's and 0's and N random numbers to make the
            set size up to that given
            """
        max_ones = -1
        max_zeroes = -1
        max_ones_num = None
        max_zeroes_num = None
        alt_ones_num = None
        alt_zeroes_num = None
        num_bits = self.code_bits - 4
        all_bits_mask = (1 << num_bits) - 1
        alt_ones = 0x55555555 & all_bits_mask
        alt_zeroes = 0xAAAAAAAA & all_bits_mask
        for num in range(self.min_num+1, self.max_num):
            code = self.c.encode(num)
            if code == alt_ones:
                alt_ones_num = num
            if code == alt_zeroes:
                alt_zeroes_num = num
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
        num_set = [self.min_num, self.max_num, max_ones_num, max_zeroes_num, alt_ones_num, alt_zeroes_num]
        for _ in range(size-6):
            num_set.append(random.randrange(self.min_num+1, self.max_num-1))
        return num_set

    def code_words(self, numbers):
        """ test code-word rotation with given set plus the extremes (visual) """
        print('')
        print('******************')
        print('Check code-words (visual)')
        bin = '{:0'+str(self.code_bits)+'b}'
        frm_ok = '{}('+bin+')=('+bin+', '+bin+', '+bin+')'
        frm_bad = '{}('+bin+')=(None)'
        try:
            for n in numbers:
                rings = self.c.build(n)
                if rings is None:
                    print(frm_bad.format(n, n))
                else:
                    print(frm_ok.format(n, n, rings[0], rings[1], rings[2]))
        except:
            traceback.print_exc()
        print('******************')

    def circles(self):
        """ test accuracy of co-ordinate conversions - polar to/from cartesian,
            also check polarToCart goes clockwise
            """
        print('')
        print('******************************************')
        print('Check co-ordinate conversions (radius 100)')
        # check clockwise direction by checking sign and relative size as we go round each octant
        #          angle, x-sign, y-sign, xy-sign
        octants = [[ 45, +1, -1, -1],
                   [ 90, +1, -1, +1],
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
            scale = 360 * 10             # 0.1 degrees
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
                    print('{:.3f} degrees --> {:.3f}x, {:.3f}y --> {:.3f} degrees, {:.3f} radius: aerr={:.3f}, rerr={:.3f}, rotation={}'.
                          format(a, cx, cy, ca, cr, aerr, rerr, rotation_err))
                else:
                    good += 1
            print('{} good, {} bad'.format(good, bad))
        except:
            traceback.print_exc()
        print('******************************************')

    def rings(self, width):
        """ draw angle test ring segments (visual) """
        print('')
        print('******************')
        print('Draw an angle test ring (visual)')
        try:
            self.frame.new(width * self.num_rings * 2, width * self.num_rings * 2, max_luminance)
            x, y = self.frame.size()
            ring = Ring(x >> 1, y >> 1, self.code_bits, width, self.frame, self.contrast)
            ring.code(000, [0x5555, 0xAAAA, 0x5555])
            self.frame.unload('{}-segment-angle-test'.format(self.code_bits))
        except:
            traceback.print_exc()
        print('******************')

    def codes(self, numbers, width):
        """ draw test codes for the given test_set """
        print('')
        print('******************')
        print('Draw test codes (visual)')
        try:
            for n in numbers:
                rings = self.c.build(n)
                if rings is None:
                    print('{}: failed to generate the code rings'.format(n))
                else:
                    self.frame.new(width * self.num_rings * 2, width * self.num_rings * 2, max_luminance)
                    x, y = self.frame.size()
                    ring = Ring(x >> 1, y >> 1, self.code_bits, width, self.frame, self.contrast, self.offset)
                    ring.code(n, rings)
                    self.frame.unload('{}-segment-{}'.format(self.code_bits, n))
        except:
            traceback.print_exc()
        print('******************')

    def scan(self, angles, numbers, image, noisy=Scan.DEBUG_NONE):
        """ do a scan for the code set in image and expect the number given,
            returns an exit code to indicate what happened
            """
        print('')
        print('******************')
        print('Scan image {} for codes {}'.format(image, numbers))
        exit_code = self.EXIT_OK         # be optimistic
        try:
            self._remove_derivatives(image)
            self.frame.load(image)
            scan = Scan(self.c, self.frame, angles, debug=noisy)
            results = scan.decode_targets()
            # analyse the results
            found = [False for _ in range(len(numbers))]
            analysis = []
            for result in results:
                centre_x = result[0]
                centre_y = result[1]
                size = result[2]
                num = result[3]
                doubt = result[4]
                size = result[5]
                expected = None
                found_num = None
                for n in range(len(numbers)):
                    if numbers[n] == num:
                        # found another expected number
                        found[n] = True
                        found_num = num
                        expected = '{:b}'.format(self.c.encode(num))
                        break
                analysis.append([found_num, centre_x, centre_y, num, doubt, size, expected])
            # create dummy result for those not found
            for n in range(len(numbers)):
                if not found[n]:
                    # this one is missing
                    num = numbers[n]
                    expected = self.c.encode(num)
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
                        print('Found {} ({}) at {:.0f}x, {:.0f}y size {}, doubt level {}'.
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
                        print('{}Failed to find {} ({})'.format(prefix, num, expected))
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
                            actual_code = self.c.encode(num)
                            if actual_code is None:
                                actual_code = 'not-valid'
                                prefix = ''
                            else:
                                actual_code = '{} ({:b})'.format(num, actual_code)
                                prefix = '**** UNEXPECTED **** ---> '
                        print('{}Found {} ({}) at {:.0f}x, {:.0f}y size {}, doubt level {}'.
                              format(prefix, num, actual_code, centre_x, centre_y, size, doubt))
                        continue
        except:
            traceback.print_exc()
            exit_code = self.EXIT_EXCEPTION
        print('Scan image {} for codes {}'.format(image, numbers))
        print('******************')
        return exit_code

    def _remove_derivatives(self, filename):
        """ remove all the diagnostic image derivatives of the given file name
            a derivative is that file name prefixed by '_', suffixed by anything with any file extension
            """
        filename, _ = os.path.splitext(filename)
        filelist = glob.glob('_{}_*.*'.format(filename))
        for f in filelist:
            try:
                os.remove(f)
            except:
                print('Could not remove {}'.format(f))


# parameters
min_num = 101                            # min number we want
max_num = 999                            # max number we want (may not be achievable)
code_bits = 15                           # number of bits in our code word
parity = None                            # code word parity to apply (None, 0=even, 1=odd)
edges = 4                                # how many bit transitions we want per code word
rings = 9                                # how many rings are in our code
contrast = 1.0                           # reduce dynamic luminance range when drawing to minimise 'bleed' effects
offset = 0.0                             # offset luminance range from the mid-point, -ve=below, +ve=above

test_ring_width = 32
test_black = min_luminance + 64 #+ 32
test_white = max_luminance - 64 #- 32
test_noise = mid_luminance >> 1
test_angle_steps = 256
test_debug_mode = Scan.DEBUG_IMAGE

test = Test(code_bits, min_num, max_num, parity, edges, rings, contrast, offset)
test_num_set = test.test_set(6)

#test.coding()
#test.decoding(test_black, test_white, test_noise)
#test.circles()
#test.code_words(test_num_set)
#test.rings(test_ring_width)
#test.codes(test_num_set, test_ring_width)

test.scan(test_angle_steps, [000], '15-segment-angle-test.png', test_debug_mode)
test.scan(test_angle_steps, [000], 'photo-angle-test-flat.jpg', test_debug_mode)
test.scan(test_angle_steps, [000], 'photo-angle-test-curved-flat.jpg', test_debug_mode)

test.scan(test_angle_steps, [101], '15-segment-101.png', test_debug_mode)
test.scan(test_angle_steps, [102], '15-segment-102.png', test_debug_mode)
test.scan(test_angle_steps, [365], '15-segment-365.png', test_debug_mode)
test.scan(test_angle_steps, [640], '15-segment-640.png', test_debug_mode)
test.scan(test_angle_steps, [658], '15-segment-658.png', test_debug_mode)
test.scan(test_angle_steps, [828], '15-segment-828.png', test_debug_mode)

test.scan(test_angle_steps, [101], 'photo-101.jpg', test_debug_mode)
test.scan(test_angle_steps, [102], 'photo-102.jpg', test_debug_mode)
test.scan(test_angle_steps, [365], 'photo-365.jpg', test_debug_mode)
test.scan(test_angle_steps, [640], 'photo-640.jpg', test_debug_mode)
test.scan(test_angle_steps, [658], 'photo-658.jpg', test_debug_mode)
test.scan(test_angle_steps, [828], 'photo-828.jpg', test_debug_mode)
test.scan(test_angle_steps, [102], 'photo-102-distant.jpg', test_debug_mode)
test.scan(test_angle_steps, [365], 'photo-365-oblique.jpg', test_debug_mode)
test.scan(test_angle_steps, [365], 'photo-365-blurred.jpg', test_debug_mode)
test.scan(test_angle_steps, [658], 'photo-658-small.jpg', test_debug_mode)
test.scan(test_angle_steps, [658], 'photo-658-crumbled-bright.jpg', test_debug_mode)
test.scan(test_angle_steps, [658], 'photo-658-crumbled-dim.jpg', test_debug_mode)
test.scan(test_angle_steps, [658], 'photo-658-crumbled-close.jpg', test_debug_mode)
test.scan(test_angle_steps, [658], 'photo-658-crumbled-blurred.jpg', test_debug_mode)
test.scan(test_angle_steps, [658], 'photo-658-crumbled-dark.jpg', test_debug_mode)
test.scan(test_angle_steps, [101, 102, 365, 640, 658, 828], 'photo-all-test-set.jpg', test_debug_mode)
