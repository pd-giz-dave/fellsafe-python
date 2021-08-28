import os
import glob

import cv2
import numpy as np
import random
import math
import traceback
import copy

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

# ToDo: use AI on flat image? error function is wrong/right number and doubt level
# ToDo: once edge detector is robust, increase the test angle steps
# ToDo: generate some more extreme examples to decode, and drop the old images with the timing ring
# ToDo: don't look for ring edges, just probe for bit edges every 1/3rd nominal width between limits
# ToDo: don't rely on ring boundaries, try every sample combo in decode_targets and pick result with least doubt

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


def vstr(vector):
    """ given an array of numbers return a string representing them """
    result = ''
    for pt in vector:
        if pt is None:
            result += ', None'
        else:
            result += ', {:.2f}'.format(pt)
    return '[' + result[2:] + ']'

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
            returns the number (or None), the level of doubt and the bit classification for each bit,
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

    NUM_RINGS = 9                        # total rings in our complete code

    def __init__(self, centre_x, centre_y, segments, width, frame, contrast, offset):
        # set constant parameters
        self.s = segments          # how many bits in each ring
        self.w = width             # width of each ring in pixels
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

        # setup our angles look-up table such that get 1 pixel resolution on outermost ring
        scale = 2 * math.pi * width * self.NUM_RINGS
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
    """ image frame buffer as a 2D array of luminance values,
        it uses the opencv library to read/write and modify images at the pixel level,
        its the source of images for the Transform functions
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
            return self.buffer[y, x]     # NB: cv2 x, y are reversed

    def putpixel(self, x, y, value, with_alpha=False):
        """ put the pixel of value at x,y
            value may be a greyscale value or a colour tuple
            """
        if x < 0 or x >= self.max_x or y < 0 or y >= self.max_y:
            return
        self.buffer[y, x] = value                      # NB: cv2 x, y are reversed
        if with_alpha:
            if self.alpha is None:
                self.alpha = np.full((self.max_y, self.max_x), transparent,  # default is fully transparent
                                      dtype=np.uint8)  # NB: numpy arrays follow cv2 conventions
            self.alpha[y, x] = opaque                  # want foreground only for our pixels

    def inimage(self, x, y, r):
        """ determine if the points radius R and centred at X, Y are within the image """
        if (x-r) < 0 or (x+r) >= self.max_x or (y-r) < 0 or (y+r) >= self.max_y:
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

        max_x, max_y = source.size()

        def _x(fx):
            if fx is None:
                return None
            return int(round(fx) % max_x)

        def _y(fy):
            if fy is None:
                return None
            return int(round(fy) % max_y)

        def make_int(pt):
            return _x(pt[0]), _y(pt[1])

        def alt_colour(colour):
            return (colour[0] | 128, colour[1] | 128, colour[2] | 128)

        source.incolour()
        image = source.get()

        for obj in objects:
            if obj["type"] == self.LINE:
                start = obj["start"]
                end = obj["end"]
                if start[0] > end[0]:
                    # wraps in x, split it
                    overflow = max_x - start[1]
                    start1 = [0, start[1]]
                    end1 = [overflow, end[1]]
                    end[0] = max_x - 1
                    image = cv2.line(image, make_int(start1), make_int(end1), obj["colour"], 1)
                if start[1] > end[1]:
                    # wraps in y, split it
                    overflow = max_x - start[0]
                    start1 = [start[0], 0]
                    end1 = [end[1], overflow]
                    end[1] = max_y - 1
                    image = cv2.line(image, make_int(start1), make_int(end1), obj["colour"], 1)
                image = cv2.line(image, make_int(start), make_int(end), obj["colour"], 1)
            elif obj["type"] == self.CIRCLE:
                image = cv2.circle(image, make_int(obj["centre"]), int(round(obj["radius"])), obj["colour"], 1)
            elif obj["type"] == self.RECTANGLE:
                image = cv2.rectangle(image, make_int(obj["start"]), make_int(obj["end"]), obj["colour"], 1)
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
                image = cv2.putText(image, obj["text"], make_int(obj["start"]),
                                    cv2.FONT_HERSHEY_SIMPLEX, obj["size"], obj["colour"], 1, cv2.LINE_AA)
            else:
                raise Exception('Unknown object type {}'.format(obj["type"]))
        source.set(image)
        return source

class Target:
    """ struct to hold info about a Scan detected target """
    def __init__(self, centre_x, centre_y, blob_size, number, doubt, target_size):
        self.centre_x    = centre_x
        self.centre_y    = centre_y
        self.blob_size   = blob_size
        self.number      = number
        self.doubt       = doubt
        self.target_size = target_size

class Scan:
    """ scan an image looking for codes """

    # debug options
    DEBUG_NONE = 0             # no debug output
    DEBUG_IMAGE = 1            # just write debug annotated image files
    DEBUG_VERBOSE = 2          # do everything - generates a *lot* of output

    # directions
    TOP_DOWN = 0               # top-down y scan direction
    BOTTOM_UP = 1              # bottom-up y scan direction
    LEFT_TO_RIGHT = 2          # left-to-right x scan direction
    RIGHT_TO_LEFT = 3          # right-to-left x scan direction
    NUM_DIRECTIONS = 4         # for use if creating direction arrays

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

    # better edge reason codes
    THIS_LONGER = 1
    THAT_LONGER = 2
    THAT_THINNER = 3
    THIS_THINNER = 4
    SAME_AS_THAT = 5

    class Edge:
        """ structure to hold detected edge information """

        def __init__(self, position=0, length=0, span=0, start=0, end=0, width=0, where=0):
            self.position = position     # where its midpoint is (an X or Y co-ord)
            self.length = length         # how long it is (in pixels)
            self.span = span             # how wide it is in its bright band (in pixels)
            self.start = start           # where it starts
            self.end = end               # where it ends
            self.width = width           # how wide it is in total (in pixels)
            self.where = where           # where it was found (an X or Y co-ord actually probed)
            self.count = 1               # number of edges joining/overlapping this one (a quality measure)
            self.matches = 0             # what fraction of actual edges match expectations
            self.error = 0               # worst error between actual and ideal edges
            self.ideal = None            # ideal edges starting from position
            self.actual = None           # actual matches starting from position
            self.errors = None           # error between ideal and actual as fraction of nominal width
            self.bad = False             # may be set to True if the edge should be ignored

        def __lt__(self, other):
            """ do an edge comparison for sorting purposes,
                WARNING: the sort will fail if it contains any None entries, even if None could
                         have a meaning here (e.g. treat as very low), this function is not even
                         called when that happens!
                         IMO Python is being brain-dead here, its making an assumption that is not valid
                edges are considered to be ordered by ascending position then start then length
                return True if self is below other
                """
            if self.position < other.position:
                return True
            if self.position > other.position:
                return False
            if self.start < other.start:
                return True
            if self.start > other.start:
                return False
            return self.length < other.length

        def __eq__(self, other):
            """ do edge comparison for sameness purposes,
                to be the same their position, start, length and span must be the same,
                returns True if they are the same
                """
            if int(round(self.position)) != int(round(other.position)):
                return False
            if int(round(self.start)) != int(round(other.start)):
                return False
            if self.length != other.length:
                return False
            return int(round(self.span)) == int(round(other.span))

        def __str__(self):
            return '(at {} as {:.2f} from {:.2f} to {:.2f} for {}, s:{:.2f}, w:{:.2f}, c:{}, m:{:.2f}, e:{:.02f})'.\
                format(self.where, self.position, self.start, self.end, self.length, self.span, self.width,
                       self.count, self.matches, self.error)

    class Target:
        """ structure to hold detected target information """
        def __init__(self,centre_x, centre_y, size, scale, flattened, rings, centres):
            self.centre_x = centre_x           # x co-ord of target in original image
            self.centre_y = centre_y           # y co-ord of target in original image
            self.size = size                   # blob size originally detected by the blob detector
            self.scale = scale                 # scale of target in original image
            self.image = flattened             # the flattened image of the target
            self.rings = rings                 # y co-ord of each ring centre in flattened image
            self.bits = centres                # x co-ord of each bit centre in flattened image

    class Reject:
        """ struct to hold info about rejected targets """
        def __init__(self, centre_x, centre_y, blob_size, target_size, reason):
            self.centre_x    = centre_x
            self.centre_y    = centre_y
            self.blob_size   = blob_size
            self.target_size = target_size
            self.reason      = reason

    class EdgePoint:
        """ struct to hold info about an edge point """

        def __init__(self, midpoint=0, first=0, last=0, bright_first=0, bright_last=0):
            self.midpoint = midpoint
            self.first = first
            self.last = last
            self.bright_first = bright_first
            self.bright_last = bright_last

        def __str__(self):
            return '(m:{}, f:{}, l:{}, bf:{}, bl:{})'.\
                format(self.midpoint, self.first, self.last, self.bright_first, self.bright_last)

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

        # tuning constants
        self.min_blob_separation = 5         # smallest blob within this distance of each other are dropped
        self.blob_radius_stretch = 2.0       # how much to stretch blob radius to ensure always cover the whole lot
        self.min_target_radius = 6 * self.NUM_RINGS  # upsize target radius to at least this (3 pixels per ring)
        self.min_inner_outer_span = (self.NUM_RINGS - 3) * 2   # min tolerable gap between inner and outer edge
        self.min_flat_aspect_ratio = 0.5     # min flattened image height relative to width
        self.brighter_threshold = 0.7  # the threshold at which a bright pixel is considered brighter
        self.sample_width_factor = 0.4       # fraction of a bit width that is probed for a luminance value
        self.sample_height_factor = 0.3      # fraction of a ring height that is probed for a luminance value

        self.ring_edge_drift_limit = 1.0     # max drift in y of ring edge as tan of the angle (1.0==45 degrees)
        self.bit_edge_drift_limit = 0.12     # max drift in x of bit edge as tan of the angle (0.12==7 degrees)

        self.min_ring_edge_length = 0.3      # min length of a ring edge as a fraction of the nominal bit width
        self.min_bit_edge_length = 0.5       # min length of a bit edge as a fraction of the nominal ring width

        self.min_ring_edge_matches = 0.6     # min ring edge matches (as fraction of max) before consider best criteria
        self.min_bit_edge_matches = 0.8      # min bit edge matches (as fraction of max) before consider best criteria

        self.max_ring_edge_error = 0.5       # max error of a ring edge as a fraction of the nominal ring width
        self.max_bit_edge_error = 0.5        # max error of a bit edge as a fraction of the nominal bit width

        # luminance threshold when getting the mean luminance of an image, pixels below this are ignored
        self.min_edge_threshold = int(max_luminance * 0.1)

        # minimum luminance relative to the mean for an edge pixel to qualify as an edge when following it
        self.edge_threshold = 0.8

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

        # samples probed when detecting edge widths (not lengths), see _find_best_neighbour,
        # these co-ords are rotated by 0, 90, 180 or 270 degrees depending on the direction being followed,
        # the reference set here is for left-to-right and is considered as 0 degrees rotation
        # the co-ordinates should be defined in 'best' first order

        # x,y pairs for neighbours when looking for inner/outer radius edges
        # in this context the edges can be very 'wavy' this means they change rapidly vertically
        # so we probe quite a long way 'sideways' to ensure we continue to pick it up
        self.radius_kernel = [[0, 0], [1, 0], [2, 0],    # straight line
                              [0, -1], [0, 1],           # near diagonals
                              [1, -1], [1, 1],           # near neighbours
                              [2, -2], [2, 2],           # far diagonals
                              [0, -2], [0, 2],           # far neighbours
                              [0, -3], [0, 3]]           # distant neighbours
        self.radius_kernel_width = 3         # max offset of the pixels scanned by the radius_kernel

        # x,y pairs for neighbours when looking for ring and bit edges
        # in this context the edges are fairly straight so we do not tolerate a gap
        self.edge_kernel = [[0, 0], [1, 0],              # straight line
                            [0, -1], [0, 1]]             # diagonal
        self.edge_kernel_width = 1           # max offset of the pixels scanned by the edge_kernel

        # edge vector smoothing kernel, pairs of offset and scale factor (see _smooth_edge)
        self.edge_smoothing_kernel = [[-3, 0.5], [-2, 1], [-1, 1.5], [0, 2], [+1, 1.5], [+2, 1], [+3, 0.5]]

        # decoding context (used for logging and image saving)
        self.centre_x = 0
        self.centre_y = 0

        # logging context
        self._log_prefix = '{}: Blob'.format(self.original.source)

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
        self.image = self.transform.downsize(blurred, 1080)    # re-size (to HD video convention)

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
                    if math.fabs(x1 - x2) < self.min_blob_separation \
                    and math.fabs(y1 - y2) < self.min_blob_separation:
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
        max_size = 3                                           # do one for at least 3 pixels
        for blob in blobs:
            if blob.size > max_size:
                max_size = blob.size

        max_radius = (max_size / 4) * self.NUM_RINGS * self.blob_radius_stretch
        max_circumference = min(2 * math.pi * max_radius, 3600)  # good enough for 0.1 degree resolution
        angle = Angle(int(round(max_circumference)))
        self.angle_xy = angle.polarToCart

        if self.show_log:
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
            if the target is very small we upsize it, this helps later when calculating pixel
            addresses (e.g. mid-points can be more accurate),
            returns the projected image and the diameter extracted from the original image
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

        # find our marker edges in the radius
        ring_inner_edge, ring_outer_edge, reason = self._find_extent(target)
        if reason is not None:
            return None, None, reason

        # get the edge limits we need
        max_x, projected_y = target.size()
        # ref point for inner is image edge, i.e. 0
        max_inner_edge = 0
        # ref point for outer is the corresponding inner
        max_outer_edge = 0
        max_outer_inner_edge = 0         # inner edge at max outer
        max_inner_outer_distance = 0
        min_inner_outer_distance = projected_y
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
            if distance < min_inner_outer_distance:
                min_inner_outer_distance = distance
            if distance > max_outer_inner_edge:
                max_inner_outer_distance = distance
                max_outer_edge = outer_edge
                max_outer_inner_edge = inner_edge

        # chuck it if its too small (it'd just time wasting to process it further)
        if min_inner_outer_distance < self.min_inner_outer_span:
            # we're looking at junk
            return None, None, 'inner to outer edge too small {}, limit {}'.\
                                format(min_inner_outer_distance, self.min_inner_outer_span)

        if self.show_log:
            self._log('inner (0,{}), outer ({},{})'.
                      format(max_inner_edge, max_outer_inner_edge, max_outer_edge))

        stretched_size = self._get_estimated_ring_sizes(0, max_inner_edge,
                                                        max_outer_inner_edge, max_outer_edge,
                                                        stretch=True)

        if self.show_log:
            self._log('stretched_size with ({}, {}) is {}'.
                      format(max_outer_inner_edge, max_outer_edge, stretched_size))

        # create a new image to flatten into
        flat_y = int(round(sum(stretched_size)))
        code = self.original.instance().new(max_x, flat_y, mid_luminance)

        # build flat image
        truncated_y = 0
        for x in range(max_x):

            actual_size = self._get_estimated_ring_sizes(0, ring_inner_edge[x],
                                                         ring_inner_edge[x], ring_outer_edge[x])
            if self.show_log:
                self._log('actual_size at {}x with ({:.2f}, {:.2f}) is {}'.
                          format(x, ring_inner_edge[x], ring_outer_edge[x], vstr(actual_size)))

            in_y = 0
            out_y = 0
            in_ring_start = 0
            out_ring_start = 0
            for ring in range(self.NUM_RINGS):
                # change each ring from its size now to its stretched size
                in_ring_end = in_ring_start + actual_size[ring]        # these may be fractional
                in_pixels = [None for _ in range(int(round(in_ring_end - in_y)))]
                for dy in range(len(in_pixels)):
                    in_pixels[dy] = target.getpixel(x, in_y)
                    in_y += 1
                out_ring_end = out_ring_start + stretched_size[ring]   # these may be fractional
                out_pixels = self._stretch(in_pixels, int(round(out_ring_end - out_y)))
                # the out ring sizes have been arranged to be whole pixel widths, so no fancy footwork here
                for dy in range(len(out_pixels)):
                    pixel = out_pixels[dy]
                    code.putpixel(x, out_y, pixel)
                    out_y += 1
                if self.show_log:
                    self._log('    ring {}, in_y {}, out_y {}: in_pixels {:.2f}..{:.2f}, out_pixels {:.2f}..{:.2f}'.
                              format(ring, in_y - len(in_pixels), out_y - len(out_pixels),
                                     in_ring_start, in_ring_end, out_ring_start, out_ring_end))
                in_ring_start += actual_size[ring]
                out_ring_start += stretched_size[ring]
            if in_y > truncated_y:
                truncated_y = in_y
            continue                     # statement here as a debug hook

        # upsize such that meet required aspect ratio
        new_y = int(round(max_x * self.min_flat_aspect_ratio))
        code = self.transform.upsize(code, new_y)

        _, target_y = code.size()

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

        # return flattened image
        return code, scale, None

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

    def _get_within_threshold(self, target, x, y, direction, threshold):
        """ get a list of pixels from x,y in the given direction that are over the given threshold,
            x is wrapped at image edge, y is not, if given an excessive y None is returned,
            direction specifies which way the edge should be probed:
                TOP_DOWN = x static, y increasing
                BOTTOM_UP = x static, y decreasing
                LEFT_TO_RIGHT = x increasing, y static
                RIGHT_TO_LEFT = x decreasing, y static
            threshold is used to find the extremes of the edge,
            returns a list of pixels or an empty list if given x,y is not over the threshold
            """

        max_x, max_y = target.size()
        x = int(round(x % max_x))
        y = int(round(y))
        if y >= max_y:
            return []

        if direction == self.TOP_DOWN:
            dx = 0
            dy = +1
        elif direction == self.BOTTOM_UP:
            dx = 0
            dy = -1
        elif direction == self.LEFT_TO_RIGHT:
            dx = +1
            dy = 0
        elif direction == self.RIGHT_TO_LEFT:
            dx = -1
            dy = 0
        else:
            raise Exception('illegal direction {}'.format(direction))

        pixels = []
        while True:
            pixel = target.getpixel(x, y)            # NB: returns None when reach image edge
            if pixel is None or pixel < threshold:
                break
            pixels.append(pixel)
            x += dx
            y += dy

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
                if pixel * self.brighter_threshold > brightest:
                    # its changed by enough to justify pushing the threshold up
                    brightest = pixel * self.brighter_threshold
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

    def _find_width_midpoint(self, target, x, y, direction, threshold):
        """ from x,y in the target image find the width midpoint x or y that is over the given threshold,
            the pixel at x,y must be over else None is returned,
            this function is used when following edges, the direction given here is the edge direction
            being followed, so width mid-points of interest are at 90 degrees to that,
            direction is TOP_DOWN or BOTTOM_UP if looking for an x midpoint,
            direction is LEFT_TO_RIGHT or RIGHT_TO_LEFT if looking for a y midpoint,
            returns a fully populated EdgePoint instance or None if no midpoint
            """

        max_x, max_y = target.size()

        if direction == self.LEFT_TO_RIGHT or direction == self.RIGHT_TO_LEFT:
            up = self.BOTTOM_UP
            down = self.TOP_DOWN
            direction_coord = 1
            max_coord = max_y
        elif direction == self.TOP_DOWN or direction == self.BOTTOM_UP:
            up = self.RIGHT_TO_LEFT
            down = self.LEFT_TO_RIGHT
            direction_coord = 0
            max_coord = max_x
        else:
            raise Exception('illegal direction {}'.format(direction))

        xy = [x, y]

        pixels_up = self._get_within_threshold(target, xy[0], xy[1], up, threshold)
        if len(pixels_up) == 0:
            return None

        # note how far up or right we went
        first = xy[direction_coord]
        if len(pixels_up) > 0:
            first -= (len(pixels_up) - 1)
        if first < 0:
            # we wrapped
            first += max_coord

        xy[direction_coord] += 1         # +1 always correct for the 'down' part
        pixels_down = self._get_within_threshold(target, xy[0], xy[1], down, threshold)

        # note how far down or left we went
        last = xy[direction_coord]
        if len(pixels_down) > 0:
            last += (len(pixels_down) - 1)
        if last > max_coord:
            # we wrapped
            last -= max_coord

        # join the pixel lists together, up is backwards, down is forwards
        pixels_up.reverse()
        pixels = pixels_up + pixels_down

        # find their midpoint
        midpoint, bright_first, bright_last = self._get_midpoint(pixels)
        if midpoint is None:
            return None

        bright_first += first
        bright_last += first

        return self.EdgePoint(first + midpoint,
                              min(first, last), max(first, last),
                              min(bright_first, bright_last), max(bright_first, bright_last))

    def _find_best_neighbour(self, target, x, y, direction, threshold, kernel):
        """ find best neighbour in target from x,y in given direction,
            kernel is the matrix to use to determine that a neighbour is 'connected',
            returns a fully populated EdgePoint instance,
            returns None if there is not one (ie. no neighbours are connected),
            direction affects kernel orientation,
            it finds the best ongoing x or y co-ordinate to follow,
            the 'best' x or y is the brightest between the first and last y that is over our threshold
            """

        for dx, dy in self.Kernel(kernel, direction):
            best = self._find_width_midpoint(target, x + dx, y + dy, direction, threshold)
            if best is not None:
                return best

        return None

    def _follow_edge(self, target, x, y, direction, threshold, kernel, drift_limit=None, context='anon'):
        """ follow the edge at x,y until come to its end or its span becomes too large,
            direction is left-to-right or right-to-left for following edges in x (rings),
            direction is top_down or bottom_up for following edges in y (bits),
            threshold is the threshold to apply to determine if a pixel is a candidate for an edge,
            kernel specifies the matrix to use to detect connected neighbours,
            drift_limit is the maximum sideways drift of the edge tolerated or None if no limit,
            drift_limit is specified as a ratio of the length and is effectively an angle (an approximate tan),
            no limit is 90 degrees, a limit of 1 is 45 degrees, a limit of 0.2 is 11 degrees,
            its followed for at most for the image width/height or until its drift limit is reached,
            context is purely for diagnostic messages,
            x co-ordinates wrap, y does not,
            return a vector of its co-ords, or an empty vector if none, and the min/max co-ord probed,
            the pixel at x,y must be over else an empty vector is returned
            """

        best = self._find_width_midpoint(target, x, y, direction, threshold)
        if best is None:
            return []

        max_x, max_y = target.size()

        if direction == self.TOP_DOWN:
            # bits forwards
            x_inc = 0
            y_inc = +1
            probe_range = max_y - 1
            best_coord = 0
            scan_coord = 1
            if drift_limit is None:
                drift_limit = max_x
        elif direction == self.BOTTOM_UP:
            # bits backwards
            x_inc = 0
            y_inc = -1
            probe_range = max_y - 1
            best_coord = 0
            scan_coord = 1
            if drift_limit is None:
                drift_limit = max_x
        elif direction == self.LEFT_TO_RIGHT:
            # rings forward
            x_inc = +1
            y_inc = 0
            probe_range = max_x - 1
            best_coord = 1
            scan_coord = 0
            if drift_limit is None:
                drift_limit = max_y
        elif direction == self.RIGHT_TO_LEFT:
            # rings backwards
            x_inc = -1
            y_inc = 0
            probe_range = max_x - 1
            best_coord = 1
            scan_coord = 0
            if drift_limit is None:
                drift_limit = max_y
        else:
            raise Exception('illegal direction: {}'.format(direction))

        edge = [best]
        best_xy = [0, 0]
        min_best = best.midpoint
        max_best = best.midpoint
        xy = [x, y]
        for _ in range(probe_range):
            xy[0] = (xy[0] + x_inc) % max_x
            xy[1] += y_inc
            xy[best_coord] = best.midpoint
            best_xy[best_coord] = xy[best_coord]
            best_xy[scan_coord] = xy[scan_coord]
            best = self._find_best_neighbour(target, best_xy[0], best_xy[1], direction, threshold, kernel)
            if best is None:
                # no more qualifying neighbors
                break
            if best.midpoint < min_best:
                min_best = best.midpoint
            if best.midpoint > max_best:
                max_best = best.midpoint
            drift = math.fabs(min_best - max_best)
            if drift > 1 and (drift / len(edge)) > drift_limit:
                # we've drifted too far sideways, so chop it here
                if self.show_log:
                    self._log('{}: edge drifted too far ({:.2f}..{:.2f} across {}={:.2f}), limit is {:.2f}'.
                              format(context, min_best, max_best, len(edge), drift / len(edge), drift_limit))
                break
            # it qualifies
            edge.append(best)

        return edge

    def _find_radius_edge(self, target, direction, threshold, context):
        """ look for a continuous edge in the given target either top-down (inner) or bottom-up (outer),
            direction is top-down or bottom-up, an 'edge' is vertical, i.e along the radius,
            context is purely for debug log messages,
            to qualify as an 'edge' it must be continuous across all angles (i.e. no dis-connected jumps)
            to find the edge we scan the radius in both directions, if we get full 360 without a break we've found it
            the returned edge is smoothed
            """
        max_x, max_y = target.size()
        if direction == self.TOP_DOWN:
            y = -1
            increment = +1
        elif direction == self.BOTTOM_UP:
            y = max_y
            increment = -1
        else:
            raise Exception('illegal direction {}'.format(direction))
        for _ in range(max_y):
            y += increment
            edge = self._follow_edge(target, 0, y, self.LEFT_TO_RIGHT, threshold, self.radius_kernel, context=context)
            if len(edge) == 0:
                # nothing here, move on
                # NB: if nothing going forwards there will also be nothing going backwards
                continue
            if len(edge) == max_x:
                # found an edge that goes all the way, so that's it
                if self.show_log:
                    self._log('{}: radius edge length: {} from 0,{}'.format(context, len(edge), y))
                edge_points = [edge[x].midpoint for x in range(max_x)]
                return self._smooth_edge(edge_points, True)

            # this one too short, see if can extend it by going the other way
            edge_extension = self._follow_edge(target, 0, y, self.RIGHT_TO_LEFT, threshold, self.radius_kernel, context=context)
            if len(edge_extension) < 2:
                # NB: we expect at least 1 here as it overlaps the initial edge, so to be an extension we need 2+
                if self.show_log:
                    self._log('{}: radius edge length: {} from 0,{}'.format(context, len(edge), y))
                # nothing more here, move y past this duff edge
                if increment > 0:
                    # going top-down
                    y = edge[0].last + 1       # we know this is below our threshold
                else:
                    # going bottom-up
                    y = edge[0].first - 1      # we know this is below our threshold
                continue
            if self.show_log:
                self._log('{}: radius edge length: {} from 0,{} (as {} backwards + {} forwards)'.
                          format(context, len(edge) + len(edge_extension), y, len(edge_extension), len(edge)))
            if len(edge) + len(edge_extension) + self.radius_kernel_width > max_x:
                # gone right round now, merge the two edges
                # they may overlap, in which case we use the median (NB: the first sample is the same in both)
                # we interpolate across a small ending gap (it will be at most radius_kernel_width)
                edge_combined = [None for _ in range(max_x)]  # copy our edge into this
                for x in range(max_x):
                    if x >= len(edge):
                        break
                    edge_combined[x] = edge[x].midpoint
                dropped = False
                for x in range(1, len(edge_extension)):        # skip first sample, we know that's the same
                    slot = max_x - x                           # where this goes in edge
                    if edge_combined[slot] is None:
                        edge_combined[slot] = edge_extension[x].midpoint
                    else:
                        # got an overlap, if they deviate too much abandon it
                        deviation = math.fabs(edge_combined[slot] - edge_extension[x].midpoint)
                        if deviation > self.radius_kernel_width:
                            # they've diverged too far to join up, drop it
                            if self.show_log:
                                self._log('edge[{}] diverges too far from edge_extension[{}], limit is {} - dropping'.
                                          format(edge_combined[slot], edge_extension[x].midpoint,
                                                 self.radius_kernel_width))
                            dropped = True
                            break
                        # they are close enough to combine
                        edge_combined[slot] = min(edge_combined[slot] + (deviation / 2), max_y - 1)
                if dropped:
                    continue
                # any None's left at the join need interpolating across
                if len(edge) + len(edge_extension) > max_x:    # NB: edge and edge_extension overlap by 1
                    # no gap, so that's it
                    return self._smooth_edge(edge_combined)
                # gap is at len(edge)..(max_x-len(edge_extension))
                gap = (max_x - len(edge_extension)) - len(edge) + 1  # +1 'cos edge_extension and edge overlap
                start_y = edge_combined[len(edge) - 1]
                stop_y = edge_combined[max_x - len(edge_extension) + 1]
                delta_y = (start_y - stop_y) / gap
                next_y = start_y
                for x in range(len(edge), max_x - len(edge_extension)):
                    next_y += delta_y
                    if edge_combined[x] is None:
                        edge_combined[x] = next_y
                    else:
                        # eh? got me sums wrong sumplace
                        raise Exception('got me sums wrong sumplace')
                # all done now
                return self._smooth_edge(edge_combined)

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
            returns an image of the edges and a threshold to detect edges
            """

        edges = self.transform.edges(target, x_order, y_order, 3, inverted)
        image_mean = self._get_edge_threshold(edges)
        threshold = max(min(int(round(image_mean * self.edge_threshold)), max_luminance), 1)

        return edges, threshold

    def _find_extent(self, target):
        """ find the inner and outer edges of the given target,
            the inner edge is the first white to black transition that goes all the way around,
            the outer edge is the first black to white transition that goes all the way around,
            returns two vectors, y co-ord for every angle of the edges, or a reason if one or both not found,
            we look for the outer first as that will fail when looking at junk whereas the inner may not
            """

        # look for the outer edge
        b2w_edges, threshold = self._get_transitions(target, 0, 1, False)
        ring_outer_edge = self._find_radius_edge(b2w_edges, self.BOTTOM_UP, threshold, 'radius-outer')

        if ring_outer_edge is None:
            # don't bother looking for the inner if there is no outer
            ring_inner_edge = None
            w2b_edges = None
        else:
            # found an outer, so worth looking for an inner
            w2b_edges, threshold = self._get_transitions(target, 0, 1, True)
            ring_inner_edge = self._find_radius_edge(w2b_edges, self.TOP_DOWN, threshold, 'radius-inner')

        if ring_outer_edge is None:
            reason = 'no outer edge'
        elif ring_inner_edge is None:
            reason = 'no inner edge'
        else:
            reason = None

        if self.save_images:
            if w2b_edges is not None:
                plot = self._draw_below(w2b_edges, threshold, (0, 0, 255))
                if ring_inner_edge is not None:
                    points = [[0, ring_inner_edge]]
                else:
                    points = None
                plot = self._draw_plots(plot, points, None, (0, 255, 0))
                self._unload(plot, 'inner')

            if b2w_edges is not None:
                plot = self._draw_below(b2w_edges, threshold, (0, 0, 255))
                if ring_outer_edge is not None:
                    points = [[0, ring_outer_edge]]
                else:
                    points = None
                plot = self._draw_plots(plot, points, None, (0, 0, 255))
                self._unload(plot, 'outer')

            plot = target
            if ring_inner_edge is not None:
                plot = self._draw_plots(plot, [[0, ring_inner_edge]], None, (0, 255, 0))
            if ring_outer_edge is not None:
                plot = self._draw_plots(plot, [[0, ring_outer_edge]], None, (0, 0, 255))
            self._unload(plot, 'wavy')

        return ring_inner_edge, ring_outer_edge, reason

    def _stretch(self, pixels, size):
        """ given a vector of pixels stretch them such that the resulting pixels is size long,
            consecutive pixel values are interpolated as necessary when stretched into the new size,
            this is a helper function for _flatten, in that context we are stretching pixels in the
            'y' direction, the names used here reflect that but this logic is totally generic
            """

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
                pixel_delta = (int(end_pixel) - int(start_pixel)) / (span + 1)   # NB: use int() to stop numpy minus
                gap_pixel = start_pixel
                for dy in range(span):
                    gap_pixel += pixel_delta
                    dest[start_gap + dy] = np.uint8(max(min(round(gap_pixel), max_luminance), min_luminance))
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
        ring_span = self.NUM_RINGS - 3
        outer_ring_size = (outer_end - outer_start) / ring_span  # average outer ring size
        inner_ring_size = (inner_end - inner_start) / 2          # average inner ring size

        ring_sizes = [None for _ in range(self.NUM_RINGS)]

        if stretch:
            # we want all rings the same size here and as big as the max
            outer_ring_size = max(outer_ring_size, inner_ring_size)
            inner_ring_size = outer_ring_size

        ring_sizes[self.DODGY_WHITE] = inner_ring_size
        ring_sizes[self.INNER_WHITE] = inner_ring_size
        # inner edge is here
        ring_sizes[self.INNER_BLACK] = outer_ring_size
        ring_sizes[self.DATA_RING_1] = outer_ring_size
        ring_sizes[self.DATA_RING_2] = outer_ring_size
        ring_sizes[self.DATA_RING_3] = outer_ring_size
        ring_sizes[self.DODGY_BLACK] = outer_ring_size
        ring_sizes[self.OUTER_BLACK] = outer_ring_size
        # outer edge is here
        ring_sizes[self.OUTER_WHITE] = outer_ring_size

        return ring_sizes

    def _measure_edge(self, scan_coord, down_edge, up_edge, probe_coord, max_probe):
        """ this is a helper function for _get_edges,
            it analyses the given edge co-ordinates and returns an instance of Edge that characterises it,
            the down_edge and up_edge are assumed to be the results of scans going 'backwards' and 'forwards'
            from the given scan_coord,
            probe_coord is the 'other' coordinate and max_probe is its limit,
            these are used to determine the edge start/end limits with due regard for wrapping,
            down_edge or up_edge may be empty
            """

        if len(up_edge) == 0:
            edge_length = len(down_edge)
            edge_start = probe_coord
        elif len(down_edge) == 0:
            edge_length = len(up_edge)
            edge_start = probe_coord - (len(up_edge) - 1)
        else:
            edge_length = len(up_edge) + len(down_edge)
            edge_start = probe_coord - (len(up_edge) - 1)

        if edge_length >= max_probe:
            # limit length and move start to the beginning (so all full length edges start ar the same place)
            edge_length = max_probe
            edge_start = 0

        max_bright_edge = 0
        min_bright_edge = 31 * 1024            # arbitrary large number bigger than any legit co-ord
        max_edge = 0
        min_edge = 31 * 1024
        midpoints = 0
        samples = 0
        edges = up_edge + down_edge
        for edge in edges:
            midpoints += edge.midpoint
            if edge.bright_first < min_bright_edge:
                min_bright_edge = edge.bright_first
            if edge.bright_last > max_bright_edge:
                max_bright_edge = edge.bright_last
            if edge.first < min_edge:
                min_edge = edge.first
            if edge.last > max_edge:
                max_edge = edge.last
            samples += 1

        midpoint = midpoints / samples
        edge_bright_span = max_bright_edge - min_bright_edge + 1
        edge_span = max_edge - min_edge + 1

        if edge_start < 0:
            # start wraps
            edge_start += max_probe
        edge_end = edge_start + edge_length - 1
        if edge_end >= max_probe:
            # end wraps
            edge_end -= max_probe

        return self.Edge(midpoint, edge_length, edge_bright_span, edge_start, edge_end, edge_span, scan_coord)

    def _get_edges(self, target, direction, centres, threshold):
        """ return a vector of the edges in the given target within the centres given,
            direction is top-down or bottom-up if looking for ring edges,
            direction is left-to-right or right-to-left if looking for bit edges,
            centres provides a list of co-ordinates that should be probed,
            the target must consist of either white-to-black edges or black-to-white edges or both,
            each edge in the returned vector consists of an Edge instance,
            the result vector is sorted by position then start then length,
            returns the result and an image with detections drawn on it (None if not in debug mode),
            NB: ring edges wrap and bit edges do not
            """

        max_x, max_y = target.size()

        if direction == self.TOP_DOWN or direction == self.BOTTOM_UP:
            # looking for ring edges
            nominal_width = max_x / self.size
            max_length = max_x
            min_length = int(max(nominal_width * self.min_ring_edge_length, 2))
            drift_limit = self.ring_edge_drift_limit
            down = self.LEFT_TO_RIGHT
            up = self.RIGHT_TO_LEFT
            scan_coord = 1
            probe_coord = 0
            min_scan_coord = 0
            max_scan_coord = max_y
            max_probe_coord = max_x
            if direction == self.TOP_DOWN:
                context = 'ring-down'
                start_scan_coord = min_scan_coord
                scan_direction = +1
            else:
                context = 'ring-up'
                start_scan_coord = max_scan_coord - 1
                scan_direction = -1
        elif direction == self.LEFT_TO_RIGHT or direction == self.RIGHT_TO_LEFT:
            # looking for bit edges
            nominal_width = max_y / self.NUM_RINGS
            max_length = max_y
            min_length = int(max(nominal_width * self.min_bit_edge_length, 2))
            drift_limit = self.bit_edge_drift_limit
            down = self.TOP_DOWN
            up = self.BOTTOM_UP
            scan_coord = 0
            probe_coord = 1
            min_scan_coord = 0
            max_scan_coord = max_x
            max_probe_coord = max_y
            if direction == self.LEFT_TO_RIGHT:
                context = 'bit-left'
                start_scan_coord = min_scan_coord
                scan_direction = +1
            else:
                context = 'bit-right'
                start_scan_coord = max_scan_coord - 1
                scan_direction = -1
        else:
            raise Exception('illegal direction {}'.format(direction))

        if self.show_log:
            self._log('{}: probe centres are: {}'.format(context, centres))
        if self.save_images:
            # highlight pixels that are below our threshold
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

        # ToDo: implement ignoring pixels already visited? will it eat too much?

        edges = []
        xy = [0, 0]
        for probe in centres:
            scan_inc = 1
            xy[scan_coord] = start_scan_coord - (scan_direction * scan_inc) # - to cancel upcoming +
            while True:
                xy[scan_coord] += int((scan_direction * scan_inc))
                if xy[scan_coord] >= max_scan_coord or xy[scan_coord] < min_scan_coord:
                    break
                scan_inc = 1
                xy[probe_coord] = probe
                px = xy[0]
                py = xy[1]
                down_edge = self._follow_edge(target, px, py, down, threshold, self.edge_kernel, drift_limit, context)
                if len(down_edge) == 0:
                    # found nothing going down, this means we'll also find nothing going up
                    # 'cos they overlap in the initial pixel address
                    continue
                elif len(down_edge) < max_length:
                    up_edge = self._follow_edge(target, px, py, up, threshold, self.edge_kernel, drift_limit, context + '-back')
                    if len(up_edge) == 0:
                        raise Exception('no overlap pixel!')
                    elif not self._position_same(up_edge[0].midpoint, down_edge[0].midpoint, 0):
                        raise Exception('overlap pixel not same position')
                    if len(up_edge) < 2:
                        # nothing going the other way
                        # NB: up and down overlap at px,py so get an extension we need 2+
                        up_edge = []
                    else:
                        # remove the overlap pixel
                        up_edge.pop(0)
                        # to qualify as an extension the now up_edge top pixel must be within the
                        # edge_kernel_width of the down_edge top pixel
                        if self._position_same(up_edge[0].midpoint, down_edge[0].midpoint, self.edge_kernel_width-1):
                            # its connected, so OK
                            pass
                        else:
                            # the up does not join the down, so ignore it
                            if self.show_log:
                                self._log('{}: ignoring non-connected up edge at {},{} (up-len:{}, down-len:{}) gap is {:.2f}, limit is {}'.
                                          format(context,
                                                 px, py, len(up_edge), len(down_edge),
                                                 math.fabs(up_edge[0].midpoint - down_edge[0].midpoint),
                                                 self.edge_kernel_width-1))
                            up_edge = []
                else:
                    # gone right round, so don't bother going the other way
                    up_edge = []
                edge = self._measure_edge(xy[scan_coord], down_edge, up_edge, xy[probe_coord], max_probe_coord)
                if edge.length < min_length:
                    # too small ignore it
                    if self.show_log:
                        self._log('{}: ignoring short edge of {} at {},{} (up-len:{}, down-len:{}), limit is {}'.
                                  format(context, edge, px, py, len(up_edge), len(down_edge), min_length))
                    # move scan past this edge
                    scan_inc = max(edge.width / 2, 1)
                    continue
                if edge.length > max_length:
                    # we've wrapped, truncate so its not too long
                    if self.show_log:
                        self._log('{}: edge greater than image dimension! Got {} at {},{} (up-len:{}, down-len:{}) when limit is {}, truncating'.
                                  format(context, edge, px, py, len(up_edge), len(down_edge), max_length))
                    if len(up_edge) < max_length:
                        up_edge = up_edge[:-(edge.length - max_length + 1)]
                        edge.length = len(down_edge) + len(up_edge)
                    else:
                        # going the other way went right round!
                        down_edge = []         # drop the short one
                if len(edges) > 0 and edge == edges[-1]:
                    # got a duplicates (this should not happen!)
                    if self.show_log:
                        self._log('{}: ignoring duplicate edge of {} at {},{} (up-len:{}, down-len:{})'.
                                  format(context, edge, px, py, len(up_edge), len(down_edge)))
                    # move scan past this edge
                    scan_inc = max(edge.width / 2, 1)
                    continue
                # the edge qualifies, add it to our list
                edges.append(edge)
                if self.show_log:
                    self._log('{}: adding edge {} at {},{} (up-len:{}, down-len:{})'.
                              format(context, edges[-1], px, py, len(up_edge), len(down_edge)))
                if self.save_images:
                    # draw the edge we detected in green and its ends in yellow
                    down_plots = []
                    for plot in range(len(down_edge)):
                        down_plots.append(down_edge[plot].midpoint)
                    up_plots = []
                    for plot in range(len(up_edge)-1, -1, -1):
                        up_plots.append(up_edge[plot].midpoint)
                    xy_plots = [None, None]
                    xy_plots[probe_coord] = [[xy[probe_coord], down_plots],
                                             [xy[probe_coord] - len(up_edge), up_plots]]
                    grid = self._draw_plots(grid, xy_plots[0], xy_plots[1], (0, 255, 0))
                    if len(up_edge) > 0:
                        up_start = xy[probe_coord] - len(up_edge)
                        up_val = up_plots[0]
                    else:
                        up_start = xy[probe_coord]
                        up_val = down_plots[0]
                    if len(down_edge) > 0:
                        down_end = xy[probe_coord] + len(down_edge) - 1
                        down_val = down_plots[-1]
                    else:
                        down_end = xy[probe_coord]
                        down_val = up_plots[-1]
                    if edge.length >= max_length:
                        # normalise to 0..max-1 else display looks odd
                        up_start = 0
                        down_end = max_length - 1
                    xy_plots = [None, None]
                    xy_plots[probe_coord] = [[up_start, [up_val]],
                                             [down_end, [down_val]]]
                grid = self._draw_plots(grid, xy_plots[0], xy_plots[1], (0, 255, 255))
                scan_inc = max(edge.width / 2, 1)

        if len(edges) > 0:

            # sort by position+start+length (see Edge.__lt__ class function)
            edges.sort()

            # drop duplicates (this happens when an edge spans more than one probe centre)
            unique_edges = [edges.pop(0)]
            while len(edges) > 0:
                edge = edges.pop(0)
                if edge == unique_edges[-1]:
                    if self.show_log:
                        self._log('{}: dropping duplicate edge {} (duplicate of {})'.
                                  format(context, edge, unique_edges[-1]))
                else:
                    unique_edges.append(edge)

            # set the unique set
            edges = unique_edges

        return edges, grid

    def _get_edge_overlap(self, this_edge, that_edge, length_limit):
        """ given two edges return how much they overlap each other and the combined start/end,
            this is a helper function for _merge_edge, in that context the edge co-ordinates
            cannot be relied upon for determining length, the length attribute must be used for that,
            the overlap returned is the actual edge overlap not the extent overlap,
            the extent is the start to end range which is different from the start+length range
            """

        this_start, this_end, that_start, that_end = self._remap_edge_pair(this_edge, that_edge, length_limit)

        # determine the actual edge ends, this is start + length, these may also wrap
        this_actual_end = this_start + this_edge.length - 1
        that_actual_end = that_start + that_edge.length - 1

        if (this_actual_end + 1) >= that_start and this_start <= that_actual_end:
            # they abutt or overlap
            joined_start = min(this_start, that_start)         # extent start and actual start always coincide
            joined_end = max(this_end, that_end)               # NB: this is the extent end, not the actual end
            # the joined length is the distance between the actual start and actual end
            actual_end = max(this_actual_end, that_actual_end)
            joined_length = actual_end - joined_start + 1
            if joined_length > length_limit:
                joined_length = length_limit
            if joined_length == length_limit:
                joined_start = 0
                joined_end = length_limit - 1
            overlap = (this_edge.length + that_edge.length) - joined_length
            return overlap, joined_start, joined_end
        else:
            # they do not overlap
            return 0, None, None

    def _position_same(self, this_position, that_position, limit=1):
        """ given two positions, determine if they are the same within the given limit,
            returns True if they are,
            to be the same the pixel gap between them must not be more than 1
            """

        gap = math.fabs(this_position - that_position)

        if gap > limit:
            # they are not the same position
            return False
        else:
            return True

    def _merge_edge(self, this_edge, that_edge, length_limit):
        """ given two edges (this and that) merge them if they have the same position and abutt or overlap,
            returns the merged edge if they overlap or None if they don't,
            this is a helper function for _get_combined_edges
            """

        # if we get coincident edges combine them into a single longer edge, this can happen when a
        # black-to-white edge is immediately followed by a white-to-black edge, an edge is coincident
        # if their positions are within 1 pixel and their length extents abutt or overlap.

        if not self._position_same(this_edge.position, that_edge.position, 1):
            return None

        overlap, joined_start, joined_end = self._get_edge_overlap(this_edge, that_edge, length_limit)
        if joined_start is None:
            # they do not overlap
            return None

        # they abutt and overlap in extent, join them
        joined_edge = self.Edge()

        joined_edge.position = min(this_edge.position, that_edge.position)  # NB: Do not do a mean, it causes edge drift

        # keep the span of the longest edge or the biggest if they are the same length
        if this_edge.length > that_edge.length:
            joined_edge.span = this_edge.span
        elif this_edge.length < that_edge.length:
            joined_edge.span = that_edge.span
        else:
            joined_edge.span = max(this_edge.span, that_edge.span)

        # number of segments is the sum of those being joined minus one (the join itself is considered a single segment)
        joined_edge.count = this_edge.count + that_edge.count - 1

        if int(round(joined_end)) == int(round(joined_start)):
            # ToDo: this is not possible?
            raise Exception('joined_end == joined_start for {} and {}'.format(this_edge, that_edge))
        elif overlap == 0:
            # this means they overlap in extent but not actually, we increase the extent by the
            # increase in the end co-ordinates between this and that
            joined_edge.length = this_edge.length + that_edge.length
        else:
            joined_edge.length = this_edge.length + that_edge.length - overlap
        if joined_edge.length > length_limit:
            # limit length
            joined_edge.length = length_limit

        if joined_edge.length == length_limit:
            # normalise start/end
            joined_start = 0
            joined_end = length_limit - 1

        if joined_start >= length_limit:
            joined_edge.start = joined_start - length_limit
        else:
            joined_edge.start = joined_start

        if joined_end >= length_limit:
            joined_edge.end = joined_end - length_limit
        else:
            joined_edge.end = joined_end

        # return joined edges
        return joined_edge

    def _remap_edge_pair(self, this_edge, that_edge, length_limit):
        """ given a pair of edges re-map their start/end co-ordinates to remove wrapping issues,
            this is done by doubling the range so that wrapped ends move into the extended range,
            the returned co-ordinates can then be used in compares to detect overlaps, et al,
            without having to deal with edge wrapping issues,
            returns a tuple of (this start, this end, that start, that end)
            """

        this_start = math.floor(this_edge.start)
        this_end = math.ceil(this_edge.end)
        that_start = math.floor(that_edge.start)
        that_end = math.ceil(that_edge.end)

        if this_end < this_start and that_end < that_start:
            # both wrap, just extend the ends, leave starts as is
            this_end += length_limit
            that_end += length_limit
        elif this_end < this_start:
            # just this wraps, extend this, re-locate that
            this_end += length_limit
            that_start += length_limit
            that_end += length_limit
        elif that_end < that_start:
            # just that wraps, extend that, re-locate this
            this_start += length_limit
            this_end += length_limit
            that_end += length_limit
        else:
            # neither wrap, re-locate both
            this_start += length_limit
            this_end += length_limit
            that_start += length_limit
            that_end += length_limit

        return this_start, this_end, that_start, that_end

    def _get_end_extension(self, this_edge, that_edge, length_limit):
        """ given two edges determine the end extension from a consideration of their end coordinates,
            this should be called where an extent is being expanded due to their ends growing,
            extension is how much the ends are growing, with due regard for wrapping, it may be 0,
            returns the extension
            """

        this_start, this_end, that_start, that_end = self._remap_edge_pair(this_edge, that_edge, length_limit)

        if that_end > this_end:
            # the end is growing
            extension = that_end - this_end
        else:
            # the end is not growing
            extension = 0

        return extension

    def _extend_edge(self, this_edge, that_edge, length_limit):
        """ given two edges update the extent of this to include the extent of that,
            length_limit restricts the maximum extent to that limit,
            returns the modified this_edge,
            this is a helper function for _get_combined_edges and is used in the context where
            this_edge is replacing that_edge because it is better in some way
            """

        if int(round(this_edge.start)) == int(round(that_edge.start))\
        and int(round(this_edge.end)) == int(round(that_edge.end)):
            # no new extent, do nothing
            return this_edge

        # note its happened, this contributes to the 'quality' of the edge
        this_edge.count += that_edge.count

        # update the start/end such that it spans both this and that
        # this helps _get_combined_edges to create an accurate extent on the assumption that
        # edges are being considered in lowest start first order, so its just the end that is
        # growing and any subsequent edge will either overlap this one or be beyond it
        this_edge.end += self._get_end_extension(this_edge, that_edge, length_limit)
        if this_edge.end > length_limit:
            # we've now wrapped
            if this_edge.start > 0:
                # bring end to just before start
                this_edge.end = this_edge.start - 1
            else:
                # this has become full length
                this_edge.end = length_limit - 1

        return this_edge

    def _compare_edges(self, this_edge, that_edge, min_length):
        """ given two edges determine which is the better and why,
            returns the best edge and a reason code
            """

        # the length for comparison purposes is stretched by the number of overlaps it has
        # this makes lots of short edges better than one long one, each overlap represents
        # at least the minimum edge allowed, so we stretch by that for every overlap
        this_length = this_edge.length + ((this_edge.count - 1) * min_length)
        that_length = that_edge.length + ((that_edge.count - 1) * min_length)

        if this_length > that_length:
            return this_edge, self.THIS_LONGER
        if this_length < that_length:
            return that_edge, self.THAT_LONGER
        if this_edge.span > that_edge.span:
            return that_edge, self.THAT_THINNER
        if this_edge.span < that_edge.span:
            return this_edge, self.THIS_THINNER
        return this_edge, self.SAME_AS_THAT

    def _get_combined_edges(self, target, direction, centres, injections=[]):
        """ find all the edges (black to white (up) and white to black (down)) in the given target,
            direction is either top-down (looking for ring edges) or left-to-right (looking for bit edges),
            centres is just passed on to the edge detector,
            injections is an array of implied edges that are merged with the real edges,
            this function handles the logistics of finding both 'up' and 'down' edges and merging them into
            a single vector for returning to the caller, an empty vector is returned if no edges found,
            the returned vector consists of Edge instances
            """

        max_x, max_y = target.size()

        if direction == self.TOP_DOWN:
            # looking for ring edges in y
            context = 'ring'
            x_order = 0
            y_order = 1
            length_limit = max_x
            nominal_width = max_y / self.NUM_RINGS
            min_length = int(max(nominal_width * self.min_ring_edge_length, 2))
            reverse_direction = self.BOTTOM_UP
        elif direction == self.LEFT_TO_RIGHT:
            # looking for bit edges in x
            context = 'bit'
            x_order = 1
            y_order = 0
            length_limit = max_y
            nominal_width = (max_x / self.size)
            min_length = int(max(nominal_width * self.min_bit_edge_length, 2))
            reverse_direction = self.RIGHT_TO_LEFT
        else:
            raise Exception('illegal direction {}'.format(direction))

        # get black-to-white edges (go backwards for these)
        b2w_edges, threshold = self._get_transitions(target, x_order, y_order, False)
        edges_up, b2w_grid = self._get_edges(b2w_edges, reverse_direction, centres, threshold)
        if self.save_images:
            if b2w_grid is not None:
                self._unload(b2w_grid, '{}s-b2w-edges'.format(context))

        # get white_to_black edges (go forwards for these)
        w2b_edges, threshold = self._get_transitions(target, x_order, y_order, True)
        edges_down, w2b_grid = self._get_edges(w2b_edges, direction, centres, threshold)
        if self.save_images:
            if w2b_grid is not None:
                self._unload(w2b_grid, '{}s-w2b-edges'.format(context))

        if self.show_log:
            self._log('{}: found {} total edges, {} up and {} down'.
                      format(context, len(edges_up) + len(edges_down), len(edges_up), len(edges_down)))
            # show a sorted list of edges, highlighting if it was an up or down detection
            edge_up = None
            up_index = 0
            edge_down = None
            down_index = 0
            while True:
                if edge_up is None and up_index < (len(edges_up) - 1):
                    edge_up = edges_up[up_index]
                    up_index += 1
                if edge_down is None and down_index < (len(edges_down) - 1):
                    edge_down = edges_down[down_index]
                    down_index += 1
                if edge_up is None and edge_down is None:
                    break
                elif edge_up is None:
                    self._log('    down: {}'.format(edge_down))
                    edge_down = None
                elif edge_down is None:
                    self._log('    up: {}'.format(edge_up))
                    edge_up = None
                elif edge_up < edge_down:
                    self._log('    up: {}'.format(edge_up))
                    edge_up = None
                else:
                    self._log('    down: {}'.format(edge_down))
                    edge_down = None

        # merge the two sets of edges so that got a single list in ascending order, each is separately in order now,
        # we get them separately and merge rather than do both at once 'cos they can get very close in small images
        # we also drop overlapping edges (the 'best' is kept, 'best' is thinner or longer)
        rejects = []
        edges = []
        edge_up = None
        edge_down = None
        this_edge = None
        edge_injected = None
        while True:
            last_edge = this_edge

            # get the next group to consider
            if edge_up is None and len(edges_up) > 0:
                edge_up = edges_up.pop(0)
            if edge_down is None and len(edges_down) > 0:
                edge_down = edges_down.pop(0)
            if edge_injected is None and len(injections) > 0:
                edge_injected = injections.pop(0)

            # determine which one to add next (when got a choice, pick the lowest)
            if edge_up is None and edge_down is None and edge_injected is None:                # 000
                # nothing left, we're done
                break
            elif edge_up is None and edge_down is None and edge_injected is not None:          # 001
                this_edge = edge_injected
                edge_injected = None
            elif edge_up is None and edge_down is not None and edge_injected is None:          # 010
                this_edge = edge_down
                edge_down = None
            elif edge_up is None and edge_down is not None and edge_injected is not None:      # 011
                if edge_down < edge_injected:
                    this_edge = edge_down
                    edge_down = None
                else:
                    this_edge = edge_injected
                    edge_injected = None
            elif edge_up is not None and edge_down is None and edge_injected is None:          # 100
                this_edge = edge_up
                edge_up = None
            elif edge_up is not None and edge_down is None and edge_injected is not None:      # 101
                if edge_up < edge_injected:
                    this_edge = edge_up
                    edge_up = None
                else:
                    this_edge = edge_injected
                    edge_injected = None
            elif edge_up is not None and edge_down is not None and edge_injected is None:      # 110
                if edge_up < edge_down:
                    this_edge = edge_up
                    edge_up = None
                else:
                    this_edge = edge_down
                    edge_down = None
            else:                                                                              # 111
                # got all 3
                if edge_up < edge_down < edge_injected:
                    this_edge = edge_up
                    edge_up = None
                elif edge_up < edge_injected < edge_down:
                    this_edge = edge_up
                    edge_up = None
                elif edge_down < edge_up < edge_injected:
                    this_edge = edge_down
                    edge_down = None
                elif edge_down < edge_injected < edge_up:
                    this_edge = edge_down
                    edge_down = None
                elif edge_injected < edge_down < edge_up:
                    this_edge = edge_injected
                    edge_injected = None
                elif edge_injected < edge_up < edge_down:
                    this_edge = edge_injected
                    edge_injected = None
                else:
                    raise Exception('my understanding of chained operators is wrong!')

            # add this edge if it does not coincide or overlap with what is already there
            if len(edges) == 0:
                # nothing to compare, just add it
                if self.show_log:
                    self._log('{}: adding edge {} (initial edge)'.format(context, this_edge))
                edges.append(this_edge)
                continue
            that_edge = edges[-1]              # get what is already there

            if last_edge is not None and this_edge == last_edge:  # see Edge.__eq__ for == meaning here
                # got dup drop it
                if self.show_log:
                    self._log('{}: keeping edge {} (ignoring duplicate edge {} (duplicate of {})'.
                              format(context, that_edge, this_edge, last_edge))
                rejects.append(this_edge)
                continue

            joined_edge = self._merge_edge(this_edge, that_edge, length_limit)
            if joined_edge is not None:
                # we've got a coincident edge, replace end of list with this joined edge
                if self.show_log:
                    self._log('{}: extending edge {} (joining coincident edges of {} and {})'.
                              format(context, joined_edge, this_edge, that_edge))
                edges[-1] = joined_edge
                continue

            # if got overlapping span edges, keep the 'best', the best is longer and/or thinner,
            # length takes priority over thickness
            if not self._position_same(that_edge.position, this_edge.position, 1):
                # no overlap, add it
                if self.show_log:
                    self._log('{}: adding edge {} (non-overlapping with {})'.
                              format(context, this_edge, that_edge))
                edges.append(this_edge)
                continue
            # got an overlap, pick the best
            best_edge, best_reason = self._compare_edges(this_edge, that_edge, min_length)
            if best_reason == self.THIS_LONGER:
                # replace end of list with this one
                this_edge = self._extend_edge(this_edge, that_edge, length_limit)
                if self.show_log:
                    self._log('{}: updating edge {} (replacing shorter overlapping edge {})'.
                              format(context, this_edge, that_edge))
                rejects.append(edges[-1])
                edges[-1] = this_edge
            elif best_reason == self.THAT_LONGER:
                # just ignore this one
                that_edge = self._extend_edge(that_edge, this_edge, length_limit)
                if self.show_log:
                    self._log('{}: keeping edge {} (dropping shorter overlapping edge {})'.
                              format(context, that_edge, this_edge))
                rejects.append(this_edge)
            elif best_reason == self.THIS_THINNER:
                # replace end of list with this one
                this_edge = self._extend_edge(this_edge, that_edge, length_limit)
                if self.show_log:
                    self._log('{}: updating edge {} (replacing wider overlapping edge {})'.
                              format(context, this_edge, that_edge))
                rejects.append(edges[-1])
                edges[-1] = this_edge
            elif best_reason == self.THAT_THINNER:
                # just ignore this one
                that_edge = self._extend_edge(that_edge, this_edge, length_limit)
                if self.show_log:
                    self._log('{}: keeping edge {} (dropping wider overlapping edge {})'.
                              format(context, that_edge, this_edge))
                rejects.append(this_edge)
            elif best_reason == self.SAME_AS_THAT:
                # same, arbitrarily drop the this edge
                # just ignore this one
                that_edge = self._extend_edge(that_edge, this_edge, length_limit)
                if self.show_log:
                    self._log('{}: keeping edge {} (dropping similar overlapping edge {})'.
                              format(context, that_edge, this_edge))
                rejects.append(this_edge)
            else:
                raise Exception('unknown best edge reason {}'.format(best_reason))

        return edges, rejects

    def _get_error(self, reference, position, nominal_width, distance_max, allow_wrap):
        """ given two positions determine the relative error between them,
            the error is relative to the nominal width,
            if allow_wrap is True edges wrap (i.e. edge N+m wraps to m-1 when N > max),
            it does not matter if the reference is out of range, but position must not be
            """

        error = math.fabs(position - reference)
        if allow_wrap:
            other_error = math.fabs(reference + distance_max - position)
            error = min(error, other_error)

        # turn error into a ratio relative to nominal width
        error /= nominal_width

        return error

    def _match_edges(self, target, edges, direction):
        """ given a list of actual edge detections, for each build a list of probable matches to expectations,
            the edges list provided must be sorted into lowest position first order,
            target is present purely to find image limits,
            direction is TOP_DOWN (matching ring edges) or LEFT_TO_RIGHT (matching bit edges),
            returns a list of accepted and rejected edges,
            if there is no (close enough) match for an edge expectation None is set
            NB: bit edges wrap (x co-ord), rings do not (y co-ord)
            algorithm overview:
                allocate first edge as the actual
                allocate the next edge as the best within the error band of the first plus the nominal width
                continue until all edges allocated
                no edge is allowed more than once per matched set
            """

        max_x, max_y = target.size()
        if direction == self.TOP_DOWN:
            context = 'ring'
            distance_max = max_y
            wanted_edges = self.NUM_RINGS
            nominal_width = max_y / self.NUM_RINGS
            min_length = int(max(nominal_width * self.min_ring_edge_length, 2))
            max_error = self.max_ring_edge_error
        elif direction == self.LEFT_TO_RIGHT:
            context = 'bit'
            distance_max = max_x
            wanted_edges = self.size
            nominal_width = max_x / self.size
            min_length = int(max(nominal_width * self.min_bit_edge_length, 2))
            max_error = self.max_bit_edge_error
        else:
            raise Exception('illegal direction {}'.format(direction))

        # plus/minus this over our expected next neighbour position specifies the neighbour candidates
        error_margin = nominal_width * max_error

        # build ideal edge lists based on each actual edge
        for this in range(len(edges)):
            this_edge = edges[this]      # NB: this is a reference to the edge object not a copy
            ideal_edges = [None for _ in range(wanted_edges)]
            ideal_edges[0] = this_edge.position  # first one is our reference
            for edge in range(1, wanted_edges):
                ideal_edge = (ideal_edges[edge - 1] + nominal_width) % distance_max
                ideal_edges[edge] = ideal_edge
            this_edge.ideal = ideal_edges
            this_edge.bad = True         # this gets cleared if it gets used

        # for each actual edge build a list of best fit neighbours
        for this in range(len(edges)):
            this_edge = edges[this]

            # build neighbour set for this edge
            reference = this_edge.position
            actual_edges = [reference]
            candidate_edges = [edge for edge in edges]
            candidate_edges.pop(this)    # take self out
            while len(actual_edges) < wanted_edges:
                # set the allowed error band for our next reference
                reference += nominal_width
                reference = reference % distance_max
                low_1 = reference - error_margin
                high_1 = reference + error_margin
                if high_1 >= distance_max:
                    low_2 = 0
                    high_2 = high_1 - distance_max
                    high_1 = distance_max - 1
                elif low_1 < 0:
                    low_2 = low_1 + distance_max
                    high_2 = distance_max - 1
                    low_1 = 0
                else:
                    low_2 = low_1
                    high_2 = high_1

                # find first edge to be considered (first in the range low_1..high_1 or low_2..high_2)
                first_edge = None
                for that in range(len(candidate_edges)):
                    that_edge = candidate_edges[that]
                    that_position = that_edge.position
                    if low_1 <= that_position <= high_1:
                        first_edge = that
                        break
                    if low_2 <= that_position <= high_2:
                        first_edge = that
                        break

                # find last edge to be considered (first in the range low_1..high_1 or low_2..high_2)
                last_edge = None
                for that in range(len(candidate_edges)-1, -1, -1):
                    that_edge = candidate_edges[that]
                    that_position = that_edge.position
                    if low_1 <= that_position <= high_1:
                        last_edge = that
                        break
                    if low_2 <= that_position <= high_2:
                        last_edge = that
                        break

                # find the best candidate within the allowed set
                if first_edge is None or last_edge is None:
                    # no candidates
                    best_candidate = None
                else:
                    # there are candidates, evaluate them
                    best_candidate = first_edge
                    best_edge = candidate_edges[best_candidate]
                    for that in range(first_edge + 1, last_edge + 1):
                        # NB: every candidate here is known to be within our tolerated error band
                        that_edge = candidate_edges[that]
                        best_edge, better_reason = self._compare_edges(best_edge, that_edge, min_length)
                        if better_reason == self.THIS_LONGER:
                            # what we got is still the best
                            pass
                        elif better_reason == self.THIS_THINNER:
                            # what we got is still the best
                            pass
                        elif better_reason == self.THAT_LONGER:
                            # found a better one
                            best_candidate = that
                        elif better_reason == self.THAT_THINNER:
                            # found a better one
                            best_candidate = that
                        elif better_reason == self.SAME_AS_THAT:
                            # what we got no better
                            pass
                        else:
                            raise Exception('unknown best reason {}'.format(better_reason))

                if best_candidate is None:
                    # found no choice for this neighbour
                    actual_edges.append(None)
                else:
                    # allocate this edge
                    reference = candidate_edges[best_candidate].position   # set the new reference
                    actual_edges.append(reference)
                    # note we used it
                    candidate_edges[best_candidate].bad = False
                    # remove this edge as a further candidate
                    candidate_edges.pop(best_candidate)
                continue

            this_edge.actual = actual_edges

        # return list of good and bad edges
        return self._filter_bad_edges(edges, context)

    def _select_best_edge(self, best, edge, direction):
        """ given two edges select the better of the two,
            if edge is given but best is not, just return edge with an 'initial' description,
            if best is given but edge is not, just return best with a 'final' description,
            direction is TOP_DOWN (ring edges) or LEFT_TO_RIGHT (bit edges),
            returns the best edge and a reason for picking it,
            best is treated as the prevailing best choice and edge as the next candidate,
            this orientation is purely for reason text generation
            """

        if direction == self.TOP_DOWN:
            min_matches = self.min_ring_edge_matches
        elif direction == self.LEFT_TO_RIGHT:
            min_matches = self.min_bit_edge_matches
        else:
            raise Exception('illegal direction {}'.format(direction))

        if best is None and edge is None:
            raise Exception('at least one of best or edge must be given')

        if self.show_log:
            if edge is None:
                me = best
            else:
                me = edge
            criteria = ' (l:{}, s:{}, m:{:.2f}, e:{:.2f})'.format(me.length, me.span, me.matches, me.error)
        else:
            criteria = ''
        if best is None:
            return edge, 'picking {} as initial best candidate{}'.format(vstr(edge.actual), criteria)
        if edge is None:
            return best, 'picking {} as final best candidate{}'.format(vstr(best.actual), criteria)

        # the number of matches is only better if its below the minimum,
        # above the minimum a smaller error is better
        if best.matches < min_matches:
            # we haven't got many matches yet, so look for more
            if edge.matches > best.matches:
                # this one has more matches
                return edge, 'best now {} (more matches){}'.format(vstr(edge.actual), criteria)
            elif edge.matches < best.matches:
                # worse match, ignore it
                return best, 'best staying (more matches than {}){}'.format(vstr(edge.actual), criteria)
            else:
                # no better in matches, look for more criteria
                pass
        else:
            # we've got enough, so candidate must also have enough to be considered
            if edge.matches < min_matches:
                return best, 'best staying (more matches than {}){}'.format(vstr(edge.actual), criteria)

        # pick smallest error
        if edge.error < best.error:
            return edge, 'best now {} (smaller error){}'.format(vstr(edge.actual), criteria)
        elif edge.error > best.error:
            return best, 'best staying (smaller error than {}){}'.format(vstr(edge.actual), criteria)

        # pick longest
        if edge.length > best.length:
            return edge, 'best now {} (longer edge){}'.format(vstr(edge.actual), criteria)
        elif edge.length < best.length:
            return best, 'best staying (longer edge than {}){}'.format(vstr(edge.actual), criteria)

        # pick thinnest
        if edge.span < best.span:
            return edge, 'best now {} (thinner edge){}'.format(vstr(edge.actual), criteria)
        elif edge.span > best.span:
            return best, 'best staying (thinner edge than {}){}'.format(vstr(edge.actual), criteria)

        # pick more matches
        if edge.matches > best.matches:
            return edge, 'best now {} (more matches) {}'.format(vstr(edge.actual), criteria)
        elif edge.matches < best.matches:
            return best, 'best staying (more matches than {}){}'.format(vstr(edge.actual), criteria)

        # run out of criteria, keep the first one we found
        return best, 'best staying (no better than {}){}'.format(vstr(edge.actual), criteria)

    def _pick_best_edge_set(self, edges, direction, context):
        """ given a vector of edges pick the best one,
            edges may be ring edges or bit edges,
            direction is TOP_DOWN (ring edges) or LEFT_TO_RIGHT (bit edges),
            returns the best match from the set given,
            context is only present for debug messages
            """

        best = edges[0]

        if self.show_log:
            _, reason = self._select_best_edge(None, best, direction)
            self._log('{}: {}'.format(context, reason))

        for candidate in range(1, len(edges)):
            edge = edges[candidate]
            best, reason = self._select_best_edge(best, edge, direction)
            if self.show_log:
                if reason is not None:
                    self._log('{}: {}'.format(context, reason))

        if self.show_log:
            _, reason = self._select_best_edge(best, None, direction)
            self._log('{}: {}'.format(context, reason))

        return best

    def _fill_edge_gaps(self, target, actual_edges, direction):
        """ given an edge list interpolate across gaps to fill with an expectation,
            a 'gap' is a None value, the expectation is direction dependent,
            direction is TOP_DOWN (for rings) or LEFT_TO_RIGHT (for bits),
            target is the image the edges were detected in,
            its used to get dimension limits and to add a highlight of inserted edges,
            returns an edge list with no None values and in ascending order
            along with the modified target image when in debug mode (else None)
            """

        max_x, max_y = target.size()
        if direction == self.TOP_DOWN:
            context = 'ring'
            extent = self.NUM_RINGS
            edge_limit = max_y
            allow_wrap = False
        elif direction == self.LEFT_TO_RIGHT:
            context = 'bit'
            extent = self.size
            edge_limit = max_x
            allow_wrap = True
        else:
            raise Exception('given illegal direction {}'.format(direction))

        if self.save_images:
            grid = target
        else:
            grid = None

        start_at = None
        gap = 0
        for edge in range(extent + 1):         # +1 to get the wrapped case
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
                grid = self._draw_grid(grid, plot_x, plot_y, (128, 0, 0))
            if start_at is not None:
                # end of gap, fill in the missing edges
                stop_at = actual_edges[edge % extent]
                if stop_at < start_at:
                    # we've wrapped
                    if allow_wrap:
                        span = (stop_at + edge_limit) - start_at
                    else:
                        span = edge_limit - start_at
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
                        grid = self._draw_grid(grid, plot_x, plot_y, (0, 0, 128))
                    if self.show_log:
                        self._log('{}: inserting edge at {}'.format(context, new_edge))
                    actual_edges[(gap + extra) % extent] = new_edge
            start_at = None
            gap = 0

        actual_edges.sort()

        if self.show_log:
            self._log('{}: final actual edges: {}'.format(context, vstr(actual_edges)))

        return actual_edges, grid

    def _set_edge_errors(self, edges, direction, max_x, max_y):
        """ set the error for each of the edges,
            direction is top-down for rings or left-to-right for bits,
            max_x/y are the associated image dimensions,
            the error is represented as a fraction of the nominal width
            for the distance between actual and ideal edges,
            returns updated edges list
            """

        if direction == self.TOP_DOWN:
            # doing rings
            full_length = max_x
            nominal_width = max_y / self.NUM_RINGS
            allow_error_wrap = False
        elif direction == self.LEFT_TO_RIGHT:
            # doing bits
            full_length = max_y
            nominal_width = (max_x / self.size)
            allow_error_wrap = True
        else:
            raise Exception('illegal direction {}'.format(direction))

        # set edge error
        for edge in edges:
            # set error and matches
            error = 0
            errors = [None for _ in range(len(edge.actual))]
            for sample in range(len(edge.actual)):
                reference = edge.ideal[sample]
                actual_edge = edge.actual[sample]
                if actual_edge is not None:
                    err = self._get_error(actual_edge, reference, nominal_width, full_length, allow_error_wrap)
                    errors[sample] = err
                    if err > error:
                        error = err
            edge.errors = errors
            edge.error = error

        return edges

    def _count_matches(self, edges):
        """ count the actual/ideal edge matches for edges in the given list,
            returns the modified edges
            """

        for edge in edges:
            matches = 0
            for sample in range(len(edge.actual)):
                actual_edge = edge.actual[sample]
                if actual_edge is not None:
                    matches += 1
            edge.matches = matches / len(edge.actual)

        return edges

    def _filter_bad_edges(self, edges, context):
        """ split the given edges list into good and bad,
            bad edges have been so marked by _match_edges,
            once _match_edges is complete we do not want them, this function splits them out,
            context is purely for diagnostic messages,
            returns a separate good list and bad list,
            bad edges are due to them not being used in any match,
            in debug mode this is logged here
            """

        good_edges = []
        bad_edges = []
        for edge in edges:
            if edge.bad:
                bad_edges.append(edge)
                if self.show_log:
                    self._log('{}: dropping un-used edge {}'.format(context, edge))
            else:
                good_edges.append(edge)

        return good_edges, bad_edges

    def _select_edges(self, edges, direction, target, rejects, context):
        """ given a set of detected edges, pick the best set to continue with and fill in any gaps
            edges is the full set of edges as found by _get_combined_edges,
            rejects are edges that have been rejected (only present for diagnostic purposes),
            direction is TOP_DOWN (ring edges) or LEFT_TO_RIGHT (bit edges),
            returns a list of edge co-ordinates (y for rings, x for bits) of the selected/extended edges
            """

        max_x, max_y = target.size()

        # match actual edges to expectations
        edges, bad_edges = self._match_edges(target, edges, direction)

        # set the edge errors (used as a selection criteria in _pick_best_edge_set)
        edges = self._set_edge_errors(edges, direction, max_x, max_y)

        # set the edge matches (used as a selection criteria in _pick_best_edge_set)
        edges = self._count_matches(edges)

        if self.show_log:
            self._log('{}: found {} good edges'.format(context, len(edges)))
            for edge in edges:
                self._log('    edge: {}'.format(edge))
                self._log('        actual: {}'.format(vstr(edge.actual)))
                self._log('        ideals: {}'.format(vstr(edge.ideal)))
                self._log('        errors: {}'.format(vstr(edge.errors)))
        if self.save_images:
            grid = target
            lines = []
            for x in range(10, max_x, 10):
                lines.append([x, 0, x, max_y - 1])
            grid = self._draw_lines(grid, lines, (128, 0, 0))
            lines = []
            for y in range(10, max_y, 10):
                lines.append([0, y, max_x - 1, y])
            grid = self._draw_lines(grid, lines, (128, 0, 0))
            lines = []
            for edge in rejects:
                if direction == self.TOP_DOWN:
                    # doing rings
                    lines.append([edge.start, edge.position,
                                  (edge.start + edge.length - 1) % max_x, edge.position])
                else:
                    # doing bits
                    lines.append([edge.position, edge.start,
                                  edge.position, (edge.start + edge.length - 1) % max_y])
            grid = self._draw_lines(grid, lines, (0, 0, 255))
            lines = []
            for edge in bad_edges:
                if direction == self.TOP_DOWN:
                    # doing rings
                    lines.append([edge.start, edge.position,
                                  (edge.start + edge.length - 1) % max_x, edge.position])
                else:
                    # doing bits
                    lines.append([edge.position, edge.start,
                                  edge.position, (edge.start + edge.length - 1) % max_y])
            grid = self._draw_lines(grid, lines, (0, 0, 128))
            lines = []
            for edge in edges:
                if direction == self.TOP_DOWN:
                    # doing rings
                    lines.append([edge.start, edge.position,
                                  (edge.start + edge.length - 1) % max_x, edge.position])
                else:
                    # doing bits
                    lines.append([edge.position, edge.start,
                                  edge.position, (edge.start + edge.length - 1) % max_y])
            grid = self._draw_lines(grid, lines, (0, 255, 0))
            self._unload(grid, '{}s-edges'.format(context))

        # pick the best actual edge set
        best = self._pick_best_edge_set(edges, direction, context)

        if self.save_images:
            # draw all detected edges in green on our target
            detections = []
            for edge in edges:
                detections.append(edge.position)
            if direction == self.TOP_DOWN:
                grid = self._draw_grid(target, detections, None, (0, 128, 0))
            else:
                grid = self._draw_grid(target, None, detections, (0, 128, 0))
        else:
            grid = target

        # fill in the gaps in the set we've chosen
        edges, grid = self._fill_edge_gaps(grid, best.actual, direction)

        if self.save_images:
            self._unload(grid, '{}s'.format(context))

        return edges

    def _get_centres(self, edges, max_coord):
        """ given a list of edge co-ordinates, return their centres and widths,
            a centre is halfway between each co-ordinate where the last is assumed to wrap to the first
            """

        centres = [None for _ in range(len(edges))]
        for edge in range(len(edges)):
            here = edges[edge]
            there = edges[(edge + 1) % len(edges)]
            if there < here:
                # we've wrapped
                width = (there + max_coord) - here
            else:
                width = there - here
            centres[edge] = ((here + (width / 2) % max_coord), width)

        return centres

    def _drop_noise_ring_edges(self, rings, inner_ring, outer_ring):
        """ given a set of ring edges, drop any that are above the inner out below the outer,
            returns the modified edge set
            """

        if inner_ring is not None:
            for ring in rings:
                if (ring.position + 1) >= inner_ring:
                    # gone past inner ring
                    break
                # got a noise ring above the inner ring to get rid of
                if self.show_log:
                    self._log('ring: dropping edge {} above the inner ring at {}'.
                              format(ring, inner_ring))
                rings.pop(0)
        else:
            if self.show_log:
                self._log('no inner ring edge detected!')

        if outer_ring is not None:
            for ring in rings:
                if (ring.position - 1) <= outer_ring:
                    # not got to the outer ring yet
                    continue
                # got a noise ring below the outer ring to get rid of
                if self.show_log:
                    self._log('ring: dropping edge {} below the outer ring at {}'.
                              format(ring, outer_ring))
                rings.pop(-1)
        else:
            if self.show_log:
                self._log('ring: no outer edge detected!')

        return rings

    def _find_targets(self):
        """ find targets within our image,
            the detection consists of several steps:
                1. find all the image 'blobs'
                2. project the circular target into a rectangle of radius x angle
                3. flatten the image to adjust for perspective distortions (a circle can look like an ellipsis)
                5. find the ring edges
                6. find the bit edges
            there are validation constraints in most steps that may result in a target being rejected,
            the objective is to achieve 100% confidence in the result or reject it,
            returns a list of target candidates found which may be empty if none found (as instances of Target)
            and when debugging an image with all rejects labelled with why rejected (None if not debugging)
            """

        blobs = self._find_blobs()
        if len(blobs) == 0:
            # no blobs here
            return [], None

        targets = []
        rejects = []
        for blob in blobs:
            self.centre_x = blob.pt[0]
            self.centre_y = blob.pt[1]
            blob_size = blob.size / 2          # change diameter to radius

            # do the polar to cartesian projection
            projected, orig_radius = self._project(self.centre_x, self.centre_y, blob.size)  # this does not fail

            # do the perspective correction
            flattened, scale, reason = self._flatten(projected, orig_radius)
            if reason is not None:
                # failed - this means some constraint was not met
                if self.show_log:
                    self._log('{} - rejecting'.format(reason))
                if self.save_images:
                    rejects.append(self.Reject(self.centre_x, self.centre_y, blob_size, None, reason))
                continue
            max_x, max_y = flattened.size()
            nominal_width = max_y / self.NUM_RINGS

            if self.save_images:
                grid = self._draw_lines(flattened, [[0, nominal_width * 2, max_x - 1, nominal_width * 2]], (0, 128, 0))
                grid = self._draw_lines(grid, [[0, max_y - nominal_width, max_x - 1, max_y - nominal_width]], (0, 0, 128))
                self._unload(grid, 'flat')

            # get the ring edges
            probe_width = int(max((max_x / self.size) * self.min_ring_edge_length, 1))
            probe_centres = [x for x in range(probe_width, max_x - probe_width + 1, probe_width)]
            # add these implied edges that never exist in the detected edges
            injections = [self.Edge(0, max_x, 1, 0, max_x - 1),
                          self.Edge(max_y / self.NUM_RINGS, max_x, 1, 0, max_x - 1)]
            if self.show_log:
                self._log('ring: injecting implied edges')
                for injection in injections:
                    self._log('    {}'.format(injection))
            edges, rejected_edges = self._get_combined_edges(flattened, self.TOP_DOWN, probe_centres, injections)
            if len(edges) == 0:
                # we did not find any edges
                if self.show_log:
                    self._log('no ring edges found - rejecting')
                if self.save_images:
                    rejects.append(self.Reject(self.centre_x, self.centre_y,
                                               blob_size, None, 'no ring edges found'))
                continue

            # pick the best ring edge set to use
            rings = self._select_edges(edges, self.TOP_DOWN, flattened, rejected_edges, 'ring')

            # calculate the ring centres to probe for bit edges
            ring_centres = self._get_centres(rings, max_y)

            # get the bit edges
            probe_width = max(int(nominal_width * self.min_bit_edge_length), 1)
            probe_min = int(nominal_width * 2) + probe_width
            probe_max = int(max_y - nominal_width) - probe_width
            probe_centres = [y for y in range(probe_min, probe_max, probe_width)]
            bits, rejected_bits = self._get_combined_edges(flattened, self.LEFT_TO_RIGHT, probe_centres)
            if len(bits) == 0:
                # we did not find any edges
                if self.show_log:
                    self._log('bit: no edges found - rejecting')
                if self.save_images:
                    rejects.append(self.Reject(self.centre_x, self.centre_y,
                                               blob_size, rings[self.OUTER_BLACK], 'no bit edges found'))
                continue

            # pick the best bit edge set to use
            bits = self._select_edges(bits, self.LEFT_TO_RIGHT, flattened, rejected_bits, 'bit')

            # find the centres of each bit, these wrap last to first
            bit_centres = self._get_centres(bits, max_x)

            targets.append(self.Target(self.centre_x, self.centre_y, blob_size, scale,
                                       flattened, ring_centres, bit_centres))

        if self.save_images:
            # label all the blobs we processed that were rejected
            labels = self.transform.copy(self.image)
            for reject in rejects:
                x = reject.centre_x
                y = reject.centre_y
                blob_size = reject.blob_size / 2    # assume blob detected is just inner two white rings
                ring = reject.target_size
                reason = reject.reason
                if ring is None:
                    # it got rejected before its inner/outer ring was detected
                    ring = blob_size * 4
                # show blob detected
                colour = (255, 0, 0)           # blue
                labels = self.transform.label(labels, (x, y, blob_size), colour)
                # show reject reason
                colour = (0, 0, 255)           # red
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
            self.centre_x = target.centre_x
            self.centre_y = target.centre_y
            blob_size = target.size
            target_scale = target.scale
            image = target.image
            rings = target.rings
            bits = target.bits

            # set ring centre and sample size for each bit in each ring
            max_x, _ = image.size()
            ring_centres = [None for _ in range(len(rings))]
            sample_size = [[None for _ in range(len(bits))] for _ in range(len(rings))]
            for ring in range(self.INNER_WHITE, self.OUTER_BLACK + 1):
                ring_width = rings[ring][1]
                ring_centres[ring] = rings[ring][0]
                sample_height = ring_width * self.sample_height_factor
                for bit in range(len(bits)):
                    sample_width = bits[bit][1] * self.sample_width_factor
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
                self._unload(grid, 'grid')

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

            # calculate the target size relative to the original image
            target_size = ring_centres[self.OUTER_BLACK]  # this is the size in the target image
            target_size *= target_scale                   # scale to size in original image

            # add this result
            numbers.append(Target(self.centre_x, self.centre_y, blob_size, number, doubt, target_size))

            if self.show_log:
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
                    colour = (0, 255, 0)       # green
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
        if fatal:
            raise Exception(message)
        else:
            print(message)

    def _unload(self, image, suffix, centre_x=None, centre_y=None):
        """ unload the given image with a name that indicates its source and context,
            suffix is the file name suffix (to indicate context),
            centre_x/y identify the blob the image represents, if None use decodiing context,
            centre_x/y of 0,0 means no x/y identification on the image
            """
        if centre_x is None:
            centre_x = self.centre_x
        if centre_y is None:
            centre_y = self.centre_y
        if centre_x > 0 and centre_y > 0:
            name = '{:.0f}x{:.0f}y-{}'.format(centre_x, centre_y, suffix)
        else:
            name = suffix
        filename = image.unload(self.original.source, name)
        if self.show_log:
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
            if colours is None it defaults to dark red, dark blue, drak green,
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
                                "end": (max_x-1, h)})

        if vertical is not None:
            for v in vertical:
                objects.append({"colour": colours[1],
                                "type": self.transform.LINE,
                                "start": (v, 0),
                                "end": (v, max_y-1)})

        if horizontal is not None and vertical is not None:
            if radius is not None:
                for h in horizontal:
                    for v in vertical:
                        objects.append({"colour": colours[2],
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
        scale = (colour[0]/255, colour[1]/255, colour[2]/255)
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
            ring = Ring(x >> 1, y >> 1, self.code_bits, width, self.frame, self.contrast, self.offset)
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
                        print('Found {} ({}) at {:.0f}x, {:.0f}y size {:.2f}, doubt level {}'.
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
                        print('{}Found {} ({}) at {:.0f}x, {:.0f}y size {:.2f}, doubt level {}'.
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
test_scan_angle_steps = 64
test_debug_mode = Scan.DEBUG_IMAGE
test_debug_mode = Scan.DEBUG_VERBOSE

test = Test(code_bits, min_num, max_num, parity, edges, rings, contrast, offset)
test_num_set = test.test_set(6)

#test.coding()
#test.decoding(test_black, test_white, test_noise)
#test.circles()
#test.code_words(test_num_set)
#test.rings(test_ring_width)
#test.codes(test_num_set, test_ring_width)

#test.scan(test_scan_angle_steps, [000], '15-segment-angle-test.png', test_debug_mode)
#test.scan(test_scan_angle_steps, [000], 'photo-angle-test-flat.jpg', test_debug_mode)
#test.scan(test_scan_angle_steps, [000], 'photo-angle-test-curved-flat.jpg', test_debug_mode)

#test.scan(test_scan_angle_steps, [101], '15-segment-101.png', test_debug_mode)
#test.scan(test_scan_angle_steps, [102], '15-segment-102.png', test_debug_mode)
#test.scan(test_scan_angle_steps, [365], '15-segment-365.png', test_debug_mode)
#test.scan(test_scan_angle_steps, [640], '15-segment-640.png', test_debug_mode)
#test.scan(test_scan_angle_steps, [658], '15-segment-658.png', test_debug_mode)
#test.scan(test_scan_angle_steps, [828], '15-segment-828.png', test_debug_mode)
#test.scan(test_scan_angle_steps, [101], 'photo-101.jpg', test_debug_mode)
#test.scan(test_scan_angle_steps, [102], 'photo-102.jpg', test_debug_mode)
#test.scan(test_scan_angle_steps, [365], 'photo-365.jpg', test_debug_mode)
#test.scan(test_scan_angle_steps, [640], 'photo-640.jpg', test_debug_mode)
#test.scan(test_scan_angle_steps, [658], 'photo-658.jpg', test_debug_mode)
#test.scan(test_scan_angle_steps, [828], 'photo-828.jpg', test_debug_mode)
#test.scan(test_scan_angle_steps, [102], 'photo-102-distant.jpg', test_debug_mode)
test.scan(test_scan_angle_steps, [365], 'photo-365-oblique.jpg', test_debug_mode)
#test.scan(test_scan_angle_steps, [365], 'photo-365-blurred.jpg', test_debug_mode)
#test.scan(test_scan_angle_steps, [658], 'photo-658-small.jpg', test_debug_mode)
#test.scan(test_scan_angle_steps, [658], 'photo-658-crumbled-bright.jpg', test_debug_mode)
#test.scan(test_scan_angle_steps, [658], 'photo-658-crumbled-dim.jpg', test_debug_mode)
#test.scan(test_scan_angle_steps, [658], 'photo-658-crumbled-close.jpg', test_debug_mode)
#test.scan(test_scan_angle_steps, [658], 'photo-658-crumbled-blurred.jpg', test_debug_mode)
#test.scan(test_scan_angle_steps, [658], 'photo-658-crumbled-dark.jpg', test_debug_mode)
#test.scan(test_scan_angle_steps, [101, 102, 365, 640, 658, 828], 'photo-all-test-set.jpg', test_debug_mode)
