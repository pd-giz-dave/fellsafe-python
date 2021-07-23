import os.path

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
        surrounded by a solid ring of 'black' and width R,
        surrounded by 3 concentric data rings of width R and divided into N (typically 14..16) equal segments,
        surrounded by a bit-timing ring of alternating black/white half bits and width R,
        enclosed by a solid 'black' ring of width R and, finally, a solid 'white' ring of radius R. 
    Total radius is 9R.
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
    See Scan.find_targets for the detection algorithm in the image.
    The bit-timing ring is used to determine all the bit boundaries in the ring (by an angle) even when distorted
    by perspective or being wrapped around someones leg. The bits are extracted by thresholding. There are two
    thresholds, min-grey and max-grey, which are determined by dividing the luminance range detected across the
    radius for each bit by three.  The result of the thresholding is 3 levels: black (0), white (1), grey (?).
    The bit skew between rings is known, so all three rings can be decoded into these three levels. They are
    then decoded as follows:
        three 0's         = 0
        three 1's         = 1
        two zeroes + grey = maybe 0
        two ones + grey   = maybe 1
        anything else     = junk (a give-up condition)
    This results in 5 states for each bit: 0, 1, maybe 0 (0?), maybe 1 (1?), junk (!).
    A potential alignment marker is sought in this order:
        exact match - 0110 (can only be one in a valid detection)
        single maybe bit - i.e. one of the market bits is a (0?) or (1?)      (could be several candidates)
        double maybe bit - i.e. two of the market bits is a (0?) or (1?)      (could be several candidates)
        triple maybe bit - i.e. three of the market bits is a (0?) or (1?)    (could be several candidates)
        quadruple maybe bit - i.e. four of the market bits is a (0?) or (1?)  (could be several candidates)
    When there are several maybe candidates, each is tried to extract a code word. When doing that (0?) is
    treated as 0, (1?) is treated as 1, if more than one succeed its a give-up situation as there is too much
    ambiguity. When giving up it means consider the next bullseye candidate.
    
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
            returns the number (or None) and the bit classification for each bit (as a CSV string)
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
                        return None, self._show_bits(bits)
                    found = code         # note the first one we find
            if found is not None:
                # only got 1 from this maybe level, go with it
                return found, self._show_bits(bits)

        # no candidates qualify
        return None, self._show_bits(bits)

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
            """
        if a < 0 or a > 360:
            return None, None
        if r == 0:
            return 0, 0
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
    """ draw a ring of width w and radius r from centre x,y with s segments containing bits b,
        all bits 1 is solid white, all 0 is solid black
        bit 0 (MSB) is drawn first, then 1, etc, up to bit s-1 (LSB), this is big-endian and is
        considered to be clockwise
        """

    def __init__(self, centre_x, centre_y, segments, width, frame):
        # set constant parameters
        self.s = segments                # how many bits in each ring
        self.w = width                   # width of each ring
        self.c = frame                   # where to draw it
        self.x = centre_x                # where the centre of the rings are
        self.y = centre_y                # ..
        # setup our angles look-up table
        scale = 2 * math.pi * width * 7
        self.angle_xy = Angle(scale).polarToCart
        self.edge = 360 / self.s         # the angle at which a bit edge occurs (NB: not an int)

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
            a bit value of other than 0 or 1 is drawn as 'grey'
            """
        if bit == 0:
            colour = min_luminance
        elif bit == 1:
            colour = max_luminance
        else:
            colour = mid_luminance
        self._pixel(x, y, colour)

    def _draw(self, radius, bits):
        """ draw a ring at radius of bits, if bits is None the bit timing ring is drawn
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
                        segment = int(a / self.edge)           # full-bit segments for data (or unicolour) rings
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
            if data_bits is None a grey ring is drawn (see _draw)
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
                                self._point(x+dx, y+dy, 0)
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
        # draw the bit timing ring
        self.draw(6, None)
        # draw the outer black/white rings
        self.draw(7,  0)
        self.draw(8, -1)
        # draw a human readable label
        self.label(number)

class Frame:
    """ image frame buffer as a 2D array of luminance values """

    source = None
    buffer = None
    alpha  = None
    max_x  = None
    max_y  = None

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
        """ unload the frame buffer to a PNG image file """
        if len(self.buffer.shape) == 2:
            # its a grey scale image, convert to RGBA
            image = self.colourize()
        else:
            # assume its already colour
            image = self.buffer
        filename, ext = os.path.splitext(image_file)
        if suffix is not None:
            cv2.imwrite('_{}-{}.png'.format(filename, suffix), image)
        else:
            cv2.imwrite('{}.png'.format(filename), image)

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
        """ determine of the points are radius R and centred at X, Y are within the image """
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
    """ various image transforming operations """

    # mnemonics for annotate object types
    LINE = 0
    CIRCLE = 1
    RECTANGLE = 2
    PLOTX = 3
    PLOTY = 4
    TEXT = 5

    def blur(self, source, size=3):
        """ apply a median blur to the given cv2 image with a kernel of the given size """
        target = source.instance()
        target.set(cv2.medianBlur(source.get(), size))
        return target

    def downSize(self, source):
        """ blur and downsize (by half in both directions) the given image """
        target = source.instance()
        target.set(cv2.pyrDown(source.get()))
        return target

    def resize(self, source, new_size):
        """ resize the given image such that either its width or height is at most that given,
            the aspect ration is preserved,
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
        target.set(cv2.resize(source.get(), (new_width, new_height), interpolation=cv2.INTER_LINEAR))
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

    def blobs(self, source, low=min_luminance, high=max_luminance, levels=32):
        """ find bright blobs in the given image,
            low, high are the min and max thresholds for the blob detector,
            levels is how many binarizations to perform (used to calculate threshold step)
            returns a keypoints array, each keypoint has:
                .pt[1] = x co-ord of the centre
                .pt[0] = y co-ord of centre
                .size = diameter of blob
            all floats
            """

        # Setup SimpleBlobDetector parameters.
        # These have been tuned for detecting circular blobs of a certain size range
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = low
        params.maxThreshold = high
        params.thresholdStep = int(round((high - low) / levels))
        params.filterByCircularity = True      # needed to stop detection of blobs that go off the image edge
        params.minCircularity = 0.8
        params.filterByConvexity = False       # we want to find distorted blobs, so do not filter on this
        params.minConvexity = 0.9
        params.filterByInertia = False         # we want to find stretched out blobs, so do not filter on this
        params.minInertiaRatio = 0.5
        params.filterByArea = True
        params.minArea = 30                    # needed to stop detection of very small blobs
        params.maxArea = 500000                # needed to make sure we find huge ones (whole image is one blob)
        params.filterByColor = True
        params.blobColor = max_luminance       # we want bright blobs

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs
        return detector.detect(source.get())

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

    def label(self, source, keypoint, colour=(0, 0, 255), title=None):
        """ return an image with a coloured ring around the given key point in the given image
            and a textual title at the key point centre if one is given,
            keypoint is a list of centre x,y and circle radius
            """
        objects = []
        org = (int(round(keypoint[0])), int(round(keypoint[1])))
        objects.append({"type": self.CIRCLE,
                        "centre": org,
                        "radius": int(round(keypoint[2]))})
        if title is not None:
            objects.append({"type": self.TEXT,
                            "start": org,
                            "size": 0.5,
                            "text": title})
        return self.annotate(source, objects, colour)

    def annotate(self, source, objects, colour=(0, 0, 255)):
        """ annotate an image with 'objects' (an array) in the given colour, each object is a dictionary of:
                type, - line, circle, rectangle, plot-x, plot-y, text
                start or centre position (x,y) or first plot x,
                end position (x,y) or radius or plot-x y points or plot-y x points or font-size,
                text string for the text type
            returns the source updated, it'll be converted to colour if required,
            """
        image = source.get()
        if len(image.shape) == 2:
            # its a grey scale image, convert to RGB (actually BGR)
            image = cv2.merge([image, image, image])
        for obj in objects:
            if obj["type"] == self.LINE:
                image = cv2.line(image, obj["start"], obj["end"], colour, 1)
            elif obj["type"] == self.CIRCLE:
                image = cv2.circle(image, obj["centre"], obj["radius"], colour, 1)
            elif obj["type"] == self.RECTANGLE:
                image = cv2.rectangle(image, obj["start"], obj["end"], colour, 1)
            elif obj["type"] == self.PLOTX:
                points = obj["points"]
                if points is not None:
                    x = obj["start"]
                    for pt in range(len(points)):
                        y = points[pt]
                        if y is not None:
                            image[y, x + pt] = colour
            elif obj["type"] == self.PLOTY:
                points = obj["points"]
                if points is not None:
                    y = obj["start"]
                    for pt in range(len(points)):
                        x = points[pt]
                        if x is not None:
                            image[y + pt, x] = colour
            elif obj["type"] == self.TEXT:
                image = cv2.putText(image, obj["text"], obj["start"], cv2.FONT_HERSHEY_SIMPLEX, obj["size"], colour, 1, cv2.LINE_AA)
            else:
                raise Exception('Unknown object type {}'.format(obj["type"]))
        source.set(image)
        return source

class Scan:
    """ scan an image looking for codes """

    # directions
    TOP_DOWN = +1              # top-down y scan direction, see _find_brightest_radii and _find_radius_edge
    BOTTOM_UP = -1             # bottom-up y scan direction, see _find_brightest_radii and _find_radius_edge
    LEFT_TO_RIGHT = +1         # left-to-right x scan direction, see _follow_edge_in_x, must be +1
    RIGHT_TO_LEFT = -1         # right-to-left x scan direction, see _follow_edge_in_x, must be -1

    # ring numbers of perspective corrected image
    INNER_WHITE = 0
    INNER_BLACK = 1
    DATA_RING_1 = 2
    DATA_RING_2 = 3
    DATA_RING_3 = 4
    TIMING_RING = 5
    OUTER_BLACK = 6

    # total number of rings in the whole code (central blob counts as 2)
    NUM_RINGS = 9

    def __init__(self, code, frame, angles=360, debug=False):
        """ code is the code instance defining the code structure,
            frame is the frame instance containing the image to be scanned,
            do all the pre-processing here, the pre-processing just resizes to HD video standard
            """
        # constants
        self.min_inner_outer_span = self.NUM_RINGS - 2   # min tolerable gap between inner and outer target edge
        self.blob_radius_stretch = 1.6     # how much to stretch blob radius to ensure always cover the whole lot
        self.min_timing_length = 3         # min length of a segment edge in the timing ring
        self.min_edge_thickness = 2        # minimum timing ring edge thickness
        self.min_timing_gap = 0.4          # the fraction of the timing ring mean edge gap that is 'too-close'
        self.sample_width_factor = 0.4     # fraction of a segment width that is probed for a luminance value
        self.sample_height_factor = 0.3    # fraction of a ring height that is probed for a luminance value
        self.min_run_length = 4            # minimum consecutive samples when looking for ring edges
        self.min_slope = 3                 # minimum slope when looking for ring edge maxima
        self.min_ring_image_gap = 3        # minimum gap between image edge and start of ring edge probing

        # ring position fudge factors,
        # the inner white ring end-y is moved up a bit to get it away from the white/black edge,
        # the inner black ring start-y is moved up a bit to get it away from the first data ring edge,
        # the timing ring start-y is moved down a bit to get it away from the last data ring edge,
        # the outer black ring is moved down a bit to get it away from the timing ring
        # these are fudge factors tuned by experimentation, see _find_ring_edges
        # for each ring an offset is defined for moving the ring in y (as a width ratio), +ve=down, -ve=up
        # and a stretch factor for the ring width (also as a ratio), +ve=bigger, -ve=smaller
        self.ring_adjustments = [[ 0  , -0.1],   # inner white
                                 [-0.1, -0.1],   # inner black
                                 [-0.1, +0.0],   # data ring 1
                                 [-0.1, +0.0],   # data ring 2
                                 [-0.1, +0.0],   # data ring 3
                                 [+0.1, +0.1],   # timing ring
                                 [+0.1,  0  ]]   # outer black

        # these thresholds are multipliers on the mean luminance of the edge image,
        # < 1 == make is smaller, > 1 == make it bigger, 0 == leave as is
        self.min_edge_threshold = int(round(max_luminance * 0.1))  # ignore edge pixels below this when getting mean

        # minimum luminance for an edge pixel to qualify as an edge when following in x
        self.edge_threshold_x = 0.6

        # minimum luminance for an edge pixel to qualify as an edge when following in y
        self.edge_threshold_y = 0.9

        # minimum luminance for an edge pixel to qualify as an edge when looking for ring boundaries
        self.edge_threshold_ring = 0.8

        # params
        self.angle_steps = angles                          # angular resolution when 'projecting'
        self.original = frame
        self.debug = debug
        self.c = code
        self.size = code.size                              # total ring code size in bits

        # context
        self.transform = Transform()                       # make a new frame instance
        max_circumference = 3600                           # good enough for 0.1 degree resolution
        angle = Angle(int(round(max_circumference)))
        self.angle_xy = angle.polarToCart

        # samples probed (x,y offsets from self) when following for edges
        # the samples are arranged in an order that tends to keep things in a straight line
        # x,y pairs for following edges in x, it is looser than that in y in that it tolerates small gaps
        self.edge_kernel_x = [[0,  0], [1,  0], [2,  0],   # same y
                              [0, +1], [1, +1], [0, +2],   # +1 on y
                              [0, -1], [1, -1], [0, -2]]   # +2 on y
        # y,x pairs for following edges in y, this one will not tolerate a gap
        self.edge_kernel_y = [[0,  0],                     # same x
                              [0, +1],                     # +1 on x
                              [0, -1]]                     # -1 on x

        # edge vector smoothing kernel, pairs of offset and scale factor (see _smooth_edge)
        # NB: the smoothing is done in-place, so only look ahead here
        self.edge_smoothing_kernel = [[0, 1], [+1, 1], [+2, 1], [+3, 1]]

        # the timing ring consists of an edge per bit, edges that are too close are rejected,
        # 'too close' is defined as a fraction the expected width which is the ring circumference divided
        # by the number of bits in the ring, the circumference is the angle_steps, the bits is size
        self.min_timing_width = int(round((self.angle_steps / self.size) * 0.66))

    def _find_blobs(self):
        """ find the target blobs in our image,
            creates a blob list each of which is a 'keypoint' tuple of:
                .pt[1] = x co-ord of the centre
                .pt[0] = y co-ord of centre
                .size = diameter of blob
            returns a list of blobs found
            """

        blurred = self.transform.blur(self.original)           # de-noise
        self.image = self.transform.resize(blurred, 1080)      # re-size (to HD video convention)
        blobs = self.transform.blobs(self.image)               # find the blobs

        return blobs

    def _project(self, centre_x, centre_y, blob_size):
        """ 'project' a potential target from its circular shape to a rectangle of radius (y) x angle (x),
            blob_size is used as a guide to limit the radius projected,
            we assume the blob-size is (roughly) the diameter of the inner two white rings
            but err on the side of going too big,
            when projecting small radii the same pixel will be sampled several times,
            when projecting large radii some pixels may be missed,
            we just sample a straight line between the last x,y co-ordinate pair to the next for each angle
            and calculate the mean of those
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

        # iterate every angle at every radius (ToDo: optimise angle per radius? Its wasteful to do all at small radii)
        angle_delta = 360 / self.angle_steps
        code = self.original.instance().new(self.angle_steps, limit_radius, min_luminance)
        for radius in range(limit_radius):
            last_dx = int(round(centre_x))
            last_dy = int(round(centre_y))
            for angle in range(self.angle_steps):
                degrees = angle * angle_delta
                x, y = self.angle_xy(degrees, radius)
                if x is None:
                    pixel = mid_luminance
                else:
                    dx = int(round(centre_x + x))
                    dy = int(round(centre_y + y))
                    if last_dy < dy:
                        start_y = last_dy + 1
                        stop_y = dy
                    elif last_dy > dy:
                        start_y = dy
                        stop_y = last_dy - 1
                    else:
                        start_y = dy
                        stop_y = dy
                    if last_dx < dx:
                        start_x = last_dx + 1
                        stop_x = dx
                    elif last_dx > dx:
                        start_x = dx
                        stop_x = last_dx - 1
                    else:
                        start_x = dx
                        stop_x = dx
                    samples = max((stop_y - start_y) + (stop_x - start_x) + 1, 2)
                    line = np.linspace((start_x, start_y), (stop_x, stop_y), samples)
                    pixel = 0
                    count = 0
                    for sample_x, sample_y in line:
                        sample = self.image.getpixel(int(round(sample_x)), int(round(sample_y)))
                        if sample is not None:
                            pixel += sample
                            count += 1
                    if count > 0:
                        pixel = int(round(pixel / count))
                    else:
                        pixel = mid_luminance
                last_dx = dx
                last_dy = dy
                code.putpixel(angle, radius, pixel)
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
                    if self.debug:
                        print('Blob: {:.0f}x{:.0f}y - edge[{}] is None! - info'.format(centre_x, centre_y, (x + dx) % extent))
                else:
                    v += int(round(sample * f))
                    d += f
            if d > 0:
                edge[x] = int(round(v / d))
        return edge

    def _find_brightest_in_y(self, target, x, y, direction, threshold_y):
        """ from x,y in the target image find the brightest radii that is over the given threshold,
            the pixel at x,y must be over else None is returned,
            direction is top-down or bottom-up,
            returns the y co-ord of the brightest that is still over the threshold,
            """
        brightest = target.getpixel(x, y)
        if brightest is None or brightest < threshold_y:
            return None
        brightest_at = y
        if direction == self.TOP_DOWN:
            _, end_y = target.size()
            radius = (dy for dy in range(y, end_y))
        else:
            # bottom-up
            end_y = 0
            radius = (dy for dy in range(y, end_y, -1))
        for dy in radius:
            pixel = target.getpixel(x, dy)
            if pixel < threshold_y:
                break
            if pixel > brightest:  # find first occurrence
                brightest = pixel
                brightest_at = dy
        return brightest_at

    def _find_brightest_in_x(self, target, x, y, direction, threshold_x):
        """ from x,y in the target image find the brightest angle that is over our threshold,
            the pixel at x,y must be over else None is returned,
            direction is left-to-right or right-to-left
            returns the x co-ord of the brightest that is still over the threshold and its level,
            this is used to find timing ring edges, in that context an edge has to have a minimum
            thickness to qualify
            NB: angles wrap 360..0 in target
            """
        brightest = target.getpixel(x, y)
        if brightest is None or brightest < threshold_x:
            return None, None
        start_x = x
        brightest_at = x
        limit_at = x
        max_x, _ = target.size()
        for _ in range(max_x-1):
            x += direction
            pixel = target.getpixel(x, y)
            if pixel < threshold_x:
                break
            if pixel >= brightest:        # find last occurrence
                brightest = pixel
                brightest_at = x
            limit_at = x
        thickness = math.fabs(start_x - limit_at) + 1
        if thickness < self.min_edge_thickness:
            # too feeble an edge to consider
            return None, None
        return brightest_at, brightest

    def _find_brightest_neighbour_in_x(self, target, x, y, direction_in_x, threshold_x):
        """ find brightest edge in the forward (i.e. increasing x) or backward neighbour in target from x, y,
            returns the pixel level and y co-ord of the brightest neighbour that is over our edge threshold,
            returns 0,0 if there is not one,
            this function is intended to be used when following edges in the x direction,
            direction_in_x is either LEFT_TO_RIGHT or RIGHT_TO_LEFT,
            it finds the best ongoing y co-ordinate to follow, it tolerates small gaps in x
            """
        brightest = 0
        brightest_at = 0
        max_x, _ = target.size()
        for dx, dy in self.edge_kernel_x:
            # scan our forward neighbours and follow the brightest that is over our threshold
            tx = (x + (dx * direction_in_x)) % max_x
            ty = y + dy
            pixel = target.getpixel(tx, ty)
            if pixel is not None and pixel >= threshold_x:
                # found an ongoing neighbour also in range, is it the brightest?
                if pixel > brightest:    # note first occurrence of brightest
                    brightest = pixel
                    brightest_at = ty
        return brightest, brightest_at

    def _find_brightest_neighbour_in_y(self, target, x, y, direction_in_y, threshold_y):
        """ find brightest edge in the downwards (i.e. increasing y) or upwards neighbour in target from x, y,
            returns the pixel level and x co-ord of the brightest neighbour that is over our edge threshold,
            returns 0,0 if there is not one,
            this function is intended to be used when following edges in the y direction,
            it finds the best ongoing x co-ordinate to follow
            """
        brightest = 0
        brightest_at = 0
        max_x, max_y = target.size()
        for dy, dx in self.edge_kernel_y:
            # scan our forward neighbours and follow the brightest that is over our threshold
            ty = (y + (dy * direction_in_y))
            tx = (x + dx) % max_x
            pixel = target.getpixel(tx, ty)
            if pixel is not None and pixel >= threshold_y:
                # found an ongoing neighbour also in range, is it the brightest?
                if pixel > brightest:    # note first occurrence of brightest
                    brightest = pixel
                    brightest_at = tx

        return brightest, brightest_at

    def _follow_edge_in_x(self, target, x, y, direction_in_x, direction_in_y, threshold_x, threshold_y):
        """ follow the edge at x,y until come to its end in x (its followed for up to a full revolution - 360 deg),
            return a vector of its y co-ords, or an empty vector if none,
            the pixel at x,y must be over else None is returned,
            y direction is top-down or bottom-up, x direction is left-to-right or right-to-left
            """
        edge_y = self._find_brightest_in_y(target, x, y, direction_in_y, threshold_y)
        if edge_y is None:
            return []
        edge = [edge_y]
        increment = direction_in_x
        max_x, _ = target.size()
        for _ in range(max_x - 1):
            x = (x + increment) % max_x
            brightest, brightest_dy = self._find_brightest_neighbour_in_x(target, x, edge_y, direction_in_x, threshold_x)
            if brightest == 0:
                # no more qualifying neighbors
                return edge
            else:
                edge_y = brightest_dy
                edge.append(edge_y)
        return edge

    def _follow_edge_in_y(self, target, x, y, direction_in_x, direction_in_y, threshold_x, threshold_y):
        """ follow the edge at x,y until come to its end in y (its followed until it hits the image edge),
            return a vector of its x co-ords, or an empty vector if none, and the average brightness,
            the pixel at x,y must be over the brightness threshold else None is returned,
            y direction is top-down or bottom-up, x direction is left-to-right or right-to-left
            """
        edge_x, edge_level = self._find_brightest_in_x(target, x, y, direction_in_x, threshold_x)
        if edge_x is None:
            return [], None
        _, max_y = target.size()
        edge = [edge_x]
        if direction_in_y == self.TOP_DOWN:
            radius = (dy for dy in range(y+1, max_y))
            increment = +1
        else:
            # bottom-up
            radius = (dy for dy in range(y-1, -1, -1))
            increment = -1
        samples = 1
        edge_brightness = int(edge_level)
        for _ in radius:
            y += increment
            brightest, brightest_dx = self._find_brightest_neighbour_in_y(target, edge_x, y, direction_in_y, threshold_y)
            if brightest == 0:
                # no more qualifying neighbors
                break
            else:
                edge_x = brightest_dx
                edge.append(edge_x)
                edge_brightness += brightest
                samples += 1
        return edge, edge_brightness / samples

    def _find_radius_edge(self, target, direction, centre_x, centre_y, threshold_x, threshold_y):
        """ look for a continuous edge in the given target either top-down (inner) or bottom-up (outer),
            direction is top-down or bottom-up, an 'edge' is vertical, i.e along the radius,
            centre_x/y are purely for diagnostic reporting,
            to qualify as an 'edge' it must be continuous across all angles (i.e. no dis-connected jumps)
            to find the edge we scan the radius at angle 0 looking for a pixel above our edge threshold,
            then follow that along the angle axis, if we get to 360 without a break we've found it
            """
        max_x, max_y = target.size()
        if direction == self.TOP_DOWN:
            radius = (y for y in range(max_y))
        else:
            radius = (y for y in range(max_y-1, -1, -1))
        for y in radius:
            edge = self._follow_edge_in_x(target, 0, y, self.LEFT_TO_RIGHT, direction, threshold_x, threshold_y)
            if len(edge) == max_x:
                # found an edge that goes all the way, so that's it, just smooth it and go
                edge = self._smooth_edge(edge, centre_x, centre_y)
                return edge
        # we did not find one
        return None

    def _get_edge_threshold(self, target):
        """ find the edge threshold to apply in the given target,
            target is assumed to be an edge detected image,
            this function finds the mean luminance (ignoring near black pixels)
            and returns that (turned into an int in the range min_luminance..max_luminance)
            """
        buffer = target.get().reshape(-1)
        buffer_pixels = len(buffer)
        filtered = buffer[np.where(buffer > self.min_edge_threshold)]  # remove all near black pixels
        if len(filtered) == 0:
            # nothing qualifies
            mean = 0.0
        else:
            mean = np.mean(filtered)
        return mean

    def _find_extent(self, target, centre_x, centre_y):
        """ find the inner and outer edges of the given target,
            the inner edge is the first white to black transition that goes all the way around,
            the outer edge is the first black to white transition that goes all the way around,
            the centre_x and centre_y params are only present for naming the debug image files created,
            returns two vectors, y co-ord for every angle of the edges, or a reason if one or both not found,
            we look for the outer first as that will fail when looking at junk whereas the inner may not
            """

        # create a vector of the outer edge radius for every angle (this can be distorted by curvature)
        v_b2w_edge = self.transform.edges(target, 0, 1, 3, False)
        b2w_mean = self._get_edge_threshold(v_b2w_edge)
        b2w_threshold_x = int(round(b2w_mean * self.edge_threshold_x))
        b2w_threshold_y = int(round(b2w_mean * self.edge_threshold_y))
        ring_outer_edge = self._find_radius_edge(v_b2w_edge, self.TOP_DOWN,
                                                 centre_x, centre_y, b2w_threshold_x, b2w_threshold_y)

        if ring_outer_edge is None:
            # don't bother looking for the inner if there is no outer
            ring_inner_edge = None
        else:
            # found an outer, so worth looking for an inner
            # create a vector of the inner edge radius for every angle (this is usually accurate)
            v_w2b_edge = self.transform.edges(target, 0, 1, 3, True)
            w2b_mean = self._get_edge_threshold(v_w2b_edge)
            w2b_threshold_x = int(round(w2b_mean * self.edge_threshold_x))
            w2b_threshold_y = int(round(w2b_mean * self.edge_threshold_y))
            ring_inner_edge = self._find_radius_edge(v_w2b_edge, self.TOP_DOWN,
                                                     centre_x, centre_y, w2b_threshold_x, w2b_threshold_y)

        if ring_outer_edge is None:
            reason = 'no outer edge'
        elif ring_inner_edge is None:
            reason = 'no inner edge'
        else:
            reason = None

        if self.debug:
            if ring_inner_edge is not None:
                plot = self._draw_plots(v_w2b_edge, [[0, ring_inner_edge]], None, (0, 255, 0))
                plot.unload(self.original.source, '{:.0f}x{:.0f}y-inner'.format(centre_x, centre_y))

            if ring_outer_edge is not None:
                plot = self._draw_plots(v_b2w_edge, [[0, ring_outer_edge]], None, (0, 0, 255))
                plot.unload(self.original.source, '{:.0f}x{:.0f}y-outer'.format(centre_x, centre_y))

            plot = target
            if ring_inner_edge is not None:
                plot = self._draw_plots(plot, [[0, ring_inner_edge]], None, (0, 255, 0))
            if ring_outer_edge is not None:
                plot = self._draw_plots(plot, [[0, ring_outer_edge]], None, (0, 0, 255))
            plot.unload(self.original.source, '{:.0f}x{:.0f}y-wavy'.format(centre_x, centre_y))

        return ring_inner_edge, ring_outer_edge, reason

    def _perspective(self, target, centre_x, centre_y):
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
            the returned image is just enough to contain the (reduced) image pixels,
            it consists of: white, black, data 1..N, timing, black rings,
            the (estimated) leading edge and width (in y) of each ring are also returned
            """

        # find our marker edges in the radius
        ring_inner_edge, ring_outer_edge, reason = self._find_extent(target, centre_x, centre_y)
        if reason is not None:
            return None, reason

        # get the edge and distance limits we need
        max_x, max_y = target.size()
        min_inner_edge = max_y
        max_distance = 0
        for x in range(max_x):
            inner_edge = ring_inner_edge[x]
            outer_edge = ring_outer_edge[x]
            if inner_edge is None or outer_edge is None:
                if self.debug:
                    print('{:.0f}x{:.0f}y ring_inner_edge[{}]={} or ring_outer_edge[{}]={} is None!'.
                          format(centre_x, centre_y, x, inner_edge, x, outer_edge))
                continue
            distance = outer_edge - inner_edge
            if distance > max_distance:
                max_distance = distance
            if inner_edge < min_inner_edge:
                min_inner_edge = inner_edge

        # create radius scaling vector for every angle based on the maximum inner to outer distance
        # this creates as a large an image as possible in the radius direction
        radius_scale = [None for _ in range(max_x)]
        for x in range(max_x):
            inner_edge = ring_inner_edge[x]
            outer_edge = ring_outer_edge[x]
            if inner_edge is None or outer_edge is None:
                # this has already been reported (see above)
                continue
            distance = outer_edge - inner_edge
            if distance < self.min_inner_outer_span:
                # we're looking at junk
                return None, 'inner to outer edge too small {}, limit {}'.format(distance,self.min_inner_outer_span)
            radius_scale[x] = max_distance / distance

        # calculate estimated ring width as the scaled distance divided by number of rings in that distance
        # max_distance is that between the inner white/black edge and the outer black/white edge, that spans
        # the black, data 1..N, timing, black rings, the two inner white and one outer white are not included
        one_ring = max_distance / (self.NUM_RINGS - 3)

        # create a new image scaled as appropriate
        code = self.original.instance().new(max_x, max_y, mid_luminance)
        max_new_y = 0                    # will be max radius of the image after perspective corrections
        min_new_y = max_y                # will be min radius of the image after perspective corrections
        for x in range(max_x):
            edge_y = ring_inner_edge[x]
            stop_y = ring_outer_edge[x]
            if edge_y is None or stop_y is None:
                # this has already been reported as a screw up, so ignore this angle
                continue
            min_y_limit = max(edge_y - one_ring, 0)      # only include one white ring in final image
            start_y = max(min_y_limit - one_ring, 0)     # include both the inner white rings for probing
            y_scale = radius_scale[x]
            last_y = int(round(start_y - 1))
            for y in range(int(round(start_y)), int(round(stop_y+1))):
                if y < min_inner_edge:
                    # we're inside the inner white rings, do not scale these
                    span = y - start_y
                    new_y = int(round((span * y_scale) + start_y))
                else:
                    # scale this one
                    span = y - edge_y
                    new_y = int(round((span * y_scale) + min_inner_edge))
                pixel = target.getpixel(x, y)
                if pixel is None:
                    # this means the image has narrow white bands and we've gone off the top
                    continue
                while last_y < new_y:
                    last_y += 1
                    code.putpixel(x, last_y, pixel)
                if new_y < min_y_limit:
                    # we're in the inner white ring, do not count these
                    pass
                elif new_y < min_new_y:
                    min_new_y = new_y
                if new_y > max_new_y:
                    max_new_y = new_y
        # crop the image to the max limits we actually filled
        old_buffer = code.get()
        new_buffer = old_buffer[min_new_y:max_new_y+1, 0:max_x]

        code.set(new_buffer)

        rings = self._find_ring_edges(code, one_ring, centre_x, centre_y)

        # return flattened image
        return (code, rings), None

    def _get_horizontal_edges(self, target, centre_x, centre_y):
        """ get an image of the horizontal edges in the given image,
            the centre_x and centre_y params are purely for diagnostic messages,
            we do this by getting the black to white and white to black edges then merging the result
            """
        b2w_edges = self.transform.edges(target, 0, 1, 3, False)
        w2b_edges = self.transform.edges(target, 0, 1, 3, True)
        all_edges = self.transform.merge(w2b_edges, b2w_edges)
        if self.debug:
            all_edges.unload(self.original.source, '{:.0f}x{:.0f}y-edges'.format(centre_x, centre_y))

        return all_edges

    def _find_ring_edges(self, target, one_ring, centre_x, centre_y):
        """ find the ring edges in the given target,
            the target must have been perspective corrected and cropped,
            estimated_width is an initial guess at where the ring edges might be,
            the centre_x, and centre_y params are only present for naming the debug image files created,
            this deduces the ring edges by finding all horizontal edges, both white to black and black to white,
            in the radius range and building a histogram of edges values, the image given is almost flat so
            there should be distinct edge radii that fall on the ring edges, they should fall into four groups,
            we interpolate across the gaps to yield ring edges for every ring for every angle
            """

        # get edges for every radii
        all_edges = self._get_horizontal_edges(target, centre_x, centre_y)

        threshold_ring = int(round(self._get_edge_threshold(all_edges) * self.edge_threshold_ring))

        # count samples at each radii that are over our edge threshold where there is a run length of at least N
        # the extremities are ignored
        max_x, max_y = all_edges.size()
        samples = [0 for _ in range(max_y)]
        for y in range(self.min_ring_image_gap, max_y - self.min_ring_image_gap):
            run_length = 0
            for x in range(max_x):
                pixel = all_edges.getpixel(x, y)
                if pixel is not None and pixel > threshold_ring:
                    if run_length == 0:
                        # starting a new run
                        run_length = 1
                    else:
                        run_length += 1
                elif run_length > self.min_run_length:
                    samples[y] += run_length
                    run_length = 0
                else:
                    run_length = 0
            # do the run that may have hit the end
            if run_length > self.min_run_length:
                samples[y] += run_length

        # determine the slope at every angle
        slopes = [0 for _ in range(max_y)]
        for y in range(self.min_ring_image_gap, max_y - self.min_ring_image_gap):
            slope = samples[y] - samples[y-1]
            if slope > 0+self.min_slope or slope < 0-self.min_slope:
                # slope is significant enough to count
                slopes[y] = slope
            else:
                # consider this as noise and treat as 0
                slopes[y] = 0

        # find the maxima (slope changes from +ve to -ve), end samples ignored, so range is 1,len-1
        edges = [0]                      # edge at 0 is implied
        plateau_at = 0
        plateau_slope = 0
        for y in range(self.min_ring_image_gap, max_y - self.min_ring_image_gap):
            slope = slopes[y]
            if slope == 0:
                # this means its the same as before, it could be a plateau or the start of a flat maxima,
                # we don't care about a plateau but do about a maxima, so note the y and previous slope for later,
                # we do this 'cos we want the leading edge of any maxima not the trailing edge
                if plateau_at == 0:
                    # this is a new plateau start, note it
                    plateau_at = y
                    plateau_slope = slopes[y-1]
                else:
                    # we're continuing in the same plateau
                    pass
            else:
                # we're not on a plateau, see if we're leaving one
                if plateau_at > 0:
                    # we're leaving a plateau - see if its a maxima
                    if plateau_slope > 0 and slope < 0:
                        # its a maxima, note where it starts
                        edges.append(plateau_at-1)
                    else:
                        # we're continuing in the same direction, so not a maxima
                        pass
                    # no longer in a plateau
                    plateau_at = 0
                    plateau_slope = 0
                else:
                    # we're not leaving a plateau
                    pass
                if slope > 0 and slopes[y+1] < 0:
                    # this is a maxima
                    edges.append(y)
                else:
                    # not a maxima
                    pass

        # calculate the nominal ring leading edges
        nominal = [0]                    # first one always at image edge
        for ring in range(1, self.NUM_RINGS - 2):
            nominal.append(int(round(ring * one_ring)))

        # check if found enough
        while len(edges) < (self.NUM_RINGS - 2):
            # we have not found enough
            # find more edges by looking for the lowest slope between known edges
            # consider the two rings around the widest width to be a candidate for adding
            widest = edges[1]  # first width is to second edge
            widest_ring = 1
            last_y = edges[0]
            for ring in range(1, len(edges)):
                y = edges[ring]
                w = y - last_y
                if w > widest:
                    widest = w
                    widest_ring = ring
                last_y = y
            # last ring width is to the image edge
            w = max_y - edges[-1]
            if w > widest:
                # last ring is the widest
                widest_ring = len(edges)
                start_y = edges[-1] + 1
                stop_y = max_y - 1
            else:
                start_y = edges[widest_ring - 1] + 1
                stop_y = edges[widest_ring] - 1
            # widest_ring is now the ring number of the edge at the end of the widest gap,
            # so we want to insert an edge before that, between these edges we are assuming there
            # is a 'failed' maxima, that will manifest as a minimum, or even zero, slope, we look
            # for the longest and lowest slope between our edges
            plateaus = []
            plateau_start = start_y
            plateau_length = 0
            plateau_distance = 0
            plateau_slope = int(math.fabs(slopes[plateau_start]) / self.min_slope)
            for y in range(start_y, stop_y+1):
                slope = int(round(math.fabs(slopes[y]) / self.min_slope))
                if plateau_length > 0 and slope != plateau_slope:
                    # got a slope change, note previous length and start a new one
                    plateaus.append((plateau_start, plateau_slope, plateau_length, plateau_distance))
                    plateau_start = y
                    plateau_slope = slope
                    plateau_length = 1
                    plateau_distance = min(y - start_y, stop_y - y)
                else:
                    # still in same slope
                    plateau_length += 1
            plateaus.append((plateau_start, plateau_slope, plateau_length, plateau_distance))
            # find smallest with the longest run
            edge = None
            smallest = max_luminance
            longest = 0
            distance = 0
            for plateau in plateaus:
                if plateau[1] < smallest:
                    # found a lower slope, use it
                    edge = plateau[0]
                    smallest = plateau[1]
                    longest =  plateau[2]
                    distance = plateau[3]
                elif plateau[1] == smallest:
                    # found another at smallest
                    if plateau[2] > longest:
                        # this one is longer, use it
                        edge = plateau[0]
                        smallest = plateau[1]
                        longest = plateau[2]
                        distance = plateau[3]
                    elif plateau[2] == longest:
                        # found another with same length
                        if plateau[3] > distance:
                            # this one if further from our gap edges, use it
                            edge = plateau[0]
                            smallest = plateau[1]
                            longest = plateau[2]
                            distance = plateau[3]
                        elif plateau[3] < distance:
                            # this one is closer to our gap edges, so stick with what we got
                            pass
                        else:
                            # two the same, this is only possible if they are adjacent pixels in y
                            # so it does not matter which we use, so just stick with what we got
                            pass
                else:
                    # bigger, ignore it
                    pass
            if edge is not None:
                edges.insert(widest_ring, edge)  # NB: list.insert() does an insert *before* the given index
                if self.debug:
                    print('Blob {:.0f}x{:.0f}y - inserted missing edge before ring {} at {}'.
                          format(centre_x, centre_y, widest_ring, edge))
                # now see if found enough
                continue
            else:
                # eh?
                if self.debug:
                    print('Blob {:.0f}x{:.0f}y - cannot find candidate for missing edge between rings {} and {}'.
                          format(centre_x, centre_y, widest_ring - 1, widest_ring))
                break

        # check if found too many
        while len(edges) > (self.NUM_RINGS - 2):
            # we have found too many
            # filter out the 'worst' ones
            # NB: If we get this wrong the code just will not be recognised, which it would be anyway,
            #     so always safe to drop something
            # consider the 'strength' of an edge to be the number of samples in it
            strengths = [0 for _ in range(len(edges))]
            for ring in range(len(edges)):
                y = edges[ring]
                strengths[ring] = samples[y]
            # consider the two rings around the narrowest width to be candidates for dumping
            narrowest = edges[1]         # first width is to second edge
            narrowest_ring = 1
            last_y = edges[0]
            for ring in range(1, len(edges)):
                y = edges[ring]
                w = y - last_y
                if w < narrowest:
                    narrowest = w
                    narrowest_ring = ring
                last_y = y
            # NB: last ring is never a candidate for dumping, so its width is not determined/tested
            # narrowest_ring is now the ring number of the edge at the end of the narrowest gap
            if strengths[narrowest_ring - 1] > strengths[narrowest_ring]:
                # predecessor is best, so dump this one
                drop = narrowest_ring
            else:
                # this is best, so dro ppredecessor
                drop = narrowest_ring - 1
            if self.debug:
                print('Blob {:.0f}x{:.0f}y - dropping weak ring edge {}, strength {}, width {}'.
                      format(centre_x, centre_y, drop, strengths[drop], narrowest))
            edges.pop(drop)
            # now see if we've still got too many
            continue

        # set leading edge and width of each ring of interest:
        #  one inner white, inner black, data 1..N, timing, outer black
        # the central white and outer white rings are not included (-2 rings)
        # step 1 - set nominal position and size
        if len(edges) == (self.NUM_RINGS - 2):
            # we found the right number of rings - use them
            # edges is the leading edge of each ring, the width is to the next ring
            rings = []
            for ring in range(len(edges)-1):
                y = edges[ring]
                width = edges[ring+1] - y
                rings.append([y, width])
            rings.append([edges[-1], max_y - edges[-1]])   # last ring goes to the image edge
        else:
            # fall back to estimates, this means we failed to make up the differences
            if self.debug:
                print('Blob {:.0f}x{:.0f}y - ring detection failed, falling back to estimates'.
                      format(centre_x, centre_y))
            rings = [[0, int(round(one_ring))]]  # first one starts at image edge
            for ring in range(1, self.NUM_RINGS - 2):
                rings.append([int(round(ring * one_ring)), int(round(one_ring))])
        # step 2 - make adjustments
        for ring in range(len(rings)):
            rings[ring][0] += int(round((rings[ring][1] * self.ring_adjustments[ring][0])))
            rings[ring][1] += int(round((rings[ring][1] * self.ring_adjustments[ring][1])))

        if self.debug:
            same = [0]
            for ring in nominal:
                for edge in edges:
                    if ring == edge:
                        same.append(ring)
                        break
            actual = []
            for ring in range(len(rings)):
                actual.append(rings[ring][0])
            grid = self._draw_grid(target, edges, None, colour=(0, 0, 255))    # red
            grid = self._draw_grid(grid, nominal, None, colour=(255, 0, 0))    # blue
            grid = self._draw_grid(grid, same, None, colour=(0, 255, 0))       # green
            grid = self._draw_grid(grid, actual, None, colour=(255, 255, 0))   # cyan
            grid.unload(self.original.source, '{:.0f}x{:.0f}y-rings'.format(centre_x, centre_y))

        return rings

    def _find_segment_edges(self, target, y, centre_x, centre_y):
        """ find the segment edges in the given timing ring y position in the given target,
            the target must have been perspective corrected,
            the centre_x, and centre_y params are only present for naming the debug image files created,
            this function is only concerned with the bit timing ring, it returns a vector of the angle
            co-ordinate of each bit centre, it calculates these by looking for edges in the timing ring,
            there must be an edge per bit else we're looking at junk, the timing ring has a white to black
            transition in the centre of each bit segment, its assumed any angular distortion is constant
            at these edges so the data bits should lie across these edges, only the best edges are returned,
            if too many are found the weakest are dropped
            """

        # get white to black edges for every angle
        all_edges = self.transform.edges(target, 1, 0, 3)
        max_x, max_y = all_edges.size()

        edge_mean = self._get_edge_threshold(all_edges)
        threshold_x = int(round(edge_mean * self.edge_threshold_x))
        threshold_y = int(round(edge_mean * self.edge_threshold_y))

        if self.debug:
            grid = self._draw_grid(all_edges, [y])

        # get a vector of the edges for the timing ring (NB: the ring wraps at end back to start)
        bit_edges = []
        bit_edge_metrics = []
        x = 0
        while x < max_x:
            down_edge, brightest_down = self._follow_edge_in_y(all_edges, x, y, self.LEFT_TO_RIGHT, self.TOP_DOWN, threshold_x, threshold_y)
            up_edge, brightest_up = self._follow_edge_in_y(all_edges, x, y, self.LEFT_TO_RIGHT, self.BOTTOM_UP, threshold_x, threshold_y)
            edge_length = len(down_edge) + len(up_edge) - 1
            if edge_length < 0:
                # nothing here, move on
                x += 1
                continue
            if edge_length < self.min_timing_length:   # timing edge must be at least this long
                # too small ignore it
                if self.debug:
                    print('Blob {:.0f}x{:.0f}y - timing edge at {} too small ({}..{}) - rejecting'.
                          format(centre_x, centre_y, x, y - len(up_edge) + 1, y + len(down_edge) - 1))
                x += 1
                continue
            # it qualifies, note the median x as the edge
            min_edge_x = max_x
            max_edge_x = 0
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
                edge_start = max_edge_x
            else:
                edge_start = min_edge_x
            edge_span += 1
            edge_x = int(round(min_edge_x + (edge_span / 2)))
            bit_edges.append(edge_x)
            bit_edge_metrics.append([edge_start, edge_length, edge_span, max(brightest_up, brightest_down), None])
            if self.debug:
                up_edge.reverse()
                grid = self._draw_plots(grid, None, [[y, down_edge], [y-(len(up_edge)-1), up_edge]], (0, 255, 0))
            # we move on from this edge
            x = max_edge_x + self.min_edge_thickness

        # we've now found *all* edges, if got too many filter out the worst ones
        while len(bit_edges) > self.size:
            # got too many keep the best ones, each edge is assessed for its robustness based on:
            #   its length - longer is better
            #   its width - thinner is better
            #   its level - brighter is better
            #   its proximity - further from a neighbour is better
            # the weakest one is dropped until we have the number we want
            # edge metrics array is: start-x, length, width, brightness, space for neighbour gap
            # step 1 - find the median separation
            # first separation is the wrap around from the last to the first
            separation = bit_edge_metrics[0][0] + self.angle_steps - bit_edge_metrics[len(bit_edge_metrics) - 1][0]
            bit_edge_metrics[len(bit_edge_metrics) - 1][4] = separation
            for edge in range(len(bit_edge_metrics) - 1):
                gap = (bit_edge_metrics[edge + 1][0] - bit_edge_metrics[edge][0])
                separation += gap
                bit_edge_metrics[edge][4] = gap                        # note how close it is to its next neighbour
            separation /= len(bit_edge_metrics)                        # this is now the mean separation
            too_close = int(round(separation * self.min_timing_gap))   # min neighbour gap tolerated
            # step 2 - find closest neighbours
            smallest_gap = self.angle_steps
            smallest_edge = 0
            for edge in range(len(bit_edge_metrics)):
                if bit_edge_metrics[edge][4] < smallest_gap:
                    smallest_gap = bit_edge_metrics[edge][4]
                    smallest_edge = edge
            if smallest_gap < too_close:
                # this and the next are too close, keep the best, adjust its gap and drop the other
                pred = (smallest_edge + len(bit_edge_metrics) - 1) % len(bit_edge_metrics)
                here = smallest_edge
                succ = (smallest_edge + 1) % len(bit_edge_metrics)
                pred_edge = bit_edge_metrics[pred]
                here_edge = bit_edge_metrics[here]
                succ_edge = bit_edge_metrics[succ]
                if here_edge[3] > succ_edge[3]:
                    # this edge is brighter - so drop next
                    drop = succ
                    adjust = here
                elif here_edge[3] < succ_edge[3]:
                    # next edge is brighter - so drop this
                    drop = here
                    adjust = pred
                else:
                    # same brightness - check width
                    if here_edge[2] < succ_edge[2]:
                        # this edge is thinner - so drop next
                        drop = succ
                        adjust = here
                    elif here_edge[2] > succ_edge[2]:
                        # next edge is thinner - so drop this
                        drop = here
                        adjust = pred
                    else:
                        # same width - check length
                        if here_edge[1] > succ_edge[1]:
                            # this edge is longer - so drop next
                            drop = succ
                            adjust = here
                        elif here_edge[1] < succ_edge[1]:
                            # next is longer - so drop this
                            drop = here
                            adjust = here
                        else:
                            # same length - check neighbour distances
                            if pred_edge[4] > succ_edge[4]:
                                # this is further from its predecessor than next is to its successor - so drop next
                                drop = succ
                                adjust = here
                            if pred_edge[4] < succ_edge[4]:
                                # next is further from its predecessor than this is to its successor - so drop this
                                drop = here
                                adjust = pred
                            else:
                                # nothing to choose between them!
                                drop = here
                                adjust = pred
                # drop now specifies which one to get rid of, and adjust which one to adjust the gap of
                if self.debug:
                    # draw a blue line near/over the edge we are going to dump
                    print('Blob {:.0f}x{:.0f}y - dropping extra segment edge at {},{}'.
                          format(centre_x, centre_y, bit_edges[drop], y))
                    dropped = bit_edge_metrics[drop]
                    sx = dropped[0]
                    sy = int(round(y - (dropped[1] / 2)))
                    ex = sx
                    ey = int(round(y + (dropped[1] / 2)))
                    grid = self._draw_lines(grid, [[sx, sy, ex, ey]], (255, 0, 0))
                bit_edge_metrics[adjust][4] += bit_edge_metrics[drop][4]   # adjust the gap
                bit_edge_metrics.pop(drop)                                 # drop our metrics
                bit_edges.pop(drop)                                        # drop the edge
                continue                                                   # look for another (or not
            else:
                # everything is spaced out sufficiently, give up
                break

        if self.debug:
            grid.unload(self.original.source, '{:.0f}x{:.0f}y-segments'.format(centre_x, centre_y))
            # grid.show()

        # check what we found
        if len(bit_edges) != self.size:
            return None, 'found {} bit edges when expecting {}'.format(len(bit_edges), self.size)

        # we've done it
        return bit_edges, None

    def _find_targets(self):
        """ find targets within our image,
            the detection consists of several steps:
                0. find all the image 'blobs'
                1. detect the central bullseye (as an opencv 'blob', done in __init__)
                2. project the circular target into a rectangle of radius x angle
                3. adjust for perspective distortions (a circle can look like an ellipsis)
                4. adjust for luminance variations
                5. classify the data ring bit segments (black, white or grey)
                6. decode the data bits
            there are validation constraints in most steps that may result in a target being rejected,
            the objective is to achieve 100% confidence in the result or reject it,
            returns a list of target images found
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

            # do the polar to cartesian projection
            projected = self._project(centre_x, centre_y, blob.size)   # this does not fail

            # do the perspective correction
            corrected, reason = self._perspective(projected, centre_x, centre_y)
            if corrected is None:
                # failed - this means some constraint was not met
                if self.debug:
                    print('Blob {:.0f}x{:.0f}y - {} - rejecting'.
                          format(centre_x, centre_y, reason))
                    raw_width = (blob.size / 2) * self.NUM_RINGS   # assume blob is just the 2 inner white rings
                    status.append([centre_x, centre_y, raw_width, reason])
                continue

            # got a corrected image
            perspected = corrected[0]
            rings = corrected[1]

            # work out where our timing ring centre is
            timing_ring_start = rings[self.TIMING_RING][0]
            timing_ring_width = rings[self.TIMING_RING][1]
            timing_ring_centre = int(round(timing_ring_start + (timing_ring_width / 2)))

            if self.debug:
                perspected.unload(self.original.source, '{:.0f}x{:.0f}y-flat'.format(centre_x, centre_y))
                # perspected.show()

            # find the bit segment boundaries
            segments, reason = self._find_segment_edges(perspected, timing_ring_centre, centre_x, centre_y)
            if segments is None:
                # failed - this means some constraint was not met
                if self.debug:
                    print('Blob {:.0f}x{:.0f}y - {} - rejecting'.
                          format(centre_x, centre_y, reason))
                    status.append([centre_x, centre_y, rings[self.TIMING_RING][0], reason])
                continue

            # set the co-ordinates and sizes for our bit samples
            # radius co-ordinate is based on the estimated ring width, its sample size is 1/3rd ring width

            targets.append([centre_x, centre_y, perspected, rings, segments])

        if self.debug:
            # label all the blobs we processed that were rejected
            labels = self.transform.copy(self.image)
            for blob in status:
                x = blob[0]
                y = blob[1]
                k = blob[2]
                reason = blob[3]
                colour = (0, 0, 255)       # red
                labels = self.transform.label(labels, (x, y, k), colour, '{:.0f}x{:.0f}y {}'.format(x, y, reason))
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
            returns a list of x,y blob co-ordinates and the encoded number there, or None if not valid
            """

        targets, labels = self._find_targets()
        if len(targets) == 0:
            if self.debug:
                print('Image {} does not contain any target candidates'.format(self.original.source))
                self.image.unload(self.original.source, 'targets')
            return []

        numbers = []
        for blob in targets:
            centre_x = blob[0]
            centre_y = blob[1]
            image = blob[2]
            rings = blob[3]
            segments = blob[4]

            # calculate ring centre and sample size for each ring
            max_x, _ = image.size()
            # as an expedient the sample width is the same for all as the variation in width is small
            # and we only sample a small part of it
            # ToDo: calculate sample widths for each segment based on distance to its neighbours?
            sample_width = int(round((max_x / self.size) * self.sample_width_factor))
            ring_centres = [None for _ in range(len(rings))]
            sample_size = [None for _ in range(len(rings))]
            for ring in range(len(rings)):
                ring_width = rings[ring][1]
                sample_height = int(round(ring_width * self.sample_height_factor))
                sample_size[ring] = (sample_width, sample_height)
                ring_centres[ring] = rings[ring][0] + int(round(ring_width / 2))

            if self.debug:
                grid = image
                # all except timing have a sample box
                for ring in range(len(rings) - 2):   # -2 to skip timing ring and outer black ring
                    grid = self._draw_grid(grid, [ring_centres[ring]], segments, box=sample_size[ring])
                # timing has no box
                grid = self._draw_grid(grid, [ring_centres[self.TIMING_RING]], None)
                # outer black ring has a sample box
                grid = self._draw_grid(grid, [ring_centres[self.OUTER_BLACK]], segments, box=sample_size[self.OUTER_BLACK])

                grid.unload(self.original.source, '{:.0f}x{:.0f}y-grid'.format(centre_x, centre_y))
                # grid.show()

            # the sample_size box around the intersection of each ring and each segment is what we look at
            # in the perspective corrected image,
            white_level = [None for _ in range(len(segments))]
            black_level = [None for _ in range(len(segments))]
            data_ring_1 = [None for _ in range(len(segments))]
            data_ring_2 = [None for _ in range(len(segments))]
            data_ring_3 = [None for _ in range(len(segments))]
            for bit in range(len(segments)):
                x = segments[bit]
                white_level[bit] = self._get_sample(image, x, ring_centres[self.INNER_WHITE], sample_size[self.INNER_WHITE])
                data_ring_1[bit] = self._get_sample(image, x, ring_centres[self.DATA_RING_1], sample_size[self.DATA_RING_1])
                data_ring_2[bit] = self._get_sample(image, x, ring_centres[self.DATA_RING_2], sample_size[self.DATA_RING_2])
                data_ring_3[bit] = self._get_sample(image, x, ring_centres[self.DATA_RING_3], sample_size[self.DATA_RING_3])
                # we have two rings that contain the black level, both get 'corrupted' by neighbouring white rings
                # bleeding into them, as an expedient we take the lower estimate from both of them as our black level
                black1 = self._get_sample(image, x, ring_centres[self.INNER_BLACK], sample_size[self.INNER_BLACK])
                black2 = self._get_sample(image, x, ring_centres[self.OUTER_BLACK], sample_size[self.OUTER_BLACK])
                black_level[bit] = min(black1, black2)

            # now decode what we got
            number, bits = self.c.unbuild([data_ring_1, data_ring_2, data_ring_3], [white_level, black_level])
            numbers.append([centre_x, centre_y, number, rings[self.TIMING_RING][0]])

            if self.debug:
                print('Blob {:.0f}x{:.0f}y - number:{}, bits:{}'.format(centre_x, centre_y, number, bits))
                if number is None:
                    print('**Blob {:.0f}x{:.0f}y - white samples:{}'.format(centre_x, centre_y, white_level))
                    print('**Blob {:.0f}x{:.0f}y - black samples:{}'.format(centre_x, centre_y, black_level))
                    print('**Blob {:.0f}x{:.0f}y - ring1 samples:{}'.format(centre_x, centre_y, data_ring_1))
                    print('**Blob {:.0f}x{:.0f}y - ring2 samples:{}'.format(centre_x, centre_y, data_ring_2))
                    print('**Blob {:.0f}x{:.0f}y - ring3 samples:{}'.format(centre_x, centre_y, data_ring_3))
                    colour = (0, 0, 255, 0)  # red
                    label = 'invalid code'
                else:
                    colour = (0, 255, 0)     # green
                    label = 'code is {}'.format(number)
                if labels is not None:
                    k = (centre_x, centre_y, rings[self.OUTER_BLACK][0])
                    labels = self.transform.label(labels, k, colour, '{:.0f}x{:.0f}y {}'
                                                  .format(centre_x, centre_y, label))
                    labels.unload(self.original.source, 'targets')

        return numbers

    def _draw_plots(self, source, plots_x=None, plots_y=None, colour=(0, 0, 255)):
        """ draw plots in the given colour, each plot is a set of points and a start x,
            returns a new colour image of the result
            """
        objects = []
        if plots_x is not None:
            for plot in plots_x:
                objects.append({"type": self.transform.PLOTX,
                                "start": plot[0],
                                "points": plot[1]})
        if plots_y is not None:
            for plot in plots_y:
                objects.append({"type": self.transform.PLOTY,
                                "start": plot[0],
                                "points": plot[1]})
        target = self.transform.copy(source)
        return self.transform.annotate(target, objects, colour)

    def _draw_lines(self, source, lines, colour=(0, 0, 255)):
        """ draw lines in given colour,
            lines param is an array of start-x,start-y,end-x,end-y tuples
            """
        objects = []
        for line in lines:
            objects.append({"type": self.transform.LINE,
                            "start": (line[0], line[1]),
                            "end": (line[2], line[3])})
        target = self.transform.copy(source)
        return self.transform.annotate(target, objects, colour)

    def _draw_grid(self, source, horizontal=None, vertical=None, colour=(0, 0, 255), radius=None, box=None):
        """ draw horizontal and vertical lines across the given source image,
            if a radius is given: circles of that radius are drawn at each horizontal/vertical intersection,
            if a box is given: rectangles of that size are drawn centred at each horizontal/vertical intersection,
            lines are drawn in the given colour,
            returns a new colour image of the result
            """
        max_x, max_y = source.size()
        objects = []
        if horizontal is not None:
            for h in horizontal:
                objects.append({"type": self.transform.LINE,
                                "start": (0, h),
                                "end": (max_x-1, h)})

        if vertical is not None:
            for v in vertical:
                objects.append({"type": self.transform.LINE,
                                "start": (v, 0),
                                "end": (v, max_y-1)})

        if horizontal is not None and vertical is not None:
            if radius is not None:
                for h in horizontal:
                    for v in vertical:
                        objects.append({"type": self.transform.CIRCLE,
                                        "centre": (h, v),
                                        "radius": radius})
            if box is not None:
                width  = box[0]
                height = box[1]
                for h in horizontal:
                    for v in vertical:
                        start = (int(round((v - width/2))), int(round((h - height/2))))
                        end   = (int(round((v + width/2))), int(round((h + height/2))))
                        objects.append({"type": self.transform.RECTANGLE,
                                        "start": start,
                                        "end": end})

        target = self.transform.copy(source)
        return self.transform.annotate(target, objects, colour)


class Test:
    """ test the critical primitives """
    def __init__(self, code_bits, min_num, max_num, parity, edges, rings):
        self.code_bits = code_bits
        self.min_num = min_num
        self.c = Codes(self.code_bits, min_num, max_num, parity, edges)
        self.frame = Frame()
        self.max_num = min(max_num, self.c.num_limit)
        self.num_rings = rings
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
                m, bits = self.c.unbuild(samples, levels)
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
            ring = Ring(x >> 1, y >> 1, self.code_bits, width, self.frame)
            ring.code(999, [0x5555, 0xAAAA, 0x5555])
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
                    ring = Ring(x >> 1, y >> 1, self.code_bits, width, self.frame)
                    ring.code(n, rings)
                    self.frame.unload('{}-segment-{}'.format(self.code_bits, n))
        except:
            traceback.print_exc()
        print('******************')

    def scan(self, angles, numbers, image):
        """ do a scan for the code set in image and expect the number given"""
        print('')
        print('******************')
        print('Scan image {} for codes {}'.format(image, numbers))
        try:
            self.frame.load(image)
            scan = Scan(self.c, self.frame, angles, True)
            results = scan.decode_targets()
            found = [False for _ in range(len(numbers))]
            for result in results:
                centre_x = result[0]
                centre_y = result[1]
                num = result[2]
                size = result[3]
                for n in range(len(numbers)):
                    if numbers[n] == num:
                        # found another expected number
                        found[n] = True
                        print('**** Found {} ({:b}) at {:.0f}x, {:.0f}y size {}'.
                              format(num, self.c.encode(num), centre_x, centre_y, size))
                        break
            for n in range(len(numbers)):
                if not found[n]:
                    coded_as = self.c.encode(numbers[n])
                    if coded_as is None:
                        # not a legal code
                        print('**** Failed to find {} (not-valid)'.format(numbers[n]))
                    else:
                        print('**** Failed to find {} ({:b})'.format(numbers[n], self.c.encode(numbers[n])))
        except:
            traceback.print_exc()
        print('******************')


# parameters
min_num = 101                            # min number we want
max_num = 999                            # max number we want (may not be achievable)
code_bits = 15                           # number of bits in our code word
parity = None                            # code word parity to apply (None, 0=even, 1=odd)
edges = 4                                # how many bit transitions we want per code word
rings = 9                                # how many rings are in our code

test_ring_width = 64
test_black = min_luminance + 64 #+ 32
test_white = max_luminance - 64 #- 32
test_noise = mid_luminance >> 1
test_angle_steps = 256

test = Test(code_bits, min_num, max_num, parity, edges, rings)

#test.coding()
#test.decoding(test_black, test_white, test_noise)
#test.circles()
#test_num_set = test.test_set(6)
#test.code_words(test_num_set)
#test.rings(test_ring_width)
#test.codes(test_num_set, test_ring_width)

#test.scan(test_angle_steps, [999], '15-segment-angle-test.png')
#test.scan(test_angle_steps, [999], 'photo-angle-test-flat.jpg')
#test.scan(test_angle_steps, [999], 'photo-angle-test-curved-flat.jpg')

#test.scan(test_angle_steps, [101], '15-segment-101.png')
#test.scan(test_angle_steps, [102], '15-segment-102.png')
#test.scan(test_angle_steps, [365], '15-segment-365.png')
#test.scan(test_angle_steps, [640], '15-segment-640.png')
#test.scan(test_angle_steps, [658], '15-segment-658.png')
#test.scan(test_angle_steps, [828], '15-segment-828.png')
test.scan(test_angle_steps, [101], 'photo-101.jpg')
test.scan(test_angle_steps, [102], 'photo-102.jpg')
test.scan(test_angle_steps, [365], 'photo-365.jpg')
test.scan(test_angle_steps, [640], 'photo-640.jpg')
test.scan(test_angle_steps, [658], 'photo-658.jpg')
test.scan(test_angle_steps, [828], 'photo-828.jpg')
test.scan(test_angle_steps, [102], 'photo-102-distant.jpg')
test.scan(test_angle_steps, [365], 'photo-365-oblique.jpg')
test.scan(test_angle_steps, [658], 'photo-658-small.jpg')
test.scan(test_angle_steps, [658], 'photo-658-crumbled-bright.jpg')
test.scan(test_angle_steps, [658], 'photo-658-crumbled-dim.jpg')
test.scan(test_angle_steps, [658], 'photo-658-crumbled-close.jpg')
test.scan(test_angle_steps, [658], 'photo-658-crumbled-blurred.jpg')
test.scan(test_angle_steps, [658], 'photo-658-crumbled-dark.jpg')
test.scan(test_angle_steps, [101, 102, 365, 640, 658, 828], 'photo-all-test-set.jpg')
