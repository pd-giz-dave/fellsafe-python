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

    def __init__(self, size, min_num, max_num, parity, edges):
        """ create the valid code set for a number in the range min_num..max_num for code_size
            a valid code is one where there are no embedded start/stop bits bits but contains at least one 1 bit,
            we create two arrays - one maps a number to its code and the other maps a code to its number.
            """
        # blob value categories
        self.black = 0
        self.white = 1
        self.grey = 2

        # bit value categories
        self.is_zero = 0
        self.is_one = 1
        self.maybe_zero = 2
        self.maybe_one = 3
        self.is_neither = 4

        # params
        self.size = size                                       # total ring code size in bits
        self.min_num = min_num                                 # minimum number we want to be able to encode
        self.max_num = max_num                                 # maximum number we want to be able to encode
        self.parity = parity                                   # None, 0 (even) or 1 (odd)
        self.edges = edges                                     # how many bit transitions we want per code
        self.skew = max(int(self.size / 3),1)                  # ring to ring skew in bits
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

        # thresholds (set by set_thresholds)
        self.grey_min = None                                   # luminance below this is considered 'black'
        self.grey_max = None                                   # luminance above this is considered 'white'

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

    def unbuild(self, samples):
        """ given an array of 3 code-word rings with random alignment return the encoded number or None
            each ring must be given as an array of blob values in bit number order
            """
        # step 1 - decode the 3 rings bits (present them to bit())
        bits = [None for _ in range(self.size)]
        for n in range(self.size):
            rings = self.ring_bits_pos(n)
            bits[n] = self.blob(samples[0][rings[0]], samples[1][rings[1]], samples[2][rings[2]])
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
                        return None, bits
                    found = code         # note the first one we find
            if found is not None:
                # only got 1 from this maybe level, go with it
                return found, bits
        # no candidates qualify
        return None, bits

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
        if b1 == self.is_zero:
            pass                         # exact match
        elif b1 == self.maybe_zero:
            maybes += 1                  # maybe match
        else:
            return None                  # not a marker
        if b2 == self.is_one:
            pass                         # exact match
        elif b2 == self.maybe_one:
            maybes += 1                  # maybe match
        else:
            return None                  # not a match
        if b3 == self.is_one:
            pass                         # exact match
        elif b3 == self.maybe_one:
            maybes += 1                  # maybe match
        else:
            return None                  # not a match
        if b4 == self.is_zero:
            pass                         # exact match
        elif b4 == self.maybe_zero:
            maybes += 1                  # maybe match
        else:
            return None                  # not a match
        return maybes

    def data_bits(self, n, bits):
        """ return an array of the data-bits from bits array starting at bit position n """
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
            if (val == self.is_one) or (val == self.maybe_one):
                code += 1                # got a one bit
            elif (val == self.is_zero) or (val == self.maybe_zero):
                pass                     # got a zero bit
            else:
                return None              # got junk
        return self.decode(code)

    def check(self, num):
        """ check encode/decode is symetric
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

    def bit(self, s1, s2, s3):
        """ given 3 blob values determine the most likely bit value
            the blobs are designated as 'black', 'white' or 'grey'
            the return bit is one of is_zero, is_one, maybe_zero, maybe_one, or is_neither
            the middle sample (s2) is expected to be inverted (i.e. black is considered as white and visa versa)
            """
        zeroes = 0
        ones   = 0
        greys  = 0
        # count states
        if s1 == self.grey:
            greys += 1
        elif s1 == self.black:
            zeroes += 1
        elif s1 == self.white:
            ones += 1
        if s2 == self.grey:
            greys += 1
        elif s2 == self.black:
            ones += 1                    # s2 is inverted
        elif s2 == self.white:
            zeroes += 1                  # s2 is inverted
        if s3 == self.grey:
            greys += 1
        elif s3 == self.black:
            zeroes += 1
        elif s3 == self.white:
            ones += 1
        # test definite cases
        if zeroes == 3:
            return self.is_zero
        elif ones == 3:
            return self.is_one
        # test maybe cases
        if zeroes == 2 and greys == 1:
            return self.maybe_zero
        elif ones == 2 and greys == 1:
            return self.maybe_one
        # the rest are junk
        return self. is_neither

    def blob(self, s1, s2, s3):
        """ given 3 luminance samples determine the most likely blob value
            each sample is checked against the grey threshold to determine if its black, grey or white
            then decoded as a bit
            """
        return self.bit(self.category(s1), self.category(s2), self.category(s3))

    def category(self, level):
        """ given a luminance level categorize it as black, white or grey """
        if self.grey_max is None or self.grey_min is None:
            # we haven't been given the thresholds, treat all as grey
            return self.grey
        if level is None:
            return self.grey
        elif level < self.grey_min:
            return self.black
        elif level > self.grey_max:
            return self.white
        else:
            return self.grey

    def set_thresholds(self, min_grey, max_grey):
        """ set the grey thresholds (for use by category) """
        self.grey_min = min_grey
        self.grey_max = max_grey

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

    def unload(self, image_file):
        """ unload the frame buffer to a PNG image file """
        if len(self.buffer.shape) == 2:
            # its a grey scale image, convert to RGBA
            image = self.colourize()
        else:
            # assume its already colour
            image = self.buffer
        filename, ext = os.path.splitext(image_file)
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
    PLOT = 3
    TEXT = 4

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
            the aspect ration is preserved
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

    def copy(self, source):
        """ make a copy of the given image """
        target = source.instance()
        target.set(source.get().copy())
        return target

    def blobs(self, source):
        """ find bright blobs in the given image,
            returns a keypoints array, each keypoint has:
                .pt[1] = x co-ord of the centre
                .pt[0] = y co-ord of centre
                .size = diameter of blob
            all floats
            """

        # Setup SimpleBlobDetector parameters.
        # These have been tuned for detecting circular blobs of a certain size range
        params = cv2.SimpleBlobDetector_Params()
        # 20/06/21 DCN: These parameters have been tuned heuristically by experimentation on a few images
        params.minThreshold = 0
        params.maxThreshold = 255
        params.thresholdStep = 8
        params.filterByArea = True
        params.minArea = 500
        params.maxArea = 500000
        params.filterByCircularity = True
        params.minCircularity = 0.8
        params.filterByConvexity = False
        params.minConvexity = 0.9
        params.filterByInertia = False
        params.minInertiaRatio = 0.5

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs (NB: reversing image as we want bright blobs not dark ones)
        return detector.detect(255 - source.get())

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

    def label(self, source, keypoints, colour=(0, 0, 255), title=None):
        """ return an image with a coloured ring around the given key points in the given image
            and a textual title at each key point centre
            """
        objects = []
        for k in keypoints:
            org = (int(round(k.pt[0])), int(round(k.pt[1])))
            objects.append({"type": self.CIRCLE,
                            "centre": org,
                            "radius": int(round(k.size / 2))})
            if title is not None:
                objects.append({"type": self.TEXT,
                                "start": org,
                                "size": 0.5,
                                "text": title})
        return self.annotate(source, objects, colour)

    def annotate(self, source, objects, colour=(0, 0, 255)):
        """ annotate an image with 'objects' (an array) in the given colour, each object is a dictionary of:
                type, - line, circle, rectangle, plot, text
                start or centre position (x,y) or first plot x,
                end position (x,y) or radius or plot y points or font-size,
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
            elif obj["type"] == self.PLOT:
                x = obj["start"]
                for pt in range(len(obj["points"])):
                    image[obj["points"][pt], x+pt] = colour
            elif obj["type"] == self.TEXT:
                image = cv2.putText(image, obj["text"], obj["start"], cv2.FONT_HERSHEY_SIMPLEX, obj["size"], colour, 1, cv2.LINE_AA)
            else:
                raise Exception('Unknown object type {}'.format(obj["type"]))
        source.set(image)
        return source

class Scan:
    """ scan an image looking for codes """

    def __init__(self, size, frame, debug=False):
        """ frame is the frame instance containing the image to be scanned
            do all the pre-processing here, the pre-processing just isolates
            the areas of interest (by looking for bright blobs)
            """
        # constants
        self.min_border_pixels = 1       # minimum border pixels when sampling an area
        self.min_ring_width = 1 + 3 + 1  # must be wide enough to have an ignored border and still leave enough
        self.min_timing_width = 3        # min width of a timing half-bit to be valid
        self.min_edge_neighbours = 4     # minimum required neighbours when following bit timing edges
        self.min_black_white_diff = 21   # must be divisible by 3 and still be big enough to be obvious
        self.min_white_ratio = 0.8       # percent pixels that must be white in the central circle (ring 0)
        self.min_black_ratio = 0.6       # percent pixels that must be black in the central border (ring 2)
        self.num_rings = 9               # total number of rings in the whole code (central blob counts as 2)
        self.radius_stretch = 1.5        # how much to stretch the radius tested to cope with perspective distortion
        self.angle_steps = 360           # angular resolution when 'projecting'
        self.min_distance_scale = 1      # the minimum inner to outer edge distance as a proportion of blob width
        self.min_inner_edge_scale = 0.3  # the minimum inner edge distance as a proportion of blob width
        self.edge_threshold = int(max_luminance * 0.4)  # minimum luminance for an edge pixel to qualify as an edge
        self.circle_scale = 0.9          # circle radius scale when calculating differentials from an ellipse

        # params
        self.original = frame
        self.debug = debug
        self.size = size                                       # total ring code size in bits

        # pre-process
        self.transform = Transform()                           # make a new frame instance
        blurred = self.transform.blur(self.original)           # de-noise
        self.image = self.transform.resize(blurred, 1080)      # re-size (to HD video convention)
        self.blobs = self.transform.blobs(self.image)          # find the blobs

        # context
        max_radius, _ = self.image.size()
        max_circumference = min(2 * math.pi * max_radius, 3600)
        self.angle = Angle(int(round(max_circumference)))
        self.angle_xy = self.angle.polarToCart
        self.arc_len = self.angle.arcLength
        self.targets = []                # list of targets we've found
        self.status = []                 # list of blobs and their accepted/rejected status

        # semi-circle of samples probed when looking for horizontal edges (see _find_brightest_neighbour)
        # we sample in a semi-circle around each pixel over the threshold to find the best candidate to follow
        self.broad_edge_kernel = [[0, -2],
                                  [0, -1], [1, -1],
                                  [0,  0], [1,  0], [2, 0],
                                  [0, +1], [1,  1],
                                  [0, +2]]
        self.tight_edge_kernel = [[0, -1], [1, -1],
                                  [0,  0], [1,  0],
                                  [0, +1], [1,  1]]

        # addresses of neighbours for checking timing edges
        self.timing_edge_neighbours = [[-1, -1], [ 0, -1], [+1, -1],
                                       [-1,  0], [ 0,  0], [+1,  0],
                                       [-1, +1], [ 0, +1], [+1, +1]]

    def _luminance(self, image, start_x, start_y, end_x, end_y):
        """ get the average luminance in the given image rectangle """
        level = 0
        pixels = 0
        for y in range(int(round(end_y - start_y))):
            for x in range(int(round(end_x - start_x))):
                pixel = image.getpixel(x, y)
                if pixel is not None:
                    level += pixel
                    pixels += 1
        return int(round(level / pixels))

    def _threshold(self, image, start_x, start_y, end_x, end_y, min_grey, max_grey):
        """ count pixels in the given image rectangle that are below, within or above the given thresholds """
        black = 0
        grey = 0
        white = 0
        for y in range(int(round(end_y - start_y))):
            for x in range(int(round(end_x - start_x))):
                pixel = image.getpixel(x, y)
                if pixel is not None:
                    if pixel < min_grey:
                        black += 1
                    elif pixel > max_grey:
                        white += 1
                    else:
                        grey += 1
        return black, grey, white

    def _project(self, centre_x, centre_y, width):
        """ 'project' a potential target from its circular shape to a rectangle of radius (y) x angle (x),
            we do a gaussian mean on the sampled pixels with its N/S/W/E neighbours of:
                - 1 -
                1 2 1
                - 1 -
            """
        # ToDo: make this independent of width - i.e. auto detect it
        angle_delta = 360 / self.angle_steps
        limit_radius = int(round((self.num_rings * self.radius_stretch) * width))
        code = self.original.instance().new(self.angle_steps, limit_radius, min_luminance)
        for radius in range(limit_radius):
            for angle in range(self.angle_steps):
                degrees = angle * angle_delta
                x, y = self.angle_xy(degrees, radius)
                if x is None:
                    pixel = mid_luminance
                else:
                    dx = int(round(centre_x + x))
                    dy = int(round(centre_y + y))
                    c = self.image.getpixel(dx, dy)
                    n = self.image.getpixel(dx, dy-1)
                    s = self.image.getpixel(dx, dy+1)
                    w = self.image.getpixel(dx - 1, dy)
                    e = self.image.getpixel(dx+1, dy)
                    pixel = 0
                    count = 0
                    if c is not None:
                        pixel += c * 2
                        count += 2
                    if n is not None:
                        pixel += n
                        count += 1
                    if s is not None:
                        pixel += s
                        count += 1
                    if w is not None:
                        pixel += w
                        count += 1
                    if e is not None:
                        pixel += e
                        count += 1
                    if count > 0:
                        pixel = int(round(pixel / count))
                    else:
                        pixel = mid_luminance
                code.putpixel(angle, radius, pixel)
        return code

    def _smooth_edge(self, edge):
        """ smooth the given edge vector by doing a mean across N pixels
            """
        extent = len(edge)
        for x in range(extent):
            v = 0  # value accumulator
            d = 0  # divisor accumulator
            for dx, f in [[-2, 1], [-1, 1], [0, 1], [+1, 1], [+2, 1]]:
                sample = edge[(x + dx) % extent]
                if sample is None:
                    # eh?
                    pass
                else:
                    v += int(round(sample * f))
                    d += f
            if d > 0:
                edge[x] = int(round(v / d))
        return edge

    def _find_brightest_neighbour(self, target, x, y, tight=False):
        """ find brightest edge forward (i.e. increasing x) neighbour in target from x, y,
            returns the pixel level and y co-ord of the brightest neighbour that is over our edge threshold,
            returns 0,0 if there is not one,
            """
        brightest = 0
        brightest_dy = 0
        if tight:
            span = self.tight_edge_kernel
        else:
            span = self.broad_edge_kernel
        for dx, dy in span:
            # scan our forward neighbours and follow the brightest that is over our threshold
            pixel = target.getpixel(x + dx, y + dy)
            if pixel is not None and pixel > self.edge_threshold:
                # found an ongoing neighbour also in range, is it the brightest?
                if pixel > brightest:
                    brightest = pixel
                    brightest_dy = dy
        return brightest, y + brightest_dy

    def _find_radius_edge(self, target, direction):
        """ look for a continuous edge in the given target either top-down (inner) or bottom-up (outer),
            direction is +1 for top-down or -1 for bottom-up, an 'edge' is vertical, i.e along the radius,
            to qualify as an 'edge' it must be continuous across all angles (i.e. no dis-connected jumps)
            to find the edge we scan the radius at angle 0 looking for a pixel above our edge threshold,
            then follow that along the angle axis, if we get to 360 without a break we've found it
            """
        max_x, max_y = target.size()
        edge = [None for _ in range(max_x)]
        if direction > 0:
            radius = (y for y in range(max_y))
        else:
            radius = (y for y in range(max_y-1, -1, -1))
        for y in radius:
            pixel = target.getpixel(0, y)
            if pixel is not None and pixel > self.edge_threshold:
                # found a potential edge, follow it
                edge_y = y + 2*direction       # +/-2 to get into the middle of our sample semi-circle
                for x in range(max_x):
                    brightest, brightest_dy = self._find_brightest_neighbour(target, x, edge_y, tight=False)
                    if brightest == 0:
                        # no qualifying neighbors, try next y
                        edge = [None for _ in range(max_x)]  # zap the previous candidate
                        break
                    else:
                        edge_y = brightest_dy
                        edge[x] = edge_y
                if edge[0] is not None:
                    # we found one, smooth it
                    edge = self._smooth_edge(edge)
                    return edge
        # we did not find one
        return None

    def _find_radius_timing_edge(self, target, inner_edge, outer_edge, centre_x, centre_y):
        """ given a vertical black/white edges image and the radius limits for every angle, find the
            'true' outer edge as the outer edge of the timing ring, the centre_x and centre_y params
            are purely for file naming debug images,
            this is the edge defined by the smallest black/white transition from our outer edge
            (looking inwards) within each bit segment, due to longitudinal stretching (in the x direction
            due to perspective effects) we can't just scan fixed chunks of x, so we just look for an edge
            pixel over our threshold for every angle, this will generate a series of segments at various
            distances from the outer edge, we expect to find N that are closest to the outer edge (N being
            the number of bit segments) with N gaps, we just interpolate across these gaps, job done!
            if we don't find N, we're looking at junk and fail (by returning None)
            """
        max_x, max_y = target.size()

        # find all the radii that are over our edge threshold for every angle
        edge = [None for _ in range(max_x)]
        min_edge = max_y
        min_at = None
        max_edge = 0
        max_at = None
        for x in range(max_x):
            start_y = int(round(outer_edge[x] - (self.min_ring_width / 2)))
            end_y = int(round(inner_edge[x] + (self.min_ring_width / 2)))
            for y in range(start_y, end_y, -1):
                pixel = target.getpixel(x, y)
                if pixel is not None:
                    brightest, brightest_at = self._find_brightest_neighbour(target, x, y, tight=True)
                    if brightest == 0:
                        # does not qualify as an edge
                        continue
                    edge[x] = brightest_at
                    if brightest_at < min_edge:
                        min_edge = brightest_at
                        min_at = x
                    if brightest_at > max_edge:
                        max_edge = brightest_at
                        max_at = x
                    break
        if max_at is None or min_at is None:
            return None, 'no timing ring edges detected'

        # remove noise, noise is any 'connected' sequence that is too short or has insufficient neighbours
        x = max_at - 1  # start at max-1 'cos we do +1 immediately inside the loop
        last_edge_y = edge[max_at % max_x]  # initial reference is self
        edge_length = 0
        for _ in range(max_x):
            x = int((x + 1) % max_x)
            c = 0
            for n in self.timing_edge_neighbours:
                dx = (x+n[0]) % max_x
                dy = edge[x]
                if dy is not None:
                    v = target.getpixel(dx, dy+n[1])
                    if v is not None and v > self.edge_threshold:
                        c += 1
            if c < self.min_edge_neighbours:
                # not enough neighbours, chuck it but treat it as a transition
                edge[x] = None
                distance = max_y
                last_edge_y = 0
                width_threshold = 0
            elif edge[x] is None:
                # this means we did not detect an edge, treat this is a transition too
                distance = max_y
                last_edge_y = 0
                width_threshold = 0
            else:
                distance = math.fabs(edge[x] - last_edge_y)
                last_edge_y = edge[x]
                width_threshold = max(int(round(((outer_edge[x] - inner_edge[x]) / (self.num_rings - 3))) * 0.2), 3)
            if distance > width_threshold:
                # this is a ring transition, x is now the first sample of the next 'ring'
                if edge_length < self.min_timing_width:
                    # its too small, zap it
                    for dx in range(edge_length):
                        edge[(x-(dx+1)) % max_x] = None
                edge_length = 1
            else:
                edge_length += 1

        if self.debug:
            plots = []
            gaps = []
            start_x = 0
            points = []
            if edge[0] is None:
                in_gap = True
            else:
                in_gap = False
            for x in range(max_x):
                if in_gap and edge[x] is None:
                    # continue in the gap
                    points.append(0)
                elif in_gap and edge[x] is not None:
                    # coming out of gap, end gap and start a plot
                    gaps.append([start_x, points])
                    start_x = x
                    points = [edge[x]]
                    in_gap = False
                elif edge[x] is None:
                    # coming out of plot, end plot and start a gap
                    plots.append([start_x, points])
                    start_x = x
                    points = [0]
                    in_gap = True
                else:
                    # continue in the plot
                    points.append(edge[x])
            if in_gap:
                gaps.append([start_x, points])
            else:
                plots.append([start_x, points])
            plot = self._draw_plots(target, plots, colour=(0, 255, 0))
            plot = self._draw_plots(plot, gaps, colour=(0, 0, 255))
            plot.unload('target-timing-{:.0f}x{:.0f}y-{}'.format(centre_x, centre_y, self.original.source))
            plot.show()

        # reject edges that are outside the timing ring and count the gaps
        keep = True                            # we know we're starting inside the timing ring
        gap_edge = []                          # x,y (out) and x,y (in) for each gap found
        x = max_at - 1                         # start at max-1 'cos we do +1 immediately inside the loop
        last_edge_y = edge[max_at % max_x]     # initial reference is self
        for _ in range(max_x):
            x = int((x + 1) % max_x)
            width_threshold = max(int(round(((outer_edge[x] - inner_edge[x]) / (self.num_rings - 3))) * 0.2), 3)
            if edge[x] is None:
                # this means there was no edge detected at all before we got to the inner edge or it
                # was rejected with insufficient neighbours or as being too small, just ignore it
                continue
            distance = edge[x] - last_edge_y
            if not keep and 0+distance > width_threshold:
                # this one is a jump outwards, so we're' going from outside to inside the timing ring
                # keep this one and those near unless its too small
                keep = True
                # add gap edge co-ords of where we jump back in
                # (we're updating the record created when we jumped out - see below)
                gap_edge[-1][2] = x
                gap_edge[-1][3] = edge[x]
            elif 0-distance > width_threshold:
                if keep:
                    # this one is a jump inwards, so we're going from inside to outside the timing ring
                    # do not want this one or any near it
                    keep = False
                    # note where it happened
                    gap_edge.append([x, last_edge_y, None, None])
                else:
                    # we're already outside and going further outside, nothing to do here
                    pass
            last_edge_y = edge[x]
            if keep:
                # this one is inside the timing ring, so keep it
                pass
            else:
                # this one is outside the timing ring, so drop it
                edge[x] = None

        if self.debug:
            gaps = []
            for gap in gap_edge:
                gaps.append(gap[0])
            grid = self._draw_grid(target, None, gaps, colour=(0, 255, 0))
            gaps = []
            for gap in gap_edge:
                gaps.append(gap[2])
            grid = self._draw_grid(grid, None, gaps, colour=(0, 0, 255))
            grid.unload('target-gaps-{:.0f}x{:.0f}y-{}'.format(centre_x, centre_y, self.original.source))
            grid.show()

        # interpolate across all the gaps we found, all the required info is in gap_edge
        # NB: a gap may span the angle edges
        for gap in gap_edge:
            x_out = gap[0]
            y_out = gap[1]
            x_in = (gap[2] - 1) % max_x
            y_in = gap[3]
            span = x_in - x_out + 1
            if y_in > y_out:
                # edge is going downwards (i.e. increment is +ve)
                distance = y_in - y_out
                increment = distance / span
            else:
                # edge is going upwards (i.e. increment is -ve)
                distance = y_out - y_in
                increment = 0 - (distance / span)
            y = y_out
            x = (x_out - 1) % max_x
            while x != x_in:
                x = (x + 1) % max_x
                y += increment
                edge[x] = int(round(y))

        # smooth the edge
        edge = self._smooth_edge(edge)

        return edge, None

    def _find_extent(self, target, centre_x, centre_y):
        """ find the inner and outer edges of the given target,
            the inner edge is the white to black transition at min radius,
            the outer edge is timing ring to black transition at max radius,
            the centre_x and centre_y params are only present for naming the debug image files created,
            returns two vectors, inner co-ord and outer co-ord for every angle, or a message if fails
            """

        # find our marker edges in the radius
        v_w2b_edge = self.transform.edges(target, 0, 1, 3, True)   # get white to black edges along the radius
        v_b2w_edge = self.transform.edges(target, 0, 1, 3, False)  # get black to white edges along the radius
        if self.debug:
            v_w2b_edge.unload('target-inner-{:.0f}x{:.0f}y-{}'.format(centre_x, centre_y, self.original.source))
            v_b2w_edge.unload('target-outer-{:.0f}x{:.0f}y-{}'.format(centre_x, centre_y, self.original.source))
            v_w2b_edge.show()
            v_b2w_edge.show()

        # create a vector of the first inner edge radius for every angle (this is usually accurate)
        ring_inner_edge = self._find_radius_edge(v_w2b_edge, +1)
        if ring_inner_edge is None:
            return None, None, 'no inner edge'

        # create a vector of the last outer edge radius for every angle (this can be distorted by curvature)
        ring_outer_edge = self._find_radius_edge(v_b2w_edge, -1)
        if ring_outer_edge is None:
            return None, None, 'no outer edge'

        # create a vector of the 'true' outer edge
        ring_true_edge, reason = self._find_radius_timing_edge(v_w2b_edge,
                                                               ring_inner_edge, ring_outer_edge,
                                                               centre_x, centre_y)
        if ring_true_edge is None:
            return None, None, reason

        if self.debug:
            plot = self._draw_plots(target, [[0, ring_inner_edge]], (0, 255, 0))
            plot = self._draw_plots(plot, [[0, ring_outer_edge]], (0, 0, 255))
            plot = self._draw_plots(plot, [[0, ring_true_edge]], (255, 0, 0))
            plot.unload('target-span-{:.0f}x{:.0f}y-{}'.format(centre_x, centre_y, self.original.source))
            plot.show()

        return ring_inner_edge, ring_true_edge, None

    def _perspective(self, target, centre_x, centre_y):
        """ remove perspective distortions from the given 'projected' image, the centre_x and centre_y params
            are only present for naming the debug image files created,
            a circle when not viewed straight on appears as an ellipse, when that is projected into a rectangle
            the radius edges becomes 'wavy' (a sine wave), this function straightens those wavy edges, other
            distortions can arise if the target is curved (due to it being wrapped around someones leg), in
            this case the outer black ring can appear narrow in some parts of the circle (this does not happen
            with the inner ring because it is inside the code and curvature there would need to be vast to
            make any significant difference),
            the outer black ring is preceded by the bit timing ring, this guarantees a black/white edge within
            every bit segment, we look for that as the 'true' code edge for each bit segment,
            the 'straightening' is then just a matter of scaling such that all angles span the same range from
            the inner edge to the 'true' outer edge,
            the returned image is just enough to contain the (reduced) image pixels
            """

        # find our marker edges in the radius
        ring_inner_edge, ring_outer_edge, reason = self._find_extent(target, centre_x, centre_y)
        if ring_inner_edge is None:
            # failed
            return None, reason

        # get the edge and distance limits we need
        max_x, max_y = target.size()
        min_inner_edge = max_y
        min_outer_edge = max_y
        max_inner_edge = 0
        max_outer_edge = 0
        min_distance = max_y
        max_distance = 0
        for x in range(max_x):
            inner_edge = ring_inner_edge[x]
            outer_edge = ring_outer_edge[x]
            distance = outer_edge - inner_edge
            if distance < min_distance:
                min_distance = distance
            if distance > max_distance:
                max_distance = distance
            if inner_edge < min_inner_edge:
                min_inner_edge = inner_edge
            if inner_edge > max_inner_edge:
                max_inner_edge = inner_edge
            if outer_edge < min_outer_edge:
                min_outer_edge = outer_edge
            if outer_edge > max_outer_edge:
                max_outer_edge = outer_edge

        # create radius scaling vector for every angle based on the maximum inner to outer distance
        # this creates as a large an image as possible in the radius direction
        radius_scale = [None for _ in range(max_x)]
        for x in range(max_x):
            distance = ring_outer_edge[x] - ring_inner_edge[x]
            radius_scale[x] = max_distance / distance

        # calculate our nominal ring width as the scaled distance divided by number of rings in that distance
        one_ring = max_distance / (self.num_rings - 4)   # -4 'cos inner white rings and outer black/white rings not included

        # create a new image scaled as appropriate
        code = self.original.instance().new(max_x, max_y, mid_luminance)
        max_new_y = 0                    # will be max radius of the image after perspective corrections
        min_new_y = max_y                # will be min radius of the image after perspective corrections
        for x in range(max_x):
            start_y = ring_inner_edge[x] - one_ring            # include one of the inner white rings
            stop_y = ring_outer_edge[x] + one_ring             # include the outer black ring
            y_scale = radius_scale[x]
            last_y = int(round(start_y - 1))
            for y in range(max_y):
                if y > stop_y:
                    # we're done when we get to the outer edge (inc the enclosing white ring)
                    break
                if y < start_y:
                    # ignore inner ring (its just used for blob detection)
                    continue
                elif y < min_inner_edge:
                    # we're inside the inner white ring
                    span = y - start_y
                    new_y = int(round((span * y_scale) + start_y))
                else:
                    # scale this one
                    span = y - ring_inner_edge[x]
                    new_y = int(round((span * y_scale) + min_inner_edge))
                pixel = target.getpixel(x, y)
                while last_y < new_y:
                    last_y += 1
                    code.putpixel(x, last_y, pixel)
                if new_y < min_new_y:
                    min_new_y = new_y
                if new_y > max_new_y:
                    max_new_y = new_y

        # crop the image to the max limits we actually filled
        old_buffer = code.get()
        new_buffer = old_buffer[min_new_y:max_new_y+1, 0:max_x]

        code.set(new_buffer)

        # return flattened image
        return (code, min_inner_edge, one_ring), None

    def _find_segment_edges(self, target, timing_ring_pos, centre_x, centre_y):
        """ find the segment edges in the given timing ring position in the given target,
            the target must have been perspective corrected,
            the centre_x, and centre_y  params are only present for naming the debug image files created,
            this function is only concerned with the bit timing ring, it returns a vector of the angle
            co-ordinate of each bit centre, it calculates these by looking for edges in the timing ring,
            there must be an edge per bit else we're looking at junk, the timing ring has a white to black
            transition in the centre of each bit segment, its assumed any angular distortion is constant
            at these edges so the data bits should lie across these edges
            """
        # get white to black edges for every angle
        all_edges = self.transform.edges(target, 1, 0, 3)
        max_x, max_y = all_edges.size()

        # get a vector of the edges for the timing ring (NB: the ring wraps at end back to start)
        bit_edges = []
        y = int(round(timing_ring_pos))        # must be centre of timing ring
        x = 0
        for _ in range(max_x):
            pixel = all_edges.getpixel(x, y)
            if pixel is not None and pixel > self.edge_threshold:
                # got one over our threshold, find the brightest connected until go below threshold
                brightest = pixel
                brightest_at = x
                dx = x
                # scan for the end allowing fpr wrapping around the end
                for _ in range(max_x):
                    dx = (dx + 1) % max_x
                    pixel = all_edges.getpixel(dx, y)
                    if pixel is not None and pixel > self.edge_threshold:
                        if pixel > brightest:
                            brightest = pixel
                            brightest_at = dx
                    else:
                        # gone past the edge
                        break
                # note the brightest we found as an edge
                bit_edges.append(brightest_at)
                if dx < x:
                    # we've wrapped, so done
                    break
                x = dx
            else:
                x += 1
                if x >= max_x:
                    # we're done
                    break
        if self.debug:
            grid = self._draw_grid(all_edges, [int(round(timing_ring_pos))], bit_edges)
            grid.unload(
                'target-segments-{:.0f}x{:.0f}y-{}'.format(centre_x, centre_y, self.original.source))
            grid.show()

        # check what we found
        if len(bit_edges) != self.size:
            return None, 'found {} bit edges when expecting {}'.format(len(bit_edges), self.size)

        # we've done it
        return bit_edges, None

    def _draw_plots(self, source, plots, colour=(0, 0, 255)):
        """ draw plots in the given colour, each plot is a set of points and a start x,
            returns a new colour image of the result
            """
        objects = []
        for plot in plots:
            objects.append({"type": self.transform.PLOT,
                            "start": plot[0],
                            "points": plot[1]})
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

    def find_targets(self):
        """ find targets within our image,
            the detection consists of several steps:
                1. detect the central bullseye (as an opencv 'blob', done in __init__)
                2. project the circular target into a rectangle of radius x angle
                3. adjust for perspective distortions (a circle can look like an ellipsis)
                4. adjust for luminance variations
                5. classify the data ring bit segments (black, white or grey)
                6. decode the data bits
            there are validation constraints in most steps that may result in a target being rejected,
            the objective is to achieve 100% confidence in the result or reject it,
            returns a count of how many found.
            """
        self.targets = []
        self.status = []
        # HACK
        white_ratio = 1
        black_ratio = 1
        min_grey = 128 - 64
        max_grey = 128 + 64
        # ENDHACK
        timing_ring_centre = 3.6         # this should be 3.5 but that tends under-estimate,
                                         # 3.6 works better but no idea why!
        for blob in self.blobs:
            centre_x = blob.pt[0]
            centre_y = blob.pt[1]
            raw_width = (blob.size - self.min_border_pixels) / 4   # white bleeds into black so blob detector comes up big

            # do the polar to cartesian projection
            projected = self._project(centre_x, centre_y, raw_width)     # this does not fail
            if self.debug:
                projected.unload('target-wavy-{:.0f}x{:.0f}y-{}'.format(centre_x, centre_y, self.original.source))
                projected.show()

            # do the perspective correction
            corrected, reason = self._perspective(projected, centre_x, centre_y)
            if corrected is None:
                # failed - this means some constraint was not met
                if self.debug:
                    print('Rejecting blob at {:4.1f}x, {:4.1f}y - {}'.
                          format(centre_x, centre_y, reason))
                    self.status.append([blob, reason, centre_x, centre_y])
                continue
            perspected = corrected[0]
            start_at = corrected[1]
            new_width = corrected[2]
            max_x, max_y = perspected.size()
            if new_width < self.min_ring_width:
                # too small - ignore it
                if self.debug:
                    print('Rejecting blob at {:4.1f}x, {:4.1f}y - too small'.
                          format(centre_x, centre_y))
                    self.status.append([blob, 'too small', centre_x, centre_y])
                continue
            if self.debug:
                perspected.unload('target-flat-{:.0f}x{:.0f}y-{}'.format(centre_x, centre_y, self.original.source))
                perspected.show()

            # find the bit segment boundaries
            segments, reason = self._find_segment_edges(perspected,
                                                        start_at + (new_width * timing_ring_centre), centre_x, centre_y)
            if segments is None:
                # failed - this means some constraint was not met
                if self.debug:
                    print('Rejecting blob at {:4.1f}x, {:4.1f}y - {}'.
                          format(centre_x, centre_y, reason))
                    self.status.append([blob, reason, centre_x, centre_y])
                continue

            # set the co-ordinates and sizes for our bit samples
            # radius co-ordinate is based on the estimated ring width, its sample size is 1/3rd ring width
            data_rings = []
            for ring in range(2, 5):
                data_rings.append(int(round((ring + 0.5) * new_width)))  # plus .5 to get to centre of ring
            # angle co-ordinate is the segments vector, its sample size is 1/3rd the average distance between bits
            sample_size = (int(round((max_x / self.size) / 3)), int(round(new_width / 3)))

            if self.debug:
                grid = self._draw_grid(perspected, data_rings, segments, box=sample_size)
                grid.unload('target-grid-{:.0f}x{:.0f}y-{}'.format(centre_x, centre_y, self.original.source))
                grid.show()

            # we now know where the data rings are and where the bit boundaries are,
            # so we can sample an area inside each bit to get its value, but first we
            # need to know what the black/white levels are, we find those for each bit
            # by sampling an area either side of each bit boundary in the timing ring
            # as well as the inner white and black rings, from that a luminance gradient
            # can be calculated across the three data rings
            # ToDo: determine the black/white gradient across the data rings

            # do the luminance correction - is it required?
            # the luminance varies mostly around the radius but its only across the radius we care about

            # ToDo: do this on a s segment by segment basis, not around the whole ring
            max_x, _ = perspected.size()
            white_level = self._luminance(perspected, 0, 0, max_x, new_width)
            # black_level = self._luminance(perspected, 0, new_width*2, max_x, new_width)
            """
                to qualify as a target the luminance sequence across the radius must be:
                    white, black, ?, ?, ?, black
                """
            """
            rings = [[(width*ring_num)+border, (width*(ring_num+1))-border] for ring_num in range(self.num_rings)]
            if not self.image.inimage(int(round(centre_x)), int(round(centre_y)), int(round(width * self.num_rings))):
                # whole code would go off the image edge
                if self.debug:
                    print('Rejecting blob at {:4.1f}, {:4.1f}, radius {:4.1f}, overflows image edge'.
                          format(centre_x, centre_y, width))
                    self.status.append([blob, 'over edge', width])
                continue
            white_level = self._luminance(centre_x, centre_y, rings[0][0], rings[1][1])
            # get luminance of black inner ring
            black_level = self._luminance(centre_x, centre_y, rings[2][0], rings[2][1])
            # check we have a sufficient luminance gap
            level_range = white_level - black_level
            if level_range < self.min_black_white_diff:
                # luminance levels too close (or even reversed!)
                if self.debug:
                    print('Rejecting blob at {:4.1f}, {:4.1f}, radius {:4.1f}, luminance diff black to white is too small {:4.1f}'.
                          format(centre_x, centre_y, width, level_range))
                    self.status.append([blob, 'black/white diff too small', width])
                continue
            # determine the grey thresholds
            min_grey = black_level + int(round(level_range / 3))
            max_grey = white_level - int(round(level_range / 3))
            # check threshold on central blob
            black, grey, white = self._threshold(centre_x, centre_y, rings[0][0], rings[1][1], min_grey, max_grey)
            white_ratio = (grey + white) / (black + grey + white)
            if white_ratio < self.min_white_ratio:
                # central circle has too many non-white pixels
                if self.debug:
                    print('Rejecting blob at {:4.1f}, {:4.1f}, radius {:4.1f}, too many non-white central pixels {:4.1f}% (b={},g={},w={})'.
                          format(centre_x, centre_y, width, white_ratio*100, black, grey, white))
                    self.status.append([blob, 'centre not white enough', width])
                continue
            # check threshold on inner black ring
            black, grey, white = self._threshold(centre_x, centre_y, rings[2][0], rings[2][1], min_grey, max_grey)
            black_ratio = (black + grey) / (black + grey + white)
            if black_ratio < self.min_black_ratio:
                # inner ring has too many non-black pixels
                if self.debug:
                    print('Rejecting blob at {:4.1f}, {:4.1f}, radius {:4.1f}, too many non-black inner ring pixels {:4.1f}% (b={},g={},w={})'.
                          format(centre_x, centre_y, width, black_ratio*100, black, grey, white))
                    self.status.append([blob, 'inner ring not black enough', width])
                continue
            """
            if self.debug:
                print('**** Accepting blob at {:4.1f}x, {:4.1f}y, min_grey={}, max_grey={}, white={:4.1f}%, black={:4.1f}%'.
                      format(centre_x, centre_y, min_grey, max_grey, white_ratio*100, black_ratio*100))
                self.status.append([blob, None, centre_x, centre_y])
            self.targets.append([centre_x, centre_y, new_width, min_grey, max_grey])
        if self.debug:
            # label all the blobs we processed and draw their rings
            labels = self.transform.copy(self.image)
            for blob in self.status:
                k = blob[0]
                reason = blob[1]
                x = blob[2]
                y = blob[3]
                if reason is not None:
                    # got a reject
                    colour = (0, 0, 255)       # red
                else:
                    # got a good'un
                    reason = 'good target'
                    colour = (0, 255, 0)       # green
                labels = self.transform.label(labels, [k], colour, '{} {:.0f}x{:.0f}y'.format(reason, x, y))
            labels.unload('targets-status-{}'.format(self.original.source))
            labels.show()
        return len(self.targets)


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
            grey = int((white - black) / 3)
            self.c.set_thresholds(min(black + grey, max_luminance), max(white - grey, min_luminance))
            colours = [black, white]
            good = 0
            fail = 0
            bad = 0
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
                m, bits = self.c.unbuild(samples)
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
        """ test code-word rotation with given set plus the extremes(visual) """
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

    def scan(self, n, image):
        """ do a scan for the code in image and expect the number given"""
        print('')
        print('******************')
        print('Scan image {} for code {}'.format(image, n))
        try:
            self.frame.load(image)
            scan = Scan(self.code_bits, self.frame, True)
            scan.find_targets()
            raise Exception('not yet')
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

test = Test(code_bits, min_num, max_num, parity, edges, rings)
#test.coding()
#test.decoding(test_black, test_white, test_noise)
#test.circles()
#test_num_set = test.test_set(6)
#test.code_words(test_num_set)
#test.rings(test_ring_width)
#test.codes(test_num_set, test_ring_width)
#test.scan(999, '15-segment-angle-test.png')
test.scan(101, '15-segment-101.png')
#test.scan(365, '15-segment-365.png')
#test.scan(101, 'photo-101.jpg')
#test.scan(365, 'photo-365-oblique.jpg')
#test.scan(101, 'photo-all-test-set.jpg')
#test.scan(999, 'photo-angle-test-flat.jpg')
#test.scan(999, 'photo-angle-test-curved-flat.jpg')
#test.scan(999, 'photo-angle-test-oblique.jpg')
#test.scan(999, 'photo-angle-test-small-curve.jpg')
#test.scan(999, 'photo-angle-test-big-curve.jpg')
#test.scan(999, 'photo-angle-test-crumbled.jpg')
