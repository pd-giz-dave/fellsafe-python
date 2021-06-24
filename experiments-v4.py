import cv2
import numpy as np
import random
import math
import traceback

""" coding scheme
    
    This coding scheme is intended to be easy to detect and robust against noise and luminance artifacts.
    The code is designed for use as competitor numbers in club level fell races, as such the number range
    is limited to between 101 and 999 inclusive, i.e. always a 3 digit number and max 899 competitors.
    The actual implementation limit may be less depending on coding constraints chosen (parity, edges, code size).
    The code is circular (so orientation does not matter) and consists of:
        a solid circular centre of 'white' with a radius 2R,
        surrounded by a solid ring of 'black' and width R,
        surrounded by 3 concentric data rings of width R and divided into N (typically 14..16) equal segments,
        enclosed by a solid 'black' ring of width R (this is used to calculate the ring widths when detecting)
    Total radius is 7R.
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
    The central 'bullseye' candidates are detected using a 'blob detector' (via opencv). This provides the radius
    and the luminance levels for a potential target. The potential code rings are then extracted and examined.
    When probing for bits an area is examined that approximates a circle segment but is sized such that small
    errors in the width do not cross a ring boundary. From an area boundary (start position is arbitrary), edges
    are detected using a standard edge detecting convolution (there must be at least 2 for the alignment marker).
    From the code size (which is known) a single edge is sufficient to determine all the bit boundaries in the
    ring (by an angle), and thence the bits are extracted by thresholding. There are two thresholds, min-grey
    and max-grey, which are determined by dividing the luminance range detected in the bullseye by three. The
    bit area sampled is arranged such that there is at least a 2 pixel border that it ignored, the remaining
    pixels are averaged to determine the luminance level.
    The result of the thresholding is 3 levels: black (0), white (1), grey (? - could be either).
    The bit skew between rings is known, so all three rings can be decoded into these three levels. They are
    then decoded as follows:
        three 0's         = 0
        three 1's         = 1
        two zeroes + grey = maybe 0
        two ones + grey   = maybe 1
        anything else     = junk (a give-up condition)
    This results in 5 states for each bit: 0, 1, maybe 0 (0?), maybe 1 (1?), junk (!).
    The ambiguities are (partially) resolved by pattern matching for the start/stop bit pattern (0110).
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

class Octant:
    """ an iterator that returns a series of x,y co-ordinates for an octant of circle of radius r
        it uses the Bresenham algorithm: https://www.geeksforgeeks.org/bresenhams-circle-drawing-algorithm/
        """
    def __init__(self, r):
        # define a circle of radius r
        self.r = r

    def __iter__(self):
        # init and return self
        self.x = 0
        self.y = self.r
        self.d = 3 - 2 * self.r
        return self

    def __next__(self):
        # return next tuple or raise StopIteration
        ret = (self.x, self.y)
        if self.y >= self.x:
            self.x += 1
            if self.d > 0:
                self.y -= 1
                self.d = self.d + 4 * (self.x - self.y) + 10
            else:
                self.d = self.d + 4 * self.x + 6
            return ret
        else:
            raise StopIteration

class Circle:
    """ an iterator that returns a series of x,y co-ordinates for a circle of radius r """
    def __init__(self, r):
        octant1 = []
        octant2 = []
        octant3 = []
        octant4 = []
        octant5 = []
        octant6 = []
        octant7 = []
        octant8 = []
        for dx, dy in Octant(r):
            octant1.append([ dx,  dy])   # forwards
            octant2.append([ dy,  dx])   # backwards
            octant3.append([ dy, -dx])   # forwards
            octant4.append([ dx, -dy])   # backwards
            octant5.append([-dx, -dy])   # forwards
            octant6.append([-dy, -dx])   # backwards
            octant7.append([-dy,  dx])   # forwards
            octant8.append([-dx,  dy])   # backwards
        # remove duplicate edges
        if octant1[-1] == octant2[-1]:  octant1.pop(-1)
        if octant2[ 0] == octant3[ 0]:  octant2.pop( 0)
        if octant3[-1] == octant4[-1]:  octant3.pop(-1)
        if octant4[ 0] == octant5[ 0]:  octant4.pop( 0)
        if octant5[-1] == octant6[-1]:  octant5.pop(-1)
        if octant6[ 0] == octant7[ 0]:  octant6.pop( 0)
        if octant7[-1] == octant8[-1]:  octant7.pop(-1)
        if octant8[ 0] == octant1[ 0]:  octant8.pop( 0)
        # construct circle
        self._circle = []
        for p in          octant1 : self._circle.append(p)
        for p in reversed(octant2): self._circle.append(p)
        for p in          octant3 : self._circle.append(p)
        for p in reversed(octant4): self._circle.append(p)
        for p in          octant5 : self._circle.append(p)
        for p in reversed(octant6): self._circle.append(p)
        for p in          octant7 : self._circle.append(p)
        for p in reversed(octant8): self._circle.append(p)

    def __iter__(self):
        # init and return self
        self._i = -1
        return self

    def __next__(self):
        # return next tuple or raise StopIteration
        self._i += 1
        if self._i >= len(self._circle):
            raise StopIteration
        else:
            return self._circle[self._i][0], self._circle[self._i][1]

class Angle:
    """ a fast mapping (i.e. uses lookup tables and not math functions) from angles to co-ordinates
        and co-ordinates to angles for a circle
        """
    def __init__(self, scale):
        """ build the lookup tables with the resolution required for a single octant, from this octant
            the entire circle can be calculated by rotation and reflection (see angle() and ratio()),
            scale defines the accuracy required, the bigger the more accurate, it must be a +ve integer
            """
        # NB: This code is only executed once so clarity over performance is preferred
        self.ratio_scale = scale
        self.angles = [None for _ in range(self.ratio_scale)]
        self.angles[0] = 0
        for step in range(1,len(self.angles)):
            # each step here represents 1/scale of an octant
            # the index is the ratio of x/y, the result is the angle (in degrees)
            self.angles[step] = math.degrees(math.atan(step / self.ratio_scale))
        self.ratios = [[None, None] for _ in range(self.ratio_scale+1)]
        self.step_angle = 45 / self.ratio_scale  # the angle represented by each step in the lookup table
        for step in range(len(self.ratios)):
            # each octant here consists of scale steps
            # the index is an angle 0..45, the result is the x,y co-ordinates for circle of radius 1
            self.ratios[step][0] = math.sin(math.radians(step * self.step_angle))
            self.ratios[step][1] = math.cos(math.radians(step * self.step_angle))
        # Parameters for ratio() for each octant:
        #   edge angle, offset, 'a' multiplier', reverse x/y, x multiplier, y multiplier
        self.octants = [[45 ,   0,+1,False,+1,+1],   # octant 0
                        [90 , +90,-1,True ,+1,+1],   # octant 1
                        [135, -90,+1,True ,-1,+1],   # octant 2
                        [180,+180,-1,False,+1,-1],   # octant 3
                        [225,-180,+1,False,-1,-1],   # octant 4
                        [270,+270,-1,True ,-1,-1],   # octant 5
                        [315,-270,+1,True ,+1,-1],   # octant 6
                        [360,+360,-1,False,-1,+1]]   # octant 7

    def ratio(self, a, r):
        """ get the x,y co-ordinates on the circumference of a circle of radius 'r' for angle 'a'
            'a' is in degrees (0..360), 'r' is in pixels
              0..45   octant 0 --> [    a    ] =  x/ y ==   +0 + (+a)
             45..90   octant 1 --> [ 90-a    ] =  y/ x ==  +90 + (-a)
             90..135  octant 2 --> [    a-90 ] = -y/ x ==  -90 + (+a)
            135..180  octant 3 --> [180-a    ] =  x/-y == +180 + (-a)
            180..225  octant 4 --> [    a-180] = -x/-y == -180 + (+a)
            225..270  octant 5 --> [270-a    ] = -y/-x == +270 + (-a)
            270..315  octant 6 --> [    a-270] =  y/-x == -270 + (+a)
            315..360  octant 7 --> [360-a    ] = -x/ y == +360 + (-a)
            """
        if a < 0 or a > 360:
            return None, None
        if r == 0:
            return 0, 0
        for octant in self.octants:
            if a < octant[0]:
                ratio = self.ratios[int(((octant[1] + (a * octant[2])) / self.step_angle) + 0.5)]
                if octant[3]:
                    x = ratio[1]
                    y = ratio[0]
                else:
                    x = ratio[0]
                    y = ratio[1]
                x *= octant[4]
                y *= octant[5]
                return int(x * r), int(y * r)
        return None, None

    def angle(self, x, y):
        """ get the angle from these x,y co-ordinates around a circle
                +x, +y, y >  x -->   0..45   octant 0 = 0   + [ x/ y]
                +x, +y, y <  x -->  45..90   octant 1 = 90  - [ y/ x]
                +x, -y, x > -y -->  90..135  octant 2 = 90  + [-y/ x]
                +x, -y, x < -y --> 135..180  octant 3 = 180 - [ x/-y]
                -x, -y, y <  x --> 180..225  octant 4 = 180 + [-x/-y]
                -x, -y, y >  x --> 225..270  octant 5 = 270 - [-y/-x]
                -x, +y, y < -x --> 270..315  octant 6 = 270 + [ y/-x]
                -x, +y, y > -x --> 315..360  octant 7 = 360 - [-x/ y]
            edge cases:
                x = 0, y = 0 --> None        edge 0
                x > 0, y = 0 --> 90          edge 1
                x = 0, y < 0 --> 180         edge 2
                x < 0, y = 0 --> 270         edge 3
                x = 0, y > 0 --> 0 (or 360)  edge 4
        """
        def _ratio2angle(offset, sign, ratio):
            """ do a lookup on the given ratio, changes its sign (+1 or -1) and add the offset (degrees) """
            return offset + (self.angles[int(ratio * self.ratio_scale)] * sign)

        # edge cases
        if x == 0:
            if y == 0: return None       # edge 0
            if y  < 0: return 180        # edge 2
            else:      return 0          # edge 4
        elif y == 0:  # and x != 0
            if x  > 0: return  90        # edge 1
            else:      return 270        # edge 3
        # which octant?
        if x > 0:
            if y > 0:
                if   y >  x: return _ratio2angle(0, +1, x / y)
                elif y <  x: return _ratio2angle(90, -1, y / x)
                else:        return  45
            else:  # y < 0
                if   x > -y: return _ratio2angle(90, +1, -y / x)
                elif x < -y: return _ratio2angle(180, -1, x / -y)
                else:        return 135
        else:  # x < 0
            if y < 0:
                if   y <  x: return _ratio2angle(180, +1, -x / -y)
                elif y >  x: return _ratio2angle(270, -1, -y / -x)
                else:        return 225
            else:  # y > 0
                if   y < -x: return _ratio2angle(270, +1, y / -x)
                elif y > -x: return _ratio2angle(360, -1, -x / y)
                else:        return 315

    def check_angle(self, x, y):
        """ check the angle mapping for co-ords x,y is within 1 resolution step """
        def _atan(x, y):
            """ do the angle determination the math way """
            if x > 0 and y > 0:  # q1
                return math.degrees(math.atan(x / y))
            elif x > 0 and y < 0:  # q2
                return math.degrees(math.atan(-y / x)) + 90
            elif x < 0 and y < 0:  # q3
                return math.degrees(math.atan(-x / -y)) + 180
            elif x < 0 and y > 0:  # q4
                return math.degrees(math.atan(y / -x)) + 270
            elif x == 0 and y > 0:
                return 0
            elif x == 0 and y < 0:
                return 180
            elif y == 0 and x > 0:
                return 90
            elif y == 0 and x < 0:
                return 270
            else:
                raise Exception('Invalid co-ordinates {},{}'.format(x, y))

        a = self.angle(x, y)                     # do our lookup conversion of x,y to an angle
        m = _atan(x, y)                          # do the same thing via atan()
        d = a - m                                # calc the difference
        if math.fabs(d) >= 0.1:
            # got a bad one
            return a, m, d
        else:
            return None, None, None

    def check_ratio(self, a, r):
        """ check the ratio mapping for angles 'a' and radius 'r' is within 1 pixel """
        def _sincos(a, r):
            """ do the x,y determination the math way """
            if a < 0 or a > 360:
                return None, None
            if r == 0:
                return 0, 0
            for octant in self.octants:
                if a < octant[0]:
                    p = math.radians(octant[1] + (a * octant[2]))
                    ratio = [math.sin(p), math.cos(p)]
                    if octant[3]:
                        x = ratio[1]
                        y = ratio[0]
                    else:
                        x = ratio[0]
                        y = ratio[1]
                    x *= octant[4]
                    y *= octant[5]
                    return int(x * r), int(y * r)

        x, y = self.ratio(a, r)                  # do our lookup conversion of a,r to x,y
        mx, my = _sincos(a, r)                   # do the same thing via sin() and cos()
        dx = x - mx                              # calc the difference
        dy = y - my                              # ..
        if (math.fabs(dx) > 0) or (math.fabs(dy) > 0):
            # got a bad one
            return x, y, mx, my, dx, dy
        else:
            return None, None, None, None, None, None

class Ring:
    """ draw a ring of width w and radius r from centre x,y with s segments containing bits b,
        all bits 1 is solid white, all 0 is solid black
        bit 0 (MSB) is drawn first, then 1, etc, up to bit s-1 (LSB), this is big-endian and is
        considered to be clockwise
        """

    def __init__(self, centre_x, centre_y, segments, width, scale, frame):
        # set constant parameters
        self.s = segments                # how many bits in each ring
        self.w = width                   # width of each ring
        self.c = frame                   # where to draw it
        self.x = centre_x                # where the centre of the rings are
        self.y = centre_y                # ..
        # setup our angles look-up table
        self.angle = Angle(scale).angle
        self.edge = 360 / self.s         # the angle at which a bit edge occurs (NB: not an int)

    def _pixel(self, x, y, colour):
        """ draw a pixel at x,y from the image centre with the given luminance """
        x += self.x
        y += self.y
        self.c.putpixel(x  , y  , colour)
        self.c.putpixel(x+1, y  , colour)
        self.c.putpixel(x  , y+1, colour)
        self.c.putpixel(x+1, y+1, colour)

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
        """ draw a ring at radius of bits, if bits is None a grey solid ring is drawn """
        if radius <= 0:
            # special case - just draw a dot at x,y of the LSB colour of bits
            self._point(0, 0, bits & 1)
        else:
            for x, y in Circle(radius):
                a = self.angle(x, y)
                if a > 0:
                    segment = int(a / self.edge)
                else:
                    segment = 0
                mask = 1 << segment
                if bits is None:
                    self._point(x, y, -1)
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
        # draw the outer black ring
        self.draw(6,  0)
        # draw a human readable label
        self.label(number)

class Frame:
    """ image frame buffer as a 2D array of luminance values """

    def __init__(self):
        self.source = None
        self.buffer = None
        self.max_x  = None
        self.max_y  = None

    def instance(self):
        """ return a new instance of self """
        return Frame()

    def new(self, width, height, luminance):
        """ prepare a new buffer of the given size and luminance """
        self.buffer = np.full((height, width), luminance, dtype=np.uint8)  # NB: numpy arrays follow cv2 conventions
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

    def size(self):
        """ return the x,y size of the current frame buffer """
        return self.max_x, self.max_y

    def load(self, image_file):
        """ load frame buffer from a JPEG image file """
        self.buffer = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        self.max_x = self.buffer.shape[1]      # NB: cv2 x, y are reversed
        self.max_y = self.buffer.shape[0]      # ..
        self.source = image_file

    def unload(self, image_file):
        """ unload the frame buffer to a JPEG image file """
        cv2.imwrite(image_file, self.buffer)

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

    def putpixel(self, x, y, value):
        """ put the pixel of value at x,y """
        if x < 0 or x >= self.max_x or y < 0 or y >= self.max_y:
            pass
        else:
            self.buffer[y, x] = min(max(value, min_luminance), max_luminance)  # NB: cv2 x, y are reversed

    def inimage(self, x, y, r):
        """ determine of the points are radius R and centred at X, Y are within the image """
        if (x-r) < 0 or (x+r) >= self.max_x or (y-r) < 0 or (y+r) >= self.max_y:
            return False
        else:
            return True

class Transform:
    """ various image transforming operations """

    def blur(self, source, size=3):
        """ apply a gaussian blur to the given cv2 image with a kernel of the given size """
        target = source.instance()
        target.set(cv2.GaussianBlur(source.get(), (size, size), 0))
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
        params.maxArea = 50000
        params.filterByCircularity = True
        params.minCircularity = 0.8
        params.filterByConvexity = True
        params.minConvexity = 0.9
        params.filterByInertia = True
        params.minInertiaRatio = 0.5

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs (NB: reversing image as we want bright blobs not dark ones)
        return detector.detect(255 - source.get())

    def label(self, source, keypoints, colour=(0, 0, 255), title=None):
        """ return an image with a coloured ring around the given key points in the given image
            and a textual title at each key point centre
            """
        image = source.get()
        if len(image.shape) == 2:
            image = cv2.merge([image, image, image])
        for k in keypoints:
            org = (int(round(k.pt[0])), int(round(k.pt[1])))
            image = cv2.circle(image, org, int(round(k.size / 2)), colour, 1)
            if title is not None:
                image = cv2.putText(image, title, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        target = source.instance()
        target.set(image)
        return target

class Scan:
    """ scan an image looking for codes """

    def __init__(self, frame, debug=False):
        """ frame is the frame instance containing the image to be scanned
            do all the pre-processing here, the pre-processing just isolates
            the areas of interest (by looking for bright blobs)
            """
        # constants
        self.min_border_pixels = 2       # minimum border pixels when sampling rings
        self.min_ring_width = 9          # must be wide enough to have a 2 pixel ignored border and still leave 3
        self.min_black_white_diff = 21   # must be divisible by 3 and still be big enough to be obvious
        self.min_white_ratio = 0.8       # percent pixels that must be white in the central circle
        self.min_black_ratio = 0.6       # percent pixels that must be black in the innermost ring
        self.min_grey_ratio = 0.6        # percent pixels that must be grey in the outermost ring
        self.num_rings = 7               # total number of rings in the whole code (central blob counts as 2)

        # params
        self.original = frame
        self.debug = debug

        # pre-process
        self.transform = Transform()                                             # make a new frame instance
        self.blurred = self.transform.blur(self.original, 3)                     # de-noise
        self.image = self.transform.resize(self.blurred, 1080)                   # re-size
        self.blobs = self.transform.blobs(self.image)                            # find the blobs

        # context
        self.targets = []  # list of potential targets we've found
        self.status = []  # list of blobs and their accepted/rejected status

    def _prepare_ring_scan(self, inner_radius, outer_radius):
        """ setup the params to scan a ring """
        inner_limit = inner_radius * inner_radius
        outer_limit = outer_radius * outer_radius
        return int(round(inner_radius)), int(round(outer_radius)), int(round(inner_limit)), int(round(outer_limit))

    def _luminance(self, centre_x, centre_y, inner_radius, outer_radius):
        """ get the average luminance in the given ring at the given centre """
        level = 0
        pixels = 0
        inner_radius, outer_radius, inner_limit, outer_limit = self._prepare_ring_scan(inner_radius, outer_radius)
        for y in range(-outer_radius, +outer_radius):
            for x in range(-outer_radius, +outer_radius):
                if inner_limit <= (x*x + y*y) <= outer_limit:
                    # we're inside the ring
                    pixel = self.image.getpixel(int(round(centre_x+x)), int(round(centre_y+y)))
                    if pixel is not None:
                        level += pixel
                        pixels += 1
        return int(round(level / pixels))

    def _threshold(self, centre_x, centre_y, inner_radius, outer_radius, min_grey, max_grey):
        """ count pixels in the given ring that are below, within or above the given thresholds """
        black = 0
        grey = 0
        white = 0
        inner_radius, outer_radius, inner_limit, outer_limit = self._prepare_ring_scan(inner_radius, outer_radius)
        for y in range(-outer_radius, +outer_radius):
            for x in range(-outer_radius, +outer_radius):
                if inner_limit <= (x*x + y*y) <= outer_limit:
                    # we're inside the ring
                    pixel = self.image.getpixel(int(round(centre_x+x)), int(round(centre_y+y)))
                    if pixel is not None:
                        if pixel < min_grey:
                            black += 1
                        elif pixel > max_grey:
                            white += 1
                        else:
                            grey += 1
        return black, grey, white

    def find_targets(self):
        """ find target candidates from our image,
            a target is a bright blob surrounded by a uniform darker area with an outer uniform black ring,
            result is an array of potentials in self.targets,
            returns a count of how many found.
            method:
                1. assume found blobs are mostly circular and mostly white, note its width R (keypoint size/4)
                1.1 reject those with a width that is too small
                1.2 reject those that would cross the image edge
                2 get the average luminance for the inner 2/3rds of the blob
                2.1 take this to be white
                3. get the average luminance of the middle third of the ring at radius R for a width R
                3.1 take this to be black
                3.2 reject the target if black is not sufficiently dimmer than white
                4. determine the grey thresholds from the black and white levels
                4.1 apply to central blob - reject target if not enough are white
                4.2 apply to black ring at R - reject target if not enough are black
                4.3 apply to outer black ring - reject target if not enough are black
                5. the target qualifies as a candidate
            """
        self.targets = []
        self.status = []
        for blob in self.blobs:
            centre_x = blob.pt[0]
            centre_y = blob.pt[1]
            width = (blob.size - self.min_border_pixels) / 4   # white bleeds into black so blob detector comes up big
            border = max(width / 3, self.min_border_pixels)
            if width < self.min_ring_width:
                # too small - ignore it
                if self.debug:
                    print('Rejecting blob at {:4.1f}, {:4.1f}, radius {:4.1f}, too small'.
                          format(centre_x, centre_y, width))
                    self.status.append([blob, 'too small', width])
                continue
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
            if self.debug:
                print('**** Accepting blob at {:4.1f}, {:4.1f}, radius {:4.1f}, min_grey={}, max_grey={}, white={:4.1f}%, black={:4.1f}%'.
                      format(centre_x, centre_y, width, min_grey, max_grey, white_ratio*100, black_ratio*100))
                self.status.append([blob, None, width])
            self.targets.append([centre_x, centre_y, width, min_grey, max_grey])
        if self.debug:
            # label all the blobs we processed and draw their rings
            self.labels = self.transform.copy(self.image)
            for blob in self.status:
                k = blob[0]
                reason = blob[1]
                width = blob[2] * 2
                if reason is not None:
                    # got a reject
                    colour = (0, 0, 255)       # red
                else:
                    # got a good'un
                    reason = 'potential target'
                    colour = (0, 255, 0)       # green
                self.labels = self.transform.label(self.labels, [k], colour, reason)
                rings = []
                rings.append(cv2.KeyPoint(k.pt[0], k.pt[1], width * 2))
                rings.append(cv2.KeyPoint(k.pt[0], k.pt[1], width * 3))
                rings.append(cv2.KeyPoint(k.pt[0], k.pt[1], width * 4))
                rings.append(cv2.KeyPoint(k.pt[0], k.pt[1], width * 5))
                rings.append(cv2.KeyPoint(k.pt[0], k.pt[1], width * 6))
                self.labels = self.transform.label(self.labels, rings, colour)
            self.labels.unload('targets-'+self.original.source)
            self.labels.show('targets')
        return len(self.targets)

    def find_codes(self):
        """ extract potential code targets from our image,
            we do this on an angle and radius basis, this creates a rectangle with radius as one axis (x)
            and angle as the other (y), its then easy to do edge detection in that to find our alignment
            pattern, we do this on the outer ring as that has more pixels and will be more accurate, the
            radius resolution is one pixel, the angle resolution is dependant on the radius, it is chosen
            such that every pixel in the outer ring will be sample at least once, this means the inner
            rings will be over-sampled, that is considered benign
            """

        rings = [[], [], []]


class Test:
    """ test the critical primitives """
    def __init__(self, code_bits, min_num, max_num, ratio_scale, parity, edges):
        self.code_bits = code_bits
        self.min_num = min_num
        self.ratio_scale = ratio_scale
        self.c = Codes(self.code_bits, min_num, max_num, parity, edges)
        self.frame = Frame()
        self.max_num = min(max_num, self.c.num_limit)
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

    def angles(self, circle_size):
        """ test accuracy of angle lookup table """
        print('')
        print('******************')
        print('Check angle look-up')
        try:
            i = 0
            good = 0
            bad = 0
            check = Angle(self.ratio_scale).check_angle
            for x,y in Circle(circle_size):
                a, m, d = check(x, y)
                if a is not None:
                    bad += 1
                    print('{:+03},{:+03}: lookup={:+6.1f}  atan={:+6.1f}  err={:+8.3f}        '.format(x, y, a, m, d), end='')
                    i += 1
                    if i % 5 == 0:
                        print('')
                        i = 0
                else:
                    good += 1
            if bad > 0: print('')
            print('{} good, {} bad'.format(good, bad))
        except:
            traceback.print_exc()
        print('******************')

    def ratios(self, circle_size):
        """ test accuracy of ratio lookup table """
        print('')
        print('******************')
        print('Check ratio look-up')
        try:
            r = circle_size >> 1
            i = 0
            good = 0
            bad = 0
            scale = 10
            check = Angle(self.ratio_scale).check_ratio
            for a in range(360 * scale):
                x, y, mx, my, dx, dy = check(a/scale, r)
                if x is not None:
                    bad += 1
                    print('{:+7.2f},{:+03}: lookup=({:+03},{:+03})  sincos=({:+03},{:+03})  err=({:+03},{:+03})        '.\
                          format(a/scale, r, x, y, mx, my, dx, dy), end='')
                    i += 1
                    if i % 4 == 0:
                        print('')
                        i = 0
                else:
                    good += 1
            if bad > 0: print('')
            print('{} good, {} bad'.format(good, bad))
        except:
            traceback.print_exc()
        print('******************')

    def rings(self, width):
        """ draw angle test ring segments (visual) """
        print('')
        print('******************')
        print('Draw an angle test ring (visual)')
        try:
            self.frame.new(width * 14, width * 14, max_luminance)      # 14 == 6 rings in radius + a border
            x, y = self.frame.size()
            ring = Ring(x >> 1, y >> 1, self.code_bits, width, self.ratio_scale, self.frame)
            ring.code(999, [0x5555, 0xAAAA, 0x5555])
            self.frame.unload('{}-segment-angle-test.jpg'.format(self.code_bits))
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
                    self.frame.new(width * 14, width * 14, max_luminance)  # 14 == 6 rings in radius + a border
                    x, y = self.frame.size()
                    ring = Ring(x >> 1, y >> 1, self.code_bits, width, self.ratio_scale, self.frame)
                    ring.code(n, rings)
                    self.frame.unload('{}-segment-{}.jpg'.format(self.code_bits, n))
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
            scan = Scan(self.frame, True)
            scan.find_targets()
            raise Exception('not yet')
        except:
            traceback.print_exc()
        print('******************')


# parameters
min_num = 101                            # min number we want
max_num = 999                            # max number we want (may not be achievable)
code_bits = 15                           # number of bits in our code word
ratio_scale = 3600                       # scale factor for ratio and angle calculations (good for 0.1 degree)
parity = None                            # code word parity to apply (None, 0=even, 1=odd)
edges = 4                                # how many bit transitions we want per code word

test_circle_size = 120
test_ring_width = 64
test_black = min_luminance + 64 #+ 32
test_white = max_luminance - 64 #- 32
test_noise = mid_luminance >> 1

test = Test(code_bits, min_num, max_num, ratio_scale, parity, edges)
#test.coding()
#test.decoding(test_black, test_white, test_noise)
#test.angles(test_circle_size)
#test.ratios(test_circle_size)
test_num_set = test.test_set(6)
#test.code_words(test_num_set)
#test.rings(test_ring_width)
#test.codes(test_num_set, test_ring_width)
#test.scan(101, '15-segment-101.jpg')
#test.scan(101, 'photo-101.jpg')
test.scan(101, 'photo-all-test-set.jpg')
