from PIL import Image
import random
import math
import traceback

"""
13/06/21 DCN Abandoned 15 segment scheme to use a 16 segment with parity.
             This is because without parity in the presence of lots of noise invalid codes are not detected
             with sufficient robustness, leading to too many incorrect decodes.  
"""
""" coding scheme
    
    This coding scheme is intended to be easy to detect and robust against noise and luminance artifacts.
    The code is designed for use as competitor numbers in club level fell races, as such the number range
    is limited to between 101 and 999 inclusive, i.e. always a 3 digit number and max 899 competitors.
    The actual implementation limits the max competitors to 883, still plenty for our use case.
    The code is circular (so orientation does not matter) and consists of:
        a solid circular centre of 'white' with a radius R,
        surrounded by a solid ring of 'black' and width R,
        surrounded by 3 concentric rings of width R and divided into 15 equal segments (24 degrees each)
    Total radius is 5R.
    The radial segments form 3 bits, which are used as a triple redundant data bit copy.
    Segments around the ring provide for 15 bits. Each ring is skewed clockwise by 5 bits (a kind of interleave).
    A one-bit is white (i.e. high luminance) and a zero-bit is black (i.e. low luminance).
    An alignment marker of 0110 (4 bits) is used to identify the start/stop of the code word encoded in a ring.
    The remaining 11 bits are the payload (big-endian) and must not contain the alignment marker and also must
    not end in 011 and must not start with 110 (else they look like 0110 when adjacent to the alignment marker).
    The 3 payload rings are skewed round by 5 bits so that the alignment marker is not aligned radially. This 
    evens out the luminance levels and also has the effect of interleaving the redundant bits. So the 3 rings
    have bits like this:
        0110abcdefghijk
        ghijk0110abcdef
        bcdefghijk0110a
    The code is detected by searching for the 'bullseye' (central circle and its enclosing ring). The luminance
    in the bullseye must be approximately the same for all its pixels and its surrounding ring must also have an
    approximately even luminance but be significantly dimmer than the central circle. These two luminance levels
    determine the black/white/grey thresholds. The radius of the circle determines the width of subsequent rings.
    The algorithm for detecting the central circle first does a simple quadrant probe out from some point looking
    for a 'sharp' luminance change. If all quadrants change at a similar radius its a potential bullseye. Else
    try the 'next' location. Check all points in the circle have similar luminance. Else try the 'next' location.
    Next check the enclosing ring has consistent luminance all the way around. Else try the 'next' location.
    The black, white, grey luminance levels and the circle radius (R) are now known. 
    Each bit position is examined clockwise (3 * 15 = 45) to determine its colour, which can be black, white or
    grey (i.e. don't know - could be a 0 or 1 to be determined later).
    When probing for bits a 'blob' is examined that approximates a circle segment but is sized such that small
    errors in the width do not cross a ring boundary. From a blob boundary (start position is arbitrary), take
    average of N pixels, slide up 1, repeat until get M the same-ish averages or moved by 2 segments (this is a
    'give-up' condition). That average is then the blob value. The next blob starts 1 segment away from the start
    of this blob.
    Each pixel value is the average of 9 pixels (central one and its 8 neighbours), the pixel value is the average
    of this set. The rings are then de-skewed and the 3 redundant bits merged as follows (0=black, 1=white, ?=grey):
        ??? = either 0 or 1
        ??0 = maybe 0
        ??1 = maybe 1
        ?0? = maybe 0
        ?00 = 0
        ?01 = either 0 or 1
        ?1? = maybe 1
        ?10 = either 0 or 1
        ?11 = 1
        0?? = maybe 0
        0?0 = 0
        0?1 = either 0 or 1
        00? = 0
        000 = 0
        001 = 0
        01? = either 0 or 1
        010 = 0
        011 = 1
        1?? = maybe 1
        1?0 = either 0 or 1
        1?1 = 1
        10? = either 0 or 1
        100 = 0
        101 = 1
        11? = 1
        110 = 1
        111 = 1
    This results in 5 states for each bit: 0, 1, maybe 0 (0?), maybe 1 (1?), could be either (?).
    The ambiguities are (partially) resolved by pattern matching for the start/stop bit (0110).
    A potential alignment marker is sought in this order:
        exact match - 0110 (can only be one in a valid detection)
        single maybe bit - i.e. one of the market bits is a (0?) or (1?)      (could be several candidates)
        double maybe bit - i.e. two of the market bits is a (0?) or (1?)      (could be several candidates)
        triple maybe bit - i.e. three of the market bits is a (0?) or (1?)    (could be several candidates)
        quadruple maybe bit - i.e. four of the market bits is a (0?) or (1?)  (could be several candidates)
    When there are several maybe candidates, each is tried to extract a code word, if more than one succeed
    its a give-up situation as there is too much ambiguity.
    When trying maybe candidates, all other candidates are demoted (by changing (0?) and (1?) to (?)).
    When an alignment candidate has been found, then the potential code-word is extracted.
    When doing that (0?) is treated a 0, (1?) is treated as 1 and every possible combination of (?) as
    0 or 1. If more than one succeed in extracting a valid code-word its another give-up situation.
    When giving up it means re-start looking for the bullseye from a different position. How different that
    position is depends on what was found before giving up.
    
    Note: The decoder can cope with several targets in the same image. It uses the relative size of
          the targets to present the results in distance order, nearest (i.e. biggest) first. This is
          important to determine finish order.
                
    """

# colours
max_luminance = 255
min_luminance = 0
mid_luminance = (max_luminance - min_luminance) >> 1


class Codes:
    """ Encode and decode a number or a bit or a blob
        a number is a payload, it can be encoded and decoded
        a bit is a raw bit decoded from 3 blobs
        a blob is decoded from 3 luminance level samples
        this class encapsulates all the encoding and decoding and their constants
        """

    # blob value categories
    black = 0
    white = 1
    grey = 2

    # bit value categories
    is_zero = 0
    is_one = 1
    maybe_zero = 2
    maybe_one = 3
    maybe_either = 4

    def __init__(self, size, min_num, max_num):
        """ create the valid code set for a number in the range min_num..max_num for code_size
            a valid code is one where there are no embedded start/stop bits bits but contains at least one 1 bit,
            we create two arrays - one maps a number to its code and the other maps a code to its number.
            """
        self.size = size                                       # total ring code size in bits
        self.min_num = min_num                                 # minimum number we want to be able to encode
        self.max_num = max_num                                 # maximum number we want to be able to encode
        self.skew = max(int(self.size / 3),1)                  # ring to ring skew in bits
        self.grey_min = None                                   # luminance below this is considered 'black'
        self.grey_max = None                                   # luminance above this is considered 'white'
        self.marker_bits = 4                                   # number of bits in our alignment marker
        self.code_bits = self.size - self.marker_bits          # code word bits is what's left
        self.marker = 6 << self.code_bits                      # 0110 in MS 4 bits of code
        self.code_range = 1 << self.code_bits
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
                    # got a good code, give it to next number
                    num += 1
                    if num > self.max_num:
                        # found enough, don't use this one
                        pass
                    else:
                        self.codes[code] = num                 # decode code as num
                        self.nums[num] = code                  # encode num as code
        self.num_limit = num
        print('Within 0..{} there are {} valid codes, mapped to {}..{} numbers'.format(self.code_range,
                                                                                       num - (self.min_num - 1),
                                                                                       self.min_num,
                                                                                       min(num, self.max_num)))

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
            rings in the same order).
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
        return r1, r2, r3

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
                for m_at in maybe_at:
                    for m in m_at:
                        self.demote_marker(m, word)
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

    def demote_marker(self, n, bits):
        """ demote the candidate alignment marker at n to not a candidate
            this is done by changing any (0?) and (1?) bits to (?)
            """
        i = self.marker_bits_pos(n)
        for bit in range(self.marker_bits):
            self.demote_bit(i[bit], bits)

    def demote_bit(self, n, bits):
        """ demote the bit at n to (?) from (0?) or (1?) """
        if bits[n] == self.maybe_one or bits[n] == self.maybe_zero:
            bits[n] = self.maybe_either

    def data_bits(self, n, bits):
        """ return an array of the data-bits from bits array starting at bit position n """
        return [bits[int(pos % self.size)] for pos in range(n+self.marker_bits, n+self.size)]

    def extract_word(self, n, bits):
        """ given an array of bit values with the alignment marker at position n
            extract the code word and decode it (via decode()), returns None if cannot,
            this may take several attempts if the bit values contain maybe_either bits
            maybe_zero is treated as is_zero and maybe_one is treated as is_one but each
            maybe_either is tried for both (along with all the others, if X bits can be
            either then 2^X combinations are tried), more than one valid combination is
            a no-can-do condition (too ambiguous)
            """
        word = self.data_bits(n, bits)
        partial = [None for _ in range(len(word))]
        options = 0
        for bit in range(len(word)):
            val = word[bit]
            if (val == self.is_one) or (val == self.maybe_one):
                partial[bit] = 1
            elif (val == self.is_zero) or (val == self.maybe_zero):
                partial[bit] = 0
            else:
                options += 1
        # try every possible combination of bit options discovered
        found = None
        for mask in range(max(1 << options, 1)):
            # NB: Building candidate big-endian
            candidate = 0
            for bit in partial:
                candidate <<= 1                    # make room for next bit
                if bit is None:
                    candidate += (mask & 1)        # add the next option bit
                    mask >>= 1                     # prepare the next optional bit
                else:
                    candidate += bit               # we know what this one is
            result = self.decode(candidate)        # note: returns None if not valid
            if result is not None:
                if found is not None:
                    # must only be 1 good one, else ambiguous
                    return None
                found = result                     # note the first good one we find
        # every possibility has now been evaluated
        return found

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
            the return bit is one of is_zero, is_one, maybe_either, maybe_zero, maybe_one, or None
            """
        if s1 == self.grey:
            if s2 == self.grey:
                if s3 == self.grey:
                    return self.maybe_either
                elif s3 == self.black:
                    return self.maybe_zero
                elif s3 == self.white:
                    return self.maybe_one
            elif s2 == self.black:
                if s3 == self.grey:
                    return self.maybe_zero
                elif s3 == self.black:
                    return self.is_zero
                elif s3 == self.white:
                    return self.maybe_either
            elif s2 == self.white:
                if s3 == self.grey:
                    return self.maybe_one
                elif s3 == self.black:
                    return self.maybe_either
                elif s3 == self.white:
                    return self.is_one
        elif s1 == self.black:
            if s2 == self.grey:
                if s3 == self.grey:
                    return self.maybe_zero
                elif s3 == self.black:
                    return self.is_zero
                elif s3 == self.white:
                    return self.maybe_either
            elif s2 == self.black:
                if s3 == self.grey:
                    return self.is_zero
                elif s3 == self.black:
                    return self.is_zero
                elif s3 == self.white:
                    return self.is_zero
            elif s2 == self.white:
                if s3 == self.grey:
                    return self.maybe_either
                elif s3 == self.black:
                    return self.is_zero
                elif s3 == self.white:
                    return self.is_one
        elif s1 == self.white:
            if s2 == self.grey:
                if s3 == self.grey:
                    return self.maybe_one
                elif s3 == self.black:
                    return self.maybe_either
                elif s3 == self.white:
                    return self.is_one
            elif s2 == self.black:
                if s3 == self.grey:
                    return self.maybe_either
                elif s3 == self.black:
                    return self.is_zero
                elif s3 == self.white:
                    return self.is_one
            elif s2 == self.white:
                if s3 == self.grey:
                    return self.is_one
                elif s3 == self.black:
                    return self.is_one
                elif s3 == self.white:
                    return self.is_one
        # none of the above, so...
        return None

    def blob(self, s1, s2, s3):
        """ given 3 luminance samples determine the most likely blob value
            each sample is checked against the grey threshold to determine if its black, grey or white
            then decoded as a bit
            """
        return self.bit(self.category(s1), self.category(s2), self.category(s3))

    def category(self, level):
        """ given a luminance level categorize it as black, white or grey """
        if self.grey_max is None or self.grey_min is None:
            # we haven't been given the thresholds, so no-can-do
            return None
        if level is None:
            return None
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
        bit 0 is drawn first, then 1, etc, up to bit s-1, this is considered to be clockwise
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
        self.edge = 360 / self.s         # the angle at which a bit edge occurs
        # draw the bullseye and its enclosing ring
        self.draw(0, -1)
        self.draw(-1, 0)

    def _point(self, x, y, bit):
        """ draw a point at offset x,y from our centre with the given bit (0 or 1) colour (black or white) """
        if bit == 0:
            colour = min_luminance
        else:
            colour = max_luminance
        x += self.x
        y += self.y
        self.c.putpixel(x  , y  , colour)
        self.c.putpixel(x+1, y  , colour)
        self.c.putpixel(x  , y+1, colour)
        self.c.putpixel(x+1, y+1, colour)

    def _draw(self, radius, bits):
        """ draw a ring at radius of bits """
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
                if bits & mask:
                    self._point(x, y, 1)
                else:
                    self._point(x, y, 0)

    def draw(self, ring_num, data_bits):
        """ draw a data ring, ring 0 is the centre, -1 is its enclosing circle, 1..3 are the data rings """
        if ring_num == 0:
            draw_ring = 0                # central bullseye
        elif ring_num == -1:
            draw_ring = 1                # enclosing ring
        elif ring_num < 1 or ring_num > 3:
            raise Exception('Ring number must be 1..3 not {}'.format(ring_num))
        else:
            draw_ring = ring_num + 1
        for radius in range(draw_ring*self.w,(draw_ring+1)*self.w):
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
        max_rings = 5
        start_x = -(max_rings * self.w)
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
        self.draw(1, rings[0])
        self.draw(2, rings[1])
        self.draw(3, rings[2])
        self.label(number)


class Frame:
    """ image frame buffer as a 2D array of luminance values """
    buffer = None

    def new(self, width, luminance):
        self.buffer = [[luminance for _ in range(width)] for _ in range(width)]

    def size(self):
        """ return the x,y size of the current frame buffer """
        return len(self.buffer), len(self.buffer[0])

    def load(self, image_file):
        """ load frame buffer from a JPEG image file """
        im = Image.open(image_file)
        im.convert(mode='L')
        # im.show()

        max_x = im.size[0]
        max_y = im.size[1]

        # Note: axis arranged such that buffer[x][y] yields pixel(x,y)
        self.buffer = [[im.getpixel((x, y))[0] for y in range(max_y)] for x in range(max_x)]

        im.close()

    def unload(self, image_file):
        """ unload the frame buffer to a JPEG image file """
        max_x, max_y = self.size()
        im = Image.new("RGB", (max_x, max_y), (mid_luminance, mid_luminance, mid_luminance))
        for x in range(max_x):
            for y in range(max_y):
                pixel = self.buffer[x][y]
                im.putpixel((x, y), (pixel, pixel, pixel))
        im.save(image_file)
        im.show()
        im.close()

    def getpixel(self, x, y):
        """ get the pixel value at x,y """
        if x < 0 or x >= len(self.buffer) or y < 0 or y >= len(self.buffer[0]):
            return None
        else:
            return self.buffer[x][y]

    def putpixel(self, x, y, value):
        """ put the pixel of value at x,y """
        if x < 0 or x >= len(self.buffer) or y < 0 or y >= len(self.buffer[0]):
            pass
        else:
            self.buffer[x][y] = min(max(value, min_luminance), max_luminance)


class Test:
    """ test the critical primitives """
    def __init__(self, code_bits, min_num, max_num, ratio_scale):
        self.code_bits = code_bits
        self.min_num = min_num
        self.ratio_scale = ratio_scale
        self.c = Codes(self.code_bits, min_num, max_num)
        self.frame = Frame()
        self.max_num = min(max_num, self.c.num_limit)

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
        print('Check build/unbuild from {} to {}'.format(self.min_num, self.max_num))
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

    def code_words(self, test_set):
        """ test code-word rotation with given set plus the extremes(visual) """
        print('')
        print('******************')
        print('Check code-words (visual)')
        numbers = [self.min_num] + test_set + [self.max_num]
        bin = '{:0'+str(self.code_bits)+'b}'
        frm = '{}('+bin+')=('+bin+', '+bin+', '+bin+')'
        try:
            for n in numbers:
                rings = self.c.build(n)
                print(frm.format(n, n, rings[0], rings[1], rings[2]))
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
        """ test drawn ring segments (visual) """
        print('')
        print('******************')
        print('Draw an angle test ring (visual)')
        try:
            self.frame.new(width * 11, mid_luminance)          # 11 == 5 rings in radius + a border
            x, y = self.frame.size()
            ring = Ring(x >> 1, y >> 1, self.code_bits, width, self.ratio_scale, self.frame)
            ring.draw(1,0x5555)          # alternating 0's and 1's
            ring.draw(2,0xAAAA)          # opposite
            ring.draw(3,0x5555)          # ..
            ring.label(999)
            self.frame.unload('{}-segment-angle-test.jpg'.format(self.code_bits))
        except:
            traceback.print_exc()
        print('******************')

    def codes(self, test_set, width):
        """ test drawn codes for extremes and given test_set """
        print('')
        print('******************')
        print('Draw test codes (visual)')
        try:
            numbers = [self.min_num] + test_set + [self.max_num]
            for n in numbers:
                rings = self.c.build(n)
                self.frame.new(width * 11, mid_luminance)      # 11 == 5 rings in radius + a border
                x, y = self.frame.size()
                ring = Ring(x >> 1, y >> 1, self.code_bits, width, self.ratio_scale, self.frame)
                ring.code(n, rings)
                self.frame.unload('{}-segment-{}.jpg'.format(self.code_bits, n))
        except:
            traceback.print_exc()
        print('******************')


# parameters
min_num = 101                            # min number we want
max_num = 999                            # max number we want (may not be achievable)
code_bits = 15                           # number of bits in our code word
ratio_scale = 3600                       # scale factor for ratio and angle calculations (good for 0.1 degree)

test_circle_size = 120
test_code_bits = code_bits
test_ratio_scale = ratio_scale
test_ring_width = 64
test_num_set = [341, 511, 682, 795, 877]
test_black = min_luminance + 64 + 32
test_white = max_luminance - 64 - 32
test_noise = mid_luminance

test = Test(test_code_bits, min_num, max_num, test_ratio_scale)
test.coding()
test.decoding(test_black, test_white, test_noise)
test.angles(test_circle_size)
test.ratios(test_circle_size)
test.code_words(test_num_set)
test.rings(test_ring_width)
test.codes(test_num_set, test_ring_width)
