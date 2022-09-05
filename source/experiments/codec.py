
""" coding scheme

    This coding scheme is intended to be easy to detect and robust against noise, distortion and luminance artifacts.
    The code is designed for use as competitor numbers in club level fell races, as such the number range
    is limited to between 101 and 999 inclusive, i.e. always a 3 digit number and max 899 competitors.
    The actual implementation limit may be less depending on coding constraints chosen.
    The code is circular (so orientation does not matter) and consists of:
        a solid circular centre of 'white' with a radius 2R (a 'blob'),
        surrounded by a solid ring of 'black' and width 1R,
        surrounded by a data ring of width 4R and divided into N equal segments,
        surrounded by a solid ring of 'black' and width 1R
    Total radius is 8R.
    The code consists of a 5 digit number in base 6 (or 7), these 5 digits are repeated 3 times yielding
    a 15 digit number with triple redundancy. Each base 6 (or 7) digit is represented by the ratio of the
    length of the black region and white region following the inner blob. The lengths are chosen to
    maximise the differences between the resulting ratios. For base 6 the ratios are:
        1:2, 1:3, 1:4, 2:1, 3:1, 4:1
    For base 7 a ratio of 1:1 is added.
    Each ratio is assigned a number 1..6 (or 7) - the digit. In a 5 digit code, the most significant digit
    is always the biggest, this is the mechanism to detect code boundaries. Also, within the 5 digit code
    no two adjacent digits may be the same, this ensures there is a digit 'edge' (black/white transition)
    for every digit and is the mechanism to detect digit boundaries.

    Some codes are reserved for special purposes - start, check, retired, finish, etc.
    Codes are allocated to numbers pseudo randomly (but deterministically) to ensure a spread of codes
    irrespective of the numbers used in an event.

    Note: The decoder can cope with several targets in the same image. It uses the relative size of
          the targets to present the results in distance order, nearest (i.e. biggest) first. This is
          important to determine finish order.

    This Python implementation is just a proof-of-concept. In particular, it does not represent good
    coding practice, nor does it utilise the Python language to its fullest extent, and performance
    is mostly irrelevant.

    """

import math
import rand


class Codec:
    """ Encode and decode a code-word or a code-block,
        a code-word is a number with specific properties,
        a code-block is an encoding of N code-word's with specific properties,
        this class encapsulates all the encoding and decoding and their constants
        """

    # region constants...
    DIGITS_PER_WORD = 5  # how many digits per encoded code-word
    COPIES_PER_BLOCK = 3  # number of copies in a code-word (DIGITS_PER_WORD * COPIES_PER_BLOCK must be odd)

    # encoding is a bit pattern across the four data rings for each digit (a 0 at either end is implied)
    # of the 16 combinations many are not allowed:
    #   1 0000 - treated as 6:1 (ratio 6 and above, detected by running into the white border)
    #   2 0001 - 4:1
    #   3 0010 - 3:1
    #   4 0011 - 3:2 (ratio 1.5)
    #   5 0100 - 2:1
    #     0101 - not allowed (double pulse)
    #   6 0110 - 1:1
    #   7 0111 - 2:3
    #     1000 - not allowed (1:1 ambiguous with 0110)
    #     1001 - not allowed (double pulse)
    #     1010 - not allowed (double pulse)
    #     1011 - not allowed (double pulse)
    #   8 1100 - 1:2 (ratio 0.5)
    #     1101 - not allowed (double pulse)
    #   9 1110 - 1:3 (ration 0.33)
    #  10 1111 - 1:4 (ratio 0.25 or below, or anything that runs into the border)
    # of the 10 possibilities the 7 with the biggest error differential have been chosen
    # NB: use None for trailing zeroes (so ratio calculator knows true 0's length)
    ENCODING = [
                [1, 1, 1,    1   ],
                [0, 0, 1,    1   ],
                [0, 0, 0,    1   ],
                [0, 1, 1,    None],
                [0, 1, None, None],
                [0, 0, 1,    None],
                [0, 0, 0,    0   ],
               ]

    # base 6 encoding yields 452 usable codes, base 7 yields 1202, base 5 yields 132
    BASE_MAX = len(ENCODING)
    BASE_MIN = 2
    RINGS_PER_DIGIT = len(ENCODING[0])  # number of rings spanning the variable data portion of a code-word
    SPAN = RINGS_PER_DIGIT + 2  # total span of the code including its margin black rings
    DIGITS = DIGITS_PER_WORD * COPIES_PER_BLOCK  # number of digits in a full code-block (must be odd)
    DOUBT_LIMIT = int(COPIES_PER_BLOCK / 2)  # we want the majority to agree, so doubt must not exceed half
    # endregion

    def __init__(self, min_num, max_num, base: int = BASE_MAX):
        """ create the valid code set for a number in the range min_num..max_num,
            the min_num cannot be zero,
            we create two arrays - one maps a number to its code and the other maps a code to its number.
            """

        self.base = max(min(base, Codec.BASE_MAX), Codec.BASE_MIN)
        self.code_range = int(math.pow(self.base, Codec.DIGITS_PER_WORD))

        # params
        self.min_num = max(min_num, 1)  # minimum number we want to be able to encode (0 not allowed)
        self.max_num = max(max_num, self.min_num)  # maximum number we want to be able to encode

        # build ratios table
        # ratios is the relative length, across a radial, of the black rings (including the inner black)
        # and white rings following the blob, the number of trailing black rings is irrelevant but must
        # be >0 (its just a 'filler')
        # count the number of black and white elements for each code
        counts = [[1, 0] for _ in range(len(Codec.ENCODING))]  # NB: inner black ring is implied
        for digit, rings in enumerate(Codec.ENCODING):
            for ring in rings:
                if ring is None:
                    # this is a don't care 0
                    continue
                if ring == 0:
                    counts[digit][0] += 1  # one more black
                else:
                    counts[digit][1] += 1  # one more white
        # calculate the black/white ratio
        self.ratios = [None for _ in range(len(Codec.ENCODING))]
        for digit, (blacks, whites) in enumerate(counts):
            if whites == 0:
                self.ratios[digit] = blacks + 1  # +1 for outer black
            else:
                self.ratios[digit] = blacks / whites
        # calc the ratio limits (used for testing)
        self.min_ratio = self.ratios[0]
        self.max_ratio = self.ratios[0]
        for ratio in self.ratios:
            if ratio < self.min_ratio:
                self.min_ratio = ratio
            if ratio > self.max_ratio:
                self.max_ratio = ratio

        # build code tables
        # we first build a list of all allowable codes then allocate them pseudo randomly to numbers.
        # its expected that the first N codes are by convention reserved for special purposes
        # (start, check, retired, finish, cp's, etc)
        self.allowed = [False for _ in range(self.code_range)]  # set all invalid initially
        all_codes = []
        for code in range(self.code_range):
            if self._allowable(code):
                all_codes.append(code)
        self.code_limit = len(all_codes)
        code_index = {}
        generate = rand.Rand()  # use the default seed
        for num in range(self.code_limit):
            while True:
                index = int(round(generate.rnd() * (self.code_limit - 1)))
                if code_index.get(index) is None:
                    # got a unique index
                    code_index[index] = num
                    break
        # code_index now contains indexes into all_codes in number order
        self.codes = [None for _ in range(self.code_limit)]
        for index, num in code_index.items():
            # index is an index into all_codes
            # code is the number for that code (in range 0..N)
            self.codes[num] = all_codes[index]
        # self.codes[number] -> code is now the mapping of numbers (offset by self.min_num) to codes
        # we also need the opposite - codes to numbers - self.nums[code] -> number
        self.nums = [None for _ in range(self.code_range)]  # NB: This array is sparsely populated
        for num, code in enumerate(self.codes):
            self.nums[code] = num

        if self.code_limit > 0:
            self.code_limit -= 1  # change from a range to a limit
        if self.code_limit + self.min_num > self.max_num:
            # got more codes than we need
            self.num_limit = self.max_num
        else:
            # not got enough
            self.num_limit = self.code_limit + self.min_num

    def encode(self, num):
        """ get the code for the given number, returns None if number not valid """
        if num is None or num < self.min_num or (num - self.min_num) >= len(self.codes):
            return None
        return self.codes[num - self.min_num]

    def decode(self, code):
        """ get the number for the given code, if not valid returns None """
        if code is None or code < 0 or code >= len(self.nums):
            return None
        return self.nums[code] + self.min_num

    def build(self, num):
        """ build the code block needed for the data rings
            returns the N digits required to build a 'target'
            """
        if num is None:
            return None
        code = self.encode(num)
        if code is None:
            return None
        code_block = self._make_code_block(code)
        return self._rings(code_block)

    def unbuild(self, digits):
        """ given an array of N digits with random alignment return the encoded number or None,
            N must be Codec.DIGITS
            returns the number (or None) and the level of doubt
            """

        if len(digits) != Codec.DIGITS:
            return None, Codec.DIGITS

        # step 1 - split into its copies
        copies = [[None for _ in range(Codec.DIGITS_PER_WORD)] for _ in range(Codec.COPIES_PER_BLOCK)]
        for item, value in enumerate(digits):
            copy = int(item / Codec.DIGITS_PER_WORD)
            digit = int(item % Codec.DIGITS_PER_WORD)
            copies[copy][digit] = value

        # step 2 - amalgamate digit copies into most likely with a doubt
        merged = [[None, None] for _ in range(Codec.DIGITS_PER_WORD)]  # number and doubt for each digit in a word
        for digit in range(len(copies[0])):
            # the counts structure contains the digit and a count of how many copies of that digit exist
            counts = [[0, idx] for idx in range(self.base)]
            for copy in range(len(copies)):
                digit_copy = copies[copy][digit]
                if digit_copy is None:
                    # no count for this
                    continue
                counts[digit_copy][0] += 1
            # pick digit with the biggest count (i.e. first in sorted counts)
            counts.sort(key=lambda c: (c[0], c[1]), reverse=True)
            doubt = Codec.COPIES_PER_BLOCK - counts[0][0]
            # possible doubt values are: 0==all copies the same, 1==1 different, 2+==2+ different
            merged[digit] = (counts[0][1], doubt)

        # step 3 - look for the biggest digit
        biggest_at = None
        biggest = -1
        for idx, (digit, _) in enumerate(merged):
            if digit > biggest:
                biggest = digit
                biggest_at = idx
        if biggest_at is None:
            # this should not be possible - one of them has to be biggest!
            return None, doubt

        # step 4 - extract the code and its doubt
        idx = biggest_at  # this is the MS digit
        code = 0
        doubt = 0
        for _ in range(Codec.DIGITS_PER_WORD):
            idx = (idx - 1) % Codec.DIGITS_PER_WORD  # scan backwards do we go LS to MS digit
            digit, digit_doubt = merged[idx]
            code *= self.base
            code += digit
            doubt += digit_doubt

        # step 5 - lookup number
        number = self.decode(code)
        if number is None:
            if doubt == 0:
                doubt = Codec.DIGITS

        # that's it
        return number, doubt, code

    def ratio(self, digit):
        """ given a digit return its black/white ratio, this represents the ideal ratio """
        for candidate, ideal in enumerate(self.ratios):
            if candidate == digit:
                return ideal
        return None

    def classify(self, actual):
        """ given a ratio measurement, return a list of the most likely digits it represents with an error,
            every possible digit is returned in least error first order,
            all errors are in the range 0..1, with 0=perfect and 1=utter crap
            """

        def error(actual, ideal):
            """ calculate an error between the actual ratio and the ideal,
                ideal is in range min-ratio to max_ratio,
                actual could be anything, but we clamp it to the same limits,
                we want a number that is in the range 0..1 where 0 is no error and 1 is a huge error
                """
            if actual > self.max_ratio:
                actual = self.max_ratio
            elif actual < self.min_ratio:
                actual = self.min_ratio
            if ideal > actual:
                err = actual / ideal
            else:
                err = ideal / actual
            err = 1 - err  # convert 1..0 to 0..1
            err *= err  # go quadratic so big errors spread more (still 0..1)
            return err

        digits = [(None, 1) for _ in range(self.base)]
        for digit in range(self.base):
            ideal = self.ratios[digit]
            digits[digit] = (digit, error(actual, ideal))
        digits.sort(key=lambda d: d[1])
        return digits

    def digits(self, code):
        """ given a code return the digits for that code """
        partial = [None for _ in range(Codec.DIGITS_PER_WORD)]
        for digit in range(Codec.DIGITS_PER_WORD):
            partial[digit] = code % self.base
            code = int(code / self.base)
        return partial

    def digit(self, rings):
        """ given a set of data rings for a single digit return the corresponding digit,
            this is only used for testing, it is the opposite to _rings for a single digit
            """
        for digit, coding in enumerate(Codec.ENCODING):
            same = 0
            for ring, bit in enumerate(coding):
                if bit is None:
                    # treat like 0
                    bit = 0
                if rings[ring] == bit:
                    same += 1
            if same == len(coding):
                return digit
        return None

    def _rings(self, code_block):
        """ build the data ring encoding for the given code-block,
            code_block must be a list of digits in the range 0..BASE-1,
            each digit is encoded into 5 bits, one for each ring,
            returns a list of integers representing each ring
            """

        # build code block
        rings = [0 for _ in range(Codec.RINGS_PER_DIGIT)]
        data = 1 << (Codec.DIGITS - 1)  # start at the msb
        for digit in code_block:
            coding = Codec.ENCODING[digit]
            for ring, bit in enumerate(coding):
                if bit is None:
                    # this is a don't care 0
                    continue
                if bit == 1:
                    # want a 1 bit in this position in this ring
                    rings[ring] += data
            data >>= 1  # move to next bit position

        return rings

    def _make_code_block(self, code):
        """ make the code-block from the given code,
            a code-block is the given code encoded as base N and copied N times,
            the encoding is little-endian (LS digit first),
            it is returned as a list of digits
            """
        partial = self.digits(code)
        code_block = []
        for _ in range(Codec.COPIES_PER_BLOCK):
            code_block += partial
        return code_block

    def _allowable(self, candidate):
        """ given a code word candidate return True if its allowable as the basis for a code-word,
            code-words are not allowed if they do not meet our property requirements around the rings,
            the requirement is that the most significant digit must be the biggest and no two adjacent
            digits can be the same, and at least one digit must have 'black' element as the first data ring
            returns True iff its allowable
            """

        digits = self.digits(candidate)
        first_digit = digits[0]
        last_digit = first_digit
        if Codec.ENCODING[first_digit][0] == 0:  # NB: first element in an encoding is never None
            has_black = True
        else:
            has_black = False
        for idx in range(1, len(digits)):
            digit = digits[idx]
            if digit >= first_digit:
                # does not meet requirement that first digit is the biggest
                return False
            if digit == last_digit:
                # does not meet requirement that adjacent digits are different
                return False
            last_digit = digit
            if Codec.ENCODING[digit][0] == 0:  # NB: first element in an encoding is never None
                has_black = True
        if not has_black:
            return False

        # all good
        return True
