
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
    INNER_BLACK_RINGS = 1  # number of inner black rings (defined here 'cos encoding relies on it)
    OUTER_BLACK_RINGS = 1  # number of outer black rings (defined here 'cos encoding relies on it)
    EDGE_RINGS = INNER_BLACK_RINGS + OUTER_BLACK_RINGS

    # encoding is a bit pattern across the data rings for each digit (a 0 at either end is implied)
    # **** DO NOT CHANGE THIS - it'll invalidate existing codes
    ENCODING = [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],  # the only one that creates a double radial 'pulse'
                [1, 1, 0],
                None,  # [1, 1, 1],
               ]

    SYNC_DIGIT = ENCODING[5]  # must not be a None above, otherwise arbitrary
    BASE_MAX = len(ENCODING)
    BASE_MIN = 2
    RINGS_PER_DIGIT = len(SYNC_DIGIT)  # number of rings spanning the variable data portion of a code-word
    SPAN = RINGS_PER_DIGIT + EDGE_RINGS  # total span of the code including its margin black rings
    DIGITS = DIGITS_PER_WORD * COPIES_PER_BLOCK  # number of digits in a full code-block (must be odd)
    DOUBT_LIMIT = int(COPIES_PER_BLOCK / 2)  # we want the majority to agree, so doubt must not exceed half
    DOUBLE_HEAD_GAP_LIMIT = 0.1  # (see make_ratio) head gap ratio required to consider it as 2 distinct heads

    # ratio quantisation mitigation
    RATIO_QUANTA = 0  # this is +/- added to lengths when calculating ratios (see make_ratio)

    # error weights to apply to the ratio components (see Error() function)
    LEAD_ERROR_WEIGHT = 1.0
    HEAD_ERROR_WEIGHT = 1.0
    GAP_ERROR_WEIGHT  = 1.0
    TAIL_ERROR_WEIGHT = 1.0
    # endregion

    class Ratio:
        """ encapsulate the digit pulse element (lead, head, second_head, tail) ratios """

        def __init__(self, lead: float, head1: float, gap: float, head2: float, tail: float, parts=None):
            self.lead  = lead   # this is a min/max tuple
            self.head1 = head1  # ..
            self.gap   = gap    # ..
            self.head2 = head2  # ......
            self.tail  = tail   # ........
            self.parts = parts  # the lengths used to determine these ratios (params to make_ratio())

        def __str__(self):
            return '(({:.2f}, {:.2f}), ({:.2f}, {:.2f}), ({:.2f}, {:.2f}), ({:.2f}, {:.2f}), ({:.2f}, {:.2f}) ' \
                   'from {})'.format(self.lead[0], self.lead[1],
                                     self.head1[0], self.head1[1],
                                     self.gap[0], self.gap[1],
                                     self.head2[0], self.head2[1],
                                     self.tail[0], self.tail[1],
                                     self.parts)

        def lead_ratio(self):
            """ return the average ratio for the min/max lead """
            return (self.lead[0] + self.lead[1]) / 2

        def head1_ratio(self):
            """ return the average ratio for the min/max head1 """
            return (self.head1[0] + self.head1[1]) / 2

        def gap_ratio(self):
            """ return the average ratio for the min/max gap """
            return (self.gap[0] + self.gap[1]) / 2

        def head2_ratio(self):
            """ return the average ratio for the min/max head2 """
            return (self.head2[0] + self.head2[1]) / 2

        def tail_ratio(self):
            """ return the average ratio for the min/max tail """
            return (self.tail[0] + self.tail[1]) / 2

    class Error:
        """ encapsulate the digit pulse element (lead, head1, gap, head2, tail) errors """

        def __init__(self, lead_error: float,
                           head1_error: float,
                           gap_error: float,
                           head2_error: float,
                           tail_error: float):
            self.lead_error = lead_error
            self.head1_error = head1_error
            self.gap_error = gap_error
            self.head2_error = head2_error
            self.tail_error = tail_error
            # overall error is the average of the errors that are not zero
            count = 0
            for err in [self.lead_error, self.head1_error, self.gap_error, self.head2_error, self.tail_error]:
                if err > 0:
                    count += 1
            if count > 0:
                self.error = (self.lead_error + self.head1_error + self.gap_error + self.head2_error + self.tail_error) \
                             / count
            else:
                self.error = 0.0

        def __str__(self):
            return '[{:.2f} from {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]'.\
                   format(self.error,
                          self.lead_error,
                          self.head1_error,
                          self.gap_error,
                          self.head2_error,
                          self.tail_error)

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

        # region build ratios table...
        # ratios are the relative lengths, across a radial, of the leading black rings (including the
        # inner black), the central white rings and the trailing black rings (including the outer black
        # following the blob, the lead ratio is the leading black over central white, the tail ratio is
        # the trailing black over central white
        self.ratios = [None for _ in range(len(Codec.ENCODING))]
        for digit, bits in enumerate(Codec.ENCODING):
            if bits is None:
                # illegal digit has no ratios
                continue
            # count the number of elements in each part: lead, head1, gap, head2, tail
            # we're assuming 3 rings here with at most 2 pulses (i.e. 101)
            parts = [1, 0, 0, 0, 1]  # lead and tail have one each in the inner and outer black rings
            need  = [0, 1, 0, 1, 0]  # what bit state needed for each part
            part  = 0  # which part we are looking for
            for bit in bits:
                if bit != need[part]:
                    part += 1
                parts[part] += 1
            # calculate the lead, head1, head2 and tail ratios
            if parts[1] == 0:
                # this is a zero, it has no tail
                parts[0] += parts[4]
                parts[4]  = 0
            elif parts[3] == 0:
                # got no head2, so gap is part of tail
                parts[4] += parts[2]
                parts[2]  = 0
            self.ratios[digit] = self.make_ratio(parts[0], parts[4], parts[1], parts[3], sum(parts))
        # endregion

        # region build code tables...
        # we first build a list of all allowable codes then allocate them pseudo randomly to numbers.
        # its expected that the first N codes are by convention reserved for special purposes
        # (start, check, retired, finish, cp's, etc)
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
        # endregion

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
        base = self.nums[code]
        if base is None:
            return None
        return base + self.min_num

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
        """ given an array of N digits with random alignment return the corresponding code or None,
            in a sense its the opposite of digits(),
            N must be Codec.DIGITS,
            returns the code (or None) and the level of doubt
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

        # step 3 - look for the sync digit
        sync_at = None
        for idx, (digit, _) in enumerate(merged):
            if Codec.ENCODING[digit] == Codec.SYNC_DIGIT:
                sync_at = idx
        if sync_at is None:
            # got a duff code
            return None, Codec.DIGITS_PER_WORD * Codec.COPIES_PER_BLOCK

        # step 4 - extract the code and its doubt
        idx = sync_at  # this is the MS digit
        code = 0
        doubt = 0
        for _ in range(Codec.DIGITS_PER_WORD):
            idx = (idx - 1) % Codec.DIGITS_PER_WORD  # scan backwards so we go LS to MS digit
            digit, digit_doubt = merged[idx]
            code *= self.base
            code += digit
            doubt += digit_doubt

        # that's it
        return code, doubt

    def to_ratio(self, digit: int) -> Ratio:
        """ given a digit return its ratios, these represent the ideal ratios """
        for candidate, ideal in enumerate(self.ratios):
            if candidate == digit:
                return ideal
        return None

    def make_ratio(self, lead: float, tail: float, head1: float, head2: float, total: float = None) -> Ratio:
        """ given several lengths return the corresponding classification ratios,
            this function is aware of the expected pulse shapes/positions,
            total is the total measured length of the data area (including the marker black rings),
            lead is the length of the leading black area (including the inner black ring),
            head1 is the length of the white area as measured going down from the inner (may be 0),
            head2 is the length of the white area as measured going up from the outer (may be 0),
            tail is the length of the trailing black area (including the outer black ring),
            if head1 and head2 are distinct we've got a double pulse,
            the result given here can be fed into classify to get a list of likely digits,
            all ratios are in the range 0..1, where 1 is the entire inner to outer span
            """
        if total is None:
            total = lead + head1 + head2 + tail
        gap = total - (lead + head1 + head2 + tail)
        parts = (lead, head1, gap, head2, tail)
        second_head = 0
        # if head1 == 0 or head2 == 0 or tail == 0:
        #     # special case for a '0'
        #     return Codec.Ratio(1, 0, 0, parts)
        # have we got a double pulse?
        # if we have lead + head1 + head2 + tail will be less than total (by the pulse gap)
        if gap > 0:
            # a small gap is treated like noise and the true head is head1 + head2 + gap
            # a bigger gap is a double pulse
            gap_ratio = gap / total
            if gap_ratio > Codec.DOUBLE_HEAD_GAP_LIMIT:
                # we've got a double pulse, head1 is the top pulse, head2 is the bottom pulse
                head = head1
                second_head = head2
            else:
                # its a noise gap
                head = head1 + head2 + gap
                gap = 0
        elif head2 > 0:
            # not a double pulse but heads overlap, this can happen with distorted images with grey areas,
            # consider head to be the average of head1 and head2 and spread the difference between lead and tail
            head = (head1 + head2) / 2
            extra = (head - head1) / 2
            lead -= extra
            tail += extra
            gap = 0
        else:
            # not a double pulse and no second head, so no gap either
            head = head1
            gap = 0
        # each ratio consists of a min/max pair based on jiggling the lengths by +/- 1 pixel,
        # this is to mitigate quantisation effects when only a few pixels are involved
        if lead > 0:
            lead_ratio = ((lead - Codec.RATIO_QUANTA) / total, (lead + Codec.RATIO_QUANTA) / total)
        else:
            lead_ratio = (0, 0)
        if head > 0:
            head_ratio = ((head - Codec.RATIO_QUANTA) / total, (head + Codec.RATIO_QUANTA) / total)
        else:
            head_ratio = (0, 0)
        if gap > 0:
            gap_ratio = ((gap - Codec.RATIO_QUANTA) / total, (gap + Codec.RATIO_QUANTA) / total)
        else:
            gap_ratio = (0, 0)
        if second_head > 0:
            second_head_ratio = ((second_head - Codec.RATIO_QUANTA) / total, (second_head + Codec.RATIO_QUANTA) / total)
        else:
            second_head_ratio = (0, 0)
        if tail > 0:
            tail_ratio = ((tail - Codec.RATIO_QUANTA) / total, (tail + Codec.RATIO_QUANTA) / total)
        else:
            tail_ratio = (0, 0)
        return Codec.Ratio(lead_ratio, head_ratio, gap_ratio, second_head_ratio, tail_ratio, parts)

    def classify(self, actual):
        """ given a ratio measurement, return a list of the most likely digits it represents with an error,
            every possible digit is returned in least error first order,
            all errors are in the range 0..1, with 0=perfect and 1=utter crap
            """

        def error(actual: Codec.Ratio, ideal: Codec.Ratio) -> Codec.Error:
            """ calculate an error between the actual ratio and the ideal,
                all ratios are in the range 0..1,
                we want a number that is in the range 0..1 where 0 is no error and 1 is a huge error,
                """

            def error_from_ratio(ideal: (float, float), actual: (float, float)) -> float:
                """ calculate the least error between the two given ratio tuples,
                    the ratio tuple is the min/max, in the case of the ideal we
                    take the average, and test that against the min/max of actual,
                    the average is taken as the result
                    """
                ideal_ratio = (ideal[0] + ideal[1]) / 2
                if ideal_ratio > actual[0]:
                    min_ratio = actual[0] / ideal_ratio
                elif ideal_ratio < actual[0]:
                    min_ratio = ideal_ratio / actual[0]
                else:
                    min_ratio = 1
                if ideal_ratio > actual[1]:
                    max_ratio = actual[1] / ideal_ratio
                elif ideal_ratio < actual[1]:
                    max_ratio = ideal_ratio / actual[1]
                else:
                    max_ratio = 1
                # min/max_ratio are in range 1..0 (1==good, 0==bad)
                best_ratio = (min_ratio + max_ratio) / 2
                # error must be in range 0..1 (0=none, 1==lots)
                return 1 - best_ratio

            lead_err  = error_from_ratio(ideal.lead , actual.lead ) * Codec.LEAD_ERROR_WEIGHT
            head1_err = error_from_ratio(ideal.head1, actual.head1) * Codec.HEAD_ERROR_WEIGHT
            gap_err   = error_from_ratio(ideal.gap  , actual.gap  ) * Codec.GAP_ERROR_WEIGHT
            head2_err = error_from_ratio(ideal.head2, actual.head2) * Codec.HEAD_ERROR_WEIGHT
            tail_err  = error_from_ratio(ideal.tail , actual.tail ) * Codec.TAIL_ERROR_WEIGHT

            # return average error and its components
            err = Codec.Error(lead_err, head1_err, gap_err, head2_err, tail_err)

            return err

        digits = [(None, Codec.Error(1, 1, 1, 1, 1)) for _ in range(self.base)]
        if actual is not None:
            for digit in range(self.base):
                ideal = self.ratios[digit]
                if ideal is None:
                    # this is an illegal digit
                    continue
                digits[digit] = (digit, error(actual, ideal))
            digits.sort(key=lambda d: d[1].error)  # put into least error order
        return digits

    def digits(self, code):
        """ given a code return the digits for that code """
        partial = [None for _ in range(Codec.DIGITS_PER_WORD)]
        if code is not None:
            for digit in range(Codec.DIGITS_PER_WORD):
                partial[digit] = code % self.base
                code = int(code / self.base)
        return partial

    def digit(self, rings):
        """ given a set of data rings for a single digit return the corresponding digit,
            it is, in a sense, the opposite to _rings for a single digit
            """
        for digit, coding in enumerate(Codec.ENCODING):
            if coding is None:
                # illegal digit
                continue
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

    def is_sync_digit(self, this_digit):
        """ determine if the given digit is the sync digit,
            this is abstracted so outside callers can use it
            """
        if Codec.ENCODING[this_digit] == Codec.SYNC_DIGIT:
            return True
        return False

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
            code-words are not allowed if they do not meet our property requirements,
            returns True iff its allowable
            """

        digits = self.digits(candidate)
        has_sync = False
        has_white = [0 for _ in range(Codec.RINGS_PER_DIGIT)]
        has_black = [0 for _ in range(Codec.RINGS_PER_DIGIT)]
        for this in range(len(digits)):
            this_digit = digits[this]
            if Codec.ENCODING[this_digit] is None:
                # not allowed to use this digit
                return False
            if self.is_sync_digit(this_digit):
                if this != 0:
                    # does not meet only first digit can be the sync digit requirement (so can sync correctly)
                    return False
                has_sync = True
            before_digit = digits[(this - 1) % len(digits)]
            if this_digit == before_digit:
                # does not meet adjacent digits must be different requirement (to ensure sufficient vertical edges)
                return False
            for ring in range(Codec.RINGS_PER_DIGIT):
                # no black bit is allowed to be fully surrounded by white ('cos it can disappear in neighbour smudges)
                if ring == 0 or ring == (Codec.RINGS_PER_DIGIT - 1):
                    # we know there is black above the first ring and below the last ring, so not a problem here
                    pass
                elif Codec.ENCODING[this_digit][ring] == 0:
                    # there are 4 neighbours here, at least one must be black
                    black_neighbours = 0
                    for x, y in [          (0, -1),
                                 (-1,  0),          (+1,  0),
                                           (0, +1)          ]:
                        test_digit = digits[(this + x) % len(digits)]
                        test_ring = ring + y
                        bits = Codec.ENCODING[test_digit]
                        if bits is None:
                            # hot an illegal digit
                            return None
                        if bits[test_ring] == 0:
                            black_neighbours += 1
                    if black_neighbours < 1:
                        # does not meet black neighbours requirement
                        return False
                # there must be at least one black and one white bit per ring (to ensure sufficient horizontal edges)
                if Codec.ENCODING[this_digit][ring] == 0:
                    has_black[ring] += 1
                if Codec.ENCODING[this_digit][ring] == 1:
                    has_white[ring] += 1
        if not has_sync:
            # does not meet first digit must be the sync digiti requirement
            return False
        for is_black in has_black:
            if is_black == 0:
                # does not meet each ring must have at least one black segment per copy requirement
                return False
        for is_white in has_white:
            if is_white == 0:
                # does not meet each ring must have at least one white segment per copy requirement
                return False
        # all good
        return True
