
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
    DIGITS_PER_WORD = 6  # how many digits per encoded code-word
    COPIES_PER_BLOCK = 3  # number of copies in a code-word (DIGITS_PER_WORD * COPIES_PER_BLOCK must be odd)
    INNER_BLACK_RINGS = 1  # number of inner black rings (defined here 'cos encoding relies on it)
    OUTER_BLACK_RINGS = 1  # number of outer black rings (defined here 'cos encoding relies on it)
    EDGE_RINGS = INNER_BLACK_RINGS + OUTER_BLACK_RINGS

    # encoding is a bit pattern across the data rings for each digit (a 0 at either end is implied)
    # **** DO NOT CHANGE THIS - it'll invalidate existing codes
    ENCODING = [
                [0, 0, 0],  # 0  must be first
                [0, 0, 1],  # 1
                [0, 1, 0],  # 2
                [0, 1, 1],  # 3
                [1, 0, 0],  # 4
                [1, 1, 0],  # 5
                [1, 1, 1],  # 6
    ]

    ZERO_DIGIT = ENCODING[0]
    BASE_MAX = len(ENCODING)
    BASE_MIN = 2
    RINGS_PER_DIGIT = len(ENCODING[0])  # number of rings spanning the variable data portion of a code-word
    SPAN = RINGS_PER_DIGIT + EDGE_RINGS  # total span of the code including its margin black rings
    DIGITS = DIGITS_PER_WORD * COPIES_PER_BLOCK  # number of digits in a full code-block (must be odd)
    DOUBT_LIMIT = int(COPIES_PER_BLOCK / 2)  # we want the majority to agree, so doubt must not exceed half
    DOUBLE_HEAD_GAP_LIMIT = 0.1  # (see make_ratio) head gap ratio required to consider it as 2 distinct heads
    # endregion

    class Ratio:
        """ encapsulate the digit pulse element (lead, head, tail) ratios """

        def __init__(self, lead: float, head: float, tail: float):
            self.lead = lead
            self.head = head
            self.tail = tail

        def __str__(self):
            return '({:.2f}, {:.2f}, {:.2f})'.format(self.lead, self.head, self.tail)

    class Error:
        """ encapsulate the digit pulse element (lead, head, tail) errors """

        def __init__(self, lead_error: float, head_error: float, tail_error: float):
            self.lead_error = lead_error
            self.head_error = head_error
            self.tail_error = tail_error
            self.error = (self.lead_error + self.head_error + self.tail_error) / 3  # overall average

        def __str__(self):
            return '[{:.2f}, {:.2f}, {:.2f}, {:.2f}]'.\
                   format(self.error, self.lead_error, self.head_error, self.tail_error)

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
        for digit, rings in enumerate(Codec.ENCODING):
            # count the number of leading black, central white and trailing black elements
            leading = 1  # inner black is assumed
            central = 0
            trailing = 1  # outer black is assumed
            for ring in rings:
                if ring == 0 and central == 0:
                    # one more leading black
                    leading += 1
                elif ring == 0:  # and central > 0
                    # one more trailing black
                    trailing += 1
                else:  # ring != 0 and central > 0
                    # one more central white
                    central += 1
            # calculate the lead, head and tail ratios
            self.ratios[digit] = self.make_ratio(leading, central, trailing)
        # calc the ratio limits and their span (used in the error function)
        self.min_lead_ratio = 1
        self.max_lead_ratio = 0
        self.min_head_ratio = 1
        self.max_head_ratio = 0
        self.min_tail_ratio = 1
        self.max_tail_ratio = 0
        for ratio in self.ratios:
            if ratio.lead < self.min_lead_ratio:
                self.min_lead_ratio = ratio.lead
            if ratio.head < self.min_head_ratio:
                self.min_head_ratio = ratio.head
            if ratio.tail < self.min_tail_ratio:
                self.min_tail_ratio = ratio.tail
            if ratio.lead > self.max_lead_ratio:
                self.max_lead_ratio = ratio.lead
            if ratio.head > self.max_head_ratio:
                self.max_head_ratio = ratio.head
            if ratio.tail > self.max_tail_ratio:
                self.max_tail_ratio = ratio.tail
        self.lead_ratio_span = self.max_lead_ratio - self.min_lead_ratio
        self.head_ratio_span = self.max_head_ratio - self.min_head_ratio
        self.tail_ratio_span = self.max_tail_ratio - self.min_tail_ratio
        # endregion

        # region build code tables...
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

        # that's it
        return code, doubt

    def to_ratio(self, digit: int) -> Ratio:
        """ given a digit return its ratios, these represent the ideal ratios """
        for candidate, ideal in enumerate(self.ratios):
            if candidate == digit:
                return ideal
        return None

    def make_ratio(self, lead: float, head1: float, tail: float, head2: float = None, total: float = None) -> Ratio:
        """ given several lengths return the corresponding classification ratios,
            this function is aware of the expected pulse shapes/positions,
            total is the total measured length of the data area (including the marker black rings),
            lead is the length of the leading black area (including the inner black ring),
            head1 is the length of the white area as measured going down from the inner (may be 0),
            head2 is the length of the white area as measured going up from the outer (may be),
            tail is the length of the trailing black area (including rhe outer black ring),
            if head1 and head2 are distinct we've got a double pulse (which is not expected),
            the result given here can be fed into classify to get a list of likely digits
            """
        if head2 is None:
            head2 = head1
        if total is None:
            total = lead + head1 + tail
        if head1 == 0 or head2 == 0 or tail == 0:
            # special case for a '0'
            return Codec.Ratio(1, 0, 0)
        # have we got a double pulse?
        # if we have lead + head1 + head2 + tail will be less than total (by the pulse gap)
        gap = total - (lead + head1 + head2 + tail)
        if gap > 0:
            # a small gap is treated like noise and the true head is head1 + head2 + gap
            # a bigger gap is an overlap, we then use the greater of head1 and head2
            gap_ratio = gap / total
            if gap_ratio > Codec.DOUBLE_HEAD_GAP_LIMIT:
                # we've got two heads, use the biggest and add the other to the lead or tail
                if head1 > head2:
                    # use head1 and add head2 and the gap to the tail
                    head = head1
                    tail += (head2 + gap)
                else:
                    # use head2 and add head1 and the gap to the lead
                    head = head2
                    lead += (head1 + gap)
            else:
                # its a noise gap
                head = head1 + head2 + gap
        else:
            # not a double pulse, consider head to be the average of head1 and head2
            # and spread the difference between lead and tail
            head = (head1 + head2) / 2
            extra = (head - head1) / 2
            lead -= extra
            tail += extra
        lead_ratio = lead / total
        head_ratio = head / total
        tail_ratio = tail / total
        return Codec.Ratio(lead_ratio, head_ratio, tail_ratio)

    def classify(self, actual):
        """ given a ratio measurement, return a list of the most likely digits it represents with an error,
            every possible digit is returned in least error first order,
            all errors are in the range 0..1, with 0=perfect and 1=utter crap
            """

        def error(actual: Codec.Ratio, ideal: Codec.Ratio) -> float:
            """ calculate an error between the actual ratio and the ideal,
                ideal elements are in the range min-ratio to max_ratio, actual could be anything,
                we want a number that is in the range 0..1 where 0 is no error and 1 is a huge error,
                """

            # actual figures could be anything, but we must constrain them to our min/max
            # limits in order to end up with error differences within a known range
            actual_lead = min(self.max_lead_ratio, max(self.min_lead_ratio, actual.lead))
            actual_head = min(self.max_head_ratio, max(self.min_head_ratio, actual.head))
            actual_tail = min(self.max_tail_ratio, max(self.min_tail_ratio, actual.tail))

            if ideal.lead > actual_lead:
                lead_err = ideal.lead - actual_lead
            else:
                lead_err = actual_lead - ideal.lead
            lead_err /= self.lead_ratio_span  # range now 0..1

            if ideal.head > actual_head:
                head_err = ideal.head - actual_head
            else:
                head_err = actual_head - ideal.head
            head_err /= self.head_ratio_span  # range now 0..1

            if ideal.tail > actual_tail:
                tail_err = ideal.tail - actual_tail
            else:
                tail_err = actual_tail - ideal.tail
            tail_err /= self.tail_ratio_span  # range now 0..1

            # return average error and its components
            err = Codec.Error(lead_err, head_err, tail_err)

            return err

        digits = [(None, 1) for _ in range(self.base)]
        if actual is not None:
            for digit in range(self.base):
                ideal = self.ratios[digit]
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
            returns True iff its allowable
            """

        digits = self.digits(candidate)
        first_digit = digits[0]
        last_digit = first_digit
        if first_digit == 0:
            zeroes = 1
        else:
            zeroes = 0
        if Codec.ENCODING[first_digit][0] == 0:
            has_black = True
        else:
            has_black = False
        for idx in range(1, len(digits)):
            digit = digits[idx]
            if digit >= first_digit:
                # does not meet first digit is the biggest requirement
                return False
            if digit == last_digit:
                # does not meet adjacent digits different requirement
                return False
            if digit == 0:
                zeroes += 1
            if Codec.ENCODING[digit][0] == 0:
                has_black = True
            last_digit = digit

        if zeroes != 1:
            # does not meet our one and only one digit must be 0 requirement
            return False
        if not has_black:
            # does not meet our inner data ring having a black segment requirement
            return False

        # all good
        return True
