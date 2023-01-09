
""" coding scheme

    This coding scheme is intended to be easy to detect and be robust against noise, distortion and luminance artifacts.
    The code is designed for use as competitor numbers in club level fell races, as such the number range
    is limited to between 101 and 999 inclusive, i.e. always a 3 digit number and max 899 competitors.
    The actual implementation limit may be less depending on coding constraints chosen.
    The code is circular (so orientation does not matter) and consists of:
        a solid circular centre of 'white' with a radius 2R (a 'blob'),
        surrounded by a solid ring of 'black' and width 1R,
        surrounded by a data ring of width 3R and divided into 18 equal segments,
        surrounded by a solid ring of 'black' and width 1R
    Total radius is 8R.
    The code consists of a 6 digit number in base 8, with the first digit always 0 (the sync digit), these 6 digits
    are repeated 3 times yielding an 18 digit number with triple redundancy.

    Some codes are reserved for special purposes - start, check, retired, finish, etc.
    Codes are allocated to numbers pseudo randomly (but deterministically) to ensure a spread of codes
    irrespective of the numbers used in an event.

    Note: The decoder can cope with several targets in the same image. It uses the relative size of
          the targets to present the results in distance order, nearest (i.e. biggest) first. This is
          important to determine finish order.

    This Python implementation is just a proof-of-concept. In particular, it does not represent good
    coding practice, nor does it utilise the Python language to its fullest extent, and performance
    is mostly irrelevant. It represents the specification for the actual implementation.

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
    # if you do change it, also make appropriate changes to BIT_WEIGHTS and NOT_ALLOWED_ERROR
    ENCODING = [  # allowed, coding, weight (used to scale errors for easily confused classifications, e.g. 3->1 or 2)
                (True,  [0, 0, 0], 2  ),  # 0 - we need to be very confident to classify as a sync digit
                (True,  [0, 0, 1], 1  ),  # 1
                (True,  [0, 1, 0], 1  ),  # 2
                (True,  [0, 1, 1], 1.2),  # 3 - can be confused with 2 and 1
                (True,  [1, 0, 0], 1  ),  # 4
                (False, [1, 0, 1], 1  ),  # 5 never use this, it creates a double radial 'pulse'
                (True,  [1, 1, 0], 1.2),  # 6 - can be confused with 2 and 4
                (False, [1, 1, 1], 1  ),  # 7 do not use this, its too easily confused with 3 or 6
               ]

    SYNC_DIGIT = 0  # must be allowed above, otherwise arbitrary, zero is the easiest to detect
    DIGIT_BASE = len(ENCODING)  # how many digits in our encoding
    RINGS_PER_DIGIT = len(ENCODING[SYNC_DIGIT][1])  # number of rings spanning the variable data portion of a code-word
    SPAN = RINGS_PER_DIGIT + EDGE_RINGS  # total span of the code including its margin black rings
    DIGITS = DIGITS_PER_WORD * COPIES_PER_BLOCK  # number of digits in a full code-block (must be odd)
    DOUBT_LIMIT = int(COPIES_PER_BLOCK / 2)  # we want the majority to agree, so doubt must not exceed half
    # endregion
    # region qualify/classify tuning constants
    # error weights when classifying pixel slices across the ring SPAN (i.e. including the edge rings),
    # all the weights should sum to 1, see classify() function for usage
    # NB: if ENCODING changes, make appropriate changes here too (e.g. if change number of rings)
    VIRTUAL_RINGS_PER_RING = 3  # how many times to split a data ring for classification purposes
    BIT_WEIGHTS = [0.00, 0.00, 0.00,  # rings are split into N 'virtual rings'
                   0.08, 0.16, 0.08,  # centre of bit has more weight
                   0.09, 0.18, 0.09,  # centre ring has more weight
                   0.08, 0.16, 0.08,
                   0.00, 0.00, 0.00]

    # VIRTUAL_RINGS_PER_RING = 2
    # BIT_WEIGHTS = [0.00, 0.00,
    #                0.16, 0.17,
    #                0.17, 0.17,
    #                0.16, 0.17,
    #                0.00, 0.00]

    # max proportion of the data portion of a slice that can be white (i.e. 1's),
    # more than this disqualifies the slice for digit classification (removes 7's)
    MAX_SLICE_WHITE = 0.9

    # min proportion of the data portion of a slice that is white (i.e. 1's),
    # less than this qualifies the slice for digit classification (its a 0)
    MIN_SLICE_WHITE = 0.1

    # 1's run length, as a fraction of the nominal ring width, on either side of a 0's run when qualifying a slice
    ONES_RUN_LENGTH = 0.8
    ONES_THRESHOLD = 0.7  # proportion of ONES_RUN_LENGTH that must be 1 to qualify the run as a 1
    # 0's run length, as a fraction of the nominal ring width, within a pair of 1's runs when qualifying a slice
    ZERO_RUN_LENGTH = 0.8
    ZERO_THRESHOLD = 0.7  # proportion of ZERO_RUN_LENGTH that must be 0 to qualify the run as a 0

    # min correct digit bit samples, if less than this many samples are correct, the digit classification fails
    MIN_CORRECT_SAMPLES = 1
    # endregion

    def __init__(self, min_num, max_num):
        """ create the valid code set for a number in the range min_num..max_num,
            the min_num cannot be zero,
            we create two arrays - one maps a number to its code and the other maps a code to its number.
            """

        # sanity check
        bits = self._make_slice_bits(Codec.SYNC_DIGIT)  # only digit we can guarantee is valid
        if len(Codec.BIT_WEIGHTS) != len(bits):
            raise Exception('BIT_WEIGHTS size ({}) is not same as slice bits ({})'.
                            format(len(Codec.BIT_WEIGHTS), len(bits)))
        if sum(Codec.BIT_WEIGHTS) != 1.0:
            raise Exception('BIT_WEIGHTS must sum to 1.0 not {}'.format(sum(Codec.BIT_WEIGHTS)))

        # params
        self.min_num = max(min_num, 1)  # minimum number we want to be able to encode (0 not allowed)
        self.max_num = max(max_num, self.min_num)  # maximum number we want to be able to encode

        self.code_range = int(math.pow(Codec.DIGIT_BASE, Codec.DIGITS_PER_WORD))

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
            if num > (self.max_num - self.min_num):
                # not in required range, so its not valid
                continue
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
            counts = [[0, idx] for idx in range(Codec.DIGIT_BASE)]
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
            if self.is_sync_digit(digit):
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
            code *= Codec.DIGIT_BASE
            code += digit
            doubt += digit_doubt

        # that's it
        return code, doubt

    def make_slice(self, digit, length, index=None):
        """ build a bit slice for the given digit of the given length,
            this represents an 'ideal' slice that can be used for diagnostic purposes,
            the slice entries are 0 or 1 for expected bits,
            """
        if digit is None:
            return None
        bits = self._make_slice_bits(digit)
        if bits is None:
            # illegal digit, no slice for these
            return None
        if index is None:
            index = self._make_slice_index(length)
        if index is None:
            # not enough length to make a slice
            return None
        slice = [None for _ in range(length)]
        for entry in range(length):
            bit_num = int(index[entry])
            bit_value = bits[bit_num]
            slice[entry] = bit_value
        return slice

    def _make_slice_bits(self, digit):
        """ make the complete slice bits for the given digit,
            the complete bits include the inner/outer ring,
            there is 1 bit returned for each 'virtual ring'
            """
        if digit is None:
            return None
        bits = self.coding(digit, only_if_allowed=False)  # we want the coding regardless of it being allowed
        prefix = [0 for _ in range(Codec.INNER_BLACK_RINGS)]
        suffix = [0 for _ in range(Codec.OUTER_BLACK_RINGS)]
        slice = prefix + bits + suffix
        virtual_slice = []
        for bit in slice:
            virtual_slice += [bit for _ in range(Codec.VIRTUAL_RINGS_PER_RING)]  # add bit copied N times
        return virtual_slice

    def _get_slice_scale(self, length):
        """ determine if a slice of the given length is eligible for classification,
            if it is, return the 'scale' for that length, else None
            the scale is used to map ideal slice bit indices to actual
            """
        # ToDo: this is constant - move it to __init__
        bits = self._make_slice_bits(Codec.SYNC_DIGIT)  # only digit we can guarantee is valid
        scale = length / len(bits)
        if scale < 1.0:
            # not enough length to make a slice
            return None
        return scale

    def _make_slice_index(self, length):
        """ make a bit index of the given length,
            a bit index is the virtual bit number to associate with every offset into a slice of 'length'
            the indices generated here match the bits produced by _make_slice_bits
            """

        scale = self._get_slice_scale(length)
        if scale is None:
            # not enough length to make a slice
            return None

        slice = [None for _ in range(length)]

        for entry in range(length):
            bit_num = entry / scale
            slice[entry] = bit_num

        return slice

    def qualify(self, actual: [int]):
        """ given a pixel slice, determine if it meets the minimum criteria required to be able to classify it,
            the minimum criteria is that there should be either nothing or a single 'pulse' of 1's with at least
            one zero on either end, also the sequence of 1's should have no significant holes in it,
            the coding scheme is such that there should only be single sequence of ones, but we must allow for noise,
            this function should be called before the slice is presented to classify(),
            returns True if the slice qualifies
            """

        def is_101(here: int, ones_length: int, zero_length: int, ones: [int]) -> bool:
            """ test for a 101 sequence in the ones array starting from here with ones_length and zero_length,
                ones is an integration of the 1 bits in a slice,
                so ones[b] - ones[a] is the number of 1's between a and b (a < b)
                """
            limit = len(ones)
            # check for the one's lead-in
            pre_ones_start_at = here
            pre_ones_end_at = pre_ones_start_at + ones_length
            if pre_ones_end_at > (limit - 1):
                # run out of room
                return False
            pre_ones = (ones[pre_ones_end_at] - ones[pre_ones_start_at]) / ones_length
            if pre_ones < Codec.ONES_THRESHOLD:
                # have not got a 1's lead-in run from here
                return False
            # check for the zero run (we keep extending it from the minimum until it fails)
            zero_start_at = pre_ones_end_at - 1
            found_zero = False
            while True:
                zero_start_at += 1  # keep extending until it fails
                zero_end_at = zero_start_at + zero_length
                if zero_end_at > (limit - 1):
                    # run out of room
                    return False
                zeroes = 1 - ((ones[zero_end_at] - ones[zero_start_at]) / zero_length)
                if zeroes < Codec.ZERO_THRESHOLD:
                    # have not got a 0's run from here
                    if not found_zero:
                        # we did not find a minimum zero length
                        return False
                    # we've now found the end of the 0 run, so go look for the end ones run
                    zero_start_at -= 1  # backup to last good position
                    zero_end_at = zero_start_at + zero_length
                    break
                # note we found a minimum length zero run
                found_zero = True
            # check for the ones lead-out
            post_ones_start_at = zero_end_at
            post_ones_end_at = post_ones_start_at + ones_length
            if post_ones_end_at > (limit - 1):
                # run out of room
                return False
            post_ones = (ones[post_ones_end_at] - ones[post_ones_start_at]) / ones_length
            if post_ones < Codec.ONES_THRESHOLD:
                # have not got a 1's lead-out run
                return False
            # we've got a 101
            return True

        length = len(actual)
        if self._get_slice_scale(length) is None:
            # not enough room for a slice
            return False

        ring_width = length / Codec.SPAN
        if ring_width < 1:
            # length too small to differentiate each ring
            return False

        ones_length = int(round(ring_width * Codec.ONES_RUN_LENGTH))
        if ones_length < 2:
            # slice not big enough to properly qualify
            return True
        zero_length = int(round(ring_width * Codec.ZERO_RUN_LENGTH))
        if zero_length < 2:
            # slice not big enough to properly qualify
            return True
        data_length = int(round(ring_width * Codec.RINGS_PER_DIGIT))

        # the objective of this function is to detect slices that could be construed as a '5' (i.e. 2 pulses)
        # or a '7' (i.e. too big a pulse), the detection here differs from classify() in that here we are not
        # concerned with ring boundaries, we look for bit boundaries of the form ..1..1[0..0]1..1.. where the
        # [0..0] portion is a significant fraction of a ring width and is 'mostly' 0's.

        # run a sliding window across the data bits looking for a 101 pattern
        # integrate the slice (so we can get a 1's count between any 2 indices by a simple subtract)
        count = 0
        ones = [None for _ in range(length)]  # how many ones there are *before* each x
        for x in range(int(ring_width)+1):
            # no more 1's in the outer ring
            ones[x] = 0
        for x in range(int(ring_width)+1, length - int(ring_width) + 1):
            count += actual[x-1]
            ones[x] = count
        for x in range(length - int(ring_width) + 1, length):
            # no more 1's in the outer ring
            ones[x] = ones[x-1]
        # test for edge cases
        if count <= int((ring_width * Codec.RINGS_PER_DIGIT) * Codec.MIN_SLICE_WHITE):
            # its all, or mostly, black, always OK (its a 0)
            return True
        if count >= int(round((ring_width * Codec.RINGS_PER_DIGIT) * Codec.MAX_SLICE_WHITE)):
            # its all, or mostly, white, always crap (its a 7)
            return False
        # test every offset in the data portion of a 101 sequence
        start_data = int(ring_width * Codec.INNER_BLACK_RINGS)
        end_data = min(start_data + data_length, length)
        for x in range(start_data, end_data):
            if is_101(x, ones_length, zero_length, ones):
                # got a 101 sequence, this is crap (its a 5)
                return False

        return True

    def classify(self, actual: [int], source_error: float=0.0):
        """ given a pixel slice, return a list of the most likely digits it represents with an error,
            the pixel list must be a list of 0's, 1's or None's and must have at least SPAN bits,
            every viable digit is returned in least error first order, the list may be empty
            all errors are in the range 0..1, with 0=perfect and 1=utter crap,
            the given source error is just added to the digit error (and capped at 1.0)
            """

        def compare_pixel_slices(ideal, actual, spans, weights):
            """ compare ideal and actual slices,
                ideal is an array of bits for a digit, 1 bit per 'virtual ring',
                actual is an array of N bits that has been extracted from some image,
                spans is a mapping of bit indices to a set of actual indices to be matched for each bit,
                the length of spans must be the same as ideal,
                spans entries must be valid actual indices,
                weights is a weight factor for each bit error (0..1, 0=ignore, 1=use full),
                slices consist of 0's, 1's and None's, a None means do-not-care and are ignored,
                returns an error for the 0 bits and 1 bits, both in range 0..1 (0=no error, 1=utter crap),
                the term 'virtual ring' is used here to reflect the fact that this function operates on
                rings as implied by the length of ideal and may bear no resemblance to the code rings
                """

            zero_errors = [0.0 for _ in range(len(ideal))]  # init to 0 so skipped bits do not distort the result
            ones_errors = [0.0 for _ in range(len(ideal))]  # ditto

            for bit, ideal_value in enumerate(ideal):
                span_start, span_end = spans[bit]
                zero_samples = 0
                zero_matches = 0
                one_samples = 0
                one_matches = 0
                for entry in range(span_start, span_end + 1):
                    actual_value = actual[entry]
                    if actual_value is None:
                        continue
                    if ideal_value == 0:
                        zero_samples += 1
                        if actual_value == 0:
                            zero_matches += 1
                    elif ideal_value == 1:
                        one_samples += 1
                        if actual_value == 1:
                            one_matches += 1
                if zero_samples > 0:
                    zero_errors[bit] = 1 - (zero_matches / zero_samples)  # range 0..1 (0==good, 1==crap)
                else:
                    zero_errors[bit] = 0.0  # this means ideal has no zeroes, so we cannot have a zero error
                if one_samples > 0:
                    ones_errors[bit] = 1 - (one_matches / one_samples)  # range 0..1 (0==good, 1==crap)
                else:
                    ones_errors[bit] = 0.0  # this means ideal has no ones, so we cannot have a ones error
                zero_errors[bit] *= weights[bit]
                ones_errors[bit] *= weights[bit]

            zero_error = min(sum(zero_errors), 1.0)  # range 0..1
            ones_error = min(sum(ones_errors), 1.0)  # range 0..1

            pass  # only here as a debug hook, as placing it on the return gets missed :-(

            return zero_error, ones_error

        def error_acceptable(error, samples):
            """ test if the given error is acceptable for the given sample size,
                error is in the range 0..1, 0==no bits wrong, 1==all wrong,
                'acceptable' is at least MIN_CORRECT_SAMPLES bits are correct,
                less than that is interpreted as junk
                """
            if samples < (Codec.MIN_CORRECT_SAMPLES + 1):
                # not enough samples to set a meaningful max, so anything is OK
                return True
            max_error = 1 - (Codec.MIN_CORRECT_SAMPLES / samples)
            if error > max_error:
                return False
            else:
                return True

        # ToDo: this is constant - move it to __init__
        # build the ideal pixel list for each digit possibility
        ideals = [None for _ in range(Codec.DIGIT_BASE)]
        for digit in range(Codec.DIGIT_BASE):
            # comment here purely so the IDE will collapse this one line for loop :-(
            ideals[digit] = self._make_slice_bits(digit)

        # get the ideal bit index and split into a start/stop pair for each bit
        index = self._make_slice_index(len(actual))
        if index is None:
            # not enough length
            return []
        bit_edges = [[None, None] for _ in range(len(ideals[Codec.SYNC_DIGIT]))]
        for i, bit in enumerate(index):  # NB: bit index can be fractional
            start_bit_num = max(int(bit), 0)  # truncate fractional for the start
            end_bit_num = min(int(round(bit)), len(bit_edges) - 1)  # round for the end
            for edge_bit in [start_bit_num, end_bit_num]:
                if bit_edges[edge_bit][0] is None:
                    bit_edges[edge_bit][0] = i  # set start edge for this bit
                bit_edges[edge_bit][1] = i  # set stop edge for this bit

        # test actual to ideal match for every digit
        digits = []
        ring_width = len(actual) / Codec.SPAN
        for digit in range(Codec.DIGIT_BASE):
            zero_error, ones_error = compare_pixel_slices(ideals[digit], actual, bit_edges, Codec.BIT_WEIGHTS)
            if error_acceptable(zero_error, ring_width) and error_acceptable(ones_error, ring_width):
                # we only want to keep classifications that are 'reasonable'
                # NB: adding errors is correct as their spans are independent, the result is still 0..1
                error = zero_error + ones_error
                if error > 1.0:
                    raise Exception('Zero error ({:.2f}) plus ones error ({:.2f}) is greater than 1.0 for digit {}'.
                                    format(zero_error, ones_error, digit))
                error *= Codec.ENCODING[digit][2]  # scale by the digit weight
                if error > 1.0:
                    error = 0.999  # what else can we do?
                    # raise Exception('Scaled error ({:.2f}) is greater than 1.0 for digit {} (scale {})'.
                    #                 format(error, digit, Codec.ENCODING[digit][2]))
                digits.append((digit, min(error + source_error, 1.0)))

        # put into least error first order
        digits.sort(key=lambda d: d[1])

        return digits

    def digits(self, code):
        """ given a code return the digits for that code """
        partial = [None for _ in range(Codec.DIGITS_PER_WORD)]
        if code is not None:
            for digit in range(Codec.DIGITS_PER_WORD):
                partial[digit] = code % Codec.DIGIT_BASE
                code = int(code / Codec.DIGIT_BASE)
        return partial

    def digit(self, rings):
        """ given a set of data rings for a single digit return the corresponding digit,
            it is, in a sense, the opposite to _rings for a single digit
            """
        for digit, (allowed, coding, weight) in enumerate(Codec.ENCODING):
            if not allowed:
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
        if this_digit is None:
            return False
        if this_digit == Codec.SYNC_DIGIT:
            return True
        return False

    @staticmethod
    def coding(digit, only_if_allowed=True):
        """ given a digit return the coding for that digit or None if its a not allowed digit,
            if only_if_allowed is True allowed digit coding is returned else None is returned,
            if only_if_allowed is False the coding is returned regardless of its allowed status,
            an out of range digit always return None
            """
        if digit < 0 or digit >= len(Codec.ENCODING):
            return None
        allowed, coding, weight = Codec.ENCODING[digit]
        if only_if_allowed and not allowed:
            return None
        return coding

    def _rings(self, code_block):
        """ build the data ring encoding for the given code-block,
            code_block must be a list of digits in the range 0..BASE-1,
            the Nth bit of each digit is encoded into the Nth data ring,
            returns a list of integers representing each ring
            """

        # build code block (regardless of the validity of the given digits)
        rings = [0 for _ in range(Codec.RINGS_PER_DIGIT)]
        data = 1 << (Codec.DIGITS - 1)  # start at the msb
        for digit in code_block:
            coding = self.coding(digit, only_if_allowed=False)  # we want the coding regardless of it being allowed
            for ring, bit in enumerate(coding):
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
        has_white = [0 for _ in range(Codec.RINGS_PER_DIGIT)]
        has_black = [0 for _ in range(Codec.RINGS_PER_DIGIT)]
        for this in range(len(digits)):
            this_digit = digits[this]
            coding = self.coding(this_digit)
            if coding is None:
                # not allowed to use this digit
                return False
            if self.is_sync_digit(this_digit):
                if this != 0:
                    # does not meet only first digit can be the sync digit requirement
                    return False
            elif this == 0:
                # does not meet first digit must be the sync digit requirement
                return False
            before_digit = digits[(this - 1) % len(digits)]
            if this_digit == before_digit:
                # does not meet consecutive digits must be different requirement
                return False
            for ring in range(Codec.RINGS_PER_DIGIT):
                # consecutive digits cannot be the same
                # no black bit is allowed to be fully surrounded by white
                # because it can disappear in neighbour smudges, neighbours that are white smudge the corner,
                # if all 4 corners get smudged the whole thing can disappear!
                if ring == 0 or ring == (Codec.RINGS_PER_DIGIT - 1):
                    # we know there is black above the first ring and below the last ring, so not a problem here
                    pass
                elif coding[ring] == 0:
                    # there are 4 neighbours here, at least one must be black (so at least one corner survives)
                    black_neighbours = 0
                    for x, y in [          (0, -1),
                                 (-1,  0),          (+1,  0),
                                           (0, +1)          ]:
                        test_digit = digits[(this + x) % len(digits)]
                        test_ring = ring + y
                        bits = self.coding(test_digit)
                        if bits is None:
                            # not a legal digit
                            return None
                        if bits[test_ring] == 0:
                            black_neighbours += 1
                    if black_neighbours < 1:
                        # does not meet black neighbours requirement
                        return False
                # there must be at least one black and one white bit per ring,
                # to prevent continuous edges around a ring that could be confused with the target inner/outer edges
                if coding[ring] == 0:
                    has_black[ring] += 1
                if coding[ring] == 1:
                    has_white[ring] += 1
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
