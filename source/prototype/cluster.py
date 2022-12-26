""" decode digits in an extent

    the 'extent' describes the target as the inner and outer edges
    (i.e. where the inner and outer black rings are)
    this module analyses the pixels between these edges to find the likely digit sequence

"""

# ToDo: different approach:
#       after find inner/outer extent - extract target and flatten (in greyscale)
#       then threshold that (tighter)
#       then find ring (radial) edges by clustering rising and falling edge offsets
#       then find bit (annular) edges by clustering rising and falling edge offsets
#       now got a rectangle for each cell, count pixels in those and threshold for 0/1
#       job done!

import const
import structs
import codec

class Cluster:
    """ provide functions to detect digits within an extent """

    # region constants...
    DIGIT_BASE = codec.Codec.DIGIT_BASE  # number base for our digits
    NUM_SEGMENTS = codec.Codec.DIGITS  # total number of segments in a ring (segment==cell in length)
    COPIES = codec.Codec.COPIES_PER_BLOCK  # number of copies in a code-word
    DIGITS_PER_NUM = codec.Codec.DIGITS_PER_WORD  # how many digits per encoded number
    MAX_NOT_ALLOWED_ERROR_DIFF = 0.15  # a not-allowed choice error within this of its neighbour is noise, else junk
    MAX_DIGIT_ERROR = 0.5  # digits with an error of more than this are dropped
    MAX_ZERO_ERROR_DIFF = 0.25  # a zero with a choice with a smaller error difference than this is dropped
    MAX_DIGIT_ERROR_DIFF = 0.05  # if two digit choices have an error difference less than this its ambiguous
    MAX_DIGIT_WIDTH = 2.0  # maximum width of a keep-able digit relative to the nominal digit width
    MIN_DIGIT_WIDTH = 0.3  # minimum width of a keep-able digit relative to the nominal digit width
    MIN_DROPPABLE_WIDTH = MIN_DIGIT_WIDTH * 3  # minimum width of a droppable digit
    MIN_SPLITTABLE_DIGIT_WIDTH = MIN_DIGIT_WIDTH * 2  # minimum width of a splittable digit (/2 must be >=1)
    # endregion

    def __init__(self, scanner):
        # dependency injection
        self.scan        = scanner
        self.logging     = scanner.logging
        self.save_images = scanner.save_images
        self._log        = scanner._log
        self._unload     = scanner._unload
        self.decoder     = scanner.decoder
        self.transform   = scanner.transform
        self._draw_lines = scanner._draw_lines

    # region helpers...
    @staticmethod
    def drop_illegal_digits(digits):
        """ the initial digit classification takes no regard of not allowed digits,
            this is so we can differentiate junk from noise, we filter those here,
            returns the modified digits list and a count of digits dropped,
            this is public to allow test harnesses access
            """
        dropped = 0
        not_allowed = True
        while not_allowed and len(digits) > 0:
            not_allowed = False
            for choice, (digit, error) in enumerate(digits):
                coding = codec.Codec.coding(digit)
                if coding is None:
                    # we've got a not-allowed classification,
                    # if the error difference between this and some other choices is small-ish
                    # consider it noise and drop just that classification,
                    # if the error difference is large-ish it means we're looking at junk, so drop the lot
                    if choice == 0:
                        if choice < (len(digits) - 1):
                            # we're the first choice and there is a following choice, check it
                            digit2, error2 = digits[choice + 1]
                            coding2 = codec.Codec.coding(digit2)
                            if coding2 is None:
                                # got another illegal choice, drop that
                                dropped += 1
                                del digits[choice + 1]
                            else:
                                diff = max(error, error2) - min(error, error2)
                                if diff < Cluster.MAX_NOT_ALLOWED_ERROR_DIFF:
                                    # its not a confident classification, so just drop this choice (as noise)
                                    dropped += 1
                                    del digits[choice]
                                else:
                                    # its confidently a not allowed digit, so drop the lot ('cos we're looking at junk)
                                    dropped += len(digits)
                                    digits = []
                        else:
                            # we're the only choice and illegal, drop it (not silent)
                            dropped += 1
                            del digits[choice]
                    else:
                        # not first choice, just drop it as a choice (silently)
                        del digits[choice]
                    not_allowed = True
                    break
        return digits, dropped

    @staticmethod
    def drop_bad_digits(digits):
        """ check the error of the given digits and drop those that are 'excessive',
            returns the modified digits list and a count of digits dropped,
            this is public to allow test harnesses access
            """
        dropped = 0
        for digit in range(len(digits) - 1, -1, -1):
            if digits[digit][1] > Cluster.MAX_DIGIT_ERROR:
                # error too big to be considered as real
                dropped += 1
                del digits[digit]
        return digits, dropped

    @staticmethod
    def drop_bad_zero_digit(digits):
        """ drop a zero if it has a choice with a small-ish error,
            returns the modified digits list and a count of digits dropped,
            this is public to allow test harnesses access
            """
        dropped = 0
        if len(digits) > 1:
            # there is a choice, check if first is a 0 with a 'close' choice
            digit1, error1 = digits[0]
            if digit1 == 0:
                digit2, error2 = digits[1]  # NB: we know error2 is >= error1 (due to earlier sort)
                diff = error2 - error1
                if diff < Cluster.MAX_ZERO_ERROR_DIFF:
                    # second choice is 'close' to first of 0, so treat 0 as dodgy and drop it
                    dropped += 1
                    del digits[0]
                    # there can only be one zero digit, so we're now done
        return digits, dropped

    @staticmethod
    def is_ambiguous(slice):
        """ test if the top choices in the given slice are ambiguous,
            this is public to allow test harnesses access
            """
        if len(slice) > 1:
            _, error1 = slice[0]
            _, error2 = slice[1]
            diff = max(error1, error2) - min(error1, error2)
            if diff < Cluster.MAX_DIGIT_ERROR_DIFF:
                # its ambiguous when only a small error difference
                return True
        return False

    @staticmethod
    def show_options(options):
        """ produce a formatted string to describe the given digit options,
            this is public to allow test harnesses access
            """
        if options is None:
            return 'None'
        if len(options) > 0:
            msg = ''
            for digit, error in options:
                msg = '{}, ({}, {:.2f})'.format(msg, digit, error)
            return msg[2:]
        else:
            return '()'

    @staticmethod
    def show_bits(slice):
        """ produce a formatted string for the given slice of bits,
            this is public to allow test harnesses access
            """
        ring_width = int(round(len(slice) / codec.Codec.SPAN))
        bits = ''
        for ring in range(codec.Codec.SPAN):
            block = ''
            for dx in range(ring_width):
                x = (ring * ring_width) + dx
                if x >= len(slice):
                    block = '{}.'.format(block)
                else:
                    block = '{}{}'.format(block, slice[x])
            bits = '{} {}'.format(bits, block)
        return '[{}]'.format(bits[1:])
    # endregion

    def find_all_digits(self, extent: structs.Extent) -> ([structs.Digit], str):
        """ find all the digits from an analysis of the given extent,
            returns a digit list and None or a partial list and a fail reason,
            the extent is also updated with the slices involved
            """

        if self.logging:
            header = 'find_all_digits:'

        buckets = extent.buckets
        max_x, max_y = buckets.size()
        reason = None

        # a 0 in these bit positions mean treat grey as black, else treat as white
        # these are AND masks, when we get a grey in the 'before', 'centre' or 'after' context in a slice
        # the result of the AND on the option being performed is used to determine what to do,
        # 0=treat as black, 1=treat as white
        GREY_BEFORE_MASK = 1  # grey option bit mask to specify how to handle grey before the first white
        GREY_CENTRE_MASK = 2  # grey option bit mask to specify how to handle grey between before and after white
        GREY_AFTER_MASK  = 4  # grey option bit mask to specify how to handle grey after the last white
        GREY_ONLY_MASK   = 8  # grey option bit mask to specify how to handle grey when there is no white
        GREY_BEFORE = 0  # GREY_BEFORE_MASK
        GREY_CENTRE = 0  # GREY_CENTRE_MASK
        GREY_AFTER  = 0  # GREY_AFTER_MASK
        GREY_ONLY   = GREY_ONLY_MASK

        def drop_dodgy_zero(x, option, digits, logging=False):
            nonlocal header
            if logging:
                before = Cluster.show_options(digits)
            digits, dropped = Cluster.drop_bad_zero_digit(digits)
            if logging and dropped > 0:
                if header is not None:
                    self._log(header)
                    header = None
                self._log('    {} (option {}): dropping {} zero choices from {} leaving {}'.
                          format(x, option, dropped, before, Cluster.show_options(digits)))
            return dropped

        def first_good_choice(slice):
            """ get the first choice in the given slice,
                if there are two or more choices and they are ambiguous, return None,
                if the top two choices are not ambiguous return the first choice,
                else return None,
                """
            if Cluster.is_ambiguous(slice):
                return None
            if len(slice) > 0:
                return slice[0]
            else:
                return None

        def show_grey_option(option):
            """ option is a bit mask of 3 bits, bit 0 = before white, 1 = between, 2 = after
                a 1 in that bit means treat grey as white and a 0 means treat grey as black
                """
            if option & GREY_BEFORE_MASK == 0:
                before = '0'
            else:
                before = '1'
            if option & GREY_CENTRE_MASK == 0:
                centre = '0'
            else:
                centre = '1'
            if option & GREY_AFTER_MASK == 0:
                after = '0'
            else:
                after = '1'
            if option & GREY_ONLY_MASK == 0:
                only = '0'
            else:
                only = '1'
            return '(grey {}{}{}{})'.format(before, centre, after, only)

        # region generate likely digit choices for each x...
        inner = extent.inner
        outer = extent.outer
        slices = []
        for x in range(max_x):
            start_y = inner[x] + 1  # get past the white bullseye, so start at first black
            end_y   = outer[x]      # this is the first white after the inner black
            # region get the raw pixels and their luminance edge extremes...
            pixels = []
            has_white = 0
            has_grey = 0
            first_white = None  # location of first transition from black or grey to white
            last_white = None  # location of last transition from white to black or grey
            this_pixel = buckets.getpixel(x, start_y)  # make sure we do not see a transition on the first pixel
            for y in range(start_y, end_y):
                prev_pixel = this_pixel
                dy = y - start_y
                this_pixel = buckets.getpixel(x, y)
                pixels.append(this_pixel)
                if this_pixel != prev_pixel:
                    # got a transition
                    if prev_pixel == const.MAX_LUMINANCE:
                        # got a from white transition
                        last_white = dy - 1
                    if this_pixel == const.MAX_LUMINANCE and first_white is None:
                        # got first to white transition
                        first_white = dy
                if this_pixel == const.MAX_LUMINANCE:
                    has_white += 1
                elif this_pixel == const.MID_LUMINANCE:
                    has_grey += 1
            # adjust for leading/trailing white
            if first_white is None:
                # there was no transition to white
                if last_white is None:
                    # there is no transition from white either
                    if has_white > 0:
                        # this means its all white
                        first_white = 0
                        last_white = len(pixels) - 1
                    else:
                        # this means its all black or grey
                        # set an out of range value, so we do not have to check for None in our later loops
                        first_white = len(pixels) + 1
                        last_white = first_white
                else:
                    # there is transition from white but not to white, this means its all white from the start
                    first_white = 0
            elif last_white is None:
                # there is a transition to white but not from white, that means it ran into the end
                last_white = len(pixels) - 1
            # endregion
            # region ToDo: HACK experiment on edge 'snapping' - does not work
            # options = []
            # # ToDo: do options to treat rising to mid as min or max and separate option for falling to mid
            # for mask, grey_as in ((GREY_BEFORE_MASK, const.MIN_LUMINANCE), (GREY_AFTER_MASK, const.MAX_LUMINANCE)):
            #     slice = Allocate(pixels, codec.Codec.SPAN, grey_as)
            #     options.append((mask, slice.bits, slice.error))
            # endregion ToDo: HACKEND
            # region build the options for the grey treatment...
            # the options are: grey before first white as black or white
            #                  grey after last white as black or white
            #                  grey between first and last white as black or white
            #                  grey when no white
            # we classify the pixels adjusted for these options, then pick the best,
            # this has the effect of sliding the central pulse up/down around the grey boundaries
            # there are 16 combinations of greys - before b/w * between b/w * after b/w * no white
            options = []
            for option in range(16):
                # option is a bit mask of 4 bits, bit 0 = before white, 1 = between, 2 = after, 3 = no white
                # a 1 in that bit means treat grey as white and a 0 means treat grey as black
                # NB: option 0 must be to consider *all* greys as black
                slice = []
                for y, pixel in enumerate(pixels):
                    if pixel == const.MID_LUMINANCE:
                        if has_white == 0:
                            if option & GREY_ONLY == 0:
                                # treat as black
                                pixel = const.MIN_LUMINANCE
                            else:
                                # treat as white
                                pixel = const.MAX_LUMINANCE
                        elif y < first_white:
                            if option & GREY_BEFORE == 0:
                                # treat before as black
                                pixel = const.MIN_LUMINANCE
                            else:
                                # treat before as white
                                pixel = const.MAX_LUMINANCE
                        elif y > last_white:
                            if option & GREY_AFTER == 0:
                                # treat after as black
                                pixel = const.MIN_LUMINANCE
                            else:
                                # treat after as white
                                pixel = const.MAX_LUMINANCE
                        else:  # if y >= first_white and y <= last_white:
                            if option & GREY_CENTRE == 0:
                                # treat between as black
                                pixel = const.MIN_LUMINANCE
                            else:
                                # treat between as white
                                pixel = const.MAX_LUMINANCE
                    if pixel == const.MIN_LUMINANCE:
                        slice.append(0)
                    elif pixel == const.MAX_LUMINANCE:
                        slice.append(1)
                    else:
                        # can't get here
                        raise Exception('Got unexpected MID_LUMINANCE')
                options.append((option, slice, 0.0))
                if has_grey == 0:
                    # there are no greys, so don't bother with the rest
                    break
            # endregion
            # region get rid of duplicates...
            if len(options) > 1:
                for option in range(len(options)-1, 0, -1):
                    _, this_bits, this_error = options[option]
                    for other in range(option):
                        _, other_bits, other_error = options[other]
                        if this_bits == other_bits:
                            if this_error < other_error:
                                # keep this, chuck other
                                del options[other]
                            else:
                                # chuck this, keep other
                                del options[option]
                            break
            # endregion
            # region build the digits for each option...
            slice = []
            if self.logging:
                prefix = '{:3n}:'.format(x)
                disqualified = []
                illegal = []
                big_error = []
                bad_zero = []
            for option, (_, bits, error) in enumerate(options):
                if not self.decoder.qualify(bits):
                    if self.logging:
                        disqualified.append(option)
                    slice.append([])  # set an empty digit list
                    continue
                raw_digits = self.decoder.classify(bits, error)
                digits = raw_digits.copy()  # protect original raw_digits for log messages
                # drop digits with an excessive error (reduces subsequent work load)
                digits, dropped = Cluster.drop_bad_digits(digits)
                # Don't log these, it generates too much noise with very little meaning
                # if self.logging and dropped > 0:
                #     big_error.append((option, raw_digits, dropped))
                # drop illegal digits
                digits, dropped = Cluster.drop_illegal_digits(digits)
                if self.logging and dropped > 0:
                    illegal.append((option, raw_digits, dropped))
                # drop a zero if it has a choice with a small-ish error
                dropped = drop_dodgy_zero(x, option, digits)
                if self.logging and dropped > 0:
                    bad_zero.append((option, raw_digits, dropped))
                slice.append(digits)
            # log what happened
            if self.logging:
                if len(disqualified) > 0 or len(illegal) > 0 or len(big_error) > 0 or len(bad_zero) > 0:
                    if header is not None:
                        self._log(header)
                        header = None
                    for option in disqualified:
                        mask, bits, error = options[option]
                        self._log('    {}{}(err={:.2f}) ignoring non-qualifying bits {}'.
                                  format(prefix, show_grey_option(mask), error, Cluster.show_bits(bits)))
                        prefix = '    '
                    for option, raw_digits, dropped in illegal:
                        mask, bits, error = options[option]
                        self._log('    {}{}(err={:.2f}) dropping {} illegal choices from {}'.
                                  format(prefix, show_grey_option(mask), error, dropped, Cluster.show_options(raw_digits)))
                        prefix = '    '
                    for option, raw_digits, dropped in big_error:
                        mask, bits, error = options[option]
                        self._log('    {}{}(err={:.2f}) dropping {} big error choices from {}'.
                                  format(prefix, show_grey_option(mask), error, dropped, Cluster.show_options(raw_digits)))
                        prefix = '    '
                    for option, raw_digits, dropped in bad_zero:
                        mask, bits, error = options[option]
                        self._log('    {}{}(err={:.2f}) dropping {} zero choices from {}'.
                                  format(prefix, show_grey_option(mask), error, dropped, Cluster.show_options(raw_digits)))
                        prefix = '    '
            # endregion
            # region merge the grey options...
            # get rid of dodgy 0's, a 'dodgy' 0 is one where option 0 is black but some other is not
            # NB: option 0 is where all greys are considered to be black
            first_digits = slice[0]
            if len(first_digits) > 0 and first_digits[0][0] == 0:
                # it thinks its a zero, so check others
                invalid = 0
                for option in range(1, len(slice)):
                    if len(slice[option]) == 0:
                        # this means this option had no valid digits, but for the classifier to say this there
                        # must be white or grey pixels present, so the 0 is in doubt, if all other options are
                        # also invalid, we drop the 0, so just count them here
                        invalid += 1
                        continue
                    other_digits = slice[option]
                    if other_digits[0][0] != 0:
                        # some other option thinks it is not 0, so its a dodgy 0, drop all in that slice
                        if self.logging:
                            if header is not None:
                                self._log(header)
                                header = None
                            self._log('    {}{} dropping zero slice: {} in favour of {}'.
                                      format(prefix, show_grey_option(option),
                                             Cluster.show_options(first_digits), Cluster.show_options(other_digits)))
                            prefix = '    '
                        del slice[0]
                        break
                if invalid > 0 and invalid == (len(slice) - 1):
                    # all other options are invalid, so do not trust the initial 0, drop it
                    if self.logging:
                        if header is not None:
                            self._log(header)
                            header = None
                        self._log('    {}{} dropping zero: {} all its other {} options are invalid'.
                                  format(prefix, show_grey_option(0), Cluster.show_options(first_digits), invalid))
                        prefix = '    '
                    del slice[0]
            # join all the (remaining) slices so we can find the best choice for each digit
            choices = []
            for digits in slice:
                for digit, error in digits:
                    choices.append((digit, error))
            choices.sort(key=lambda k: (k[0], k[1]))  # put like digits together in least error order
            for choice in range(len(choices) - 1, 0, -1):
                this_digit , _ = choices[choice]
                other_digit, _ = choices[choice-1]
                if this_digit == other_digit:
                    # duplicate digit, we're on the worst one, so dump that
                    del choices[choice]
            choices.sort(key=lambda k: k[1])  # put merged list into least error order
            drop_dodgy_zero(x, 0, choices, self.logging)  # JIC some percolated up
            # endregion
            slices.append(choices)
        # region check for and resolve ambiguous choices...
        # ambiguity can lead to false positives, which we want to avoid,
        # an ambiguous choice is when a digit has (nearly) equal error choices,
        # we try to find a choice that matches one of its non-ambiguous neighbours,
        # if found, resolve to that, otherwise drop the digit
        # there may be a series of consecutive ambiguities, in which case the ends of
        # series get resolved but not the centre, so we iterate until either the whole
        # series has been resolved or no more resolutions are possible, the remaining
        # ambiguous digits are dropped
        # region resolve ambiguities...
        resolutions = 1
        passes      = 0
        while resolutions > 0:
            resolutions = 0
            passes     += 1
            for x, slice in enumerate(slices):
                best_choice = first_good_choice(slice)
                if best_choice is None:
                    # got no, or an ambiguous, choice, check its neighbours
                    left_x = (x - 1) % max_x
                    right_x = (x + 1) % max_x
                    left_choice = first_good_choice(slices[left_x])
                    right_choice = first_good_choice(slices[right_x])
                    if left_choice is None and right_choice is None:
                        # both neighbours are ambiguous too, leave for next pass
                        continue
                    # there is scope to resolve this ambiguity
                    if self.logging:
                        if header is not None:
                            self._log(header)
                            header = None
                    # if only 1 choice, inherit the other (that makes them the same in comparisons below)
                    if left_choice is None:
                        left_choice = right_choice
                        left_x      = right_x
                    if right_choice is None:
                        right_choice = left_choice
                        right_x      = left_x
                    # choose best neighbour
                    left_digit , left_error  = left_choice
                    right_digit, right_error = right_choice
                    if len(slice) < 2:
                        # we've got no choice, so just inherit our best neighbour
                        if left_error < right_error:
                            if self.logging:
                                self._log('    {:3n}:(pass {}) resolving ambiguity of {} with neighbour {}: {}'.
                                          format(x, passes, Cluster.show_options(slices[x]),
                                                 left_x, Cluster.show_options(slices[left_x])))
                            slices[x] = [left_choice]
                        else:
                            if self.logging:
                                self._log('    {:3n}:(pass {}) resolving ambiguity of {} with neighbour {}: {}'.
                                          format(x, passes, Cluster.show_options(slices[x]),
                                                 right_x, Cluster.show_options(slices[right_x])))
                            slices[x] = [right_choice]
                        resolutions += 1
                        continue
                    # we have choices to resolve
                    digit1, error1 = slice[0]
                    digit2, error2 = slice[1]
                    left_match  = (digit1 == left_digit or digit2 == left_digit)
                    right_match = (digit1 == right_digit or digit2 == right_digit)
                    if left_match and right_match:
                        # both sides match, pick the one with the least error
                        if left_error < right_error:
                            # left is best, so drop right
                            right_match = False
                        else:
                            # right is best so drop left
                            left_match = False
                    if left_match:
                        # got a left match use it
                        if self.logging:
                            self._log('    {:3n}:(pass {}) resolving ambiguity of {} with neighbour {}: {}'.
                                      format(x, passes, Cluster.show_options(slices[x]),
                                             left_x, Cluster.show_options(slices[left_x])))
                        if digit1 == left_digit:
                            # first choice is best, dump the 2nd
                            del slice[1]
                        else:
                            # second choice is best, dump the first
                            del slice[0]
                        resolutions += 1
                        continue
                    if right_match:
                        # got a right match use it
                        if self.logging:
                            self._log('    {:3n}:(pass {}) resolving ambiguity of {} with neighbour {}: {}'.
                                      format(x, passes, Cluster.show_options(slices[x]),
                                             right_x, Cluster.show_options(slices[right_x])))
                        if digit1 == right_digit:
                            # first choice is best, dump the second
                            del slice[1]
                        else:
                            # second choice is best, dump the first
                            del slice[0]
                        resolutions += 1
                        continue
                    # neither side matches, leave to next pass
                    continue
        # endregion
        # region do final pass to drop remaining ambiguous choices
        for x, slice in enumerate(slices):
            if Cluster.is_ambiguous(slice):
                # unresolvable ambiguity - drop it
                if self.logging:
                    if header is not None:
                        self._log(header)
                        header = None
                    self._log('    {:3n}:(pass {}) dropping unresolvable ambiguous choices: {}'.
                              format(x, passes, Cluster.show_options(slices[x])))
                slices[x] = []
        if self.logging and header is None:
            self._log('    {} ambiguity passes'.format(passes))
        # endregion
        # endregion
        # region resolve singletons...
        # a singleton is a single slice bordered both sides by some valid digit
        # these are considered to be noise and inherit their best neighbour
        for x, this_slice in enumerate(slices):
            left_x = (x - 1) % max_x
            right_x = (x + 1) % max_x
            left_slice = slices[left_x]
            right_slice = slices[right_x]
            if len(left_slice) == 0 or len(right_slice) == 0:
                # we have not got both neighbours
                continue
            left_digit, left_error = left_slice[0]
            right_digit, right_error = right_slice[0]
            if len(this_slice) == 0:
                # inherit best neighbour
                pass
            else:
                this_digit, _ = this_slice[0]
                if left_digit == this_digit or right_digit == this_digit:
                    # not a potential singleton
                    continue
            # inherit best neighbour
            # do not inherit sync unless both sides are sync
            if self.decoder.is_sync_digit(left_digit) and self.decoder.is_sync_digit(right_digit):
                # both sides are sync, so allow inherit
                pass
            elif self.decoder.is_sync_digit(left_digit):
                # do not allow inherit of left
                left_slice = right_slice
                left_x = right_x
            else:  # self.decoder.is_sync_digit(right_digit):
                # do not allow inherit of right
                right_slice = left_slice
                right_x = left_x
            if self.logging:
                if header is not None:
                    self._log(header)
                    header = None
            if left_error < right_error:
                if self.logging:
                    self._log('    {:3n}: resolve singleton {} to {}: {}'.
                              format(x, Cluster.show_options(this_slice),
                                     left_x, Cluster.show_options(left_slice)))
                slices[x] = left_slice
            else:
                if self.logging:
                    self._log('    {:3n}: resolve singleton {} to {}: {}'.
                              format(x, Cluster.show_options(this_slice),
                                     right_x, Cluster.show_options(right_slice)))
                slices[x] = right_slice
            continue
        # endregion
        if self.logging:
            if header is not None:
                self._log(header)
                header = None
            self._log('    initial slices (fail reason {}):'.format(reason))
            for x, options in enumerate(slices):
                self._log('        {:3n}: options={}'.format(x, Cluster.show_options(options)))
        # endregion
        # region build digit list...
        digits = []
        last_digit = None
        for x, options in enumerate(slices):
            if len(options) == 0:
                # this is junk - treat like an unknown digit
                best_digit = None
                best_error = 1.0
            else:
                # we only look at the best choice and only iff the next best has a worse error
                best_digit, best_error = options[0]
            if last_digit is None:
                # first digit
                last_digit = structs.Digit(best_digit, best_error, x, 1)
            elif best_digit == last_digit.digit:
                # continue with this digit
                last_digit.error += best_error  # accumulate error
                last_digit.samples += 1         # and sample count
            else:
                # save last digit
                last_digit.error /= last_digit.samples  # set average error
                digits.append(last_digit)
                # start a new digit
                last_digit = structs.Digit(best_digit, best_error, x, 1)
        # deal with last digit
        if last_digit is None:
            # nothing to see here...
            reason = 'no digits'
        elif len(digits) == 0:
            # its all the same digit - this must be junk
            last_digit.error /= last_digit.samples  # set average error
            digits = [last_digit]
            reason = 'single digit'
        elif last_digit.digit == digits[0].digit:
            # its part of the first digit
            last_digit.error /= last_digit.samples  # set average error
            digits[0].error = (digits[0].error + last_digit.error) / 2
            digits[0].start = last_digit.start
            digits[0].samples += last_digit.samples
        else:
            # its a separate digit
            last_digit.error /= last_digit.samples  # set average error
            digits.append(last_digit)
        # endregion

        # save slices in the extent for others to use  ToDo: get rid of the need for this
        extent.slices = slices

        if self.logging:
            if header is not None:
                self._log(header)
                header = None
            self._log('    {} digits (fail reason {}):'.format(len(digits), reason))
            for item, digit in enumerate(digits):
                self._log('        {:2n}: {}'.format(item, digit))

        if self.save_images:
            plot = self.transform.copy(buckets)
            ones = []
            zeroes = []
            for x, options in enumerate(slices):
                if len(options) == 0:
                    continue
                start_y = inner[x] + 1
                end_y = outer[x]
                slice = self.decoder.make_slice(options[0][0], end_y - start_y)
                if slice is None:
                    continue
                last_bit = None
                for dy, bit in enumerate(slice):
                    if bit is None:
                        continue
                    if last_bit is None:
                        last_bit = (dy, bit)
                    elif bit != last_bit[1]:
                        # end of a run
                        if last_bit[1] == 0:
                            zeroes.append((x, last_bit[0] + start_y, x, start_y + dy - 1))
                        else:
                            ones.append((x, last_bit[0] + start_y, x, start_y + dy - 1))
                        last_bit = (dy, bit)
                if last_bit[1] == 0:
                    zeroes.append((x, last_bit[0] + start_y, x, end_y - 1))
                else:
                    ones.append((x, last_bit[0] + start_y, x, end_y - 1))
            plot = self._draw_lines(plot, ones, colour=const.RED)
            plot = self._draw_lines(plot, zeroes, colour=const.GREEN)
            self._unload(plot, '05-slices')

        return digits, reason

    def find_best_digits(self, digits: [structs.Digit], extent: structs.Extent = None) -> ([structs.Digit], str):
        """ analyse the given digits to isolate the 'best' ones,
            extent provides the slices found by _find_all_digits,
            the 'best' digits are those that conform to the known code structure,
            specifically one zero per copy and correct number of digits per copy,
            returns the revised digit list and None if succeeded
            or a partial list and a reason if failed
            """

        buckets = extent.buckets
        max_x, max_y = buckets.size()
        nominal_digit_width = (max_x / Cluster.NUM_SEGMENTS)
        max_digit_width = nominal_digit_width * Cluster.MAX_DIGIT_WIDTH
        min_digit_width = nominal_digit_width * Cluster.MIN_DIGIT_WIDTH
        min_splittable_digit_width = nominal_digit_width * Cluster.MIN_SPLITTABLE_DIGIT_WIDTH
        min_droppable_digit_width = nominal_digit_width * Cluster.MIN_DROPPABLE_WIDTH
        reason = None  # this is set to a reason mnemonic if we fail

        if self.logging:
            header = 'find_best_digits: digit width: nominal={:.2f}, limits={:.2f}..{:.2f}, ' \
                     'min splittable: {:.2f}, droppable: {:.2f}'.\
                     format(nominal_digit_width, min_digit_width, max_digit_width,
                            min_splittable_digit_width, min_droppable_digit_width)

        def find_best_2nd_choice(slices, slice_start, slice_end):
            """ find the best 2nd choice in the given slices,
                return the digit, how many there are and the average error,
                the return info is sufficient to create a Scan.Digit
                """
            # ToDo: this should be a separate function and/or part of find_all_digits,
            #       as is here, the structure of slices is bleeding too far
            second_choice = [[0, 0] for _ in range(Cluster.DIGIT_BASE)]
            for x in range(slice_start, slice_end):
                options = slices[x % len(slices)]
                if len(options) > 1:
                    # there is a 2nd choice
                    digit, error = options[1]
                    second_choice[digit][0] += 1
                    second_choice[digit][1] += error
                elif len(options) > 0:
                    # no second choice, use the first
                    digit, error = options[0]
                else:
                    # nothing here at all
                    continue
                second_choice[digit][0] += 1
                second_choice[digit][1] += error
            best_digit = None
            best_count = 0
            best_error = 0
            for digit, (count, error) in enumerate(second_choice):
                if count > best_count:
                    best_digit = digit
                    best_count = count
                    best_error = error
            if best_count > 0:
                best_error /= best_count
            return best_digit, best_count, best_error

        def shrink_digit(slices, digit, start, samples) -> structs.Digit:
            """ shrink the indicated digit to the start and size given,
                this involves moving the start, updating the samples and adjusting the error,
                the revised digit is returned
                """
            # calculate the revised error
            error = 0
            for x in range(start, start + samples):
                options = slices[x % len(slices)]
                if len(options) == 0:
                    # treat nothing as a worst case error
                    error += 1.0
                else:
                    error += options[0][1]
            if samples > 0:
                error /= samples
            # create a new digit
            new_digit = structs.Digit(digit.digit, error, start % len(slices), samples)
            return new_digit

        # translate digits from [Digit] to [[Digit]] so we can mess with it and not change indices,
        # indices into digits_list must remain constant even while we are adding/removing digits
        # we achieve this by having a list of digits for each 'digit', that list is emptied when
        # a digit is removed or extended when a digit is added, for any index 'x' digits[x] is
        # the original digit and digits_list[x] is a list of 1 or 2 digits or None, None=removed,
        # 2=split, and 1=no change
        digits_list = [[options] for options in digits]

        # we expect to find Scan.COPIES of the sync digit
        copies = []
        for x, digit in enumerate(digits):
            if self.decoder.is_sync_digit(digit.digit):
                copies.append([x])
        if len(copies) < Cluster.COPIES:
            # not enough sync digits - that's a show-stopper
            reason = 'too few syncs'
        else:
            # if too many sync digits - dump the smallest droppable sync digit with the biggest error
            while len(copies) > Cluster.COPIES:
                smallest_x = None  # index into copy of the smallest sync digit
                for x in range(len(copies)):
                    xx = copies[x][0]
                    if digits[xx].samples >= min_droppable_digit_width:
                        # too big to drop
                        continue
                    if smallest_x is None:
                        smallest_x = x
                        continue
                    smallest_xx = copies[smallest_x][0]
                    if digits[xx].samples < digits[smallest_xx].samples:
                        # less samples
                        smallest_x = x
                    elif digits[xx].samples == digits[smallest_xx].samples:
                        if digits[xx].error > digits[smallest_xx].error:
                            # same samples but bigger error
                            smallest_x = x
                if smallest_x is None:
                    # nothing small enough to drop - that's a show stopper
                    reason = 'too many syncs'
                    break
                smallest_xx = copies[smallest_x][0]
                if self.logging:
                    if header is not None:
                        self._log(header)
                        header = None
                    self._log('    {}: dropping excess sync {}'.
                              format(smallest_xx, digits[smallest_xx]))
                digits_list[smallest_xx] = None
                del copies[smallest_x]
        if reason is None:
            # find the actual digits between the syncs
            for copy_num, copy in enumerate(copies):
                xx = copy[0]
                while True:
                    xx = (xx + 1) % len(digits)
                    if digits_list[xx] is None:
                        # this one has been dumped
                        continue
                    if self.decoder.is_sync_digit(digits[xx].digit):
                        # ran into next copy
                        break
                    copy.append(xx)
                digits_required = Cluster.DIGITS_PER_NUM
                while len(copy) < digits_required:
                    # not enough digits - split the biggest with the biggest error
                    biggest_x = None
                    for x in range(1, len(copy)):  # never consider the initial 0
                        xx = copy[x]
                        digits_xx = digits_list[xx]
                        if digits_xx is None:
                            # this one has been dumped
                            continue
                        if len(digits_xx) > 1:
                            # this one has already been split
                            continue
                        if biggest_x is None:
                            # found first split candidate
                            biggest_x = x
                            biggest_digit = digits[copy[biggest_x]]
                            if biggest_digit.samples < min_splittable_digit_width:
                                # too small to split:
                                biggest_x = None
                            continue
                        biggest_digit = digits[copy[biggest_x]]
                        if digits[xx].samples > biggest_digit.samples:
                            # found a bigger candidate
                            biggest_x = x
                        elif digits[xx].samples == biggest_digit.samples:
                            if digits[xx].error > biggest_digit.error:
                                # find a same size candidate with a bigger error
                                biggest_x = x
                    if biggest_x is None:
                        # everything splittable has been split and still not enough - this is a show stopper
                        reason = 'too few digits'
                        break
                    biggest_xx = copy[biggest_x]
                    biggest_digit = digits[biggest_xx]
                    # we want to split the biggest using the second choice in the spanned slices
                    # count 2nd choices in the first half and second half of the biggest span
                    # use the option with the biggest count, this represents the least error 2nd choice
                    # this algorithm is very crude and is only reliable when we are one digit short
                    # this is the most common case, eg. when a 100 smudges into a 010
                    # we only split a sequence once so digits[x] and digits_list[x][0] are the same here
                    slices = extent.slices
                    slice_start = digits[biggest_xx].start
                    slice_full_span = digits[biggest_xx].samples
                    slice_first_span = int(round(slice_full_span / 2))
                    slice_second_span = slice_full_span - slice_first_span
                    best_1 = find_best_2nd_choice(slices, slice_start, slice_start + slice_first_span)
                    best_2 = find_best_2nd_choice(slices, slice_start + slice_first_span, slice_start + slice_full_span)
                    if best_1[1] > best_2[1]:
                        # first half is better, create a digit for that, insert before the other and shrink the other
                        digit_1 = structs.Digit(best_1[0], best_1[2], slice_start, slice_first_span)
                        digit_2 = shrink_digit(slices, biggest_digit, slice_start + slice_first_span, slice_second_span)
                    else:
                        # second half is better, create a digit for that
                        digit_1 = shrink_digit(slices, biggest_digit, slice_start, slice_first_span)
                        digit_2 = structs.Digit(best_2[0], best_2[2],
                                             (slice_start + slice_first_span) % len(slices), slice_second_span)
                    if self.logging:
                        if header is not None:
                            self._log(header)
                            header = None
                        self._log('    {}: splitting {} into {} and {}'.
                                  format(biggest_xx, biggest_digit, digit_1, digit_2))
                    digits_list[biggest_xx] = [digit_1, digit_2]
                    digits_required -= 1
                while len(copy) > digits_required:
                    # too many digits - drop the smallest with the biggest error that is not a sync digit
                    smallest_x = None
                    for x in range(1, len(copy)):  # never consider the initial sync digit
                        xx = copy[x]
                        digits_xx = digits_list[xx]
                        if digits_xx is None:
                            # this one has been dumped
                            continue
                        if len(digits_xx) > 1:
                            # this one has been split - not possible to see that here!
                            continue
                        if smallest_x is None:
                            # found first dump candidate
                            smallest_x = x
                            smallest_digit = digits[copy[smallest_x]]
                            if smallest_digit.samples >= min_droppable_digit_width:
                                # too big to drop
                                smallest_x = None
                            continue
                        smallest_digit = digits[copy[smallest_x]]
                        if digits[xx].samples < smallest_digit.samples:
                            # found a smaller candidate
                            smallest_x = x
                        elif digits[xx].samples == smallest_digit.samples:
                            if digits[xx].error > smallest_digit.error:
                                # found a similar size candidate with a bigger error
                                smallest_x = x
                    if smallest_x is None:
                        # nothing (left) small enough to drop, that's a show stopper
                        reason = 'too many digits'
                        break
                    smallest_xx = copy[smallest_x]
                    if self.logging:
                        if header is not None:
                            self._log(header)
                            header = None
                        self._log('    {}: dropping excess digit {}'.
                                  format(smallest_xx, digits[smallest_xx]))
                    digits_list[smallest_xx] = None
                    digits_required += 1
                if reason is not None:
                    # we've given up someplace
                    if self.logging:
                        continue  # carry on to check other copies so logs are more intelligible
                    else:
                        break  # no point looking at other copies

        # build the final digit list
        best_digits = []
        for digits in digits_list:
            if digits is None:
                # this has been deleted
                continue
            for digit in digits:
                best_digits.append(digit)

        if reason is None:
            # check we're not left with digits that are too small or too big
            for digit in best_digits:
                if digit.samples > max_digit_width:
                    # too big
                    reason = 'digit too big'
                    break
                elif digit.samples < min_digit_width:
                    if self.decoder.is_sync_digit(digit.digit):
                        # we tolerate small syncs
                        pass
                    else:
                        # too small
                        reason = 'digit too small'
                        break

        if self.logging:
            if header is not None:
                self._log(header)
                header = None
            self._log('    {} digits (fail reason: {}):'.format(len(best_digits), reason))
            for x, digit in enumerate(best_digits):
                self._log('        {}: {}'.format(x, digit))

        if self.save_images:
            buckets = extent.buckets
            max_x, _ = buckets.size()
            plot = self.transform.copy(buckets)
            ones = []
            zeroes = []
            nones = []
            for digit in best_digits:
                for x in range(digit.start, digit.start + digit.samples):
                    x %= max_x
                    start_y = extent.inner[x] + 1
                    end_y = extent.outer[x]
                    digit_slice = self.decoder.make_slice(digit.digit, end_y - start_y)
                    if digit_slice is None:
                        # this is a 'None' digit - draw those as blue
                        nones.append((x, start_y, x, end_y - 1))
                        continue
                    last_bit = None
                    for dy, bit in enumerate(digit_slice):
                        if bit is None:
                            continue
                        if last_bit is None:
                            last_bit = (dy, bit)
                        elif bit != last_bit[1]:
                            # end of a run
                            if last_bit[1] == 0:
                                zeroes.append((x, last_bit[0] + start_y, x, start_y + dy - 1))
                            else:
                                ones.append((x, last_bit[0] + start_y, x, start_y + dy - 1))
                            last_bit = (dy, bit)
                    if last_bit[1] == 0:
                        zeroes.append((x, last_bit[0] + start_y, x, end_y - 1))
                    else:
                        ones.append((x, last_bit[0] + start_y, x, end_y - 1))
            plot = self._draw_lines(plot, nones, colour=const.BLUE)
            plot = self._draw_lines(plot, ones, colour=const.RED)
            plot = self._draw_lines(plot, zeroes, colour=const.GREEN)
            self._unload(plot, '06-digits')
            self._log('find_best_digits: 06-digits: green==zeroes, red==ones, blue==nones, black/white==ignored')

        return best_digits, reason

    def decode_digits(self, digits: [structs.Digit]) -> [structs.Result]:
        """ decode the digits into their corresponding code and doubt """

        bits = []
        error = 0
        samples = 0
        for digit in digits:
            bits.append(digit.digit)
            error += digit.error
            samples += digit.samples
        error /= samples  # average error per sample - 0..1
        code, doubt = self.decoder.unbuild(bits)
        number = self.decoder.decode(code)
        if self.logging:
            msg = ''
            for digit in digits:
                msg = '{}, {}'.format(msg, digit)
            self._log('decode_digits: num:{}, code:{}, doubt:{}, error:{:.3f} from: {}'.
                      format(number, code, doubt, error, msg[2:]))
        return structs.Result(number, doubt + error, code, digits)

class Allocate:
    """ given a ring span and a pixel slice allocate transitions to ring edges,
        transitions can be either rising or falling,
        a detected transition is allocated to its nearest ring edge (min squared distance),
        along with its distance (squared) and a 'strength' (3 steps -> strongest, strong, weak),
        ring edges are calculated from the span and the length of the pixel slice,
        the edge list is a list of offsets into the pixel slice where a transition is expected,
    """

    NO_EDGE        = 0
    STRONGEST_EDGE = 1  # lists are sorted ascending on this, so make best lowest number
    STRONG_EDGE    = 2
    WEAK_EDGE      = 3

    RISING_EDGE    = 1  # is the bit value of the edge 'destination' (i.e. rises to 1)
    FALLING_EDGE   = 0  # ..similar (i.e. falls to 0)

    def __init__(self, pixels, span, grey_as):
        self.pixels        = pixels
        self.span          = span
        self.width         = len(pixels) / span
        self.ideal_rising  = self.make_ideal_edges(self.width, span, -1)
        self.ideal_falling = self.make_ideal_edges(self.width, span, 0)

        self.actual_rising   = self.make_actual_edges(pixels, const.MIN_LUMINANCE, const.MAX_LUMINANCE, grey_as, 0)
        self.actual_falling  = self.make_actual_edges(pixels, const.MAX_LUMINANCE, const.MIN_LUMINANCE, grey_as, 0)

        self.matched_rising  = self.match_edges(self.ideal_rising , self.actual_rising , self.width, Allocate.RISING_EDGE)
        self.matched_falling = self.match_edges(self.ideal_falling, self.actual_falling, self.width, Allocate.FALLING_EDGE)

        self.edges            = self.make_combined_edges(self.matched_rising, self.matched_falling)
        self.bits, self.error = self.make_slice(self.edges, len(pixels))

    @staticmethod
    def make_ideal_edges(width, span, exclude=None):
        # width is the width of a single ring
        # span is how many rings there are
        edges = []
        for edge in range(1, span):  # NB: the inner and outer edges are not included
            edges.append(edge * width)
        if exclude is not None:
            del edges[exclude]
        # returned edges are in ascending offset order
        return edges

    @staticmethod
    def make_actual_edges(samples, from_level, to_level, mid_level=const.MID_LUMINANCE, offset=0):
        # there are 3 possible transitions in either direction:
        #  from --> to   (strongest +3)
        #  from --> mid  (weak      +1)
        #  mid  --> to   (strong    +2)
        #  nothing       (no edge    0)
        edges = []
        prev_level = None
        for bit, this_level in enumerate(samples):
            if this_level == const.MID_LUMINANCE:
                this_level = mid_level
            if prev_level is None:
                prev_level = this_level  # don't want a transition on the first sample (yet)
            if prev_level == from_level and this_level == to_level:
                strength = Allocate.STRONGEST_EDGE
            elif prev_level == from_level and this_level == const.MID_LUMINANCE:
                strength = Allocate.WEAK_EDGE
            elif prev_level == const.MID_LUMINANCE and this_level == to_level:
                strength = Allocate.STRONG_EDGE
            else:
                strength = Allocate.NO_EDGE
            if strength != Allocate.NO_EDGE:
                edges.append((bit+offset, strength))
            prev_level = this_level
        if len(edges) == 0:
            # found no edges, what to do depends on context (first sample value and the edge direction)
            first_sample = samples[0]
            if first_sample == const.MID_LUMINANCE:
                first_sample = mid_level
            if from_level > to_level:
                # looking for falling edges and found none
                if first_sample == const.MIN_LUMINANCE:
                    # got leading 0's, treat initial sample as falling
                    edges.append((0, Allocate.STRONG_EDGE))
                else:
                    # its got leading not-0's, treat final sample as falling
                    edges.append((len(samples), Allocate.STRONG_EDGE))
            else:
                # looking for rising edges and found none
                if first_sample == const.MIN_LUMINANCE:
                    # its all 0's, treat final sample as rising
                    edges.append((len(samples), Allocate.STRONG_EDGE))
                else:
                    # its got leading not-0's, treat initial sample as rising
                    edges.append((0, Allocate.STRONG_EDGE))
        # returned edges are in ascending offset order
        return edges

    @staticmethod
    def match_edges(ideals, actuals, width, direction):
        # ideal is a list of expected edge offsets
        # actual is a list of actual offsets and their strengths
        # make a list of adjusted edges with an error (in range 0..1)
        max_error = width * width
        edges = []
        for bit, ideal in enumerate(ideals):
            for actual, strength in actuals:
                error = ideal - actual
                error *= error
                error /= max_error  # now in range 0..1
                if error < 1.0:
                    # beyond one width error is meaningless, so only want those within a width of expectation
                    edges.append((bit, ideal, actual, strength, error))
        # we may have one actual assigned to two ideals,
        # in this case we want to chuck out the worst,
        # so sort on actual then error then strength then ideal and filter duplicates
        edges.sort(key=lambda k: (k[2], k[4], k[3], k[1]))
        for edge in range(len(edges)-1, 0, -1):
            _, _, this_actual, this_strength, this_error = edges[edge]
            _, _, prev_actual, prev_strength, prev_error = edges[edge-1]
            if this_actual == prev_actual:
                # got a dup
                if this_error > prev_error:
                    # this is worse, chuck it
                    del edges[edge]
                elif this_strength > prev_strength:
                    # this is same error but weaker, chuck it
                    del edges[edge]
                else:
                    # same error, same strength, so we've got ambiguity,
                    # ambiguity only matters if there is no other choice at an edge
                    # or other choices have a bigger error,
                    # to be ambiguous an actual edge must be equidistant to two ideal edges,
                    # to do that its error must be at a maximum, so if there are other choices
                    # they should be better, for a rising edge we dump the latter, for a falling the former
                    if direction == Allocate.RISING_EDGE:
                        del edges[edge]
                    else:
                        del edges[edge-1]
        # re-sort by ideal then error then strength, so we can produce ideal ordered result
        edges.sort(key=lambda k: (k[1], k[4], k[3]))
        matched = []
        for bit, ideal, actual, strength, error in edges:
            matched.append((ideal, actual, strength, error))
        # NB: an actual may match one or two ideals (i.e. can be ambiguous)
        #     an ideal may have 0 or more (if the pixel slice has lots of short spikes)
        # returned matched edges are in ascending offset order
        return matched

    @staticmethod
    def make_combined_edges(rising_matches, falling_matches):
        NO_POSITION = 1 << 30  # an edge position we know does not exist
        edges       = []
        rising_bit  = 0
        falling_bit = 0
        while rising_bit < len(rising_matches) or falling_bit < len(falling_matches):
            if rising_bit < len(rising_matches):
                rising     = rising_matches[rising_bit]
                rising_inc = 1
            else:
                rising     = []
                rising_inc = 0
            if len(rising) > 0:
                rising_pos = rising[0]
            else:
                rising_pos = NO_POSITION
            if falling_bit < len(falling_matches):
                falling     = falling_matches[falling_bit]
                falling_inc = 1
            else:
                falling     = []
                falling_inc = 0
            if len(falling) > 0:
                falling_pos = falling[0]
            else:
                falling_pos = NO_POSITION
            if rising_pos < falling_pos:
                # got only a rising edge here
                rising_bit += rising_inc
                edges.append((Allocate.RISING_EDGE, rising))
            elif falling_pos < rising_pos:
                # got only a falling edge here
                falling_bit += falling_inc
                edges.append((Allocate.FALLING_EDGE, falling))
            else:
                # got both here, resolve which to keep
                rising_bit  += rising_inc
                falling_bit += falling_inc
                if len(rising) > len(falling):
                    # got a rising edge
                    edges.append((Allocate.RISING_EDGE, rising))
                elif len(rising) < len(falling):
                    # got a falling edge
                    edges.append((Allocate.FALLING_EDGE, falling))
                else:
                    # both same, ignore it (they cancel out)
                    continue
        return edges

    @staticmethod
    def make_slice(edges, length):
        # a slice is a pixel list that spans the original pixels and an error (0..1)
        # they are constructed from rising edges to their subsequent falling edges
        # there can be zero or more matches for every ideal edge position
        # edges reflect a perfect digit, translate that into the equivalent pixel list
        slice            = [None for _ in range(length)]
        current_pixel    = 0
        current_position = 0
        current_error    = 0
        current_ideal    = -1  # an offset we know does not exist
        for direction, (ideal, actual, strength, error) in edges:
            # ToDo: extend to cater for ambiguity, how?
            if ideal == current_ideal:
                # already done this one (the least error version was first)
                continue
            next_position = int(round(ideal))
            for pos in range(current_position, next_position):
                if pos >= length:
                    # gone off the end
                    break
                slice[pos] = current_pixel
            current_ideal = ideal
            current_position = next_position
            current_pixel    = direction
            current_error   += error
        error = current_error / max(len(edges), 1)
        # now fill to the end (NB: this also catches the no edges case, i.e. a '0' or a '7')
        for pos in range(current_position, length):
            slice[pos] = current_pixel
        # slice is now in a form that can be given to Codec.qualify and Codec.classify
        return slice, error
