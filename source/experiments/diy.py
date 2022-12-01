import os
import glob
import shutil

import rand
import codec
import angle as angles
import ring as rings
import frame
import transform
import scan as scanner
# import matplotlib.pyplot as plt
import random
import math
import traceback

""" test harness

    Test the codec and the scanner.
    See codec for coding scheme description.
            
    """

# ToDo: create test background with random noise blobs and random edges
# ToDo: add a blurring option when drawing test targets
# ToDo: add a rotation (vertically and horizontally) option when drawing test targets
# ToDo: add a curving (vertically and horizontally) option when drawing test targets
# ToDo: generate lots of (extreme) test targets using these options
# ToDo: draw the test targets on a relevant scene photo background
# ToDo: try frame grabbing from a movie
# ToDo: in the final product speak the number detected (just need 10 sound snips - spoken 0 to 9)
# ToDo: in the final product use special codes to mark the finish, retired, et al
#       (runner crosses the line when his target size matches that of the finish code)
#       (runner retires by holding the retired target beside theirs in front of the camera)
#       (mark CP's by runners running between two CP (n) targets)
#       (record starters by placing start targets at several places near the race start)
# ToDo: generate some test images in heavy rain (and snow?)

# colours
MAX_LUMINANCE = 255
MIN_LUMINANCE = 0
MID_LUMINANCE = (MAX_LUMINANCE - MIN_LUMINANCE) >> 1


class Test:
    # exit codes from scan
    EXIT_OK = 0  # found what was expected
    EXIT_FAILED = 1  # did not find what was expected
    EXIT_EXCEPTION = 2  # an exception was raised

    def __init__(self, log=None):
        self.min_num = None
        self.coder = None
        self.max_num = None
        self.frame = None
        self.transform = transform.Transform()     # our opencv wrapper
        self.angles = None
        self.video_mode = None
        self.debug_mode = None
        self.log_folder = log
        self.log_file = None
        self._log('')
        self._log('******************')
        self._log('Rings: {}, Digits: {}'.format(rings.Ring.NUM_RINGS, codec.Codec.DIGITS_PER_WORD))

    def __del__(self):
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def encoder(self, min_num, max_num):
        """ create the coder/decoder and set its parameters """
        self.min_num = min_num
        self.codec = codec.Codec(min_num, max_num)
        self.base = self.codec.DIGIT_BASE
        if self.codec.num_limit is None:
            self.max_num = None
            self._log('Codec: available numbers are None!')
        else:
            self.max_num = min(max_num, self.codec.num_limit)
            self._log('Codec: {} digits, available numbers are {}..{}, available codes 0..{}, base={}'.
                      format(codec.Codec.DIGITS, min_num, self.max_num, self.codec.code_limit, self.base))

    def folders(self, read=None, write=None):
        """ create an image frame and set the folders to read input media and write output media """
        self.frame = frame.Frame(read, write)
        self._log('Frame: media in: {}, media out: {}'.format(read, write))

    def options(self, cells=None, mode=None, debug=None, log=None):
        """ set processing options, only given options are changed """
        if cells is not None:
            self.cells = cells
        if mode is not None:
            self.video_mode = mode
        if debug is not None:
            self.debug_mode = debug
        if log is not None:
            self.log_folder = log
        self._log('Options: cells {}, video mode {}, debug {}, log {}'.
                  format(self.cells, self.video_mode, self.debug_mode, self.log_folder))

    def rand(self, limit=None):
        """ check the random number generator is deterministic """
        if limit is None:
            limit = int(math.pow(self.base, codec.Codec.DIGITS_PER_WORD))
        self._log('')
        self._log('******************')
        self._log('Check random number generator is deterministic across {} numbers'.format(limit))
        try:
            generate = rand.Rand(seed=self.base)
            nums = []
            for _ in range(limit):
                num = int(round(generate.rnd() * limit))
                nums.append(num)
            generate = rand.Rand(seed=self.base)  # start again from same seed
            good = 0
            bad = 0
            for item in range(limit):
                num = int(round(generate.rnd() * limit))
                if nums[item] == num:
                    good += 1
                else:
                    bad += 1
            self._log('{} not repeated, {} correct'.format(bad, good))
        except:
            traceback.print_exc()
        self._log('******************')

    def coding(self):
        """ test for encode/decode symmetry """
        self._log('')
        self._log('******************')
        self._log('Check encode/decode from {} to {}'.format(self.min_num, self.max_num))

        def check(num):
            """ check encode/decode is symmetric
                returns None if check fails or the coded number if OK
                """
            encoded = self.codec.encode(num)
            if encoded is None:
                self._log('{} encodes as None'.format(num))
                return None
            decoded = self.codec.decode(encoded)
            if decoded != num:
                self._log('{} encodes to {} but decodes as {}'.format(num, encoded, decoded))
                return None
            return encoded

        try:
            good = 0
            bad = 0
            if self.max_num is not None:
                for n in range(self.min_num, self.max_num + 1):
                    if check(n) is None:
                        bad += 1
                    else:
                        good += 1
            self._log('{} good, {} bad'.format(good, bad))
        except:
            traceback.print_exc()
        self._log('******************')

    def decoding(self):
        """ test build/unbuild symmetry """

        def rotate(word, shift, bits):
            """ rotate the given word by shift right within a length of bits """
            msb = 1 << (bits - 1)
            for bit in range(shift):
                if word & 1 == 1:
                    word >>= 1
                    word += msb
                else:
                    word >>= 1
            return word

        self._log('')
        self._log('******************')
        self._log('Check build/unbuild from {} to {} with random rotations'.format(self.min_num, self.max_num))
        try:
            good = 0
            doubted = 0
            fail = 0
            bad = 0
            if self.max_num is not None:
                for n in range(self.min_num, self.max_num + 1):
                    rings = self.codec.build(n)
                    if rings is None:
                        continue  # not a valid code
                    # do a random rotation to test it gets re-aligned correctly
                    rotation = random.randrange(0, codec.Codec.DIGITS - 1)
                    for ring in range(len(rings)):
                        rings[ring] = rotate(rings[ring], rotation, codec.Codec.DIGITS)
                        # ToDo: add random errors to test doubt feedback
                    # turn rings into digits
                    digits = [None for _ in range(codec.Codec.DIGITS)]
                    mask = 1 << (codec.Codec.DIGITS - 1)  # 1 in the msb
                    for bit in range(codec.Codec.DIGITS):
                        digit_bits = [None for _ in range(len(rings))]
                        for ring, bits in enumerate(rings):
                            if bits & mask != 0:
                                # got a 1 here
                                digit_bits[ring] = 1
                            else:
                                digit_bits[ring] = 0
                        digits[bit] = self.codec.digit(digit_bits)
                        mask >>= 1  # move down a bit
                    m, doubt = self.codec.unbuild(digits)
                    if m is None:
                        # failed to decode
                        fail += 1
                        self._log('****FAIL: {:03}-->None, build={}, doubt={}, code={}, digits={}'.
                                  format(n, rings, doubt, m, digits))
                    elif self.codec.decode(m) != n:
                        # incorrect decode
                        bad += 1
                        self._log('****BAD!: {:03}-->{} , build={}, doubt={}, code={}, digits={}'.
                                  format(n, self.codec.decode(m), rings, doubt, m, digits))
                    elif doubt > 0:
                        # got doubt
                        doubted += 1
                        self._log('****DOUBT!: {:03}-->{} , build={}, doubt={}, code={}, digits={}'.
                                  format(n, self.codec.decode(m), rings, doubt, m, digits))
                    else:
                        good += 1
            self._log('{} good, {} doubted, {} bad, {} fail'.format(good, doubted, bad, fail))
        except:
            traceback.print_exc()
        self._log('******************')

    def errors(self):
        """ test the Codec error tolerance, we do this by generating every possible digit slice """

        # slice states
        DISQUALIFIED = '0-disq'
        INVALID      = '1-invd'
        AMBIGUOUS    = '2-ambi'
        VALID        = '3-good'

        class Slice:
            """ a test slice """
            def __init__(self, width, size, bits):
                self.width = width  # ring width for this slice
                self.size = size  # pulse size for this slice
                self.bits = bits.copy()  # bits of this slice
                self.error = None  # classification best error
                self.digit = None  # classification best digit
                self.raw_digits = None  # classification raw digits
                self.legal_digits = None  # after illegal digits removed
                self.good_digits = None  # after bad digits removed
                self.clean_digits = None  # after dodgy zeroes removed

            def __str__(self):
                return self.show_slice()

            def set_digits(self, digits, raw=False, legal=False, good=False, clean=False):
                if raw:
                    self.raw_digits = digits.copy()
                if legal:
                    self.legal_digits = digits.copy()
                if good:
                    self.good_digits = digits.copy()
                if clean:
                    self.clean_digits = digits.copy()

            def has_digits(self, raw=False, legal=False, good=False, clean=False):
                digit_sets = 0
                if raw:
                    if self.raw_digits is not None:
                        digit_sets += 1
                if legal:
                    if self.legal_digits is not None:
                        digit_sets += 1
                if good:
                    if self.good_digits is not None:
                        digit_sets += 1
                if clean:
                    if self.clean_digits is not None:
                        digit_sets += 1
                return digit_sets

            def show_slice(self):
                if self.error is None:
                    err = 'None'
                else:
                    err = '{:.2f}'.format(self.error)
                return '(width {}, bits {}, size {}, digit {}, error {})'.\
                       format(self.width, len(self.bits), self.size, self.digit, err)

            def show_bits(self):
                return scanner.Scan.show_bits(self.bits)

            def show_digits(self, raw=False, legal=False, good=False, clean=False):
                if raw:
                    digits = self.raw_digits
                elif legal:
                    digits = self.legal_digits
                elif good:
                    digits = self.good_digits
                elif clean:
                    digits = self.clean_digits
                else:
                    digits = []
                return scanner.Scan.show_options(digits)

        class Result:
            """ a test result set """

            def __init__(self, slice: Slice):
                self.width = slice.width
                self.size = slice.size
                self.digit = slice.digit
                self.state = slice.state
                self.first_slice: Slice = slice
                self.last_slice: Slice = slice
                self.min_slice: Slice = slice
                self.max_slice: Slice = slice
                self.count = 1

            def __str__(self):
                return self.show_result()

            def update(self, slice: Slice):
                self.count += 1
                if slice.error < self.min_slice.error:
                    self.min_slice = slice
                if slice.error > self.max_slice.error:
                    self.max_slice = slice
                self.last_slice = slice
                # self.slices.append(slice)

            def show_result(self):
                min_err = '{:.2f}'.format(self.min_slice.error)
                max_err = '{:.2f}'.format(self.max_slice.error)
                return '(digit {}: state {}, width {}, bits {}, size {}, count {}, min err {}, max err {})'.\
                       format(self.digit, self.state, self.width, len(self.last_slice.bits), self.size, self.count,
                              min_err, max_err)

            def has_digits(self, min=False, max=False, raw=False, legal=False, good=False, clean=False):
                if min:
                    return self.min_slice.has_digits(raw=raw, legal=legal, good=good, clean=clean)
                if max:
                    return self.max_slice.has_digits(raw=raw, legal=legal, good=good, clean=clean)
                return 0

            def show_digits(self, min=False, max=False, raw=False, legal=False, good=False, clean=False):
                if min:
                    return self.min_slice.show_digits(raw=raw, legal=legal, good=good, clean=clean)
                if max:
                    return self.max_slice.show_digits(raw=raw, legal=legal, good=good, clean=clean)
                return '[]'

            def show_bis(self, min=False, max=False, first=False, last=False):
                if min:
                    return self.min_slice.show_bits()
                if max:
                    return self.max_slice.show_bits()
                if first:
                    return self.first_slice.show_bits()
                if last:
                    return self.last_slice.show_bits()
                return '[]'

        def make_result_key(slice):
            """ make the results dictionary key for the given slice """
            key = 'W{}S{}D{}T{}'.format(slice.width, slice.size, slice.digit, slice.state)
            return key

        self._log('')
        self._log('******************')
        self._log('Check error tolerance on digit slices')
        try:
            results = {}
            # run the tests
            for ring_width in range(1, self.cells[1] + 2):
                # set pulse limits
                slice_length = ring_width * codec.Codec.SPAN  # this is size the scanner creates
                min_start = ring_width
                max_start = slice_length - ring_width - 1
                max_end = slice_length - ring_width
                min_size = 0
                max_size = slice_length - (2 * ring_width)
                for size in range(min_size, max_size + 1):
                    for start in range(min_start, max_start + 1):
                        if start + size > max_end:
                            # ran out of room, so that's it for this size
                            break
                        # set next slice
                        digit_slice = [0 for _ in range(slice_length)]  # set initial empty slice
                        if size > 0:
                            for x in range(start, start + size):
                                digit_slice[x] = 1
                        # put a 'smudge' on the inner/outer edge
                        for x in range(int(ring_width * 0.8)):  # almost the whole ring
                            digit_slice[x] = 1
                            digit_slice[(len(digit_slice) - 1) - x] = 1
                        # note our first slice for testing
                        holes = [digit_slice.copy()]
                        if size < 3:
                            # no room for a hole
                            pass
                        else:
                            # make slices with ever-increasing holes starting at ever-increasing offsets
                            max_hole_size = min(int(ring_width * 1.5), size - 2)
                            for hole_size in range(1, max_hole_size + 1):
                                for hole_start in range(start + 1, start + size - 1):
                                    if (hole_start + hole_size) > (start + size - 1):
                                        # ran out of room for this hole
                                        break
                                    noisy_pulse = digit_slice.copy()
                                    for x in range(hole_start, hole_start + hole_size):
                                        noisy_pulse[x] = 0
                                    holes.append(noisy_pulse)
                        # test these slices
                        for hole in holes:
                            slice = Slice(ring_width, size, hole)
                            if not self.codec.qualify(slice.bits):
                                slice.state = DISQUALIFIED
                                slice.error = 1.0
                            else:
                                raw_digits = self.codec.classify(slice.bits)
                                legal_digits, legal_dropped = scanner.Scan.drop_illegal_digits(raw_digits.copy())
                                good_digits, good_dropped = scanner.Scan.drop_bad_digits(legal_digits.copy())
                                clean_digits, clean_dropped = scanner.Scan.drop_bad_zero_digit(good_digits.copy())
                                if len(clean_digits) > 0:
                                    if scanner.Scan.is_ambiguous(clean_digits):
                                        slice.state = AMBIGUOUS
                                        slice.error = 1.0
                                    else:
                                        slice.state = VALID
                                        slice.digit, slice.error = clean_digits[0]
                                else:
                                    slice.state = INVALID
                                    slice.error = 1.0
                                slice.set_digits(raw_digits, raw=True)
                                if legal_dropped:
                                    slice.set_digits(legal_digits, legal=True)
                                if good_dropped:
                                    slice.set_digits(good_digits, good=True)
                                if clean_dropped:
                                    slice.set_digits(clean_digits, clean=True)
                            result_key = make_result_key(slice)
                            if result_key in results:
                                results[result_key].update(slice)
                            else:
                                # first time for this size, digit, state
                                results[result_key] = Result(slice)
                        if slice.size == 0:
                            # no point moving the start for this
                            break
                    pass  # do next size
                pass  # do next width
            # log the results
            result_list = list(results.values())  # convert to a list
            result_list.sort(key=lambda r: (-1 if r.digit is None else r.digit, r.state, r.width, r.size))  # sort into a useful order
            self._log('{} test results'.format(len(result_list)))
            digits = {}
            for result in result_list:
                if result.digit not in digits:
                    digits[result.digit] = {}
                states = digits[result.digit]
                if result.state not in states:
                    states[result.state] = 0
                states[result.state] += result.count
                self._log(result)
                self._log('    first bits: {}'.format(result.show_bis(first=True)))
                self._log('     last bits: {}'.format(result.show_bis(last=True)))
                self._log('      min bits: {}'.format(result.show_bis(min=True)))
                self._log('      max bits: {}'.format(result.show_bis(max=True)))
                self._log('    min raw digits: {}'.format(result.show_digits(min=True, raw=True)))
                self._log('    max raw digits: {}'.format(result.show_digits(max=True, raw=True)))
                if result.has_digits(min=True, legal=True) == 1:
                    self._log('    min legal digits: {}'.format(result.show_digits(min=True, legal=True)))
                if result.has_digits(max=True, legal=True) == 1:
                    self._log('    max legal digits: {}'.format(result.show_digits(max=True, legal=True)))
                if result.has_digits(min=True, good=True) == 1:
                    self._log('    min good digits: {}'.format(result.show_digits(min=True, good=True)))
                if result.has_digits(max=True, good=True) == 1:
                    self._log('    max good digits: {}'.format(result.show_digits(max=True, good=True)))
                if result.has_digits(min=True, clean=True) == 1:
                    self._log('    min clean digits: {}'.format(result.show_digits(min=True, clean=True)))
                if result.has_digits(max=True, clean=True) == 1:
                    self._log('    max clean digits: {}'.format(result.show_digits(max=True, clean=True)))
            self._log('Summary:')
            total_slices = 0
            for digit, states in digits.items():
                for state, count in states.items():
                    total_slices += count
                    self._log('    digit {}, state {}, count {}'.format(digit, state, count))
            self._log('Total slices {}'.format(total_slices))

        except:
            traceback.print_exc()
        self._log('******************')

    def test_set(self, size, presets=None):
        """ make a set of test codes,
            the test codes consist of the given presets plus the minimum and maximum numbers
            and N random numbers to make the set size up to that given
            """
        if self.max_num is None:
            return []
        if size < 2:
            size = 2
        if self.max_num - self.min_num <= size:
            return [num for num in range(self.min_num, self.max_num + 1)]
        num_set = [self.min_num, self.max_num]
        if presets is not None:
            for preset in presets:
                if preset < self.min_num or preset > self.max_num:
                    # don't use illegal preset
                    self._log('ignoring invalid test set number: {}'.format(preset))
                    continue
                if preset in num_set:
                    # don't want a duplicate
                    continue
                num_set.append(preset)
        while len(num_set) < size:
            num = random.randrange(self.min_num + 1, self.max_num - 1)
            if num in num_set:
                # don't want a duplicate
                continue
            num_set.append(num)
        num_set.sort()
        return num_set

    def code_words(self, numbers):
        """ test code-word generation with given set (visual) """
        self._log('')
        self._log('******************')
        self._log('Check code-words (visual)')
        try:
            for n in numbers:
                if n is None:
                    # this means a test code pattern is not available
                    num = 'None'
                    code = 'None'
                    digits = 'None'
                else:
                    num = '{:03n}'.format(n)
                    code = self.codec.encode(n)
                    if code is None:
                        # should not get here!
                        code = 'None'
                        digits = 'None'
                    else:
                        digits = self.codec.digits(code)
                        code = '{:06n}'.format(code)
                        if digits is None:
                            # this should not happen
                            digits = 'None'
                self._log('    {} ({}) = {}'.format(num, code, digits))
        except:
            traceback.print_exc()
        self._log('******************')

    def circles(self):
        """ test accuracy of co-ordinate conversions - polar to/from cartesian,
            also check polarToCart goes clockwise
            """
        self._log('')
        self._log('*******************************************************')
        self._log('Check co-ordinate conversions (radius 256, 0.1 degrees)')
        # check clockwise direction by checking sign and relative size as we go round each octant
        #          angle, x-sign, y-sign, xy-sign
        octants = [[45, +1, -1, -1],
                   [90, +1, -1, +1],
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
            radius = 256
            scale = 360 * 10  # 0.1 degrees
            angle = angles.Angle(scale, radius)
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
                if rerr > 0.7 or aerr > 0.3 or rotation_err is not None:
                    bad += 1
                    self._log(
                        '{:.3f} degrees, {:.3f} radius --> {:.3f}x, {:.3f}y --> {:.3f} degrees, {:.3f} radius: aerr={:.3f}, rerr={:.3f}, rotation={}'.
                        format(a, radius, cx, cy, ca, cr, aerr, rerr, rotation_err))
                else:
                    good += 1

            self._log('{} good, {} bad'.format(good, bad))
        except:
            traceback.print_exc()
        self._log('*******************************************************')

    def rings(self, folder, width):
        """ draw pulse test rings in given folder (visual) """
        self._log('')
        self._log('******************')
        self._log('Draw pulse test rings (to test scanner edge detection)')
        self.folders(write=folder)
        try:
            x, y = self._make_image_buffer(width)
            ring = rings.Ring(x >> 1, y >> 1, width, self.frame)
            block = [0 for _ in range(codec.Codec.RINGS_PER_DIGIT)]
            digit = 0
            slice = -1
            while digit < codec.Codec.DIGITS:
                slice += 1
                bits = self.codec.coding(slice % self.base)
                if bits is None:
                    # illegal digit
                    continue
                digit_mask = 1 << digit
                for r in range(codec.Codec.RINGS_PER_DIGIT):
                    if bits[r] == 1:
                        block[r] += digit_mask
                digit += 1
            ring.code(000, block)
            self.frame.unload('test-code-000')
        except:
            traceback.print_exc()
        self._log('******************')

    def codes(self, folder, numbers, width):
        """ draw test codes for the given test_set in the given folder """
        self._log('')
        self._log('******************')
        self._log('Draw test codes (visual)')
        self.folders(write=folder)
        self._remove_test_codes(folder, 'test-code-')
        try:
            for n in numbers:
                if n is None:
                    # this means a test code pattern is not available
                    continue
                code_rings = self.codec.build(n)
                if code_rings is None:
                    self._log('  {}: failed to generate the code rings'.format(n))
                else:
                    self._log('  drawing rings for code {}'.format(n))
                    x, y = self._make_image_buffer(width)
                    ring = rings.Ring(x >> 1, y >> 1, width, self.frame)
                    ring.code(n, code_rings)
                    self.frame.unload('test-code-{}'.format(n))
        except:
            traceback.print_exc()
        self._log('******************')

    def scan_codes(self, folder):
        """ find all the test codes in the given folder and scan them,
            these are all 'perfect' images saved as drawn, each file name
            is assumed to include the code number of the image as 3 digits
            """

        filelist = glob.glob('{}/test-code-*.*'.format(folder))
        filelist.sort()
        for f in filelist:
            f = os.path.basename(f)
            num = ''.join([s for s in f if s.isdigit()])
            if num == '':
                num = 0
            else:
                num = int(num)
            self.scan(folder, [num], f, scanner.Scan.PROXIMITY_CLOSE)

    def scan_media(self, folder):
        """ find all the media in the given folder and scan them,
            these are video frames of codes in various states of distortion,
            each file name is assumed to include the code number in the image,
            code numbers must be 3 digits, if an image contains more than one code include
            all numbers separated by a non-digit
            """

        filelist1 = glob.glob('{}/*.jpg'.format(folder))
        filelist2 = glob.glob('{}/*.png'.format(folder))
        filelist = filelist1 + filelist2
        filelist.sort()
        for f in filelist:
            f = os.path.basename(f)
            digits = ''.join([s for s in f if s.isdigit()])
            # every 3 digits is assumed to be a code number
            codes = []
            while len(digits) >= 3:
                num = int(digits[0:3])
                if num < self.min_num or num > self.max_num:
                    num = 0
                codes.append(num)
                digits = digits[3:]
            if len(codes) == 0:
                codes = [0]
            self.scan(folder, codes, f, scanner.Scan.PROXIMITY_FAR)

    def scan(self, folder, numbers, image, proximity=scanner.Scan.PROXIMITY_FAR):
        """ do a scan for the code set in image in the given folder and expect the number given,
            proximity specifies how close to the camera the targets can be, 'far' is suitable for
            normal video capture, 'close' is suitable for test images that consist of just a single
            target covering the whole frame,
            returns an exit code to indicate what happened
            """
        self._log('')
        self._log('******************')
        self._log('Scan image {} with proximity {} for codes {}'.format(image, proximity, numbers))
        if not os.path.isfile('{}/{}'.format(folder, image)):
            self._log('Image {} does not exist in {}'.format(image, folder))
            exit_code = self.EXIT_FAILED
        else:
            debug_folder = 'debug_images'
            self.folders(read=folder, write=debug_folder)
            exit_code = self.EXIT_OK  # be optimistic
            scan = None
            try:
                self._remove_debug_images(debug_folder, image)
                self.frame.load(image)
                scan = scanner.Scan(self.codec, self.frame, self.transform, self.cells, self.video_mode,
                                    proximity=proximity, debug=self.debug_mode, log=self.log_folder)
                results = scan.decode_targets()
                # analyse the results
                found = [False for _ in range(len(numbers))]
                analysis = []
                for result in results:
                    centre_x = result.centre_x
                    centre_y = result.centre_y
                    num = result.result.number
                    doubt = result.result.doubt
                    code = result.result.code
                    digits = result.result.digits
                    size = result.target_size
                    expected = None
                    found_num = None
                    for n in range(len(numbers)):
                        if numbers[n] == num:
                            # found another expected number
                            found[n] = True
                            found_num = num
                            expected = 'code={}, digits={}'.format(code, self.codec.digits(code))
                            break
                    analysis.append([found_num, centre_x, centre_y, num, doubt, size, expected, digits])
                # create dummy result for those not found
                for n in range(len(numbers)):
                    if not found[n]:
                        # this one is missing
                        num = numbers[n]
                        code = self.codec.encode(num)
                        if code is None:
                            # not a legal code
                            expected = 'not-valid'
                        else:
                            expected = 'code={}, digits={}'.format(code, self.codec.digits(code))
                        analysis.append([None, 0, 0, numbers[n], 0, 0, expected, None])
                # print the results
                for loop in range(3):
                    for result in analysis:
                        found_num = result[0]
                        centre_x = result[1]
                        centre_y = result[2]
                        num = result[3]
                        doubt = result[4]
                        size = result[5]
                        expected = result[6]
                        digits = result[7]
                        if found_num is not None:
                            if loop != 0:
                                # don't want these in this loop
                                continue
                            # got what we are looking for
                            self._log('Found {} ({}) at {:.0f}x, {:.0f}y size {:.2f}, doubt {:.3f}, digits={}'.
                                      format(num, expected, centre_x, centre_y, size, doubt, self._show_list(digits)))
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
                            self._log('{}Failed to find {} ({})'.format(prefix, num, expected))
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
                                actual_code = self.codec.encode(num)
                                if actual_code is None:
                                    actual_code = 'not-valid'
                                    prefix = ''
                                else:
                                    actual_code = 'code={}, digits={}'\
                                                  .format(actual_code, self.codec.digits(actual_code))
                                    prefix = '**** UNEXPECTED **** ---> '
                            self._log('{}Found {} ({}) at {:.0f}x, {:.0f}y size {:.2f}, doubt {:.3f}, digits={}'.
                                      format(prefix, num, actual_code, centre_x, centre_y, size, doubt,
                                             self._show_list(digits)))
                            continue
                if len(results) == 0:
                    self._log('**** FOUND NOTHING ****')
            except:
                traceback.print_exc()
                exit_code = self.EXIT_EXCEPTION
            finally:
                if scan is not None:
                    del (scan)  # needed to close log files
        self._log('Scan image {} for codes {}'.format(image, numbers))
        self._log('******************')
        return exit_code

    def _show_list(self, this):
        """ helper to display a list of classes where that class is assumed to have a __str__ function """
        if this is None:
            return 'None'
        msg = ''
        for item in this:
            msg = '{}, {}'.format(msg, item)
        return '[' + msg[2:] + ']'

    def _log(self, message):
        """ print a log message and maybe write it to a file too """
        print(message)
        if self.log_folder is not None:
            if self.log_file is None:
                log_file = '{}/test.log'.format(self.log_folder)
                self._remove_file(log_file, silent=True)
                self.log_file = open(log_file, 'w')
            self.log_file.write('{}\n'.format(message))

    def _make_image_buffer(self, width):
        """ make an image buffer suitable for drawing our test images within,
            width is the width to allow for each ring,
            returns the buffer x, y size
            """

        image_width = int(round(width * rings.Ring.NUM_RINGS * 2))
        self.frame.new(image_width, image_width, MIN_LUMINANCE)  # NB: must be initialised to black
        x, y = self.frame.size()

        return x, y

    def _remove_file(self, f, silent=False):
        try:
            os.remove(f)
        except:
            if silent:
                pass
            else:
                traceback.print_exc()
                self._log('Could not remove {}'.format(f))

    def _remove_test_codes(self, folder, pattern):
        """ remove all the test code images with file names containing the given pattern in the given folder
            """
        filelist = glob.glob('{}/*{}*.*'.format(folder, pattern))
        for f in filelist:
            self._remove_file(f)

    def _remove_debug_images(self, folder, filename):
        """ remove all the diagnostic images of the given file name in the given folder,
            diagnostic images are in a folder with that base file name prefixed by '_'
            """
        if os.path.isdir(folder):
            dir_list = os.listdir('{}/'.format(folder))
            filename, _ = os.path.splitext(filename)
            exists = '_{}'.format(filename) in dir_list
            if exists:
                debug_folder = '{}/_{}'.format(folder, filename)
                try:
                    shutil.rmtree(debug_folder)
                except:
                    traceback.print_exc()
                    self._log('could not remove {}'.format(debug_folder))


def verify():
    # parameters
    min_num = 101  # min number we want
    max_num = 999  # max number we want (may not be achievable)

    test_ring_width = 34  # this makes a target that fits on A5

    # cell size is critical,
    # too small and 1 pixel errors become significant,
    # too big and takes too long and uses a lot of memory
    test_scan_cells = (8, 8)

    # reducing the resolution means targets have to be closer to be detected,
    # increasing it takes longer to process, most modern smartphones can do 4K at 30fps, 2K is good enough
    test_scan_video_mode = scanner.Scan.VIDEO_2K

    # test_debug_mode = scanner.Scan.DEBUG_IMAGE
    test_debug_mode = scanner.Scan.DEBUG_VERBOSE

    # test log/image folders
    test_log_folder = 'logs'
    test_codes_folder = 'codes'
    test_media_folder = 'media'

    try:
        test = None
        # setup test params
        test = Test(log=test_log_folder)
        test.encoder(min_num, max_num)
        test.options(cells=test_scan_cells,
                     mode=test_scan_video_mode,
                     debug=test_debug_mode)

        # build a test code set
        # test_num_set = test.test_set(20, [111, 222, 333, 444, 555, 666, 777, 888, 999])

        # test.rand()
        # test.coding()
        # test.decoding()
        # test.errors()
        # test.circles()
        # test.code_words(test_num_set)
        # test.codes(test_codes_folder, test_num_set, test_ring_width)
        # test.rings(test_codes_folder, test_ring_width)  # must be after test.codes (else it gets deleted)

        # test.scan_codes(test_codes_folder)
        # test.scan_media(test_media_folder)

        # test.scan(test_codes_folder, [000], 'test-code-000.png')
        # test.scan(test_codes_folder, [222], 'test-code-222.png')

        test.scan(test_media_folder, [101,102,111,116,222,298,333,387,401,444,555,666,673,732,746,756,777,888,892,999],
                  'distant-101-102-111-116-222-298-333-387-401-444-555-666-673-732-746-756-777-888-892-999.jpg')

    except:
        traceback.print_exc()

    finally:
        if test is not None:
            del (test)  # needed to close the log file(s)


if __name__ == "__main__":
    verify()
