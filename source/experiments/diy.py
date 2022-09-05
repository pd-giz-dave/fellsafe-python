import os
import glob
import shutil

import rand
import codec
import ring
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
        self.contrast = None
        self.offset = None
        self.debug_mode = None
        self.log_folder = log
        self.log_file = None
        self._log('')
        self._log('******************')
        self._log('Rings: {}, Digits: {}'.format(ring.Ring.NUM_RINGS, codec.Codec.DIGITS_PER_WORD))

    def __del__(self):
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def encoder(self, min_num, max_num, base):
        """ create the coder/decoder and set its parameters """
        self.min_num = min_num
        self.codec = codec.Codec(min_num, max_num, base)
        self.base = self.codec.base
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

    def options(self, cells=None, mode=None, contrast=None, offset=None, debug=None, log=None):
        """ set processing options, only given options are changed """
        if cells is not None:
            self.cells = cells
        if mode is not None:
            self.video_mode = mode
        if contrast is not None:
            self.contrast = contrast
        if offset is not None:
            self.offset = offset
        if debug is not None:
            self.debug_mode = debug
        if log is not None:
            self.log_folder = log
        self._log('Options: cells {}, video mode {}, contrast {}, offset {}, debug {}, log {}'.
                  format(self.cells, self.video_mode, self.contrast, self.offset, self.debug_mode, self.log_folder))

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
                    m, doubt, code = self.codec.unbuild(digits)
                    if m is None:
                        # failed to decode
                        fail += 1
                        self._log('****FAIL: {:03}-->{}, build={}, doubt={}, code={}, digits={}'.
                                  format(n, m, rings, doubt, code, digits))
                    elif m != n:
                        # incorrect decode
                        bad += 1
                        self._log('****BAD!: {:03}-->{} , build={}, doubt={}, code={}, digits={}'.
                                  format(n, m, rings, doubt, code, digits))
                    elif doubt > 0:
                        # got doubt
                        doubted += 1
                        self._log('****DOUBT!: {:03}-->{} , build={}, doubt={}, code={}, digits={}'.
                                  format(n, m, rings, doubt, code, digits))
                    else:
                        good += 1
            self._log('{} good, {} doubted, {} bad, {} fail'.format(good, doubted, bad, fail))
        except:
            traceback.print_exc()
        self._log('******************')

    def errors(self):
        """ test the Codec ratio tolerance,
            we use this to test different bit ratios to find those with the best error differential,
            the differential is the minimum error difference between the best two digits for each ratio
            """

        self._log('')
        self._log('******************')
        self._log('Check error tolerance on digit ratios (visual)')
        # in a real target the radial cells can only be a few pixels high (self.cell[1])
        # so a one pixel error can be significant, we test ratios for every possibility
        # black/white lengths. The biggest length is cell-height * cells-in-radius
        try:
            max_span = self.cells[1] * (codec.Codec.RINGS_PER_DIGIT + 1)  # ToDo: is this correct for 0000 case?
            min_length = 2
            max_length = max_span - min_length
            min_ratio = min_length / max_length
            max_ratio = max_length / min_length
            ratios = {}
            for black_length in range(min_length, max_length + 1):
                for white_length in range(min_length, (max_span - black_length) + 1):
                    ratio = black_length / white_length
                    ratio = max(min(ratio, max_ratio), min_ratio)
                    int_ratio = int(round(ratio * 20))
                    if ratios.get(int_ratio) is None:
                        ratios[int_ratio] = ratio
            digits = {}
            for ratio in ratios.values():
                candidates = self.codec.classify(ratio)
                digit = candidates[0][0]
                if digits.get(digit) is None:
                    # new digit
                    digits[digit] = [(ratio, candidates)]
                else:
                    digits[digit].append((ratio, candidates))
            errors = []
            for digit, options in digits.items():
                options.sort(key=lambda k: k[0])  # put options into ratio order
                errors.append((digit, options))
            errors.sort(key=lambda e: e[0])
            for (digit, options) in errors:
                if len(options) == 0:
                    self._log('Digit:{} has no options!'.format(digit))
                    continue
                self._log('Digit {} ideal={:.2f}: detection ratio span: {:.2f} .. {:.2f} in {} steps'.
                          format(digit, self.codec.ratio(digit), options[0][0], options[-1][0], len(options)))
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
            num_set += presets
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
        frm_bin = '{:0' + str(codec.Codec.DIGITS) + 'b}'
        frm_prefix = '{}(' + frm_bin + ')=('
        suffix = ')'
        try:
            for n in numbers:
                if n is None:
                    # this means a test code pattern is not available
                    continue
                prefix = frm_prefix.format(n, n)
                rings = self.codec.build(n)
                if rings is None:
                    infix = 'None'
                else:
                    infix = ''
                    for ring in rings:
                        bits = frm_bin.format(ring)
                        infix = '{}, {}'.format(infix, bits)
                    infix = infix[2:]  # remove leading comma space
                self._log('{}{}{}'.format(prefix, infix, suffix))
        except:
            traceback.print_exc()
        self._log('******************')

    def circles(self):
        """ test accuracy of co-ordinate conversions - polar to/from cartesian,
            also check polarToCart goes clockwise
            """
        self._log('')
        self._log('******************************************')
        self._log('Check co-ordinate conversions (radius 100)')
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
        self._log('******************************************')

    def rings(self, folder, width):
        """ draw pulse test rings in given folder (visual) """
        self._log('')
        self._log('******************')
        self._log('Draw pulse test rings (to test scanner edge detection)')
        self.folders(write=folder)
        try:
            x, y = self._make_image_buffer(width)
            ring = rings.Ring(x >> 1, y >> 1, width, self.frame, self.contrast, self.offset)
            bits = codec.Codec.ENCODING
            block = [0 for _ in range(codec.Codec.RINGS_PER_DIGIT)]
            for slice in range(codec.Codec.DIGITS):
                bit = bits[int(slice % len(bits))]
                slice_mask = 1 << slice
                for r in range(codec.Codec.RINGS_PER_DIGIT):
                    if bit[r] is None:
                        # this is a don't care 0
                        continue
                    if bit[r] == 1:
                        block[r] += slice_mask
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
                    self._log('{}: failed to generate the code rings'.format(n))
                else:
                    x, y = self._make_image_buffer(width)
                    ring = rings.Ring(x >> 1, y >> 1, width, self.frame, self.contrast, self.offset)
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
            self.scan(folder, [num], f)

    def scan_media(self, folder):
        """ find all the media in the given folder and scan them,
            these are photos of codes in various states of distortion,
            each file name is assumed to include the code number in the image,
            code numbers must be 3 digits, if an image contains more than one code include
            all numbers separated by a non-digit
            """

        filelist = glob.glob('{}/*.jpg'.format(folder))
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
            self.scan(folder, codes, f)

    def scan(self, folder, numbers, image):
        """ do a scan for the code set in image in the given folder and expect the number given,
            returns an exit code to indicate what happened
            """
        self._log('')
        self._log('******************')
        self._log('Scan image {} for codes {}'.format(image, numbers))
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
                                    debug=self.debug_mode, log=self.log_folder)
                results = scan.decode_targets()
                # analyse the results
                found = [False for _ in range(len(numbers))]
                analysis = []
                for result in results:
                    centre_x = result.centre_x
                    centre_y = result.centre_y
                    num = result.number
                    doubt = result.doubt
                    size = result.target_size
                    bits = result.digits
                    expected = None
                    found_num = None
                    for n in range(len(numbers)):
                        if numbers[n] == num:
                            # found another expected number
                            found[n] = True
                            found_num = num
                            code = self.codec.encode(num)
                            expected = 'code={}, bits={}'.format(code, self.codec.bits(code))
                            break
                    analysis.append([found_num, centre_x, centre_y, num, doubt, size, expected, bits])
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
                            expected = 'code={}, bits={}'.format(code, self.codec.bits(code))
                        analysis.append([None, 0, 0, numbers[n], 0, 0, expected, None])
                # print the results
                for loop in range(3):
                    for result in analysis:
                        found = result[0]
                        centre_x = result[1]
                        centre_y = result[2]
                        num = result[3]
                        doubt = result[4]
                        size = result[5]
                        expected = result[6]
                        bits = result[7]
                        bits = ', bits {}'.format(bits)
                        if found is not None:
                            if loop != 0:
                                # don't want these in this loop
                                continue
                            # got what we are looking for
                            self._log('Found {} ({}) at {:.0f}x, {:.0f}y size {:.2f}, doubt {:.4f}{}'.
                                      format(num, expected, centre_x, centre_y, size, doubt, bits))
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
                            elif num < 0:
                                num = 0 - num
                                actual_code = self.codec.encode(num)
                                if actual_code is None:
                                    actual_code = 'not-valid'
                                else:
                                    actual_code = 'code={}, bits={}'. \
                                                  format(actual_code, self.codec.bits(actual_code))
                                prefix = '**** AMBIGUOUS **** --> '
                            else:
                                actual_code = self.codec.encode(num)
                                if actual_code is None:
                                    actual_code = 'not-valid'
                                    prefix = ''
                                else:
                                    actual_code = 'code={}, bits={}'.\
                                                  format(actual_code, self.codec.bits(actual_code))
                                    prefix = '**** UNEXPECTED **** ---> '
                            self._log('{}Found {} ({}) at {:.0f}x, {:.0f}y size {:.2f}, doubt {:.4f}{}'.
                                      format(prefix, num, actual_code, centre_x, centre_y, size, doubt, bits))
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

        image_width = width * (rings.Ring.NUM_RINGS + 1) * 2  # rings +1 for the border
        self.frame.new(image_width, image_width, MID_LUMINANCE)
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
    code_base = 7
    contrast = 1.0  # reduce dynamic luminance range when drawing to minimise 'bleed' effects
    offset = 0.0  # offset luminance range from the mid-point, -ve=below, +ve=above

    test_codes_folder = 'codes'
    test_media_folder = 'media'
    test_log_folder = 'logs'
    test_ring_width = 30  # this makes a target that fits on A5

    # cell size is critical,
    # going small in length creates edges that are steep vertically, going more takes too long
    # going small in height creates edges that are too small and easily confused with noise
    test_scan_cells = (7, 5)

    # reducing the resolution means targets have to be closer to be detected,
    # increasing it takes longer to process, most modern smartphones can do 4K at 30fps, 2K is good enough
    test_scan_video_mode = scanner.Scan.VIDEO_2K

    # test_debug_mode = scanner.Scan.DEBUG_IMAGE
    test_debug_mode = scanner.Scan.DEBUG_VERBOSE

    try:
        test = None
        # setup test params
        test = Test(log=test_log_folder)
        test.encoder(min_num, max_num, code_base)
        test.options(cells=test_scan_cells,
                     mode=test_scan_video_mode,
                     contrast=contrast,
                     offset=offset,
                     debug=test_debug_mode)

        # build a test code set
        test_num_set = test.test_set(20, [111, 222, 333, 444, 555])

        test.rand()
        test.coding()
        test.decoding()
        test.errors()
        test.circles()
        test.code_words(test_num_set)
        test.codes(test_codes_folder, test_num_set, test_ring_width)
        test.rings(test_codes_folder, test_ring_width)  # must be after test.codes (else it gets deleted)

        # test.scan_codes(test_codes_folder)
        # test.scan_media(test_media_folder)

        # test.scan(test_codes_folder, [000], 'test-code-000.png')
        # test.scan(test_codes_folder, [332], 'test-code-332.png')
        # test.scan(test_codes_folder, [222], 'test-code-222.png')
        # test.scan(test_codes_folder, [555], 'test-code-555.png')
        # test.scan(test_codes_folder, [800], 'test-code-800.png')
        # test.scan(test_codes_folder, [574], 'test-code-574.png')
        # test.scan(test_codes_folder, [371], 'test-code-371.png')
        # test.scan(test_codes_folder, [757], 'test-code-757.png')
        # test.scan(test_codes_folder, [611], 'test-code-611.png')
        # test.scan(test_codes_folder, [620], 'test-code-620.png')
        # test.scan(test_codes_folder, [132], 'test-code-132.png')

        # test.scan(test_media_folder, [301], 'photo-301.jpg')
        # test.scan(test_media_folder, [775, 592, 184, 111, 101, 285, 612, 655, 333, 444], 'photo-775-592-184-111-101-285-612-655-333-444.jpg')
        # test.scan(test_media_folder, [332, 222, 555, 800, 574, 371, 757, 611, 620, 132], 'photo-332-222-555-800-574-371-757-611-620-132-mid.jpg')
        # test.scan(test_media_folder, [332, 222, 555, 800, 574, 371, 757, 611, 620, 132], 'photo-332-222-555-800-574-371-757-611-620-132-distant.jpg')
        # test.scan(test_media_folder, [332, 222, 555, 800, 574, 371, 757, 611, 620, 132], 'photo-332-222-555-800-574-371-757-611-620-132.jpg')
        # test.scan(test_media_folder, [332, 222, 555, 800, 574, 371, 757, 611, 620, 132], 'photo-332-222-555-800-574-371-757-611-620-132-near.jpg')

    except:
        traceback.print_exc()

    finally:
        if test is not None:
            del (test)  # needed to close the log file(s)


if __name__ == "__main__":
    verify()
