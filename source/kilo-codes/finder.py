""" Finder for the locators timing marks
    This module takes a list of locator detections and finds the associated timing marks
"""

import math
import const
import utils
import codes
import locator
import canvas


class Params(locator.Params):
    def __init__(self):
        self.finder = None

class Finder:

    # region geometry...
    CELL_SPAN    = codes.Codes.LOCATOR_SPAN      # number of cell units between locators
    TIMING_CELLS = codes.Codes.TIMING_CELLS      # physical timing mark cell positions between locators
    TIMING_SCALE = codes.Codes.TIMING_SCALE      # size of timing marks relative to locators
    TIMING_MARKS = codes.Codes.TIMING_PER_CODE   # how many timing marks per code
    # endregion
    # region tuning...
    MIN_IMAGE_SIZE = 96  # min target image size, smaller ones are upsized to at least this
    # distant=30 or less, far=40 or less, near=50 or less, close=100 or less, native=300+
    SMALL_IMAGE_SIZE = 40  # threshold for a small detection image
    LARGE_IMAGE_SIZE = 80  # ditto for a big one
    # these ratios have to be sloppy for small images 'cos 1 pixel diff can be huge with low resolution images
    # these ratios are indexed by the image size index (0=small, 1=medium, 2=large)
    MIN_RADIUS_RATIO = (0.2, 0.4, 0.6)  # min radius deviation from expected (1=perfect, 0=rubbish, 0.5=2:1)
    MAX_DISTANCE_RATIO = (3.6, 3.5, 3.0)  # max distance between actual and expected mark as a multiple of expected radius
    MIN_QUALITY_RATIO = (0.3, 0.3, 0.4)  # 'quality' is radius-error (1..0) times distance-error (1..0), 1 is best, 0 is worst
    DISTANCE_QUALITY_WEIGHT = 0.5  # weighting factor for the distance component of quality (must be <= 1)
    SIZE_QUALITY_WEIGHT = 1.0  # weighting factor for the size component of quality (must be <= 1)
    MATCH_METRIC_SCALE = 2  # the scale factor to apply to the above when matching (scale is 1 when filtering)
    MIN_MARK_HITS = 6/TIMING_MARKS  # minimum number of matched marks for a detection to qualify as a ratio of the maximum, 1==all
    LOCATOR_MARGIN = 1 + TIMING_SCALE  # timing marks must be separated from locators by at least 1r
    MIN_TIMING_MARK_GAP = 0.5  # minimum gap between timing marks as a fraction of their expected radii
    # endregion
    # region field offsets in a blob...
    X_COORD = locator.Locator.X_COORD
    Y_COORD = locator.Locator.Y_COORD
    R_COORD = locator.Locator.R_COORD
    # endregion

    def __init__(self, source, image, detections, logger=None):
        self.source             = source      # originating image file name (for diagnostics)
        self.image              = image       # grayscale image the detections were detected within (and co-ordinates apply to)
        self.detections         = detections  # qualifying detections
        self.logger             = logger      # for diagnostics
        self.sub_images         = None        # grayscale extractions of all detections and their scale
        self.good_images        = None        # grayscale extractions of all accepted detections and their scale
        self.found_timing       = None        # candidate timing marks discovered for each detection (random ordering)
        self.expected_timing    = None        # expected timing mark centres for each detection (clockwise from top-left)
        self.matched_timing     = None        # discovered/expected pairings in best to worst order
        self.filtered_timing    = None        # list of detections that pass the qualification filter and their matched marks
        self.code_grids         = None        # list of detected code grids for all filtered timings
        self.min_radius_ratio   = None        # appropriate radius ratio for each sub-image size
        self.max_distance_ratio = None        # appropriate distance ratio for each sub-image size
        self.min_quality_ratio  = None        # appropriate quality ratio for each sub-image size

    def get_grayscale(self, detection=None):
        """ extract the grayscale sub-image of the given, or all, detection,
            returns the image and its scale (if it got resized)
            """

        def log_size(detection, size, size_name):
            if self.logger is not None:
                self.logger.log('Detection {} is {} ({} pixels, small limit is {}, large limit is {}), '
                                'min_radius_ratio={:.2f}, max_distance_ratio={:.2f}, min_quality_ratio={:.2f}'.
                                format(detection, size_name, size, Finder.SMALL_IMAGE_SIZE, Finder.LARGE_IMAGE_SIZE,
                                       self.min_radius_ratio[detection], self.max_distance_ratio[detection],
                                       self.min_quality_ratio[detection]))

        if self.sub_images is None:
            self.sub_images         = []
            self.min_radius_ratio   = []
            self.max_distance_ratio = []
            self.min_quality_ratio  = []
            for d_num, d in enumerate(self.detections):
                if self.logger is not None:
                    folder = utils.image_folder(target=d.box_tl)
                    self.logger.push(context=folder)
                image = canvas.extract(self.image, box=(d.box_tl, d.box_br))
                max_x, max_y = canvas.size(image)
                size = max(max_x, max_y)
                if size <= Finder.SMALL_IMAGE_SIZE:
                    self.min_radius_ratio.append(Finder.MIN_RADIUS_RATIO[0])
                    self.max_distance_ratio.append(Finder.MAX_DISTANCE_RATIO[0])
                    self.min_quality_ratio.append(Finder.MIN_QUALITY_RATIO[0])
                    if self.logger is not None:
                        log_size(d_num, size, 'small')
                elif size >= Finder.LARGE_IMAGE_SIZE:
                    self.min_radius_ratio.append(Finder.MIN_RADIUS_RATIO[2])
                    self.max_distance_ratio.append(Finder.MAX_DISTANCE_RATIO[2])
                    self.min_quality_ratio.append(Finder.MIN_QUALITY_RATIO[2])
                    if self.logger is not None:
                        log_size(d_num, size, 'large')
                else:  # medium
                    self.min_radius_ratio.append(Finder.MIN_RADIUS_RATIO[1])
                    self.max_distance_ratio.append(Finder.MAX_DISTANCE_RATIO[1])
                    self.min_quality_ratio.append(Finder.MIN_QUALITY_RATIO[1])
                    if self.logger is not None:
                        log_size(d_num, size, 'medium')
                if size < Finder.MIN_IMAGE_SIZE:
                    # too small, upsize it
                    if self.logger is not None:
                        self.logger.log('  Upsizing detection {} image from {} pixels to {}'.
                                        format(d_num, size, Finder.MIN_IMAGE_SIZE))
                    scale = Finder.MIN_IMAGE_SIZE / size
                    image = canvas.upsize(image, scale)
                else:
                    scale = 1.0
                self.sub_images.append((image, scale))
                if self.logger is not None:
                    self.logger.pop()
        if detection is None:
            return self.sub_images
        return self.sub_images[detection]

    def get_colour_image(self, detection):
        """ extract the grayscale sub-image of the given detection and colourize it """
        image, scale = self.get_grayscale(detection)
        return canvas.colourize(image), scale

    def get_image_size(self, detection):
        """ get the size of the sub-image for the given detection """
        image, _ = self.get_grayscale(detection)
        return canvas.size(image)

    def draw(self, image, file, detection):
        folder = utils.image_folder(target=detection.box_tl)
        self.logger.push(context=folder, folder=folder)
        self.logger.draw(image, file=file)
        self.logger.pop()

    def draw_detections(self):
        """ draw the detection rectangles and expected timing marks for diagnostic purposes """
        if self.logger is None:
            return

        def draw_marks(image, edge):
            # draw circle with a radius of the expected physical timing mark in orange
            if len(edge) > 1:
                marks = Finder.TIMING_CELLS  # assume we got a detection edge
            else:
                marks = [0]                  # its a single cell
            for mark in marks:
                if edge[mark] is None:
                    # nothing here
                    continue
                x, y, r, _ = edge[mark]
                canvas.circle(image, (x, y), r, const.ORANGE, 1)

        for i, detection in enumerate(self.detections):
            image, scale = self.get_colour_image(i)
            image = locator.draw_rectangle(image,
                                           detection.tl, detection.tr, detection.br, detection.bl,
                                           origin=detection.box_tl, scale=scale)
            expected_timing = self.expect_timing()[i]
            for edge in expected_timing:
                draw_marks(image, edge)
            self.draw(image, 'locators', detection)

    def draw_timing(self):
        """ draw the filtered timing marks for diagnostic purposes """
        if self.logger is None:
            return

        def draw_marks(image, found_timing, edge):
            # draw circle of the marks in the given edge in gree
            for tick in edge:
                if tick is None:
                    # nothing here
                    continue
                x, y, r, _ = found_timing[tick[0][0]]
                image = canvas.circle(image, (x, y), r, const.GREEN, 1)
            return image

        def draw_locator(image, detection, scale, locator, colour):
            # draw a coloured circle where the given locator is
            # locator co-ordinates are with respect to the original image, so they need to be scaled
            centre, radius = canvas.translate(locator, locator[2], self.detections[detection].box_tl, scale)
            return canvas.circle(image, centre, radius, colour)

        for detection, matches in self.filter_timing():
            image, scale = self.get_colour_image(detection)
            found_timing = self.found_timing[detection]
            top, right, bottom, left, corner = matches
            image = draw_locator(image, detection, scale, self.detections[detection].tl, const.RED)
            image = draw_locator(image, detection, scale, self.detections[detection].tr, const.ORANGE)
            image = draw_locator(image, detection, scale, self.detections[detection].bl, const.ORANGE)
            image = draw_marks(image, found_timing, top)
            image = draw_marks(image, found_timing, right)
            image = draw_marks(image, found_timing, bottom)
            image = draw_marks(image, found_timing, left)
            image = draw_marks(image, found_timing, corner)
            self.draw(image, 'timing', self.detections[detection])

    def draw_grids(self):
        """ draw the detected code grids for diagnostic purposes """
        if self.logger is None:
            return

        def draw_grid(image, here, there, pallete):
            """ draw lines of same cells from here to there on the given image """
            for cell in range(len(here)):
                src = here[cell]
                dst = there[cell]
                src_x = src[Finder.X_COORD]
                src_y = src[Finder.Y_COORD]
                dst_x = dst[Finder.X_COORD]
                dst_y = dst[Finder.Y_COORD]
                if (cell & 1) == 0:
                    colour = pallete[0]
                else:
                    colour = pallete[1]
                canvas.line(image, (src_x, src_y), (dst_x, dst_y), colour, 1)

        for detection, top, right, bottom, left in self.grids():
            image, _ = self.get_colour_image(detection)
            detection = self.detections[detection]
            draw_grid(image, top, bottom, (const.GREEN, const.BLUE))
            draw_grid(image, left, right, (const.BLUE, const.GREEN))
            tl_x, tl_y, tl_r, _ = top[0]
            canvas.circle(image, (tl_x, tl_y), tl_r, const.RED, 1)  # mark primary corner
            self.draw(image, 'grid', detection)

    def find_timing(self):  # ToDo: move timing blob tuning params to constants
        """ find candidate timing marks in all the detections,
            the co-ordinates of the found timing marks are relative to the detection box
            """
        if self.found_timing is not None:
            # already done it
            return self.found_timing
        self.found_timing = [None for _ in range(len(self.detections))]
        # setup fixed parameters
        params = Params()
        params.source_file = self.source
        params.box = None
        params.integration_width = 2
        params.black_threshold = 8  # ToDo: HACK-->10
        params.direct_neighbours = False
        params.inverted = True
        params.mode = const.RADIUS_MODE_MEAN
        params.blur_kernel_size = 0
        params.min_area = 4
        params.max_splitness  = (10, 10, 10)
        params.max_sameness   = (0.9, 0.9, 0.9)
        params.max_thickness  = (0.9, 0.8, 0.7)
        params.max_squareness = (0.7, 0.7, 0.7)
        params.max_wavyness   = (0.4, 0.4, 0.4)
        params.max_offsetness = (0.25, 0.25, 0.25)
        params.max_whiteness  = (0.6, 0.6, 0.6)
        params.max_blackness  = (0.6, 0.6, 0.6)
        # leave thickness, blackness on the defaults
        for i, detection in enumerate(self.detections):
            # do the timing mark detection
            image, _ = self.get_grayscale(i)
            if self.logger is not None:
                folder = utils.image_folder(target=detection.box_tl)
                self.logger.push(context='find_timing/{}'.format(folder), folder=folder)
                self.logger.log('')
                self.logger.log('Looking for timing marks in detection {}...'.format(i))
            params = locator.get_blobs(image, params, logger=self.logger)
            self.found_timing[i] = params.targets
            if self.logger is not None:
                self.logger.pop()
        return self.found_timing

    @staticmethod
    def get_step(a, b, spacing):
        """ from two locations, determine the x,y step size representing one code cell between them ('r'),
            spacing is number of cells between a and b, incrementing a location by the step size will move
            along a line between a and b irrespective of its rotation
            """
        span_x = b[0] - a[0]
        span_y = b[1] - a[1]
        step_x = span_x / spacing
        step_y = span_y / spacing
        return step_x, step_y

    @staticmethod
    def make_detection(detection, origin=(0, 0), image_scale=1.0, radius_scale=1.0) -> []:
        """ remove the origin and scale the radius of a detection,
            used to map image co-ordinates to detection box co-ordinates and vice-versa (if origin -ve)
            """
        x, y, r, l = detection
        r *= radius_scale
        (x, y), r = canvas.translate((x,y), r, origin, image_scale)
        # NB: create a new object, do not update existing one
        return [x, y, r, l]

    @staticmethod
    def make_steps(origin, start, end, steps: int, image_scale=1.0, radius_scale=TIMING_SCALE) -> [()]:
        """ make a series of x,y,r co-ordinate steps, in units of cells, from start to end """
        start   = Finder.make_detection(start, origin, image_scale, radius_scale)
        end     = Finder.make_detection(end, origin, image_scale, radius_scale)
        start_x = start[Finder.X_COORD]
        start_y = start[Finder.Y_COORD]
        radius  = start[Finder.R_COORD]
        step_x, step_y = Finder.get_step(start, end, steps)
        centres = []
        for step in range(steps + 1):
            centres.append([start_x + (step_x * step), start_y + (step_y * step), radius, None])
        return centres

    def expect_timing(self):
        """ make a list of expected timing blob centres for each detection,
            in clockwise order starting at 'top-left' locator (the 'primary' locator) -
            top-columns, right-rows, bottom-columns, left-rows - with a final single centre at bottom-right
            (the 'missing' corner locator - see visualisation in codes.py),
            the constructed list co-ordinates are relative to the box of the detection
            """
        if self.expected_timing is not None:
            # already done it
            return self.expected_timing

        self.expected_timing = []
        for i, detection in enumerate(self.detections):
            origin = detection.box_tl
            _, image_scale = self.get_grayscale(i)

            top_columns    = Finder.make_steps(origin, detection.tl, detection.tr, Finder.CELL_SPAN, image_scale)
            right_rows     = Finder.make_steps(origin, detection.tr, detection.br, Finder.CELL_SPAN, image_scale)
            bottom_columns = Finder.make_steps(origin, detection.bl, detection.br, Finder.CELL_SPAN, image_scale)
            left_rows      = Finder.make_steps(origin, detection.tl, detection.bl, Finder.CELL_SPAN, image_scale)

            # make the 'missing' corner 'minor' locator mark (make as a list to be type consistent with the edges)
            corners = [Finder.make_detection(detection.br, origin, image_scale)]

            expected_timing = [top_columns, right_rows, bottom_columns, left_rows, corners]
            self.expected_timing.append(expected_timing)

        if self.logger is not None:
            self.logger.push(context='expect_timing')
            self.draw_detections()
            self.logger.pop()

        return self.expected_timing

    def expected_marks(self):
        """ return the expected number of physical timing marks for a detection """
        expected_timing = self.expect_timing()
        expected = 0
        for detection in expected_timing:
            for edge, marks in enumerate(detection):
                if len(marks) > 1:
                    # its a box edge (we assume its len is compatible with Finder.TICKS)
                    for _ in Finder.TIMING_CELLS:
                        # its a physical mark
                        expected += 1
                else:
                    # its the bottom-right corner - always physical
                    expected += 1
            break  # they are all the same, so just do the first detection
        return expected

    def translate_locator(self, detection, locator):
        """ translate the given locator co-ordinates from the main image to its sub-image """
        origin = self.detections[detection].box_tl
        _, scale = self.get_colour_image(detection)
        centre, radius = canvas.translate(locator, locator[2], origin, scale)
        return centre, radius

    def match_timing(self):
        """ match candidate timing marks with those expected, matching is done on a proximity and size basis,
            result: for each detection, a list of matching expectations that are 'reasonable', a match is an
            expected and discovered pair with a distance and size measure for each,
            the ordering of the returned list within each detection is closest distance then nearest size,
            the 'reasonable' thresholds here are expected to be looser than those applied in filter_timing,
            NB: each detected mark can be associated with several expectations, its the job of filter_timing
                to sort that out
            """

        if self.matched_timing is not None:
            # already done it
            return self.matched_timing

        def is_excluded(exclusions, candidate):
            """ return True iff the given candidate is near one of the give exclusion zones,
                its 'near' if the circumference of the candidate touches or overlaps any exclusion
                """
            c_x, c_y, c_r = candidate
            for e_x, e_y, e_r in exclusions:
                separation = utils.distance((e_x, e_y), (c_x, c_y))
                inside = e_r + c_r
                inside *= inside
                if separation <= inside:
                    return True
            return False

        self.expect_timing()  # go set all expectations
        self.find_timing()    # go find all the timing marks

        self.matched_timing = [None for _ in range(len(self.detections))]  # create slot per detection
        for detection, found_timing in enumerate(self.find_timing()):
            # measure every discovered timing mark against every (relevant) expected mark
            # make a list of all those that are within spec
            # NB: the same expected mark may be associated with more than one discovered mark,
            #     that's OK, filter_timing deals with that
            if self.logger is not None:
                folder = utils.image_folder(target=self.detections[detection].box_tl)
                self.logger.push(context='match_timing/{}'.format(folder), folder=folder)
                self.logger.log('')

            max_distance, min_size, min_quality = self.get_limits(detection)
            self.matched_timing[detection] = []

            # setup locator exclusion zones (timing marks cannot be too near a locator)
            locators = []
            for locator in (self.detections[detection].tl, self.detections[detection].tr, self.detections[detection].bl):
                (x, y), r = self.translate_locator(detection, locator)
                locators.append((x, y, r * Finder.LOCATOR_MARGIN))
            # find all qualifying timing marks
            for candidate, (fx, fy, fr, _) in enumerate(found_timing):
                if is_excluded(locators, (fx, fy, fr)):
                    # too near a locator
                    continue
                expected_timing = self.expect_timing()[detection]
                for edge, marks in enumerate(expected_timing):
                    for mark, (x, y, r, _) in enumerate(marks):
                        if len(marks) > 1:
                            # its an edge (and not a corner) - must filter for physical ticks
                            if mark not in Finder.TIMING_CELLS:
                                # its not physical - ignore it
                                continue
                        distance = utils.distance((x, y), (fx, fy))
                        size = utils.ratio(r, fr)  # closer to 1 means closer radius match
                        tick = ((candidate, edge, mark), distance / (r * r), size)  # NB: r^2 'cos distance is also squared
                        # chuck out those out of spec on any metric
                        _, metric1, metric2, metric3 = Finder.get_metrics(tick,
                                                                          max_distance, min_size, min_quality,
                                                                          Finder.MATCH_METRIC_SCALE)
                        if metric1[0] or metric2[0]:  # ToDo: HACK--> or metric3[0]:
                            continue
                        # potentially useful
                        self.matched_timing[detection].append(tick)

            # put into closest distance then nearest size order
            # (NB: size is in range 1..0, good to bad, so 1-size is 0..1 good to bad
            self.matched_timing[detection].sort(key=lambda k: (k[1], 1-k[2]))

            if self.logger is not None:
                matches = self.matched_timing[detection]
                self.logger.log('Detection {}: has {} matches'.format(detection, len(matches)))
                for tick in matches:
                    self.log_tick(detection, tick)

            if self.logger is not None:
                self.logger.pop()

        return self.matched_timing

    @staticmethod
    def get_metrics(tick, max_distance, min_size, min_quality, scale=1):
        """ get the metrics for the given tick and test against the given limits,
            max_distance is the maximum tolerable distance of the tick position from the expected position (squared)
            min_size is the minimum actual/expected size ratio tolerated
            min_quality is the minimum quality tolerated,
            scale is a scaling factor to apply before testing the limits, 1==as is, >1==be looser, <1==be stricter
            for each test a tuple of good/bad and number is returned preceded by the found timing mark number
            """
        scale_squared = scale * scale  # for use with squared metrics
        match, distance, size = tick
        if distance > max_distance:  # NB: do not used scaled numbers here
            distance_ratio = 0  # quality is 0 if gap too big
        else:
            distance_ratio = 1 - distance / max_distance  # 0==bad, 1==perfect
        # determine quality
        size_quality     = size * size  # size squared to be compatible with distance which is also squared
        # we want the weights to 'suppress' the effect of the component, the components are in the range 0..1
        # where 0 is bad and 1 is good, we want to bias things towards good, so we apply the weight to the
        # 'badness' to make it less bad, the badness is 1-goodness, the weights are in the range 0..1 so
        # multiplying the badness by the weight reduces the badness, the 1-thing converts badness into goodness
        size_quality     = 1 - ((1 - size_quality  ) * Finder.SIZE_QUALITY_WEIGHT    )
        distance_quality = 1 - ((1 - distance_ratio) * Finder.DISTANCE_QUALITY_WEIGHT)
        quality          = size_quality * distance_quality
        # NB: true quality is the square root of the above, 0==bad, 1==perfect
        distance_bad = distance                  > (max_distance * scale_squared)
        size_bad     = (size    * scale        ) < min_size
        quality_bad  = (quality * scale_squared) < min_quality
        return match, (distance_bad, distance, distance_ratio), (size_bad, size), (quality_bad, quality)

    def log_tick(self, detection, tick, prefix='    '):
        """ log a tick for diagnostic purposes, returns good/bad state for distance, size and quality """
        if self.logger is None:
            return None
        max_distance, min_size, min_quality = self.get_limits(detection)
        (candidate, edge, mark), distance, size, quality = Finder.get_metrics(tick, max_distance, min_size, min_quality)
        actual   = self.found_timing[detection][candidate]
        expected = self.expected_timing[detection][edge][mark]
        span     = math.sqrt(utils.distance(actual, expected))
        if distance[0]:
            distance_status = '(bad)'
        else:
            distance_status = ''
        if size[0]:
            size_status = '(bad)'
        else:
            size_status = ''
        if quality[0]:
            quality_status = '(bad)'
        else:
            quality_status = ''
        self.logger.log('{}Candidate {}: {:.2f}x, {:.2f}y, {:.2f}r  -->  '
                        ' Edge {}, Mark {}: expected {:.2f}x, {:.2f}y, {:.2f}r'
                        ' (distance ({:.2f}->{:.2f}) ratio{} {:.2f}, size ratio{} {:.2f}, quality{} {:.2f})'.
                        format(prefix, candidate, actual[0], actual[1], actual[2],
                               edge, mark, expected[0], expected[1], expected[2],
                               span, math.sqrt(distance[1]), distance_status, math.sqrt(distance[2]),
                               size_status, size[1],
                               quality_status, math.sqrt(quality[1])))
        return distance, size, quality

    def get_limits(self, detection):
        """ get the distance, size and quality limits for the given detection in units required for comparisons """
        min_quality_ratio   = self.min_quality_ratio[detection]
        min_quality_ratio  *= min_quality_ratio  # squared 'cos that's what get_metrics produces
        min_size_ratio      = self.min_radius_ratio[detection]
        max_distance_ratio  = self.max_distance_ratio[detection]
        max_distance_ratio *= max_distance_ratio  # squared to be units compatible with matched timing distance
        return max_distance_ratio, min_size_ratio, min_quality_ratio

    def filter_timing(self):
        """ filter candidates that are too distant from expectations and detections with too few matches """
        # the lists we work with here (found_timing) are a list of all expected/discovered pairs that are a
        # 'reasonable' match, our job is to extract the best set of pairings,
        # drop all those too distant, too dissimilar in size, or too close together to be considered a potential
        # match (tuning constants), if not enough matches (another tuning constant) drop the detection as junk
        if self.filtered_timing is not None:
            # already done it
            return self.filtered_timing

        if self.logger is not None:
            span  = max(Finder.MAX_DISTANCE_RATIO)+1
            slots = int(span / 0.1)  # want each slot to be 0.1
            dist_good_stats = utils.Stats(slots, (0, span))
            dist_bad_stats  = utils.Stats(slots, (0, span))
            size_good_stats = utils.Stats(10, (0, 1))
            size_bad_stats  = utils.Stats(10, (0, 1))
            qual_good_stats = utils.Stats(10, (0, 1))
            qual_bad_stats  = utils.Stats(10, (0, 1))

        def log_tick(detection, tick, prefix='    '):
            if self.logger is None:
                return
            distance, size, quality = self.log_tick(detection, tick, prefix)
            if size[0]:
                size_bad_stats.count(size[1])
            else:
                size_good_stats.count(size[1])
            if distance[0]:
                dist_bad_stats.count(distance[1])
            else:
                dist_good_stats.count(distance[1])
            if quality[0]:
                qual_bad_stats.count(quality[1])
            else:
                qual_good_stats.count(quality[1])

        def log_locator(detection, locator, name):
            if self.logger is None:
                return
            centre, radius = self.translate_locator(detection, locator)
            self.logger.log('  Locator ({}): {:.2f}x, {:.2f}y, {:.2f}r'.format(name, centre[0], centre[1], radius))

        def get_tick_circle(detection, tick):
            """ get the centre and radius of the given tick """
            (candidate, edge, mark), distance, size = tick
            circle = self.found_timing[detection][candidate]
            return circle

        def tick_gap(detection, from_tick, to_tick):
            """ determine the separation between two ticks,
                the separation is the gap between their circumferences in pixels,
                negative means they are overlapping
                (as the distance squared between their centres)
                """
            from_circle = get_tick_circle(detection, from_tick)
            to_circle   = get_tick_circle(detection, to_tick)
            separation  = utils.distance(from_circle, to_circle)  # centre to centre distance squared
            touching    = from_circle[2] + to_circle[2]  # sum of radii
            touching   *= touching  # squared
            gap         = separation - touching
            return gap

        self.filtered_timing = []
        max_hits = self.expected_marks()  # maximum possible marks per detection
        for detection, matches in enumerate(self.match_timing()):
            max_distance, min_size, min_quality = self.get_limits(detection)
            if self.logger is not None:
                folder = utils.image_folder(target=self.detections[detection].box_tl)
                self.logger.push(context='filter_timing/{}'.format(folder), folder=folder)
                self.logger.log('')
            # make allocation tracking lists (so we only allocate one pairing once)
            committed_candidates = [None for _ in range(len(self.find_timing()[detection]))]  # candidates we've used
            expected_timing = self.expect_timing()[detection]
            matched_expectation = [None for _ in range(len(expected_timing))]
            for edge, marks in enumerate(expected_timing):
                matched_expectation[edge] = [None for _ in range(len(marks))]  # expectations we've done
            # matches is a distance/size ordered list of expected/discovered pairings
            got_hits = 0
            for tick in matches:
                (candidate, edge, mark), distance, size, quality = Finder.get_metrics(tick,
                                                                                      max_distance,
                                                                                      min_size,
                                                                                      min_quality)
                # check allowed to use this one
                if committed_candidates[candidate] is not None:
                    # already used this, so cannot use it again here
                    if self.logger is not None:
                        self.logger.log('Detection {}, candidate {} already used in:'.format(detection, candidate))
                        log_tick(detection, committed_candidates[candidate])
                    continue
                if matched_expectation[edge][mark] is not None:
                    # already allocated this edge/mark, so do not allocate this candidate to it
                    if self.logger is not None:
                        self.logger.log('Detection {}, edge {}, mark {} already paired with:'.
                                        format(detection, edge, mark))
                        log_tick(detection, matched_expectation[edge][mark])
                    continue
                # we've not used either side of this pairing yet, see if its in spec
                if distance[0]:
                    # distance too far from expected - ignore it
                    if self.logger is not None:
                        self.logger.log('Detection {}: candidate {} to edge {}, mark {} pairing'
                                        ' distance ratio ({:.2f}) too high, limit is {}'.
                                        format(detection, candidate, edge, mark,
                                               math.sqrt(distance[1]), math.sqrt(max_distance)))
                        log_tick(detection, tick)
                    continue
                if size[0]:
                    # size too divergent from expected - ignore it
                    if self.logger is not None:
                        self.logger.log('Detection {}, candidate {} to edge {}, mark {} pairing'
                                        ' size ratio ({:.2f}) too low, limit is {}'.
                                        format(detection, candidate, edge, mark, size[1], min_size))
                        log_tick(detection, tick)
                    continue
                if quality[0]:
                    # quality too low - ignore it
                    if self.logger is not None:
                        self.logger.log('Detection {}, candidate {} to edge {}, mark {} pairing'
                                        ' quality ({:.2f}) too low, limit is {}'.
                                        format(detection, candidate, edge, mark,
                                               math.sqrt(quality[1]), math.sqrt(min_quality)))
                        log_tick(detection, tick)
                    continue
                # found a pairing we're allowed to use that is in 'spec'
                # we further check it is not too close one that has already been allocated,
                # allocations so far will always be closer to their expectation than this one,
                # if this one is not 'separated' from that by some threshold we must ignore it
                too_close = None
                expected_radius  = expected_timing[edge][mark][2]
                expected_radius *= expected_radius  # squared for comparison with measured gap
                minimum_gap      = expected_radius * Finder.MIN_TIMING_MARK_GAP
                for matched_edge, edge_marks in enumerate(matched_expectation):
                    for matched_mark, matched_tick in enumerate(edge_marks):
                        if matched_tick is None:
                            continue
                        gap = tick_gap(detection, matched_tick, tick)
                        # the gap must be +ve and at least 1r (expected timing mark radius)
                        if gap < minimum_gap:
                            too_close = (matched_edge, matched_mark, gap)
                            break
                    if too_close is not None:
                        break
                if too_close is not None:
                    if self.logger is not None:
                        self.logger.log('Detection {}, candidate {} too close ({:2f}, limit is {:.2f})'
                                        ' to existing allocation at edge {}, mark {}'.
                                        format(detection, candidate, too_close[2],
                                               minimum_gap, too_close[0], too_close[1]))
                        log_tick(detection, tick)
                    continue
                # passed all filters, so use it
                committed_candidates[candidate] = tick
                matched_expectation[edge][mark] = tick
                got_hits += 1
                if self.logger is not None:
                    self.logger.log('Detection {}, candidate {} paired to edge {}, mark {}'.
                                    format(detection, candidate, edge, mark))
                    log_tick(detection, tick)
            # done this detection
            if (got_hits / max_hits) < Finder.MIN_MARK_HITS:
                # not enough mark hits for this detection to qualify
                if self.logger is not None:
                    self.logger.log('')
                    self.logger.log('Detection {}: Has too few detectable timing marks - require at least {:.0f} of {},'
                                    ' only found {} ({:.2f}%, threshold is {:.2f}%):'.
                                    format(detection, Finder.MIN_MARK_HITS * max_hits, max_hits, got_hits,
                                           (got_hits / max_hits) * 100, Finder.MIN_MARK_HITS * 100))
            else:
                self.filtered_timing.append((detection, matched_expectation))
                if self.logger is not None:
                    self.logger.log('')
                    self.logger.log('Detection {}: Found {} (of {}, {:.2f}%) timing marks:'.
                                    format(detection, got_hits, max_hits, (got_hits / max_hits) * 100))
            if self.logger is not None:
                log_locator(detection, self.detections[detection].tl, 'top-left')
                log_locator(detection, self.detections[detection].tr, 'top-right')
                log_locator(detection, self.detections[detection].bl, 'bottom-left')
                for edge, marks in enumerate(matched_expectation):
                    self.logger.log('  Edge {} targets:'.format(edge))
                    for mark, tick in enumerate(marks):
                        if tick is None:
                            continue  # no tick here
                        self.logger.log('    Mark {}:'.format(mark))
                        log_tick(detection, tick, '      ')
                self.logger.pop()

        if self.logger is not None:
            self.logger.push(context='filter_timing')
            self.logger.log('')
            self.logger.log('Size good stats: {}'.format(size_good_stats.show()))
            self.logger.log('Size bad stats: {}'.format(size_bad_stats.show()))
            self.logger.log('Distance good stats: {}'.format(dist_good_stats.show()))
            self.logger.log('Distance bad stats: {}'.format(dist_bad_stats.show()))
            self.logger.log('Quality good stats: {}'.format(qual_good_stats.show()))
            self.logger.log('Quality bad stats: {}'.format(qual_bad_stats.show()))
            self.logger.log('')
            self.draw_timing()
            self.logger.pop()

        return self.filtered_timing

    def grids(self):
        """ return a list of code area grid lines discovered,
            the list consists of row, column co-ordinates and radii for each 'line' inside the locators,
            the co-ordinates are relative to the grayscale image and consist of top columns, right rows,
            bottom columns, left rows (i.e. clockwise), the 'grid' is formed by joining like cell address
            top to bottom and left to right
            """
        if self.code_grids is not None:
            # already done it
            return self.code_grids

        def get_found_mark(detection, found_marks, mark):
            """ get the identified found mark """
            if found_marks[mark] is None:
                # there isn't one here
                return None
            found_mark_number = found_marks[mark][0][0]
            found = self.found_timing[detection][found_mark_number]
            return found

        def make_columns(detection, found_marks, found_locators):
            """ make the full column repertoire from the found marks and locators for the given detection """
            columns = [None for _ in range(len(found_marks))]  # create every possible column between locators (incl.)
            for mark in range(len(found_marks)):
                columns[mark] = get_found_mark(detection, found_marks, mark)  # NB: maybe None
            columns[0]  = found_locators[0]   # discovered locator
            columns[-1] = found_locators[-1]  # ..
            # we now have the detected location of the timing marks and locators in columns,
            # we estimate a missing column from its neighbours, gaps away from the locators span a single cell,
            # gaps adjacent to the locators span two cells (see visualisation in codes.py)
            for col in range(1, len(columns)-1):  # we know the first and last are present (the locators)
                if columns[col] is None:
                    # got a gap that needs filling from col-1 to next non-None column
                    gap = 1  # start a new gap
                    end_at = None
                    for end_gap in range(col+1, len(columns)):
                        if columns[end_gap] is None:
                            # gap continues
                            gap += 1
                        else:
                            # found gap end - we *know* we get here 'cos the last column is never None
                            end_at = end_gap
                            break
                    if end_at is None:
                        raise Exception('end locator missing')
                    a = columns[col-1]
                    b = columns[end_at]
                    step = Finder.get_step(a, b, gap+1)
                    x = a[Finder.X_COORD] + step[0]
                    y = a[Finder.Y_COORD] + step[1]
                    r = (a[Finder.R_COORD] + b[Finder.R_COORD]) / 2  # radius is average of a and b
                    columns[col] = [x, y, r, None]
            return columns

        self.code_grids = []
        for detection, matches in self.filter_timing():
            top, right, bottom, left, corner = matches
            top_ref, right_ref, bottom_ref, left_ref, corner_ref = self.expected_timing[detection]
            # region calculate actual bottom right refs from the discovered corner
            br = get_found_mark(detection, corner, 0)
            if br is not None:
                # found a matching mark, overwrite the estimate
                right_ref [-1] = Finder.make_detection(br, (0,0), 1.0, 1.0)
                bottom_ref[-1] = Finder.make_detection(br, (0,0), 1.0, 1.0)
            # endregion
            top_cols    = make_columns(detection, top,    top_ref)
            right_cols  = make_columns(detection, right,  right_ref)
            bottom_cols = make_columns(detection, bottom, bottom_ref)
            left_cols   = make_columns(detection, left,   left_ref)
            self.code_grids.append([detection, top_cols, right_cols, bottom_cols, left_cols])

        if self.logger is not None:
            self.logger.push('grids')
            self.draw_grids()
            self.logger.pop()

        return self.code_grids

    def images(self):
        """ get the sub-images for all the accepted detections """
        if self.good_images is None:
            self.good_images = []
            for detection, _ in self.filter_timing():
                self.good_images.append(self.get_grayscale(detection))
        return self.good_images

    def get_detections(self):
        """ return all our detections for upstream use """
        ids = []  # these are only used for logging diagnostics
        for detection, _ in self.filter_timing():
            ids.append(self.detections[detection].box_tl)
        return self.grids(), self.images(), ids

def find_codes(src, params, image, detections, logger):
    """ find the valid code areas within the given detections """
    if logger is not None:
        logger.push('find_codes')
        logger.log('')
    finder = Finder(src, image, detections, logger)
    finder.get_detections()
    params.finder = finder
    if logger is not None:
        logger.pop()
    return params  # for upstream access


def _test(src, proximity, blur, mode, params=None, logger=None, create_new=True):
    """ ************** TEST **************** """
    
    if logger.depth() > 1:
        logger.push('finder/_test')
    else:
        logger.push('_test')
    logger.log('')
    logger.log('Finding timing marks (create new {})'.format(create_new))

    # get the detections
    if not create_new:
        params = logger.restore(file='finder', ext='params')
        if params is None or params.source_file != src:
            create_new = True
    if create_new:
        # this is very slow
        if params is None:
            params = Params()
        params = locator._test(src, proximity, blur=blur, mode=mode, params=params, logger=logger, create_new=create_new)
        if params is None:
            logger.log('Locator failed on {}'.format(src))
            logger.pop()
            return None
        logger.save(params, file='finder', ext='params')

    located = params.locator
    image = params.source
    detections = located.get_detections()

    # process the detections
    params = find_codes(src, params, image, detections, logger)
    logger.pop()
    return params  # for upstream test harness


if __name__ == "__main__":
    """ test harness """

    #src = '/home/dave/precious/fellsafe/fellsafe-image/source/kilo-codes/codes/test-alt-bits.png'
    #src = '/home/dave/precious/fellsafe/fellsafe-image/source/kilo-codes/codes/test-code-145.png'
    src = '/home/dave/precious/fellsafe/fellsafe-image/media/kilo-codes/kilo-codes-close-150-257-263-380-436-647-688-710-777.jpg'
    #proximity = const.PROXIMITY_CLOSE
    proximity = const.PROXIMITY_FAR

    logger = utils.Logger('finder.log', 'finder/{}'.format(utils.image_folder(src)))

    _test(src, proximity, blur=const.BLUR_KERNEL_SIZE, mode=const.RADIUS_MODE_MEAN, logger=logger, create_new=False)
