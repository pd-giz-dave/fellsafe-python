""" Finder for the code area
    This module takes a list of locator detections and finds the associated data cell co-ordinates
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
    MIN_IMAGE_SIZE = 128  # min target image size, smaller ones are upsized to at least this
    # distant=30 or less, far=40 or less, near=50 or less, close=100 or less, native=300+
    SMALL_IMAGE_SIZE = 40  # threshold for a small detection image
    LARGE_IMAGE_SIZE = 80  # ditto for a big one
    # these ratios have to be sloppy for small images 'cos 1 pixel diff can be huge with low resolution images
    # these ratios are indexed by the image size index (0=small, 1=medium, 2=large)
    MIN_RADIUS_RATIO = (0.4, 0.6, 0.6)  # min radius deviation from expected (1=perfect, 0=rubbish, 0.5=2:1)
    MAX_DISTANCE_RATIO = (5*5, 3*3, 3*3)  # max distance (squared) between actual and expected mark as a multiple of expected radius
    MIN_MARK_HITS = 5/TIMING_MARKS  # minimum number of matched marks for a detection to qualify as a ratio of the maximum, 1==all
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
        self.matched_timing     = None        # shouldbe_timing's matched with closest maybe_timing's
        self.filtered_timing    = None        # list of detections that pass the qualification filter
        self.code_grids         = None        # list of detected code grids
        self.code_cells         = None        # list of cell co-ords inside the locators
        self.code_circles       = None        # list of all code 'circles' (max enclosed circle within detected locators)
        self.min_radius_ratio   = None        # appropriate radius ratio for each sub-image size
        self.max_distance_ratio = None        # appropriate distance ratio for each sub-image size

    def get_grayscale(self, detection=None):
        """ extract the grayscale sub-image of the given, or all, detection """

        def log_size(detection, size, size_name):
            if self.logger is not None:
                self.logger.log('Detection {} is {} ({} pixels, small limit is {}, large limit is {}), '
                                'min_radius_ratio={:.2f}, max_distance_ratio={:.2f}'.
                                format(detection, size_name, size, Finder.SMALL_IMAGE_SIZE, Finder.LARGE_IMAGE_SIZE,
                                       self.min_radius_ratio[detection], math.sqrt(self.max_distance_ratio[detection])))

        if self.sub_images is None:
            self.sub_images         = []
            self.min_radius_ratio   = []
            self.max_distance_ratio = []
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
                    if self.logger is not None:
                        log_size(d_num, size, 'small')
                elif size >= Finder.LARGE_IMAGE_SIZE:
                    self.min_radius_ratio.append(Finder.MIN_RADIUS_RATIO[2])
                    self.max_distance_ratio.append(Finder.MAX_DISTANCE_RATIO[2])
                    if self.logger is not None:
                        log_size(d_num, size, 'large')
                else:  # medium
                    self.min_radius_ratio.append(Finder.MIN_RADIUS_RATIO[1])
                    self.max_distance_ratio.append(Finder.MAX_DISTANCE_RATIO[1])
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
        """ extract the grayscale sub-image of the given detection and coliurize it """
        image, scale = self.get_grayscale(detection)
        return canvas.colourize(image), scale

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
            for mark in edge:
                if mark is None or len(mark) == 0:
                    # nothing here
                    continue
                x, y, r, _ = found_timing[mark[0][0]]  # ToDo: euch! all these [0]'s
                image = canvas.circle(image, (x, y), r, const.GREEN, 1)
            return image

        def draw_locator(image, detection, scale, locator, colour):
            # draw a coloured circle where the given locator is
            # locator co-ordinates are with respect to the original image, so they need to be scaled
            centre, radius = canvas.translate(locator, locator[2], self.detections[detection].box_tl, scale)
            return canvas.circle(image, centre, radius, colour)

        for detection in self.filter_timing():
            image, scale = self.get_colour_image(detection)
            found_timing = self.found_timing[detection]
            top, right, bottom, left, corner = self.matched_timing[detection]
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
                src   = here [cell]
                dst   = there[cell]
                src_x = src[Finder.X_COORD]
                src_y = src[Finder.Y_COORD]
                dst_x = dst[Finder.X_COORD]
                dst_y = dst[Finder.Y_COORD]
                if (cell & 1) == 0:
                    colour = pallete[0]
                else:
                    colour = pallete[1]
                canvas.line(image, (src_x, src_y), (dst_x, dst_y), colour, 1)

        for detection, (top, right, bottom, left) in enumerate(self.grids()):
            detection = self.filter_timing()[detection]
            image, _  = self.get_colour_image(detection)
            detection = self.detections[detection]
            draw_grid(image, top, bottom, (const.GREEN, const.BLUE))
            draw_grid(image, left,right, (const.BLUE, const.GREEN))
            tl_x, tl_y, tl_r, _ = top[0]
            canvas.circle(image, (tl_x, tl_y), tl_r, const.RED, 1)  # mark primary corner
            self.draw(image, 'grid', detection)

    def draw_cells(self):
        """ draw the detected code cells for diagnostic purposes """
        if self.logger is None:
            return

        for detection, rows in enumerate(self.cells()):
            detection = self.filter_timing()[detection]
            image, _  = self.get_colour_image(detection)
            detection = self.detections[detection]
            for row, cells in enumerate(rows):
                for col, (x, y, r, _) in enumerate(cells):
                    if row == 0 and col == 0:
                        # highlight reference point
                        colour = const.RED
                    elif (row & 1) == 0 and (col & 1) == 0:
                        # even row and even col
                        colour = const.GREEN
                    elif (row & 1) == 0 and (col & 1) == 1:
                        # even row and odd col
                        colour = const.BLUE
                    elif (row & 1) == 1 and (col & 1) == 0:
                        # odd row and even col
                        colour = const.BLUE
                    elif (row & 1) == 1 and (col & 1) == 1:
                        # odd row and odd col
                        colour = const.GREEN
                    image = canvas.circle(image, (x, y), r, colour, 1)
            self.draw(image, 'cells', detection)

    def circles(self, detection=None):
        """ get the radius and centre of the maximum enclosed circle of the given, or all, accepted detections """
        if self.code_circles is None:
            self.code_circles = []
            for accepted in self.filter_timing():
                d = self.detections[accepted]
                # centre is mid-point of the bottom-left (bl) to top-right (br) locator diagonal as:
                #   centre_x = (tr_x-bl_x)/2 + bl_x and centre_y = (tr_y-bl_y)/2 + bl_y
                bl_x, bl_y, _, _ = d.bl
                tr_x, tr_y, _, _ = d.tr
                centre = ((tr_x - bl_x) / 2 + bl_x, (tr_y - bl_y) / 2 + bl_y)
                # radius is the average of the centre to top-left or centre to bottom-left distance
                c2tl = utils.distance(centre, d.tl)    # NB: returns the squared distance
                c2bl = utils.distance(centre, d.bl)    #     ..
                radius = math.sqrt((c2tl + c2bl) / 2)  #     so square root to get actual distance
                origin = d.box_tl
                self.code_circles.append((centre, radius, origin))  # origin used for diagnostics (as an id)
        if detection is None:
            return self.code_circles
        return self.code_circles[detection]

    def find_timing(self):
        """ find candidate timing marks in all the detections,
            the co-ordinates of the found timing marks are relative to the detection box
            """
        if self.found_timing is not None:
            # already done it
            return self.found_timing
        self.found_timing = [None for _ in range(len(self.detections))]
        # setup fixed parameters
        params = locator.Params()
        params.source_file = self.source
        params.box = None
        params.integration_width = 2
        params.black_threshold = 10
        params.direct_neighbours = False
        params.inverted = True
        params.mode = const.RADIUS_MODE_MEAN
        params.blur_kernel_size = 0
        params.min_area = 8
        params.min_size = 4
        params.blur_kernel_size = 0
        params.max_squareness = 0.7
        params.max_wavyness = 0.4
        params.max_offsetness = 0.25
        params.max_whiteness = 0.6
        for i, detection in enumerate(self.detections):
            # do the timing mark detection
            image, _ = self.get_grayscale(i)
            if self.logger is not None:
                folder = utils.image_folder(target=detection.box_tl)
                self.logger.push(context='find_timing/{}'.format(folder), folder=folder)
                self.logger.log('')
                self.logger.log('Looking for timing marks...')
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
        radius  = (start[Finder.R_COORD] + end[Finder.R_COORD]) / 2  # use average of locators as the radius
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
            corners = [Finder.make_detection(detection.br, origin, image_scale, Finder.TIMING_SCALE)]

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
            result: for each detection, for each candidate the indices of the closest expectation,
            if there are more expectations than candidates, some will not have a match
            """

        if self.matched_timing is not None:
            # already done it
            return self.matched_timing

        def distance(a, b):
            """ return the squared distance between a and b """
            ab_x = b[0] - a[0]
            ab_y = b[1] - a[1]
            return (ab_x * ab_x) + (ab_y * ab_y)

        def ratio(a, b):
            """ return the ratio of a/b or b/a whichever is smaller,
                this provides a measure of how close to each other the two numbers are,
                the nearer to 1 the ratio is the closer the numbers are to each other
                """
            result = min(a, b) / max(a, b)  # range 0..1, 0=distant, 1=close
            return result

        def is_locator(detection, locator):
            """ return True iff the given locator is one of the locators for the given detection,
                its a locator if it is inside one of the detection locators or vice-versa,
                to be inside the distance between the centres is less than the largest radius
                """
            tl = self.translate_locator(detection, self.detections[detection].tl)
            tr = self.translate_locator(detection, self.detections[detection].tr)
            bl = self.translate_locator(detection, self.detections[detection].bl)
            l_x, l_y, l_r = locator
            for (x, y), r in (tl, tr, bl):
                separation = utils.distance((x, y), (l_x, l_y))
                inside = max(l_r, r)
                inside *= inside
                if separation <= inside:
                    return True
            return False

        self.matched_timing = [None for _ in range(len(self.detections))]  # create slot per detection
        for detection, found_timing in enumerate(self.find_timing()):
            for candidate, (fx, fy, fr, _) in enumerate(found_timing):
                if is_locator(detection, (fx, fy, fr)):
                    # ignore locators
                    continue
                # find the best expectation to give this candidate to
                best_edge = None
                best_mark = None
                best_gap  = None
                best_size = None
                best_r    = None  # used to calculate ratios for filtering, see filter_timing() - all same for an edge
                expected_timing = self.expect_timing()[detection]
                if self.matched_timing[detection] is None:
                    self.matched_timing[detection] = [None for _ in range(len(expected_timing))]  # create edge slots
                for edge, marks in enumerate(expected_timing):
                    if self.matched_timing[detection][edge] is None:
                        self.matched_timing[detection][edge] = [None for _ in range(len(marks))]  # create mark slots
                    for mark, (x, y, r, _) in enumerate(marks):
                        if len(marks) > 1:
                            # its an edge - must filter for physical ticks
                            if mark not in Finder.TIMING_CELLS:
                                # its not physical - ignore it
                                continue
                        if self.matched_timing[detection][edge][mark] is None:
                            self.matched_timing[detection][edge][mark] = []  # create empty mark list
                        gap = distance((x, y), (fx, fy))
                        size = ratio(r, fr)  # closer to 1 means closer radius match
                        if best_edge is None:
                            best_edge = edge
                            best_mark = mark
                            best_gap = gap
                            best_size = size
                            best_r = r * r
                        elif gap < best_gap:
                            # this one is closer in distance
                            best_edge = edge
                            best_mark = mark
                            best_gap = gap
                            best_size = size
                            best_r = r * r
                        elif gap > best_gap:
                            # this one is further away in distance
                            continue
                        elif size > best_size:
                            # this one is same distance but is closer in size
                            best_edge = edge
                            best_mark = mark
                            best_gap = gap
                            best_size = size
                            best_r = r * r
                    # now get best match along this edge
                # now got best match along all edges of this detection for this candidate
                self.matched_timing[detection][best_edge][best_mark].append((candidate, best_gap, best_size, best_r))
            # now allocated all found marks to expected marks
            # NB: There maybe multiple matches for each mark, the resultant list is in an arbitrary order
        # now done all detections

        return self.matched_timing

    def filter_timing(self):
        """ filter candidates that are too distant from expectations and detections with too few matches """
        # drop all but the closest match for each expectation
        # drop all those too distant to be considered a potential match (a tuning constant)
        # if not enough matches (another tuning constant) drop the detection as junk
        if self.filtered_timing is not None:
            # already done it
            return self.filtered_timing

        if self.logger is not None:
            size_good_stats = utils.Stats(10, (0, 1))
            size_bad_stats  = utils.Stats(10, (0, 1))
            span = math.sqrt(max(Finder.MAX_DISTANCE_RATIO))+1
            slots = int(span / 0.1)  # want each slot to be 0.1
            dist_good_stats = utils.Stats(slots, (0, span))
            dist_bad_stats  = utils.Stats(slots, (0, span))

        def log_tick(detection, edge, mark, tick):
            if self.logger is None:
                return
            candidate, distance, size_ratio, rr = tick
            actual    = self.found_timing[detection][candidate]
            expected  = self.expected_timing[detection][edge][mark]
            gap_ratio = distance / rr  # convert units of pixels^2 into expected radius^2
            gap_bad   = gap_ratio > self.max_distance_ratio[detection]
            size_bad  = size_ratio < self.min_radius_ratio[detection]
            gap_ratio = math.sqrt(gap_ratio)
            self.logger.log('    {}: {:.2f}x, {:.2f}y, {:.2f}r'
                            ' (expected {:.2f}x, {:.2f}y, {:.2f}r)'
                            ' distance ratio {:.2f}, size ratio {:.2f}'.
                            format(candidate, actual[0], actual[1], actual[2],
                                   expected[0], expected[1], expected[2],
                                   gap_ratio, size_ratio))
            if size_bad:
                size_bad_stats.count(size_ratio)
            else:
                size_good_stats.count(size_ratio)
            if gap_bad:
                dist_bad_stats.count(gap_ratio)
            else:
                dist_good_stats.count(gap_ratio)

        def log_locator(detection, locator, name):
            if self.logger is None:
                return
            centre, radius = self.translate_locator(detection, locator)
            self.logger.log('  Locator ({}): {:.2f}x, {:.2f}y, {:.2f}r'.format(name, centre[0], centre[1], radius))

        self.filtered_timing = []
        max_hits = self.expected_marks()  # maximum possible marks per detection
        for detection, edges in enumerate(self.match_timing()):
            max_distance_ratio = self.max_distance_ratio[detection]
            min_size_ratio     = self.min_radius_ratio  [detection]
            if self.logger is not None:
                folder = utils.image_folder(target=self.detections[detection].box_tl)
                self.logger.push(context='filter_timing/{}'.format(folder), folder=folder)
                self.logger.log('')
            got_hits = 0  # how many marks actually qualify for this detection
            if edges is not None:
                for edge, marks in enumerate(edges):
                    for mark in range(len(marks)):
                        if marks[mark] is None:
                            continue  # no marks here
                        # what we have in marks[mark] is an un-ordered list of detected blobs close to our expectation
                        # drop all but closest that is within acceptable limits
                        closest_mark = None
                        closest_gap = None
                        closest_size = None
                        for m, tick in enumerate(marks[mark]):
                            candidate, distance, size_ratio, rr = tick
                            gap_ratio = distance / rr  # convert units of pixels^2 into expected radius^2
                            if gap_ratio > max_distance_ratio:
                                # distance too far from expected - ignore it
                                if self.logger is not None:
                                    self.logger.log('Detection {}, edge {}, mark {}: Distance ratio ({:.2f}) too high,'
                                                    ' limit is {}'.format(detection, edge, mark, math.sqrt(gap_ratio),
                                                                          math.sqrt(max_distance_ratio)))
                                    log_tick(detection, edge, mark, tick)
                                continue
                            if size_ratio < min_size_ratio:
                                # size too divergent from expected - ignore it
                                if self.logger is not None:
                                    self.logger.log('Detection {}, edge {}, mark {}: Size ratio ({:.2f}) too low,'
                                                    ' limit is {}'.format(detection, edge, mark,
                                                                          size_ratio, min_size_ratio))
                                    log_tick(detection, edge, mark, tick)
                                continue
                            if closest_mark is None:
                                closest_mark = m
                                closest_gap = gap_ratio
                                closest_size = size_ratio
                                continue
                            if gap_ratio < closest_gap:
                                closest_mark = m
                                closest_gap = gap_ratio
                                closest_size = size_ratio
                                continue
                            if gap_ratio > closest_gap:
                                continue
                            if size_ratio < closest_size:
                                closest_mark = m
                                closest_gap = gap_ratio
                                closest_size = size_ratio
                                continue
                        if closest_mark is None:
                            # none are close enough to qualify
                            marks[mark] = []
                        else:
                            marks[mark] = [marks[mark][closest_mark]]
                        # mark is now either empty or contains one item
                        got_hits += len(marks[mark])
            if (got_hits / max_hits) < Finder.MIN_MARK_HITS:
                # not enough mark hits for this detection to qualify
                if self.logger is not None:
                    self.logger.log('Detection {}: Has too few detectable timing marks - require at least {:.0f} of {},'
                                    ' only found {} ({:.2f}%, threshold is {:.2f}%):'.
                                    format(detection, Finder.MIN_MARK_HITS * max_hits, max_hits, got_hits,
                                           (got_hits / max_hits) * 100, Finder.MIN_MARK_HITS * 100))
            else:
                self.filtered_timing.append(detection)
                if self.logger is not None:
                    self.logger.log('Detection {}: Found {} (of {}, {:.2f}%) timing marks:'.
                                    format(detection, got_hits, max_hits, (got_hits / max_hits) * 100))
            if self.logger is not None:
                log_locator(detection, self.detections[detection].tl, 'top-left')
                log_locator(detection, self.detections[detection].tr, 'top-right')
                log_locator(detection, self.detections[detection].bl, 'bottom-left')
                if edges is not None:
                    for edge, marks in enumerate(edges):
                        self.logger.log('  Edge {} targets:'.format(edge))
                        for mark, ticks in enumerate(marks):
                            if ticks is None:
                                continue  # no ticks here
                            for tick in ticks:
                                log_tick(detection, edge, mark, tick)
                else:
                    self.logger.log('  No blobs detected!')
                self.logger.pop()

        if self.logger is not None:
            self.logger.push(context='filter_timing')
            self.logger.log('')
            self.logger.log('Size ratio good stats: {}'.format(size_good_stats.show()))
            self.logger.log('Size ratio bad stats: {}'.format(size_bad_stats.show()))
            self.logger.log('Distance ratio good stats: {}'.format(dist_good_stats.show()))
            self.logger.log('Distance ratio bad stats: {}'.format(dist_bad_stats.show()))
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
            if found_marks[mark] is None or len(found_marks[mark]) == 0:
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
        for detection in self.filter_timing():
            top, right, bottom, left, corner = self.matched_timing[detection]
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
            self.code_grids.append([top_cols, right_cols, bottom_cols, left_cols])

        if self.logger is not None:
            self.logger.push('grids')
            self.draw_grids()
            self.logger.pop()

        return self.code_grids

    def cells(self):
        """ produce the co-ordinates of all cells inside the locators relative to the original image,
            grid addresses are in clockwise order starting at the primary corner (top-left when not rotated)
            and relative to the sub-image of the extracted target, cell addresses are row (top to bottom) then
            column (left to right), i.e. 'array' format, and relative to the original image,
            cell co-ordinates represent the centre of the cell, and a radius is the maximum circle radius that
            fits inside the cell
            """
        if self.code_cells is not None:
            # already done it
            return self.code_cells

        def intersect(l1_start, l1_end, l2_start, l2_end):
            # the cells are at the crossing points of the grid lines, we know where each line starts and end, so we can
            # use determinants to find each intersection (see https://en.wikipedia.org/wiki/Line-line_intersection)
            # intersection (Px,Py) between two non-parallel lines (x1,y1 -> x2,y2) and (x3,y3 -> x4,y4) is:
            #   Px = (x1y2 - y1x2)(x3-x4) - (x1-x2)(x3y4 - y3x4)
            #        -------------------------------------------
            #             (x1-x2)(y3-y4) - (y1-y2)(x3-x4)
            #
            #   Py = (x1y2 - y1x2)(y3-y4) - (y1-y2)(x3y4 - y3x4)
            #        -------------------------------------------
            #             (x1-x2)(y3-y4) - (y1-y2)(x3-x4)
            x1        = l1_start[0]
            y1        = l1_start[1]
            x2        = l1_end  [0]
            y2        = l1_end  [1]
            x3        = l2_start[0]
            y3        = l2_start[1]
            x4        = l2_end  [0]
            y4        = l2_end  [1]
            x1y2      = x1 * y2
            y1x2      = y1 * x2
            x3y4      = x3 * y4
            y3x4      = y3 * x4
            x3_x4     = x3 - x4
            x1_x2     = x1 - x2
            y3_y4     = y3 - y4
            y1_y2     = y1 - y2
            x1y2_y1x2 = x1y2 - y1x2
            x3y4_y3x4 = x3y4 - y3x4
            divisor   = (x1_x2 * y3_y4) - (y1_y2 * x3_x4)
            Px        = ((x1y2_y1x2 * x3_x4) - (x1_x2 * x3y4_y3x4)) / divisor
            Py        = ((x1y2_y1x2 * y3_y4) - (y1_y2 * x3y4_y3x4)) / divisor
            return Px, Py

        self.code_cells = []
        for top, right, bottom, left in self.grids():
            code_rows = []
            code_rows.append(top)  # first row is the top edge
            for row in range(1, len(left)-1):
                cells = []
                cells.append(left[row])  # first cell is left edge
                for col in range(1, len(top)-1):
                    cell = intersect(top[col], bottom[col], left[row], right[row])  # cell is at intersection
                    radius = (top[col][2] + bottom[col][2] + left[row][2] + right[row][2]) / 4  # radius is average
                    cells.append([cell[0], cell[1], radius, None])
                cells.append(right[row])  # last cell is right edge
                code_rows.append(cells)
            code_rows.append(bottom)  # last row is the bottom edge
            self.code_cells.append(code_rows)
        if self.logger is not None:
            self.logger.push('cells')
            self.draw_cells()
            self.logger.pop()
        return self.code_cells

    def images(self):
        """ get the sub-images for all the accepted detections """
        if self.good_images is None:
            self.good_images = []
            for detection in self.filter_timing():
                self.good_images.append(self.get_grayscale(detection))
        return self.good_images

def find_codes(src, params, image, detections, logger):
    """ find the valid code areas within the given detections """
    if logger is not None:
        logger.push('find_codes')
        logger.log('')
    finder = Finder(src, image, detections, logger)
    finder.cells()
    finder.circles()
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
    logger.log('Finding targets')

    # get the detections
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
    else:
        params = logger.restore(file='finder', ext='params')

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
    src = '/home/dave/precious/fellsafe/fellsafe-image/media/kilo-codes/kilo-codes-close-150-257-263-380-436-647-688-710-777.jpg'
    #proximity = const.PROXIMITY_CLOSE
    proximity = const.PROXIMITY_FAR

    logger = utils.Logger('finder.log', 'finder/{}'.format(utils.image_folder(src)))

    _test(src, proximity, blur=3, mode=const.RADIUS_MODE_MEAN, logger=logger, create_new=True)
