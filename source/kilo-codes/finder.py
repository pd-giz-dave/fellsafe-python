""" Finder for the code area
    This module takes a list of locator detections and finds the associated data cell co-ordinates
"""

import const
import utils
import codes
import locator
import cv2      # only for diagnostics


class Finder:

    # region geometry...
    CELL_SPAN     = codes.Codes.LOCATOR_SPAN      # number of cell units between locators
    TIMING_CELLS  = codes.Codes.TIMING_CELLS      # physical timing mark cell positions between locators
    TIMING_SCALE  = codes.Codes.TIMING_SCALE      # size of timing marks relative to locators
    # endregion
    # region tuning...
    MAX_SIZE_RATIO = 1.5*1.5  # max size diff squared between actual and expected size as a fraction of expected radius
    MAX_DISTANCE_RATIO = 2.0*2.0  # max dist squared between actual and expected mark as a fraction of expected radius
    MIN_MARK_HITS = 0.7  # minimum number of matched marks for a detection to qualify as a ratio of the maximum, 1==all
    # endregion
    # region field offsets in a blob...
    X_COORD = locator.Locator.X_COORD
    Y_COORD = locator.Locator.Y_COORD
    R_COORD = locator.Locator.R_COORD
    L_COORD = locator.Locator.L_COORD
    # endregion

    def __init__(self, source, image, detections, logger=None):
        self.source          = source      # originating image file name (for diagnostics)
        self.image           = image       # grayscale image the detections were detected within (and co-ordinates apply to)
        self.detections      = detections  # qualifying detections
        self.logger          = logger      # for diagnostics
        self.found_timing    = None        # candidate timing marks discovered for each detection (random ordering)
        self.expected_timing = None        # expected timing mark centres for each detection (clockwise from top-left)
        self.matched_timing  = None        # shouldbe_timing's matched with closest maybe_timing's
        self.filtered_timing = None        # list of detections that pass the qualification filter
        self.code_grids      = None        # list of detected code grids
        self.code_cells      = None        # list of cell co-ords inside the locators

    def draw(self, image, file, detection):
        folder = utils.image_folder(target=detection.tl)
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
                cv2.circle(image, (int(round(x)), int(round(y))), int(round(r)), const.ORANGE, 1)

        for i, detection in enumerate(self.detections):
            grayscale = locator.extract_box(self.image, box=(detection.box_tl, detection.box_br))
            image = cv2.merge([grayscale, grayscale, grayscale])
            image = locator.draw_rectangle(image,
                                           detection.tl, detection.tr, detection.br, detection.bl,
                                           origin=detection.box_tl)
            expected_timing = self.expect_timing()[i]
            for edge in expected_timing:
                draw_marks(image, edge)
            self.draw(image, 'locators', detection)

    def draw_grids(self):
        """ draw the detected code grids for diagnostic purposes """
        if self.logger is None:
            return

        def draw_grid(image, here, there):
            """ draw lines of same cells from here to there on the given image """
            for cell in range(len(here)):
                src   = here [cell]
                dst   = there[cell]
                src_x = int(round(src[Finder.X_COORD]))
                src_y = int(round(src[Finder.Y_COORD]))
                dst_x = int(round(dst[Finder.X_COORD]))
                dst_y = int(round(dst[Finder.Y_COORD]))
                cv2.line(image, (src_x, src_y), (dst_x, dst_y), const.GREEN, 1)

        for detection, (top, right, bottom, left) in enumerate(self.grids()):
            detection = self.detections[detection]
            grayscale = locator.extract_box(self.image, box=(detection.box_tl, detection.box_br))
            image = cv2.merge([grayscale, grayscale, grayscale])
            draw_grid(image, top, bottom)
            draw_grid(image, left,right)
            tl_x, tl_y, tl_r, _ = top[0]
            cv2.circle(image, (int(round(tl_x)), int(round(tl_y))), int(round(tl_r)), const.RED, 1)  # mark primary corner
            self.draw(image, 'grid', detection)

    def draw_cells(self):
        """ draw the detected code cells for diagnostic purposes """
        if self.logger is None:
            return

        for detection, rows in enumerate(self.code_cells):
            detection = self.detections[detection]
            origin_x, origin_y = detection.box_tl
            grayscale = locator.extract_box(self.image, box=(detection.box_tl, detection.box_br))
            image = cv2.merge([grayscale, grayscale, grayscale])
            for row, cells in enumerate(rows):
                for col, (x, y, r, _) in enumerate(cells):
                    if row == 0 and col == 0:
                        # highlight reference point
                        colour = const.RED
                    else:
                        colour = const.GREEN
                    cv2.circle(image, (int(round(x-origin_x)), int(round(y-origin_y))), int(round(r)), colour, 1)
            self.draw(image, 'cells', detection)

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
        params.integration_width = 2
        params.black_threshold = 2
        params.direct_neighbours = False
        params.inverted = True
        params.mode = const.RADIUS_MODE_INSIDE
        params.blur_kernel_size = 0
        params.min_area = 1
        params.min_size = 1
        #max_area
        #max_size
        params.blur_kernel_size = 0
        params.max_squareness = 0.5
        params.max_wavyness = 0.4
        params.max_offsetness = 0.25
        params.max_whiteness = 0.6
        for i, detection in enumerate(self.detections):
            # do the timing mark detection
            params.box = (detection.box_tl, detection.box_br)
            if self.logger is None:
                self.found_timing[i] = locator.get_targets(self.image, params)
            else:
                folder = utils.image_folder(target=detection.tl)
                self.logger.push(context='find_timing/{}'.format(folder), folder=folder)
                self.logger.log('\nLooking for timing marks...')
                self.found_timing[i] = locator.get_targets(self.image, params, logger=self.logger)
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
    def make_detection(detection, origin=(0, 0), scale=1.0) -> []:
        """ remove the origin and scale the radius of a detection,
            used to map image co-ordinates to detection box co-ordinates
            """
        x, y, r, l = detection
        r *= scale
        # NB: create a new object, do not update existing one
        return [x - origin[Finder.X_COORD], y - origin[Finder.Y_COORD], r, l]

    @staticmethod
    def make_steps(origin, start, end, steps: int, radius_scale=TIMING_SCALE) -> [()]:
        """ make a series of x,y,r co-ordinate steps, in units of cells, from start to end """
        start   = Finder.make_detection(start, origin, radius_scale)
        end     = Finder.make_detection(end, origin, radius_scale)
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
        for detection in self.detections:
            origin = detection.box_tl

            top_columns    = Finder.make_steps(origin, detection.tl, detection.tr, Finder.CELL_SPAN)
            right_rows     = Finder.make_steps(origin, detection.tr, detection.br, Finder.CELL_SPAN)
            bottom_columns = Finder.make_steps(origin, detection.bl, detection.br, Finder.CELL_SPAN)
            left_rows      = Finder.make_steps(origin, detection.tl, detection.bl, Finder.CELL_SPAN)

            # make the 'missing' corner 'minor' locator mark (make as a list to be type consistent with the edges)
            corners = [Finder.make_detection(detection.br, origin, Finder.TIMING_SCALE)]

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

        self.matched_timing = [None for _ in range(len(self.find_timing()))]  # create slot per detection
        for detection, (found_timing, _) in enumerate(self.find_timing()):
            for candidate, (fx, fy, fr, _) in enumerate(found_timing):
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
                        self.matched_timing[detection][edge] = [None for _ in
                                                                range(len(marks))]  # create mark slots
                    for mark, (x, y, r, _) in enumerate(marks):
                        if len(marks) > 1:
                            # its an edge - must filter for physical ticks
                            if mark not in Finder.TIMING_CELLS:
                                # its not physical - ignore it
                                continue
                        if self.matched_timing[detection][edge][mark] is None:
                            self.matched_timing[detection][edge][mark] = []  # create empty mark list
                        gap = distance((x, y), (fx, fy))
                        size = r - fr
                        size *= size   # closer to 0 means closer radius match (used when equi-distant to 2 expected)
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
                        elif size < best_size:
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

        def log_tick(detection, edge, mark, candidate):
            actual   = self.found_timing[detection][0][candidate]
            expected = self.expected_timing[detection][edge][mark]
            self.logger.log('    {}: {:.2f}x, {:.2f}y, {:.2f}r'
                            ' (expected {:.2f}x, {:.2f}y, {:.2f}r)'.
                            format(candidate, actual[0], actual[1], actual[2],
                                   expected[0], expected[1], expected[2]))

        self.filtered_timing = []
        max_hits = self.expected_marks()  # maximum possible marks per detection
        for detection, edges in enumerate(self.match_timing()):
            if self.logger is not None:
                folder = utils.image_folder(target=self.detections[detection].tl)
                self.logger.push(context='filter_timing/{}'.format(folder), folder=folder)
                self.logger.log('')
            got_hits = 0  # how many marks actually qualify for this detection
            for edge, marks in enumerate(edges):
                for mark in range(len(marks)):
                    if marks[mark] is None:
                        continue  # no marks here
                    # drop all but closest that is within acceptable limits
                    closest_mark = None
                    closest_gap = None
                    closest_size = None
                    for m, (candidate, distance, size, r) in enumerate(marks[mark]):
                        size_ratio = size / r
                        gap_ratio = distance / r
                        if size_ratio > Finder.MAX_SIZE_RATIO:
                            # size too divergent from expected - ignore it
                            if self.logger is not None:
                                self.logger.log('Detection {}, edge {}, mark {}: Size ratio ({:.2f}) too high,'
                                                ' limit is {}'.format(detection, edge, mark, size_ratio,
                                                                      Finder.MAX_SIZE_RATIO))
                                log_tick(detection, edge, mark, candidate)
                            continue
                        if gap_ratio > Finder.MAX_DISTANCE_RATIO:
                            # distance too far from expected - ignore it
                            if self.logger is not None:
                                self.logger.log('Detection {}, edge {}, mark {}: Gap ratio ({:.2f}) too high,'
                                                ' limit is {}'.format(detection, edge, mark, gap_ratio,
                                                                      Finder.MAX_DISTANCE_RATIO))
                                log_tick(detection, edge, mark, candidate)
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
                    self.logger.log('Detection {}: Has too few detectable timing marks - require {}, '
                                    'only found {} ({:.2f}%, threshold is {:.2f}%):'.
                                    format(detection, max_hits, got_hits,
                                           (got_hits / max_hits) * 100, Finder.MIN_MARK_HITS * 100))
            else:
                self.filtered_timing.append(detection)
                if self.logger is not None:
                    self.logger.log('Detection {}: Found {} (of {}, {:.2f}%) timing marks:'.
                                    format(detection, got_hits, max_hits, (got_hits / max_hits) * 100))
            if self.logger is not None:
                for edge, marks in enumerate(edges):
                    self.logger.log('  Edge {} targets:'.format(edge))
                    for mark, ticks in enumerate(marks):
                        if ticks is None:
                            continue  # no ticks here
                        for (candidate, distance, size, r) in ticks:
                            log_tick(detection, edge, mark, candidate)
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
            found = self.found_timing[detection][0][found_mark_number]
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
                right_ref [-1] = Finder.make_detection(br, (0,0), 1.0)
                bottom_ref[-1] = Finder.make_detection(br, (0,0), 1.0)
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

        self.code_cells = []
        grids = self.grids()
        for detection, (top, right, bottom, left) in enumerate(grids):
            code_cells = []
            origin = self.detections[detection].box_tl  # start of our box
            offset = [-origin[Finder.X_COORD], origin[-Finder.X_COORD]]  # offset to map back to orig image
            steps = len(top) - 1
            for col in range(len(top)):
                start = left[col]
                end = right[col]
                code_cells.append(Finder.make_steps(offset, start, end, steps=steps, radius_scale=1.0))
            self.code_cells.append(code_cells)
        if self.logger is not None:
            self.logger.push('cells')
            self.draw_cells()
            self.logger.pop()
        return self.code_cells

def find_codes(src, image, detections, logger):
    """ find the valid code areas within the given detections """
    if logger is not None:
        logger.push('find_codes')
        logger.log('')
    found = Finder(src, image, detections, logger)
    result = found.cells()
    if logger is not None:
        logger.pop()
    return result


def _test(src, proximity, blur=3, logger=None, create_new=True):
    """ ************** TEST **************** """
    import pickle

    if logger.depth() > 1:
        logger.push('finder/_test')
    else:
        logger.push('_test')
    logger.log("\nFinding targets")

    # get the detections
    if create_new:
        # this is very slow
        params = locator._test(src, proximity, logger, blur=blur, create_new=create_new)
        located = params.locator
        image = params.source
        image_dump = open('locator.image', 'wb')
        pickle.dump(image, image_dump)
        image_dump.close()
        detections = located.detections()
        detections_dump = open('locator.detections','wb')
        pickle.dump(detections, detections_dump)
        detections_dump.close()
    else:
        image_dump = open('locator.image', 'rb')
        image = pickle.load(image_dump)
        image_dump.close()
        detections_dump = open('locator.detections', 'rb')
        detections = pickle.load(detections_dump)
        detections_dump.close()

    # process the detections
    found = find_codes(src, image, detections, logger)
    logger.pop()
    return found, image


if __name__ == "__main__":
    """ test harness """

    #src = "/home/dave/precious/fellsafe/fellsafe-image/media/kilo-codes/kilo-codes-distant.jpg"
    src = "/home/dave/precious/fellsafe/fellsafe-image/source/kilo-codes/test-alt-bits.png"
    proximity = const.PROXIMITY_CLOSE
    #proximity = const.PROXIMITY_FAR

    logger = utils.Logger('finder.log', 'finder/{}'.format(utils.image_folder(src)))

    _test(src, proximity, blur=3, logger=logger, create_new=False)
