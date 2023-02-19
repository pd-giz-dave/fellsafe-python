""" Finder code area
    This module takes a list of locator detections and finds the timing marks associated with the code area
"""

import const
import utils
import codes
import locator
import os       # only for diagnostics
import cv2      # only for diagnostics

class Finder:

    # region geometry...
    SPACING = codes.Codes.LOCATOR_SPACING  # number of 'r' units between locators
    COLUMNS = codes.Codes.DATA_SHAPE[0]    # number of data columns between locators
    ROWS    = codes.Codes.DATA_SHAPE[1]    # number of data rows between locators
    # endregion
    # region tuning...
    MAX_SIZE_RATIO = 1.5  # max size difference between actual and expected size as a fraction of expected radius
    MAX_DISTANCE_RATIO = 1.5  # max distance between actual and expected mark as a fraction of expected radius
    MIN_MARK_HITS = 0.7  # minimum number of matched marks for a detection to qualify as a ratio of the maximum
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

    def get_pixel(self, x: float, y: float) -> int:
        """ get the interpolated pixel value at x,y,
            x,y are fractional so the pixel value returned is a mixture of the 4 pixels around x,y,
            the mixture is based on the ratio of the neighbours to include, the ratio of all 4 is 1,
            code based on:
                void interpolateColorPixel(double x, double y) {
                    int xL, yL;
                    xL = (int) Math.floor(x);
                    yL = (int) Math.floor(y);
                    xLyL = ipInitial.getPixel(xL, yL, xLyL);
                    xLyH = ipInitial.getPixel(xL, yL + 1, xLyH);
                    xHyL = ipInitial.getPixel(xL + 1, yL, xHyL);
                    xHyH = ipInitial.getPixel(xL + 1, yL + 1, xHyH);
                    for (int rr = 0; rr < 3; rr++) {
                        double newValue = (xL + 1 - x) * (yL + 1 - y) * xLyL[rr];
                        newValue += (x - xL) * (yL + 1 - y) * xHyL[rr];
                        newValue += (xL + 1 - x) * (y - yL) * xLyH[rr];
                        newValue += (x - xL) * (y - yL) * xHyH[rr];
                        rgbArray[rr] = (int) newValue;
                    }
                }
            from here: https://imagej.nih.gov/ij/plugins/download/Polar_Transformer.java
            explanation:
            x,y represent the top-left of a 1x1 pixel
            if x or y are not whole numbers the 1x1 pixel area overlaps its neighbours,
            the returned pixel value is the sum of the overlap fractions of its neighbour
            pixel squares, P is the fractional pixel address in its pixel, 1, 2 and 3 are
            its neighbours, dotted area is contribution from neighbours:
                +------+------+
                |  P   |   1  |
                |  ....|....  |  Ax = 1 - (Px - int(Px) = 1 - Px + int(Px) = (int(Px) + 1) - Px
                |  . A | B .  |  Ay = 1 - (Py - int(Py) = 1 - Py + int(Py) = (int(Py) + 1) - Py
                +------+------+  et al for B, C, D
                |  . D | C .  |
                |  ....|....  |
                |  3   |   2  |
                +----- +------+
            """
        cX: float = x
        cY: float = y
        xL: int = int(cX)
        yL: int = int(cY)
        xH: int = xL + 1
        yH: int = yL + 1
        pixel_xLyL = self.image[yL][xL]
        pixel_xLyH = self.image[yH][xL]
        pixel_xHyL = self.image[yL][xH]
        pixel_xHyH = self.image[yH][xH]
        if pixel_xLyL is None:
            pixel_xLyL = const.MIN_LUMINANCE
        if pixel_xLyH is None:
            pixel_xLyH = const.MIN_LUMINANCE
        if pixel_xHyL is None:
            pixel_xHyL = const.MIN_LUMINANCE
        if pixel_xHyH is None:
            pixel_xHyH = const.MIN_LUMINANCE
        ratio_xLyL = (xH - cX) * (yH - cY)
        ratio_xHyL = (cX - xL) * (yH - cY)
        ratio_xLyH = (xH - cX) * (cY - yL)
        ratio_xHyH = (cX - xL) * (cY - yL)
        part_xLyL = pixel_xLyL * ratio_xLyL
        part_xHyL = pixel_xHyL * ratio_xHyL
        part_xLyH = pixel_xLyH * ratio_xLyH
        part_xHyH = pixel_xHyH * ratio_xHyH
        pixel = int(round(part_xLyL + part_xHyL + part_xLyH + part_xHyH))
        return pixel

    def draw_detections(self):
        """ draw the detection rectangles and expected timing marks for diagnostic purposes """

        def draw_marks(image, marks):
            # draw circle with a radius of the expected mark in orange
            for mark, (x, y, r, _) in enumerate(marks):
                r = max(r-2, 1)  # show inner radius (i.e. drawn pixels are *inside* the circle)
                cv2.circle(image, (int(round(x)), int(round(y))), int(round(r)), const.ORANGE, 1)

        for i, detection in enumerate(self.detections):
            grayscale = locator.extract_box(self.image, box=(detection.box_tl, detection.box_br))
            draw = cv2.merge([grayscale, grayscale, grayscale])
            draw = locator.draw_rectangle(draw,
                                          detection.tl, detection.tr, detection.br, detection.bl,
                                          origin=detection.box_tl)
            expected_timing = self.expect_timing()[i]
            for edge in expected_timing:
                draw_marks(draw, edge)
            folder = image_folder(target=detection.tl)
            self.logger.push(folder=folder)
            self.logger.draw(draw, file='locators')
            self.logger.pop()

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
                folder = image_folder(target=detection.tl)
                self.logger.push(context='find_timing/{}'.format(folder), folder=folder)
                self.logger.log('\nLooking for timing marks...')
                self.found_timing[i] = locator.get_targets(self.image, params, logger=self.logger)
                self.logger.pop()
        return self.found_timing

    def expect_timing(self):
        """ make a list of expected timing blob centres for each detection,
            in clockwise order starting at 'top-left' locator (the 'primary' locator) -
            top-columns, right-rows, bottom-columns, left-rows - with a final single centre at bottom-right
            (the 'missing' corner locator - see visualisation below),
            the constructed list co-ordinates are relative to the box of the detection
            """
        if self.expected_timing is not None:
            # already done it
            return self.expected_timing

        # region visualisation...
        # in each detection we have x,y co-ords for top-left, top-right, bottom-right and bottom-left corners
        # these may be in any rotation, we expect two timing marks equally spaced between these corners
        # the timing marks are aligned with the 'outer' halves of the locators (locators have diameter 2R, timing 1R)
        # e.g:
        #   +--XXXXXXXX                    XXXXXXXX--+
        #   |  ---tl--- c1  c2  c3  c4  c5 ---tr---  |
        #   X  XXXX####....****....****....####XXXX  X
        #   X  XXXX####....****....****....####XXXX  X   X..X is detected by locator
        #   X  ####XXXX....................XXXX####  X   E..E is estimated by locator (note expected mark within it)
        #   X  ####XXXX....................XXXX####  X   *..* is expected timing marks
        #  r1  ....................................  r1  #..# is the reference part of corner detections
        #      ....................................      c1..c5 are data column positions
        #  r2  ****............................****  r2  r1..r5 are data row positions
        #      ****............................****
        #  r3  ....................................  r3
        #      ....................................
        #  r4  ****............................****  r4
        #      ****............................****
        #  r5  ....................................  r5
        #      ....................................
        #   X  ####XXXX....................****####  E
        #   X  ####XXXX....................****####  E
        #   X  XXXX####....****....****....####EEEE  E
        #   X  XXXX####....****....****....####EEEE  E
        #   |  ---bl--- c1  c2  c3  c4  c5 ---br---  |
        #   +--XXXXXXXX                    EEEEEEEE--+
        # endregion

        def get_step(a, b):
            """ from two locators, determine the x,y step size representing one code cell between them ('r') """
            ax, ay, _, _ = a
            bx, by, _, _ = b
            span_x = bx - ax
            span_y = by - ay
            step_x = span_x / Finder.SPACING
            step_y = span_y / Finder.SPACING
            return step_x, step_y

        def get_offset(a_step, b_step, a_scale, b_scale):
            """ get the offset from two orthogonal steps to move a locator centre to one of its quadrant centres,
                the a/b_scale are scaling factors of +/- 0.5 (to move half an 'r' in a positive or negative direction)
                """
            offset_a = (a_step[0] * a_scale, a_step[1] * a_scale)
            offset_b = (b_step[0] * b_scale, b_step[1] * b_scale)
            return offset_a[0] + offset_b[0], offset_a[1] + offset_b[1]

        def apply_offset(origin, detection, offset, scale):
            """ apply an x,y offset and a radius scale to a detection then remove the origin """
            x, y, r, l = detection
            x += offset[0]
            y += offset[1]
            r *= scale
            return [x-origin[0], y-origin[1], r, l]

        def make_steps(origin, start, end, start_offset, end_offset, steps: int) -> [()]:
            """ make a series of x,y,r co-ordinate steps from start+offset to end+offset """
            start_x, start_y, start_r, _ = apply_offset(origin, start, start_offset, 0.5)  # timing marks are half the
            end_x,   end_y,   end_r,   _ = apply_offset(origin, end,   end_offset,   0.5)  # radius of the locators
            span_x  = end_x - start_x
            span_y  = end_y - start_y
            step_x  = span_x / (steps + 1)  # +1 for the two 0.5R's of the locators
            step_y  = span_y / (steps + 1)
            radius  = (start_r + end_r) / 2  # use average of locators as the radius
            centres = []
            for step in range(2, steps + 1, 2):  # only want every other step
                centres.append([start_x + (step_x * step), start_y + (step_y * step), radius, None])
            return centres

        self.expected_timing = []
        for detection in self.detections:
            origin = detection.box_tl

            tl2tr_step = get_step(detection.tl, detection.tr)
            tr2br_step = get_step(detection.tr, detection.br)
            bl2br_step = get_step(detection.bl, detection.br)
            tl2bl_step = get_step(detection.tl, detection.bl)

            tlc_offset = get_offset(tl2tr_step, tl2bl_step, +0.5, -0.5)
            trc_offset = get_offset(tl2tr_step, tr2br_step, -0.5, -0.5)
            brc_offset = get_offset(bl2br_step, tr2br_step, -0.5, +0.5)
            blc_offset = get_offset(tl2bl_step, bl2br_step, +0.5, +0.5)

            tlr_offset = get_offset(tl2tr_step, tl2bl_step, -0.5, +0.5)
            trr_offset = get_offset(tl2tr_step, tr2br_step, +0.5, +0.5)
            brr_offset = get_offset(bl2br_step, tr2br_step, +0.5, -0.5)
            blr_offset = get_offset(bl2br_step, tl2bl_step, -0.5, -0.5)

            top_columns    = make_steps(origin, detection.tl, detection.tr, tlc_offset, trc_offset, Finder.COLUMNS)
            right_rows     = make_steps(origin, detection.tr, detection.br, trr_offset, brr_offset, Finder.ROWS)
            bottom_columns = make_steps(origin, detection.bl, detection.br, blc_offset, brc_offset, Finder.COLUMNS)
            left_rows      = make_steps(origin, detection.tl, detection.bl, tlr_offset, blr_offset, Finder.ROWS)

            # make the 'missing' corner locator mark (make as a list to be type consistent with the edges)
            corners = [apply_offset(origin, detection.br, trc_offset, 0.5)]

            expected_timing = [top_columns, right_rows, bottom_columns, left_rows, corners]
            self.expected_timing.append(expected_timing)

        return self.expected_timing

    def expected_marks(self):
        """ return the expected number of timing marks for a detection """
        expected_timing = self.expect_timing()
        marks = 0
        for detection in expected_timing:
            for edge in detection:
                marks += len(edge)
            break  # they are all the same, so just do the first one
        return marks

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

        self.matched_timing = [None for _ in range(len(self.find_timing()))]  # create detection slots
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
        self.filtered_timing = []
        max_hits = self.expected_marks()  # maximum possible marks per detection
        for detection, edges in enumerate(self.match_timing()):
            if self.logger is not None:
                folder = image_folder(target=self.detections[detection].tl)
                self.logger.push(context='filter_timing/{}'.format(folder), folder=folder)
                self.logger.log('')
            got_hits = 0  # how many marks actually qualify for this detection
            for edge, marks in enumerate(edges):
                for mark in range(len(marks)):
                    # drop all but closest that is within acceptable limits
                    closest_mark = None
                    closest_gap = None
                    closest_size = None
                    for m, (candidate, distance, size, r) in enumerate(marks[mark]):
                        size_ratio = size / r
                        gap_ratio = distance / r
                        if size_ratio > Finder.MAX_SIZE_RATIO:
                            # size too divergent from expected - ignore it
                            continue
                        if gap_ratio > Finder.MAX_DISTANCE_RATIO:
                            # distance too far from expected - ignore it
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
                    self.logger.log('Dropping detection for having too few detectable timing marks - require '
                                    '{}, only found {} (threshold is {:.2f}%)'.format(max_hits, got_hits,
                                                                                      Finder.MIN_MARK_HITS * 100))
            else:
                self.filtered_timing.append(detection)
                if self.logger is not None:
                    self.logger.log('Detected {} (of {}) timing marks:'.format(got_hits, max_hits))
                    for edge, marks in enumerate(edges):
                        self.logger.log('  Edge {} targets:'.format(edge))
                        for mark, ticks in enumerate(marks):
                            for (candidate, distance, size, r) in ticks:
                                actual = self.found_timing[detection][0][candidate]
                                expected = self.expected_timing[detection][edge][mark]
                                self.logger.log('    {}: {:.2f}x, {:.2f}y, {:.2f}r'
                                                ' (expected {:.2f}x, {:.2f}y, {:.2f}r)'.
                                                format(candidate, actual[0], actual[1], actual[2],
                                                       expected[0], expected[1], expected[2]))
            if self.logger is not None:
                self.logger.pop()
        return self.filtered_timing


def _unload(image, source=None, file='', target=(0,0), logger=None):
    """ unload the given image with a name that indicates its source and context,
        file is the file base name to save the image as,
        target identifies the x,y of the primary locator the image represents,
        target of 0,0 means no x/y identification for the image name,
        """

    folder = image_folder(source=source, target=target)
    logger.draw(image, folder=folder, file=file)

def image_folder(source=None, target=(0,0)):
    """ build folder name for diagnostic images for the given target """
    if target[0] > 0 and target[1] > 0:
        # use a sub-folder for this image
        folder = '{:.0f}x{:.0f}y'.format(target[0], target[1])
    else:
        folder = ''
    if source is not None:
        # construct parent folder to save images in for this source
        pathname, _ = os.path.splitext(source)
        _, basename = os.path.split(pathname)
        folder = '{}{}'.format(basename, folder)
    return folder

def _test(src, proximity, blur=3, logger=None, create_new_detections=True):
    """ ************** TEST **************** """
    import pickle

    logger.log("\nFinding targets")

    # get the detections
    if create_new_detections:
        # this is very slow
        params = locator._test(src, proximity, logger, blur=blur, create_new_blobs=create_new_detections)
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
    extractor = Finder(src, image, detections, logger)
    extractor.draw_detections()
    result = extractor.filter_timing()

if __name__ == "__main__":
    """ test harness """

    src = "/home/dave/precious/fellsafe/fellsafe-image/media/square-codes/square-codes-far.jpg"
    #src = "/home/dave/precious/fellsafe/fellsafe-image/source/kilo-codes/test-alt-bits.png"
    #proximity = const.PROXIMITY_CLOSE
    proximity = const.PROXIMITY_FAR
    create_new = True

    logger = utils.Logger('finder.log', 'finder/{}'.format(image_folder(src)))
    logger.log('_test')

    _test(src, proximity, blur=3, logger=logger, create_new_detections=create_new)
