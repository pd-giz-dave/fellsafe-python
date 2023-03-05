""" Find locator blobs
    This module is responsible for finding the locator blobs in an image.
    The locator blobs are positioned on 3 corners of a square (similar to QR codes).
    There may be other (smaller) blobs within the square that must be ignored.
    The square may be rotated through any angle, e.g. upside down would be a 180 degree rotation.
    The module is given a list of blobs, each of which has a centre and a radius.
    The geometry sought is:
    (A)  x                   x  (B)
       x   x               x   x
     x<- 2r->x<--- d --->x<-2r ->x
       x   x               x   x
         x                   x
         |                  /
         |                /
                        /
         d         sqrt(2d^2)
                    /
         |        /
         |      /
         x    /
       x   x
     x<- 2r->x
       x   x
         x  (C)
    where 'r' is the blob radius (+/- some margin)
    'd' is the distance between two of the three blob centres and must be in the region of LOCATOR_SPACING*r,
    'sqrt(2d^2)' is the distance to the other blob centre of a group of three,
    blobs smaller than some factor (<1) of 'r' between the three locator blobs are ignored
"""

import math
import canvas    # purely for diagnostic aids

import codes     # for the geometry constants
import const     # for the proximity constants
import utils     # for the logger
import contours  # for providing a test source of blobs

class Params(contours.Targets):
    def __init__(self):
        self.locator = None

get_targets = contours.get_targets  # only here to hide contours from upstream
extract_box = contours.extract_box  # ..

class Detection:
    """ a 'detection' is the four points of a rectangle formed from the three detected corners, and its
        estimated fourth corner, in clockwise order from the primary corner along with its enclosing box,
        the enclosing box encloses the rectangle and a margin sufficient to include the white space around
        the code area (equivalent to the radius of the biggest blob)
        """

    MARGIN_SCALE = 1.7  # scale factor from the maximum blob radius to the enclosing box edges

    @staticmethod
    def constrain(value, min_value, max_value):
        """ constrain the given value (as an int) to be between the given min/max (int) limits """
        value = int(min(max(int(round(value)), min_value), max_value))
        return value

    def __init__(self, rectangle, image_width, image_height):
        """ rectangle is as created by Locator.rectangles() - 4 blob of x,y,radius,label,
            image width/height is used to constrain the enclosing box for detections close to the image edges
            the rectangle co-ordinates may be fractional, the box co-ordinates are not
            """
        self.tl = rectangle[0]
        self.tr = rectangle[1]
        self.br = rectangle[2]
        self.bl = rectangle[3]
        self.squareness = rectangle[5]  # maybe used as a discriminator downstream
        margin = max(rectangle[0][2], rectangle[1][2], rectangle[2][2], rectangle[3][2]) * Detection.MARGIN_SCALE
        min_x = Detection.constrain(min(self.tl[0], self.tr[0], self.br[0], self.bl[0]) - margin, 0, image_width-1)
        max_x = Detection.constrain(max(self.tl[0], self.tr[0], self.br[0], self.bl[0]) + margin, 0, image_width-1)
        min_y = Detection.constrain(min(self.tl[1], self.tr[1], self.br[1], self.bl[1]) - margin, 0, image_height-1)
        max_y = Detection.constrain(max(self.tl[1], self.tr[1], self.br[1], self.bl[1]) + margin, 0, image_height-1)
        self.box_tl = (min_x, min_y)
        self.box_br = (max_x, max_y)


class Locator:

    # region geometry contraints...
    LOCATOR_DISTANCE    = codes.Codes.LOCATOR_SPACING  # distance between locators in units of locator radius
    LOCATOR_DISTANCE   *= LOCATOR_DISTANCE             # ..squared to be units compatible with distance squared
    MIN_NEIGHBOURS      = codes.Codes.LOCATORS_PER_CODE - 1  # expected neighbours per 'primary' locator
    MIN_DISTANCE_RATIO  = 0.5  # min distance ratio (actual/expected) between locators to be considered neighbours
    MIN_DISTANCE_RATIO *= MIN_DISTANCE_RATIO  # ..squared to be units compatible with distance squared (1==best)
    MIN_RADIUS_RATIO    = 0.6  # min size ratio between blobs for them to be considered of similar size (1==best)
    MIN_LENGTH_RATIO    = 0.7  # min length ratio of two triangle sides to be considered equal (1==best)
    MIN_DIAGONAL_RATIO  = 0.8  # min diagonal ratio between expected and actual length to be considered equal (1==best)
    NEAR_CORNER_SCALE   = 1.0  # scaling factor to consider two corners as overlapping
    # endregion

    # blob tuple indexes
    X_COORD = 0
    Y_COORD = 1
    R_COORD = 2
    L_COORD = 3

    def __init__(self, blobs, logger=None):
        self.blobs       = blobs
        self.logger      = logger
        self._neighbours = None
        self._corners    = None
        self._triangles  = None
        self._rectangles = None
        self._detections = None
        self._width      = None
        self._height     = None
        self._dropped    = None
        self._neighbour_size_stats = None
        self._neighbour_dist_stats = None
        self._corner_side_stats = None
        self._corner_diag_stats = None

    @staticmethod
    def is_same(a, b, limit=0.0) -> float:
        """ determine if the two given numbers are considered equal within the given limit,
            if they are, return their ratio, else return None
            """
        ratio = min(a, b) / max(a, b)
        if ratio < limit:
            return None
        else:
            return ratio

    @staticmethod
    def distance(here, there) -> float:
        """ calculate the distance between the two given blobs,
            returns the distance squared,
            we use squared result so we do not need to do a square root (which is slow)
            """
        distance_x  = here[Locator.X_COORD] - there[Locator.X_COORD]
        distance_x *= distance_x
        distance_y  = here[Locator.Y_COORD] - there[Locator.Y_COORD]
        distance_y *= distance_y
        distance    = distance_x + distance_y
        return distance

    @staticmethod
    def is_near(a, b) -> bool:
        """ determine if corner a is near corner b, 'near' is when their areas overlap,
            i.e. the distance between them is less than the sum of their radii times some scaling constant
            """
        gap = Locator.distance(a, b)
        limit = (a[Locator.R_COORD] + b[Locator.R_COORD]) * Locator.NEAR_CORNER_SCALE
        limit *= limit  # gap is distance squared, so the limit must be too
        return gap < limit

    @staticmethod
    def quality(rectangle):
        """ return a measure of the quality of the given rectangle,
            a high quality rectangle is more square and has more evenly sized locators
            """
        tl, tr, br, bl, _, squareness = rectangle
        tl_r = tl[Locator.R_COORD]
        tr_r = tr[Locator.R_COORD]
        # don't look at br radius as that is an estimate (so no quality information in it)
        bl_r = bl[Locator.R_COORD]
        max_r = max(tl_r, tr_r, bl_r)
        tl_ratio = tl_r / max_r  # range 0..1
        tr_ratio = tr_r / max_r
        bl_ratio = bl_r / max_r
        ratio = tl_ratio * tr_ratio * bl_ratio  # range 0..1, 0==bad, 1==good
        return ratio * squareness               # range 0..1, 0==bad, 1==good

    def neighbours(self):
        """ return a list of similar size near neighbours for each blob """
        if self._neighbours is not None:
            # already done it
            return self._neighbours

        # this is a crude O(N^2) algorithm, I'm sure there are better ways!, eg. a k-d tree
        self._neighbours = []
        self._neighbour_size_stats = [utils.Stats(20), utils.Stats(20)]  # good and bad
        self._neighbour_dist_stats = [utils.Stats(20), utils.Stats(20)]  # ..
        for blob, here in enumerate(self.blobs):
            neighbour = []
            for candidate, there in enumerate(self.blobs):
                if candidate == blob:
                    # ignore self
                    continue
                # check size first (as its cheap)
                here_r  = here [Locator.R_COORD]
                there_r = there[Locator.R_COORD]
                if Locator.is_same(here_r, there_r, Locator.MIN_RADIUS_RATIO) is None:
                    # sizes too dis-similar
                    if self.logger is not None:
                        # log bad ones
                        ratio = Locator.is_same(here_r, there_r)
                        # self.logger.log('Neighbours (bad): Here blob {} ({:.2f}r), there blob {} ({:.2f}r):'
                        #                 ' sizes too dis-similar (ratio is {:.2f}, limit is {:.2f})'.
                        #                 format(blob, here_r, candidate, there_r, ratio, Locator.MIN_RADIUS_RATIO))
                        self._neighbour_size_stats[1].count(ratio)
                    continue
                if self.logger is not None:
                    # log good ones
                    ratio = Locator.is_same(here_r, there_r)
                    # self.logger.log('Neighbours (good): Here blob {} ({:.2f}r), there blob {} ({:.2f}r):'
                    #                 ' sizes match (ratio is {:.2f}, limit is {:.2f})'.
                    #                 format(blob, here_r, candidate, there_r, ratio, Locator.MIN_RADIUS_RATIO))
                    self._neighbour_size_stats[0].count(ratio)
                # size OK, now check distance
                distance   = Locator.distance(here, there)
                one_unit   = (here_r + there_r) / 2  # one locator distance unit
                one_unit  *= one_unit                # square it to be compatible with distance
                separation = distance / one_unit     # this locator pair separation in distance units squared
                if Locator.is_same(separation, Locator.LOCATOR_DISTANCE, Locator.MIN_DISTANCE_RATIO) is None:
                    # distance out of range
                    if self.logger is not None:
                        # log bad ones
                        ratio = Locator.is_same(separation, Locator.LOCATOR_DISTANCE)
                        # self.logger.log('Neighbours (bad): Here blob {} ({:.2f}r), there blob {} ({:.2f}r):'
                        #                 ' distance ({:.2f}) too far from expected (ratio is {:.2f}, limit is {:.2f})'.
                        #                 format(blob, here_r, candidate, there_r, math.sqrt(separation),
                        #                        math.sqrt(ratio), math.sqrt(Locator.MIN_DISTANCE_RATIO)))
                        self._neighbour_dist_stats[1].count(ratio)
                    continue
                # log good ones
                ratio = Locator.is_same(separation, Locator.LOCATOR_DISTANCE)
                # self.logger.log('Neighbours (good): Here blob {} ({:.2f}r), there blob {} ({:.2f}r):'
                #                 ' distance ({:.2f}) close to expected (ratio is {:.2f}, limit is {:.2f})'.
                #                 format(blob, here_r, candidate, there_r, math.sqrt(separation),
                #                        math.sqrt(ratio), math.sqrt(Locator.MIN_DISTANCE_RATIO)))
                self._neighbour_dist_stats[0].count(ratio)
                # size and distance OK
                neighbour.append((candidate, distance))
            if len(neighbour) >= Locator.MIN_NEIGHBOURS:
                self._neighbours.append((blob, neighbour))
        if self.logger is not None:
            self.logger.log('')
            self.logger.log('Neighbour bad size ratio stats')
            self.logger.log('  ' + self._neighbour_size_stats[1].show())
            self.logger.log('Neighbour good size ratio stats')
            self.logger.log('  ' + self._neighbour_size_stats[0].show())
            self.logger.log('')
            self.logger.log('Neighbour bad distance ratio stats')
            self.logger.log('  ' + self._neighbour_dist_stats[1].show())
            self.logger.log('Neighbour good distance ratio stats (across {} neighbours)'.format(len(self._neighbours)))
            self.logger.log('  ' + self._neighbour_dist_stats[0].show())
        return self._neighbours

    def corners(self):
        """ return a list of triplets that meet our corner requirements,
            that is a triangle A,B,C such that A->B == A->C == d and B->C == sqrt(2d^2),
            i.e. a right-angle triangle with 2 equal sides
            """
        if self._corners is not None:
            # already done it
            return self._corners

        self._corners = []
        self._corner_side_stats = [utils.Stats(20), utils.Stats(20)]  # good, bad
        self._corner_diag_stats = [utils.Stats(20), utils.Stats(20)]  # good, bad
        for a, neighbour in self.neighbours():
            pivot_x, pivot_y, _, _ = self.blobs[a]
            for b, a2b in neighbour:
                for c, c2a in neighbour:
                    if c == b:
                        # ignore self
                        continue
                    b2c = Locator.distance(self.blobs[b], self.blobs[c])
                    # a2b is side ab length, c2a is side ac length, b2c is side bc length
                    # we want two sides to be d^2 and the other to be 2d^2
                    if Locator.is_same(a2b, c2a, Locator.MIN_LENGTH_RATIO) is None:
                        if Locator.is_same(a2b, b2c, Locator.MIN_LENGTH_RATIO) is None:
                            if Locator.is_same(c2a, b2c, Locator.MIN_LENGTH_RATIO) is None:
                                # no two sides the same, so not a corner
                                if self.logger is not None:
                                    # log bad ones
                                    ac_ratio = Locator.is_same(a2b, c2a)
                                    ab_ratio = Locator.is_same(a2b, b2c)
                                    cb_ratio = Locator.is_same(c2a, b2c)
                                    # self.logger.log('Corners (bad): neighbours a:{}, b:{}, c:{} have no two sides the same length'
                                    #                 ' a2b/c2a={:.2f}, a2b/b2c={:.2f}, c2a/b2c={:.2f}, limit is {:.2f}'.
                                    #                 format(a, b, c, ac_ratio, ab_ratio, cb_ratio, Locator.MIN_LENGTH_RATIO))
                                    self._corner_side_stats[1].count(max(ac_ratio, ab_ratio, cb_ratio))
                                continue
                            else:
                                # c2a==b2c, c is 'primary' corner, so a2b is the long side, it should be c2a+b2c
                                primary         = c
                                actual_length   = a2b
                                expected_length = c2a + b2c
                        else:
                            # a2b==b2c, b is 'primary' corner, so c2a is the long side, it should be a2b+b2c
                            primary         = b
                            actual_length   = c2a
                            expected_length = a2b + b2c
                    else:
                        # a2b==c2a, a is 'primary' corner, so b2c is the long side, it should be a2b+c2a
                        primary         = a
                        actual_length   = b2c
                        expected_length = a2b + c2a
                    # log good ones
                    if self.logger is not None:
                        ac_ratio = Locator.is_same(a2b, c2a)
                        ab_ratio = Locator.is_same(a2b, b2c)
                        cb_ratio = Locator.is_same(c2a, b2c)
                        # self.logger.log('Corners (good): neighbours a:{}, b:{}, c:{} has two sides the same length'
                        #                 ' a2b/c2a={:.2f}, a2b/b2c={:.2f}, c2a/b2c={:.2f}, limit is {:.2f}'.
                        #                 format(a, b, c, ac_ratio, ab_ratio, cb_ratio, Locator.MIN_LENGTH_RATIO))
                        self._corner_side_stats[0].count(max(ac_ratio, ab_ratio, cb_ratio))
                    squareness = Locator.is_same(expected_length, actual_length, Locator.MIN_DIAGONAL_RATIO)
                    if squareness is None:
                        # not required distance
                        if self.logger is not None:
                            # log bad ones
                            squareness = Locator.is_same(expected_length, actual_length)
                            # self.logger.log('Corners (bad): primary blob {} (of {}, {}, {}) diagonal '
                            #                 '(actual_length/expected_length) '
                            #                 'too far from square at {:.2f} (limit is {:.2f})'.
                            #                 format(primary, a, b, c, squareness, Locator.MIN_DIAGONAL_RATIO))
                            self._corner_diag_stats[1].count(squareness)
                        continue
                    # log good ones
                    if self.logger is not None:
                        squareness = Locator.is_same(expected_length, actual_length)
                        # self.logger.log('Corners (good): primary blob {} (of {}, {}, {}) diagonal '
                        #                 '(actual_length/expected_length) '
                        #                 'is square at {:.2f} (limit is {:.2f})'.
                        #                 format(primary, a, b, c, squareness, Locator.MIN_DIAGONAL_RATIO))
                        self._corner_diag_stats[0].count(squareness)
                    # found a qualifying corner set
                    # save corner in blob number order so can easily find duplicates
                    corners = [a, b, c]
                    corners.sort()
                    corners.append(primary)
                    corners.append(squareness)  # this is used to filter duplicate rectangles, the closer to 1 the better
                    self._corners.append(corners)
        if self.logger is not None:
            self.logger.log('')
            self.logger.log('Corner bad side stats')
            self.logger.log('  ' + self._corner_side_stats[1].show())
            self.logger.log('Corner good side stats')
            self.logger.log('  ' + self._corner_side_stats[0].show())
            self.logger.log('')
            self.logger.log('Corner bad diagonal stats')
            self.logger.log('  ' + self._corner_diag_stats[1].show())
            self.logger.log('Corner good diagonal stats (across {} corners)'.format(len(self._corners)))
            self.logger.log('  ' + self._corner_diag_stats[0].show())
        if len(self._corners) > 0:
            # remove duplicates (NB: relying on blobs within corners being sorted into blob order)
            self._corners.sort(key=lambda k: (k[0], k[1], k[2]))
            ref_a, ref_b, ref_c, _, _ = self._corners[-1]
            for corner in range(len(self._corners)-2, -1, -1):
                a, b, c, _, _ = self._corners[corner]
                if a == ref_a and b == ref_b and c == ref_c:
                    del self._corners[corner+1]
                else:
                    ref_a = a
                    ref_b = b
                    ref_c = c
        return self._corners

    def triangles(self):
        """ return a clockwise right-angle triangle from each corner set """
        if self._triangles is not None:
            # already done it
            return self._triangles

        def quadrant(x, y, px, py):
            if x == px and y == py: return 0
            if x <  px and y <  py: return 1
            if x >= px and y <  py: return 2
            if x >= px and y >= py: return 3
            if x <  px and y >= py: return 4
            # can't get here!
            raise Exception('Logic error! No quadrant for x={}, y={}, px={}, py={}'.format(x, y, px, py))

        def clockwise(qa, qb):
            """ return True iff quadrant b is clockwise from quadrant a """
            # clockwise is Q1->Q2 or Q2->Q3 or Q3->Q4 or Q4->Q1
            return (qa==1 and qb==2) or (qa==2 and qb==3) or (qa==3 and qb==4) or (qa==4 and qb==1)

        self._triangles = []
        for a, b, c, primary, squareness in self.corners():
            # primary is the corner where the right-angle is (one of a or b or c)
            # if we know A is a right-angle it means B and C will be in adjacent quadrants
            # clockwise means B is 'before' C in this:
            #     +----+----+   Q1 = x < A and y < A
            #     | Q1 | Q2 |   Q2 = x > A and y < A
            #     +----A----+   Q3 = x > A and y > A
            #     | Q4 | Q3 |   Q4 = x < A and y > A
            #     +----+----+   Q0 = x = A and y = A
            ax, ay, _, _ = self.blobs[a]
            bx, by, _, _ = self.blobs[b]
            cx, cy, _, _ = self.blobs[c]
            px, py, _, _ = self.blobs[primary]
            qa = quadrant(ax, ay, px, py)
            qb = quadrant(bx, by, px, py)
            qc = quadrant(cx, cy, px, py)
            if qa == 0:
                # its between qb and qc
                if clockwise(qb, qc):
                    # c is after b
                    triangle = [a, b, c]
                else:
                    # b is after c
                    triangle = [a, c, b]
            elif qb ==0:
                # its between qa and qc
                if clockwise(qa, qc):
                    # c is after a
                    triangle = [b, a, c]
                else:
                    # a is after c
                    triangle = [b, c, a]
            elif qc == 0:
                # its between qa and qb
                if clockwise(qa, qb):
                    # b is after a
                    triangle = [c, a, b]
                else:
                    # a is after b
                    triangle = [c, b, a]
            else:
                # bang! can't get here
                raise Exception('Logic error! Primary blob {} is none of {}, {}, {}'.format(primary, a, b, c))
            tl = self.blobs[triangle[0]]
            tr = self.blobs[triangle[1]]
            bl = self.blobs[triangle[2]]
            self._triangles.append((tl, tr, bl, triangle, squareness))
        self.triangles().sort(key=lambda k: (k[3][0], k[3][1], k[3][2]))  # put into blob order purely to help debugging
        return self._triangles

    def rectangles(self):
        """ return a list of (approximate) rectangles for the discovered triangles,
            this involves estimating the position of the fourth corner like so:
               B              A,B,C are the discovered triangle vertices
              / \             D is the fourth corner whose position is to be estimated
             /   \            E is the centre of the BC line as Ex = (Cx-Bx)/2 + Bx and Ey = (Cy-By)/2 + By
            A     \           Dx is then (Ex-Ax)*2 + Ax
             \  E  \          Dy is then (Ey-Ay)*2 + Ay
              \     D
               \   /
                \ /
                 C
            """
        if self._rectangles is not None:
            # already done it
            return self._rectangles
        self._rectangles = []
        for tl, tr, bl, triangle, squareness in self.triangles():
            Ax, Ay, Ar, _ = tl
            Bx, By, _ , _ = tr
            Cx, Cy, _ , _ = bl
            Ex = ((Cx - Bx) / 2) + Bx
            Ey = ((Cy - By) / 2) + By
            Dx = ((Ex - Ax) * 2) + Ax
            Dy = ((Ey - Ay) * 2) + Ay
            br = [Dx, Dy, Ar, None]  # build a blob for the 'bottom-right' corner (same size as 'top-left')
            self._rectangles.append([tl, tr, br, bl, triangle, squareness])
        return self._rectangles

    def detections(self, width=None, height=None):
        """ return a list of unique detections, to be unique all four corners must be distinct,
            when not, the best is kept where 'best' is that with the most equal corner radii and most 'square'
            """
        if width is None:
            width = self._width
        if height is None:
            height = self._height
        if self._detections is not None and self._width == width and self._height == height:
            # already done it
            return self._detections

        self._detections = []
        self._dropped    = []
        self._width      = width
        self._height     = height
        # resolve ambiguity when an estimated corner is very near an actual one
        # (it means a timing mark got detected as a contour)
        for r1, rectangle in enumerate(self.rectangles()):
            if rectangle is None:
                # this one has been dumped
                continue
            r1_quality = Locator.quality(rectangle)
            (_, _, br1, _, _, _) = rectangle
            for r2, rectangle in enumerate(self.rectangles()):
                if r2 == r1:
                    # ignore self
                    continue
                if rectangle is None:
                    # this one has been dumped
                    continue
                (tl, tr, _, bl, _, _) = rectangle
                if Locator.is_near(tl, br1) or Locator.is_near(tr, br1) or Locator.is_near(bl, br1):
                    # got a real corner near our estimate - choose which to dump
                    r2_quality = Locator.quality(rectangle)
                    if r1_quality > r2_quality:
                        # r1 better than r2 - drop r2
                        self._dropped.append(self._rectangles[r2])
                        self._rectangles[r2] = None
                        continue  # carry on looking for more
                    elif r1_quality < r2_quality:
                        # r2 better than r1 - drop r1
                        self._dropped.append(self._rectangles[r1])
                        self._rectangles[r1] = None
                        break  # stop looking for more
                    else:
                        # both same quality! - tough, stay ambiguous
                        continue
        # remove dropped rectangles
        for rectangle in range(len(self._rectangles)-1, -1, -1):
            if self._rectangles[rectangle] is None:
                del self._rectangles[rectangle]
        # build detection list
        for rectangle in self.rectangles():
            if rectangle is None:
                # this one has been dumped
                continue
            self._detections.append(Detection(rectangle, width, height))
        return self._detections

    def dropped(self):
        """ return list of dropped detections (diagnostic aid) """
        self.rectangles()  # make sure we've worked it out
        return self._dropped


def locate_targets(image, params: Params, logger=None) -> [Detection]:
    """ locate targets in the given image blobs using the given params,
        returns a (possibly empty) list of detections
        NB: params are modified
        """
    if logger is not None:
        logger.push('locate_targets')
        logger.log('')
    locator = Locator(params.targets, logger)
    max_x = image.shape[1]  # NB: x is columns and y is rows in the array
    max_y = image.shape[0]  # ..
    detections = locator.detections(max_x, max_y)
    if logger is not None:
        show_results(params, locator, logger)
        logger.pop()
    params.locator = locator  # for upstream diagnostics
    return detections

def show_results(params, locator, logger):
    """ diagnostic aid - log stats and draw images """
    source = params.source
    max_x  = source.shape[1]  # NB: x, y are reversed in numpy arrays
    max_y  = source.shape[0]  # ..
    image  = canvas.colourize(source)  # make into a colour image

    logger.log('')
    logger.log('Locator:')
    neighbours = locator.neighbours()
    logger.log('{} blobs with {} or more neighbours:'.format(len(neighbours), Locator.MIN_NEIGHBOURS))
    for blob, neighbour in neighbours:
        x, y, r, _ = locator.blobs[blob]
        logger.log('  {}: centre: {:.2f}, {:.2f}, radius: {:.2f}, neighbours: {}:'.
                   format(blob, x, y, r, len(neighbour)))
        for there, distance in neighbour:
            x, y, r, _ = locator.blobs[there]
            canvas.circle(image, (x, y), r, const.RED, 1)
            logger.log('      {}: centre: {:.2f}, {:.2f}, radius: {:.2f}, distance: {:.2f} '.
                       format(there, x, y, r, math.sqrt(distance)))
    for blob, neighbour in neighbours:
        x, y, r, _ = locator.blobs[blob]
        canvas.circle(image, (x, y), r, const.YELLOW, 1)
    corners = locator.corners()
    logger.log('{} corner triplets found:'.format(len(corners)))
    for a, b, c, primary, squareness in corners:
        logger.log('  {} -> {} -> {}: primary: {}, squareness: {:.2f}'.format(a, b, c, primary, squareness))
    triangles = locator.triangles()
    logger.log('{} triangles:'.format(len(triangles)))
    for tl, tr, bl, (a, b, c), squareness in triangles:
        ax, ay, _, _ = tl
        bx, by, _, _ = tr
        cx, cy, _, _ = bl
        logger.log('  {} -> {} -> {}: {:.2f} x {:.2f} -> {:.2f} x {:.2f} -> {:.2f} x {:.2f} (squareness={:.2f})'.
                   format(a, b, c, ax, ay, bx, by, cx, cy, squareness))
    rectangles = locator.rectangles()
    logger.log('{} good rectangles:'.format((len(rectangles))))
    for rectangle in rectangles:
        if rectangle is None:
            continue
        tl, tr, br, bl, (a, b, c), squareness = rectangle
        ax, ay, _, _ = tl
        bx, by, _, _ = tr
        cx, cy, _, _ = bl
        dx, dy, dr, _ = br
        logger.log('  {} -> {} -> D -> {}: {:.2f} x {:.2f} -> {:.2f} x {:.2f} -> {:.2f} x {:.2f} -> {:.2f} x {:.2f}'
                   ' (squareness={:.2f})'.format(a, b, c, ax, ay, bx, by, dx, dy, cx, cy, squareness))
        draw_rectangle(image, tl, tr, br, bl)
        canvas.circle(image, (dx, dy), dr, const.GREEN, 1)  # mark our estimated blob
    dropped = locator.dropped()
    logger.log('{} dropped rectangles:'.format((len(dropped))))
    for tl, tr, br, bl, (a, b, c), squareness in dropped:
        ax, ay, _, _ = tl
        bx, by, _, _ = tr
        cx, cy, _, _ = bl
        dx, dy, dr, _ = br
        logger.log('  {} -> {} -> D -> {}: {:.2f} x {:.2f} -> {:.2f} x {:.2f} -> {:.2f} x {:.2f} -> {:.2f} x {:.2f}'
                   ' (squareness={:.2f})'.format(a, b, c, ax, ay, bx, by, dx, dy, cx, cy, squareness))
    detections = locator.detections(max_x, max_y)
    logger.log('{} detections:'.format(len(detections)))
    for detection in detections:
        logger.log('  rectangle: {:.2f} x {:.2f} -> {:.2f} x {:.2f} -> {:.2f} x {:.2f} -> {:.2f} x {:.2f}'.
                   format(detection.tl[0], detection.tl[1],
                          detection.tr[0], detection.tr[1],
                          detection.br[0], detection.br[1],
                          detection.bl[0], detection.bl[1]))
        logger.log('    enclosing box: tl: {} x {} -> br: {} x {} (squareness={:.2f})'.
                   format(detection.box_tl[0], detection.box_tl[1], detection.box_br[0], detection.box_br[1],
                          detection.squareness))
        canvas.rectangle(image, detection.box_tl, (detection.box_br[0]+1, detection.box_br[1]+1), const.GREEN, 1)
    logger.draw(image, file='locators')

def draw_rectangle(image, tl, tr, br, bl, origin=(0,0)):
    """ draw the given located code rectangle corner points on the given image for diagnostic purposes,
        origin is the co-ordinate reference for the given corner points, this is subtracted to gain the
        equivalent co-ordinates in the image (the image may be a 'box' extracted from a larger image)
        """
    ax, ay = tl[0] - origin[0], tl[1] - origin[1]
    bx, by = tr[0] - origin[0], tr[1] - origin[1]
    cx, cy = bl[0] - origin[0], bl[1] - origin[1]
    dx, dy = br[0] - origin[0], br[1] - origin[1]
    canvas.line(image, (int(round(ax)), int(round(ay))), (int(round(bx)), int(round(by))), const.GREEN , 1)
    canvas.line(image, (int(round(bx)), int(round(by))), (int(round(dx)), int(round(dy))), const.YELLOW, 1)
    canvas.line(image, (int(round(dx)), int(round(dy))), (int(round(cx)), int(round(cy))), const.RED   , 1)
    canvas.line(image, (int(round(cx)), int(round(cy))), (int(round(ax)), int(round(ay))), const.BLUE  , 1)
    return image


def _test(src, proximity, logger, blur=3, create_new=True):
    # create_new_blobs=True to create new blobs, False to re-use existing blobs
    import pickle

    if logger.depth() > 1:
        logger.push('locator/_test')
    else:
        logger.push('_test')

    if create_new:
        # this is very slow
        params = contours._test(src, size=const.VIDEO_2K, proximity=proximity, black=const.BLACK_LEVEL[proximity],
                                inverted=True, blur=blur, logger=logger, params=Params())
        blobs_dump = open('locator.blobs','wb')
        pickle.dump(params, blobs_dump)
        blobs_dump.close()
    else:
        blobs_dump = open('locator.blobs', 'rb')
        params = pickle.load(blobs_dump)
        blobs_dump.close()

    params.source_file = src
    locate_targets(params.source, params, logger)
    logger.pop()
    return params  # for upstream test harness


if __name__ == "__main__":
    """ test harness """

    #src = "/home/dave/precious/fellsafe/fellsafe-image/source/kilo-codes/test-alt-bits.png"
    src = '/home/dave/precious/fellsafe/fellsafe-image/media/kilo-codes/kilo-codes-far-150-257-263-380-436-647-688-710-777.jpg'
    #proximity = const.PROXIMITY_CLOSE
    proximity = const.PROXIMITY_FAR

    logger = utils.Logger('locator.log', 'locator')

    _test(src, proximity, logger, blur=3, create_new=True)
