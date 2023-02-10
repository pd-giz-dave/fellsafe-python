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
    'd' is the distance between two of the three blob centres and must be in the region of 7r,
    'sqrt(2d^2)' is the distance to the other blob centre of a group of three,
    blobs smaller than some factor (<1) of 'r' between the three locator blobs are ignored
"""

import math

import codes     # for the geometry constants
import const     # for the proximity constants
import utils     # for the logger
import contours  # for providing a test source of blobs

Params = contours.Targets  # upstream of here is not allowed to know about contours

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
        value = int(min(max(int(value), min_value), max_value))
        return value

    def __init__(self, rectangle, image_width, image_height):
        """ rectangle is as created by Locator.rectangles() - 4 blob of x,y,radius,label,
            image width/height is used to constrain the enclosing box for detections close to the image edges
            the rectangle co-ordinates may be fractional, the box co-ordinates are not
            """
        self.tl = (rectangle[0][0], rectangle[0][1])
        self.tr = (rectangle[1][0], rectangle[1][1])
        self.br = (rectangle[2][0], rectangle[2][1])
        self.bl = (rectangle[3][0], rectangle[3][1])
        margin = max(rectangle[0][2], rectangle[1][2], rectangle[2][2], rectangle[3][2]) * Detection.MARGIN_SCALE
        min_x = Detection.constrain(min(self.tl[0], self.tr[0], self.br[0], self.bl[0]) - margin, 0, image_width-1)
        max_x = Detection.constrain(max(self.tl[0], self.tr[0], self.br[0], self.bl[0]) + margin, 0, image_width-1)
        min_y = Detection.constrain(min(self.tl[1], self.tr[1], self.br[1], self.bl[1]) - margin, 0, image_height-1)
        max_y = Detection.constrain(max(self.tl[1], self.tr[1], self.br[1], self.bl[1]) + margin, 0, image_height-1)
        self.box_tl = (min_x, min_y)
        self.box_br = (max_x, max_y)


class Locator:

    # region geometry...
    STRETCH_FACTOR       = 1.3  # how much image 'stretch' to allow for (as a result of distortion and viewing angle)
    MIN_RADIUS_RATIO     = 0.7  # min size ratio between blobs for them to be considered of similar size
    MAX_LOCATOR_DISTANCE = codes.Codes.LOCATOR_SPACING * STRETCH_FACTOR  # max distance between corner locators
    MIN_LOCATOR_DISTANCE = codes.Codes.LOCATOR_SPACING / STRETCH_FACTOR  # min distance between corner locators
    MIN_NEIGHBOURS       = codes.Codes.LOCATORS_PER_CODE - 1  # expected neighbours per locator
    MIN_LENGTH_RATIO     = 0.7  # min length ratio of two triangle sides to be considered equal
    MIN_DIAGONAL_RATIO   = 0.8  # min diagonal ratio between expected and actual length to be considered equal
    # endregion

    # blob tuple indexes
    X_COORD = 0
    Y_COORD = 1
    R_COORD = 2

    def __init__(self, blobs, logger=None):
        self.blobs      = blobs
        self.logger     = logger
        self._neighbours = None
        self._corners    = None
        self._triangles  = None
        self._rectangles = None
        self._detections = None

    @staticmethod
    def is_same(a, b, limit) -> bool:
        """ determine if the two given numbers are considered equal within the given limit """
        ratio = min(a, b) / max(a, b)
        return ratio >= limit

    @staticmethod
    def distance(here, there) -> float:
        """ calculate the distance between the two given blobs,
            returns the distance squared or None if they are too far apart,
            we use squared result so we do not need to do a square root (which is slow)
            """
        distance_x  = here[Locator.X_COORD] - there[Locator.X_COORD]
        distance_x *= distance_x
        distance_y  = here[Locator.Y_COORD] - there[Locator.Y_COORD]
        distance_y *= distance_y
        distance    = distance_x + distance_y
        return distance

    def neighbours(self):
        """ return a list of similar size near neighbours for each blob """
        if self._neighbours is not None:
            # already done it
            return self._neighbours

        # this is a crude O(N^2) algorithm, I'm sure there are better ways!, eg. a k-d tree
        self._neighbours = []
        for blob, here in enumerate(self.blobs):
            neighbour = []
            for candidate, there in enumerate(self.blobs):
                if candidate == blob:
                    # ignore self
                    continue
                # check size first (as its cheap)
                here_r  = here [Locator.R_COORD]
                there_r = there[Locator.R_COORD]
                if not Locator.is_same(here_r, there_r, Locator.MIN_RADIUS_RATIO):
                    # sizes too dis-similar
                    continue
                # size OK, now check distance
                max_distance = max(here_r, there_r) * Locator.MAX_LOCATOR_DISTANCE
                min_distance = min(here_r, there_r) * Locator.MIN_LOCATOR_DISTANCE
                max_distance *= max_distance
                min_distance *= min_distance
                distance = Locator.distance(here, there)
                if distance > max_distance or distance < min_distance:
                    # length out of range
                    continue
                neighbour.append((candidate, distance))
            if len(neighbour) >= Locator.MIN_NEIGHBOURS:
                self._neighbours.append((blob, neighbour))
        return self._neighbours

    def corners(self):
        """ return a list of triplets that meet our corner requirements,
            that is a triangle A,B,C such that A->B == A->C == d and B->C == sqrt(2d^2)
            """
        if self._corners is not None:
            # already done it
            return self._corners

        self._corners = []
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
                    if not Locator.is_same(a2b, c2a, Locator.MIN_LENGTH_RATIO):
                        if not Locator.is_same(a2b, b2c, Locator.MIN_LENGTH_RATIO):
                            if not Locator.is_same(c2a, b2c, Locator.MIN_LENGTH_RATIO):
                                # no two sides the same, so not a corner
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
                    if not Locator.is_same(expected_length, actual_length, Locator.MIN_DIAGONAL_RATIO):
                        # not required distance
                        continue
                    # found a qualifying corner set
                    # save corner in blob number order so can easily find duplicates
                    corners = [a, b, c]
                    corners.sort()
                    corners.append(primary)
                    self._corners.append(corners)
        if len(self._corners) > 0:
            # remove duplicates (NB: relying on blobs within corners being sorted into blob order)
            self._corners.sort(key=lambda k: (k[0], k[1], k[2]))
            ref = self._corners[-1]
            for corner in range(len(self._corners)-2, -1, -1):
                abc = self._corners[corner]
                if abc == ref:
                    del self._corners[corner+1]
                else:
                    ref = abc
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
        for a, b, c, primary in self.corners():
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
            self._triangles.append((tl, tr, bl, triangle))
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
        for tl, tr, bl, blobs in self.triangles():
            Ax, Ay, Ar, _ = tl
            Bx, By, _ , _ = tr
            Cx, Cy, _ , _ = bl
            Ex = ((Cx - Bx) / 2) + Bx
            Ey = ((Cy - By) / 2) + By
            Dx = ((Ex - Ax) * 2) + Ax
            Dy = ((Ey - Ay) * 2) + Ay
            br = (Dx, Dy, Ar, None)  # build a blob for the 'bottom-right' corner (same size as 'top-left')
            self._rectangles.append((tl, tr, br, bl, blobs))
        # ToDo: resolve ambiguity when an estimated blob is very near an actual one
        #       (it means a timing mark got detected as a contour)
        return self._rectangles

    def detections(self, width, height):
        """ return a list of detections """
        if self._detections is not None:
            # already done it
            return self._detections
        self._detections = []
        for rectangle in self.rectangles():
            self._detections.append(Detection(rectangle, width, height))
        return self._detections


def locate_targets(image, params: Params, logger=None) -> (Params, [Detection]):
    """ locate targets in the given image using the given params,
        returns the updated params (for diagnostics) and a (possibly empty) list of detections
        """
    contours.find_targets(image, params, logger=logger)
    locator = Locator(params.blobs, logger.log)
    max_x = image.shape[1]  # NB: x is columns and y is rows in the array
    max_y = image.shape[0]  # ..
    detections = locator.detections(max_x, max_y)
    return params, detections


def _test(src, proximity, logger, blur=3, create_new_blobs=True):
    # create_new_blobs=True to create new blobs, False to re-use existing blobs
    import cv2
    import pickle

    if create_new_blobs:
        # this is very slow
        results = contours._test(src, size=const.VIDEO_2K, proximity=proximity, black=const.BLACK_LEVEL[proximity],
                                 inverted=True, blur=blur, logger=logger)
        blobs_dump = open('locator.blobs','wb')
        pickle.dump(results, blobs_dump)
        blobs_dump.close()
    else:
        blobs_dump = open('locator.blobs', 'rb')
        results = pickle.load(blobs_dump)
        blobs_dump.close()
    blobs = results.targets
    image = results.source
    max_x = image.shape[1]  # NB: x, y are reversed in mumpy arrays
    max_y = image.shape[0]  #     ..
    image = cv2.merge([image, image, image])  # make into a colour image

    logger.log('\n\nLocator:')
    blobs.sort(key=lambda k: (k[0], k[1]))  # put in x,y order purely to help debugging
    locator = Locator(blobs, logger.log)
    neighbours = locator.neighbours()
    logger.log('{} blobs with {} or more neighbours:'.format(len(neighbours), Locator.MIN_NEIGHBOURS))
    for blob, neighbour in neighbours:
        x, y, r, _ = locator.blobs[blob]
        logger.log('  {}: centre: {:.2f}, {:.2f}, radius: {:.2f}, neighbours: {}:'.
                   format(blob, x, y, r, len(neighbour)))
        for there, distance in neighbour:
            x, y, r, _ = locator.blobs[there]
            cv2.circle(image, (int(round(x)), int(round(y))), int(round(r)), const.RED, 1)
            logger.log('      {}: centre: {:.2f}, {:.2f}, radius: {:.2f}, distance: {:.2f} '.
                       format(there, x, y, r, math.sqrt(distance)))
    for blob, neighbour in neighbours:
        x, y, r, _ = locator.blobs[blob]
        cv2.circle(image, (int(round(x)), int(round(y))), int(round(r)), const.BLUE, 1)
    corners = locator.corners()
    logger.log('{} corner triplets found:'.format(len(corners)))
    for a, b, c, primary in corners:
        logger.log('  {} -> {} -> {}: primary: {}'.format(a, b, c, primary))
    triangles = locator.triangles()
    logger.log('{} triangles:'.format(len(triangles)))
    for tl, tr, bl, (a, b, c) in triangles:
        ax, ay, _, _ = tl
        bx, by, _, _ = tr
        cx, cy, _, _ = bl
        logger.log('  {} -> {} -> {}: {:.2f} x {:.2f} -> {:.2f} x {:.2f} -> {:.2f} x {:.2f}'.
                   format(a, b, c, ax, ay, bx, by, cx, cy))
    rectangles = locator.rectangles()
    logger.log('{} rectangles:'.format((len(rectangles))))
    for tl, tr, br, bl, (a, b, c) in rectangles:
        ax, ay, _ , _ = tl
        bx, by, _ , _ = tr
        cx, cy, _ , _ = bl
        dx, dy, dr, _ = br
        logger.log('  {} -> {} -> D -> {}: {:.2f} x {:.2f} -> {:.2f} x {:.2f} -> {:.2f} x {:.2f} -> {:.2f} x {:.2f}'.
                   format(a, b, c, ax, ay, bx, by, dx, dy, cx, cy))
        cv2.line(image, (int(round(ax)), int(round(ay))), (int(round(bx)), int(round(by))), const.RED   , 1)
        cv2.line(image, (int(round(bx)), int(round(by))), (int(round(dx)), int(round(dy))), const.GREEN , 1)
        cv2.line(image, (int(round(dx)), int(round(dy))), (int(round(cx)), int(round(cy))), const.YELLOW, 1)
        cv2.line(image, (int(round(cx)), int(round(cy))), (int(round(ax)), int(round(ay))), const.BLUE  , 1)
        cv2.circle(image, (int(round(dx)), int(round(dy))), int(round(dr)), const.GREEN, 1)  # mark our estimated blob
    detections = locator.detections(max_x, max_y)
    logger.log('{} detections:'.format(len(detections)))
    for detection in detections:
        logger.log('  rectangle: {:.2f} x {:.2f} -> {:.2f} x {:.2f} -> {:.2f} x {:.2f} -> {:.2f} x {:.2f}'.
                   format(detection.tl[0], detection.tl[1],
                          detection.tr[0], detection.tr[1],
                          detection.br[0], detection.br[1],
                          detection.bl[0], detection.bl[1]))
        logger.log('    enclosing box: tl: {} x {} -> br: {} x {}'.format(detection.box_tl[0], detection.box_tl[1],
                                                                          detection.box_br[0], detection.box_br[1]))
        cv2.rectangle(image, detection.box_tl, detection.box_br, const.GREEN, 1)
    cv2.imwrite('locators.png', image)
    return locator, results.source  # for upstream test harness


if __name__ == "__main__":
    """ test harness """

    src = "/home/dave/precious/fellsafe/fellsafe-image/media/square-codes/square-codes-distant.jpg"
    #src = "/home/dave/precious/fellsafe/fellsafe-image/source/square-codes/test-alt-bits.png"
    #proximity = const.PROXIMITY_CLOSE
    proximity = const.PROXIMITY_FAR

    logger = utils.Logger('locator.log')
    logger.log('_test')

    _test(src, proximity, logger, create_new_blobs=True)
