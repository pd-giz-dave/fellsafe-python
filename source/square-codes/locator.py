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
    blobs smaller than 2r/3 between the three locator blobs are ignored
"""
import math

import codes     # for the geometry constants
import const     # for the proximity constants
import utils     # for the logger
import contours  # for providing a test source of blobs

class Locator:

    # geometry
    STRETCH_FACTOR       = 1.3  # how much image 'stretch' to allow for (as a result of distortion and viewing angle)
    MIN_RADIUS_RATIO     = 0.7  # min size ratio between blobs for them to be considered of similar size
    MAX_LOCATOR_DISTANCE = codes.Codes.LOCATOR_SPACING * STRETCH_FACTOR  # max distance between corner locators
    MIN_LOCATOR_DISTANCE = codes.Codes.LOCATOR_SPACING / STRETCH_FACTOR  # min distance between corner locators
    MIN_NEIGHBOURS       = codes.Codes.LOCATORS_PER_CODE - 1  # expected neighbours per locator
    MIN_LENGTH_RATIO     = 0.7  # min length ratio of two triangle sides to be considered equal
    MIN_DIAGONAL_RATIO   = 0.8  # min diagonal ratio between expected and actual length to be considered equal

    # blob tuple indexes
    X_COORD = 0
    Y_COORD = 1
    R_COORD = 2

    def __init__(self, blobs, logger=None):
        self.blobs      = blobs
        self.logger     = logger
        self.neighbours = None
        self.corners    = None


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

    def build_neighbours(self):
        """ build a list of similar size near neighbours for each blob """
        # this is a crude O(N^2) algorithm, I'm sure there are better ways!, eg. a k-d tree
        self.neighbours = []
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
                self.neighbours.append((blob, neighbour))
        return self.neighbours

    def find_corners(self):
        """ from the neighbour lists find the triplets that meet our corner requirements,
            that is a triangle A,B,C such that A->B == A->C == d and B->C == sqrt(2d^2)
            """
        self.corners = []
        for a, neighbour in self.neighbours:
            pivot_x, pivot_y, _, _ = self.blobs[a]
            for b, a2b in neighbour:
                for c, a2c in neighbour:
                    if c == b:
                        # ignore self
                        continue
                    b2c = Locator.distance(self.blobs[b], self.blobs[c])
                    # a2b is side ab length, a2c is side ac length, b2c is side bc length
                    # we want two sides to be d^2 and the other to be 2d^2
                    if not Locator.is_same(a2b, a2c, Locator.MIN_LENGTH_RATIO):
                        if not Locator.is_same(a2b, b2c, Locator.MIN_LENGTH_RATIO):
                            if not Locator.is_same(a2c, b2c, Locator.MIN_LENGTH_RATIO):
                                # no two sides the same, so not a corner
                                continue
                            else:
                                # a2b is the long side, it should be a2c+b2c
                                actual_length   = a2b
                                expected_length = a2c + b2c
                        else:
                            # a2c is the long side, it should be a2b+b2c
                            actual_length   = a2c
                            expected_length = a2b + b2c
                    else:
                        # b2c  is the long side, it should be a2b+a2c
                        actual_length   = b2c
                        expected_length = a2b + a2c
                    if not Locator.is_same(expected_length, actual_length, Locator.MIN_DIAGONAL_RATIO):
                        # not required distance
                        continue
                    # found a qualifying corner set
                    # save corner in blob number order so can easily find duplicates
                    corners = [a, b, c]
                    corners.sort()
                    self.corners.append(corners)
        if len(self.corners) > 0:
            # remove duplicates (NB: relying on blobs within corners being sorted into blob order)
            self.corners.sort(key=lambda k: (k[0], k[1], k[2]))
            ref = self.corners[-1]
            for corner in range(len(self.corners)-2, -1, -1):
                abc = self.corners[corner]
                if abc == ref:
                    del self.corners[corner+1]
                else:
                    ref = abc
        return self.corners

if __name__ == "__main__":
    """ test harness """
    import cv2
    import pickle

    #src = "/home/dave/precious/fellsafe/fellsafe-image/media/square-codes/square-codes-close.jpg"
    src = "/home/dave/precious/fellsafe/fellsafe-image/source/square-codes/test-alt-bits.png"
    proximity = const.PROXIMITY_CLOSE
    #proximity = const.PROXIMITY_FAR

    logger = utils.Logger('locator.log')
    logger.log('_test')

    if True:  # True to create new blobs, False to re-use existing blobs
        results = contours._test(src, size=const.VIDEO_2K, proximity=proximity, black=const.BLACK_LEVEL[proximity],
                                 inverted=True, logger=logger)
        blobs_dump = open('locator.blobs','wb')
        pickle.dump(results, blobs_dump)
        blobs_dump.close()
    else:
        blobs_dump = open('locator.blobs', 'rb')
        results = pickle.load(blobs_dump)
        blobs_dump.close()
    blobs = results.targets
    image = results.source
    image = cv2.merge([image, image, image])  # make into a colour image

    logger.log('\n\nLocator:')
    blobs.sort(key=lambda k: (k[0], k[1]))  # put in x,y order purely to help debugging
    locator = Locator(blobs, logger.log)
    neighbours = locator.build_neighbours()
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
    corners = locator.find_corners()
    logger.log('{} corner triplets found:'.format(len(corners)))
    for a, b, c in corners:
        logger.log('  {} -> {} -> {}'.format(a, b, c))
        ax, ay, _, _ = locator.blobs[a]
        bx, by, _, _ = locator.blobs[b]
        cx, cy, _, _ = locator.blobs[c]
        cv2.line(image, (int(round(ax)), int(round(ay))), (int(round(bx)), int(round(by))), const.GREEN, 1)
        cv2.line(image, (int(round(bx)), int(round(by))), (int(round(cx)), int(round(cy))), const.GREEN, 1)
        cv2.line(image, (int(round(cx)), int(round(cy))), (int(round(ax)), int(round(ay))), const.GREEN, 1)
    cv2.imwrite('locators.png', image)


