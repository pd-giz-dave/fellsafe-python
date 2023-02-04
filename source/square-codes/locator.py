""" Find locator blobs
    This module is responsible for finding the locator blobs in an image.
    The locator blobs are positioned on 3 corners of a square (similar to QR codes).
    There may be other (smaller) blobs within the square that must be ignored.
    The square may be rotated through any angle, e.g. upside down would be a 180 degree rotation.
    The module is given a list of blobs, each of which has a centre and a radius.
    The geometry sought is:
         x                   x
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
         x
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
    MIN_RADIUS_RATIO     = 2/3  # min size ratio between blobs for them to be considered of similar size
    MAX_LOCATOR_DISTANCE = codes.Codes.LOCATOR_SPACING * STRETCH_FACTOR  # max distance between corner locators
    MIN_LOCATOR_DISTANCE = codes.Codes.LOCATOR_SPACING / STRETCH_FACTOR  # min distance between corner locators
    MIN_NEIGHBOURS       = codes.Codes.LOCATORS_PER_CODE - 1  # expected neighbours per locator

    # blob tuple indexes
    X_COORD = 0
    Y_COORD = 1
    R_COORD = 2

    def __init__(self, blobs, logger=None):
        self.blobs = blobs
        self.logger = logger
        self.neighbours = None

    def distance(self, here, there) -> float:
        """ calculate the distance between the two given blobs,
            returns the distance squared or None if they are too far apart or too dis-similar in size,
            we use squared result so we do not need to do a square root (which is slow)
            """
        # check size first (as its cheap)
        here_r  = here [Locator.R_COORD]
        there_r = there[Locator.R_COORD]
        if here_r > there_r:
            ratio = there_r / here_r
        else:
            ratio = here_r / there_r
        if ratio < Locator.MIN_RADIUS_RATIO:
            # sizes too dis-similar
            return None
        # check distance
        max_distance  = max(here_r, there_r) * Locator.MAX_LOCATOR_DISTANCE
        max_distance *= max_distance
        # max_distance *= 2  # to get the diagonal
        min_distance  = min(here_r, there_r) * Locator.MIN_LOCATOR_DISTANCE
        min_distance *= min_distance
        distance_x    = here[Locator.X_COORD] - there[Locator.X_COORD]
        distance_x   *= distance_x
        distance_y    = here[Locator.Y_COORD] - there[Locator.Y_COORD]
        distance_y   *= distance_y
        distance      = distance_x + distance_y
        if distance > max_distance or distance < min_distance:
            # too far apart or too close
            return None
        # they are potential neighbours
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
                distance = self.distance(here, there)
                if distance is None:
                    # not a neighbour
                    continue
                neighbour.append((candidate, distance))
            if len(neighbour) >= Locator.MIN_NEIGHBOURS:
                self.neighbours.append((blob, neighbour))
        return self.neighbours


if __name__ == "__main__":
    """ test harness """
    import cv2
    import pickle

    #src = "/home/dave/precious/fellsafe/fellsafe-image/media/square-codes/square-codes-distant.jpg"
    src = "/home/dave/precious/fellsafe/fellsafe-image/source/square-codes/test-alt-bits.png"
    proximity = const.PROXIMITY_CLOSE
    #proximity = const.PROXIMITY_FAR

    logger = utils.Logger('locator.log')
    logger.log('_test')

    if True:  # True:  # True to create new blobs, False to re-use existing blobs
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
        cv2.circle(image, (int(round(x)), int(round(y))), int(round(r)), const.GREEN, 1)
    cv2.imwrite('locator-with-neighbours.png', image)

