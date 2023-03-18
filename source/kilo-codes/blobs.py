""" Find qualifying blobs
    Finds all contours then filters for the relevant ones
"""
import math

import const
import utils
import shapes
import canvas
import contours

class Params(contours.Params):

    depinch: bool = True  # iff True de-pinch blobs by splitting them

    # pre-filter params
    min_size: float = 2  # min number of pixels across the contour box
    max_size: float = 128  # max number of pixels across the contour box
    max_internals: int = 1  # max number of internal contours that is tolerated to be a blob

    # all these 'ness' parameters are in the range 0..1, where 0 is perfect and 1 is utter crap
    max_squareness = 0.5  # how close to square the bounding box has to be (0.5 is a 2:1 rectangle)
    max_wavyness   = 0.35  # how close to not wavy a contour perimeter must be
    max_offsetness = 0.05  # how close the centroid has to be to the enclosing box centre
    max_whiteness  = 0.3  # whiteness of the enclosing circle
    max_blackness  = 0.6  # whiteness of the enclosing box (0.5 is worst case for a 45 deg rotated sq)
    targets: [tuple] = None  # qualifying blobs

class Target:
    """ properties/methods for reducing blobs (as detected by contours.py) to potential targets """

    # region directions...
    NORTH      = 'N'
    SOUTH      = 'S'
    EAST       = 'E'
    WEST       = 'W'
    NORTH_EAST = 'NE'
    NORTH_WEST = 'NW'
    SOUTH_EAST = 'SE'
    SOUTH_WEST = 'SW'

    # opposite codes, higher numbers are better
    NOT_OPPOSITE     = 0  # worst, <early
    EARLY_OPPOSITE   = 1  # relative to clockwise
    LATE_OPPOSITE    = 2  # ..
    EXACTLY_OPPOSITE = 3  # best, >late

    # the opposite is the direct opposite compass point and +/- 1 point either side,
    # e.g the opposite of N is S and SW and SE
    #         N
    #         x
    #   NW  x | x  NE
    #        \|/
    #   W  x--+--x E
    #        /|\
    #   SW  x | x  SE
    #         x
    #         S

    # list of opposites for each direction, first in list is 'exact', rest are 'nearly' and in clockwise order
    OPPOSITES = {NORTH     : (SOUTH     , SOUTH_EAST, SOUTH_WEST),
                 SOUTH     : (NORTH     , NORTH_WEST, NORTH_EAST),
                 EAST      : (WEST      , SOUTH_WEST, NORTH_WEST),
                 WEST      : (EAST      , NORTH_EAST, SOUTH_EAST),
                 NORTH_EAST: (SOUTH_WEST, SOUTH     , WEST      ),
                 NORTH_WEST: (SOUTH_EAST, EAST      , SOUTH     ),
                 SOUTH_EAST: (NORTH_WEST, WEST      , NORTH     ),
                 SOUTH_WEST: (NORTH_EAST, NORTH     , EAST      )}
    # endregion

    PINCHPOINT_MAX_DISTANCE  = 0.2  # max fraction of max opposite distance considered to be a pinchpoint
    PINCHPOINT_MAX_DISTANCE *= PINCHPOINT_MAX_DISTANCE  # squared
    PINCHPOINT_MIN_LENGTH    = 0.1  # min fraction of contour length considered to be a pinchpoint
    PINCHPOINT_MIN_SAMPLES   = max(int(1/PINCHPOINT_MIN_LENGTH)+1,4)  # minimum samples to trigger a pinchpoint search
                               # above must be at least 4 to make sure a split always creates a distinct contour

    def __init__(self, blob: shapes.Blob):
        self.blob        = blob
        self.perimeter   = self.blob.get_points()  # NB: need all contour points so can calculate accurate direction
        self.direction   = None  # the 'direction' of each contour sample
        self.opposites   = None  # the 'best' opposite sample for each sample
        self.pinchpoints = None  # pinchpoints of the contour (if there are any)

    @staticmethod
    def is_opposite(here, there):
        """ determine if direction here is opposite to that at there,
            returns not opposite, or nearly opposite or exactly opposite
            """
        opposites = Target.OPPOSITES[here]
        if opposites[0] == there:
            return Target.EXACTLY_OPPOSITE
        elif opposites[-1] == there:
            return Target.LATE_OPPOSITE
        elif there in opposites:
            return Target.EARLY_OPPOSITE
        else:
            return Target.NOT_OPPOSITE

    def get_direction(self, sample=None):
        """ determine the direction a contour is moving in at the given, or all, sample location """
        # there are 8 possible directions calculated from the difference of the x,y co-ordinate in successive samples
        # they are referred as compass bearings: N, NE, E, SE, S, SW, W, NW
        if self.direction is None:
            self.direction = []
            max_samples = len(self.perimeter)
            for sample in range(max_samples):
                here_x , here_y  = self.perimeter[sample]
                there_x, there_y = self.perimeter[(sample+1)%max_samples]
                east  = there_x > here_x
                west  = there_x < here_x
                south = there_y > here_y
                north = there_y < here_y
                if east and south:
                    direction = Target.SOUTH_EAST
                elif east and north:
                    direction = Target.NORTH_EAST
                elif west and south:
                    direction = Target.SOUTH_WEST
                elif west and north:
                    direction = Target.NORTH_WEST
                elif east:
                    direction = Target.EAST
                elif west:
                    direction = Target.WEST
                elif north:
                    direction = Target.NORTH
                elif south:
                    direction = Target.SOUTH
                else:
                    # this means both samples are in the same place, this is legit for the last sample
                    # as its the same as the first, otherwise there's a screw up!
                    if sample == (max_samples - 1):
                        # treat direction same as last one
                        direction = self.direction[-1]
                    else:
                        raise Exception('Samples {} and {} are both the same! (N={:.2f}x{:.2f}y, N+1={:.2f}x{:.2f}y)'.
                                        format(sample, (sample+1)%max_samples, here_x, here_y, there_x, there_y))
                self.direction.append(direction)
        if sample is None:
            return self.direction
        else:
            return self.direction[sample]

    def get_opposites(self):
        """ get the opposite sample with the shortest distance for each sample """
        if self.opposites is not None:
            return self.opposites

        self.get_direction()  # build the direction list (if not already)
        self.opposites = []
        max_samples = len(self.perimeter)
        for sample in range(max_samples):
            here = self.direction[sample]
            best_distance   = None
            best_opposite   = None
            best_opposition = None
            for opposite in range(max_samples):
                opposition = Target.is_opposite(here, self.direction[opposite])
                if opposition == Target.NOT_OPPOSITE:
                    continue
                distance = utils.distance(self.perimeter[sample], self.perimeter[opposite])
                if best_distance is None:
                    best_distance   = distance
                    best_opposite   = opposite
                    best_opposition = opposition
                elif distance < best_distance:
                    best_distance   = distance
                    best_opposite   = opposite
                    best_opposition = opposition
                elif distance > best_distance:
                    continue
                elif opposition > best_opposition:
                    best_distance = distance
                    best_opposite = opposite
                    best_opposition = opposition
                elif opposition < best_opposition:
                    continue
                else:
                    # same distance, same opposition, now what?
                    break
            # each opposite pair may also be found in opposite directions (due to symmetry, a->b == b->a)
            # that's benign but confusing when viewing logs and could be removed, but its not worth the effort
            if best_opposite is None:
                raise Exception('Cannot find unique opposite sample for {}'.format(sample))
            self.opposites.append((best_opposite, best_distance))
        return self.opposites

    def split_pinchpoint(self):
        """ split contour at worst pinchpoint"""

    def get_pinchpoints(self):
        """ get a list 'pinchpoint's for the blob in worst to least worst order,
            a pinchpoint is characterised by two opposite points with a distance below some threshold,
            each has two attributes: the head length and the tail length
            the head length is the length after the tail has been removed,
            the tail length is the number of samples between the opposites,
            the worst pinchpoint has the shortest distance with the biggest head or tail
            """
        if self.pinchpoints is not None:
            return self.pinchpoints

        self.pinchpoints = []
        if len(self.get_opposites()) < Target.PINCHPOINT_MIN_SAMPLES:  # NB: also triggers opposites calc
            # not enough samples to be meaningful
            return self.pinchpoints

        # find distance/length limits
        max_length  = len(self.opposites)
        max_distance = None
        for _, distance in self.opposites:
            if max_distance is None or distance > max_distance:
                max_distance = distance
        max_distance *= Target.PINCHPOINT_MAX_DISTANCE
        min_length    = max(max_length * Target.PINCHPOINT_MIN_LENGTH, Target.PINCHPOINT_MIN_SAMPLES)
        # build pinchpoint list
        for start, (finish, distance) in enumerate(self.opposites):
            if distance > max_distance:
                # too far apart to be considered a pinchpoint
                continue
            if start <= finish:
                tail = finish - start + 1
            else:
                tail = start - finish + 1
            head = max_length - tail + 2  # +2 'cos we put the start/finish samples in both
            if min(head, tail) < min_length:
                # too small to be considered a pinchpoint
                continue
            self.pinchpoints.append((start, finish, head, tail, distance))
        self.pinchpoints.sort(key=lambda k: (k[4], max_length - max(k[2], k[3])))  # +distance, -length
        return self.pinchpoints

def sanatize_blob(blob: shapes.Blob, params: Params, logger=None) -> [Target]:
    """ filter out junk and split a blob that has a pinchpoint.
        returns a list of split targets, empty if target is junk, single element if no pinchpoints,
        otherwise an element for pinchpoint split
        """

    target = Target(blob)

    # do the cheap filtering first
    if len(blob.internal) > params.max_internals:
        blob.rejected = const.REJECT_INTERNALS
        return []
    size = blob.get_size()
    if size < params.min_size:
        blob.rejected = const.REJECT_TOO_SMALL
        return []
    if size > params.max_size:
        blob.rejected = const.REJECT_TOO_BIG
        return []
    wavyness = blob.get_wavyness()
    if wavyness > params.max_wavyness:
        blob.rejected = const.REJECT_WAVYNESS
        return []

    if not params.depinch:
        # being told not go any further
        return [target]

    pinchpoints = target.get_pinchpoints()
    if len(pinchpoints) == 0:
        # no pinchpoint here
        return [target]

    start, finish, head, tail, distance = pinchpoints[0]
    head_blob = blob.extract(start, finish)
    tail_blob = blob.extract(finish, start)
    if logger is not None:
        logger.log('Blob {}\nhas {} pinchpoints, splitting worst into:\n{}\n{}'.
                   format(blob, len(pinchpoints), head_blob, tail_blob))

    head_targets = sanatize_blob(head_blob, params, logger)
    tail_targets = sanatize_blob(tail_blob, params, logger)

    blob.rejected = const.REJECT_SPLIT

    return head_targets + tail_targets

def filter_blobs(blobs: [shapes.Blob], params: Params, logger=None) -> [shapes.Blob]:
    """ filter out blobs that do no meet the target criteria,
        marks *all* blobs with an appropriate reject code and returns a list of good ones
        """

    if logger is not None:
        logger.push("filter_blobs")

    good_blobs = []                      # good blobs are accumulated in here
    all_blobs = 0
    for blob in blobs:
        targets = sanatize_blob(blob, params, logger)
        # ToDo: after splitting pinchpoints the ratio of max opposite distance and box size is telling!
        for target in targets:
            all_blobs += 1
            blob = target.blob
            # offsetness = blob.get_offsetness()
            # if offsetness > params.max_offsetness:
            #     # we've got a potential saucepan
            #     reason_code = const.REJECT_OFFSETNESS
            #     break
            squareness = blob.get_squareness()
            if squareness > params.max_squareness:
                blob.rejected = const.REJECT_SQUARENESS
                continue
            blackness = blob.get_blackness()
            if blackness > params.max_blackness:
                blob.rejected = const.REJECT_BLACKNESS
                continue
            whiteness = blob.get_whiteness()
            if whiteness > params.max_whiteness:
                blob.rejected = const.REJECT_WHITENESS
                continue
            # all filters passed
            blob.rejected = const.REJECT_NONE
            good_blobs.append(blob)

    if logger is not None:
        rejected = all_blobs - len(good_blobs)
        logger.log("Accepted blobs: {}, rejected {} ({:.2f}%) of {}".
                   format(len(good_blobs), rejected, (rejected / all_blobs) * 100, all_blobs))
        logger.pop()

    return good_blobs

def find_blobs(params, logger=None) -> Params:
    """ find targets in the detected blobs """
    if logger is not None:
        logger.push("find_blobs")
        logger.log('')
    passed = filter_blobs(params.blobs, params, logger=logger)
    params.targets = []
    for blob in passed:
        circle = blob.external.get_enclosing_circle(params.mode)
        params.targets.append([circle.centre[0], circle.centre[1], circle.radius, blob])
    if logger is not None:
        show_result(params, logger)
        logger.pop()
    return params

def get_blobs(image, params, logger=None):
    params = contours.find_contours(image, params, logger)
    params = find_blobs(params, logger=logger)
    return params

def show_result(params, logger):

    # extract the relevant part of the image
    if params.box is not None:
        source_part = canvas.extract(params.source, params.box)
    else:
        source_part = params.source

    # highlight our detections on the greyscale image
    draw = canvas.colourize(source_part)
    for (x, y, r, _) in params.targets:
        canvas.circle(draw, (x, y), r, const.GREEN, 1)
    logger.draw(draw, file='blobs')

    # show rejected blobs
    blobs   = params.blobs      # rejects will be in here
    buffer  = params.contours
    labels  = params.labels
    max_x, max_y = canvas.size(buffer)
    draw_bad = canvas.colourize(source_part)
    colours = const.REJECT_COLOURS
    for x in range(max_x):
        for y in range(max_y):
            label = buffer[y, x]
            if label > 0:
                blob = labels.get_blob(label)
                colour, _ = colours[blob.rejected]
                if blob.rejected == const.REJECT_NONE:
                    # good ones are done elsewhere
                    pass
                else:
                    # bad ones are done here
                    draw_bad[y, x] = colour
    logger.draw(draw_bad, file='blobs_rejected')

    # draw accepted blobs (this includes splits)
    draw_good = canvas.colourize(source_part)
    for (x, y, r, blob) in params.targets:
        colour, _ = colours[blob.rejected]
        for (x, y) in blob.external.points:
            draw_good[y, x] = colour
    logger.draw(draw_good, file='blobs_accepted')

    logger.log('\n')
    logger.log("All accepted blobs:")
    stats_buckets = 20
    all_squareness_stats = utils.Stats(stats_buckets)
    all_wavyness_stats = utils.Stats(stats_buckets)
    all_whiteness_stats = utils.Stats(stats_buckets)
    all_blackness_stats = utils.Stats(stats_buckets)
    all_offsetness_stats = utils.Stats(stats_buckets)
    squareness_stats = utils.Stats(stats_buckets)
    wavyness_stats = utils.Stats(stats_buckets)
    whiteness_stats = utils.Stats(stats_buckets)
    blackness_stats = utils.Stats(stats_buckets)
    offsetness_stats = utils.Stats(stats_buckets)
    reject_stats = utils.Frequencies()
    good_blobs = 0
    blobs.sort(key=lambda k: (k.external.get_enclosing_box()[0][0], k.external.get_enclosing_box()[0][1]))

    def update_count(stats, value):
        stats.count(value)

    def log_stats(name, stats):
        msg = stats.show()
        logger.log('  {:10}: {}'.format(name, msg))

    for b, blob in enumerate(blobs):
        reject_stats.count(blob.rejected)
        squareness, wavyness, whiteness, blackness, offsetness = blob.get_quality_stats()
        update_count(all_squareness_stats, squareness)
        update_count(all_wavyness_stats, wavyness)
        update_count(all_whiteness_stats, whiteness)
        update_count(all_blackness_stats, blackness)
        update_count(all_offsetness_stats, offsetness)
        if offsetness > params.max_offsetness*2:
            logger.log('HACK: blob #{} is offset by {:.2f} ({})'.format(b, offsetness, blob))
        if blob.rejected != const.REJECT_NONE:
            continue
        good_blobs += 1
        update_count(squareness_stats, squareness)
        update_count(wavyness_stats, wavyness)
        update_count(whiteness_stats, whiteness)
        update_count(blackness_stats, blackness)
        update_count(offsetness_stats, offsetness)
        logger.log("  {}: {}".format(b, blob.show(verbose=True)))
    # show stats
    logger.log('')
    logger.log("All reject frequencies (across {} blobs):".format(len(blobs)))
    logger.log('  ' + reject_stats.show())
    logger.log('')
    logger.log("All blobs stats (across {} blobs):".format(len(blobs)))
    log_stats("squareness", all_squareness_stats)
    log_stats("wavyness", all_wavyness_stats)
    log_stats("whiteness", all_whiteness_stats)
    log_stats("blackness", all_blackness_stats)
    log_stats("offsetness", all_offsetness_stats)
    logger.log('')
    logger.log("All accepted blobs stats (across {} blobs):".format(good_blobs))
    log_stats("squareness", squareness_stats)
    log_stats("wavyness", wavyness_stats)
    log_stats("whiteness", whiteness_stats)
    log_stats("blackness", blackness_stats)
    log_stats("offsetness", offsetness_stats)
    logger.log('')
    logger.log("All detected targets:")
    params.targets.sort(key=lambda k: (k[0], k[1]))
    for t, (x, y, r, target) in enumerate(params.targets):
        logger.log("  {}: centre: {:.2f}, {:.2f}  radius: {:.2f}".format(t, x, y, r))
    logger.log('')
    logger.log("Blob colours:")
    for reason, (_, name) in colours.items():
        logger.log('  {}: {}'.format(name, reason))

def _test(src, size, proximity, black, inverted, blur, mode, logger, params=None, create_new=True):
    """ ************** TEST **************** """

    if logger.depth() > 1:
        logger.push(context='blobs/_test')
    else:
        logger.push(context='_test')
    logger.log('')
    logger.log("Detecting blobs (create new {})".format(create_new))

    shrunk = contours.prepare_image(src, size, logger)
    if shrunk is None:
        logger.pop()
        return None
    logger.log("Proximity={}, blur={}".format(proximity, blur))
    if not create_new:
        params = logger.restore(file='blobs', ext='params')
        if params is None or params.source_file != src:
            create_new = True
    if create_new:
        if params is None:
            params = Params()
        params = contours.set_params(src, proximity, black, inverted, blur, mode, params)
        params = contours.find_contours(shrunk, params, logger)
        logger.save(params, file='blobs', ext='params')

    # do the actual detection
    params = find_blobs(params, logger=logger)
    logger.pop()
    return params


if __name__ == "__main__":

    #src = "/home/dave/precious/fellsafe/fellsafe-image/source/kilo-codes/test-alt-bits.png"
    src = '/home/dave/precious/fellsafe/fellsafe-image/media/kilo-codes/kilo-codes-near-150-257-263-380-436-647-688-710-777.jpg'
    #proximity = const.PROXIMITY_CLOSE
    proximity = const.PROXIMITY_FAR

    logger = utils.Logger('blobs.log', 'blobs/{}'.format(utils.image_folder(src)))

    _test(src, size=const.VIDEO_2K, proximity=proximity, black=const.BLACK_LEVEL[proximity],
          inverted=True, blur=3, mode=const.RADIUS_MODE_MEAN, logger=logger, create_new=False)
