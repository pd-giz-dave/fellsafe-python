""" Find qualifying blobs
    Finds all contours then filters for the relevant ones
"""
import math
import random

import const
import utils
import shapes
import canvas
import contours

class Params(contours.Params):

    depinch: bool = True  # iff True de-pinch blobs by splitting them

    # pre-filter params
    min_area: float = 4  # min number of pixels inside the contour box
    max_size: float = 128  # max number of pixels across the contour box
    max_perimeter = max_size * 4  # anything bigger than this creates performance problems for finding opposites
    max_internals: int = 1  # max number of internal contours that is tolerated to be a blob
    max_depth: int = 4  # max recursion depth when splitting blobs

    # all these 'ness' parameters are in the range 0..1, where 0 is perfect and 1 is utter crap
    # there are three values for each corresponding to small, medium and large blobs
    max_splitness  = ( 10,  10,  10)     # how many times a blob is allowed to split
    max_sameness   = (1.0, 0.8, 0.8)     # how much contrast we want across the contour edge (0==lots, 1==none)
    max_thickness  = (1.0, 0.9, 0.7)     # how similar the thickness has to be to the enclosing circle diameter
    max_squareness = (0.5, 0.5, 0.5)     # how close to square the bounding box has to be (0.5 is a 2:1 rectangle)
    max_wavyness   = (1.0, 0.5, 0.25)    # how close to not wavy a contour perimeter must be
    max_offsetness = (1.0, 0.05, 0.05)   # how close the centroid has to be to the enclosing box centre
    max_whiteness  = (0.3, 0.3, 0.3)     # whiteness of the enclosing circle
    max_blackness  = (0.6, 0.6, 0.6)     # whiteness of the enclosing box (0.5 is worst case for a 45 deg rotated sq)
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
    NOWHERE    = '?'

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
                 SOUTH_WEST: (NORTH_EAST, NORTH     , EAST      ),
                 NOWHERE   : (NOWHERE,)}

    # 'outside' is the direction that represents outside a contour, contours are constructed in a clockwise
    # direction, so 'outside' is 90 degrees left of whatever direction it is going in,
    # the direction is given as an x,y offset from the contour point
    # NB: reversing the sign on these offsets gets the 'inside' of the contour
    OUTSIDE = {NORTH     : (-1,  0),  # WEST
               SOUTH     : (+1,  0),  # EAST
               EAST      : ( 0, -1),  # NORTH
               WEST      : ( 0, +1),  # SOUTH
               NORTH_EAST: (-1, -1),  # NORTH_WEST
               NORTH_WEST: (-1, +1),  # SOUTH_WEST
               SOUTH_EAST: (+1, -1),  # NORTH_EAST
               SOUTH_WEST: (+1, +1),  # SOUTH_EAST
               NOWHERE   : ( 0,  0)}  # NOWHERE
    # endregion

    SMALL_BLOB_SIZE = 7*7    # threshold for a small blob in pixels (area of its contour box)
    LARGE_BLOB_SIZE = 12*12  # ditto for a big one

    OUTSIDE_WIDTH = max((const.BLUR_KERNEL_SIZE >> 1), 2)  # how far outside the contour to look for luminance contrast
    INSIDE_WIDTH  = max((const.BLUR_KERNEL_SIZE >> 1), 1)  # how far inside the contour to look for luminance contrast

    PINCHPOINT_MIN_VARIATION  = 0.1  # minimum opposites 'coefficient-of-variation' to trigger finding a pinch-point
    PINCHPOINT_MIN_VARIATION *= PINCHPOINT_MIN_VARIATION  # squared

    PINCHPOINT_MAX_DISTANCE  = 0.5  # max fraction of max opposite distance considered to be a pinch-point
                                    # bigger makes more pinch-points
    PINCHPOINT_MAX_DISTANCE *= PINCHPOINT_MAX_DISTANCE  # squared
    PINCHPOINT_MIN_LENGTH    = 0.1  # min fraction of contour length considered to be a pinch-point
                                    # bigger makes less pinch-points
    PINCHPOINT_MIN_SAMPLES   = max(int(1/PINCHPOINT_MIN_LENGTH)+1,16)  # minimum samples to trigger a pinch-point search
                               # above must be at least 4 to make sure a split always creates a distinct contour
                               # bigger makes less pinch-points
                               # ToDo: The max() 2nd param is really dependent on the image size, a fixed num is a hack
    PINCHPOINT_MIN_RATIO     = 2.0  # min length/distance ratio (pi/2==semi-circle-ish, 3==square-ish)
                                    # this must be greater than 1 to ensure there is an enclosed area

    def __init__(self, blob: shapes.Blob, source, logger=None):
        self.blob        = blob
        self.source      = source  # the source greyscale image the blob was found in
        self.logger      = logger
        self.perimeter   = self.blob.get_points()  # NB: need all contour points so can calculate accurate direction
        self.area, self.size = Target.get_size(blob)
        self.direction  = None  # the 'direction' of each contour sample
        self.opposites  = None  # the 'best' opposite sample for each sample
        self.pinchpoint = None  # pinch-point of the contour (if there is one), its a list of 0 or 1 items
        self.variation  = None  # the opposites coefficient-of-variation (squared)

    @staticmethod
    def get_size(blob):
        """ get the area and size of the given blob """
        width, height = blob.get_size()
        area = width * height
        if area <= Target.SMALL_BLOB_SIZE:
            size = 0
        elif area < Target.LARGE_BLOB_SIZE:
            size = 2
        else:
            size = 1
        return area, size

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
                    # this means we've only got a single sample or both samples are in the same place,
                    # this is legit for the last sample as its the same as the first, otherwise there's a screw up!
                    if max_samples == 1:
                        # only one sample, so no direction
                        direction = Target.NOWHERE
                    elif sample == (max_samples - 1):
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
        midway      = max_samples >> 1  # used to discriminate same distance
        for sample in range(max_samples):
            here_direction  = self.direction[sample]
            here_position   = self.perimeter[sample]
            best_distance   = None
            best_opposite   = None
            best_opposition = None
            best_span       = None
            for opposite in range(max_samples):
                opposition = Target.is_opposite(here_direction, self.direction[opposite])
                if opposition == Target.NOT_OPPOSITE:
                    continue
                if opposite < sample:
                    span = sample - opposite
                else:
                    span = opposite - sample
                span = utils.ratio(span, midway)
                distance = utils.distance(here_position, self.perimeter[opposite])
                if best_distance is None:
                    best_distance = distance
                    best_opposite = opposite
                    best_opposition = opposition
                    best_span = span
                elif distance < best_distance:
                    best_distance = distance
                    best_opposite = opposite
                    best_opposition = opposition
                    best_span = span
                elif distance > best_distance:
                    continue
                elif opposition > best_opposition:
                    best_distance = distance
                    best_opposite = opposite
                    best_opposition = opposition
                    best_span = span
                elif opposition < best_opposition:
                    continue
                elif span > best_span:
                    best_distance = distance
                    best_opposite = opposite
                    best_opposition = opposition
                    best_span = span
                elif span < best_span:
                    continue
                else:
                    # same distance, same opposition, same span, now what?
                    continue
            # each opposite pair may also be found in opposite directions (due to symmetry, a->b == b->a)
            # that's benign but confusing when viewing logs and could be removed, but its not worth the effort
            if best_opposite is None:
                raise Exception('Cannot find unique opposite sample for {}'.format(sample))
            self.opposites.append((best_opposite, best_distance))
        return self.opposites

    def get_pinchpoint(self):
        """ get the worst pinch-point for the contour (if it has one),
            a pinch-point is characterised by two opposite points with a distance below some threshold
            and a length above some threshold, each has two attributes: the head length and the tail length
            the head length is the length after the tail has been removed,
            the tail length is the number of samples between the opposites,
            the worst pinch-point has the shortest distance with the biggest head or tail
            """
        if self.pinchpoint is not None:
            return self.pinchpoint, self.variation

        self.pinchpoint = []
        self.variation  = None
        if len(self.get_opposites()) < Target.PINCHPOINT_MIN_SAMPLES:  # NB: also triggers opposites calc
            # not enough samples to be meaningful
            return self.pinchpoint, self.variation

        # find distance/length limits
        max_distance = None
        min_distance = None
        avg_distance = 0
        for _, distance in self.opposites:
            avg_distance += distance
            if max_distance is None or distance > max_distance:
                max_distance = distance
            if min_distance is None or distance < min_distance:
                min_distance = distance
        avg_distance /= len(self.opposites)

        # find the coefficient of variation (see https://en.wikipedia.org/wiki/Coefficient_of_variation)
        deviation = 0
        for _, distance in self.opposites:
            diff = distance - avg_distance
            deviation += (diff * diff)
        deviation /= len(self.opposites)               # the actual variation is the square root of this
        mean = avg_distance * avg_distance             # get the mean distance squared
        self.variation = deviation / mean              # determine the 'coefficient-of-variation' (0=none, 1+=lots)
        if self.variation < Target.PINCHPOINT_MIN_VARIATION:
            # not enough variation for a pinch-point to exist
            return self.pinchpoint, self.variation

        # build list of qualifying lengths for each qualifying distance
        max_distance *= Target.PINCHPOINT_MAX_DISTANCE
        max_distance  = max(max_distance, min_distance)
        max_length    = len(self.opposites)
        min_length    = max(max_length * Target.PINCHPOINT_MIN_LENGTH, Target.PINCHPOINT_MIN_SAMPLES)
        distances = {}
        for start, (finish, distance) in enumerate(self.opposites):
            if distance > max_distance:
                # too far apart to be considered a pinch-point
                continue
            if start <= finish:
                tail = finish - start + 1
            else:
                tail = start - finish + 1
            head = max_length - tail + 2  # +2 'cos we put the start/finish samples in both
            length = min(head, tail)
            if length < min_length:
                # too small to be considered a pinch-point
                continue
            length *= length  # square it to be units compatible with distance
            if length < (distance * Target.PINCHPOINT_MIN_RATIO):
                # insufficient enclosed area
                continue
            if distance not in distances:
                distances[distance] = []
            distances[distance].append((length, start, finish, head, tail, distance))

        # find the longest with the shortest distance
        pinchpoints = [distance for distance in distances.items()]
        pinchpoints.sort(key=lambda k: k[0])
        for distance, pinchpoint in pinchpoints:
            pinchpoint.sort(key=lambda k: k[0], reverse=True)

        if len(pinchpoints) == 0 or len(pinchpoints[0][1]) == 0:
            # nothing qualifies
            return self.pinchpoint, self.variation

        self.pinchpoint = [pinchpoints[0][1][0]]
        return self.pinchpoint, self.variation

    def get_thickness(self):
        """ the thickness is a measure of how the opposites distances relate to the enclosing box """
        if self.blob.thickness is not None:
            return self.blob.thickness

        big_acc = 0
        big_samples = 0
        small_acc = 0
        small_samples = 0
        for (_, distance) in self.get_opposites():
            if distance > 2:
                # got an opposite that is actually separated
                big_acc += math.sqrt(distance)
                big_samples += 1
            else:
                # got an out-and-back sample
                small_acc += math.sqrt(distance)
                small_samples += 1
        if big_samples == 0:
            # this means we've just got a long thin strip (out-and-back)
            thickness = small_acc / small_samples  # average distance of opposites where there is no separation
        else:
            # we've actually got a separation
            thickness = big_acc / big_samples  # average distance of opposites where there is a separation
        size  = min(self.blob.get_size())
        self.blob.thickness = 1 - utils.ratio(thickness, size)  # 0==good, 1==crap

        return self.blob.thickness

    def get_sameness(self):
        """ get a luminance change measure across the edge of the contour """
        if self.blob.sameness is not None:
            return self.blob.sameness

        inside_level    = 0
        outside_level   = 0
        inside_samples  = 0
        outside_samples = 0
        for sample, (x, y) in enumerate(self.perimeter):
            direction = self.get_direction(sample)
            dx, dy = Target.OUTSIDE[direction]
            pixel = canvas.getpixel(self.source, x - (dx * Target.INSIDE_WIDTH), y - (dy * Target.INSIDE_WIDTH))
            if pixel is not None:
                inside_level   += pixel
                inside_samples += 1
            pixel = canvas.getpixel(self.source, x + (dx * Target.OUTSIDE_WIDTH), y + (dy * Target.OUTSIDE_WIDTH))
            if pixel is not None:
                outside_level += pixel
                outside_samples += 1
        if inside_samples > 0:
            inside_level /= inside_samples
        if outside_samples > 0:
            outside_level /= outside_samples
        contrast = max(outside_level - inside_level, 0)  # this should be +ve as inside should black
        self.blob.sameness = 1 - (contrast / const.MAX_LUMINANCE)  # range 0..1, smaller number is bigger change
        return self.blob.sameness


def split_blob(blob: shapes.Blob, params: Params, logger=None, depth=0) -> [Target]:
    """ split a blob that has a pinchpoint,
        returns a list of split targets, empty if target is junk, single element if no pinchpoints,
        otherwise an element for pinchpoint split
        """

    target = Target(blob, params.source, logger)

    if not params.depinch:
        # being told not to do it
        return [target]

    if depth > params.max_depth:
        # gone too deep
        return [target]

    pinchpoint, variation = target.get_pinchpoint()
    if len(pinchpoint) == 0:
        # no pinchpoint here
        if logger is not None:
            if variation is not None:
                indent = '  '
                prefix = indent * 2 * depth
                logger.log('{}Blob {}:'.format(prefix, blob))
                logger.log('{}{}has no pinch-point (cv={})'.format(prefix, indent, utils.show_number(variation)))
        return [target]

    _, start, finish, head, tail, distance = pinchpoint[0]

    head_blob = blob.extract(start, finish)
    tail_blob = blob.extract(finish, start)
    if logger is not None:
        indent = '  '
        prefix = indent * 2 * depth
        logger.log('{}Blob {}:'.format(prefix, blob))
        logger.log('{}{}splitting at worst pinch-point {} (cv={}) into:'.format(prefix, indent,
                                                                                pinchpoint,
                                                                                utils.show_number(variation)))
        logger.log('{}{}{}{}'.format(prefix, indent, indent, head_blob))
        logger.log('{}{}{}{}'.format(prefix, indent, indent, tail_blob))

    head_targets = split_blob(head_blob, params, logger, depth+1)
    tail_targets = split_blob(tail_blob, params, logger, depth+1)

    blob.rejected = const.REJECT_SPLIT

    return head_targets + tail_targets

def filter_blobs(blobs: [shapes.Blob], params: Params, logger=None) -> [shapes.Blob]:
    """ filter out blobs that do no meet the target criteria,
        marks *all* blobs with an appropriate reject code and returns a list of good ones and all blobs
        """

    if logger is not None:
        logger.push("filter_blobs")

    good_blobs = []                      # good blobs are accumulated in here
    all_blobs  = []                      # this is everything, including splits
    for blob in blobs:
        # do these before attempt split
        if len(blob.get_points()) > params.max_perimeter:
            blob.rejected = const.REJECT_TOO_BIG
            all_blobs.append(blob)
            continue
        targets = split_blob(blob, params, logger)
        blob.splitness = len(targets)
        _, size = Target.get_size(blob)
        if blob.splitness > params.max_splitness[size]:
            blob.rejected = const.REJECT_SPLITS
            all_blobs.append(blob)
            continue
        if logger is not None:
            if blob.splitness > 1:
                logger.log('Blob label:{} has split into {} parts'.format(blob.label, blob.splitness))
        for target in targets:
            blob = target.blob
            all_blobs.append(blob)
            # do the cheap filtering first
            if len(blob.internal) > params.max_internals:
                blob.rejected = const.REJECT_INTERNALS
                continue
            if target.area < params.min_area:
                blob.rejected = const.REJECT_TOO_SMALL
                continue
            if max(blob.get_size()) > params.max_size:
                blob.rejected = const.REJECT_TOO_BIG
                continue
            wavyness = blob.get_wavyness()
            if wavyness > params.max_wavyness[target.size]:
                blob.rejected = const.REJECT_WAVYNESS
                continue
            sameness = target.get_sameness()
            if sameness > params.max_sameness[target.size]:
                blob.rejected = const.REJECT_SAMENESS
                continue
            thickness = target.get_thickness()
            if thickness > params.max_thickness[target.size]:
                blob.rejected = const.REJECT_THICKNESS
                continue
            offsetness = blob.get_offsetness()
            if offsetness > params.max_offsetness[target.size]:
                # we've got a potential saucepan
                blob.rejected = const.REJECT_OFFSETNESS
                continue
            squareness = blob.get_squareness()
            if squareness > params.max_squareness[target.size]:
                blob.rejected = const.REJECT_SQUARENESS
                continue
            blackness = blob.get_blackness()
            if blackness > params.max_blackness[target.size]:
                blob.rejected = const.REJECT_BLACKNESS
                continue
            whiteness = blob.get_whiteness()
            if whiteness > params.max_whiteness[target.size]:
                blob.rejected = const.REJECT_WHITENESS
                continue
            # all filters passed
            blob.rejected = const.REJECT_NONE
            good_blobs.append(blob)

    if logger is not None:
        if len(all_blobs) > 0:
            rejected = len(all_blobs) - len(good_blobs)
            rate     = (rejected / len(all_blobs))
        else:
            rejected = 'ALL'
            rate     = 1
        logger.log("Accepted blobs: {}, rejected {} ({:.2f}%) of {}".
                   format(len(good_blobs), rejected, rate * 100, len(all_blobs)))
        logger.pop()

    return good_blobs, all_blobs

def find_blobs(params, logger=None) -> Params:
    """ find targets in the detected blobs """
    if logger is not None:
        logger.push("find_blobs")
        logger.log('')
    passed, params.blobs = filter_blobs(params.blobs, params, logger=logger)  # NB: updating the blob list
    params.targets = []
    for blob in passed:
        circle = blob.get_enclosing_circle()
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

    colours = const.REJECT_COLOURS

    # draw rejected blobs (this includes splits)
    blobs    = params.blobs      # rejects will be in here
    draw_bad = canvas.colourize(source_part)
    for blob in blobs:
        if blob.rejected == const.REJECT_NONE:
            # good ones are done elsewhere
            continue
        colour, _ = colours[blob.rejected]
        for (x, y) in blob.external.points:
            draw_bad[y, x] = colour
    logger.draw(draw_bad, file='blobs_rejected')

    # draw accepted blobs (this includes splits) in random colours (so can see split boundaries)
    colours = (const.OLIVE, const.GREEN, const.BLUE, const.YELLOW, const.PURPLE, const.CYAN, const.ORANGE, const.PINK)
    draw_good = canvas.colourize(source_part)
    for (_, _, _, blob) in params.targets:
        colour = colours[random.randrange(0, len(colours))]
        for (x, y) in blob.external.points:
            draw_good[y, x] = colour
    logger.draw(draw_good, file='blobs_accepted')

    BUCKETS    = 20  # how many quantisation buckets in the stats
    all_stats  = None
    good_stats = None

    reject_stats = utils.Frequencies()
    good_blobs = 0
    blobs.sort(key=lambda k: (k.external.get_enclosing_box()[0][0], k.external.get_enclosing_box()[0][1]))

    def update_count(stats, value):
        stats.count(value)

    def log_stats(name, stats):
        msg = stats.show()
        logger.log('  {:10}: {}'.format(name, msg))

    used_colours = {}  # collect reject colours actually used (for the log)
    logger.log('')
    logger.log("All accepted blobs:")
    for b, blob in enumerate(blobs):
        if blob.rejected in const.REJECT_COLOURS:
            used_colours[blob.rejected] = const.REJECT_COLOURS[blob.rejected]  # for log hints later
        reject_stats.count(blob.rejected)
        blob_stats = blob.get_quality_stats()
        if all_stats is None:
            # create stats tables now we know how many we need
            all_stats  = [utils.Stats(BUCKETS) for _ in range(len(blob_stats))]
            good_stats = [utils.Stats(BUCKETS) for _ in range(len(blob_stats))]
            stat_names = [stat_name for _, stat_name in blob_stats]
        # update the stats
        for s, (blob_stat, _) in enumerate(blob_stats):
            update_count(all_stats[s], blob_stat)
            if blob.rejected != const.REJECT_NONE:
                continue
            good_blobs += 1
            update_count(good_stats[s], blob_stat)
        logger.log("  {}: {}".format(b, blob.show(verbose=True)))

    logger.log('')
    logger.log('All rejected blobs')
    for b, blob in enumerate(blobs):
        if blob.rejected == const.REJECT_NONE:
            continue
        logger.log("  {}: {}".format(b, blob.show(verbose=True)))

    # show stats
    logger.log('')
    logger.log("All reject frequencies (across {} blobs):".format(len(blobs)))
    logger.log('  ' + reject_stats.show())
    logger.log('')
    logger.log("All blobs stats (across {} blobs):".format(len(blobs)))
    for s, stats in enumerate(all_stats):
        log_stats(stat_names[s], stats)
    logger.log('')
    logger.log("All accepted blobs stats (across {} blobs):".format(good_blobs))
    for s, stats in enumerate(good_stats):
        log_stats(stat_names[s], stats)

    logger.log('')
    logger.log("All detected targets:")
    params.targets.sort(key=lambda k: (k[0], k[1]))
    for t, (x, y, r, target) in enumerate(params.targets):
        logger.log("  {}: centre: {:.2f}, {:.2f}  radius: {:.2f}".format(t, x, y, r))
    logger.log('')
    logger.log("Blob colours:")
    for reason, (_, name) in used_colours.items():
        logger.log('  {}: {}'.format(name, reason))

def _test(src, size, proximity, black, inverted, blur, mode, logger, params=None, create_new=True):
    """ ************** TEST **************** """

    if logger.depth() > 1:
        logger.push(context='blobs/_test')
    else:
        logger.push(context='_test')
    logger.log('')
    logger.log("Detecting blobs (create new {})".format(create_new))

    shrunk = canvas.prepare(src, size, logger)
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

    #src = "/home/dave/precious/fellsafe/fellsafe-image/source/kilo-codes/codes/test-alt-bits.png"
    src = '/home/dave/precious/fellsafe/fellsafe-image/media/kilo-codes/kilo-codes-close-150-257-263-380-436-647-688-710-777.jpg'
    #proximity = const.PROXIMITY_CLOSE
    proximity = const.PROXIMITY_FAR

    logger = utils.Logger('blobs.log', 'blobs/{}'.format(utils.image_folder(src)))

    _test(src, size=const.VIDEO_2K, proximity=proximity, black=const.BLACK_LEVEL[proximity],
          inverted=True, blur=const.BLUR_KERNEL_SIZE, mode=const.RADIUS_MODE_MEAN, logger=logger, create_new=False)
