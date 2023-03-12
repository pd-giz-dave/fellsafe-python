""" Find qualifying blobs
    Finds all contours then filters for the relevant ones
"""

import const
import utils
import shapes
import canvas
import contours

# ToDo: implement a mechanism to chop off tails - see e.g. in kilo-codes/near image @ code 436 bottom-left locator

class Params(contours.Params):

    # all these 'ness' parameters are in the range 0..1, where 0 is perfect and 1 is utter crap
    max_squareness = 0.5  # how close to square the bounding box has to be (0.5 is a 2:1 rectangle)
    max_wavyness   = 0.35  # how close to not wavy a contour perimeter must be
    max_offsetness = 0.03  # how close the centroid has to be to the enclosing box centre
    max_whiteness  = 0.4  # whiteness of the enclosing circle
    max_blackness  = 0.5  # whiteness of the enclosing box (0.5 is worst case for a 45 deg rotated sq)
    targets: [tuple] = None  # qualifying blobs


def filter_blobs(blobs: [shapes.Blob], params: Params, logger=None) -> [shapes.Blob]:
    """ filter out blobs that do no meet the target criteria,
        marks *all* blobs with an appropriate reject code and returns a list of good ones
        """

    if logger is not None:
        logger.push("filter_blobs")

    good_blobs = []                      # good blobs are accumulated in here
    for blob in blobs:
        while True:
            squareness = blob.get_squareness()
            if squareness > params.max_squareness:
                reason_code = const.REJECT_SQUARENESS
                break
            wavyness = blob.get_wavyness()
            if wavyness > params.max_wavyness:
                reason_code = const.REJECT_WAVYNESS
                break
            # now do the expensive checks, in cheapest first order
            offsetness = blob.get_offsetness()
            if offsetness > params.max_offsetness:
                reason_code = const.REJECT_OFFSETNESS
                break
            blackness = blob.get_blackness()
            if blackness > params.max_blackness:
                reason_code = const.REJECT_BLACKNESS
                break
            whiteness = blob.get_whiteness()
            if whiteness > params.max_whiteness:
                reason_code = const.REJECT_WHITENESS
                break
            # all filters passed
            reason_code = const.REJECT_NONE
            good_blobs.append(blob)
            break
        blob.rejected = reason_code
        # if logger is not None and reason_code != REJECT_NONE:
        #     logger.log("Rejected:{}, {}".format(reason_code, blob.show(verbose=True)))
    if logger is not None:
        rejected = len(blobs) - len(good_blobs)
        logger.log("Accepted blobs: {}, rejected {} ({:.2f}%) of {}".
                   format(len(good_blobs), rejected, (rejected / len(blobs)) * 100, len(blobs)))
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
        params.targets.append([circle.centre.x, circle.centre.y, circle.radius, blob])
    if logger is not None:
        show_result(params, logger)
        logger.pop()
    return params

def get_blobs(image, params, logger=None):
    params = contours.find_contours(image, params, logger)
    params = find_blobs(params, logger=logger)
    return params

def show_result(params, logger):
    # show what happened
    blobs   = params.blobs
    targets = params.targets
    buffer  = params.contours
    labels  = params.labels
    max_x, max_y = canvas.size(buffer)
    if params.box is not None:
        source_part = canvas.extract(params.source, params.box)
    else:
        source_part = params.source
    draw_bad = canvas.colourize(source_part)
    draw_good = canvas.colourize(source_part)
    colours = const.REJECT_COLOURS
    for x in range(max_x):
        for y in range(max_y):
            label = buffer[y, x]
            if label > 0:
                blob = labels.get_blob(label)
                if blob not in blobs:
                    continue  # this one was filtered out by contours - so do not draw again
                colour, _ = colours[blob.rejected]
                if blob.rejected == const.REJECT_NONE:
                    draw_good[y, x] = colour
                else:
                    draw_bad[y, x] = colour

    logger.draw(draw_good, file='blobs_accepted')
    logger.draw(draw_bad, file='blobs_rejected')

    # highlight our detections on the greyscale image
    draw = canvas.colourize(source_part)
    for (x, y, r, _) in targets:
        canvas.circle(draw, (x, y), r, const.GREEN, 1)
    logger.draw(draw, file='blobs')

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
    blobs.sort(key=lambda k: (k.external.top_left.x, k.external.top_left.y))

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
    logger.log("Detecting blobs")
    shrunk = contours.prepare_image(src, size, logger)
    if shrunk is None:
        logger.pop()
        return None
    logger.log("Proximity={}, blur={}".format(proximity, blur))
    if create_new:
        if params is None:
            params = Params()
        params = contours.set_params(src, proximity, black, inverted, blur, mode, params)
        params = contours.find_contours(shrunk, params, logger)
        logger.save(params, file='blobs', ext='params')
    else:
        params = logger.restore(file='blobs', ext='params')
    # do the actual detection
    params = find_blobs(params, logger=logger)
    logger.pop()
    return params


if __name__ == "__main__":

    #src = "/home/dave/precious/fellsafe/fellsafe-image/source/kilo-codes/test-alt-bits.png"
    src = '/home/dave/precious/fellsafe/fellsafe-image/media/kilo-codes/kilo-codes-near-150-257-263-380-436-647-688-710-777.jpg'
    #proximity = const.PROXIMITY_CLOSE
    proximity = const.PROXIMITY_FAR

    logger = utils.Logger('blobs.log', 'blobs')

    _test(src, size=const.VIDEO_2K, proximity=proximity, black=const.BLACK_LEVEL[proximity],
          inverted=True, blur=3, mode=const.RADIUS_MODE_MEAN, logger=logger, create_new=False)
