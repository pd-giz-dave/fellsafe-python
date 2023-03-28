""" Random useful stuff """

import os
import pathlib
import dill
import canvas

class Logger:
    """ crude logging system that saves to a file and prints to the console """

    def __init__(self, log_file: str, folder: str='.', context: str=None, prefix='  '):
        # make sure the destination folder exists
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        self.log_file = '{}/{}'.format(folder, log_file)
        self.log_handle = None
        if context is None:
            pathname, _ = os.path.splitext(log_file)
            _, context = os.path.split(pathname)
        self.context: [(str, str)] = [(context, folder)]
        self.prefix = prefix  # when logging multi-line messages prefix all lines except the first with this
        self.log('open {}'.format(self.log_file))
        self.count = 0  # incremented for every anonymous draw call and used as a file name suffix

    def __del__(self):
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    def close(self):
        self.log_handle.close()
        self.log_handle = None

    def log(self, msg: str=None):
        if msg is None:
            return
        if self.log_handle is None:
            self.log_handle = open(self.log_file, 'w')
        if msg == '\n':
            # caller just wants a blank line
            lines = ['']
        else:
            lines = msg.split('\n')
        for line, text in enumerate(lines):
            if line > 0:
                prefix = self.prefix
            else:
                prefix = ''
            log_msg = '{}: {}{}'.format(self.context[0][0], prefix, text)
            self.log_handle.write('{}\n'.format(log_msg))
            self.log_handle.flush()
            print(log_msg)

    def push(self, context=None, folder=None):
        parent_context = self.context[0][0]
        parent_folder  = self.context[0][1]
        if context is not None:
            parent_context = '{}/{}'.format(parent_context, context)
        if folder is not None:
            parent_folder = '{}/{}'.format(parent_folder, folder)
        self.context.insert(0, (parent_context, parent_folder))

    def pop(self):
        self.context.pop(0)

    def depth(self):
        return len(self.context)

    def draw(self, image, folder='', file='', ext='png', prefix=''):
        """ unload the given image into the given folder and file,
            folder, iff given, is a sub-folder to save it in (its created as required),
            the parent folder is that given when the logger was created,
            all images are saved as a sub-folder of the parent,
            file is the file name to use, blank==invent one,
            """

        filename = self.makepath(folder, file, ext)

        # save the image
        canvas.unload(image, filename)

        self.log('{}{}: image saved as: {}'.format(prefix, file, filename))
        
    def makepath(self, folder='', file='', ext='png'):
        """ make the required folder and return the fully qualified file name """
        
        if file == '':
            file = 'logger-{}'.format(self.count)
            self.count += 1

        if folder == '':
            folder = self.context[0][1]

        # make sure the destination folder exists
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

        filename = '{}/{}.{}'.format(folder, file, ext)
        return filename
        
    def save(self, object, folder='', file='', ext='object'):
        """ save the given object to the given file (so it can be restored later), 
            returns the fully qualified file name used
            """
        
        filename = self.makepath(folder, file, ext)
        dump_file = open(filename, 'wb')
        dill.dump(object, dump_file)
        dump_file.close()
        
        self.log('{}: object saved as: {}'.format(file, filename))
        
        return filename
        
    def restore(self, folder='', file='', ext='object', filename=None):
        """ restore a previously saved object, returning the object or None if it does not exist,
            NB: the logger context must be the same as when the object was saved if no filename is given
            """
        if filename is None:
            filename = self.makepath(folder, file, ext)
        try:
            dump_file = open(filename, 'rb')
            object = dill.load(dump_file)
            dump_file.close()
            self.log('Restored object from {}'.format(filename))
        except:
            object = None
            self.log('Restore of object from {} failed'.format(filename))
        return object
        
class Stats:
    """ log quantized value statistics """

    def __init__(self, buckets: int, value_range=(0,1), number_format='{:.2f}'):
        self.buckets       = buckets                          # how many buckets to quantize values into
        self.number_format = number_format                    # how to show numbers in logs
        self.value_min     = value_range[0]                   # expected minimum value
        self.value_max     = value_range[1]                   # expected maximum value
        self.value_span    = self.value_max - self.value_min  # nuff said?
        self.value_delta   = self.value_span / self.buckets   # value range in each bucket
        self.counts        = [0 for _ in range(buckets+1)]    # count of values in each bucket
        if self.value_span <= 0:
            raise Exception('value min must be >= 0 and value max must be > min (given {})'.format(value_range))

    def reset(self):
        """ clear stats to 0 """
        for bucket in range(len(self.counts)):
            self.counts[bucket] = 0

    def normalise(self, value):
        """ normalize the given value into the range 0..1 """
        norm  = min(max(value, self.value_min), self.value_max)  # ?..?     --> min..max
        norm -= self.value_min                                   # min..max --> 0..span
        norm /= self.value_span                                  # 0..span  --> 0..1
        return norm

    def bucket(self, value):
        """ return the bucket number for the given value """
        if value is None:
            return None
        norm   = self.normalise(value)
        bucket = int(norm * self.buckets)
        if bucket < 0 or bucket >= len(self.counts):
            raise Exception('bucket not in range 0..{} for value {}'.format(len(self.counts)-1, value))
        return bucket

    def span(self, bucket):
        """ return the value span (from(>=)-->to(<)) represented by the given bucket """
        span = (self.value_min + (bucket * self.value_delta), self.value_min + ((bucket+1) * self.value_delta))
        return span

    def count(self, value):
        bucket = self.bucket(value)
        if bucket is None:
            return
        self.counts[bucket] += 1

    def show(self, separator=', '):
        """ return a string showing the current stats """
        msg = ''
        for bucket, count in enumerate(self.counts):
            if count == 0:
                continue
            bucket_min, bucket_max = self.span(bucket)
            value_min = self.number_format.format(bucket_min)
            value_max = self.number_format.format(bucket_max)
            msg = '{}{}{}..{}-{}'.format(msg, separator, value_min, value_max, count)
        return msg[len(separator):]

class Frequencies:
    """ count the frequency of something within a set """

    def __init__(self, number_scale=100, number_format='{:.2f}%'):
        self.number_scale  = number_scale
        self.number_format = number_format
        self.reset()

    def reset(self):
        self.set   = {}
        self.size  = 0
        self.total = 0

    def count(self, item):
        if self.set.get(item) is None:
            # not seen this before
            self.set[item] = 1
            self.size += 1
        else:
            # seen before
            self.set[item] += 1
        self.total += 1

    def show(self, separator='\n'):
        """ return a string representing the counts """
        msg = ''
        for item, count in self.set.items():
            ratio = count / self.total
            freq = self.number_format.format(ratio * self.number_scale)
            msg = '{}{}{}: {} ({})'.format(msg, separator, item, count, freq)
        return msg[len(separator):]

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

def show_number(number, how='{:.2f}'):
    if number is None:
        return 'None'
    else:
        return how.format(number)

def show_point(point, how='{:.2f}'):
    return "({}, {})".format(show_number(point[0], how), show_number(point[1], how))

def distance(here, there) -> float:
    """ calculate the distance between the two given points,
        returns the distance squared,
        we use squared result so we do not need to do a square root (which is slow)
        """
    distance_x  = here[0] - there[0]
    distance_x *= distance_x
    distance_y  = here[1] - there[1]
    distance_y *= distance_y
    distance    = distance_x + distance_y
    return distance

def ratio(a, b):
    """ return the ratio of a/b or b/a whichever is smaller,
        this provides a measure of how close to each other the two numbers are,
        the nearer to 1 the ratio is the closer the numbers are to each other
        """
    if a == 0 and b == 0:
        # special case, we're looking for 'sameness' and 0 is the same as 0
        return 1
    result = min(a, b) / max(a, b)  # range 0..1, 0=distant, 1=close
    return result

def line(x0: int, y0: int, x1: int, y1: int) -> [(int, int)]:
    """ return a list of points that represent all pixels between x0,y0 and x1,y1 in the order x0,x0 -> x1,y1 """

    # see https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm for the algorithm implemented here

    points = []

    def line_low(x0, y0, x1, y1):
        # x0 <= x1 and slope <=1 guaranteed to get here
        dx = x1 - x0
        dy = y1 - y0
        yi = 1
        if dy < 0:
            yi = -1
            dy = -dy
        D = (2 * dy) - dx
        y = y0
        for x in range(x0, x1+1):
            points.append((x, y))
            if D > 0:
                y = y + yi
                D = D + (2 * (dy - dx))
            else:
                D = D + 2 * dy
    def line_high(x0, y0, x1, y1):
        # y0 <= y1 and slope <=1 guaranteed to get here
        dx = x1 - x0
        dy = y1 - y0
        xi = 1
        if dx < 0:
            xi = -1
            dx = -dx
        D = (2 * dx) - dy
        x = x0
        for y in range(y0, y1+1):
            points.append((x, y))
            if D > 0:
                x = x + xi
                D = D + (2 * (dx - dy))
            else:
                D = D + 2 * dx

    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            line_low(x1, y1, x0, y0)
            points.reverse()
        else:
            line_low(x0, y0, x1, y1)
    else:
        if y0 > y1:
            line_high(x1, y1, x0, y0)
            points.reverse()
        else:
            line_high(x0, y0, x1, y1)
    return points

def intersection(line1, line2):
    """ find the intersection point between line1 and line2, each line is a tuple pair of start/end points """
    l1_start, l1_end = line1
    l2_start, l2_end = line2
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
    x1 = l1_start[0]
    y1 = l1_start[1]
    x2 = l1_end[0]
    y2 = l1_end[1]
    x3 = l2_start[0]
    y3 = l2_start[1]
    x4 = l2_end[0]
    y4 = l2_end[1]
    x1y2 = x1 * y2
    y1x2 = y1 * x2
    x3y4 = x3 * y4
    y3x4 = y3 * x4
    x3_x4 = x3 - x4
    x1_x2 = x1 - x2
    y3_y4 = y3 - y4
    y1_y2 = y1 - y2
    x1y2_y1x2 = x1y2 - y1x2
    x3y4_y3x4 = x3y4 - y3x4
    divisor = (x1_x2 * y3_y4) - (y1_y2 * x3_x4)
    Px = ((x1y2_y1x2 * x3_x4) - (x1_x2 * x3y4_y3x4)) / divisor
    Py = ((x1y2_y1x2 * y3_y4) - (y1_y2 * x3y4_y3x4)) / divisor
    return Px, Py

def extend(start, end, box):
    """ extend the given line segment such that its ends meet the box walls """

    (xmin, ymin), (xmax, ymax) = box
    x1 = start[0]  # do it like this 'cos start/end may be a tuple with more then 2 fields
    y1 = start[1]  # ..
    x2 = end[0]    # ....
    y2 = end[1]    # ......

    # sanity check
    if min(x1, x2) < xmin or max(x1, x2) > xmax or min(y1, y2) < ymin or max(y1, y2) > ymax:
        raise Exception('line segment {} -> {} not within box {} x {}'.
                        format(show_point(start), show_point(end), show_point(box[0]), show_point(box[1])))

    # deal with vertical and horizontal
    if x1 == x2:
        # vertical line
        if y1 < y2:
            # heading to the bottom
            return (x1, ymin), (x2, ymax)
        elif y1 > y2:
            # heading to the top
            return (x1, ymax), (x2, ymin)
    elif y1 == y2:
        # horizontal line
        if x1 < x2:
            # heading to the right
            return (xmin, y1), (xmax, y2)
        else:
            # heading to the left
            return (xmax, y1), (xmin, y2)

    def extend_down(x1, y1, x2, y2):
        # heading towards bottom-right
        # we use similar triangles to work it out, like so:
        #  _____
        #  ^ ^ *                               Line segment is x--x
        #  | | |\                              Extending upwards to X-min and downwards to X-max reaches *
        #  D'| | \                             By similar triangles A/B = E/F = E'/F' = C/D = C'/D'
        #  | | |C'\           X-max
        #  v | +---\------------+    Y-min
        #    D |    \           |
        #    | |  C  \  A       |
        #    v +------X--+      |
        #      |       \ | B    |
        #      |        \|      |
        #      |         X------+ ^
        #      |          \  E  | |
        #      |           \    | F
        #      +------------\---+ | ^ Y-max
        #    X-min           \E'| | |
        #                     \ | | F'
        #                      \| | |
        #                       * v v
        #                       _____
        dx = x2 - x1
        dy = y2 - y1
        slope = dx / dy
        # dx = A, dy = B, E = xmax-x2, so A/B = E/F, slope = A/B, F * slope = E, F = E / slope
        y_at_xmax = y2 + ((xmax - x2) / slope)
        if y_at_xmax > ymax:
            # overshot, so back up to ymax, F' = (y2 + F) - ymax, E'/F' = A/B, E' = (A/B)*F'
            x_at_ymax = xmax - slope * (y_at_xmax - ymax)
            end = x_at_ymax, ymax
        else:
            end = xmax, y_at_xmax
        # C = x1-xmin, A/B = C/D, D = C / (A/b)
        y_at_xmin = y1 - ((x1 - xmin) / slope)
        if y_at_xmin < xmin:
            # overshot, so back up to ymin (NB: y_at_xmin may go -ve, hence the abs() below)
            x_at_ymin = xmin + (slope * abs(y_at_xmin - xmin))  # A/B = C'/D', D'= D - X-min, so C' = (A/B) * D'
            start = x_at_ymin, ymin
        else:
            start = xmin, y_at_xmin
        return start, end

    def extend_up(x1, y1, x2, y2):
        # heading towards top-right
        # we use similar triangles to work it out, like so:
        #  Line segment is x--x
        #  Extending upwards to X-min and downwards to X-max reaches *
        #  By similar triangles A/B = C/D = C'/D' = E/F = E'/F'
        #                              _____
        #                              * ^ ^
        #                             /| | |
        #                            / | | D'
        #         X-min             /  | D |
        #    Y-min  +--------------/---+ | v
        #           |             / C' | |
        #           |            /     | |
        #           |           /  C   | |
        #           |          X-------+ V
        #           |         /|       |
        #           |        / | B     |
        #           |  E    /  |       |
        #         ^ +------X---+       |
        #         | |     /  A         |
        #         | | E' /             |
        #       ^ | +---/--------------+  Y-max
        #       | F |  /             X-max
        #       F'| | /
        #       | | |/
        #       v v *
        #       -----
        dx = x2 - x1
        dy = y1 - y2
        slope = dx / dy
        # dx = A, dy = B, C = X-max - x2, A/B = C/D, slope = A/B, slope * D = C, D = C / slope
        y_at_xmax = y2 - ((xmax - x2) / slope)
        if y_at_xmax < ymin:  # NB: y_at_xmax may go -ve, hence the abs() below
            # overshot, back up to ymin, D' = y_at_xmax - ymin, C'/D' = A/B, C' = (A/B)*D'
            x_at_ymin = xmax - (slope * abs(y_at_xmax - ymin))
            end = x_at_ymin, ymin
        else:
            end = xmax, y_at_xmax
        # E = x1 - xmin, E/F = A/B, F * (A/B) = E, F = E / (A/B)
        y_at_xmin = y1 + ((x1 - xmin) / slope)
        if y_at_xmin > ymax:
            # overshot, back up to ymax, F' = F - ymax, E'/F' = A/B, E' = (A/B) * F'
            x_at_ymax = xmin + (slope * (y_at_xmin - ymax))
            start = x_at_ymax, ymax
        else:
            start = xmin, y_at_xmin
        return start, end

    # which corner is the line heading towards?
    dx = x2 - x1
    dy = y2 - y1
    if dx > 0 and dy > 0:
        # heading towards bottom-right
        return extend_down(x1, y1, x2, y2)
    elif dx < 0 and dy < 0:
        # heading towards top-left - same as bottom-right with start,end reversed
        start, end = extend_down(x2, y2, x1, y1)
        return end, start
    elif dx < 0 and dy > 0:
        # heading towards bottom-left - same as top-right with start,end reversed
        start, end = extend_up(x2, y2, x1, y1)
        return end, start
    else:  # dx > 0 and dy < 0
        # heading towards top-right
        return extend_up(x1, y1, x2, y2)


##################################################################
########################### TESTING ##############################
##################################################################

def test_line(x0, y0, x1, y1, name):
    points = line(x0, y0, x1, y1)
    if points[0] == (x0, y0) and points[-1] == (x1, y1):
        prefix = '____pass____'
    else:
        prefix = '****FAIL****'
    logger.log('  {}: {:4} {:3}x, {:3}y  -->  {:3}x, {:3}y = {}'.format(prefix, name, x0, y0, x1, y1, points))

def test_extend(start, end, box, name, expect_start, expect_end, error_allowed_squared=1):
    extended_start, extended_end = extend(start, end, box)
    error_start_x = extended_start[0] - expect_start[0]
    error_start_x *= error_start_x
    error_start_y = extended_start[1] - expect_start[1]
    error_start_y *= error_start_y
    error_end_x = extended_end[0] - expect_end[0]
    error_end_x *= error_end_x
    error_end_y = extended_end[1] - expect_end[1]
    error_end_y *= error_end_y
    if error_start_x <= error_allowed_squared and \
       error_end_x   <= error_allowed_squared and \
       error_start_y <= error_allowed_squared and \
       error_end_y   <= error_allowed_squared:
        prefix = '____pass____'
    else:
        prefix = '****FAIL****'
    logger.log('  {}: {} {} -> {} extends to {} -> {} for box {} x {} (expect {} -> {})'.
               format(prefix, name,
                      show_point(start), show_point(end),
                      show_point(extended_start), show_point(extended_end),
                      show_point(box[0]), show_point(box[1]),
                      show_point(expect_start), show_point(expect_end)))

if __name__ == "__main__":
    """ testing """

    logger = Logger('utils.log', 'utils')
    logger.log('Utils')

    # region test_line...
    logger.log('')
    logger.log('Test line()...')
    test_line(0,0,   0,-10, 'N')
    test_line(0,0,   3,-10, 'NNE')
    test_line(0,0,  10,-10, 'NE')
    test_line(0,0,  10, -3, 'ENE')
    test_line(0,0,  10,  0, 'E')
    test_line(0,0,  10,  3, 'ESE')
    test_line(0,0,  10, 10, 'SE')
    test_line(0,0,   3, 10, 'SSE')
    test_line(0,0,   0, 10, 'S')
    test_line(0,0,  -3, 10, 'SSW')
    test_line(0,0, -10,-10, 'SW')
    test_line(0,0, -10, -3, 'WSW')
    test_line(0,0, -10,  0, 'W')
    test_line(0,0, -10, -3, 'WNW')
    test_line(0,0, -10,-10, 'NW')
    test_line(0,0,  -3,-10, 'NNW')
    # endregion

    # region test_extend...
    logger.log('')
    logger.log('Test extend()...')
    tl = (5,5)
    tr = (14,5)
    br = (14,14)
    bl = (5,14)
    box = (tl,br)
    test_extend((6,6), (7,7), box, 'diagonal++', tl, br)
    test_extend((7,7), (6,6), box, 'diagonal--', br, tl)
    test_extend((12,7), (13,6), box, 'diagonal+-', bl, tr)
    test_extend((13,6), (12,7), box, 'diagonal-+', tr, bl)
    test_extend((6,9), (6,10), box, 'vertical+', (6,5),(6,14))
    test_extend((6, 10), (6, 9), box, 'vertical-', (6, 14), (6, 5))
    test_extend((9, 10), (11, 10), box, 'horizontal+', (5, 10), (14, 10))
    test_extend((11, 10), (9, 10), box, 'horizontal-', (14, 10), (5, 10))
    test_extend((6,12), (8,13), box, 'shallow slope++low', (5,11), (9,14))
    test_extend((8,13), (6,12), box, 'shallow slope--low', (9,14), (5,11))
    test_extend((6, 13), (8, 12), box, 'shallow slope+-low', (5, 13), (14, 9))
    test_extend((6, 7), (8, 8), box, 'shallow slope++high', (5, 6), (14, 11))
    test_extend((8, 8), (6, 7), box, 'shallow slope--high', (14, 11), (5, 6))
    test_extend((6, 7), (7, 10), box, 'steep slope++high', (5, 5), (8, 14))
    test_extend((13,7), (12,5), box, 'steep slope--high', (14,9), (12,5))
    test_extend((6,7), (7,5), box, 'steep slope+-high', (5,9), (7,5))
    test_extend((7,5), (6,7), box, 'steep slope-+high', (7,5), (5,9))
    # endregion

    logger.log('...done')
