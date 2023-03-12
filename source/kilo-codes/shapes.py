""" Classes that provides properties for our contours """

import math
import const

def count_pixels(image, perimeter):
    """ count how many pixels in the area bounded by perimeter within image are black and how many are white,
        perimeter is a list of x co-ords and the max/min y at that x,
        perimeter points are inside the area,
        co-ords outside the image bounds are ignored,
        """
    limit_y = image.shape[0]
    limit_x = image.shape[1]
    black = 0
    white = 0
    for x, min_y, max_y in perimeter:
        for y in range(min_y, max_y + 1):
            if y >= limit_y or y < 0:
                pass
            elif x >= limit_x or x < 0:
                pass
            elif image[y, x] == const.MIN_LUMINANCE:
                black += 1
            else:
                white += 1
    return black, white

def circumference(centre_x: float, centre_y: float, r: float) -> [(int, int)]:
    """ return a list of co-ordinates of a circle centred on x,y of radius r,
        x,y,r do not need to be integer but the returned co-ordinates will be integers,
        the co-ordinates returned are suitable for drawing the circle,
        co-ordinates returned are unique but in a random order,
        the algorithm here was inspired by: https://www.cs.helsinki.fi/group/goa/mallinnus/ympyrat/ymp1.html
        """

    centre_x: int = int(round(centre_x))
    centre_y: int = int(round(centre_y))

    points = []  # list of x,y tuples for a circle of radius r centred on centre_x,centre_y

    def plot(x_offset: int, y_offset: int):
        """ add the circle point for the given x,y offsets from the centre """
        points.append((centre_x + x_offset, centre_y + y_offset))

    def circle_points(x: int, y: int):
        """ make all 8 quadrant points from the one point given
            from https://www.cs.helsinki.fi/group/goa/mallinnus/ympyrat/ymp1.html
                Procedure Circle_Points(x,y: Integer);
                Begin
                    Plot(x,y);
                    Plot(y,x);
                    Plot(y,-x);
                    Plot(x,-y);
                    Plot(-x,-y);
                    Plot(-y,-x);
                    Plot(-y,x);
                    Plot(-x,y)
                End;
        """

        # NB: when a co-ord is 0, x and -x are the same, ditto for y

        if x == 0 and y == 0:
            plot(0, 0)
        elif x == 0:
            plot(0, y)
            plot(y, 0)
            plot(0, -y)
            plot(-y, 0)
        elif y == 0:
            plot(x, 0)
            plot(0, x)
            plot(0, -x)
            plot(-x, 0)
        elif x == y:
            plot(x, x)
            plot(x, -x)
            plot(-x, -x)
            plot(-x, x)
        elif x == -y:
            plot(x, -x)
            plot(-x, x)
            plot(-x, -x)
            plot(x, x)
        else:
            plot(x, y)
            plot(y, x)
            plot(y, -x)
            plot(x, -y)
            plot(-x, -y)
            plot(-y, -x)
            plot(-y, x)
            plot(-x, y)

    """ from https://www.cs.helsinki.fi/group/goa/mallinnus/ympyrat/ymp1.html
        Begin {Circle}
        x := r;
        y := 0;
        d := 1 - r;
        Repeat
            Circle_Points(x,y);
            y := y + 1;
            if d < 0 Then
                d := d + 2*y + 1
            Else Begin
                x := x - 1;
                d := d + 2*(y-x) + 1
            End
        Until x < y
        End; {Circle}
    """
    x = int(round(r))
    if x == 0:
        # special case
        plot(0,0)
    else:
        y = 0
        d = 1 - x
        while True:
            circle_points(x, y)
            y += 1
            if d < 0:
                d += (2 * y + 1)
            else:
                x -= 1
                d += (2 * (y - x) + 1)
            if x < y:
                break

    return points

class Point:

    def __init__(self, x: float, y: float):
        self.x: float = x
        self.y: float = y

    def __str__(self):
        return "({:.2f}, {:.2f})".format(self.x, self.y)


class Circle:

    def __init__(self, centre: Point, radius: float):
        self.centre = centre
        self.radius = radius
        self.points = None  # will become perimeter points on-demand

    def __str__(self):
        return "(centre:{}, radius:{:.2f}, area:{:.2f})".format(self.centre, self.radius, self.area())

    def area(self):
        return math.pi * self.radius * self.radius

    def perimeter(self):
        """ get the x,y co-ordinates of the perimeter of the circle,
            the co-ords are returned as a list of triplets
            where each triplet is the x co-ord and its two y co-ords,
            this format is compatible with count_pixels()
            """
        if self.points is not None:
            # already done it
            return self.points
        # this is expensive, so only do it once
        points = circumference(self.centre.x, self.centre.y, self.radius)
        # we want a min_y/max_y value pair for every x
        y_limits = {}
        min_x = None
        max_x = None
        for x, y in points:
            if min_x is None or x < min_x:
                min_x = x
            if max_x is None or x > max_x:
                max_x = x
            if y_limits.get(x) is None:
                y_limits[x] = [y, y]
            elif y < y_limits[x][0]:
                y_limits[x][0] = y
            elif y > y_limits[x][1]:
                y_limits[x][1] = y
        # build our tuple set
        self.points = []
        for x in range(min_x, max_x + 1):
            limits = y_limits.get(x)
            if limits is None:
                raise Exception('perimeter does not include x of {} when range is {}..{}'.format(x, min_x, max_x))
            min_y, max_y = limits
            min_y += 1  # we want exclusive co-ords
            max_y -= 1  # ..
            # NB: min_y > max_y can happen at the x extremes, but we chuck those, so we don't care
            self.points.append((x, min_y, max_y))
        return self.points


class Contour:
    """ properties of a contour and methods to create/access them,
        the contour only knows the co-ordinates of its points, it knows nothing of the underlying image,
        NB: most of the metrics of a contour are in the range 0..1 where 0 is good and 1 is very bad
        """

    def __init__(self, small=4):
        self.small = small  # contour perimeter of this or less is considered to be small (affects wavyness function)
        self.points: [Point] = None  # points that make up the contour (NB: contours are a 'closed' set of points)
        self.top_left: Point = None
        self.bottom_right: Point = None
        self.reset()

    def __str__(self):
        return self.show()

    def reset(self):
        """ reset the cached stuff """
        self.blob_perimeter: {Point} = None
        self.x_slices: [tuple] = None
        self.y_slices: [tuple] = None
        self.centroid: Point = None
        self.radius: [float] = [None for _ in range(const.RADIUS_MODES)]
        self.offset: float = None

    def add_point(self, point: Point):
        """ add a point to the contour """
        if self.points is None:
            self.points = [point]
        else:
            self.points.append(point)
        if self.top_left is None:
            self.top_left = Point(point.x, point.y)
        else:
            if point.x < self.top_left.x:
                self.top_left.x = point.x
            if point.y < self.top_left.y:
                self.top_left.y = point.y
        if self.bottom_right is None:
            self.bottom_right = Point(point.x, point.y)
        else:
            if point.x > self.bottom_right.x:
                self.bottom_right.x = point.x
            if point.y > self.bottom_right.y:
                self.bottom_right.y = point.y

    def show(self, verbose: bool = False, prefix: str = '    '):
        """ produce a string describing the contour for printing purposes,
            if verbose is True a multi-line response is made that describes all properties,
            lines after the first are prefixed by prfix
            """
        if self.points is None:
            return "None"
        first_line = 'start:{}, box:{}..{}, size:{}, points:{}, small:{}'.\
                     format(self.points[0], self.top_left, self.bottom_right, self.get_size(), len(self.points), self.small)
        if not verbose:
            return first_line
        second_line = 'centroid:{}, offsetness:{:.2f}, squareness:{:.2f}, wavyness:{:.2f}'.\
                      format(self.get_centroid(), self.get_offsetness(),
                             self.get_squareness(), self.get_wavyness())
        return '{}\n{}{}'.format(first_line, prefix, second_line)

    def get_wavyness(self, small=None):
        """ wavyness is a measure of how different the length of the perimeter is to the number of contour points,
            result is in range 0..1, where 0 is not wavy and 1 is very wavy,
            this is a very cheap metric that can be used to quickly drop junk,
            if small is given it overwrites that given to the Contour constructor
            """
        if self.points is None:
            return 1.0
        perimeter = len(self.get_blob_perimeter())
        if small is not None:
            self.small = small
        if perimeter <= self.small:
            # too small to be measurable
            return 0.0
        # NB: number of points is always more than the perimeter length
        return 1 - (perimeter / len(self.points))

    def get_size(self) -> Point:
        """ the size is the maximum width and height of the contour,
            i.e. the size of the enclosing box
            this is a very cheap metric that can be used to quickly drop junk
            """
        width: float = self.bottom_right.x - self.top_left.x + 1
        height: float = self.bottom_right.y - self.top_left.y + 1
        return Point(width, height)

    def get_squareness(self) -> float:
        """ squareness is a measure of how square the enclosing box is,
            result is in range 0..1, where 0 is perfect square, 1 is very thin rectangle,
            this is a very cheap metric that can be used to quickly drop junk
            """
        size = self.get_size()
        ratio = min(size.x, size.y) / max(size.x, size.y)  # in range 0..1, 0=bad, 1=good
        return 1 - ratio  # in range 0..1, 0=square, 1=very thin rectangle

    def get_offsetness(self) -> float:
        """ offsetness is a measure of the distance from the centroid to the enclosing box centre,
            result is in range 0..1, where 0 is exactly coincident, 1 is very far apart
            """
        if self.offset is None:
            box_size = self.get_size()
            box_centre = Point(self.top_left.x + (box_size.x / 2), self.top_left.y + (box_size.y / 2))
            centroid = self.get_centroid()  # NB: this cannot be outside the enclosing box
            x_diff = box_centre.x - centroid.x  # max this can be is box_size.x
            x_diff *= x_diff
            y_diff = box_centre.y - centroid.y  # max this can be is box_size.y
            y_diff *= y_diff
            distance = x_diff + y_diff  # most this can be is box_size.x^2 + box_size.y^2
            limit = max((box_size.x * box_size.x) + (box_size.y * box_size.y), 1)
            self.offset = distance / limit
        return self.offset

    def get_x_slices(self):
        """ get the slices in x,
            for every unique x co-ord find the y extent at that x,
            this function is lazy
            """
        # ToDo: extend to remove 1 pixel gaps
        if self.x_slices is not None:
            # already been done
            return self.x_slices
        x_slices = {}
        for point in self.points:
            if x_slices.get(point.x) is None:
                x_slices[point.x] = {}
            x_slices[point.x][point.y] = True
        self.x_slices = []
        for x in x_slices:
            min_y = None
            max_y = None
            for y in x_slices[x]:
                if min_y is None or y < min_y:
                    min_y = y
                if max_y is None or y > max_y:
                    max_y = y
            self.x_slices.append((x, min_y, max_y))
        return self.x_slices

    def get_y_slices(self):
        """ get the slices in y array,
            for every unique y co-ord find the x extent at that y,
            this function is lazy
            """
        # ToDo: extend to remove 1 pixel gaps
        if self.y_slices is not None:
            # already been done
            return self.y_slices
        y_slices = {}
        for point in self.points:
            if y_slices.get(point.y) is None:
                y_slices[point.y] = {}
            y_slices[point.y][point.x] = True
        self.y_slices = []
        for y in y_slices:
            min_x = None
            max_x = None
            for x in y_slices[y]:
                if min_x is None or x < min_x:
                    min_x = x
                if max_x is None or x > max_x:
                    max_x = x
            self.y_slices.append((y, min_x, max_x))
        return self.y_slices

    def get_blob_perimeter(self):
        """ get the unique contour perimeter points,
            this function is lazy
            """
        if self.blob_perimeter is not None:
            return self.blob_perimeter
        self.blob_perimeter = {}
        for point in self.points:
            self.blob_perimeter[(point.x, point.y)] = True  # NB: do NOT use point as the key, its an object not a tuple
        return self.blob_perimeter

    def get_blob_radius(self, mode) -> float:
        """ get the radius of the blob from its centre for the given mode,
            when mode is inside the radius found is the maximum that fits entirely inside the contour
            when mode is outside the radius found is the minimum that fits entirely outside the contour
            when mode is mean the radius found is the mean distance to the contour perimeter
            this is an expensive operation so its lazy, calculated on demand and then cached
            """
        if self.radius[mode] is not None:
            # already done it
            return self.radius[mode]
        centre = self.get_centroid()  # the centre of mass of the blob (assuming its solid)
        # the perimeter points are the top-left of a 1x1 pixel square, we want their centre, so we add 0.5
        perimeter = self.get_blob_perimeter()
        mean_distance_squared = 0
        min_distance_squared = None
        max_distance_squared = None
        for x, y in perimeter:
            x += 0.5
            y += 0.5
            x_distance = centre.x - x
            x_distance *= x_distance
            y_distance = centre.y - y
            y_distance *= y_distance
            distance = x_distance + y_distance
            mean_distance_squared += distance
            if min_distance_squared is None or min_distance_squared > distance:
                min_distance_squared = distance
            if max_distance_squared is None or max_distance_squared < distance:
                max_distance_squared = distance
        if len(perimeter) <= self.small:
            # too small to do anything but outer
            outside_r = max(math.sqrt(max_distance_squared), 0.5)
            inside_r = outside_r
            mean_r = outside_r
        else:
            mean_distance_squared /= len(perimeter)
            mean_r    = max(math.sqrt(mean_distance_squared), 0.5)
            inside_r  = max(math.sqrt(min_distance_squared) , 0.5)
            outside_r = max(math.sqrt(max_distance_squared) , 0.5)
        # cache all results
        self.radius[const.RADIUS_MODE_INSIDE ] = inside_r
        self.radius[const.RADIUS_MODE_OUTSIDE] = outside_r
        self.radius[const.RADIUS_MODE_MEAN   ] = mean_r
        # return just the one asked for
        return self.radius[mode]

    def get_enclosing_circle(self, mode) -> Circle:
        """ get the requested circle type """
        return Circle(self.get_centroid(), self.get_blob_radius(mode))

    def get_circle_perimeter(self, mode):
        """ get the perimeter of the enclosing circle,
            NB: the circle perimeter is expected to be cached by the Circle instance
            """
        circle = self.get_enclosing_circle(mode)
        return circle.perimeter()

    def trim_edges(self):
        """ remove single pixel extension or indentation in x or y slices at the x or y extremes,
            e.g: XXX                            XXXX
                  XX                            XXX   <-- x indentation
                  XX                            XXXX
                   X  <-- y extension           X XX
                ^                                ^
                |                                |
                +-- x extension                  +-- y indentation
            the algorithm is iterative eating away at the edges until there are no extensions or indentations
            the result may be nothing when dealing with small blobs!
            this is intended to tidy up target edges
            """
        # ToDo: implement the above then hook it into the properties (in particular wavyness)
        #       indentation is better done in x_slices and y_slices
        # an indentation is a slice with
        dropped = True
        while dropped:
            dropped = False
            x_slices = self.get_x_slices()
            while len(x_slices) > 0:
                _, min_y, max_y = x_slices[0]
                if min_y == max_y:
                    # got a protrusion at min-x, drop it
                    del x_slices[0]
                    dropped = True
            while len(x_slices) > 0:
                _, min_y, max_y = x_slices[-1]
                if min_y == max_y:
                    # got a protrusion at max-x, drop it
                    del x_slices[-1]
                    dropped = True
            y_slices = self.get_y_slices()
            while len(y_slices) > 0:
                _, min_x, max_x = y_slices[0]
                if min_x == max_x:
                    # got a protrusion at min-y, drop it
                    del y_slices[0]
                    dropped = True
            while len(y_slices) > 0:
                _, min_x, max_x = y_slices[-1]
                if min_x == max_x:
                    # got a protrusion at max-y, drop it
                    del y_slices[-1]
                    dropped = True

    def get_centroid(self) -> Point:
        """ get the centroid of the blob as: sum(points)/num(points) """
        if self.centroid is None:
            sum_x = 0
            num_x = 0
            x_slices = self.get_x_slices()
            for x, min_y, max_y in x_slices:
                samples = max_y - min_y + 1
                sum_x += samples * x
                num_x += samples
            sum_y = 0
            num_y = 0
            y_slices = self.get_y_slices()
            for y, min_x, max_x in y_slices:
                samples = max_x - min_x + 1
                sum_y += samples * y
                num_y += samples
            self.centroid = Point((sum_x / num_x) + 0.5, (sum_y / num_y) + 0.5)  # +0.5 is to get to the pixel centre
        return self.centroid


class Blob:
    """ a blob is an external contour and its properties,
        a blob has access to the underlying image (unlike a Contour)
        """

    def __init__(self, label: int, image, inverted: bool, mode, small: int):
        self.label: int = label
        self.image = image  # the binary image buffer the blob was found within
        self.inverted = inverted  # True if its a black blob, else a white blob
        self.mode = mode  # what type of circle radius required (one of Contour.RADIUS...)
        self.small = small  # any blob with a perimeter of this or less is considered to be 'small'
        self.external: Contour = None
        self.internal: [Contour] = []
        self.rejected = const.REJECT_NONE  # why it was rejected (if it was)
        self.reset()

    def __str__(self):
        return self.show()

    def reset(self):
        """ reset all the cached stuff """
        self.blob_black = None
        self.blob_white = None
        self.box_black = None
        self.box_white = None
        self.circle_black = None
        self.circle_white = None

    def add_contour(self, contour: Contour):
        """ add a contour to the blob, the first contour is the external one,
            subsequent contours are internal,
        """
        if self.external is None:
            self.external = contour
        else:
            self.internal.append(contour)

    def show(self, verbose: bool = False, prefix: str = '    '):
        """ describe the blob for printing purposes,
            if verbose is True a multi-line response is made that describes all properties,
            lines after the first are prefixed by prefix
            """
        header = "label:{}".format(self.label)
        if self.external is None:
            return header
        body = '{}, {}'.format(header, self.external.show(verbose, prefix))
        if verbose:
            size = self.get_size()
            body = '{}\n{}internals:{}, blob_pixels:{}, box_pixels:{}, size:{:.2f}'.\
                   format(body, prefix, len(self.internal), self.get_blob_pixels(), self.get_box_pixels(), size)
        return body

    def get_blob_pixels(self):
        """ get the total white area and black area within the perimeter of the blob """
        if self.external is None:
            return None
        if self.blob_black is not None:
            return self.blob_black, self.blob_white
        self.blob_black, self.blob_white = count_pixels(self.image, self.external.get_x_slices())
        return self.blob_black, self.blob_white

    def get_box_pixels(self):
        """ get the total white area and black area within the enclosing box of the blob """
        if self.external is None:
            return None
        if self.box_black is not None:
            return self.box_black, self.box_white
        # build 'x-slices' for the box
        x_slices = []
        for x in range(self.external.top_left.x, self.external.bottom_right.x + 1):
            x_slices.append((x, self.external.top_left.y, self.external.bottom_right.y))
        self.box_black, self.box_white = count_pixels(self.image, x_slices)
        return self.box_black, self.box_white

    def get_size(self) -> float:
        """ the size of a blob is the average of the width + height of the bounding box """
        if self.external is None:
            return None
        point = self.external.get_size()
        return (point.x + point.y) / 2

    def get_quality_stats(self):
        """ get all the 'quality' statistics for a blob """
        return self.get_squareness(), self.get_wavyness(),\
               self.get_whiteness(), self.get_blackness(), self.get_offsetness()

    def get_circle_pixels(self):
        """ get the total white area and black area within the enclosing circle """
        if self.external is None:
            return None
        if self.circle_black is not None:
            return self.circle_black, self.circle_white
        self.circle_black, self.circle_white = count_pixels(self.image, self.external.get_circle_perimeter(self.mode))
        return self.circle_black, self.circle_white

    def get_squareness(self) -> float:
        if self.external is None:
            return None
        return self.external.get_squareness()

    def get_wavyness(self) -> float:
        """ this allows for filtering very irregular blobs """
        if self.external is None:
            return None
        return self.external.get_wavyness(self.small)

    def get_offsetness(self) -> float:
        """ this measure how far the centre of mass is from the centre of the enclosing box,
            this allows for filtering elongated blobs (e.g. a 'saucepan')
            """
        if self.external is None:
            return None
        return self.external.get_offsetness()

    def get_whiteness(self) -> float:
        """ whiteness is a measure of how 'white' the area covered by the enclosing circle is,
            (contrast this with blackness which covers the whole enclosing box)
            when inverted is false:
                result is in range 0..1, where 0 is all white and 1 is all black,
            when inverted is True:
                result is in range 0..1, where 0 is all black and 1 is all white,
            this allows for filtering out blobs with lots of holes in it
            """
        if self.external is None:
            return None
        black, white = self.get_circle_pixels()
        if (black + white) <= self.small:
            # too small to measure
            return 0.0
        if self.inverted:
            return white / (black + white)
        else:
            return black / (black + white)

    def get_blackness(self) -> float:
        """ blackness is a measure of how 'white' the area covered by the enclosing box is,
            (contrast this with whiteness which covers the enclosing circle)
            when inverted is false:
                result is in range 0..1, where 0 is all white and 1 is all black,
            when inverted is True:
                result is in range 0..1, where 0 is all black and 1 is all white,
            this allows for filtering out sparse symmetrical blobs (e.g. a 'star')
            """
        if self.external is None:
            return None
        black, white = self.get_box_pixels()
        if (black + white) <= self.small:
            # too small to measure
            return 0.0
        if self.inverted:
            return white / (black + white)
        else:
            return black / (black + white)


class Labels:
    """ label to blob map """
    blobs = None

    def add_label(self, label: int, blob: Blob):
        if self.blobs is None:
            self.blobs = {}
        self.blobs[label] = blob

    def get_blob(self, label: int):
        if self.blobs is None:
            return None
        elif label in self.blobs:
            return self.blobs[label]
        else:
            # no such label
            return None
