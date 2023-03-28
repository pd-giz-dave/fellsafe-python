""" Classes that provides properties for our contours """

import math
import const
import utils

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

class Circle:

    def __init__(self, centre: (float, float), radius: float):
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
        points = circumference(self.centre[0], self.centre[1], self.radius)
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
        self.points: [(int, int)] = None  # points that make up the contour (NB: contours are a 'closed' set of points)
        self.reset()

    def __str__(self):
        return self.show()

    def reset(self):
        """ reset the cached stuff """
        self.enclosing_box: ((int, int), (int, int)) = None
        self.blob_perimeter: {(int, int): int} = None
        self.x_slices: [tuple] = None
        self.y_slices: [tuple] = None
        self.centroid: (float, float) = None
        self.radius: [float] = [None for _ in range(const.RADIUS_MODES)]
        self.offset: float = None

    def add_point(self, point: (int, int)):
        """ add a point to the contour """
        if self.points is None:
            self.points = [point]
        else:
            self.points.append(point)

    def get_enclosing_box(self):
        """ get the minimum sized box that encloses the contour """
        if self.enclosing_box is not None:
            return self.enclosing_box
        top_left_x     = None
        top_left_y     = None
        bottom_right_x = None
        bottom_right_y = None
        for x, y in self.points:
            if top_left_x is None or x < top_left_x:
                top_left_x = x
            if top_left_y is None or y < top_left_y:
                top_left_y = y
            if bottom_right_x is None or x > bottom_right_x:
                bottom_right_x = x
            if bottom_right_y is None or y > bottom_right_y:
                bottom_right_y = y
        self.enclosing_box = ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y))
        return self.enclosing_box

    def show(self, verbose: bool = False, prefix: str = '    '):
        """ produce a string describing the contour for printing purposes,
            if verbose is True a multi-line response is made that describes all properties,
            lines after the first are prefixed by prfix
            """
        if self.points is None:
            return "None"
        first_line = 'start:{}, box:{}..{}, size:{}, points:{}, small:{}'.\
                     format(utils.show_point(self.points[0]),
                            utils.show_point(self.get_enclosing_box()[0]), utils.show_point(self.get_enclosing_box()[1]),
                            utils.show_point(self.get_size()), len(self.points), self.small)
        if not verbose:
            return first_line
        second_line = 'centroid:{}, offsetness:{:.2f}, squareness:{:.2f}, wavyness:{:.2f}'.\
                      format(utils.show_point(self.get_centroid()), self.get_offsetness(),
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

    def get_size(self) -> (int, int):
        """ the size is the maximum width and height of the contour,
            i.e. the size of the enclosing box
            this is a very cheap metric that can be used to quickly drop junk
            """
        (tl_x, tl_y), (br_x, br_y) = self.get_enclosing_box()
        width  = br_x - tl_x + 1
        height = br_y - tl_y + 1
        return width, height

    def get_squareness(self) -> float:
        """ squareness is a measure of how square the enclosing box is,
            result is in range 0..1, where 0 is perfect square, 1 is very thin rectangle,
            this is a very cheap metric that can be used to quickly drop junk
            """
        x, y = self.get_size()
        ratio = min(x, y) / max(x, y)  # in range 0..1, 0=bad, 1=good
        return 1 - ratio  # in range 0..1, 0=square, 1=very thin rectangle

    def get_offsetness(self) -> float:
        """ offsetness is a measure of the distance from the centroid to the enclosing box centre,
            result is in range 0..1, where 0 is exactly coincident, 1 is very far apart
            """
        if self.offset is None:
            (box_top_left_x, box_top_left_y), _ = self.get_enclosing_box()
            box_size_x, box_size_y = self.get_size()
            box_centre = (box_top_left_x + (box_size_x / 2), box_top_left_y + (box_size_y / 2))
            centroid = self.get_centroid()  # NB: this cannot be outside the enclosing box
            x_diff = box_centre[0] - centroid[0]  # max this can be is box_size_x/2
            x_diff *= x_diff
            y_diff = box_centre[1] - centroid[1]  # max this can be is box_size_y/2
            y_diff *= y_diff
            distance = x_diff + y_diff  # most this can be is (box_size_x/2)^2 + (box_size_y/2)^2
            max_x = box_size_x / 2
            max_y = box_size_y / 2
            limit = max((max_x * max_x) + (max_y * max_y), 1)
            self.offset = distance / limit
        return self.offset

    def get_x_slices(self):
        """ get the slices in x,
            for every unique x co-ord find the y extent at that x,
            this function is lazy
            """
        if self.x_slices is not None:
            # already been done
            return self.x_slices
        x_slices = {}
        for (x, y) in self.points:
            if x_slices.get(x) is None:
                x_slices[x] = {}
            x_slices[x][y] = True
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
        if self.y_slices is not None:
            # already been done
            return self.y_slices
        y_slices = {}
        for (x, y) in self.points:
            if y_slices.get(y) is None:
                y_slices[y] = {}
            y_slices[y][x] = True
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
        for p, (x, y) in enumerate(self.points):
            # NB: points are in clockwise order, we want that preserved, we're relying on Python dict doing that
            self.blob_perimeter[(x, y)] = p  # NB: do NOT use point as the key, its an object not a tuple
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
        centre_x, centre_y = self.get_centroid()  # the centre of mass of the blob (assuming its solid)
        # the perimeter points are the top-left of a 1x1 pixel square, we want their centre, so we add 0.5
        perimeter = self.get_blob_perimeter()
        mean_distance_squared = 0
        min_distance_squared = None
        max_distance_squared = None
        for x, y in perimeter:
            x += 0.5
            y += 0.5
            x_distance = centre_x - x
            x_distance *= x_distance
            y_distance = centre_y - y
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

    def get_centroid(self) -> (float, float):
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
            self.centroid = ((sum_x / num_x) + 0.5, (sum_y / num_y) + 0.5)  # +0.5 is to get to the pixel centre
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
        self.internal: [Contour] = []  # list of internal contours (NB: we're only interested in how many)
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
        self.thickness = None  # property set externally, ratio of box size to average contour thickness
        self.sameness  = None  # property set externally, luminance level change across contour edge as fraction of max
        self.splitness = None  # property set externally, how many times the blob was split

    def add_contour(self, contour: Contour):
        """ add a contour to the blob, the first contour is the external one,
            subsequent contours are internal,
            all contours are guaranteed to have their first and last point the same (needed by later processes)
        """
        # the first and last point of a contour must be the same, if not add it
        points = contour.points
        if points is None or len(points) == 0:
            breakpoint()
            raise Exception('Attempt to add a contour with no points')
        if len(points) > 1:
            if points[0] != points[-1]:
                # add a last point same as first (this is important when calculating perimeter directions later)
                contour.add_point(points[0])
        if self.external is None:
            self.external = contour
        else:
            self.internal.append(contour)

    def show(self, verbose: bool = False, prefix: str = '    '):
        """ describe the blob for printing purposes,
            if verbose is True a multi-line response is made that describes all properties,
            lines after the first are prefixed by prefix
            """
        header = "label:{}({})".format(self.label, self.rejected)
        if self.external is None:
            return header
        body = '{}, {}'.format(header, self.external.show(verbose, prefix))
        if verbose:
            size = self.get_size()
            size = (size[0] + size[1]) / 2
            body = '{}\n{}internals:{}, blob_pixels:{}, box_pixels:{}, size:{:.2f}, ' \
                   'blackness:{}, whiteness:{}, thickness:{}, sameness:{}, splitness:{}'.\
                   format(body, prefix, len(self.internal),
                          self.get_blob_pixels(), self.get_box_pixels(), size,
                          utils.show_number(self.get_blackness()),
                          utils.show_number(self.get_whiteness()),
                          utils.show_number(self.thickness), utils.show_number(self.sameness),
                          utils.show_number(self.splitness))
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
        (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = self.external.get_enclosing_box()
        # build 'x-slices' for the box
        x_slices = []
        for x in range(top_left_x, bottom_right_x + 1):
            x_slices.append((x, top_left_y, bottom_right_y))
        self.box_black, self.box_white = count_pixels(self.image, x_slices)
        return self.box_black, self.box_white

    def get_size(self) -> float:
        """ get the width and height of the bounding box """
        if self.external is None:
            return None
        return self.external.get_size()

    def get_perimeter(self):
        """ get the unique points around the blob perimeter as a list of x,y points in clockwise order """
        if self.external is None:
            return None
        perimeter = self.external.get_blob_perimeter()  # NB: relying on a Python dict preserving the insertion order
        return [(x, y) for x, y in perimeter.keys()]

    def get_points(self):
        """ get all the points of the blob contour as a list of x,y points in clockwise order """
        if self.external is None:
            return None
        return self.external.points  # NB: relying on contours being 'clockwise' followed

    def get_enclosing_circle(self) -> Circle:
        """ get the enclosing circles of the blob contour """
        if self.external is None:
            return None
        return self.external.get_enclosing_circle(self.mode)

    def get_quality_stats(self):
        """ get all the 'quality' statistics for a blob """
        return (self.get_squareness(),'squareness'), \
               (self.get_wavyness()  ,'wayness'   ), \
               (self.get_whiteness() ,'whiteness' ), \
               (self.get_blackness() ,'blackness' ), \
               (self.get_offsetness(),'offsetness'), \
               (self.thickness       ,'thickness' ), \
               (self.sameness        ,'sameness'  ), \
               (self.splitness       ,'splitness' )

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

    def extract(self, start, end):
        """ create a new blob like this one but consisting only of the external points
            between start and end inclusive, the resulting external contour is made
            to be closed by joining start to end with a straight line (8-connected).
            NB: end index may be lower than start, i.e. we're wrapping, that's benign
                because contours are closed loops.
            """
        # make a new contour and populate it with the required points
        contour = Contour(self.small)
        points  = self.get_points()
        point   = start
        while point != ((end+1) % len(points)):
            if point == (len(points)-1):
                # skip the last point as we know its the same as the first (enforced by add_contour())
                pass
            else:
                contour.add_point(points[point])
            point = (point + 1) % len(points)
        # join the ends by adding more points from the end to the start
        start_x, start_y = points[start]
        end_x  , end_y   = points[end  ]
        join = utils.line(end_x, end_y, start_x, start_y)
        # NB: join will contain the first and last point, which we do not want
        for sample in range(1, len(join)-1):
            contour.add_point(join[sample])
        # make a new blob with this contour
        extracted = Blob(self.label, self.image, self.inverted, self.mode, self.small)
        extracted.add_contour(contour)  # NB: This will add the final point to be same as the first
        return extracted


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
