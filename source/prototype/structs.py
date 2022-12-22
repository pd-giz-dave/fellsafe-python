"""
various types used by the scanning algorithm
"""

class Step:
    """ a Step is a description of a luminance change
        (orientation of vertical or horizontal is defined by the context)
        """

    def __init__(self, where, type, from_pixel, to_pixel):
        self.where = where  # the y co-ord of the step
        self.type = type  # the step type, rising or falling
        self.from_pixel = from_pixel  # the 'from' pixel level
        self.to_pixel = to_pixel  # the 'to' pixel level

    def __str__(self):
        return '({} at {} {}->{})'.format(self.type, self.where, self.from_pixel, self.to_pixel)


class Edge:
    """ an Edge is a sequence of joined steps """

    def __init__(self, where, type, samples, grey_as):
        self.where = where  # the x co-ord of the start of this edge
        self.type = type  # the type of the edge, falling or rising
        self.samples = samples  # the list of connected y's making up this edge
        self.grey_as = grey_as

    def __str__(self):
        self.show()

    def show(self, in_line=10, max_x=None):
        """ generate a readable string describing the edge """
        if in_line == 0:
            # do not want to see samples
            samples = ''
        elif len(self.samples) > in_line:
            # too many samples to see all, so just show first and last few
            samples = ': {}..{}'.format(self.samples[:int(in_line / 2)], self.samples[-int(in_line / 2):])
        else:
            # can show all the samples
            samples = ': {}'.format(self.samples)
        from_x = self.where
        to_x = from_x + len(self.samples) - 1
        if max_x is not None:
            to_x %= max_x
        from_y = self.samples[0]
        to_y = self.samples[-1]
        return '{}(grey={}) at ({},{}) to ({},{}) for {}{}'. \
            format(self.type, self.grey_as, from_x, from_y, to_x, to_y, len(self.samples), samples)


class Extent:
    """ an Extent is the inner edge co-ordinates of a projected image along with
        the horizontal and vertical edge fragments it was built from """

    def __init__(self, inner=None, outer=None, inner_fail=None, outer_fail=None,
                 buckets=None, rising_edges=None, falling_edges=None, slices=None):
        self.inner: [int] = inner  # list of y co-ords for the inner edge
        self.inner_fail = inner_fail  # reason if failed to find inner edge or None if OK
        self.outer: [int] = outer  # list of y co-ords for the outer edge
        self.outer_fail = outer_fail  # reason if failed to find outer edge or None if OK
        self.rising_edges: [Edge] = rising_edges  # rising edge list used to create this extent
        self.falling_edges: [Edge] = falling_edges  # falling edge list used to create this extent
        self.buckets = buckets  # the binarized image the extent was created from
        self.slices = slices  # the slices extracted from the extent (by _find_all_digits)


class Digit:
    """ a digit is a decode of a sequence of slice samples into the most likely digit """

    def __init__(self, digit, error, start, samples):
        self.digit = digit  # the most likely digit
        self.error = error  # the average error across its samples
        self.start = start  # start x co-ord of this digit
        self.samples = samples  # the number of samples in this digit

    def __str__(self):
        return '({}, {:.2f}, at {} for {})'.format(self.digit, self.error, self.start, self.samples)


class Result:
    """ a result is the result of a number decode and its associated error/confidence level """

    def __init__(self, number, doubt, code, digits):
        self.number = number  # the number found
        self.doubt = doubt  # integer part is sum of error digits (i.e. where not all three copies agree)
        # fractional part is average bit error across all the slices
        self.code = code  # the code used for the number lookup
        self.digits = digits  # the digit pattern used to create the code


class Target:
    """ structure to hold detected target information """

    def __init__(self, centre_x, centre_y, blob_size, target_size, result):
        self.centre_x = centre_x  # x co-ord of target in original image
        self.centre_y = centre_y  # y co-ord of target in original image
        self.blob_size = blob_size  # blob size originally detected by the blob detector
        self.target_size = target_size  # target size scaled to the original image (==outer edge average Y)
        self.result = result  # the number, doubt and digits of the target


class Reject:
    """ struct to hold info about rejected targets """

    def __init__(self, centre_x, centre_y, blob_size, target_size, reason):
        self.centre_x = centre_x
        self.centre_y = centre_y
        self.blob_size = blob_size
        self.target_size = target_size
        self.reason = reason


class Detection:
    """ struct to hold info about a Scan detected code """

    def __init__(self, result, centre_x, centre_y, target_size, blob_size):
        self.result = result  # the result of the detection
        self.centre_x = centre_x  # where it is in the original image
        self.centre_y = centre_y  # ..
        self.blob_size = blob_size  # the size of the blob as detected by opencv
        self.target_size = target_size  # the size of the target in the original image (used for relative distance)
