""" Extract the code from the code area
    This module takes a list of data cell co-ordinates for each detection and extracts the code bits.
    The cells given here are all those between the locators, the data bits are within that.
    See the visualisation in codes.py
"""

import math

import const
import utils
import canvas
import cells
import codes

class Params(cells.Params):
    def __init__(self):
        self.extractor = None


class Extractor:

    DATA_CELL   = 'data'
    BLACK_CELL  = 'black'
    WHITE_CELL  = 'white'
    IGNORE_CELL = 'ignore'

    # region tuning constants...
    INTEGRATION_WIDTH = 1   # fraction of image width that is integrated when creating a tertiary image
    BLACK_THRESHOLD   = 15  # % below average to consider to be black in a tertiary image
    WHITE_THRESHOLD   = 1   # % above average to consider to be white in a tertiary image
    # gap between black and white threshold is considered to be 'gray'
    CELL_OVERSAMPLE    = 0.6  # how far inside(<1)/outside(>1) the given cell radius to sample for levels
    GREY_THRESHOLD     = 0.8  # grey ratio threshold for a cell to be considered to be a None
    ZERO_THRESHOLD     = 0.1  # black - white ratio threshold for a cell to be considered to be a '0'
    ONE_THRESHOLD      = 0.1  # white - black ratio threshold for a cell to be considered to be a '1'
    DONATION_THRESHOLD = 0.3  # grey level above which a donation to black or white considered (>=1 turns it off)
    DONATION_LEVEL     = 0.2  # what fraction of grey to donate to black or white
    # endregion

    def __init__(self, source, images, cells, logger=None, origins=None):
        self.source          = source      # the originating image (that locations will refer to)
        self.images          = images      # the grayscale sub-image for each detection (and cells refer to)
        self.cells           = cells       # cells discovered in each detection
        self.origins         = origins     # detection origins within the main image
        self.logger          = logger      # for diagnostics
        self.tertiary        = None        # tertiary (black,gray,white) image of each detection grayscale image
        self.cell_levels     = None        # cell luminance levels for each detection
        self.data_bits       = None        # data bits extracted from the cells
        self.locations       = None        # locations and sizes of every detection

    def draw_locations(self):
        """ draw all the locations detected on the original image """
        if self.logger is None:
            return
        image = canvas.colourize(self.source)
        for detection, (centre, radius) in enumerate(self.get_locations()):
            canvas.circle(image, centre, radius, const.GREEN)
            canvas.settext(image, '{}x{}y{:.0f}r'.format(centre[0], centre[1], radius), centre,
                           size=0.4, colour=const.RED)
        self.logger.draw(image, file='detections')

    def draw_bits(self):
        """ draw circle for 0 bits, 1 bits and None bits in distinct colours for all detections """
        if self.logger is None:
            return
        for detection in range(len(self.cells)):
            data_bits  = self.get_bits()[detection]
            data_cells = self.cells[detection]
            image      = self.get_tertiary(detection)
            bit_num    = 0
            for cell in codes.Codes.DATA_CELLS:
                if cell is None:
                    # just a spacer, ignore it
                    continue
                data_cell = data_cells[cell[1]][cell[0]]
                if data_cell is None: breakpoint()
                data_bit  = data_bits[bit_num]
                if data_bit == 0:
                    colour = const.GREEN
                elif data_bit == 1:
                    colour = const.BLUE
                else:
                    colour = const.RED
                if bit_num == 0:
                    # mark the first bit so can see the orientation
                    image = canvas.circle(image,(data_cell[0], data_cell[1]), 1, const.RED)
                image = canvas.circle(image, (data_cell[0], data_cell[1]), data_cell[2], colour)
                bit_num  += 1
            folder = utils.image_folder(target=self.get_origin(detection))
            self.logger.push(folder=folder)
            self.logger.draw(image, file='bits')
            self.logger.pop()

    def draw_tertiary(self):
        """ draw the tertiary image for all detections """
        for detection, image in enumerate(self.get_tertiary()):
            folder = utils.image_folder(target=self.get_origin(detection))
            self.logger.push(folder=folder)
            self.logger.draw(image, file='tertiary')
            self.logger.pop()

    def get_origin(self, detection, negate=False):
        """ return the origin of the given detection in the source image (diagnostic aid),
            the origin is the top-left cell centre in the detection,
            if negate is True the negation of the origin is returned
            """
        x, y = self.origins[detection]
        if negate:
            return -x, -y
        else:
            return x, y

    def get_tertiary(self, detection=None):
        """ get the tertiary image of the given, or all, detections """
        if self.tertiary is None:
            self.tertiary = []
            for image, _ in self.images:
                self.tertiary.append(canvas.binarize(image,
                                                     width=Extractor.INTEGRATION_WIDTH,
                                                     black=Extractor.BLACK_THRESHOLD,
                                                     white=Extractor.WHITE_THRESHOLD))
            if self.logger is not None:
                self.draw_tertiary()
        if detection is None:
            return self.tertiary
        return self.tertiary[detection]

    def get_luminance(self, detection, centre_x: float, centre_y: float, radius: float) -> (float, float, float):
        """ get the ratio of black,grey,white pixels for all pixels in the circle of radius r at location x,y """

        radius *= Extractor.CELL_OVERSAMPLE
        radius = max(radius, 1)  # do at least a 2x2 square
        radius_squared = int(math.ceil(radius * radius))  # round up so pixels are included near the edge

        def in_circle(x, y):
            """ return True iff x,y is inside, or on, the circle at centre_x/y with radius """
            if radius <= 1:
                # consider everything inside with such a small radius
                return True
            x_len  = x - centre_x
            x_len *= x_len
            y_len  = y - centre_y
            y_len *= y_len
            distance = int(x_len + y_len)  # round down so pixels near the edge are included
            if distance > radius_squared:
                # this x,y is outside our circle
                return False
            # we're on or inside
            return True

        image = self.get_tertiary(detection)
        image_x, image_y = canvas.size(image)

        min_x = max(centre_x - radius, 0)
        max_x = min(centre_x + radius, image_x-1)
        min_y = max(centre_y - radius, 0)
        max_y = min(centre_y + radius, image_y-1)

        counts  = [0, 0, 0]  # black, grey, white pixel counts
        samples = 0
        step    = 1
        x_dash = min_x
        while x_dash < max_x:
            y_dash = min_y
            while y_dash < max_y:
                if in_circle(x_dash, y_dash):
                    samples += 1
                    parts = canvas.pixelparts(image, x_dash, y_dash)
                    for pixel, ratio in parts:
                        if pixel == const.MIN_LUMINANCE:
                            counts[0] += ratio
                        elif pixel == const.MID_LUMINANCE:
                            counts[1] += ratio
                        elif pixel == const.MAX_LUMINANCE:
                            counts[2] += ratio
                        else:
                            raise Exception('Image is not tertiary, got pixel of {}'.format(pixel))
                y_dash += step
            x_dash += step
        if samples == 0:
            raise Exception('No samples in circle at {:.2f}x {:.2f}y for radius {:.2f}'.
                            format(centre_x, centre_y, radius))
        return counts[0] / samples, counts[1] / samples, counts[2] / samples

    def get_cell_levels(self, target=None):
        """ get the luminance level of all relevant cells for the given target, None==all,
            the 'level' of a cell here is the ratio of black,grey,white pixels within it (i.e. 3 numbers),
            we treat the cells as if they are circles (so we do not need to consider rotations)
            """
        if self.cell_levels is None:
            # not got levels yet, get all relevant cell levels now for all detections
            self.cell_levels = []
            for detection, rows in enumerate(self.cells):
                cell_levels = [[None for _ in range(len(rows[0]))] for _ in range(len(rows))]
                for row, cells in enumerate(self.cells[detection]):
                    for col, (x, y, r, _) in enumerate(cells):
                        if (col, row) in codes.Codes.DATA_CELLS:
                            # want this one, its a data cell
                            cell_type = Extractor.DATA_CELL
                        elif (col, row) in codes.Codes.BLACK_CELLS:
                            # want this one, its a black luminance reference cell
                            cell_type = Extractor.BLACK_CELL
                        elif (col, row) in codes.Codes.WHITE_CELLS:
                            # want this one, its a white luminance reference cell
                            cell_type = Extractor.WHITE_CELL
                        else:
                            # not a relevant cell, ignore it
                            cell_type = Extractor.IGNORE_CELL
                        level = self.get_luminance(detection, x, y, r)
                        cell_levels[row][col] = [level, cell_type]
                self.cell_levels.append(cell_levels)
            if self.logger is not None:
                for detection in range(len(self.cell_levels)):
                    rows = self.cell_levels[detection]
                    folder = utils.image_folder(target=self.get_origin(detection))
                    self.logger.push(context='get_cell_levels/{}'.format(folder), folder=folder)
                    self.logger.log('')
                    self.logger.log('Detection {} cell_levels:'.format(detection))
                    for row, cols in enumerate(rows):
                        line = ''
                        for (black, grey, white), cell_type in cols:
                            line = '{}, [({:3.0f},{:3.0f},{:3.0f}) {:6}]'.\
                                   format(line, black*100, grey*100, white*100, cell_type)
                        self.logger.log('  row {}: {}'.format(row, line[2:]))
                    self.logger.pop()

        if target is None:
            return self.cell_levels
        else:
            return self.cell_levels[target]

    def get_locations(self):
        """ get the location and size of every detection relative to the original source image,
            the locations are truncated to pixel boundaries, the size (aka radius) is not
            """
        if self.locations is not None:
            return self.locations

        # the location is the centre of the cells and its size is the radius of a circle passing through the corners,
        # the cells given to us are a square matrix located on the locators, our reference points are the
        # first and last cell in the first row and the first and last cell in the last row,
        # the centre is the intersection of the diagonals from top-left to bottom-right and top-right to
        # bottom-left (with respect to an un-rotated matrix), the radius is the average distance from the centre
        # to our reference points
        self.locations = []
        for detection, rows in enumerate(self.cells):
            top_left = rows[0][0]
            top_right = rows[0][-1]
            bottom_right = rows[-1][-1]
            bottom_left = rows[-1][0]
            sub_centre = utils.intersection((top_left, bottom_right), (top_right, bottom_left))
            rad1 = utils.distance(sub_centre, top_left)
            rad2 = utils.distance(sub_centre, top_right)
            rad3 = utils.distance(sub_centre, bottom_left)
            rad4 = utils.distance(sub_centre, bottom_right)
            sub_radius = math.sqrt((rad1 + rad2 + rad3 + rad4) / 4)
            _, scale = self.images[detection]
            centre, radius = canvas.translate(sub_centre, sub_radius, self.get_origin(detection, negate=True), scale)
            self.locations.append(((int(centre[0]), int(centre[1])), radius))
            if self.logger is not None:
                folder = utils.image_folder(target=self.get_origin(detection))
                self.logger.push(context='get_locations/{}'.format(folder), folder=folder)
                self.logger.log('Detection {}: {:.2f}x {:.2f}y {:.2f}r --> {:.2f}x {:.2f}y {:.2f}r (scale={:.2f})'.
                                format(detection, sub_centre[0], sub_centre[1], sub_radius,
                                       centre[0], centre[1], radius, scale))
                self.logger.pop()

        if self.logger is not None:
            self.draw_locations()

        return self.locations

    def get_bits(self):
        """ get the data bits for all detections,
            the bits are returned as a list of 0 (black) or 1 (white) or None (grey) in the order
            defined by Codes.DATA_CELLS
            """
        if self.data_bits is None:
            self.data_bits = []
            for detection, cell_levels in enumerate(self.get_cell_levels()):
                bits = []
                bit_num = 0
                for cell in codes.Codes.DATA_CELLS:
                    if cell is None:
                        # just a spacer, ignore it
                        continue
                    (black, grey, white), typ = cell_levels[cell[1]][cell[0]]
                    if typ != Extractor.DATA_CELL:
                        raise Exception('Expected DATA_CELL at {} in detection {}, got {}'.format(cell, detection, typ))
                    # we have a ratio that is black, is grey and is white, they sum to 1
                    # we use these relative levels to determine the bit-ness of the cell (0, 1, or None==don't know)
                    if grey >= Extractor.GREY_THRESHOLD:
                        # overwhelmingly grey
                        bit = None
                    else:
                        if grey >= Extractor.DONATION_THRESHOLD:
                            # there is substantial grey, so donate some to black or white
                            donation = grey * Extractor.DONATION_LEVEL  # what we are going to donate
                            if black > white:
                                # give some to black
                                black += donation
                                grey -= donation
                            elif white > black:
                                # give some to white
                                white += donation
                                grey -= donation
                            else:
                                # ambiguous, stay as we are
                                pass
                        if (black - white) >= Extractor.ZERO_THRESHOLD:
                            bit = 0
                        elif (white - black) >= Extractor.ONE_THRESHOLD:
                            bit = 1
                        else:
                            bit = None
                    bits.append(bit)
                    bit_num += 1
                self.data_bits.append(bits)
                if self.logger is not None:
                    folder = utils.image_folder(target=self.get_origin(detection))
                    self.logger.push(context='get_bits/{}'.format(folder), folder=folder)
                    self.logger.log('Detection {}: bits={}'.format(detection, bits))
                    self.logger.pop()
            if self.logger is not None:
                self.draw_bits()

        return self.data_bits


def find_bits(source, params, images, cells, logger, origins):
    """ find code bits within the given detected cells """
    if logger is not None:
        logger.push('find_bits')
        logger.log('')
    extractor = Extractor(source, images, cells, logger, origins)
    extractor.get_bits()
    extractor.get_locations()
    params.extractor = extractor
    if logger is not None:
        logger.pop()
    return params


def _test(src, proximity, blur, mode, params=None, logger=None, create_new=True):
    """ ************** TEST **************** """

    if logger.depth() > 1:
        logger.push('extractor/_test')
    else:
        logger.push('_test')
    logger.log('')
    logger.log("Extracting code bits (create new {})".format(create_new))

    # get the code areas
    if not create_new:
        params = logger.restore(file='extractor', ext='params')
        if params is None or params.source_file != src:
            create_new = True
    if create_new:
        if params is None:
            params = Params()
        params = cells._test(src, proximity, blur=blur, mode=mode, params=params, logger=logger, create_new=create_new)
        if params is None:
            logger.log('Cells failed on {}'.format(src))
            logger.pop()
            return None
        logger.save(params, file='extractor', ext='params')

    # process the code areas
    images = params.cells.images()
    cell_array = params.cells.cells()
    origins = params.cells.origins
    source = params.source
    params = find_bits(source, params, images, cell_array, logger, origins)

    logger.pop()
    return params

if __name__ == "__main__":
    """ test harness """

    src = '/home/dave/precious/fellsafe/fellsafe-image/media/kilo-codes/kilo-codes-close-150-257-263-380-436-647-688-710-777.jpg'
    #src = "/home/dave/precious/fellsafe/fellsafe-image/source/kilo-codes/test-alt-bits.png"
    #proximity = const.PROXIMITY_CLOSE
    proximity = const.PROXIMITY_FAR

    logger = utils.Logger('extractor.log', 'extractor/{}'.format(utils.image_folder(src)))

    _test(src, proximity, blur=const.BLUR_KERNEL_SIZE, mode=const.RADIUS_MODE_MEAN, logger=logger, create_new=False)
