""" Extract the code from the code area
    This module takes a list of data cell co-ordinates for each detection and extracts the code bits.
    The cells given here are all those between the locators, the data bits are within that.
    See the visualisation in codes.py
"""
import math

import const
import utils
import finder
import codes

class Extractor:

    DATA_CELL  = 'data'
    BLACK_CELL = 'black'
    WHITE_CELL = 'white'

    def __init__(self, source, image, detections, logger=None):
        self.source          = source      # file name of originating image (for diagnostics)
        self.image           = image       # the grayscale image the detections were made in (and cells refer to)
        self.detections      = detections  # cells discovered in each detection
        self.logger          = logger      # for diagnostics
        self.cell_levels     = None        # cell luminance levels for each detection
        self.luminance_range = None        # black/white luminance range for each detection
        self.thresholds      = None        # black/white thresholds for each detection
        self.data_bits       = None        # data bits extracted from the cells

    def get_pixel(self, x: float, y: float) -> float:
        """ get the interpolated pixel value at x,y,
            x,y are fractional so the pixel value returned is a mixture of the 4 pixels around x,y,
            the mixture is based on the ratio of the neighbours to include, the ratio of all 4 is 1,
            code based on:
                void interpolateColorPixel(double x, double y) {
                    int xL, yL;
                    xL = (int) Math.floor(x);
                    yL = (int) Math.floor(y);
                    xLyL = ipInitial.getPixel(xL, yL, xLyL);
                    xLyH = ipInitial.getPixel(xL, yL + 1, xLyH);
                    xHyL = ipInitial.getPixel(xL + 1, yL, xHyL);
                    xHyH = ipInitial.getPixel(xL + 1, yL + 1, xHyH);
                    for (int rr = 0; rr < 3; rr++) {
                        double newValue = (xL + 1 - x) * (yL + 1 - y) * xLyL[rr];
                        newValue += (x - xL) * (yL + 1 - y) * xHyL[rr];
                        newValue += (xL + 1 - x) * (y - yL) * xLyH[rr];
                        newValue += (x - xL) * (y - yL) * xHyH[rr];
                        rgbArray[rr] = (int) newValue;
                    }
                }
            from here: https://imagej.nih.gov/ij/plugins/download/Polar_Transformer.java
            explanation:
            x,y represent the top-left of a 1x1 pixel
            if x or y are not whole numbers the 1x1 pixel area overlaps its neighbours,
            the returned pixel value is the sum of the overlap fractions of its neighbour
            pixel squares, P is the fractional pixel address in its pixel, 1, 2 and 3 are
            its neighbours, dotted area is contribution from neighbours:
                +------+------+
                |  P   |   1  |
                |  ....|....  |  Ax = 1 - (Px - int(Px) = 1 - Px + int(Px) = (int(Px) + 1) - Px
                |  . A | B .  |  Ay = 1 - (Py - int(Py) = 1 - Py + int(Py) = (int(Py) + 1) - Py
                +------+------+  et al for B, C, D
                |  . D | C .  |
                |  ....|....  |
                |  3   |   2  |
                +----- +------+
            """
        cX: float = x
        cY: float = y
        xL: int = int(cX)
        yL: int = int(cY)
        xH: int = xL + 1
        yH: int = yL + 1
        pixel_xLyL = self.image[yL][xL]
        pixel_xLyH = self.image[yH][xL]
        pixel_xHyL = self.image[yL][xH]
        pixel_xHyH = self.image[yH][xH]
        if pixel_xLyL is None:
            pixel_xLyL = const.MIN_LUMINANCE
        if pixel_xLyH is None:
            pixel_xLyH = const.MIN_LUMINANCE
        if pixel_xHyL is None:
            pixel_xHyL = const.MIN_LUMINANCE
        if pixel_xHyH is None:
            pixel_xHyH = const.MIN_LUMINANCE
        ratio_xLyL = (xH - cX) * (yH - cY)
        ratio_xHyL = (cX - xL) * (yH - cY)
        ratio_xLyH = (xH - cX) * (cY - yL)
        ratio_xHyH = (cX - xL) * (cY - yL)
        part_xLyL = pixel_xLyL * ratio_xLyL
        part_xHyL = pixel_xHyL * ratio_xHyL
        part_xLyH = pixel_xLyH * ratio_xLyH
        part_xHyH = pixel_xHyH * ratio_xHyH
        pixel = part_xLyL + part_xHyL + part_xLyH + part_xHyH
        return pixel

    def get_luminance(self, centre_x: float, centre_y: float, radius: float) -> float:
        """ get the average luminance level for all pixels in the circle of radius r at location x,y in our image """

        radius_squared = radius * radius

        def in_circle(x, y):
            """ return True iff x,y is inside, or on, the circle at centre_x/y with radius """
            x_len  = x - centre_x
            x_len *= x_len
            y_len  = y - centre_y
            y_len *= y_len
            distance = x_len + y_len
            if distance > radius_squared:
                # this x,y is outside our circle
                return False
            # we're on or inside
            return True

        level   = 0
        samples = 0
        min_x = centre_x - radius
        max_x = centre_x + radius
        min_y = centre_y - radius
        max_y = centre_y + radius
        x_dash = min_x
        while x_dash < max_x:
            y_dash = min_y
            while y_dash < max_y:
                if in_circle(x_dash, y_dash):
                    level   += self.get_pixel(x_dash, y_dash)
                    samples += 1
                y_dash += 1
            x_dash += 1
        return level / samples

    def get_cell_levels(self, target=None):
        """ get the luminance level of all relevant cells for the given target, None==all
            we treat the cells as if they are circles (so we do not need to consider rotations)
            """
        if self.cell_levels is None:
            # not got levels yet, get all relevant cell levels now for all detections
            self.cell_levels = []
            self.luminance_range = []
            for detection in range(len(self.detections)):
                rows = self.detections[detection]
                cell_levels = [[None for _ in range(len(rows[0]))] for _ in range(len(rows))]
                luminance   = [0, 0]
                samples     = [0, 0]
                for row, cells in enumerate(self.detections[detection]):
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
                            continue
                        level = self.get_luminance(x, y, r)
                        cell_levels[row][col] = [level, cell_type]
                        if cell_type == Extractor.BLACK_CELL:
                            luminance[0] += level
                            samples  [0] += 1
                        elif cell_type == Extractor.WHITE_CELL:
                            luminance[1] += level
                            samples  [1] += 1
                self.cell_levels.append(cell_levels)
                if samples[0] > 0:
                    luminance[0] /= samples[0]
                if samples[1] > 0:
                    luminance[1] /= samples[1]
                self.luminance_range.append(luminance)
        if target is None:
            return self.cell_levels
        else:
            return self.cell_levels[target]

    def get_luminance_range(self, target=None):
        """ get the luminance range for the given target, None==all """
        if self.luminance_range is None:
            self.get_cell_levels()
        if target is None:
            return self.luminance_range
        else:
            return self.luminance_range[target]
        
    def get_thresholds(self, target=None):
        """ get the black/white thresholds for the given target, None==all
            these are determined by the average of the 'black' and 'white' cell types,
            the dynamic range detected is split into three: black, grey, white,
            min..black is a 'black' cell, white..max is a 'white' cell, in between is 'grey' (could be either),
            the dynamic range is assumed to be constant across all cells
            """
        if self.thresholds is None:
            self.thresholds = []
            luminance = self.get_luminance_range()
            for detection, (black_average, white_average) in enumerate(luminance):
                threshold = max((white_average - black_average) / 3, 1)
                self.thresholds.append((black_average + threshold, white_average - threshold))
        if target is None:
            return self.thresholds
        else:
            return self.thresholds[target]

    def get_origin(self, detection):
        """ return the origin of the given detection in the image (diagnostic aid),
            the origin is the top-left cell centre in the detection
            """
        origin_x, origin_y, _, _ = self.detections[detection][0][0]
        return int(round(origin_x)), int(round(origin_y))

    def get_size(self, detection):
        """ get the size of the given detection,
            the size is the distance between the top-left cell and the bottom-right
            """

        rows = self.detections[detection]
        top_left     = rows[0][0]
        bottom_right = rows[-1][-1]

        x = bottom_right[0] - top_left[0]
        y = bottom_right[1] - top_left[1]

        return math.sqrt((x*x + y*y))

    def get_origins(self):
        """ get a list of origins and sizes for all detections """
        origins = []
        for detection in range(len(self.detections)):
            origins.append((self.get_origin(detection), self.get_size(detection)))
        return origins

    def get_bits(self, target=None):
        """ get the data bits for the given target, None==all,
            the bits are returned as a list of 0 (black) or 1 (white) or None (grey) in the order
            defined by Codes.DATA_CELLS
            """
        if self.data_bits is None:
            self.data_bits = []
            for detection, cell_levels in enumerate(self.get_cell_levels()):
                thresholds = self.get_thresholds(detection)
                bits = []
                bit_num = 0
                for cell in codes.Codes.DATA_CELLS:
                    if cell is None:
                        # just a spacer, ignore it
                        continue
                    level, typ = cell_levels[cell[1]][cell[0]]
                    if typ != Extractor.DATA_CELL:
                        raise Exception('Expected DATA_CELL at {} in detection {}, got {}'.format(cell, detection, typ))
                    if level <= thresholds[0]:
                        bit = 0
                    elif level >= thresholds[1]:
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
        if target is None:
            return self.data_bits
        else:
            return self.data_bits[target]
            

def find_bits(src, image, detections, logger):
    """ find code bits within the given detected cells """
    if logger is not None:
        logger.push('find_bits')
        logger.log('')
    extractor = Extractor(src, image, detections, logger)
    result = extractor.get_bits()
    if logger is not None:
        logger.pop()
    return result, extractor.get_origins()


def _test(src, proximity, blur=3, logger=None, create_new=True):
    """ ************** TEST **************** """

    if logger.depth() > 1:
        logger.push('extractor/_test')
    else:
        logger.push('_test')
    logger.log("\nExtracting code bits")

    # get the code areas
    cells, image = finder._test(src, proximity, blur=blur, logger=logger, create_new=create_new)

    # process the code areas
    bits, origins = find_bits(src, image, cells, logger)

    logger.pop()
    return bits, image, origins

if __name__ == "__main__":
    """ test harness """

    #src = "/home/dave/precious/fellsafe/fellsafe-image/media/kilo-codes/kilo-codes-distant.jpg"
    src = "/home/dave/precious/fellsafe/fellsafe-image/source/kilo-codes/test-alt-bits.png"
    proximity = const.PROXIMITY_CLOSE
    #proximity = const.PROXIMITY_FAR
    create_new = True

    logger = utils.Logger('extractor.log', 'extractor/{}'.format(utils.image_folder(src)))

    _test(src, proximity, blur=3, logger=logger, create_new=create_new)
