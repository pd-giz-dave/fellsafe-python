""" Find the cells in the code area
    This module takes a list of locator and timing mark detections and finds the associated data cell co-ordinates
"""

import math
import const
import utils
import finder
import canvas


class Params(finder.Params):
    def __init__(self):
        self.cells = None

class Cells:

    def __init__(self, source, image, grids, images, logger=None, origins=None):
        self.source        = source   # originating image file name (for diagnostics)
        self.image         = image    # grayscale image the detections were detected within (and co-ordinates apply to)
        self.code_grids    = grids    # list of detected timing grids
        self.sub_images    = images   # list of greyscale images the grids where extracted from
        self.logger        = logger   # for diagnostics
        self.origins       = origins  # origins of all the sub-images in the originating iamge
        self.intersections = None     # all the grid line intersections, in row then column order
        self.code_cells    = None     # list of cell co-ords inside the locators for all filtered timings

    def get_grayscale(self, detection):
        """ get the grayscale sub-image of the given detection """
        image, _ = self.sub_images[detection]
        return image

    def get_colour_image(self, detection):
        """ extract the grayscale sub-image of the given detection and colourize it """
        image = self.get_grayscale(detection)
        return canvas.colourize(image)

    def get_image_size(self, detection):
        """ get the size of the sub-image for the given detection """
        image = self.get_grayscale(detection)
        return canvas.size(image)

    def draw(self, image, file, detection):
        folder = utils.image_folder(target=self.origins[detection])
        self.logger.push(context=folder, folder=folder)
        self.logger.draw(image, file=file)
        self.logger.pop()

    def draw_cells(self):
        """ draw the detected code cells for diagnostic purposes """
        if self.logger is None:
            return

        for detection, rows in enumerate(self.cells()):
            image = self.get_colour_image(detection)
            for row, cells in enumerate(rows):
                for col, (x, y, r, _) in enumerate(cells):
                    if row == 0 and col == 0:
                        # highlight reference point
                        colour = const.RED
                    elif (row & 1) == 0 and (col & 1) == 0:
                        # even row and even col
                        colour = const.GREEN
                    elif (row & 1) == 0 and (col & 1) == 1:
                        # even row and odd col
                        colour = const.BLUE
                    elif (row & 1) == 1 and (col & 1) == 0:
                        # odd row and even col
                        colour = const.BLUE
                    elif (row & 1) == 1 and (col & 1) == 1:
                        # odd row and odd col
                        colour = const.GREEN
                    image = canvas.circle(image, (x, y), r, colour)
            self.draw(image, 'cells', detection)

    def get_intersections(self):
        """ get an array of all row, column intersections for all detections """
        if self.intersections is not None:
            return self.intersections

        self.intersections = []
        for detection, (_, top, right, bottom, left) in enumerate(self.code_grids):
            size = self.get_image_size(detection)
            box  = ((0, 0), (size[0] - 1, size[1] - 1))  # x,y limits for extend
            rows = []
            for row in range(len(left)):
                cols = []
                for col in range(len(top)):
                    line1 = utils.extend(top[col], bottom[col], box)  # extend to image edge
                    line2 = utils.extend(left[row], right[row], box)  # ..to ensure they always intersect
                    cols.append(utils.intersection(line1, line2))
                rows.append(cols)
            self.intersections.append(rows)

        return self.intersections

    def cells(self):
        """ produce the co-ordinates of all cells inside the locators relative to the original image,
            grid addresses are in clockwise order starting at the primary corner (top-left when not rotated)
            and relative to the sub-image of the extracted target, cell addresses are row (top to bottom) then
            column (left to right), i.e. 'array' format, and relative to the original image,
            cell co-ordinates represent the centre of the cell, and a radius is the maximum circle radius that
            fits inside the cell
            """
        if self.code_cells is not None:
            # already done it
            return self.code_cells

        NEIGHBOURS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # cell neighbour row,col offsets from self

        def get_radius(col, row):
            """ determine the cell width for the given column and row intersection """
            origin = rows[row][col]
            distance = 0
            samples = 0
            for row_offset, col_offset in NEIGHBOURS:
                neighbour_row = row + row_offset
                if neighbour_row < 0 or neighbour_row >= len(rows):
                    # no neighbour here
                    continue
                neighbour_col = col + col_offset
                if neighbour_col < 0 or neighbour_col >= len(cols):
                    # no neighbour here
                    continue
                distance += utils.distance(origin, rows[neighbour_row][neighbour_col])
                samples += 1
            distance /= samples
            # distance is now average distance (squared) to all our neighbour cells
            # our radius is half the square root of that
            radius = math.sqrt(distance) / 2
            return radius

        self.code_cells = []
        for rows in self.get_intersections():
            code_rows = []
            for row, cols in enumerate(rows):
                cells = []
                for col, cell in enumerate(cols):
                    radius = get_radius(col, row)
                    cells.append([cell[0], cell[1], radius, None])
                code_rows.append(cells)
            self.code_cells.append(code_rows)
        if self.logger is not None:
            self.logger.push('cells')
            self.draw_cells()
            self.logger.pop()
        return self.code_cells

    def images(self):
        """ return the greyscale sub-images of all our detections """
        return self.sub_images


def find_cells(src, params, image, detections, logger):
    """ find the valid code cell areas within the given detections """
    if logger is not None:
        logger.push('find_cells')
        logger.log('')
    grids, images, origins = detections
    cells = Cells(src, image, grids, images, logger, origins)
    cells.cells()
    params.cells = cells
    if logger is not None:
        logger.pop()
    return params  # for upstream access


def _test(src, proximity, blur, mode, params=None, logger=None, create_new=True):
    """ ************** TEST **************** """
    
    if logger.depth() > 1:
        logger.push('cells/_test')
    else:
        logger.push('_test')
    logger.log('')
    logger.log('Finding cells (create new {})'.format(create_new))

    # get the detections
    if not create_new:
        params = logger.restore(file='cells', ext='params')
        if params is None or params.source_file != src:
            create_new = True
    if create_new:
        # this is very slow
        if params is None:
            params = Params()
        params = finder._test(src, proximity, blur=blur, mode=mode, params=params, logger=logger, create_new=create_new)
        if params is None:
            logger.log('Finder failed on {}'.format(src))
            logger.pop()
            return None
        logger.save(params, file='cells', ext='params')

    found = params.finder
    image = params.source
    detections = found.get_detections()

    # process the detections
    params = find_cells(src, params, image, detections, logger)
    logger.pop()
    return params  # for upstream test harness


if __name__ == "__main__":
    """ test harness """

    #src = '/home/dave/precious/fellsafe/fellsafe-image/source/kilo-codes/codes/test-alt-bits.png'
    #src = '/home/dave/precious/fellsafe/fellsafe-image/source/kilo-codes/codes/test-code-145.png'
    src = '/home/dave/precious/fellsafe/fellsafe-image/media/kilo-codes/kilo-codes-close-150-257-263-380-436-647-688-710-777.jpg'
    #proximity = const.PROXIMITY_CLOSE
    proximity = const.PROXIMITY_FAR

    logger = utils.Logger('cells.log', 'cells/{}'.format(utils.image_folder(src)))

    _test(src, proximity, blur=const.BLUR_KERNEL_SIZE, mode=const.RADIUS_MODE_MEAN, logger=logger, create_new=False)
