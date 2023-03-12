""" Draw/detect codewords primitives
    Codewords are characterised by a detection/orientation pattern and a data area.
    The detection/orientation pattern is implicit knowledge in this class.
    The data area structure is implicit knowledge, its contents come from outside
    or are delivered to outside.
    Structure overview:
    0____1____2____3____4____  5____  6____  7____  8____9____10___11___12___
    _________________________  _____  _____  _____  _________________________
    1____xxxxxxxxxxxxxxx_____  _____  _____  _____  _____xxxxxxxxxxxxxxx_____
    _____xxxxxxxxxxxxxxx_____  _____  _____  _____  _____xxxxxxxxxxxxxxx_____
    2____xxxxxxxxxxxxxxx_____  *****  _____  *****  _____xxxxxxxxxxxxxxx_____  <-- column markers
    _____xxxxxxxxxxxxxxx_____  *****  _____  *****  _____xxxxxxxxxxxxxxx_____  <--
    3____xxxxxxxxxxxxxxx_____  _____  _____  _____  _____xxxxxxxxxxxxxxx_____
    _____xxxxxxxxxxxxxxx_____  _____  _____  _____  _____xxxxxxxxxxxxxxx_____

    4         -----     _____  .....  .....  .....  _____     _____
              -----     _____  . A .  . B .  . C .  _____     _____
              -----     _____  .....  .....  .....  _____     _____

    5         *****_____.....  .....  .....  .....  ....._____*****
              *****_____. H .  . G .  . F .  . E .  . D ._____*****
              *****_____.....  .....  .....  .....  ....._____*****

    6         -----     .....  .....  .....  .....  .....     _____
              -----     . I .  . J .  . K .  . L .  . M .     _____
              -----     .....  .....  .....  .....  .....     _____

    7         *****_____.....  .....  .....  .....  ....._____*****
              *****_____. R .  . Q .  . P .  . O .  . N ._____*****
              *****_____.....  .....  .....  .....  ....._____*****

    8         _____     _____  .....  .....  .....  _____     _____
              _____     _____  . S .  . T .  . U .  _____     _____
              _____     _____  .....  .....  .....  _____     _____

    9____xxxxxxxxxxxxxxx_____  _____  _____  _____  _________________________
    _____xxxxxxxxxxxxxxx_____  _____  _____  _____  _________________________
    10___xxxxxxxxxxxxxxx_____  *****  _____  *****  __________mmmmm__________  <-- column markers
    _____xxxxxxxxxxxxxxx_____  *****  _____  *****  __________mmmmm__________  <--
    11___xxxxxxxxxxxxxxx_____  _____  _____  _____  _________________________
    _____xxxxxxxxxxxxxxx_____  _____  _____  _____  _________________________
    12_______________________  _____  _____  _____  _________________________
    0____1____2____3____4____  5____  6____  7____  8____9____10___11___12___
              ^^^^^                                           ^^^^^
              |||||                                           |||||
              +++++--------------- row markers ---------------+++++

    xx..xx are the 'major locator' blobs, top-left is the 'primary' locator,
    mm..mm is the 'minor locator' blob that is the same size as a marker blob
    **** are the row/column 'marker' blobs
    all locator and marker blobs are detected by their contour
    ....   is a 'bit box' that can be either a '1' (white) or a '0' (black)
    a '1' is white area (detected by its relative luminance)
    a '0' is black area (detected by its relative luminance)
    the background (____) is white (paper) and the blobs are black holes in it
    A..U are the bits of the codeword (21)
    their centre co-ordinates and size are calculated from the marker blobs
    numbers in the margins are 'cell' addresses (in units of the width of a marker blob)
    only cell addresses 2,2..10,10 are 'active' in the sense they are detected and processed
    Note: The bottom-right 'minor locator' is much smaller than the others but its centre
          still aligns with bottom-left and top-right 'major locator' blobs
"""

import random
import canvas
import const
import utils
import crc

# ToDo: can we filter out codewords we don't want and still have enough codes?
#       e.g. all 1's, all 0's, all 1's except one 0, all 0's except one 1, etc

class Codes:

    WHITE = const.MAX_LUMINANCE
    BLACK = const.MIN_LUMINANCE

    # region Geometry constants...
    # all drawing co-ordinates are in units of a 'cell' where a 'cell' is the minimum width between
    # luminance transitions, cell 0,0 is the top left of the canvas
    # the major locator blobs are 3x3 cells starting at 1,1
    # the minor locator blob is 1x1 cells
    # the marker blobs and data blobs are 1x1 cells, marker blobs are separated by at least 1 cell
    LOCATORS   = [(1,1), (1,2), (1,3), None,(9,1), (10,1), (11,1),  # NB: these co-ords are relative to the canvas
                  (2,1), (2,2), (2,3), None,(9,2), (10,2), (11,2),
                  (3,1), (3,2), (3,3), None,(9,3), (10,3), (11,3),
                  None,
                  (1,9) ,(2,9) ,(3,9),
                  (1,10),(2,10),(3,10),None,(10,10),
                  (1,11),(2,11),(3,11)]
    MARKERS    = [None, (5,2), None,(7,2),  # NB: these co-ords are relative to the canvas
                  None,
                  (2,5), None, None, None, (10,5),
                  None,
                  (2,7), None, None, None, (10,7),
                  None,
                  None, (5,10),None,(7,10)]
    STRUCTURE  = LOCATORS + MARKERS
    DATA_OFFSET = [2,2]  # data-bits offset from the canvas origin (0,0) - add this to DATA_BITS when drawing
    DATA_BITS  = [None, (3,2),(4,2),(5,2),None,   # NB: these co-ords are relative to the active area
                  (6,3),(5,3),(4,3),(3,3),(2,3),  # a None in here is just a spacing convenience and should be ignored
                  (2,4),(3,4),(4,4),(5,4),(6,4),
                  (6,5),(5,5),(4,5),(3,5),(2,5),
                  None, (3,6),(4,6),(5,6),None]
    NAME_CELL  = (9,12)  # canvas co-ordinate of the bottom-left of the text label
    MAX_X_CELL = 12  # canvas size
    MAX_Y_CELL = 12  # ..
    # endregion
    # region Public constants...
    LOCATORS_PER_CODE = 3  # how many 'major' locators there are per code
    TIMING_PER_CODE   = 9  # how mnay 'timing' marks there are per code (including the 'minor' locator)
    LOCATOR_SCALE = 3  # size of major locators relative to markers (so radius of enclosing circle is half this)
    TIMING_SCALE  = 1 / LOCATOR_SCALE  # size of timing marks relative to locators
    LOCATOR_SPAN = 8  # distance between locator centres in units of *marker width*
    LOCATOR_SPACING = LOCATOR_SPAN / (LOCATOR_SCALE / 2)  # locator spacing in units of *locator radius*
    # These cell positions are relative to the 'active' area of the code (see visualisation above)
    TIMING_CELLS  = [3, 5]               # timing mark cell positions along a line between locators (all 4 sides)
    DATA_CELLS    = DATA_BITS            # public face of the data bits (same as internal as it happens)
    BLACK_CELLS   = [(0,0),None,None,(3,0),None,(5,0),None,None,(8,0),  # active cell areas guaranteed to be black
                     None,
                     None,
                     (0,3),None,None,None ,None,None ,None,None,(8,3),
                     None,
                     (0,5),None,None,None ,None,None ,None,None,(8,5),
                     None,
                     None,
                     (0,8),None,None,(3,8),None,(5,8),None,None,(8,8)]
    WHITE_CELLS   = [None ,None,(2,0),None,(4,0),None,(6,0),None,None,  # active cell areas guaranteed to be white
                     None,
                     (0,2),None,None ,None,None ,None,None ,None,(8,2),
                     None,
                     (0,4),None,None ,None,None ,None,None ,None,(8,4),
                     None,
                     (0,6),None,None ,None,None ,None,None ,None,(8,6),
                     None,
                     None ,None,(2,8),None,(4,8),None,(6,8),None,None]
    # endregion

    def __init__(self, image_buffer):
        self.canvas = image_buffer  # where we do our drawing or detecting (when drawing it should be square)
        self.max_x, self.max_y = canvas.size(self.canvas)
        self.code_span   = min(self.max_x, self.max_y)
        self.cell_width  = int(round(self.code_span / (self.MAX_X_CELL + 1)))
        self.cell_height = self.cell_width

    # region Drawing functions...
    def draw_codeword(self, codeword: int, name: str):
        """ codeword is the A..P data bits,
            name is the readable version of the codeword and is drawn alongside the codeword,
            the codeword is drawn as large as possible within the canvas
            """
        self.clear_canvas()
        self.draw_structure(self.STRUCTURE)
        bit = 0
        for cell in self.DATA_BITS:
            if cell is None:
                # these are just convenience spacers
                continue
            cell = (cell[0] + Codes.DATA_OFFSET[0],  # map from active area to canvas
                    cell[1] + Codes.DATA_OFFSET[1])  # ..
            mask = 1 << bit
            if codeword & mask == 0:
                self.draw_cell(cell)
            bit += 1
        self.draw_name(self.NAME_CELL, name)

    def draw_name(self, position: (int, int), name:str):
        """ draw the codeword name """
        canvas.settext(self.canvas, name, self.cell2pixel(position), size=0.5)

    def draw_cell(self, position: (int, int), colour: int=BLACK):
        """ draw a cell - i.e. make the area black """
        start_x, start_y = self.cell2pixel(position)
        end_x = start_x + self.cell_width
        end_y = start_y + self.cell_height
        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                canvas.putpixel(self.canvas, x, y, colour)

    def clear_canvas(self):
        """ make the entire codeword structure white """
        for x in range(self.MAX_X_CELL + 1):
            for y in range(self.MAX_Y_CELL + 1):
                self.draw_cell((x, y), self.WHITE)

    def draw_structure(self, structure: [(int, int)]):
        """ draw the given structure (==a list of black cell addresses) """
        for cell in structure:
            if cell is None:
                # these are just convenience spacers
                continue
            self.draw_cell(cell)

    def cell2pixel(self, cell: (int, int)) -> (int, int):
        """ translate a cell address to its equivalent pixel address in our canvas """
        x = int(round(cell[0] * self.cell_width))
        y = int(round(cell[1] * self.cell_height))
        return x, y
    # endregion


if __name__ == "__main__":
    """ test harness """

    CELL_WIDTH  = 42  # creates an image that fits on A5
    CELL_HEIGHT = 42  # ..

    TEST_PATTERN = 0b010_01010_10101_01010_010
    TEST_CODES   = 10  # how many test codes to make

    TEST_PATTERN_NAME  = 'test-alt-bits'
    TEST_PATTERN_LABEL = 'Alt zero and one'

    logger = utils.Logger('codes.log', 'codes')
    logger.log('Creating {} codes and the test pattern with cells {} x {}'.format(TEST_CODES, CELL_WIDTH, CELL_HEIGHT))

    codec = crc.make_codec(logger)

    def draw_code(code, label, name):
        logger.log('')
        logger.log('Creating code {:21b}...'.format(code))
        image = canvas.new((Codes.MAX_X_CELL + 1) * CELL_WIDTH, (Codes.MAX_Y_CELL + 1) * CELL_HEIGHT)
        codes = Codes(image)
        codes.draw_codeword(code, label)
        logger.draw(image, file=name)

    draw_code(TEST_PATTERN, TEST_PATTERN_LABEL, TEST_PATTERN_NAME)

    for test in range(TEST_CODES):
        code = random.randrange(101, 999+1)
        codeword = codec.encode(code)
        draw_code(codeword, '{:03d}'.format(code), 'test-code-{:03d}'.format(code))

    logger.log('')
    logger.log('Done')
