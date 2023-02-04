""" Draw/detect codewords primitives
    Codewords are characterised by a detection/orientation pattern and a data area.
    The detection/orientation pattern is implicit knowledge in this class.
    The data area structure is implicit knowledge, its contents come from outside
    or are delivered to outside.
    Structure overview:
    0____1____2____3____  4____  5____  6____  7____8____9____10___
    ____________________  _____  _____  _____  ____________________
    1____xxxxxxxxxx_____  *****  _____  *****  _____xxxxxxxxxx_____  <--
    _____xxxxxxxxxx_____  *****  _____  *****  _____xxxxxxxxxx_____  <-- column markers
    2____xxxxxxxxxx_____  _____  _____  _____  _____xxxxxxxxxx_____  <--
    _____xxxxxxxxxx_____  _____  _____  _____  _____xxxxxxxxxx_____  <--

    3    -----     _____  .....  .....  .....  _____     _____
         -----     _____  . A .  . B .  . C .  _____     _____
         -----     _____  .....  .....  .....  _____     _____

    4    *****_____.....  .....  .....  .....  ....._____*****
         *****_____. H .  . G .  . F .  . E .  . D ._____*****
         *****_____.....  .....  .....  .....  ....._____*****

    5    -----     .....  .....  .....  .....  .....     _____
         -----     . I .  . J .  . K .  . L .  . M .     _____
         -----     .....  .....  .....  .....  .....     _____

    6    *****_____.....  .....  .....  .....  ....._____*****
         *****_____. R .  . Q .  . P .  . O .  . N ._____*****
         *****_____.....  .....  .....  .....  ....._____*****

    7    _____     _____  .....  .....  .....  _____     _____
         _____     _____  . S .  . T .  . U .  _____     _____
         _____     _____  .....  .....  .....  _____     _____

    8____xxxxxxxxxx_____  _____  _____  _____  _____*****_____       <--
    _____xxxxxxxxxx_____  _____  _____  _____  _____*****_____       <--
    9____xxxxxxxxxx_____  *****  _____  *****  _____     _____       <-- column markers
    _____xxxxxxxxxx_____  *****  _____  *****  _____     _____       <--
    10__________________  _____  _____  _____  ____________________
    0____1____2____3____  4____  5____  6____  7____8____9____10___
            ^^^^                                       ^^^^^
            ||||                                       |||||
            ++++-------------- row markers ------------+++++

    xx..xx ares the locator blobs that are detected via their contour
    **** are the row/column marker blobs, also detected by their contour
    ....   is a 'bit box' that can be either a '1' (white) or a '0' (black)
    a '1' is white area (detected by its relative luminance)
    a '0' is black area (detected by its relative luminance)
    the background (____) is white (paper) and the blobs are black holes in it
    A..U are the bits of the codeword (21)
    their centre co-ordinates and size are calculated from the marker blobs
    numbers in the margins are 'cell' addresses (in units of the width of a marker blob)
"""

import frame
import const

class Codes:

    WHITE = const.MAX_LUMINANCE
    BLACK = const.MIN_LUMINANCE

    # region Geometry constants...
    # all drawing co-ordinates are in units of a 'cell' where a 'cell' is the minimum width between
    # luminance transitions, cell 0,0 is the top left of the canvas
    # the locator blobs are 2x2 cells starting at 1,1
    # the marker blobs and data blobs are 1x1 cells separated by at least 1 cell
    LOCATORS   = [(1,1),(1,2),None,(8,1),(9,1),
                  (2,1),(2,2),None,(8,2),(9,2),
                  None,
                  (1,8),(2,8),
                  (1,9),(2,9)]
    MARKERS    = [None,(4,1),None,(6,1),
                  None,
                  (1,4),None,(9,4),
                  None,
                  (1,6),None,(9,6),
                  None,
                  None,None ,None,(8,8),
                  None,(4,9),None,(6,9)]
    STRUCTURE  = LOCATORS + MARKERS
    DATA_BITS  = [None, (4,3),(5,3),(6,3),None,
                  (7,4),(6,4),(5,4),(4,4),(3,4),
                  (3,5),(4,5),(5,5),(6,5),(7,5),
                  (7,6),(6,6),(5,6),(4,6),(3,6),
                  None, (4,7),(5,7),(6,7),None]
    NAME_CELL  = (8.5,10.5)  # co-ordinate of the bottom-left of the text
    MAX_X_CELL = 10
    MAX_Y_CELL = 10
    # endregion
    # region Public constants...
    CODE_EXTENT = max(MAX_X_CELL, MAX_Y_CELL)  # max size of the entire code in units of the 'radius' of a locator blob
    LOCATOR_SPACING = 7  # distance between locator blob centres in units of the 'radius' of a locator blob
    LOCATORS_PER_CODE = 3  # how many locators there are per code
    # endregion

    def __init__(self, canvas: frame.Frame):
        self.canvas = canvas  # where we do our drawing or detecting (when drawing it should be square)
        self.max_x, self.max_y = canvas.size()
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
            mask = 1 << bit
            if codeword & mask == 0:
                self.draw_cell(cell)
            bit += 1
        self.draw_name(self.NAME_CELL, name)

    def draw_name(self, position: (int, int), name:str):
        """ draw the codeword name """
        x, y = self.cell2pixel(position)
        self.canvas.settext(name, x, y, size=0.5)

    def draw_cell(self, position: (int, int), colour: int=BLACK):
        """ draw a cell - i.e. make the area black """
        start_x, start_y = self.cell2pixel(position)
        end_x = start_x + self.cell_width
        end_y = start_y + self.cell_height
        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                self.canvas.putpixel(x, y, colour)

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

    def cell2pixel(self, cell) -> (int, int):
        """ translate a cell address to its equivalent pixel address in our canvas """
        x = int(round(cell[0] * self.cell_width))
        y = int(round(cell[1] * self.cell_height))
        return x, y
    # endregion


if __name__ == "__main__":
    """ test harness """

    CELL_WIDTH  = 48
    CELL_HEIGHT = 48

    image = frame.Frame()
    image.new((Codes.MAX_X_CELL + 1) * CELL_WIDTH, (Codes.MAX_Y_CELL + 1) * CELL_HEIGHT)
    codec = Codes(image)
    codec.draw_codeword(0b010_01010_10101_01010_010, 'TEST01...')
    image.unload('test-alt-bits.png')
