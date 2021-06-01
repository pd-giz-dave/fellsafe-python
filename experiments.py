from PIL import Image, ImageDraw, ImageFont
import random

""" coding scheme

    The coding scheme is designed to be easy to detect and robust against noise, defects, lighting, perspective
    and orientation effects. Its only intended to encode a number in the range 0..1023 (i.e. 10 bits). The
    encoding is such that the average luminance across each bit pair is the same. The coding is based on the
    deviation from the mean. There are five deviation levels: big positive (H), small positive (h), zero (.),
    small negative (l) and big negative (L). A one-bit consists of a pulse with this profile: .HH.., and a
    zero-bit profile is: .lLl. Because cameras are better are detecting light than they are at darkness, the
    zero-bit is wider with a lesser slope. The average is still the same. Each deviation level consists of at
    least three pixels. Bit-pairs are encoded into four pulses such that the number of zero bits and one bits
    are the same. In four bits there are six combinations that have the same number of zero and one bits, four
    are used for data and one for a prefix and postfix marker. One symbol is not used. Each group of four bits
    is bracketed by a quiet-zone of at least two bits (zero deviation). The entire code for a ten bit number is
    thus seven groups of four: prefix data*5 postfix. The six allowed four-bit combinations are allocated as
    follows:
    
        ..0101.. = prefix and postfix  (alternating 0's and 1' to get bit timing)
        ..1010.. = not used (but is prefix backwards)
        ..0011.. = 0
        ..0110.. = 1
        ..1001.. = 2
        ..1100.. = 3
    
    The leading and trailing quiet bits are such that a sliding window of a fixed size can be used to find bit
    groups without that window overlapping more than one group. That is critical to ensure an accurate mean is
    found for the group. Also, to ensure there is always a 'middle' value as a boundary between luminance levels
    the luminance range is multiplied by four.
    
    The code is drawn as circular rings so that orientation does not matter. The code is considered to start in
    the centre (prefix) and end at the outer edge (postfix). The very centre is a large quiet-zone. The prefix
    and postfix codes can be used to determine which direction the overall code has been scanned in. The bit
    timing can be determined by dividing the quiet-zone to quiet-zone gap across a bit group by four. 
                
    """

# colours
black = 0                                # low extreme
white = 255                              # high extreme
grey  = (white - black) >> 1             # mid-point
empty = white                            # clear colour for a new canvas

# luminance levels (for drawing the code)
luminance_range = white - black
luminance_band  = luminance_range >> 2   # band width of each of our 4 bands
L = 0  # maps to black
l = 1
Z = 2  # maps to grey
h = 3
H = 4  # maps to white

# geometry (elements and their deviation)
one_bit    = [H, H, H]                         # one-bit and zero-bit are same length
zero_bit   = [L, L, L]                         # and have same average luminance deviation from a group_gap
centre     = [Z]                               # initial central element, must be an odd number of elements
group_edge = [Z, Z, Z]                         # leadin/out to a bit-group (used for group edge detection)
group_gap  = [group_edge, Z, group_edge]       # quiet gap between bit groups (must be at least 2*group_edge)
leadin     = [group_gap, group_gap]            # inner edge (must be bigger than the group-gap)
leadout    = group_gap                         # outer edge
marker     = [zero_bit, one_bit, zero_bit, one_bit]  # start/stop bit-group
# all the following bit groups are the same size and are such that their average luminance is in the middle
prefix     = [group_gap, marker, group_gap]
postfix    = [marker, group_gap]               # NB: drawn backwards, so get opposite pattern when read forwards
zero_zero  = [zero_bit, zero_bit, one_bit, one_bit, group_gap]
zero_one   = [zero_bit, one_bit, one_bit, zero_bit, group_gap]
one_zero   = [one_bit, zero_bit, zero_bit, one_bit, group_gap]
one_one    = [one_bit, one_bit, zero_bit, zero_bit, group_gap]
bit_group  = one_one                           # arbitrary bit-group for code proforma
# proforma code (used to calculate its size)
code_shape = [centre, leadin, prefix, bit_group, bit_group, bit_group, bit_group, bit_group, postfix, leadout]

####################################################
# Image encoding
####################################################


class Octant:
    """ an iterator that returns a series of x,y co-ordinates for a quadrant of circle of radius r
        it uses the Bresenham algorithm: https://www.geeksforgeeks.org/bresenhams-circle-drawing-algorithm/
        """
    def __init__(self, r):
        # define a circle of radius r
        self.r = r

    def __iter__(self):
        # init and return first x,y tuple
        self.x = 0
        self.y = self.r
        self.d = 3 - 2 * self.r
        return self

    def __next__(self):
        # return next tuple or raise StopIteration
        ret = (self.x, self.y)
        if self.y >= self.x:
            self.x += 1
            if self.d > 0:
                self.y -= 1
                self.d = self.d + 4 * (self.x - self.y) + 10
            else:
                self.d = self.d + 4 * self.x + 6
            return ret
        else:
            raise StopIteration


class Circle:
    """ draw a circle of radius r inscribed in a rectangle at x,y with a line width of w and a shade of s on canvas c
        the shade is one of L,l,Z,h,H (0..5) which is mapped to 0..255 by multiplying by the luminance_band
        """
    def __init__(self, r, x=0, y=0, w=1, s=Z, c=None):
        self.r = r
        self.x = x + r
        self.y = y + r
        self.w = w
        self.c = c
        self.s = (s * luminance_band, s * luminance_band, s * luminance_band)

    def point(self, x, y):
        """ to ensure there are no 'edge' effect artifacts all the pixel neighbours are drawn as well
            this is slow, but don't care when drawing the codes
            """
        self.c.point([x-1, y-1], fill=self.s)
        self.c.point([x  , y-1], fill=self.s)
        self.c.point([x+1, y-1], fill=self.s)
        self.c.point([x-1, y  ], fill=self.s)
        self.c.point([x  , y  ], fill=self.s)
        self.c.point([x+1, y  ], fill=self.s)
        self.c.point([x-1, y+1], fill=self.s)
        self.c.point([x  , y+1], fill=self.s)
        self.c.point([x+1, y+1], fill=self.s)

    def draw(self):
        """ draw a circle using the Bresenham algorithm
            the Octant class iterates all the raster co-ords for one quadrant then these are replicated
            in all the other 7
            """
        for r in range(self.r - self.w, self.r+1):
            for dx,dy in Octant(r):
                self.point(self.x+dx, self.y+dy)
                self.point(self.x-dx, self.y+dy)
                self.point(self.x+dx, self.y-dy)
                self.point(self.x-dx, self.y-dy)
                self.point(self.x+dy, self.y+dx)
                self.point(self.x-dy, self.y+dx)
                self.point(self.x+dy, self.y-dx)
                self.point(self.x-dy, self.y-dx)


class Encoder:
    """ class to encode a number in an image """

    # properties
    draw     = None
    image_x  = None
    image_y  = None
    centre_x = None
    centre_y = None
    width    = None
    contrast = None
    grey     = None
    pencil   = None
    pairs    = None

    def __init__(self, canvas):
        """ canvas is the (Pillow) image to draw on """

        # canvas size
        self.image_x = int(canvas.width)
        self.image_y = int(canvas.height)

        # drawing surface
        self.draw = ImageDraw.Draw(canvas)

    def draw_reset(self, clear=True, element_width=3, offset=0, contrast=1, lighting=0):
        """ prepare to draw (another) target on our canvas
            iff clear is True the existing image is removed (by drawing a zero_band rectangle over the lot)
            element_width is the size in pixels of each deviation ring
            these are all distortion properties to test the detection algorithm:
                offset is how far from centre to draw the code,
                contrast is the compression ratio for the luminance (2==halve the band gap, 4==quarter it, etc)
                lighting (0..255) is the offset from Z for the luminance of an element (can be +ve or -ve)
            """

        # where the centre of the canvas is
        self.centre_x = min(int(self.image_x / 2) + offset, self.image_x)
        self.centre_y = min(int(self.image_y / 2) + offset, self.image_y)

        # element width
        self.width = int(element_width)

        # luminance settings
        self.contrast = contrast
        self.grey = lighting

        # current ring drawing position
        self.pencil = 0

        # bit pair lookups
        self.pairs = [zero_zero, zero_one, one_zero, one_one]

        if clear:
            fill = (empty, empty, empty)
            self.draw.rectangle([(0, 0), (self.image_x, self.image_y)], fill=fill)

    def draw_ring(self, luminance):
        """ draw a ring at the current pencil position and move the pencil for the next ring
            luminance is one of H, h, Z, l, L
            """
        # calc shade by compressing the deviation as defined by contrast then adding the offset
        shade = int(max(min(((luminance - Z) / self.contrast) + Z + self.grey, white), black))
        self.pencil += self.width        # move to next drawing position
        Circle(self.pencil, self.centre_x-self.pencil, self.centre_y-self.pencil, self.width, shade, self.draw).draw()
        return self.width

    def draw_object(self, elements):
        """ draw the set of rings as defined by the given array of elements
            items in the array are luminance levels (H,h,Z,l,L) or an array of same
            """
        object_size = 0
        for element in elements:
            if type(element) == list or type(element) == tuple:
                object_size += self.draw_object(element)
            else:
                object_size += self.draw_ring(element)
        return object_size

    def draw_centre(self):
        """ draw the central element
            the central element is half size as its the point of rotation
            the width must be such that it gets drawn solid (i.e. width is more than radius)
            """
        width = self.width
        self.width = int((self.width+1)/2)  # round up
        size = self.draw_object(centre)
        self.width = width
        return size

    def draw_text(self, txt, size):
        """ draw a text string at the current pencil position of the given font size
            """
        fnt = ImageFont.truetype('/usr/share/fonts/liberation/LiberationSans-Bold.ttf', size)
        self.draw.text((self.centre_x - self.pencil, self.centre_y - self.pencil), txt, font=fnt,
                       fill=(black, black, black))

    def draw_target(self, number):
        """ draw the code image for the given number
            the target is drawn from the central pre-amble outwards
            """

        if number < 0 or number > 1023:
            raise Exception('Target can only be 1..1023, {} not allowed'.format(number))

        target_size = self.draw_centre()
        target_size += self.draw_object(leadin)
        target_size += self.draw_object(prefix)
        bits = int(number)
        for pair in range(5):
            target_size += self.draw_object(self.pairs[bits & 3])
            bits >>= 2
        target_size += self.draw_object(postfix)
        target_size += self.draw_object(leadout)
        # move drawing position such that text does not overlap our target
        self.pencil += self.width * 5
        self.draw_text('{:04}'.format(number), int(target_size / 5))

        return target_size

    def draw_noise(self, noise_level=0, noise_size=0):
        """ draw noise over the image
            noise_level is % elements of noise, 0=none
            noise_size is max size of noise in pixels (as random ellipse), 0=none
            """
        if noise_level > 0 and noise_size > 0:
            drawn = 0
            to_draw = int((self.image_x / noise_size * 2) * (self.image_y / noise_size * 2) * noise_level / 100)
            while drawn < to_draw:
                random_x = random.randrange(0, self.image_x)
                random_y = random.randrange(0, self.image_y)
                random_w = random.randrange(1, noise_size)
                random_h = random.randrange(1, noise_size)
                random_l = random.randrange(black, white)
                self.draw.ellipse([(random_x, random_y), (random_x + random_w, random_y + random_h)],
                                  fill=(random_l, random_l, random_l))
                drawn += 1


# draw test target(s) and save the result(s)
if False:
    image = Image.new("RGB", (256 * 4, 256 * 4), (grey, grey, grey))
    encoder = Encoder(image)
    test_set = [0, 341, 682, 795, 1023]  # all zeroes, all ones, alternating in each direction, every pair transition
    for num in test_set:
        encoder.draw_reset(element_width=3)
        # code.draw_noise(noise_level=100, noise_size=code.width * 3)  # under code noise
        test_size = encoder.draw_target(num)
        # code.draw_noise(noise_level=3, noise_size=code.width * 2)  # over code noise
        image.save('{:04}.jpg'.format(num))
        # image.show()
        print('Code {} is {} pixels wide, saved as {:04}.jpg'.format(num, test_size, num))
    image.close()


#########################################
# Image decoding
#########################################


class Decoder:
    """ class to decode numbers in an image """

    def __init__(self, image_file):
        """ get all the luminance into a 2D byte array for the given image name
            cells, n and bins are parameters to later analysis methods
            """

        im = Image.open(image_file)
        im.convert(mode='L')
        # im.show()

        self.max_x = im.size[0]
        self.max_y = im.size[1]
        self.centre_x = int(self.max_x/2)
        self.centre_y = int(self.max_y/2)

        self.image_array = [[im.getpixel((x, y))[0] for y in range(self.max_y)] for x in range(self.max_x)]

        # size of a bit group and a complete code in number of elements
        self.group_size = len(self.expand(bit_group))
        self.code_size  = len(self.expand(code_shape))

        # pixel size limits when scanning for prefix
        self.min_n = 3
        self.max_n = int(min(self.max_x, self.max_y) / self.code_size)

        # mapping for resolving 'maybe' levels (see binarize)
        self.h_map = [
                        [l, l, Z, Z, Z],   # L -->h?-->L,l?,Z,h?,H
                        [l, l, Z, Z, Z],   # l?-->h?-->L,l?,Z,h?,H
                        [Z, Z, Z, h, h],   # Z -->h?-->L,l?,Z,h?,H
                        [Z, Z, h, h, h],   # h?-->h?-->L,l?,Z,h?,H
                        [Z, Z, h, h, h],   # H -->h?-->L,l?,Z,h?,H
                     ]
        self.l_map = [
                        [l, l, l, Z, Z],   # L -->l?-->L,l?,Z,h?,H
                        [l, l, l, Z, Z],   # l?-->l?-->L,l?,Z,h?,H
                        [l, l, Z, Z, h],   # Z -->l?-->L,l?,Z,h?,H
                        [Z, Z, Z, h, h],   # h?-->l?-->L,l?,Z,h?,H
                        [Z, Z, h, h, H],   # H -->l?-->L,l?,Z,h?,H
                     ]

        # map deviation levels to bits (see bits)
        self.bit_map = [
                            0,             # L -->0
                            0,             # l?-->0
                            None,          # Z -->not a bit
                            1,             # h?-->1
                            1,             # H -->1
                       ]

        # valid bit combinations in a bit-group (see bits)
        valid_bits = [
            None,  # 0000
            None,  # 0001
            None,  # 0010
            0,     # 0011
            None,  # 0100
            -1,    # 0101 - marker
            1,     # 0110
            None,  # 0111
            None,  # 1000
            2,     # 1001
            None,  # 1010 - reverse marker
            None,  # 1011
            3,     # 1100
            None,  # 1101
            None,  # 1110
            None,  # 1111
        ]

    def merge(self, x, y):
        """ merge 3 x 3 pixels around x,y
            weights are 1,1,1  -,1,1 when x=0, -,-,- when y=0, 1,1,- when x=max, 1,1,1 when y=max
                        1,2,1  -,2,1           1,2,1           1,2,-             1,2,1
                        1,1,1  -,1,1           1,1,1           1,1,-             -,-,-
            """
        if x < 0 or y < 0 or x >= self.max_x or y >= self.max_y:
            return None
        weights = [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
        if x == 0:
            weights[0][0] = None
            weights[1][0] = None
            weights[2][0] = None
        if y == 0:
            weights[0] = [None, None, None]
        if x == self.max_x - 1:
            weights[0][2] = None
            weights[1][2] = None
            weights[2][2] = None
        if y == self.max_y - 1:
            weights[2] = [None, None, None]
        merged = 0
        cells = 0
        for dx in range(3):
            for dy in range(3):
                if weights[dy][dx] is not None:
                    cells += weights[dy][dx]
                    merged += self.image_array[x+dx-1][y+dy-1] * weights[dy][dx]
        return int(merged/cells)

    def downsize(self):
        """ down size the image such that ? """
        scale = int(min(self.max_x, self.max_y) / (self.code_size * 2))
        # scale is how many complete circular codes would fit side-by-side
        # the objective is to be able to decode up to 8 across (from a size pov)
        while scale > 1:


    def average(self, x, y, dx, dy, cells):
        """ get the average, max, min luminance across a range of pixels from x,y in direction dx,dy
            """
        cell_x = x
        cell_y = y
        average = 0
        max_l = black
        min_l = white
        samples = [0 for _ in range(cells)]
        added = 0
        for cell in range(cells):
            sample = self.merge(cell_x, cell_y)
            if sample is None:
                # we've gone off the end
                break
            average += sample
            added += 1
            if sample > max_l:
                max_l = sample
            if sample < min_l:
                min_l = sample
            samples[cell] = sample
            cell_x += dx
            cell_y += dy
        if added > 0:
            return min_l, int(average/added), max_l, samples
        else:
            return None, None, None, None

    def categorize(self, x, y, dx, dy, cells):
        """ categorize the luminance deviations over a range of pixels from x,y in direction dx, dy
            params are same as the average method
            the categories detected are: H,h?,Z,l?,L
            where Z is the average, H is well above average, L is well below average, h? maybe above and l? maybe below
            to determine that the max/min luminance range is divided into eight slots, four above
            average and four below, this creates eight luminance thresholds as follows:
              let S be the luminance sample being tested
              let LG == below average threshold gap, and HG == above average threshold gap
              then the luminance sample (S) can be categorized by doing the following tests:
                L if S < min+2*LG      (min..min+2*LG)
                l? if S < min+3*LG     (min+2*LG..min+3*LG)
                Z if S < max-3*HG      (min+3*LG..max-3*HG)
                h? if S < max-2*HG     (max-3*HG..max-2*HG)
                H if none of the above (max-2*HG..max)
            """
        bins = [0 for _ in range(cells)]
        min_l, average, max_l, samples = self.average(x, y, dx, dy, cells)
        if min_l is None:
            # we've gone off the end
            return bins, min_l, average, max_l, None, None
        lg = (average - min_l) >> 2
        hg = (max_l - average) >> 2
        if lg < 1 or hg < 1:
            # this implies no deviation, so its all zero, leave bins empty
            pass
        else:
            # make all the ranges
            low = min_l + 2*lg
            low_maybe = min_l + 3*lg
            zero = max_l - 3*hg
            high_maybe = max_l - 2*hg
            # high = the rest
            for cell in range(cells):
                s = samples[cell]
                if s < low:
                    bins[cell] = L
                elif s < low_maybe:
                    bins[cell] = l
                elif s < zero:
                    bins[cell] = Z
                elif s < high_maybe:
                    bins[cell] = h
                else:
                    bins[cell] = H
        return bins, min_l, average, max_l, lg, hg

    def binarize(self, x, y, dx=1, dy=0):
        """ binarize an image line in the direction given starting from x,y and extending to the image edge
            it starts by looking for the prefix one pixel at a time, the prefix is 20 elements of 3..n pixels,
            it looks for Z*2+0101+Z*2 expanding element pixels until it finds a match, the whole code is
            prefix+5*data-blocks+postfix, worst case is whole image is just one quadrant of the circle, so the
            limit on the size of an element is: image-width / (20*7 = 140), for a 4k wide image that's around 32,
            if it finds the prefix it should also exist in up to 7 other directions, use that to verify a
            detection
            """
        next_x = x
        next_y = y
        levels = []
        while 0 <= next_x < self.max_x and 0 <= next_y < self.max_y:
            for n in range(self.min_n, self.max_n):
                slot, min_l, average, max_l, lg, hg = self.categorize(next_x, next_y, dx, dy, n * self.group_size)
                """ slot now contains categorized samples across our potential group
                    if those samples conform to our prefix we've found the code start
                    step 1 - resolve the 'maybe' signals
                        prev-->this-->next
                        H -->h?-->H   no-op            H -->l?-->H   treat as H
                        H -->h?-->h?  no-op            H -->l?-->h?  treat as h?
                        H -->h?-->Z   no-op            H -->l?-->Z   treat as h?
                        H -->h?-->l?  treat as Z       H -->l?-->l?  treat as Z
                        H -->h?-->L   treat as Z       H -->l?-->L   treat as Z
                        
                        h?-->h?-->H   no-op            h?-->l?-->H   treat as h?
                        h?-->h?-->h?  no-op            h?-->l?-->h?  treat as h?
                        h?-->h?-->Z   no-op            h?-->l?-->Z   treat as Z
                        h?-->h?-->l?  treat as Z       h?-->l?-->l?  treat as Z
                        h?-->h?-->L   treat as Z       h?-->l?-->L   treat as Z
                        
                        Z -->h?-->H   no-op            Z -->l?-->H   treat as h?
                        Z -->h?-->h?  no-op            Z -->l?-->h?  treat as Z
                        Z -->h?-->Z   treat as Z       Z -->l?-->Z   treat as Z
                        Z -->h?-->l?  treat as Z       Z -->l?-->l?  no-op
                        Z -->h?-->L   treat as Z       Z -->l?-->L   no-op
                        
                        l?-->h?-->H   treat as Z       l?-->l?-->H   treat as Z
                        l?-->h?-->h?  treat as Z       l?-->l?-->h?  treat as Z
                        l?-->h?-->Z   treat as Z       l?-->l?-->Z   no-op
                        l?-->h?-->l?  treat as l?      l?-->l?-->l?  no-op
                        l?-->h?-->L   treat as l?      l?-->l?-->L   no-op
                        
                        L -->h?-->H   treat as Z       L -->l?-->H   treat as Z
                        L -->h?-->h?  treat as Z       L -->l?-->h?  treat as Z
                        L -->h?-->Z   treat as Z       L -->l?-->Z   no-op
                        L -->h?-->l?  treat as l?      L -->l?-->l?  no-op
                        L -->h?-->L   treat as l?      L -->l?-->L   no-op
                    step 2 - must begin with a leadin and end with a group_gap
                    step 3 - find data-group edges
                    step 4 - find data bits in group
                    """
                # step 1
                prev = Z
                for this in range(len(slot)-1):
                    if slot[this] == h:
                        # maybe H
                        slot[this] = self.h_map[prev][slot[this + 1]]
                    elif slot[this] == l:
                        # maybe L
                        slot[this] = self.l_map[prev][slot[this + 1]]
                    else:
                        # no ambiguity, leave as is
                        pass
                    prev = slot[this]
                # step 2
                if self.match(slot, leadin) and self.match(slot, group_gap, True):
                    # got required front/back
                    pass
                else:
                    # break  # not a candidate - move on (NB: its not a candidate no matter how big 'n' may get)
                    continue
                # step 3
                begin, end = self.edges(slot)
                print('{},{}: {}<{}>{}: {}..{}: {}'.format(next_x,next_y,min_l,average,max_l,begin,end,slot))
                if begin is None:
                    # not a candidate as the slot has no edges
                    # we need to consider a bigger window here to take in more pixels
                    continue
                # step 4
                got = self.bits(slot, begin, end)
                print('{},{}: {}x{}..{}={}: {}'.format(next_x,next_y,n,begin,end,got,slot))
                if got != -1:
                    # not found our prefix
                    # we can move the position to the trailing quiet-zone
                    # next_x += ((n * self.group_size) - 1) * dx
                    # next_y += ((n * self.group_size) - 1) * dy
                    # break
                    continue
                return slot, next_x, next_y, min_l, average, max_l, lg, hg, n  # HACK
            next_x += dx
            next_y += dy
        # return categories
        return levels, next_x, next_y, min_l, average, max_l, lg, hg, n

    def bits(self, slot, begin, end):
        """ given a slot and group boundary, extract its bits
            the slot given has already had its leading and trailing quiet-zone detected, the
            begin and end are the position of the quiet-zone edges
            for the group to be valid it must contain exactly four bits evenly spaced,
            those bits may have 2, 3 or 4 transitions, for bit pairs that are the same (0011 or 1100)
            there are 2 transitions, for bit pairs that are different (0110 or 1001) there are 3
            and for the marker (0101 or 1010) there are 4.
            """
        # count 0->1 and 1>0 transitions
        was = Z                          # NB: this followed by 0 or 1 is a transition
        crossings = 0
        for pixel in range(begin, end+1):
            now = slot[pixel]
            if now == Z or now == h or now == l:
                # don't consider this as a transition
                pass
            elif was == now:
                # not a transition
                pass
            else:
                # got a transition
                crossings += 1
                was = now
        if crossings not in [2, 3, 4]:
            # too many or too few transitions, so its not a valid group
            return None
        width = (end - begin + 1) >> 2   # 4 bits within the band, chop range into 4
        centre = max(width >> 1, 1)      # centre band is half the width
        # divide window into 4 starting at begin + half band but do it in a manner that does not drift
        width = (end - begin + 1) / 4
        offset = width / 2
        s1 = begin + offset
        s2 = s1 + width
        s3 = s2 + width
        s4 = s3 + width
        payload = 0  # the bits we discover
        for sample in [s1, s2, s3, s4]:
            bit = self.bit_map[slot[int(sample)]]
            if bit is None:
                # bit spacing is wrong, so its not a valid group
                return None
            payload = (payload << 1) + bit
        return self.valid_bits[payload]

    def edges(self, slot):
        """ given a slot that has a leading and trailing quiet zone find the enclosed data-group edges
            """
        for begin in range(len(slot)):
            if slot[begin] != Z:
                # found leading edge
                for end in range(len(slot)-1, -1, -1):
                    if slot[end] != Z:
                        # found trailing edge
                        # extent must be at least four
                        if end - begin + 1 < 4:
                            # too small, consider it to be noise
                            return None, None
                        return begin, end
        return None, None

    def match(self, slot, pattern, end=False):
        """ see if slot contains pattern, returns True iff matches
            pattern is an array of H,h?,Z,l?,L elements or nested-arrays of the same
            if end is True look for pattern at end of slot, else at start
            """
        # unwind nested patterns
        expanded = self.expand(pattern)
        if end:
            return slot[-len(expanded):] == expanded
        else:
            return slot[:len(expanded)] == expanded

    def expand(self, pattern):
        """ unwind nested elements from the given pattern """
        expanded = []
        for element in pattern:
            if type(element) == list or type(element) == tuple:
                expanded.extend(self.expand(element))
            else:
                expanded.append(element)
        return expanded

    def show_line(self, levels, x, y, min_l, average, max_l, lg, hg, n):
        """ print an image of the given binarized line """
        print('')
        print('samples: {} from {},{} with width {}'.format(len(levels), x, y, n))
        print('Raw is min={}, average={}, max={}, lg={}, hg={}:'.format(min_l, average, max_l, lg, hg))
        for s in range(x, x+len(levels)):
            print('{:03},'.format(self.image_array[s][y]), end='')
        print('')
        print('Binarized as:')
        for level in levels:
            if level == Z:
                print('-', end='')
            elif level == h:
                print('/', end='')
            elif level == H:
                print('1', end='')
            elif level == l:
                print('\\', end='')
            elif level == L:
                print('0', end='')
            else:
                print('?', end='')
        print('')
        print('')


num = '0795-photo.jpg'
print('Loading {}'.format(num))
decoder = Decoder(num)
print('Binarizing {}'.format(num))
test_rows = [-400,-200,0,200,400]
for row in test_rows:
    levelled, x, y, min_l, average, max_l, lg, hg, n = decoder.binarize(int(decoder.centre_x/2), decoder.centre_y+row)
    print('Showing {} from row {}'.format(num,row))
    decoder.show_line(levelled, x, y, min_l, average, max_l, lg, hg, n)
print('Done {}'.format(num))

