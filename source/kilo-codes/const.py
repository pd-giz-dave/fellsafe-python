"""
Globally useful constants
"""

# region colours...
MAX_LUMINANCE = 255
MIN_LUMINANCE = 0
MID_LUMINANCE = (MAX_LUMINANCE - MIN_LUMINANCE) >> 1
# endregion

# region alpha channel...
TRANSPARENT = MIN_LUMINANCE
OPAQUE      = MAX_LUMINANCE
# endregion

# region Diagnostic image colours...
# NB: cv2 colour order is BGR not RGB
BLACK      = (  0,   0,   0)
GREY       = ( 64,  64,  64)
WHITE      = (255, 255, 255)
RED        = (  0,   0, 255)
GREEN      = (  0, 255,   0)
DARK_GREEN = (  0, 128,   0)
BLUE       = (255,   0,   0)
DARK_BLUE  = ( 64,   0,   0)
YELLOW     = (  0, 255, 255)
PURPLE     = (255,   0, 255)
PINK       = (128,   0, 128)
CYAN       = (128, 128,   0)
PALE_RED   = (  0,   0, 128)
PALE_BLUE  = (128,   0,   0)
PALE_GREEN = (  0, 128,   0)
CYAN       = (255, 255,   0)
OLIVE      = (  0, 128, 128)
ORANGE     = ( 80, 127, 255)
# synonyms
LIME    = GREEN
MAGENTA = PURPLE
MAROON  = PALE_RED
NAVY    = PALE_BLUE
# endregion

# region Video modes image height...
VIDEO_SD = 480
VIDEO_HD = 720
VIDEO_FHD = 1080
VIDEO_2K = 1152
VIDEO_4K = 2160
# endregion

# region Debug options...
DEBUG_NONE = 0  # no debug output
DEBUG_IMAGE = 1  # just write debug annotated image files
DEBUG_VERBOSE = 2  # do everything - generates a *lot* of output
# endregion

# region Proximity options...
# these control the contour detection, for big targets that cover the whole image a bigger
# integration area is required (i.e. smaller image fraction), this is used for testing print images
PROXIMITY_FAR = 48  # suitable for most images (photos and videos)
PROXIMITY_CLOSE = 3  # suitable for print images
# black threshold for binarising contours
BLACK_LEVEL = {PROXIMITY_FAR: 30, PROXIMITY_CLOSE: -0.01}
# endregion

# region Blob circle radius modes...
RADIUS_MODE_INSIDE  = 0
RADIUS_MODE_MEAN    = 1
RADIUS_MODE_OUTSIDE = 2
RADIUS_MODES        = 3  # count of the number of modes
# endregion

# region CRC parameters
PAYLOAD_BITS  = 10
PAYLOAD_RANGE = 1 << PAYLOAD_BITS
POLY_BITS     = 12
POLYNOMIAL    = 0xAE3  # discovered by brute force search, has hamming distance of 7
# POLYNOMIAL    = 0xC75  # discovered by brute force search, has hamming distance of 7
# endregion