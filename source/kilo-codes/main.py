""" Kilo Codes (KC)
    See code description in codes.py for a visualisation.
    They are called 'kilo-codes' (or KC for short) as they only encode the numbers 1..1023 (i.e. 10 bits).

    This coding scheme is intended to be easy to detect and be robust against noise, distortion and luminance artifacts.
    The code is designed for use as competitor numbers in club level fell races, as such the number range
    is limited to between 101 and 999 inclusive, i.e. always a 3-digit number and max 899 competitors.

    The competitor numbers are encoded within the code image as 10 bits of data and 11 bits of CRC.
    This level of CRC allows for up to 3 bits of error correction (the CRC Hamming distance is 7),
    but that many is only used for close targets (i.e. big) where the noise level is low enough.
    See crc.py for the algorithm.

    Some codes are reserved for special purposes - start, check, retired, finish, etc.

    Note: The decoder can cope with many targets in the same image. It uses the relative size of
          the targets to present the results in distance order, nearest (i.e. biggest) first.
          This is important to determine finish order.

    This Python implementation is just a proof-of-concept. In particular, it does not represent good
    coding practice, nor does it utilise the Python language to its fullest extent, and performance
    is mostly irrelevant. It represents the specification for the actual implementation.

"""

# ToDo: create test background with random noise blobs and random edges
# ToDo: add a blurring option when drawing test targets
# ToDo: add a rotation (vertically and horizontally) option when drawing test targets
# ToDo: add a curving (vertically and horizontally) option when drawing test targets
# ToDo: generate lots of (extreme) test targets using these options
# ToDo: draw the test targets on a relevant scene photo background
# ToDo: try frame grabbing from a movie
# ToDo: in the final product speak the number detected (just need 10 sound snips - spoken 0 to 9)
# ToDo: in the final product use special codes to mark the finish, retired, et al
#       (runner crosses the line when his target size matches that of the finish code)
#       (runner retires by holding the retired target beside theirs in front of the camera)
#       (mark CP's by runners running between two CP (n) targets)
#       (record starters by placing start targets at several places near the race start)
# ToDo: generate some test images in heavy rain (and snow?)

if __name__ == "__main__":
    """ test harness """
