""" Extract the code from the code area
    This module takes a list data cell cor-ordinates for each detection and extracts the code bits.
    The cells given here are all those between the locators, the data bits are within that.
    See the visualisation in codes.py
"""

import const
import utils
import finder

class Extractor:

    def __init__(self, detections, logger=None):
        self.detections = detections
        self.logger = logger


def _test(src, proximity, blur=3, logger=None, create_new=True):
    """ ************** TEST **************** """

    logger.log("\nExtracting code bits")

    # get the code areas
    found = finder._test(src, proximity, blur=3, logger=None, create_new=create_new)

    # process the code areas
    extractor = Extractor(found, logger)  # ToDo:


if __name__ == "__main__":
    """ test harness """

    #src = "/home/dave/precious/fellsafe/fellsafe-image/media/square-codes/square-codes-distant.jpg"
    src = "/home/dave/precious/fellsafe/fellsafe-image/source/kilo-codes/test-alt-bits.png"
    proximity = const.PROXIMITY_CLOSE
    #proximity = const.PROXIMITY_FAR
    create_new = True

    logger = utils.Logger('extractor.log', 'extractor/{}'.format(utils.image_folder(src)))
    logger.log('_test')

    _test(src, proximity, blur=3, logger=logger, create_new=create_new)
