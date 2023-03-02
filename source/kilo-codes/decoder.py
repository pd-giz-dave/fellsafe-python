""" Decode detected bit string into a code (if possible)
    The bit strings detected consist of a series of 0's, 1's or None's.
    The bit strings are in the sames order as when the code was drawn (see codes.py).
    The bit strings provided are LSB first.
    Each None bit is interpreted as a 0 and a 1 to test for a valid code with a bit missing unless there are too many.
"""
import math
import random  # only used for testing
import cv2     # only used for diagnostics

import const
import utils
import crc
import extractor

class Decoder:

    GOOD_OPTIONS_THRESHOLD = 1.0  # when got multiple decode options, min ratio of good to overall to be acceptable
                                  # must be > 0.5 else becomes ambiguous (only one option must qualify)
                                  # anything less than 1 creates false positives

    def __init__(self, bits, codec=None, logger=None):
        self.bits = bits
        self.logger = logger
        self.codewords = None
        self.decodes = None
        self.best_codes = None
        if codec is None:
            self.codec = make_codec(logger)
        else:
            self.codec = codec
        self.max_options = 1 << self.codec.error_bits  # limit the number of None bits we will tolerate

    def get_codewords(self, detection=None):
        """ get all the possible code words from the bits of the given detection, or all of them,
            there could be several possibilities for each detection depending on how many bits are missing
            """

        def expand_bits(bits):
            """ expand all None bits such that each is split into a bit sequence with it a 0 and 1 """
            for bit, value in enumerate(bits):
                if value is None:
                    option_0 = bits.copy()
                    option_0[bit] = 0
                    option_0_bits = expand_bits(option_0)
                    option_1 = bits.copy()
                    option_1[bit] = 1
                    option_1_bits = expand_bits(option_1)
                    return option_0_bits + option_1_bits
            return [bits]

        if self.codewords is None:
            self.codewords = []
            for d, bits in enumerate(self.bits):
                codewords = []
                options = expand_bits(bits)
                if len(options) > self.max_options:
                    if self.logger is not None:
                        self.logger.log('Detection {} bits {} has too many missing bits '
                                        '(makes options {} where max is {})'.
                                        format(d, bits, len(options), self.max_options))
                    pass
                else:
                    for bits in options:
                        codeword = 0
                        for bit in range(len(bits)-1, -1, -1):
                            codeword <<= 1
                            if bits[bit] == 1:
                                codeword += 1
                        codewords.append(codeword)
                self.codewords.append(codewords)
        if detection is None:
            return self.codewords
        elif detection < len(self.codewords):
            return self.codewords[detection]
        else:
            return []

    def get_decodes(self, detection=None):
        """ get the decoded codewords for the given detection, or all of them,
            for each one all the decodes, errors and bits are returned for each codeword option
            """
        if self.decodes is None:
            self.decodes = []
            for codewords in self.get_codewords():
                decodes = []
                for codeword in codewords:
                    decode, errors = self.codec.decode(codeword)
                    decodes.append((decode, errors, codeword))
                self.decodes.append(decodes)
        if detection is None:
            return self.decodes
        elif detection < len(self.decodes):
            return self.decodes[detection]
        else:
            return []

    def get_codes(self, detection=None):
        """ get the best decoded codeword for the given detection, or all of them,
            returns a list of codes each with a set of useful properties of the detection
            """

        if self.best_codes is None:
            self.best_codes = []
            for decoded in self.get_decodes():
                # count code choices
                candidates = {}
                for option, (decode, errors, codeword) in enumerate(decoded):
                    if decode == 0:
                        # 0 is not legal, so ignore it
                        continue
                    if decode not in candidates:
                        candidates[decode] = [option]
                    else:
                        candidates[decode].append(option)
                # find best
                options = len(decoded)
                nones = int(math.log2(max(options,1)))
                best_decode = None
                for hits in candidates.values():
                    ratio = len(hits) / options
                    if ratio < Decoder.GOOD_OPTIONS_THRESHOLD:
                        # not enough hits to be acceptable
                        continue
                    if best_decode is not None:
                        raise Exception('Ambiguous decoder choice: '
                                        'ratio is {} when threshold is {} with {} choices'.
                                        format(ratio, Decoder.GOOD_OPTIONS_THRESHOLD, options))
                    # found the option with enough hits, find the worst case error (being pessimistic)
                    best_errors = None
                    for option in hits:  # NB: hits contains at least 1 item
                        decode, errors, codeword = decoded[option]
                        if best_errors is None or errors > best_errors:
                            best_errors = max(errors, nones)  # consider each None as an error
                            best_decode = (decode, best_errors, codeword)
                    # continue to check for ambiguity
                self.best_codes.append(best_decode)
        if detection is None:
            return self.best_codes
        elif detection < len(self.best_codes):
            return self.best_codes[detection]
        else:
            return None

def make_codec(logger=None):
    """ make the codec for decoding our bits """
    if logger is not None:
        logger.log('Preparing codec...')
    codec = crc.CRC(const.PAYLOAD_BITS, const.POLYNOMIAL, logger)
    codec.prepare()
    return codec

def decode_bits(bits, source=None, image=None, origins=None, logger=None, codec=None):
    """ decode the given bit strings into all its options """
    decoder = Decoder(bits, logger=logger, codec=codec)
    decodes = decoder.get_codes()
    if logger is not None:
        logger.push('decode_bits')
        draw = None
        for detection, decode in enumerate(decodes):
            if origins is not None:
                origin, size = origins[detection]
                folder = utils.image_folder(target=origin)
                logger.push(folder, folder)
                if image is not None:
                    if draw is None:
                        draw = cv2.merge([image, image, image])
                    top_left = (origin[0], origin[1])
                    bottom_right = (top_left[0]+128, top_left[1]+16)
                    if decode is None:
                        colour = const.RED
                        text = 'INVALID ({:.0f})'.format(size)
                    else:
                        colour = const.GREEN
                        text = '{}.{} ({:.0f})'.format(decode[0], decode[1], size)
                    cv2.rectangle(draw, top_left, bottom_right, colour, 1)
                    cv2.putText(draw, text, (top_left[0]+3, bottom_right[1]-3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
            logger.log('{} decodes as {}'.format(bits, decode))
            if origins is not None:
                logger.pop()
        if draw is not None:
            logger.draw(draw, file='decodes')
        logger.pop()
    return decodes

def _test_pipeline(src, proximity, blur=3, logger=None, create_new=True):
    """ ************** TEST **************** """

    if logger.depth() > 1:
        logger.push('decoder/_test_pipeline')
    else:
        logger.push('_test_pipeline')
    logger.log("\nDecoding code bits from the pipeline")

    # get the code bits
    bits, image, origins = extractor._test(src, proximity, blur=blur, logger=logger, create_new=create_new)

    # process the code areas
    codes = decode_bits(bits, source=src, image=image, origins=origins, logger=logger)

    logger.pop()
    return codes

def _test_decoder(logger):
    """ ************** TEST **************** """

    if logger.depth() > 1:
        logger.push('decoder/_test_decoder')
    else:
        logger.push('_test_decoder')

    MISSING_ERRORS = (0, 3)  # range of missing bits to apply (random within this) (should not exceed error bits)
    FLIPPED_ERRORS = (0, 3)  # range of flipped bits to apply (random within this) (should not exceed error bits)

    logger.log('Testing decoder with {} missing and {} flipped bits'.format(MISSING_ERRORS, FLIPPED_ERRORS))
    codec = make_codec(logger)
    # generate codewords with random missing bits and random flipped bits
    logger.log('Testing all possible codes...')
    good = 0
    bad = 0
    wrong = 0
    for code in range(1, const.PAYLOAD_RANGE):  # NB: code 0 is invalid, so do not test that one
        # region make a random error test case...
        # create a valid codeword
        encoded = codec.encode(code)
        bits = []
        for bit in range(codec.code_bits):
            mask = 1 << bit
            if encoded & mask == 0:
                bits.append(0)
            else:
                bits.append(1)
        # replace random bits with None
        missing = []
        for _ in range(random.randint(MISSING_ERRORS[0], MISSING_ERRORS[1])):
            while True:
                bit = random.randrange(len(bits))
                if bit in missing:
                    continue
                missing.append(bit)
                break
            bits[bit] = None
        # flip random bits
        flipped = []
        for _ in range(random.randint(FLIPPED_ERRORS[0], FLIPPED_ERRORS[1])):
            while True:
                bit = random.randrange(len(bits))
                if bits[bit] is None:
                    # leave alone
                    continue
                if bit in flipped:
                    continue
                flipped.append(bit)
                break
            if bits[bit] == 0:
                bits[bit] = 1
            else:
                bits[bit] = 0
        # endregion
        # region try to decode our test case...
        decoded = decode_bits([bits], codec=codec)
        if decoded[0] is None:
            # failed
            prefix = '**BAD**'
            bad += 1
        else:
            decode, errors, codeword = decoded[0]
            if decode == code:
                good += 1
                prefix = 'OK_____'
            else:
                wrong += 1
                prefix = '*WRONG*'
        logger.log('{} Code {} with bits {} missing and {} flipped ({} errors) decoded as: {}'.
                   format(prefix, code, missing, flipped, len(missing) + len(flipped), decoded[0]))
        # endregion
    logger.log('{} good decodes, {} bad, {} wrong from {}'.format(good, bad, wrong, const.PAYLOAD_RANGE-1))


if __name__ == "__main__":
    """ test harness """

    #src = "/home/dave/precious/fellsafe/fellsafe-image/media/kilo-codes/kilo-codes-distant.jpg"
    src = "/home/dave/precious/fellsafe/fellsafe-image/source/kilo-codes/test-alt-bits.png"
    proximity = const.PROXIMITY_CLOSE
    #proximity = const.PROXIMITY_FAR

    logger = utils.Logger('decoder.log', 'decoder/{}'.format(utils.image_folder(src)))

    # test Decoder
    #_test_decoder(logger)

    # test whole pipeline
    _test_pipeline(src, proximity, blur=3, logger=logger, create_new=False)
