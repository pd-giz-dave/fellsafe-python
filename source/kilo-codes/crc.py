""" CRC coding/decoding functions
    See https://en.wikipedia.org/wiki/Cyclic_redundancy_check for the encoding algorithm.
    The functions here are only suitable for single integer based payloads and CRC's (i.e. up 32 bits).
"""
import random

import const
import utils

class CRC:

    MAX_PAYLOAD_BITS = 15  # limits max lookup table sizes,
    MAX_POLY_BITS    = 15  # but is otherwise arbitrary

    def __init__(self, payload_bits: int, poly: int, logger=None):
        """ save and validate the spec  """
        self.logger       = logger  # iff not None a logging function
        self.payload_bits = payload_bits
        self.poly         = poly
        self.poly_bits    = CRC.bit_width(self.poly)
        self.crc_bits     = self.poly_bits - 1
        if self.payload_bits > CRC.MAX_PAYLOAD_BITS:
            raise Exception('max payload bits is {}, given {}'.format(CRC.MAX_PAYLOAD_BITS, self.payload_bits))
        if self.poly_bits > CRC.MAX_POLY_BITS:
            raise Exception('max polynomial bits is {}, given polynomial ({:b}) has {}'.
                            format(CRC.MAX_POLY_BITS, self.poly, self.poly_bits))
        self.crc_mask         = (1 << self.crc_bits) - 1  # AND a code word with this to isolate its CRC
        self.payload_range    = (1 << self.payload_bits)
        self.payload_mask     = self.payload_range - 1  # AND synonym with this to isolate its payload
        self.code_bits        = self.payload_bits + self.crc_bits  # total bits in the code
        self.code_range       = 1 << self.code_bits  # max code value (payload + crc)
        self.synonyms         = None  # codeword lookup table (its huge)
        self.unique           = None  # number if unique synonyms in synonyms lookup table
        self.masks            = None  # list of bit flip masks for every N bits in the code range (lazy evaluation)
        self.codewords        = None  # list of all codewords for every payload along with its distance from any other
        self.hamming_distance = None  # the Hamming distance of the polynomial
        self.error_bits       = None  # max error correction capability (==(hamming_distance-1)/2

    def prepare(self):
        """ initialise lookup tables (preempts what encode and decode would do) """
        test_code = random.randrange(1, self.payload_range)
        codeword = self.encode(test_code)
        result = self.decode(codeword)
        if result == (test_code, 0):
            # its all working OK
            return True
        raise Exception('Encode of {} decoded as {} when should be {}'.format(test_code, result, (test_code, 0)))

    @staticmethod
    def bit_width(val: int) -> int:
        """ get the minimum number of bits required to hold the given value """
        if val > ((1 << 24) - 1) or val < 1:
            raise Exception('only positive 24 bit numbers allowed, not {}'.format(val))
        k: int = 0
        if val > 0x0000FFFF: val >>= 16; k = 16
        if val > 0x000000FF: val >>= 8 ; k |= 8
        if val > 0x0000000F: val >>= 4 ; k |= 4
        if val > 0x00000003: val >>= 2 ; k |= 2
        k |= (val & 2) >> 1
        return k + 1

    def code(self, payload: int, syndrome: int) -> int:
        """ create the full code from the given parts """
        return (payload << self.crc_bits) | (syndrome & self.crc_mask)

    def encode(self, payload: int) -> int:
        """ return the CRC encoded code word for the given value """
        if self.codewords is None:
            self.build_codewords_table()
        return self.codewords[payload][0]

    def unencode(self, codeword: int) -> (int, int):
        """ decode the code word into its value and its syndrome for the code """
        payload  = codeword >> self.crc_bits
        syndrome = codeword & self.crc_mask
        return payload, self.calculate(payload, syndrome)

    def calculate(self, val: int, pad=0) -> int:
        """ calculate the CRC for the given value """
        if val >= (1 << self.payload_bits):
            raise Exception('{} is beyond the range of payload bits {}'.format(val, self.payload_bits))
        if (val + pad) < 1:
            raise Exception('value+pad must be positive, not {}+{}'.format(val, pad))
        crc = (val << self.crc_bits) | pad
        while crc > 0:
            poly_ms_shift = CRC.bit_width(crc) - self.poly_bits
            if poly_ms_shift < 0:
                break
            poly = self.poly << poly_ms_shift  # put poly in MS position
            crc ^= poly
        return crc & self.crc_mask

    @staticmethod
    def count_bits(mask: int) -> int:
        """ count number of 1 bits in the given mask """
        bits = 0
        while mask != 0:
            if mask & 1 == 1:
                bits += 1
            mask >>= 1
        return bits

    def flips(self, N: int=0) -> [int]:
        """ return a list of all possible bit flips of N bits within code-bits """
        if N > self.code_bits:
            raise Exception('flips limit is {}, cannot do {}'.format(self.code_bits, N))
        if self.masks is None:
            if self.logger is not None:
                self.logger.log('Build bit flip mask table for {} code-bits...'.format(self.code_bits))
            self.masks = [[] for _ in range(self.code_bits+1)]
            for bits in range(self.code_range):
                count = CRC.count_bits(bits)
                self.masks[count].append(bits)
            if self.logger is not None:
                total = 0
                for count in range(1, len(self.masks)):
                    self.logger.log('  N:{}={} flips'.format(count, len(self.masks[count])))
                    total += len(self.masks[count])
                self.logger.log('  total possible flips: {}'.format(total))
        return self.masks[N]

    def decode(self, codeword: int) -> (int, int):
        """ decode with error correction by a lookup table,
            returns the payload and the number of bit errors
            """
        if self.synonyms is None:
            self.build_synonyms_table()
        synonym = self.synonyms[codeword]
        return self.unmake_synonym(synonym)

    def build_codewords_table(self) -> (int, int):
        """ build the codewords for each payload and Hamming distance between each codeword and every other codeword,
            returns the Hamming distance and error correction bits
            """
        if self.codewords is not None:
            # already done it
            return self.hamming_distance
        if self.logger is not None:
            self.logger.push(context='crc')
            self.logger.log('Build codewords table for polynomial {:b}:...'.format(self.poly))
        self.codewords = [[0,self.code_bits + 1] for _ in range(self.payload_range)]  # init to huge distance
        # build initial code table
        for payload in range(1, self.payload_range):
            self.codewords[payload][0] = self.code(payload, self.calculate(payload))  # NB: do NOT use encode() here!
        # find min distance for each codeword
        for ref_payload in range(1, self.payload_range):
            for payload in range(1, self.payload_range):
                if payload == ref_payload:
                    # ignore self
                    continue
                ref_codeword = self.codewords[ref_payload][0]
                codeword     = self.codewords[payload    ][0]
                distance     = CRC.count_bits(ref_codeword ^ codeword)
                if distance < self.codewords[ref_payload][1]:
                    # found a closer code
                    self.codewords[ref_payload][1] = distance
        # find overall Hamming distance
        self.hamming_distance = self.code_bits + 1  # initially set very big minimum
        for _, distance in self.codewords:
            if distance < self.hamming_distance:
                self.hamming_distance = distance
        self.error_bits = max((self.hamming_distance - 1) >> 1, 0)
        if self.logger is not None:
            self.logger.log('  hamming distance is {}, max error-bits is {}'.format(self.hamming_distance, self.error_bits))
            self.logger.pop()
        return self.hamming_distance, self.error_bits

    def build_synonyms_table(self) -> int:
        """ build a lookup table for every possible codeword with its decoded value and its distance,
            the 'distance' is how far from correct the synonym is, a synonym of 0 indicates an
            invalid code word,
            returns the number of unique synonyms
            """
        if self.synonyms is not None:
            # already done it
            return self.unique
        if self.logger is not None:
            self.logger.push('crc')
            self.logger.log('Build synonyms table (this may take some time!)...')
        # make all possible codewords (and their distances)
        self.build_codewords_table()
        # make all possible bit flips
        self.flips()
        # make all possible synonyms
        possible = 0
        synonyms = [[] for _ in range(self.code_range)]
        for payload in range(1, self.payload_range):
            codeword, distance = self.codewords[payload]
            N_limit = (distance - 1) >> 1
            for N in range(len(self.masks)):
                if N > N_limit:
                    # too far for this payload
                    break
                for mask in self.masks[N]:
                    synonym = codeword ^ mask
                    if synonym == 0:
                        # not allowed
                        continue
                    synonyms[synonym].append(self.make_synonym(payload, N))
                    possible += 1
        if self.logger is not None:
            self.logger.log('  all possible synonyms = {}'.format(possible))
        # build unique synonym table
        self.unique = 0
        self.synonyms = [0 for _ in range(self.code_range)]  # set all invalid initially
        for codeword, candidates in enumerate(synonyms):
            if len(candidates) == 0:
                # this one is illegal (NB: codeword 0 is guaranteed to be invalid)
                continue
            elif len(candidates) > 1:
                # got ambiguity, keep lowest distance if its unique
                breakpoint()
                best_candidate = None
                for candidate, synonym in enumerate(candidates):
                    if best_candidate is None:
                        best_candidate = candidate
                        continue
                    _, best_distance = self.unmake_synonym(candidates[best_candidate])
                    payload, distance = self.unmake_synonym(synonym)
                    if distance < best_distance:
                        best_candidate = candidate
                    elif distance == best_distance:
                        # ambiguous
                        best_candidate = None
                if best_candidate is None:
                    # nothing suitable here
                    continue
                self.synonyms[codeword] = candidates[best_candidate]
                self.unique += 1
            else:
                # no ambiguity here, so keep it
                self.synonyms[codeword] = candidates[0]
                self.unique += 1
        if self.logger is not None:
            self.logger.log('  unique synonyms = {}'.format(self.unique))
            self.logger.pop()
        return self.unique

    def analyse_synonyms(self):
        """ diagnostic aid to verify synonym table is usable,
            returns a list of payloads for each distance and a list of codewords for each payload
            """
        distances = [[] for _ in range(self.code_bits + 1)]
        payloads = [[] for _ in range(self.payload_range)]
        for codeword, synonym in enumerate(self.synonyms):
            payload, distance = self.unmake_synonym(synonym)
            if payload == 0:
                # these are not valid
                continue
            distances[distance].append(payload)
            payloads[payload].append(codeword)
        return distances, payloads

    def make_synonym(self, payload, distance):
        """ a synonym is a code value and its distance """
        return (distance << self.payload_bits) | payload

    def unmake_synonym(self, synonym):
        """ undo what make_synonym did """
        payload  = synonym & self.payload_mask
        distance = synonym >> self.payload_bits
        return payload, distance

def make_codec(logger=None):
    """ make a codec for encoding/decoding bits """
    if logger is not None:
        logger.log('Preparing codec...')
    codec = CRC(const.PAYLOAD_BITS, const.POLYNOMIAL, logger)
    codec.prepare()
    return codec

def find_best_poly(poly_bits: int, payload_bits: int, logger=None) -> [(int, int)]:
    """ find best CRC polynomial (by doing an exhaustive search of all possible polynomials),
        returns a list of the best polynomials and their Hamming distance
        """
    POLY_BITS     = poly_bits
    POLY_MAX      = 1 << POLY_BITS
    POLY_MSB      = POLY_MAX >> 1
    PAYLOAD_BITS  = payload_bits
    if logger is not None:
        logger.log('Searching for best {}-bit polynomial in range {}..{} for a {}-bit payload (this may take some time!):...'.
            format(POLY_BITS, POLY_MSB, POLY_MAX, PAYLOAD_BITS))
    best_distance = 0
    best_poly     = None
    candidates = []
    for poly in range(POLY_MSB, POLY_MAX):
        codec    = CRC(PAYLOAD_BITS, poly)
        distance, _ = codec.build_codewords_table()
        if distance > best_distance:
            # got a new best
            best_distance = distance
            best_poly     = poly
            candidates    = [(best_poly, best_distance)]
            if logger is not None:
                logger.log('  new best so far: {}'.format(candidates))
        elif distance == best_distance:
            # got another just as good
            candidates.append((poly, distance))
            if logger is not None:
                logger.log('  another at same best so far: {}'.format(candidates))
    if logger is not None:
        logger.log('  best overall: {}'.format(candidates))
    return candidates

if __name__ == "__main__":
    """ test harness """
    logger = utils.Logger('crc.logger')
    logger.log('CRC test harness')
    PAYLOAD_BITS  = const.PAYLOAD_BITS
    POLY_BITS     = const.POLY_BITS
    PAYLOAD_RANGE = const.PAYLOAD_RANGE
    # find best CRC polynomial (by doing an exhaustive search of all poly-bit polynomials)
    # best_candidates = find_best_poly(POLY_BITS, PAYLOAD_BITS, logger)
    # POLYNOMIAL     = best_candidates[0][0]
    POLYNOMIAL     = const.POLYNOMIAL
    codec = CRC(PAYLOAD_BITS, POLYNOMIAL, logger)
    logger.log('CRC spec: payload bits: {}, crc bits: {}, polynomial: {:b}, payload range: 1..{}, code range 0..{}'.
               format(PAYLOAD_BITS, codec.crc_bits, POLYNOMIAL, PAYLOAD_RANGE-1, codec.code_range-1))
    codec.build_synonyms_table()
    distances, payloads = codec.analyse_synonyms()
    logger.log('Analysis:')
    for distance, synonyms in enumerate(distances):
        if len(synonyms) == 0:
            # nothing here
            continue
        logger.log('  distance:{} payloads={}'.format(distance, len(synonyms)))
    min_synonyms = codec.code_range + 1
    max_synonyms = 0
    avg_synonyms = 0
    samples = 0
    for payload, synonyms in enumerate(payloads):
        if payload == 0:
            # not allowed
            continue
        if len(synonyms) == 0:
            # nothing here
            continue
        avg_synonyms += len(synonyms)
        samples += 1
        if len(synonyms) < min_synonyms:
            min_synonyms = len(synonyms)
        if len(synonyms) > max_synonyms:
            max_synonyms = len(synonyms)
    avg_synonyms /= samples
    logger.log('  synonyms: min={}, max={}, average={:.0f}'.format(min_synonyms, max_synonyms, avg_synonyms))
    logger.log('Detection: all payloads for zero error case:')
    passes = []
    fails = []
    encoded = [None for _ in range(PAYLOAD_RANGE)]
    for code in range(1, PAYLOAD_RANGE):
        encoded[code] = codec.encode(code)
        decoded, errors = codec.decode(encoded[code])
        if errors == 0 and decoded == code:
            passes.append(code)
        else:
            fails.append(encoded)
    logger.log('  passes: {}'.format(len(passes)))
    logger.log('  fails: {}'.format(len(fails)))
    if len(fails) == 0:
        logger.log('  all codes pass encode->decode')
    else:
        logger.log('  encode->decode not symmetrical - find out what went wrong!')
    logger.log('Detection: All codeword cases:')
    passes = 0
    fails = 0
    good = {}
    for code in range(1, codec.code_range):
        decoded, errors = codec.decode(code)
        if decoded > 0 and errors == 0:
            passes += 1
            if good.get(decoded) is None:
                good[decoded] = 1
            else:
                good[decoded] += 1
        else:
            fails += 1
    logger.log('  passes: {}'.format(passes))
    logger.log('  fails: {}'.format(fails))
    if len(good) == (PAYLOAD_RANGE - 1):
        logger.log('  all, and only, expected codes detected ({})'.format(len(good)))
    else:
        logger.log('  got {} unique codes when expecting {} - find out why'.format(len(good), PAYLOAD_RANGE - 1))
    logger.log('Error recovery for every possible code word: {}'.format(codec.code_range-1))
    passes = 0
    fails = 0
    for code in range(1, codec.code_range):
        payload, errors = codec.decode(code)
        if payload == 0:
            fails += 1
        else:
            passes += 1
    total = passes + fails
    logger.log('  passes: {} ({:.0f}%), fails: {} ({:.0f}%)'.format(passes, (passes / total) * 100,
                                                             fails, (fails / total) * 100))
    logger.log('Error recovery for every possible N-bit flip for every possible codeword: 1..{} bit flips in {} code-bits'.
        format(codec.error_bits, codec.code_bits))
    errors = [[0,0] for _ in range(codec.error_bits+1)]  # correct, total for each N
    for payload in range(1, PAYLOAD_RANGE):  # every possible payload
        code = codec.encode(payload)
        for N in range(1, len(errors)):
            for error in codec.flips(N):
                bad_code = code ^ error  # flip N bits
                decode, error_bits = codec.decode(bad_code)
                if decode == payload:
                    # correct
                    errors[N][0] += 1
                errors[N][1] += 1
    for N, (good, total) in enumerate(errors):
        if total == 0:
            # nothing here
            continue
        bad = total - good
        logger.log('  {}-bit flips: {} good ({:.0f}%), {} bad ({:.0f}%)'.format(N, good, (good/total)*100,
                                                                         bad, (bad/total)*100))
    logger.log('Done')
    logger.log()  # close logger file
