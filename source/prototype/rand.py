
""" deterministic random number generator

Based on https://en.wikipedia.org/wiki/Pseudorandom_number_generator

The following is a very simple PRNG example written in JavaScript. It utilises a sequence of 
multiplications to output a seemingly random value which is than normalized to be in range 0 to 1.
In this example, 15485863 is the 1 000 000th prime number and 2038074743 the 100 000 000th one.

class PRNG
{
    seed = 0;

    Seed(seed)
    {
        this.seed = seed;
        let a = this.seed * 15485863;
        return (a * a * a % 2038074743) / 2038074743; //Will return in range 0 to 1 if seed >= 0 and -1 to 0 if seed < 0.
    }

    Next()
    {
        this.seed++;
        let a = this.seed * 15485863;
        return (a * a * a % 2038074743) / 2038074743;
    }
}

The example returns very similar results to JavaScript's Math.random() function.

"""

class Rand:
    """ generate a deterministic random sequence
        it is important that this algorithm is deterministic as it is used as part of the mapping
        of codes to numbers and that mapping must be static
        """

    SMALL_PRIME = 15485863  # must fit in 31 bits
    LARGE_PRIME = 2038074743  # must fit in 31 bits

    def __init__(self, seed=0):  # WARNING: DO NOT CHANGE THE DEFAULT SEED
        self.seed = max(seed, 0)  # we do not want a -ve seed 'cos we want a 0..1 range

    def rnd(self) -> float:
        self.seed += 1
        a = self.seed * Rand.SMALL_PRIME
        return ((a * a * a) % Rand.LARGE_PRIME) / Rand.LARGE_PRIME
