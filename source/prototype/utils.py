

def nstr(number, fmt='.2f'):
    """ given a number that may be None, return an appropriate string """
    if number is None:
        return 'None'
    else:
        fmt = '{:' + fmt + '}'
        return fmt.format(number)


def vstr(vector, fmt='.2f', open='[', close=']'):
    """ given a list of numbers return a string representing them """
    if vector is None:
        return 'None'
    result = ''
    for pt in vector:
        result += ', ' + nstr(pt)
    return open + result[2:] + close


def wrapped_gap(x1, x2, limit_x):
    """ given two x co-ords that may be wrapped return the gap between the two,
        a legitimate gap must be less than half the limit,
        the returned gap may be +ve or -ve and represents the gap from x1 to x2,
        i.e. (x1 + gap) % limit_x = x2
        """

    dx = x2 - x1
    if dx < 0:
        # x1 bigger than x2, this is OK provided the gap is less than half the limit
        if (0 - dx) > (limit_x / 2):
            # gap too big, so its a wrap, so true gap is (x2 + limit) - x1
            dx = (x2 + limit_x) - x1
    else:
        # x1 smaller than x2
        if dx > (limit_x / 2):
            # gap too big, so its wrap, so true gap is  x2 - (x1 + limit)
            dx = x2 - (x1 + limit_x)
    return dx
