from math import log2

def is_power_of_two(val):
    return log2(val).is_integer()


def default(val, d):
    if val is None:
        return d
    return val
