import math
import numpy


def auto_edges(vals, log_base=2, bins=20, round_base=10, log_space=True):
    lower = numpy.min(vals)
    upper = numpy.max(vals)
    return auto_edges2(lower, upper, log_base, bins, round_base, log_space=log_space)


MIN_VAL = 1
MAX_LIN_DIFF = 100
UPPER_ROUND_COEF = 0.99999


def auto_edges2(lower, upper, log_base=2, bins=20, round_base=10, log_space=True):
    if lower == upper:
        return numpy.array([lower * 0.9, lower * 1.1])

    if round_base and lower > MIN_VAL:
        lower = round_base ** (math.floor(math.log(lower) / math.log(round_base)))
        upper = round_base ** (math.floor(math.log(lower) / math.log(round_base) + UPPER_ROUND_COEF))

    if lower < MIN_VAL or upper / lower < MAX_LIN_DIFF or not log_space:
        return numpy.linspace(lower, upper, bins + 1)

    lower_lg = math.log(lower) / math.log(log_base)
    upper_lg = math.log(upper) / math.log(log_base)
    return numpy.logspace(lower_lg, upper_lg, bins + 1, base=log_base)
