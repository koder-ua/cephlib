from typing import cast, Union, Tuple, TypeVar, List, Callable
from fractions import Fraction


RSMAP = [('Ki', 1024),
         ('Mi', 1024 ** 2),
         ('Gi', 1024 ** 3),
         ('Ti', 1024 ** 4),
         ('Pi', 1024 ** 5),
         ('Ei', 1024 ** 6)]

RSMAP_10_low = [('f', Fraction(1, 1000**4)),
                ('n', Fraction(1, 1000**3)),
                ('u', Fraction(1, 1000**2)),
                ('m', Fraction(1, 1000))]

RSMAP_10_hight = [('', 1),
                  ('K', 1000),
                  ('M', 1000 ** 2),
                  ('G', 1000 ** 3),
                  ('T', 1000 ** 4),
                  ('P', 1000 ** 5),
                  ('E', 1000 ** 6)]

SMAP_10_hight = {ext.strip().lower(): val for ext, val in RSMAP_10_hight}
SMAP = {ext[0].lower(): val for ext, val in RSMAP}
RSMAP_10 = [(n, float(v)) for n, v in RSMAP_10_low] + RSMAP_10_hight
RSMAP_10_exact = dict(RSMAP_10_low + RSMAP_10_hight)
RSMAP_all_exact = dict(RSMAP_10_low + RSMAP_10_hight + RSMAP)


def ssize2b(ssize: Union[str, int]) -> int:
    try:
        if isinstance(ssize, int):
            return ssize
        ssize = ssize.lower()
        if ssize[-1] in SMAP:
            return int(ssize[:-1]) * SMAP[ssize[-1]]
        return int(ssize)
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Unknown size format {!r}".format(ssize))


def to3digit(cval: float) -> str:
    # detect how many digits after dot to show
    if cval > 100:
        return str(int(cval))
    if cval > 10:
        if has_next_digit_after_coma(cval):
            return "{:.1f}".format(cval)
        else:
            return str(int(cval))
    if cval >= 1:
        if has_second_digit_after_coma(cval):
            return "{:.2f}".format(cval)
        elif has_next_digit_after_coma(cval):
            return "{:.1f}".format(cval)
        return str(int(cval))
    raise AssertionError("Can't get here")


def b2ssize(value: Union[int, float]) -> str:
    if isinstance(value, float) and value < 100:
        return b2ssize_10(value)

    value = int(value)
    if value < 1024:
        return str(value) + " "

    # make mypy happy
    scale = 1
    name = ""

    for name, scale in RSMAP:
        if value < 1024 * scale:
            return to3digit(float(value) / scale) + " " + name

    return "{} {}i".format(value // scale, name)


def has_next_digit_after_coma(x: float) -> bool:
    return int(x * 10) - int(x) * 10 != 0


def has_second_digit_after_coma(x: float) -> bool:
    return int(x * 100) - int(x * 10) * 10 != 0


def b2ssize_10(value: Union[int, float]) -> str:
    # make mypy happy
    scale = 1
    name = ""

    if value == 0.0:
        return "0"

    if value / RSMAP_10[0][1] < 1.0:
        return "{:.2e}".format(value)

    for name, scale in RSMAP_10:
        cval = value / scale
        if cval < 1000:
            return to3digit(cval) + " " + name
    return "{} {}".format(int(value // scale), name)


def split_unit(units: str) -> Tuple[Union[Fraction, int], str]:
    if len(units) > 2 and units[:2] in RSMAP_all_exact:
        return RSMAP_all_exact[units[:2]], units[2:]
    if len(units) > 1 and units[0] in RSMAP_10_exact:
        return RSMAP_10_exact[units[0]], units[1:]
    else:
        return 1, units


conversion_cache = {}


def unit_conversion_coef(from_unit: str, to_unit: str) -> Union[Fraction, int]:
    key = (from_unit, to_unit)
    if key in conversion_cache:
        return conversion_cache[key]

    f1, u1 = split_unit(from_unit)
    f2, u2 = split_unit(to_unit)

    assert u1 == u2, "Can't convert {!r} to {!r}".format(from_unit, to_unit)

    if isinstance(f1, int) and isinstance(f2, int):
        if f1 % f2 != 0:
            res = Fraction(f1, f2)
        else:
            res = f1 // f2
    else:
        res = f1 / f2

    if isinstance(res, Fraction) and cast(Fraction, res).denominator == 1:
        res = cast(Fraction, res).numerator

    conversion_cache[key] = res

    return res


def unit_conversion_coef_f(from_unit: str, to_unit: str) -> float:
    return float(unit_conversion_coef(from_unit, to_unit))
