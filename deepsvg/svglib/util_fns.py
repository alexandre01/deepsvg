import math


def get_roots(a, b, c):
    if a == 0:
        if b == 0:
            return []
        return [-c / b]
    r = b * b - 4 * a * c
    if r < 0:
        return []
    elif r == 0:
        x0 = -b / (2 * a)
        return [x0]

    x1, x2 = (-b - math.sqrt(r)) / (2 * a), (-b + math.sqrt(r)) / (2 * a)
    return x1, x2
