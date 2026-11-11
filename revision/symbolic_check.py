"""Optional independent symbolic expansion of equations (20)-(22)."""

import sympy as s

t, g, b = s.symbols("t g b")
L = s.symbols("L0:9")
p = t**3 - s.Rational(3, 2) * g * t * t - 3 * t + g / 2
q = t**4 - 2 * b * t * t - 4 * g * t + b


def expectation(poly):
    return s.expand(sum(c * L[k[0]] for k, c in s.Poly(poly, t).terms())).subs(
        {L[0]: 1, L[1]: 0, L[2]: 1, L[3]: g, L[4]: b}
    )


formulas = [
    L[6] - 3 * g * L[5] + (9 * g * g / 4 - 6) * b + 35 * g * g / 4 + 9,
    L[7]
    - s.Rational(3, 2) * g * L[6]
    - (2 * b + 3) * L[5]
    + 3 * g * b * b
    + s.Rational(3, 2) * g * b
    + 12 * g
    + 6 * g**3,
    L[8] - 4 * b * L[6] - 8 * g * L[5] + 4 * b**3 - b * b + 16 * g * g * b + 16 * g * g,
]
for product, formula in zip([p * p, p * q, q * q], formulas, strict=True):
    assert s.simplify(expectation(product) - formula) == 0
print("Three covariance polynomial identities verified symbolically.")
