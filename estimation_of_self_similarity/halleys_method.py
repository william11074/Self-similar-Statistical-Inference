import math
from typing import Sequence
from numpy import sqrt

def solve_H_halley(a: Sequence[float], b: Sequence[float], *, H0: float = 0.5, tol: float = 1e-12, max_iter: int = 50) -> float:
    """
    Solve sum a_j * b_j^H = 1 for H in [0,1] using Halley's method.

    f(H)   = sum a_j b_j^H - 1
    f'(H)  = sum a_j b_j^H ln(b_j)
    f''(H) = sum a_j b_j^H (ln(b_j))^2
    
    Halley's method: 
    x_{n+1} = x_n - (2 f_n f'_n) / (2 (f'_n)^2 - f_n f''_n) with initial guess x_0 = H0

    Assumes b_j >= 1 (so ln(b_j) >= 0); requires b_j > 0 in general.
    """
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("a and b must be same non-zero length")
    if any(bj < 1.0 for bj in b):
        raise ValueError("This function assumes b_j >= 1.0 for all j")

    H = min(max(float(H0), 0.0), 1.0) # keep the initial guess in [0,1]

    for _ in range(max_iter):
        f = -1.0
        fp = 0.0
        fpp = 0.0

        for aj, bj in zip(a, b):
            if bj == 1.0:
                term = 1.0
                ln_b = 0.0
            else:
                ln_b = math.log(bj)
                term = math.exp(H * ln_b)

            contrib = aj * term
            f += contrib
            fp += contrib * ln_b
            fpp += contrib * (ln_b * ln_b)

        if abs(f) <= tol:
            return H

        denom = 2.0 * fp * fp - f * fpp
        if denom == 0.0 or not math.isfinite(denom):
            raise RuntimeError("Halley failed: denominator is zero/non-finite")

        step = (2.0 * f * fp) / denom
        H_next = H - step

        # Clamp to [0,1]
        if H_next < 0.0:
            H_next = 0.0
        elif H_next > 1.0:
            H_next = 1.0

        if abs(H_next - H) <= tol * (1.0 + abs(H_next)):
            return H_next

        H = H_next

    raise RuntimeError("Halley's method did not converge within max_iter")

# Adapted version of solve_H_halley for sub-fractional Brownian motion to account for additional terms from scaling parameter
def solve_H_halley_sfbm(a: Sequence[float], b: Sequence[float], *, H0: float = 0.5, tol: float = 1e-12, max_iter: int = 100) -> float:
    # Solve sum a_j * b_j^{2H} = 2 - 2^{2H-1} for H in [0,1] using Halley's method for sub-fractional Brownian motion.
    # Uses solve_H_halley with modified a and b lists to account for the right-hand side
    return solve_H_halley(
        [x / 2 for x in a] + [0.25],
        list(b) + [4],
        H0=H0,
        tol=tol,
        max_iter=max_iter,
    )

if __name__ == "__main__":
    # Example usage
    a = [0.03, 0.08]
    b = [2.0, 106]
    H = solve_H_halley(a, b, H0=0.5)
    print("H =", H)
    print("check =", sum(aj * (bj ** H) for aj, bj in zip(a, b)))