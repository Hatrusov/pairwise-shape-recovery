"""Reproduce Tables 6/7 and supplementary covariance checks; no bootstrap."""

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from fractions import Fraction as F
from math import factorial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pairwise_shape_recovery.inference import direct_covariance, recover_from_pairs  # noqa: E402


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def shape(x):
    z = x - x.mean(axis=-1, keepdims=True)
    v = np.mean(z * z, axis=-1)
    return np.mean(z**3, axis=-1) / v**1.5, np.mean(z**4, axis=-1) / v**2


def population_experiment(out, reps):
    rng = np.random.Generator(np.random.PCG64(260396436))
    rows = []
    raw = []
    families = [
        ("Normal", (0.0, 3.0), lambda size: rng.normal(size=size)),
        ("Uniform", (0.0, 1.8), lambda size: rng.uniform(-1, 1, size=size)),
        ("Exponential", (2.0, 9.0), lambda size: rng.exponential(size=size)),
        ("Gamma(3)", (2 / np.sqrt(3), 5.0), lambda size: rng.gamma(3, 1, size=size)),
        ("Student t(10)", (0.0, 4.0), lambda size: rng.standard_t(10, size=size)),
    ]
    for name, target, draw in families:
        for n in (10, 20, 30):
            x = draw((reps, n))
            g, b = shape(x)
            G = g * np.sqrt(n * (n - 1)) / (n - 2)
            z = x - x.mean(axis=1, keepdims=True)
            z /= np.sqrt(np.mean(z * z, axis=1, keepdims=True))
            ij = np.triu_indices(n, 1)
            recovered = []
            # Chunking preserves RNG order; limits memory for materialized pairs.
            for start in range(0, reps, 100):
                zz = z[start : start + 100]
                s = (zz[:, ij[0]] + zz[:, ij[1]]) / 2
                gs, bs = shape(s)
                gr = gs * np.sqrt(2 * n * (n - 2)) / (n - 4)
                br = -3 * n / (n - 8) + 2 * (n - 2) ** 2 / ((n - 1) * (n - 8)) * bs
                recovered.extend(zip(gr, br, strict=True))
            recovered = np.array(recovered)
            err = np.column_stack([G - target[0], b - target[1]])
            row = {"family": name, "n": n, "replications": reps}
            for j, label in enumerate(("G1", "beta2")):
                e = err[:, j]
                row.update(
                    {
                        f"bias_{label}": e.mean(),
                        f"bias_mcse_{label}": e.std(ddof=1) / np.sqrt(reps),
                        f"mse_{label}": np.mean(e * e),
                        f"mse_mcse_{label}": (e * e).std(ddof=1) / np.sqrt(reps),
                    }
                )
                row[f"max_reconstruction_error_{label}"] = np.max(
                    np.abs(recovered[:, j] - [G, b][j])
                )
            rows.append(row)
            raw.extend(
                {
                    "family": name,
                    "n": n,
                    "replication": i,
                    "G1": G[i],
                    "beta2": b[i],
                    "recovered_G1": recovered[i, 0],
                    "recovered_beta2": recovered[i, 1],
                }
                for i in range(reps)
            )
    write_csv(out / "table7_population.csv", rows)
    write_csv(out / "replication_statistics.csv", raw)
    return rows


def exact_counterexamples(out):
    rows = []
    witnesses = {}
    for q in range(3, 9):
        n = 2 ** (q - 1)
        E = [i for i in range(2 * n) if i.bit_count() % 2 == 0]
        odd = [i for i in range(2 * n) if i.bit_count() % 2]

        def pairs(x):
            n = len(x)
            return Counter(x[i] + x[j] for i in range(n) for j in range(i + 1, n))

        assert pairs(E) == pairs(odd)
        for r in range(q):
            assert sum(i**r for i in E) == sum(i**r for i in odd)
        delta = sum(i**q for i in E) - sum(i**q for i in odd)
        assert delta == (-1) ** q * factorial(q) * 2 ** (q * (q - 1) // 2)

        def moments(x):
            n = len(x)
            z = [F(i) - F(sum(x), n) for i in x]
            return [sum(t**r for t in z) / n for r in range(9)]

        e, o = moments(E), moments(odd)
        v = e[2]
        # Exact rational coefficient multiplied by a common variance power.
        if q == 3:
            functional = "gamma1"
            numerator = e[3] - o[3]
            power = 1.5
        elif q == 4:
            functional = "beta2"
            numerator = e[4] - o[4]
            power = 2
        elif q == 5:
            functional = "C12"
            numerator = (4 - 2 * e[4] / v**2) * (e[5] - o[5])
            power = 2.5
        elif q == 6:
            functional = "C11"
            numerator = e[6] - o[6]
            power = 3
        elif q == 7:
            functional = "C12"
            numerator = e[7] - o[7]
            power = 3.5
        else:
            functional = "C22"
            numerator = e[8] - o[8]
            power = 4
        assert numerator != 0
        rows.append(
            {
                "n": n,
                "q": q,
                "equal_pair_sums": True,
                "functional": functional,
                "exact_numerator": str(numerator),
                "variance": str(v),
                "variance_power": power,
                "absolute_difference": abs(float(numerator) / float(v) ** power),
            }
        )
        witnesses[str(n)] = {
            "even": E,
            "odd": odd,
            "pair_sum_counts": dict(sorted(pairs(E).items())),
        }
    write_csv(out / "table6_counterexamples.csv", rows)
    (out / "exact_witnesses.json").write_text(json.dumps(witnesses, indent=2))


def covariance_checks(out):
    rng = np.random.Generator(np.random.PCG64(260396437))
    rows = []
    for family in ("normal", "exponential", "student10"):
        for n in (3, 5, 10, 20, 30, 50, 100, 129, 256):
            errors = []
            eigen = []
            ratios = []
            for _ in range(100):
                x = (
                    rng.normal(size=n)
                    if family == "normal"
                    else (
                        rng.exponential(size=n)
                        if family == "exponential"
                        else rng.standard_t(10, size=n)
                    )
                )
                i, j = np.triu_indices(n, 1)
                s = (x[i] + x[j]) / 2
                rec = recover_from_pairs(s, n)
                c = direct_covariance(x)
                errors.append(np.max(np.abs(rec["C"] - c)))
                eigen.append(rec["min_eigenvalue"])
                ratios.append(rec["max_cancellation_ratio"])
            rows.append(
                {
                    "family": family,
                    "n": n,
                    "replications": 100,
                    "max_abs_C_error": max(errors),
                    "mean_abs_C_error": np.mean(errors),
                    "minimum_recovered_eigenvalue": min(eigen),
                    "max_cancellation_ratio": max(ratios),
                }
            )
    write_csv(out / "covariance_recovery.csv", rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replications", type=int, default=5000)
    p.add_argument("--output", type=Path, default=ROOT / "revision/results")
    args = p.parse_args()
    if args.replications < 2:
        raise ValueError("At least two replications required")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    rows = population_experiment(out, args.replications)
    exact_counterexamples(out)
    covariance_checks(out)
    (out / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "rng": "PCG64",
                "seed_population": 260396436,
                "seed_covariance": 260396437,
                "replications_per_population_cell": args.replications,
                "order": (
                    "family outer, n inner; draw all source samples "
                    "per cell before calculations"
                ),
                "floating_point": "float64",
                "bootstrap": False,
            },
            indent=2,
        )
    )
    print(
        "Completed",
        len(rows),
        "population cells; 6 exact counterexamples; 27 covariance cells",
    )
    for label in ("G1", "beta2"):
        print(
            label,
            "maximum error",
            max(r["max_reconstruction_error_" + label] for r in rows),
        )
    (out / "SHA256SUMS.txt").write_text(
        "\n".join(
            hashlib.sha256(f.read_bytes()).hexdigest() + "  " + f.name
            for f in sorted(out.iterdir())
            if f.is_file() and f.name != "SHA256SUMS.txt"
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
