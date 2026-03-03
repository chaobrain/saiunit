"""
Matplotlib examples using saiunit.Quantity values directly.

Run:
    python examples/matplotlib_quantity_basics.py
"""

from __future__ import annotations

import numpy as np

import saiunit as u

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "This example requires matplotlib. Install it with: pip install matplotlib"
    ) from exc


def main() -> None:
    t = np.linspace(0.0, 3.0, 500) * u.second

    # Displacement in centimeters.
    x = (40.0 * u.cmeter) * np.sin(2.0 * np.pi * 1.25 * t.to_decimal(u.second))

    # Velocity from numerical derivative in SI units.
    v = (
        np.gradient(x.to_decimal(u.meter), t.to_decimal(u.second))
        * (u.meter / u.second)
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Quantity axis: Matplotlib converter handles unit labels and conversion.
    ax1.plot(t, x, label="x(t) [auto unit from Quantity]")
    ax1.plot(t, x.to(u.meter), "--", label="x(t) converted to meter")
    ax1.set_title("Quantity-Aware Matplotlib")
    ax1.set_ylabel("Displacement")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    ax2.plot(t, v, color="tab:orange", label="v(t)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Velocity")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
