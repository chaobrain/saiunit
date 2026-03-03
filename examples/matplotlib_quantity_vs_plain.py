"""
Side-by-side comparison: Quantity-aware plotting vs plain numeric plotting.

Run:
    python examples/matplotlib_quantity_vs_plain.py
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
    # Same physical signal represented in two styles.
    t_q = np.linspace(0.0, 2.0, 400) * u.second
    y_q = (2.0 * u.volt) * np.cos(2.0 * np.pi * 2.0 * t_q.to_decimal(u.second))

    t_plain = t_q.to_decimal(u.second)
    y_plain = y_q.to_decimal(u.volt)

    fig, (ax_q, ax_p) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)

    ax_q.plot(t_q, y_q, color="tab:blue")
    ax_q.set_title("With Quantity")
    ax_q.set_xlabel("Time")
    ax_q.set_ylabel("Signal")
    ax_q.grid(alpha=0.3)

    ax_p.plot(t_plain, y_plain, color="tab:green")
    ax_p.set_title("Without Quantity")
    ax_p.set_xlabel("Time [s]")
    ax_p.set_ylabel("Signal [V]")
    ax_p.grid(alpha=0.3)

    fig.suptitle("Matplotlib: Quantity vs Plain Numeric")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
