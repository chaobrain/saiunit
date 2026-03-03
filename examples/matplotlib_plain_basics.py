"""
Matplotlib examples without unit-aware Quantity objects.

Run:
    python examples/matplotlib_plain_basics.py
"""

from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "This example requires matplotlib. Install it with: pip install matplotlib"
    ) from exc


def main() -> None:
    t_s = np.linspace(0.0, 3.0, 500)  # seconds

    # Explicitly keep track of units in variable names and labels.
    x_cm = 40.0 * np.sin(2.0 * np.pi * 1.25 * t_s)
    x_m = x_cm / 100.0
    v_m_s = np.gradient(x_m, t_s)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(t_s, x_cm, label="x(t) [cm]")
    ax1.plot(t_s, x_m, "--", label="x(t) [m]")
    ax1.set_title("Plain Matplotlib (No Quantity)")
    ax1.set_ylabel("Displacement")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    ax2.plot(t_s, v_m_s, color="tab:orange", label="v(t) [m/s]")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Velocity [m/s]")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
