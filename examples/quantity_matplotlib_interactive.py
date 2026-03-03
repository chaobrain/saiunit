"""
Interactive Matplotlib example with saiunit.Quantity axes and data.

Run:
    python examples/quantity_matplotlib_interactive.py
"""

from __future__ import annotations

import numpy as np

import saiunit as u

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider


def make_signal(time_axis: u.Quantity, amplitude_cm: float, frequency_hz: float) -> u.Quantity:
    """Return a sinusoid Quantity in centimeters."""
    amplitude = amplitude_cm * u.cmeter
    angular = 2.0 * np.pi * frequency_hz * time_axis.to_decimal(u.second)
    return amplitude * np.sin(angular)


def main() -> None:
    # Quantity time axis
    t = np.linspace(0.0, 2.0, 800) * u.second
    initial_unit = u.cmeter

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.14, right=0.82, bottom=0.28)

    y = make_signal(t, amplitude_cm=50.0, frequency_hz=1.5)
    (line,) = ax.plot(t, y.to(initial_unit))
    ax.set_title("SAIUnit Quantity + Matplotlib (Interactive)")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Displacement ({initial_unit.dispname})")

    # Sliders
    ax_amp = plt.axes([0.14, 0.15, 0.60, 0.03])
    ax_freq = plt.axes([0.14, 0.10, 0.60, 0.03])
    s_amp = Slider(ax_amp, "Amplitude (cm)", 1.0, 100.0, valinit=50.0)
    s_freq = Slider(ax_freq, "Frequency (Hz)", 0.1, 10.0, valinit=1.5)

    # Unit selector
    ax_units = plt.axes([0.84, 0.60, 0.14, 0.22])
    r_units = RadioButtons(ax_units, ("cm", "m", "mm"), active=0)
    selected_unit = {"value": initial_unit}

    unit_map = {
        "cm": u.cmeter,
        "m": u.meter,
        "mm": u.mmeter,
    }

    def redraw() -> None:
        y_local = make_signal(t, s_amp.val, s_freq.val).to(selected_unit["value"])
        line.set_ydata(y_local.mantissa)
        ax.set_ylabel(f"Displacement ({selected_unit['value'].dispname})")
        fig.canvas.draw_idle()

    def on_slider_change(_value: float) -> None:
        redraw()

    def on_unit_change(label: str) -> None:
        selected_unit["value"] = unit_map[label]
        redraw()

    s_amp.on_changed(on_slider_change)
    s_freq.on_changed(on_slider_change)
    r_units.on_clicked(on_unit_change)

    plt.show()


if __name__ == "__main__":
    main()
