import pytest

import saiunit as u

try:
    import matplotlib.pyplot as plt
    from matplotlib.units import ConversionError
except ImportError:
    pytest.skip("matplotlib is not installed", allow_module_level=True)


def test_matplotlib_compat():
    plt.figure()
    plt.plot([1, 2, 3])
    plt.show()

    plt.cla()
    plt.plot([1, 2, 3] * u.meter)
    plt.show()

    plt.cla()
    plt.plot([101, 125, 150] * u.cmeter)
    plt.show()

    plt.cla()
    plt.plot([101, 125, 150] * u.ms, [101, 125, 150])
    plt.plot([0.1, 0.15, 0.2] * u.second, [111, 135, 160])
    plt.show()

    plt.cla()
    plt.plot([101, 125, 150] * u.ms, [101, 125, 150] * u.cmeter)
    plt.plot([0.1, 0.15, 0.2] * u.second, [111, 135, 160] * u.cmeter)
    plt.show()

    with pytest.raises(ConversionError):
        plt.cla()
        plt.plot([101, 125, 150] * u.ms, [101, 125, 150] * u.cmeter)
        plt.plot([0.1, 0.15, 0.2] * u.second, [111, 135, 160] * u.cmeter)
        plt.plot([0.1, 0.15, 0.2] * u.second, [131, 155, 180] * u.mA)
        plt.show()

    # with pytest.raises(ConversionError):
    plt.cla()
    plt.plot([101, 125, 150], [101, 125, 151] * u.cmeter)
    plt.plot([101, 125, 150] * u.ms, [101, 125, 150] * u.cmeter)
    plt.show()

    plt.cla()
    plt.plot([101, 125, 150] * u.ms, [101, 125, 150] * u.cmeter)
    plt.plot([101, 125, 150] * u.ms, [101, 125, 150])
    plt.show()
