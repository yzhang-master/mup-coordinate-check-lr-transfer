"""Width-transfer experiments with maximal-update parametrization."""

__version__ = "0.1.0"


def width_multiplier(width: int, base_width: int) -> float:
    """Return the μP readout multiplier for a hidden width."""
    if width <= 0 or base_width <= 0:
        raise ValueError("width and base_width must be positive")
    return base_width / width
