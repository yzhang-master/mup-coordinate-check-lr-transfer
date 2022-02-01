import pytest

from mup_transfer import width_multiplier


def test_readout_multiplier_tracks_width_ratio() -> None:
    assert width_multiplier(128, 32) == 0.25
    with pytest.raises(ValueError, match="positive"):
        width_multiplier(0, 32)
