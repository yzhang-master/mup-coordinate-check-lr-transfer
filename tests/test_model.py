import pytest
import torch

from mup_transfer.coordinates import coordinate_check, coordinate_rms
from mup_transfer.model import MLPConfig, WidthMLP


def test_mup_readout_multiplier_is_applied_to_projection() -> None:
    config = MLPConfig(width=16, base_width=8, output_dim=2, depth=1)
    mup = WidthMLP(config).eval()
    standard = WidthMLP(
        MLPConfig(width=16, base_width=8, output_dim=2, depth=1, parametrization="standard")
    ).eval()
    standard.load_state_dict(mup.state_dict())
    inputs = torch.randn(5, config.input_dim)
    torch.testing.assert_close(mup(inputs), standard(inputs) * 0.5)


def test_initialization_uses_fan_in_and_zero_biases() -> None:
    torch.manual_seed(0)
    model = WidthMLP(MLPConfig(width=128, base_width=16))
    for layer in model.layers:
        assert layer.weight.std().item() == pytest.approx(layer.in_features**-0.5, rel=0.2)
        torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))


def test_coordinate_rms_preserves_feature_axis() -> None:
    values = torch.tensor([[[3.0, 4.0], [0.0, 8.0]]])
    torch.testing.assert_close(coordinate_rms(values), torch.tensor([4.5**0.5, 40.0**0.5]))


def test_hidden_coordinates_remain_bounded_as_width_grows() -> None:
    report = coordinate_check([16, 32, 64], steps=2)
    for key in ("initial_rms", "trained_rms"):
        values = report[key]
        assert min(values) > 0.2
        assert max(values) / min(values) < 2.0


def test_mup_scales_readout_gradient_by_width_ratio() -> None:
    generator = torch.Generator().manual_seed(11)
    inputs = torch.randn(16, 8, generator=generator)
    gradients = {}
    for parametrization in ("mup", "standard"):
        torch.manual_seed(3)
        model = WidthMLP(MLPConfig(width=64, base_width=16, parametrization=parametrization))
        model(inputs).sum().backward()
        gradients[parametrization] = model.layers[-1].weight.grad.detach().clone()
    torch.testing.assert_close(gradients["mup"], gradients["standard"] * 0.25)
