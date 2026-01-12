import torch

from mup_transfer.model import MLPConfig, WidthMLP


def coordinate_rms(activations: torch.Tensor) -> torch.Tensor:
    """Measure each feature coordinate independently over batch and sequence axes."""
    if activations.ndim < 2:
        raise ValueError("activations must have a feature dimension")
    axes = tuple(range(activations.ndim - 1))
    return activations.float().square().mean(dim=axes).sqrt()


def coordinate_check(
    widths: list[int],
    *,
    base_width: int = 16,
    steps: int = 3,
    learning_rate: float = 0.03,
    seed: int = 7,
) -> dict[str, list[float] | list[int]]:
    """Track hidden-coordinate RMS before and after identical synthetic Adam updates."""
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(24, 8, generator=generator)
    targets = torch.randn(24, 4, generator=generator)
    initial: list[float] = []
    final: list[float] = []
    for width in widths:
        torch.manual_seed(seed)
        model = WidthMLP(MLPConfig(width=width, base_width=base_width))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        with torch.no_grad():
            _, activations = model.forward_with_activations(inputs)
            initial.append(torch.cat([coordinate_rms(x) for x in activations]).mean().item())
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            _, activations = model.forward_with_activations(inputs)
            final.append(torch.cat([coordinate_rms(x) for x in activations]).mean().item())
    return {"widths": widths, "initial_rms": initial, "trained_rms": final}
