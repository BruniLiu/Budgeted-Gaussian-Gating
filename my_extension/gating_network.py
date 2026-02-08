import torch
import torch.nn as nn


class DensifyGate(nn.Module):
    """
    Stabilized version of gating network.

    Improvements over previous version:
    - Better initialization (bias shifted to encourage w≈1 initially)
    - Remove risky per-batch normalization
    - Add optional input scaling
    - Add debug utilities
    """

    def __init__(self, input_dim=7, hidden_dim=64):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """
        Use Xavier initialization + positive bias
        to ensure initial gate outputs are close to 1.
        """
        linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]

        for m in linear_layers[:-1]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

        # Special initialization for the last layer
        last = linear_layers[-1]
        nn.init.xavier_uniform_(last.weight)

        # This is CRUCIAL:
        # Initialize bias to a positive value so sigmoid(bias) ≈ 0.88
        nn.init.constant_(last.bias, 2.0)

    def forward(self, feat):
        """
        feat: (N, 7) tensor composed of [xyz, scaling, opacity]
        return: (N, 1) importance score in range [0, 1]
        """

        # Lightweight feature scaling instead of full normalization
        # Prevents huge scale differences between xyz and opacity
        with torch.no_grad():
            scale = feat.abs().mean(dim=0, keepdim=True) + 1e-6

        feat_scaled = feat / scale

        scores = self.net(feat_scaled)

        return scores


if __name__ == "__main__":
    # Simple sanity check
    gate = DensifyGate()

    test_input = torch.randn(1000, 7)
    output = gate(test_input)

    print("Input shape :", test_input.shape)
    print("Output shape:", output.shape)
    print("Initial mean score:", output.mean().item())
