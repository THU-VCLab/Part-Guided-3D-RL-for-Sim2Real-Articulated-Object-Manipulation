import torch
import torch.nn as nn


class ImgEncoder(nn.Module):
    """Convolutional encoder for image-based observations."""

    def __init__(self, in_ch, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        # assert len(obs_shape) == 3
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.output_dim = 8 * 15
        self.output_logits = False
        self.feature_dim = feature_dim

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_ch, self.num_filters, 3, stride=2),
            ]
        )

        for i in range(num_layers - 1):
            self.convs.append(
                nn.Conv2d(self.num_filters, self.num_filters, 3, stride=2)
            )

        self.head = nn.Sequential(
            nn.Linear(self.num_filters * self.output_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )

        self.outputs = dict()

    def forward_conv(self, obs):
        obs = obs / 255.0

        conv = torch.relu(self.convs[0](obs))

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = self.head(h)
        if not self.output_logits:
            out = torch.tanh(out)

        return out


if __name__ == "__main__":
    data = torch.randn(4, 7, 144, 256)
    net_encoder = ImgEncoder(in_ch=7, feature_dim=256, num_layers=4, num_filters=32)
    print(net_encoder)
    out = net_encoder(data)
    print(out.shape)
