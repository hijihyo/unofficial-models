import torch
from torch import Tensor, nn


def conv_batch_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    bias: bool = False,
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            *conv_batch_relu(in_channels, out_channels, kernel_size=1),
        )
        conv_kwargs = {"kernel_size": 3, "padding": 1}
        self.convs = nn.Sequential(
            *conv_batch_relu(in_channels, out_channels, **conv_kwargs),
            *conv_batch_relu(out_channels, out_channels, **conv_kwargs),
        )

    def forward(self, x: Tensor, feature: list[Tensor]) -> Tensor:
        x = self.upsample(x)
        combined = torch.cat([x, feature], dim=x.dim() - 3)
        return self.convs(combined)


class Down(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        conv_kwargs = {"kernel_size": 3, "padding": 1}
        self.convs = nn.Sequential(
            *conv_batch_relu(in_channels, out_channels, **conv_kwargs),
            *conv_batch_relu(out_channels, out_channels, **conv_kwargs),
        )
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        out = self.convs(x)
        return self.downsample(out), out


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: list[int] = [64, 128, 256, 512, 1024],
    ) -> None:
        super().__init__()
        channels = [in_channels, *channels]
        self.layers = nn.ModuleList(
            Down(channels[i - 1], channels[i]) for i, _ in enumerate(channels) if i > 0
        )

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        out_list = []
        for layer in self.layers:
            x, out = layer(x)
            out_list.append(out)
        return out, out_list


class Decoder(nn.Module):
    def __init__(
        self,
        channels: list[int] = [64, 128, 256, 512, 1024],
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        channels = channels[::-1]
        self.layers = nn.ModuleList(
            Up(channels[i - 1], channels[i]) for i, _ in enumerate(channels) if i > 0
        )
        self.layer_out = conv_batch_relu(channels[-1], num_classes, kernel_size=3, padding=1)

    def forward(self, x: Tensor, feature_list: list[Tensor]) -> Tensor:
        feature_list = feature_list[-2::-1]
        for i, layer in enumerate(self.layers):
            x = layer(x, feature_list[i])
        return self.layer_out(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        channels: list[int] = [64, 128, 256, 512, 1024],
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, channels)
        self.decoder = Decoder(channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        out, out_list = self.encoder(x)
        return self.decoder(out, out_list)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=[0],
        type=int,
    )
    parser.add_argument("--in-channels", default=1, type=int)
    parser.add_argument(
        "--channels",
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        type=int,
    )
    parser.add_argument("--num-classes", default=3, type=int)
    parser.add_argument("--batch-size", default=2, type=int)

    args = parser.parse_args()
    print("Args:")
    print("   gpus:", args.gpus)
    print("   in_channels:", args.in_channels)
    print("   channels:", args.channels)
    print("   num_classes:", args.num_classes)
    print("   batch_size:", args.batch_size)

    device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = UNet(args.in_channels, args.channels, args.num_classes).to(device)
    model = nn.DataParallel(model, device_ids=args.gpus)
    print(model)

    x = torch.randn(args.batch_size, args.in_channels, 256, 256).to(device)
    print("Input size:", x.size())
    encoded, _ = model.module.encoder(x)
    print("Encoded size:", encoded.size())
    out = model(x)
    print("Output size:", out.size())
