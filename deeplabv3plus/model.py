from typing import List, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter


def conv_batch_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
):
    """A PyTorch sequential container consisting of Conv2d, BatchNorm, and ReLU activation layer.

    Args:
        in_channels (int)
        out_channels (int)
        kernel_size (int)
        stride (int, optional): Defaults to 1.
        padding (int, optional): Defaults to 0.
        dilation (int, optional): Defaults to 1.

    Shape:
        Input: (N, C_in, H_in, W_in)
        Output: (N, C_out, H_out, W_out)
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ASPP(nn.Module):
    def __init__(
        self,
        output_stride: int = 16,
        size: Tuple[int] = (256, 256),
        dilations: List[int] = [6, 12, 18],
    ):
        """A PyTorch Atrous Spatial Pyramid Pooling (ASPP) module

        Args:
            output_stride (int, optional): Defaults to 16.
            size (Tuple[int], optional): Defaults to (256, 256).
            dilations (List[int], optional): Defaults to [6, 12, 18].

        Shape:
            Input: (N, C_in, H_in, W_in)
            Output: (N, C_out, H_out, W_out)
        """
        assert output_stride in {8, 16}
        assert all(s > 0 for s in size)
        assert all(d > 0 for d in dilations)
        super().__init__()

        layers = []
        layers.append(
            conv_batch_relu(in_channels=2048, out_channels=256, kernel_size=1)
        )

        unit_rate = output_stride // 8
        dilations = [unit_rate * dilation for dilation in dilations]
        for dilation in dilations:
            layers.append(
                conv_batch_relu(
                    in_channels=2048,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                )
            )

        new_size = tuple(s // output_stride for s in size)
        layers.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=1),
                *conv_batch_relu(in_channels=2048, out_channels=256, kernel_size=1),
                nn.Upsample(size=new_size, mode="bilinear", align_corners=False),
            )
        )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor):
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=x.dim() - 3)


class DeepLabV3PlusEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        backbone_name: str = "resnet50",
        output_stride: int = 16,
        size: Tuple[int] = (256, 256),
        dilations: List[int] = [6, 12, 18],
        dropout: float = 0.0,
    ):
        """A PyTorch DeepLabv3+ Encoder module

        Args:
            in_channels (int, optional): Defaults to 1.
            backbone_name (str, optional): Defaults to "resnet50".
            output_stride (int, optional): Defaults to 16.
            size (Tuple[int], optional): Defaults to (256, 256).
            dilations (List[int], optional): Defaults to [6, 12, 18].
            dropout (float, optional): Defaults to 0.0.

        Shape:
            Input: (N, C_in, H_in, W_in) or (C_in, H_in, W_in)
            Output: (N, C_out, H_out, W_out) or (C_out, H_out, W_out)
        """
        assert in_channels > 0
        assert backbone_name in {"resnet50", "resnet101", "resnet152"}
        assert output_stride in {8, 16}
        assert 0 <= dropout <= 1
        super().__init__()
        self.dropout = dropout

        self.layer_in = conv_batch_relu(in_channels, out_channels=3, kernel_size=1)

        backbone = getattr(models, backbone_name)(
            pretrained=True,
            progress=True,
            replace_stride_with_dilation=[
                output_stride <= 4,
                output_stride <= 8,
                output_stride <= 16,
            ],
        )
        # get the intermediate output of ResNet
        self.backbone = IntermediateLayerGetter(
            backbone,
            return_layers={
                "layer1": "low_level",
                "layer4": "out",
            },
        )

        self.aspp = ASPP(output_stride, size, dilations)
        self.layer_out = conv_batch_relu(
            in_channels=1280, out_channels=256, kernel_size=1
        )

        for m in self.layer_in.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
        for m in self.aspp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
        for m in self.layer_out.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x: Tensor):
        assert x.dim() in {3, 4}
        is_batched = x.dim() == 4
        if not is_batched:
            x = x.unsqueeze(dim=0)
        out = self.layer_in(x)
        out = self.backbone(out)
        low_level = out["low_level"]
        out = out["out"]
        out = self.aspp(out)
        out = self.layer_out(
            F.dropout(out, self.dropout, self.training) if self.dropout > 0 else out
        )
        if not is_batched:
            out = out.squeeze(dim=0)
            low_level = low_level.squeeze(dim=0)
        return out, low_level


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(
        self,
        output_stride: int = 16,
        num_classes: int = 1,
        dropout: float = 0.0,
    ):
        """A PyTorch DeepLabv3+ Decoder module

        Args:
            output_stride (int, optional): Defaults to 16.
            num_classes (int, optional): Defaults to 1.
            dropout (float, optional): Defaults to 0.0.

        Shape:
            Input: (N, C_in, H_in, W_in) or (C_in, H_in, W_in)
            Output: (N, C_out, H_out, W_out) or (C_out, H_out, W_out)
        """
        super().__init__()
        self.dropout = dropout

        self.layer_x = nn.Upsample(
            scale_factor=output_stride // 4, mode="bilinear", align_corners=False
        )
        self.layer_lowlevel = conv_batch_relu(
            in_channels=256, out_channels=256, kernel_size=1
        )
        self.layer_out = nn.Sequential(
            *conv_batch_relu(
                in_channels=512,
                out_channels=num_classes,
                kernel_size=3,
                padding=1,
            ),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
        )

        for m in self.layer_x.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
        for m in self.layer_lowlevel.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
        for m in self.layer_out.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x: Tensor, low_level: Tensor):
        in1 = self.layer_x(x)
        in2 = self.layer_lowlevel(low_level)
        combined = torch.cat([in1, in2], dim=x.dim() - 3)
        out = self.layer_out(
            F.dropout(combined, self.dropout, self.training)
            if self.dropout > 0
            else combined
        )
        return out


class DeepLabV3Plus(nn.Module):
    """DeepLabv3+ originally from [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611v3)"""

    def __init__(
        self,
        in_channels: int = 1,
        backbone_name: str = "resnet50",
        output_stride: int = 16,
        size: Tuple[int] = (256, 256),
        dilations: List[int] = [6, 12, 18],
        num_classes: int = 1,
        dropout: float = 0.0,
    ):
        assert in_channels > 0
        assert output_stride in {8, 16}
        assert num_classes > 0
        assert 0 <= dropout <= 1
        super().__init__()
        self.bias = False

        self.encoder = DeepLabV3PlusEncoder(
            in_channels, backbone_name, output_stride, size, dilations, dropout
        )
        self.decoder = DeepLabV3PlusDecoder(output_stride, num_classes, dropout)

    def forward(self, x: Tensor):
        out, low_level = self.encoder(x)
        out = self.decoder(out, low_level)
        return out

    def encode(self, x: Tensor):
        return self.encoder(x)

    def decode(self, x: Tensor, low_level: Tensor):
        return self.decoder(x, low_level)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--gpus",
    #     nargs="+",
    #     default=[0],
    #     type=int,
    # )
    parser.add_argument("--in-channels", default=1, type=int)
    parser.add_argument("--backbone-name", default="resnet50", type=str)
    parser.add_argument("--output-stride", default=16, type=int)
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--batch-size", default=2, type=int)

    args = parser.parse_args()
    print("Args:")
    # print("   gpus:", args.gpus)
    print("   in_channels:", args.in_channels)
    print("   backbone_name:", args.backbone_name)
    print("   output_stride:", args.output_stride)
    print("   num_classes:", args.num_classes)
    print("   batch_size:", args.batch_size)

    device = torch.device("cpu")
    print("Device:", device)

    model = DeepLabV3Plus(
        args.in_channels,
        args.backbone_name,
        args.output_stride,
        num_classes=args.num_classes,
    ).to(device)
    # model = nn.DataParallel(model, device_ids=args.gpus)
    print(model)

    x = torch.randn(args.batch_size, args.in_channels, 256, 256).to(device)
    print("Input size:", x.size())
    # encoded, _ = model.module.encoder(x)
    encoded, _ = model.encoder(x)
    print("Encoded size:", encoded.size())
    out = model(x)
    print("Output size:", out.size())
