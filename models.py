"""models.py — D-ResNet, ResNet-18, ResNet-34 and VGG16 built layer-by-layer from
the tables of Theerthagiri et al., "D-ResNet...", Multimedia Tools and
Applications (2024) 83:14539-14563. NO torchvision architectures, NO pretrained
weights (spec section 6).

Conventions
-----------
- forward() returns LOGITS. The softmax stated in the paper's tables is applied
  by torch.nn.CrossEntropyLoss (spec section 6, DEVIATIONS.md row 9).
- Bottleneck downsampling: stride 2 on the FIRST 1x1 of the stage's first block
  (He et al. original placement, matches the paper's 3.8e9 FLOPs figure).
- Heads (paper-specific):
  * D-ResNet-50: avgpool -> FC(2048, 1000) + ReLU -> FC(1000, num_classes)
    (the 1000-neuron layer is stated in the paper; the 6-way layer is required
    for the six classes of paper Fig. 1).
  * ResNet-18 / ResNet-34 (Table 5): avgpool -> FC(feature, num_classes).
  * VGG16 (Table 4 + Sec 4.1): flatten 25088 -> FC4096 (ReLU, dropout 0.5)
    -> FC4096 (ReLU, dropout 0.5) -> FC(num_classes). No batch norm.
- Initialisation: Kaiming normal for convolutions (fan_out, ReLU gain), BatchNorm
  weight=1 / bias=0, Linear default PyTorch initialisation. Seed set elsewhere.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Paper Fig. 2 / confusion-matrix class order (spec section 1).
CLASS_ORDER = ["Ilmenite", "Rutile", "Zircon", "Garnet", "Monazite", "Sillimanite"]

# Reference parameter counts with a 1000-class head (standard networks).
REFERENCE_PARAMS_1000_HEAD = {
    "resnet18": 11_689_512,
    "resnet34": 21_797_672,
    "dresnet50": 25_557_032,
    "vgg16": 138_357_544,
}

# Paper FLOPs column (Table 2 / Table 5): multiply-accumulates, in FLOPs.
PAPER_FLOPS = {"resnet18": 1.8e9, "resnet34": 3.6e9, "dresnet50": 3.8e9}

# Paper Table 2 / Table 5 stage output sizes at 224x224 input.
PAPER_STAGE_SIZES = [112, 56, 28, 14, 7]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    """Table 5 block: [3x3 C, 3x3 C] with BN after every convolution."""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)  # Eq. 1: y = F(x, W) + x, then ReLU


class Bottleneck(nn.Module):
    """Table 2 block: [1x1 64, 3x3 64, 1x1 256] pattern (widths x4 per stage).
    Stride 2 sits on the FIRST 1x1 when the stage downsamples (He et al.).
    """

    expansion = 4

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        width = out_ch
        self.conv1 = nn.Conv2d(in_ch, width, 1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width, width, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, width * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(width * self.expansion)
        self.downsample = None
        if stride != 1 or in_ch != width * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, width * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(width * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)  # Eq. 1: y = F(x, W) + x, then ReLU


# ---------------------------------------------------------------------------
# ResNet family (Tables 2 and 5)
# ---------------------------------------------------------------------------
class ResNet(nn.Module):
    """Stem: conv 7x7/64 stride 2 + BN + ReLU + maxpool 3x3 stride 2.
    Four stages; stride 2 at the first block of conv3.x..conv5.x.
    """

    def __init__(self, block, layers, num_classes: int = 6, head: str = "paper"):
        """head='paper': per-paper classification head for this architecture.
        head='standard': plain FC(feature, 1000) used ONLY for the parameter /
        FLOP verification against the standard reference networks.
        """
        super().__init__()
        self.in_ch = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_stage(block, 64, layers[0], stride=1)
        self.layer2 = self._make_stage(block, 128, layers[1], stride=2)
        self.layer3 = self._make_stage(block, 256, layers[2], stride=2)
        self.layer4 = self._make_stage(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        feat = 512 * block.expansion
        if head == "standard":
            self.fc = nn.Linear(feat, 1000)
            self.paper_fc1 = None
            self.paper_fc2 = None
        elif block.expansion == 4:
            # D-ResNet per the paper: 1000-neuron FC (+ReLU) then num_classes.
            self.fc = None
            self.paper_fc1 = nn.Linear(feat, 1000)
            self.relu_fc = nn.ReLU(inplace=True)
            self.paper_fc2 = nn.Linear(1000, num_classes)
        else:
            # Table 5: average pool -> FC(feature, 6).
            self.fc = nn.Linear(feat, num_classes)
            self.paper_fc1 = None
            self.paper_fc2 = None
        self._init_weights()

    def _make_stage(self, block, width: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [block(self.in_ch, width, stride)]
        self.in_ch = width * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_ch, width, 1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.fc is not None:
            return self.fc(x)
        return self.paper_fc2(self.relu_fc(self.paper_fc1(x)))


def resnet18(num_classes: int = 6, head: str = "paper") -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, head)


def resnet34(num_classes: int = 6, head: str = "paper") -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, head)


def dresnet50(num_classes: int = 6, head: str = "paper") -> ResNet:
    """D-ResNet = ResNet-50 per Table 2 (bottleneck, 3-4-6-3, x4 widths)."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, head)


# ---------------------------------------------------------------------------
# VGG16 (Table 4 + Section 4.1)
# ---------------------------------------------------------------------------
_VGG_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


class VGG16(nn.Module):
    """13 conv layers (3x3 s1 same padding, bias=True) in blocks 2,2,3,3,3 with
    64,128,256,512,512 filters, each block followed by 2x2 stride-2 max pool;
    flatten 7x7x512 = 25088; FC 25088->4096 (ReLU, dropout 0.5) ->
    FC 4096->4096 (ReLU, dropout 0.5) -> FC -> num_classes. NO batch norm
    (spec section 6; keeps the reference count 138,357,544 exact).
    """

    def __init__(self, num_classes: int = 6, head: str = "paper"):
        super().__init__()
        self.features = self._make_features()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # keeps 7x7x512 = 25088 at any input
        in_fc = 512 * 7 * 7
        # Hidden width is always 4096 (Table 4); the head flag only changes the
        # final layer: 6 outputs for the paper, 1000 for reference verification.
        self.classifier = nn.Sequential(
            nn.Linear(in_fc, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        self._init_weights()

    @staticmethod
    def _make_features() -> nn.Sequential:
        layers, in_ch = [], 3
        for v in _VGG_CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [
                    nn.Conv2d(in_ch, v, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                ]
                in_ch = v
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def vgg16(num_classes: int = 6, head: str = "paper") -> VGG16:
    return VGG16(num_classes, head)


# ---------------------------------------------------------------------------
# Verification helpers (spec section 6: every model notebook prints these)
# ---------------------------------------------------------------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model: nn.Module, input_size=(1, 3, 224, 224)) -> dict:
    """Multiply-accumulates via forward hooks on Conv2d and Linear layers."""
    totals = {"conv": 0, "linear": 0}

    def conv_hook(mod, inp, out):
        o = out.numel() // out.shape[0]
        k = mod.in_channels // mod.groups * mod.kernel_size[0] * mod.kernel_size[1]
        totals["conv"] += o * k

    def linear_hook(mod, inp, out):
        totals["linear"] += mod.in_features * mod.out_features

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(*input_size))
    model.train(was_training)
    for h in handles:
        h.remove()
    totals["total"] = totals["conv"] + totals["linear"]
    return totals


def stage_sizes(model: nn.Module, input_size=(1, 3, 224, 224)) -> list:
    """Spatial sizes after stem and each stage: [112, 56, 28, 14, 7] expected."""
    sizes = []
    module = model if isinstance(model, ResNet) else None
    if module is not None:
        # Table 2/5 stage outputs at 224x224: conv1=112, conv2.x=56 (stem
        # maxpool belongs to conv2.x in Table 2), conv3.x=28, conv4.x=14, conv5.x=7.
        x = torch.zeros(*input_size)
        with torch.no_grad():
            x = module.stem[0](x)  # conv 7x7 stride 2 -> 112
            sizes.append(x.shape[-1])
            x = module.stem[1:](x)  # BN + ReLU + maxpool stride 2
            for layer in [module.layer1, module.layer2, module.layer3, module.layer4]:
                x = layer(x)
                sizes.append(x.shape[-1])
        return sizes
    else:  # VGG16: spatial size after each max pool inside features
        handles, xs = [], []

        def rec_hook(mod, inp, out):
            xs.append(tuple(out.shape[-2:]))

        for m in model.features:
            handles.append(m.register_forward_hook(rec_hook))
        with torch.no_grad():
            model.features(torch.zeros(*input_size))
        for h in handles:
            h.remove()
        pool_idx = [i for i, m in enumerate(model.features) if isinstance(m, nn.MaxPool2d)]
        return [xs[i][0] for i in pool_idx]
    raise ValueError("stage_sizes supports ResNet family and VGG16 only")


def verify_architecture(name: str) -> dict:
    """Spec section 6 verification. Returns (and caller should print) a dict:
    stage sizes vs Table 2/5, output shape, parameter count vs the standard
    reference (1000-class head), MACs vs the paper's FLOPs column.
    """
    builders = {"resnet18": resnet18, "resnet34": resnet34, "dresnet50": dresnet50, "vgg16": vgg16}
    build = builders[name]

    model = build(num_classes=6, head="paper").eval()
    out6 = model(torch.zeros(2, 3, 224, 224))
    ref = build(num_classes=1000, head="standard").eval()
    n_params = count_params(ref)
    macs = count_macs(ref)
    sizes = stage_sizes(ref)
    paper_flops = PAPER_FLOPS.get(name)

    result = {
        "model": name,
        "stage_sizes": sizes,
        "stage_sizes_ok": sizes == PAPER_STAGE_SIZES if name != "vgg16" else sizes == [112, 56, 28, 14, 7],
        "output_shape_6class": tuple(out6.shape),
        "output_ok": tuple(out6.shape[1:]) == (6,),
        "params_1000_head": n_params,
        "params_reference": REFERENCE_PARAMS_1000_HEAD[name],
        "params_ok": n_params == REFERENCE_PARAMS_1000_HEAD[name],
        "macs_total": macs["total"],
        "paper_flops": paper_flops,
        "macs_ok": (paper_flops is None) or (0.90 <= macs["total"] / paper_flops <= 1.10),
    }
    result["all_ok"] = all(
        [result["output_ok"], result["params_ok"], result["macs_ok"]]
        + ([result["stage_sizes_ok"]] if name != "vgg16" else [])
    )
    return result


def format_verification(results: list) -> str:
    lines = []
    for r in results:
        lines.append(f"== {r['model']} ==")
        lines.append(f"  stage sizes      : {r['stage_sizes']} (paper: {PAPER_STAGE_SIZES}) -> {'OK' if r.get('stage_sizes_ok') else 'MISMATCH'}")
        lines.append(f"  output shape     : {r['output_shape_6class']} -> {'OK' if r['output_ok'] else 'MISMATCH'}")
        lines.append(
            f"  params (1000 hd) : {r['params_1000_head']:,} (ref {r['params_reference']:,}) -> {'OK' if r['params_ok'] else 'MISMATCH'}"
        )
        flops = f"{r['paper_flops']:.1e}" if r["paper_flops"] else "n/a (paper silent)"
        lines.append(f"  MACs             : {r['macs_total'] / 1e9:.2f}G (paper FLOPs {flops}) -> {'OK' if r['macs_ok'] else 'MISMATCH'}")
        lines.append(f"  ALL OK           : {r['all_ok']}")
    return "\n".join(lines)


BUILDERS = {"dresnet50": dresnet50, "resnet18": resnet18, "resnet34": resnet34, "vgg16": vgg16}
MODEL_NAMES = ["dresnet50", "resnet18", "resnet34", "vgg16"]
DISPLAY_NAMES = {"dresnet50": "D-ResNet", "resnet18": "ResNet-18", "resnet34": "ResNet-34", "vgg16": "VGG16"}
