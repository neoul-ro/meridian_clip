#!/usr/bin/env python3

"""
Meridian Perception Frontend - CLIP visual encoder ONNX export.

CLIP checkpoint(.pt)에서 image encoder만 떼어 ONNX 그래프로 내보낸다.

노드는 encode_image만 사용하므로 text encoder는 export 대상이 아니다.
이를 빼면 파라미터가 절반 이하로 줄고 TensorRT 엔진도 작아진다.

주의할 점:
    - clip.load의 기본 경로는 TorchScript JIT + fp16이라 export가 깨진다.
      반드시 device="cpu", jit=False로 읽어 fp32 nn.Module을 얻어야 한다.
    - CLIP.forward는 (image, text) 두 입력을 받아 logits를 반환하므로
      그대로 export할 수 없다. visual encoder만 감싸서 내보낸다.
    - opset 17부터 LayerNormalization이 네이티브 연산이다. ViT는 LayerNorm이
      많아 낮은 opset을 쓰면 TensorRT 퓨전 품질이 떨어진다.

L2 정규화는 그래프 안에 넣는 것이 기본값이다. 노드의 normalize_embeddings가
항상 켜져 있고, 그래프에 포함하면 런타임에서 한 단계가 줄어든다.

사용 예:
    python3 meridian_clip/export_onnx.py
    python3 meridian_clip/export_onnx.py --no-normalize --opset 17
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


# 파리티 검증에 쓸 배치 크기들. 동적 축이 실제로 동작하는지 확인하는 목적이다.
PARITY_BATCHES = (1, 5, 16)

# 이보다 코사인 유사도가 낮으면 export가 잘못된 것으로 본다.
PARITY_COSINE_THRESHOLD = 0.9999

# clip_backend.py 의 POOLING_EPS 와 같아야 한다. weighted mean 의 분모 하한.
# fp16 정규수 하한(약 6.1e-5)보다 커야 한다. 더 작게 두면 TensorRT fp16 에서
# subnormal 이 0으로 flush 되어 빈 마스크 행이 0/0 = NaN 이 된다.
POOLING_EPS = 1e-4


class CLIPVisualEncoder(torch.nn.Module):
    """CLIP image encoder만 감싼 export용 모듈."""

    def __init__(
        self,
        visual: torch.nn.Module,
        normalize: bool,
    ) -> None:
        super().__init__()
        self.visual = visual
        self.normalize = normalize

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """[N,3,H,W] 이미지를 [N,D] embedding으로 바꾼다."""
        features = self.visual(images)
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        return features


class CLIPVisualMaskPooledEncoder(torch.nn.Module):
    """patch token 을 마스크 가중치로 합쳐 임베딩을 내는 export용 모듈.

    clip_backend.py 의 세 단계를 하나의 그래프로 합친 것이다.

        extract_final_visual_tokens  →  [B, 1+P, width]
        mask_weighted_pool           →  [B, width]
        project_object_token         →  [B, D]

    가중치(patch_weights)는 입력으로 받는다. gamma 와 min_patch_occupancy 는
    호출자가 점유율에 미리 적용하므로 그래프에 박히지 않고, 값을 바꿔도
    엔진을 다시 빌드할 필요가 없다.

    두 번째 출력 cls_embeddings 는 같은 forward 의 CLS 임베딩이다. 마스크가
    비어 weighted mean 의 분모가 0이 되는 세그먼트를 노드가 이 값으로 대체하며,
    추가 추론이 필요 없다.

    use_value_tokens 가 True 면 (--part visual_pooled_value) 마지막 블록에서
    attention 혼합 / residual / MLP 를 건너뛴 value 투영을 patch feature 로
    쓴다. ln_post @ proj 는 CLS 만 보고 학습돼서 최종 patch token 을 그대로
    투영하면 텍스트 공간의 엉뚱한 곳에 떨어지는데, CLS 의 attention 출력이
    곧 value 들의 가중합이므로 value 는 같은 영역에 놓인다. 자세한 근거와
    실측값은 clip_backend.py 의 POOLING_MODES 주석에 있다.
    CLS 출력은 두 경우 모두 원본 forward 그대로다.
    """

    def __init__(
        self,
        visual: torch.nn.Module,
        normalize: bool,
        use_value_tokens: bool = False,
    ) -> None:
        super().__init__()
        self.visual = visual
        self.normalize = normalize
        self.use_value_tokens = use_value_tokens

    @staticmethod
    def value_projection(
        block: torch.nn.Module,
        tokens_lnd: torch.Tensor,
    ) -> torch.Tensor:
        """블록의 value 투영만 계산한다 (clip_backend 와 같은 식).

        nn.MultiheadAttention 은 q/k/v 가중치를 in_proj_weight
        [3*width, width] 하나에 쌓으므로 마지막 1/3 이 value 다.
        """
        normalized = block.ln_1(tokens_lnd)

        width = normalized.shape[-1]

        weight = block.attn.in_proj_weight[2 * width:]
        bias = block.attn.in_proj_bias[2 * width:]

        projected = torch.nn.functional.linear(normalized, weight, bias)

        return block.attn.out_proj(projected)

    def forward(
        self,
        images: torch.Tensor,
        patch_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """([N,3,H,W], [N,P]) 를 ([N,D], [N,D]) 로 바꾼다."""
        visual = self.visual

        x = visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # CLIP 원본 VisionTransformer.forward 와 같은 형태로 CLS 를 붙인다.
        # expand 대신 zeros 덧셈을 쓰는 것도 원본 그대로이며, 동적 batch 를
        # ONNX 로 내보낼 때 이 형태가 가장 안전하다.
        cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0],
            1,
            x.shape[-1],
            dtype=x.dtype,
            device=x.device,
        )

        x = torch.cat([cls_token, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)

        if self.use_value_tokens:
            blocks = visual.transformer.resblocks

            for block in blocks[:-1]:
                x = block(x)

            last = blocks[-1]

            # 같은 penultimate 에서 CLS 는 정상 블록, patch 는 value 로 간다.
            normal = last(x).permute(1, 0, 2)
            value = self.value_projection(last, x).permute(1, 0, 2)

            cls_features = normal[:, 0, :]
            patch_tokens = value[:, 1:, :]

        else:
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)

            # 0번이 CLS, 나머지가 patch token 이다.
            cls_features = x[:, 0, :]
            patch_tokens = x[:, 1:, :]

        weights = patch_weights.unsqueeze(-1)

        weighted_sum = (patch_tokens * weights).sum(dim=1)
        weight_sum = weights.sum(dim=1).clamp_min(POOLING_EPS)

        object_token = weighted_sum / weight_sum

        pooled = visual.ln_post(object_token) @ visual.proj
        cls_embeddings = visual.ln_post(cls_features) @ visual.proj

        if self.normalize:
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)
            cls_embeddings = cls_embeddings / cls_embeddings.norm(
                dim=-1,
                keepdim=True,
            )

        return pooled, cls_embeddings


class CLIPTextEncoder(torch.nn.Module):
    """CLIP text encoder만 감싼 export용 모듈.

    encode_text는 token 시퀀스에서 EOT 위치(argmax)를 찾아 그 자리의 feature를
    투영한다. 이 argmax가 그래프에 포함되므로 입력은 clip.tokenize가 만든
    형식(int32 [N, context_length], 뒤쪽은 0 padding)이어야 한다.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        normalize: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.normalize = normalize

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """[N, context_length] token을 [N, D] embedding으로 바꾼다."""
        features = self.model.encode_text(tokens)
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        return features


def load_text_encoder(
    checkpoint: Path,
    normalize: bool,
) -> Tuple[CLIPTextEncoder, int, int]:
    """checkpoint에서 fp32 text encoder와 context_length를 만든다."""
    import clip

    model, _ = clip.load(
        str(checkpoint),
        device="cpu",
        jit=False,
    )
    model.eval()

    context_length = int(model.context_length)
    embedding_dim = int(model.text_projection.shape[1])

    encoder = CLIPTextEncoder(
        model=model,
        normalize=normalize,
    ).eval()

    return encoder, context_length, embedding_dim


def tokenize_batch(count: int, context_length: int) -> torch.Tensor:
    """파리티 검증용 토큰 배치를 만든다. 실제 tokenizer를 그대로 쓴다."""
    import clip

    prompts = [
        f"a photo of object number {index}"
        for index in range(count)
    ]

    return clip.tokenize(prompts, context_length=context_length)


def load_visual_encoder(
    checkpoint: Path,
    normalize: bool,
) -> Tuple[CLIPVisualEncoder, int, int]:
    """checkpoint에서 fp32 visual encoder를 만들고 입력 해상도를 함께 돌려준다."""
    import clip

    # device="cpu"가 내부에서 model.float()을 호출한다. jit=False가 없으면
    # TorchScript 모듈이 나와 torch.onnx.export가 제대로 동작하지 않는다.
    model, _ = clip.load(
        str(checkpoint),
        device="cpu",
        jit=False,
    )
    model.eval()

    resolution = int(model.visual.input_resolution)
    embedding_dim = int(model.visual.output_dim)

    encoder = CLIPVisualEncoder(
        visual=model.visual,
        normalize=normalize,
    ).eval()

    return encoder, resolution, embedding_dim


def load_visual_pooled_encoder(
    checkpoint: Path,
    normalize: bool,
    use_value_tokens: bool = False,
) -> Tuple[CLIPVisualMaskPooledEncoder, int, int, int]:
    """checkpoint에서 fp32 mask-pooled visual encoder를 만든다.

    (encoder, resolution, embedding_dim, patch_count) 를 돌려준다.
    patch 격자는 하드코딩하지 않고 conv1 의 kernel 크기에서 유도하므로
    ViT-B/16 처럼 격자가 다른 모델도 그대로 동작한다.

    use_value_tokens 는 --part visual_pooled_value 용이다.
    """
    import clip

    model, _ = clip.load(
        str(checkpoint),
        device="cpu",
        jit=False,
    )
    model.eval()

    visual = model.visual

    if not hasattr(visual, "class_embedding"):
        raise TypeError(
            "mask pooling 은 Vision Transformer 체크포인트가 필요합니다. "
            "ViT-B/32 같은 ViT 가중치를 쓰세요."
        )

    resolution = int(visual.input_resolution)
    embedding_dim = int(visual.output_dim)
    patch_size = int(visual.conv1.kernel_size[0])

    if resolution % patch_size != 0:
        raise ValueError(
            f"입력 해상도 {resolution} 가 patch 크기 {patch_size} 로 "
            "나누어떨어지지 않습니다."
        )

    grid = resolution // patch_size
    patch_count = grid * grid

    encoder = CLIPVisualMaskPooledEncoder(
        visual=visual,
        normalize=normalize,
        use_value_tokens=use_value_tokens,
    ).eval()

    return encoder, resolution, embedding_dim, patch_count


def export_visual_pooled(
    encoder: CLIPVisualMaskPooledEncoder,
    resolution: int,
    patch_count: int,
    output_path: Path,
    opset: int,
) -> None:
    """Mask-pooled visual encoder를 동적 batch ONNX로 내보낸다."""
    dummy_images = torch.randn(1, 3, resolution, resolution)

    # 전부 0인 가중치는 분모 clamp 경로만 타서 그래프가 한쪽으로 치우친다.
    # 실제 점유율처럼 0..1 값을 준다.
    dummy_weights = torch.rand(1, patch_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            encoder,
            (dummy_images, dummy_weights),
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["images", "patch_weights"],
            output_names=["embeddings", "cls_embeddings"],
            # 프레임마다 segment 수가 달라지므로 batch 축은 반드시 동적이어야 한다.
            dynamic_axes={
                "images": {0: "batch"},
                "patch_weights": {0: "batch"},
                "embeddings": {0: "batch"},
                "cls_embeddings": {0: "batch"},
            },
        )


def check_pooled_parity(
    encoder: CLIPVisualMaskPooledEncoder,
    resolution: int,
    patch_count: int,
    output_path: Path,
) -> float:
    """Torch fp32 출력과 ONNX 출력을 비교한다 (두 출력 모두)."""
    import onnxruntime

    session = onnxruntime.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )

    worst_cosine = 1.0

    for batch in PARITY_BATCHES:
        images = torch.randn(batch, 3, resolution, resolution)
        weights = torch.rand(batch, patch_count)

        # 마스크가 빈 세그먼트(분모 0)도 반드시 한 줄은 섞어서 검증한다.
        weights[0] = 0.0

        with torch.inference_mode():
            expected = [tensor.numpy() for tensor in encoder(images, weights)]

        actual = session.run(
            None,
            {
                "images": images.numpy(),
                "patch_weights": weights.numpy(),
            },
        )

        for name, want, got in zip(
            ("embeddings", "cls_embeddings"),
            expected,
            actual,
        ):
            if got.shape != want.shape:
                raise RuntimeError(
                    f"{name} shape 불일치: "
                    f"torch={want.shape}, onnx={got.shape}"
                )

            cosine = float(
                np.min(
                    np.sum(want * got, axis=-1)
                    / (
                        np.linalg.norm(want, axis=-1)
                        * np.linalg.norm(got, axis=-1)
                    )
                )
            )
            max_diff = float(np.max(np.abs(want - got)))
            worst_cosine = min(worst_cosine, cosine)

            print(
                f"  batch={batch:<3d} {name:<15s} cos={cosine:.8f}  "
                f"max_abs_diff={max_diff:.3e}"
            )

    return worst_cosine


def export(
    encoder: CLIPVisualEncoder,
    resolution: int,
    output_path: Path,
    opset: int,
) -> None:
    """Visual encoder를 동적 batch ONNX로 내보낸다."""
    dummy_input = torch.randn(1, 3, resolution, resolution)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            encoder,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["embeddings"],
            # 프레임마다 segment 수가 달라지므로 batch 축은 반드시 동적이어야 한다.
            dynamic_axes={
                "images": {0: "batch"},
                "embeddings": {0: "batch"},
            },
        )


def export_text(
    encoder: CLIPTextEncoder,
    context_length: int,
    output_path: Path,
    opset: int,
) -> None:
    """Text encoder를 동적 batch ONNX로 내보낸다."""
    dummy_input = tokenize_batch(1, context_length)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            encoder,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["tokens"],
            output_names=["embeddings"],
            # 프롬프트 개수는 실행할 때마다 다르다.
            dynamic_axes={
                "tokens": {0: "batch"},
                "embeddings": {0: "batch"},
            },
        )


def check_text_parity(
    encoder: CLIPTextEncoder,
    context_length: int,
    output_path: Path,
) -> float:
    """Torch fp32 출력과 ONNX 출력을 비교한다 (텍스트)."""
    import onnxruntime

    session = onnxruntime.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )

    worst_cosine = 1.0

    for batch in PARITY_BATCHES:
        tokens = tokenize_batch(batch, context_length)

        with torch.inference_mode():
            expected = encoder(tokens).numpy()

        actual = session.run(
            None,
            {"tokens": tokens.numpy()},
        )[0]

        if actual.shape != expected.shape:
            raise RuntimeError(
                f"shape 불일치: torch={expected.shape}, onnx={actual.shape}"
            )

        cosine = float(
            np.min(
                np.sum(expected * actual, axis=-1)
                / (
                    np.linalg.norm(expected, axis=-1)
                    * np.linalg.norm(actual, axis=-1)
                )
            )
        )
        max_diff = float(np.max(np.abs(expected - actual)))
        worst_cosine = min(worst_cosine, cosine)

        print(
            f"  batch={batch:<3d} cos={cosine:.8f}  max_abs_diff={max_diff:.3e}"
        )

    return worst_cosine


def simplify(output_path: Path) -> bool:
    """onnxsim으로 그래프를 정리한다. 실패하면 원본을 유지한다."""
    try:
        import onnx
        import onnxsim
    except ImportError:
        print("[warn] onnxsim이 없어 단순화를 건너뜁니다.")
        return False

    model = onnx.load(str(output_path))

    try:
        simplified, ok = onnxsim.simplify(model)
    except Exception as error:
        print(f"[warn] 단순화 실패, 원본을 유지합니다: {error}")
        return False

    if not ok:
        print("[warn] 단순화 검증 실패, 원본을 유지합니다.")
        return False

    onnx.save(simplified, str(output_path))
    return True


def describe(output_path: Path) -> Tuple[int, List[str]]:
    """ONNX 모델을 검증하고 연산자 구성을 요약한다."""
    import onnx

    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)

    counts: dict = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: -item[1])
    summary = [f"{name}x{count}" for name, count in ranked[:8]]

    # batch 축이 동적으로 남아 있는지 확인한다.
    batch_dim = (
        model.graph.input[0]
        .type.tensor_type.shape.dim[0]
    )
    if not batch_dim.dim_param:
        raise RuntimeError(
            "batch 축이 고정되었습니다. 동적 배치가 필요합니다."
        )

    return len(model.graph.node), summary


def check_parity(
    encoder: CLIPVisualEncoder,
    resolution: int,
    output_path: Path,
) -> float:
    """Torch fp32 출력과 ONNX 출력을 비교한다."""
    import onnxruntime

    session = onnxruntime.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )

    worst_cosine = 1.0

    for batch in PARITY_BATCHES:
        images = torch.randn(batch, 3, resolution, resolution)

        with torch.inference_mode():
            expected = encoder(images).numpy()

        actual = session.run(
            None,
            {"images": images.numpy()},
        )[0]

        if actual.shape != expected.shape:
            raise RuntimeError(
                f"shape 불일치: torch={expected.shape}, onnx={actual.shape}"
            )

        cosine = float(
            np.min(
                np.sum(expected * actual, axis=-1)
                / (
                    np.linalg.norm(expected, axis=-1)
                    * np.linalg.norm(actual, axis=-1)
                )
            )
        )
        max_diff = float(np.max(np.abs(expected - actual)))
        worst_cosine = min(worst_cosine, cosine)

        print(
            f"  batch={batch:<3d} cos={cosine:.8f}  max_abs_diff={max_diff:.3e}"
        )

    return worst_cosine


def main() -> int:
    """CLI 진입점."""
    package_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="CLIP encoder를 ONNX로 내보낸다 (visual 또는 text).",
    )
    parser.add_argument(
        "--part",
        choices=("visual", "visual_pooled", "visual_pooled_value", "text"),
        default="visual",
        help=(
            "내보낼 인코더. visual=이미지(CLS), "
            "visual_pooled=이미지(mask-weighted patch pooling), "
            "visual_pooled_value=같은 pooling 이되 마지막 블록의 value "
            "투영을 patch feature 로 사용 (MaskCLIP), "
            "text=프롬프트"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=package_root / "models" / "ViT-B-32.pt",
        help="입력 CLIP checkpoint (.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="출력 ONNX 경로 (기본: models/clip_vit_b32_<part>.onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset 버전 (기본 17, LayerNormalization 네이티브 지원)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="L2 정규화를 그래프에 포함하지 않는다",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="onnxsim 단순화를 건너뛴다",
    )

    arguments = parser.parse_args()

    if not arguments.checkpoint.is_file():
        print(
            f"[fail] checkpoint가 없습니다: {arguments.checkpoint}\n"
            "       python3 meridian_clip/download_weights.py 를 먼저 실행하세요.",
            file=sys.stderr,
        )
        return 1

    normalize = not arguments.no_normalize

    output_path = arguments.output or (
        package_root / "models" / f"clip_vit_b32_{arguments.part}.onnx"
    )

    print(f"[load] {arguments.checkpoint}  part={arguments.part}")

    if arguments.part == "text":
        encoder, context_length, embedding_dim = load_text_encoder(
            checkpoint=arguments.checkpoint,
            normalize=normalize,
        )
        parameters = sum(p.numel() for p in encoder.parameters())
        print(
            f"       context_length={context_length} "
            f"embedding_dim={embedding_dim} "
            f"params={parameters / 1e6:.1f}M normalize={normalize}"
        )

        print(f"[expt] opset={arguments.opset} dynamic batch")
        export_text(
            encoder=encoder,
            context_length=context_length,
            output_path=output_path,
            opset=arguments.opset,
        )

    elif arguments.part in ("visual_pooled", "visual_pooled_value"):
        use_value_tokens = arguments.part == "visual_pooled_value"

        (
            encoder,
            resolution,
            embedding_dim,
            patch_count,
        ) = load_visual_pooled_encoder(
            checkpoint=arguments.checkpoint,
            normalize=normalize,
            use_value_tokens=use_value_tokens,
        )
        parameters = sum(p.numel() for p in encoder.parameters())
        grid = int(round(patch_count ** 0.5))
        print(
            f"       resolution={resolution} embedding_dim={embedding_dim} "
            f"patches={grid}x{grid}={patch_count} "
            f"params={parameters / 1e6:.1f}M normalize={normalize} "
            f"patch_feature={'value(MaskCLIP)' if use_value_tokens else 'final'}"
        )

        print(f"[expt] opset={arguments.opset} dynamic batch")
        export_visual_pooled(
            encoder=encoder,
            resolution=resolution,
            patch_count=patch_count,
            output_path=output_path,
            opset=arguments.opset,
        )

    else:
        encoder, resolution, embedding_dim = load_visual_encoder(
            checkpoint=arguments.checkpoint,
            normalize=normalize,
        )
        parameters = sum(p.numel() for p in encoder.parameters())
        print(
            f"       resolution={resolution} embedding_dim={embedding_dim} "
            f"params={parameters / 1e6:.1f}M normalize={normalize}"
        )

        print(f"[expt] opset={arguments.opset} dynamic batch")
        export(
            encoder=encoder,
            resolution=resolution,
            output_path=output_path,
            opset=arguments.opset,
        )

    if not arguments.no_simplify:
        if simplify(output_path):
            print("[simp] onnxsim 적용")

    nodes, summary = describe(output_path)
    print(f"[graph] nodes={nodes}  {' '.join(summary)}")

    print("[parity] torch fp32 vs onnxruntime")
    try:
        if arguments.part == "text":
            worst = check_text_parity(
                encoder=encoder,
                context_length=context_length,
                output_path=output_path,
            )
        elif arguments.part in ("visual_pooled", "visual_pooled_value"):
            worst = check_pooled_parity(
                encoder=encoder,
                resolution=resolution,
                patch_count=patch_count,
                output_path=output_path,
            )
        else:
            worst = check_parity(
                encoder=encoder,
                resolution=resolution,
                output_path=output_path,
            )
    except Exception as error:
        print(f"[fail] 파리티 검증 실패: {error}", file=sys.stderr)
        return 1

    if worst < PARITY_COSINE_THRESHOLD:
        print(
            f"[fail] 코사인 유사도 {worst:.8f} < {PARITY_COSINE_THRESHOLD}",
            file=sys.stderr,
        )
        return 1

    size_megabytes = output_path.stat().st_size / (1 << 20)

    print()
    print(f"onnx  : {output_path}")
    print(f"size  : {size_megabytes:.1f} MiB")
    print(f"cosine: {worst:.8f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
