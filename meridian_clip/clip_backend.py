#!/usr/bin/env python3

"""
Meridian Perception Frontend - CLIP image encoder 백엔드.

clip_inference_node 가 쓰는 "이미지 → [N, D] 임베딩" 기능을 두 가지 구현으로
제공한다. 노드는 어느 쪽을 쓰든 같은 인터페이스만 본다.

    TorchBackend      : models/ViT-B-32.pt 를 clip.load 로 로드 (기존 경로)
    TensorRTBackend   : models/*.engine 를 로드 (visual encoder만, fp16)

두 백엔드 모두 다음을 제공한다.
    .embedding_dim : 임베딩 차원 (ViT-B/32 는 512)
    .device        : "cuda" 또는 "cpu"
    .encode(regions, batch_size, normalize, ...) -> np.ndarray [N, D] float32

전처리(224 정사각형 만들기 + CLIP mean/std 정규화)는 양쪽이 동일해야 한다.
두 백엔드 모두 build_preprocess() 를 쓴다 -- torch 백엔드도 clip.load 가 준
preprocess 를 버리고 이것으로 덮는다. 마스크와 RGB 가 같은 기하를 거쳐야
patch occupancy 가 맞기 때문이다. crop_fit 은 CROP_FITS 주석 참고.

pooling_mode 는 49개 patch token 을 하나로 합치는 방법을 고른다.

    cls                 : CLS token 을 그대로 쓴다 (CLIP 원본 경로)
    mask_weighted_patch : 각 패치의 객체 점유율을 가중치로 768차원 공간에서
                          weighted mean 을 한 뒤 ln_post / visual.proj 를
                          태운다. 텍스트 정렬이 깨져 있다 (아래 실측).
    mask_weighted_value : 같은 가중평균이되 마지막 블록의 value 투영을 patch
                          feature 로 쓴다 (MaskCLIP). **기본값**이다.

세 모드 모두 torch 와 TensorRT 양쪽에서 된다. 뒤의 두 모드는 전용 엔진이
필요하며(export_onnx.py 의 --part visual_pooled / visual_pooled_value), 어느
엔진을 넘길지는 노드가 pooling_mode 를 보고 고른다.
세 경로 모두 계속 유지된다. 어느 모드로 바꾸든 --pooling-mode 한 줄이면 되고
동작은 서로 독립적이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

import torch

import torch.nn.functional as F

from PIL import Image as PILImage


# clip/clip.py 의 _transform 과 동일해야 한다. 어긋나면 임베딩이 달라진다.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

DEFAULT_RESOLUTION = 224

# export_onnx.py 가 지정한 ONNX 입출력 이름. 엔진도 이 이름을 그대로 쓴다.
ENGINE_INPUT_NAME = "images"
ENGINE_TEXT_INPUT_NAME = "tokens"
ENGINE_OUTPUT_NAME = "embeddings"

# --part visual_pooled 로 export 한 엔진에만 있는 입출력.
# 이 두 이름의 존재 여부로 pooling 지원 엔진인지 판별한다.
ENGINE_WEIGHTS_INPUT_NAME = "patch_weights"
ENGINE_CLS_OUTPUT_NAME = "cls_embeddings"

# 49개 patch token 을 하나로 합치는 방법.
#   cls                 : CLS token (CLIP 원본 경로)
#   mask_weighted_patch : 마지막 블록의 최종 patch token 을 점유율로 가중평균
#   mask_weighted_value : 같은 가중평균이되, 마지막 블록에서 attention 혼합 /
#                         residual / MLP 를 건너뛴 value 투영을 patch feature
#                         로 쓴다 (MaskCLIP 방식). 아래 설명 참고.
POOLING_MODES = (
    "cls",
    "mask_weighted_patch",
    "mask_weighted_value",
)

# 기본값이 mask_weighted_value 인 이유 (VOC2012 val, GT 인스턴스 3420개 실측,
# crop_policy=bbox / crop_fit=pad 즉 노드 기본값):
#
#   pooling                top-1     macro     AUC      분리도
#   cls                    83.10%   88.12%   0.9775   0.0662
#   mask_weighted_patch     7.89%   10.55%   0.4633   0.1158
#   mask_weighted_value    87.98%   90.20%   0.9897   0.1392   <- 기본값
#   value + 정렬행렬 W      90.61%   91.25%   0.9887   0.1392
#
# value 가 네 지표 전부에서 최고다. crop_policy 가 masked_bbox 이던 시절에는
# top-1 만 cls 에 졌었는데(81.67% vs 83.39%) bbox 로 바꾸면서 그 맞바꿈이
# 사라졌다 -- 가중평균이 이미 마스크 밖을 배제하는데 픽셀까지 검게 칠하면
# 물체 경계와 맥락만 잃기 때문이다. zero-shot 라벨 정확도가 더 필요하면
# 정렬행렬을 얹어 90.61% 를 쓴다.
#
# 위 AUC 는 열 방향(text->image)이다. 프롬프트를 고정하고 인스턴스 3420개를
# 줄 세운 값이라 맵에 말로 질의를 던지는 방향과 같다. 반대 방향(인스턴스를
# 고정하고 프롬프트 20개를 줄 세우는 image->text):
#   cls 0.9817 / patch 0.4057 / value 0.9816 / value+Wᵀ 0.9883
# patch 의 0.4057 은 무작위(0.5)보다 나쁘다 -- 축 어긋남의 더 직접적인
# 증거다. masked_bbox 시절에는 이 방향에서 value 가 cls 에 졌지만
# (0.9582 vs 0.9670) bbox 에서는 동률이라 방향별 순위 역전이 사라졌다.
# 프롬프트 표현을 바꿨을 때의 강건성까지 포함해 tools/benchmark_language.py
# 에서 잰다 (value 가 언어 변화에 가장 덜 흔들린다).
#
# mask_weighted_patch 가 무너지는 이유 (정보 손실이 아니라 축 어긋남):
# ln_post @ proj 는 학습 내내 CLS token 만 입력으로 받았다. 마지막 블록을
# 통과한 patch token 은 residual(자기 원래 값) + attention 혼합 + MLP 가
# 뒤섞인 것이라 CLS 와 전혀 다른 영역에 있고(실측 cos 0.1206), 그대로
# 투영하면 텍스트 공간의 엉뚱한 곳에 떨어진다. 클래스 정보 자체는 온전히
# 남아 있다 -- 선형 프로브 89.12% 로 cls(89.77%) 와 동급이다. 그런데 정답
# 프롬프트가 나머지 19개보다 겨우 +0.0009 높고(cls 는 +0.0593), 프로브가
# 찾은 클래스 방향은 텍스트 프롬프트 방향과 직교한다(cos -0.0098).
# 그래서 AUC 가 0.4633, 즉 무작위와 구분되지 않는다. 정렬행렬 하나로
# image->text AUC 가 0.4057 -> 0.9843 으로 돌아오는 것이 그 증거다.
# (프로브 수치는 masked_bbox 시절 값이고 재측정하지 않았다.)
#
# 반면 CLS 의 attention 출력은 각 패치 value 벡터의 가중합이다. 즉 value
# 벡터들은 이미 투영이 읽을 수 있는 형태다. 마지막 블록에서 value 투영만
# 꺼내 쓰면 patch feature 가 CLS 와 같은 영역에 놓인다(실측 cos 0.2744,
# 투영 후 0.5804). 이것이 mask_weighted_value 다.
#
# 가중치를 주는 대상만 바뀌고 가중평균 식(mask_weighted_pool)은 그대로다.
# 주의: 기본값에서는 encode() 에 masks 가 반드시 있어야 한다. 마스크 없이
# 이미지만 인코딩하려면 pooling_mode="cls" 를 명시적으로 넘긴다.

DEFAULT_POOLING_MODE = "mask_weighted_value"

# weighted mean 의 분모가 0이 되는 경우(마스크가 비었거나 threshold 가 너무 높음)
# 어떻게 처리할지.
#   cls   : 같은 forward 에서 나온 CLS 임베딩으로 대체 (기본)
#   skip  : 해당 세그먼트를 결과에서 제외 (노드가 stats.keep 을 보고 거른다)
#   error : 예외를 던진다
EMPTY_MASK_FALLBACKS = (
    "cls",
    "skip",
    "error",
)

DEFAULT_EMPTY_MASK_FALLBACK = "cls"

# weighted mean 분모의 하한. 0으로 나누는 것을 막는다.
#
# fp16 에서 표현 가능한 최소 정규수가 약 6.1e-5 라, 1e-6 같은 값을 쓰면
# TensorRT fp16 엔진에서 subnormal 이 0으로 flush 되어 0/0 = NaN 이 된다.
# (마스크가 완전히 빈 세그먼트에서 실제로 재현되었다.) torch 경로와 엔진
# 경로가 같은 값을 내야 하므로 양쪽 모두 이 상수를 쓴다.
# export_onnx.py 의 POOLING_EPS 와 반드시 같아야 한다.
POOLING_EPS = 1e-4


# crop 을 224x224 정사각형으로 만드는 방법.
#   pad        : 긴 변을 224 에 맞추고 남는 곳을 채운다 (기본값).
#                물체가 잘리지 않고, CenterCrop 이 no-op 이 된다.
#   centercrop : CLIP 원본. 짧은 변을 224 에 맞춘 뒤 가운데를 오려낸다.
#   stretch    : 224x224 로 늘린다. 종횡비가 깨진다.
#
# 셋 다 종횡비 r = 긴변/짧은변 에 비례해 무언가를 잃는다. 잃는 대상이 다르다.
#   centercrop -> 물체의 '범위'   (긴 축의 1/r 만 남는다)
#   pad        -> 물체의 '해상도' (물체가 1/r 크기로 렌더링된다)
#   stretch    -> 물체의 '형태'   (종횡비 왜곡, CLIP 학습 분포 밖)
#
# 기본이 pad 인 이유: CLIP 원본 전처리는 사진 한 장을 통째로 분류하는 전제라
# 주제가 가운데 있다고 본다. 우리는 세그먼트 bbox 를 crop 하므로 전제가 다르다.
# bbox 는 물체 모양을 따라가 r 이 제각각이고, 대각선 물체는 정보가 가운데가
# 아니라 양 끝에 있다 (연필의 심과 지우개가 정확히 그 자리다).
# 실측: dogs.jpg 21개 세그먼트 중 12개(57%)가 r>2 라 centercrop 에서 내용의
# 절반 이상을 잃었고, 평균 보존율은 40.5% 였다.
#
# 정답 라벨이 분명한 세그먼트로 잰 비교 (mask_weighted_value, 2등과의 격차):
#   개    r=1.24  centercrop +0.0766 / pad +0.0764   <- 사실상 동일
#   연필  r=1.91  centercrop +0.0147 / pad +0.0596   <- 4배
# 정사각형에 가까우면 손해가 없고 길쭉하면 크게 이득이라 pad 를 기본으로 둔다.
CROP_FITS = (
    "pad",
    "centercrop",
    "stretch",
)

DEFAULT_CROP_FIT = "pad"


class PadToSquare:
    """짧은 변에 여백을 넣어 정사각형으로 만든다.

    RGB 는 mask_fill(기본 검정), 마스크는 0(배경)으로 채워야 하는데 둘 다
    0 이라 fill 값을 나눌 필요가 없다. 마스크와 RGB 가 같은 변환을 거쳐야
    patch occupancy 가 맞으므로 이 클래스도 build_geometry 안에서만 쓴다.
    """

    def __init__(self, fill: int = 0) -> None:
        self.fill = fill

    def __call__(self, image):
        """PIL 이미지를 정사각형으로 패딩한다. 이미 정사각이면 그대로 둔다."""
        width, height = image.size

        if width == height:
            return image

        from PIL import Image as PILImage

        side = max(width, height)

        canvas = PILImage.new(image.mode, (side, side), self.fill)
        canvas.paste(image, ((side - width) // 2, (side - height) // 2))

        return canvas


def build_geometry(
    resolution: int,
    interpolation,
    crop_fit: str = DEFAULT_CROP_FIT,
) -> list:
    """resize/crop 기하만 담은 변환 목록.

    RGB, 마스크, 디버그 저장이 모두 이 함수 하나를 거치게 해서 세 경로의
    기하가 갈라지지 않도록 한다. patch occupancy 는 마스크가 RGB 와 정확히
    같은 픽셀 위치에 놓인다는 전제 위에 서 있으므로, 정의가 한 곳에 있어야
    한다. 출력 크기와 crop 오프셋은 입력 크기만으로 결정되므로 보간 방식만
    달라지는 두 경로는 서로 정렬된다.
    """
    if crop_fit not in CROP_FITS:
        raise ValueError(
            f"crop_fit must be one of {CROP_FITS}, got '{crop_fit}'"
        )

    from torchvision.transforms import CenterCrop, Resize

    if crop_fit == "centercrop":
        # CLIP 원본. Resize 가 짧은 변을 224 로 맞추므로 결과는 224*r x 224
        # 이고, CenterCrop 이 거기서 가운데 224 만 남긴다 (긴 축의 1/r).
        return [
            Resize(resolution, interpolation=interpolation),
            CenterCrop(resolution),
        ]

    if crop_fit == "stretch":
        # 튜플을 주면 종횡비를 무시하고 정확히 그 크기로 만든다.
        return [
            Resize((resolution, resolution), interpolation=interpolation),
        ]

    # pad: 먼저 정사각형으로 만들어 두면 이후 Resize 가 아무것도 자르지 않고,
    # CenterCrop 자체가 필요 없어진다.
    return [
        PadToSquare(),
        Resize((resolution, resolution), interpolation=interpolation),
    ]


def build_preprocess(
    resolution: int = DEFAULT_RESOLUTION,
    crop_fit: str = DEFAULT_CROP_FIT,
):
    """CLIP 의 _transform 과 같은 전처리 파이프라인을 만든다.

    crop_fit 만 CLIP 원본과 다를 수 있다 (CROP_FITS 주석 참고).
    """
    from torchvision.transforms import Compose, Normalize, ToTensor
    from torchvision.transforms import InterpolationMode

    return Compose(
        [
            *build_geometry(resolution, InterpolationMode.BICUBIC, crop_fit),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def build_mask_preprocess(
    resolution: int = DEFAULT_RESOLUTION,
    crop_fit: str = DEFAULT_CROP_FIT,
):
    """마스크에 build_preprocess() 와 똑같은 기하를 적용한다.

    RGB 와 다른 점은 두 가지뿐이다.
        - 보간이 BICUBIC 이 아니라 NEAREST (0/1 이 뭉개지지 않게)
        - CLIP mean/std 정규화를 하지 않음 (0..1 을 그대로 유지)

    입력은 mode "L" PIL 이미지이고 출력은 [1, resolution, resolution] float 이다.
    """
    from torchvision.transforms import Compose, ToTensor
    from torchvision.transforms import InterpolationMode

    return Compose(
        [
            *build_geometry(resolution, InterpolationMode.NEAREST, crop_fit),
            ToTensor(),
        ]
    )


def build_debug_geometry(
    resolution: int = DEFAULT_RESOLUTION,
    crop_fit: str = DEFAULT_CROP_FIT,
):
    """디버그 이미지 저장용 (rgb, mask) PIL→PIL 변환 쌍.

    정규화와 텐서화를 빼고 기하만 적용하므로, 저장된 PNG 세 장(crop, mask,
    occupancy)이 모두 같은 좌표계에 놓여 그대로 겹쳐볼 수 있다.
    """
    from torchvision.transforms import Compose
    from torchvision.transforms import InterpolationMode

    return (
        Compose(build_geometry(resolution, InterpolationMode.BICUBIC, crop_fit)),
        Compose(build_geometry(resolution, InterpolationMode.NEAREST, crop_fit)),
    )


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """행 단위 L2 정규화. 0 벡터는 그대로 둔다."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0

    return matrix / norms


# ====================================================================
# Mask-weighted patch pooling
#
# ViT-B/32 는 224x224 입력을 32x32 패치 49개(7x7)로 자른다. 마지막 블록
# 출력은 [B, 50, 768] 이고 0번이 CLS, 1~49번이 patch token 이다.
# 아래 함수들은 그 patch token 을 객체 점유율로 가중평균해 512차원
# 임베딩을 만드는 경로를 구성한다.
# ====================================================================

# ViT-B/32 기준값. 실제 값은 모델에서 유도하므로 로그/기본값 용도다.
DEFAULT_PATCH_GRID = 7


def visual_geometry(visual) -> Tuple[int, int, int]:
    """CLIP visual tower 에서 (patch grid, patch 개수, token 차원)을 유도한다.

    ViT-B/32 면 (7, 49, 768) 이다. 값을 하드코딩하지 않으므로 ViT-B/16 처럼
    격자가 다른 모델도 그대로 동작한다. ResNet 계열(RN50 등)은 conv1 구조가
    달라 patch token 개념이 없으므로 여기서 걸러진다.
    """
    required = (
        "conv1",
        "class_embedding",
        "positional_embedding",
        "ln_pre",
        "transformer",
        "ln_post",
        "proj",
    )

    missing = [name for name in required if not hasattr(visual, name)]

    if missing:
        raise TypeError(
            "mask-weighted pooling requires a Vision Transformer "
            f"visual tower; this model is missing {missing}. "
            "Use a ViT checkpoint such as ViT-B/32."
        )

    patch_size = int(visual.conv1.kernel_size[0])
    resolution = int(visual.input_resolution)

    if patch_size <= 0 or resolution % patch_size != 0:
        raise ValueError(
            f"input resolution {resolution} is not divisible by patch size "
            f"{patch_size}; cannot form a square patch grid."
        )

    grid = resolution // patch_size

    return grid, grid * grid, int(visual.conv1.out_channels)


def extract_final_visual_tokens(
    model,
    image_batch: torch.Tensor,
) -> torch.Tensor:
    """마지막 Transformer 블록을 통과한 뒤의 전체 토큰을 꺼낸다.

    clip/model.py 의 VisionTransformer.forward 를 ln_post 직전까지 그대로
    재현한 것이다. 원본은 여기서 ln_post(x[:, 0, :]) 로 CLS 만 남기고
    proj 를 곱해 512차원을 만드는데, 우리는 그 앞 단계의 [B, 50, 768] 이
    필요하다. 계산 순서가 동일하므로 CLS 성분은 encode_image() 와 일치한다.

    model 은 jit=False 로 로드된 OpenAI CLIP 모델이고, image_batch 는 CLIP
    정규화가 끝난 [B, 3, 224, 224] 텐서다. 반환값은
    [B, 1 + patch_count, width] 이며 ViT-B/32 면 [B, 50, 768] 이다.
    """
    visual = model.visual

    _, patch_count, width = visual_geometry(visual)

    x = image_batch.to(
        device=visual.conv1.weight.device,
        dtype=visual.conv1.weight.dtype,
    )

    x = visual.conv1(x)
    # [B, width, grid, grid]

    x = x.flatten(start_dim=2).transpose(1, 2)
    # [B, patch_count, width]

    batch_size = x.shape[0]

    cls_token = visual.class_embedding.to(x.dtype)
    cls_token = cls_token.view(1, 1, -1)
    cls_token = cls_token.expand(batch_size, 1, -1)

    x = torch.cat([cls_token, x], dim=1)
    # [B, 1 + patch_count, width]

    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)

    expected = (batch_size, patch_count + 1, width)

    if tuple(x.shape) != expected:
        raise RuntimeError(
            f"unexpected visual token shape: expected {expected}, "
            f"got {tuple(x.shape)}"
        )

    return x


def last_block_value_projection(block, tokens_lnd: torch.Tensor) -> torch.Tensor:
    """블록의 value 투영만 계산한다 -- attention 혼합 없이 (MaskCLIP).

    원래 블록은 x + attention(ln_1(x)) 로 49개 토큰을 서로 섞고, 다시
    x + mlp(ln_2(x)) 를 더한다. 여기서는 셋 다 건너뛰고

        v = out_proj(W_v @ ln_1(x) + b_v)

    만 낸다. nn.MultiheadAttention 은 q/k/v 가중치를 in_proj_weight
    [3*width, width] 하나에 쌓아두므로 마지막 1/3 이 value 다.

    tokens_lnd 는 transformer 내부 표현인 [L, N, width] 이고 반환도 같다.
    """
    normalized = block.ln_1(tokens_lnd)

    width = normalized.shape[-1]

    weight = block.attn.in_proj_weight[2 * width:]
    bias = block.attn.in_proj_bias[2 * width:]

    return block.attn.out_proj(F.linear(normalized, weight, bias))


def extract_value_visual_tokens(
    model,
    image_batch: torch.Tensor,
) -> torch.Tensor:
    """마지막 블록의 patch 자리를 value 투영으로 바꾼 토큰을 낸다.

    extract_final_visual_tokens() 와 같은 형태를 돌려주므로 호출하는 쪽은
    인덱싱을 바꿀 필요가 없다. 다만 내용이 다르다.

        [:, 0, :]  CLS. **원래 forward 그대로**다. empty_mask_fallback="cls"
                   가 기존 경로와 완전히 같은 값을 내야 하므로 건드리지 않는다.
        [:, 1:, :] 마지막 블록의 value 투영 (attention 혼합/residual/MLP 없음)

    마지막 블록 직전까지는 원본과 동일하게 흐르고, 마지막 블록만 두 가지로
    갈라 계산한다 -- CLS 는 정상 블록에서, patch 는 value 투영에서 가져온다.
    """
    visual = model.visual

    _, patch_count, width = visual_geometry(visual)

    x = image_batch.to(
        device=visual.conv1.weight.device,
        dtype=visual.conv1.weight.dtype,
    )

    x = visual.conv1(x)
    x = x.flatten(start_dim=2).transpose(1, 2)

    batch_size = x.shape[0]

    cls_token = visual.class_embedding.to(x.dtype)
    cls_token = cls_token.view(1, 1, -1).expand(batch_size, 1, -1)

    x = torch.cat([cls_token, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)

    blocks = visual.transformer.resblocks

    for block in blocks[:-1]:
        x = block(x)

    last = blocks[-1]

    # CLS 는 정상 경로, patch 는 value 경로. 둘 다 같은 penultimate 에서 나온다.
    normal = last(x).permute(1, 0, 2)
    value = last_block_value_projection(last, x).permute(1, 0, 2)

    tokens = torch.cat([normal[:, :1, :], value[:, 1:, :]], dim=1)

    expected = (batch_size, patch_count + 1, width)

    if tuple(tokens.shape) != expected:
        raise RuntimeError(
            f"unexpected visual token shape: expected {expected}, "
            f"got {tuple(tokens.shape)}"
        )

    return tokens


def compute_patch_occupancy(
    masks: torch.Tensor,
    grid: int = DEFAULT_PATCH_GRID,
) -> torch.Tensor:
    """마스크를 patch 격자로 average pooling 해 점유율을 만든다.

    patch i 의 점유율은 그 패치 안의 객체 픽셀 비율이므로 0..1 이다.
    average pooling 이 곧 비율 계산이다.

    masks 는 [B, 1, H, W] 이고 값은 0/1 이다 (0..255 로 들어와도 받아준다).
    grid 는 patch 격자 한 변의 길이로 ViT-B/32 면 7 이다.
    반환값은 [B, grid * grid] 이며 각 원소는 0..1 이다.
    """
    if masks.ndim != 4:
        raise ValueError(
            f"Expected masks [B,1,H,W], got {tuple(masks.shape)}"
        )

    if masks.shape[1] != 1:
        raise ValueError(
            f"Expected single-channel masks, got {masks.shape[1]} channels"
        )

    values = masks.float()

    # PIL 경유로 0..255 가 들어오는 경우를 받아준다.
    if values.numel() and float(values.max()) > 1.0:
        values = values / 255.0

    values = values.clamp(0.0, 1.0)

    occupancy = F.adaptive_avg_pool2d(
        values,
        output_size=(grid, grid),
    )

    return occupancy.flatten(start_dim=1)


def compute_patch_weights(
    patch_occupancy: torch.Tensor,
    gamma: float = 1.0,
    min_patch_occupancy: float = 0.0,
) -> torch.Tensor:
    """점유율에 threshold 와 gamma 를 적용해 실제 가중치를 만든다.

    torch 경로(mask_weighted_pool)와 TensorRT 경로가 같은 가중치를 쓰도록
    이 계산을 한 곳에 둔다. TensorRT 엔진은 가중치를 입력으로 받으므로
    gamma / min_patch_occupancy 가 그래프에 박히지 않고, 값을 바꿔도
    엔진을 다시 빌드할 필요가 없다.

    w_i = (r_i if r_i >= min_patch_occupancy else 0) ** gamma
    """
    weights = torch.where(
        patch_occupancy >= min_patch_occupancy,
        patch_occupancy,
        torch.zeros_like(patch_occupancy),
    )

    return weights.float().pow(gamma)


def mask_weighted_pool(
    patch_tokens: torch.Tensor,
    patch_occupancy: torch.Tensor,
    gamma: float = 1.0,
    min_patch_occupancy: float = 0.0,
    eps: float = POOLING_EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """점유율을 가중치로 patch token 을 768차원 공간에서 가중평균한다.

    projection 이전 공간에서 평균을 낸다. 패치마다 ln_post/proj 를 먼저
    적용하고 평균내는 것과는 LayerNorm 때문에 결과가 다르며, 이 구현은
    전자를 쓴다.

    patch_tokens 는 CLS 를 제외한 [B, P, width] 이고, patch_occupancy 는
    compute_patch_occupancy() 가 낸 [B, P] 다. gamma 는 가중치 지수로
    w_i = r_i ** gamma 이며 1.0 이면 점유율을 그대로 쓴다.
    min_patch_occupancy 보다 낮은 점유율의 패치는 가중치가 0이 되고,
    eps 는 분모 하한이다.

    반환값은 (object_token [B, width], effective_weights [B, P]) 쌍이다.
    가중치 합이 0인 행의 object_token 은 0 벡터이며, 호출자가 fallback 을
    결정해야 한다.
    """
    if patch_tokens.ndim != 3:
        raise ValueError(
            f"patch_tokens must be [B,P,width], got {tuple(patch_tokens.shape)}"
        )

    if patch_occupancy.ndim != 2:
        raise ValueError(
            f"patch_occupancy must be [B,P], got {tuple(patch_occupancy.shape)}"
        )

    if patch_tokens.shape[:2] != patch_occupancy.shape:
        raise ValueError(
            "Patch token count and occupancy count do not match: "
            f"tokens={tuple(patch_tokens.shape[:2])}, "
            f"occupancy={tuple(patch_occupancy.shape)}"
        )

    weights = compute_patch_weights(
        patch_occupancy=patch_occupancy,
        gamma=gamma,
        min_patch_occupancy=min_patch_occupancy,
    )

    weighted_sum = (
        patch_tokens.float() * weights.unsqueeze(-1)
    ).sum(dim=1)

    weight_sum = weights.sum(dim=1, keepdim=True)

    object_token = weighted_sum / weight_sum.clamp_min(eps)

    return object_token, weights


def project_object_token(
    model,
    object_token: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """768차원 객체 토큰에 CLIP 의 ln_post 와 visual.proj 를 적용한다.

    visual.proj 는 CLIP checkpoint 에 들어 있는 사전학습 이미지 projection
    가중치 [width, output_dim] 이다. transpose 도 pseudoinverse 도 쓰지 않고
    그대로 곱한다.

    dtype 은 ln_post 가 아니라 proj 에 맞춘다. CLIP 은 fp16 모델에서도
    LayerNorm 가중치를 fp32 로 남기고(clip/model.py 의 LayerNorm 서브클래스가
    내부에서만 fp32 로 계산한 뒤 입력 dtype 으로 되돌린다), proj 만 fp16 이다.
    ln_post.weight.dtype 을 따라가면 CUDA 에서 fp32 @ fp16 이 되어 실패하고,
    성공하더라도 원본 forward 와 수치가 달라진다.

    object_token 은 [B, width] 이고, normalize 가 True 면 float32 로 L2
    정규화까지 한다. 반환값은 [B, output_dim] 이며 ViT-B/32 면 [B, 512] 다.
    """
    visual = model.visual

    x = object_token.to(
        device=visual.proj.device,
        dtype=visual.proj.dtype,
    )

    x = visual.ln_post(x)
    x = x @ visual.proj

    if normalize:
        x = F.normalize(x.float(), dim=-1)

    return x


@dataclass
class PoolingStats:
    """가중평균 pooling 한 프레임의 진단값. 노드가 로그/필터링에 쓴다."""

    occupancy: np.ndarray
    """[N, P] 세그먼트별 패치 점유율."""

    weight_sum: np.ndarray
    """[N] 유효 가중치 합. 0이면 빈 마스크."""

    active_patches: np.ndarray
    """[N] 가중치가 0보다 큰 패치 개수."""

    fallback: np.ndarray
    """[N] bool. CLS 임베딩으로 대체된 행."""

    keep: np.ndarray
    """[N] bool. False 면 노드가 결과에서 제외해야 한다."""

    def select(self, rows: np.ndarray) -> "PoolingStats":
        """행을 골라낸 새 PoolingStats 를 만든다.

        노드가 skip 된 세그먼트를 버린 뒤에도 로그 행 순서가 맞도록 쓴다.
        """
        return PoolingStats(
            occupancy=self.occupancy[rows],
            weight_sum=self.weight_sum[rows],
            active_patches=self.active_patches[rows],
            fallback=self.fallback[rows],
            keep=self.keep[rows],
        )


class TorchBackend:
    """clip.load 로 .pt 를 읽어 encode_image 를 돌리는 기존 경로."""

    name = "torch"

    def __init__(
        self,
        checkpoint: str,
        use_cuda: bool = True,
        crop_fit: str = DEFAULT_CROP_FIT,
    ) -> None:
        import clip

        self.device = (
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )

        # clip.load 는 "~" 를 확장하지 않으므로 여기서 풀어준다.
        # 기본 경로가 ~/meridian/... 형태라 이게 없으면 파일을 못 찾는다.
        self.model, _clip_preprocess = clip.load(
            str(Path(checkpoint).expanduser()),
            device=self.device,
        )
        self.model.eval()

        self.embedding_dim = int(self.model.visual.output_dim)

        self.resolution = int(self.model.visual.input_resolution)

        self.crop_fit = crop_fit

        # clip.load 가 준 preprocess 는 쓰지 않고 build_preprocess 로 덮는다.
        # crop_fit 을 반영해야 하고, 무엇보다 마스크와 RGB 가 **같은** 기하를
        # 거쳐야 patch occupancy 가 맞기 때문이다. clip 의 것을 그대로 두면
        # crop_fit="pad" 일 때 마스크만 패딩되어 두 이미지가 어긋난다.
        # crop_fit="centercrop" 이면 clip 의 _transform 과 완전히 같다.
        self.preprocess = build_preprocess(self.resolution, crop_fit)

        # RGB 전처리와 같은 기하를 쓰는 마스크 전처리.
        # 가중평균 모드에서만 쓰이지만 생성 비용이 없어 항상 만든다.
        self.mask_preprocess = build_mask_preprocess(self.resolution, crop_fit)

        # torch 경로는 배치 상한이 없다. 메모리가 허용하는 만큼 넣는다.
        self.max_batch = 0

        # 직전 가중평균 pooling 호출의 진단값. cls 경로에서는 None.
        self.last_pooling_stats: Optional[PoolingStats] = None

        self.description = f"torch checkpoint: {checkpoint}"

    def patch_geometry(self) -> Tuple[int, int, int]:
        """(patch grid, patch 개수, token 차원). ViT 가 아니면 예외."""
        return visual_geometry(self.model.visual)

    def encode(
        self,
        regions: Sequence[PILImage.Image],
        batch_size: int,
        normalize: bool,
        masks: Optional[Sequence[PILImage.Image]] = None,
        pooling_mode: str = DEFAULT_POOLING_MODE,
        gamma: float = 1.0,
        min_patch_occupancy: float = 0.0,
        empty_mask_fallback: str = DEFAULT_EMPTY_MASK_FALLBACK,
    ) -> np.ndarray:
        """이미지 목록을 [N, D] float32 임베딩으로 바꾼다.

        기본값(mask_weighted_value)과 mask_weighted_patch 는 masks 가
        필요하다. pooling_mode 가 "cls" 면 masks 이후의 인자는 무시되며
        기존 경로와 완전히 같은 값을 낸다.
        """
        if pooling_mode not in POOLING_MODES:
            raise ValueError(
                f"pooling_mode must be one of {POOLING_MODES}, "
                f"got '{pooling_mode}'"
            )

        if not regions:
            self.last_pooling_stats = None

            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        if pooling_mode == "cls":
            self.last_pooling_stats = None

            return self.encode_cls(
                regions=regions,
                batch_size=batch_size,
                normalize=normalize,
            )

        return self.encode_mask_weighted(
            regions=regions,
            masks=masks,
            batch_size=batch_size,
            normalize=normalize,
            use_value_tokens=(pooling_mode == "mask_weighted_value"),
            gamma=gamma,
            min_patch_occupancy=min_patch_occupancy,
            empty_mask_fallback=empty_mask_fallback,
        )

    def encode_cls(
        self,
        regions: Sequence[PILImage.Image],
        batch_size: int,
        normalize: bool,
    ) -> np.ndarray:
        """CLIP encode_image 를 그대로 쓰는 기존 경로."""
        tensors = [self.preprocess(region) for region in regions]

        features: List[torch.Tensor] = []

        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(
                tensors[start:start + batch_size]
            ).to(self.device)

            with torch.inference_mode():
                batch_features = self.model.encode_image(batch)

            # CUDA 에서 CLIP 은 fp16 으로 동작하므로 float32 로 맞춘다.
            features.append(batch_features.to(dtype=torch.float32))

        stacked = torch.cat(features, dim=0).cpu().numpy().astype(np.float32)

        if normalize:
            stacked = l2_normalize(stacked)

        return stacked

    def encode_mask_weighted(
        self,
        regions: Sequence[PILImage.Image],
        masks: Optional[Sequence[PILImage.Image]],
        batch_size: int,
        normalize: bool,
        gamma: float,
        min_patch_occupancy: float,
        empty_mask_fallback: str,
        use_value_tokens: bool = False,
    ) -> np.ndarray:
        """패치 점유율로 가중평균한 객체 임베딩을 만든다.

        use_value_tokens 가 True 면 마지막 블록의 value 투영을 patch
        feature 로 쓴다 (mask_weighted_value). 가중평균 식은 그대로이고
        가중치를 곱할 대상만 바뀐다.

        진단값은 self.last_pooling_stats 에 남긴다. 노드가 로그와
        empty_mask_fallback="skip" 처리에 쓴다.
        """
        if empty_mask_fallback not in EMPTY_MASK_FALLBACKS:
            raise ValueError(
                f"empty_mask_fallback must be one of {EMPTY_MASK_FALLBACKS}, "
                f"got '{empty_mask_fallback}'"
            )

        if masks is None:
            raise ValueError(
                "mask-weighted pooling requires masks; "
                "build_regions() must return one mask per region. "
                "Use pooling_mode='cls' to encode without masks."
            )

        if len(masks) != len(regions):
            raise ValueError(
                f"region/mask count mismatch: regions={len(regions)}, "
                f"masks={len(masks)}"
            )

        grid, _, _ = self.patch_geometry()

        image_tensors = [self.preprocess(region) for region in regions]
        mask_tensors = [self.mask_preprocess(mask) for mask in masks]

        features: List[torch.Tensor] = []
        occupancy_chunks: List[np.ndarray] = []
        weight_chunks: List[np.ndarray] = []
        empty_chunks: List[np.ndarray] = []

        for start in range(0, len(image_tensors), batch_size):
            stop = start + batch_size

            image_batch = torch.stack(
                image_tensors[start:stop]
            ).to(self.device)

            mask_batch = torch.stack(
                mask_tensors[start:stop]
            ).to(self.device)

            with torch.inference_mode():
                extract = (
                    extract_value_visual_tokens
                    if use_value_tokens
                    else extract_final_visual_tokens
                )

                tokens = extract(self.model, image_batch)

                # 0번은 CLS. 나머지가 patch token 이다. 두 extractor 모두
                # 0번은 원본 forward 의 CLS 라 fallback 값이 달라지지 않는다.
                patch_tokens = tokens[:, 1:, :]

                occupancy = compute_patch_occupancy(mask_batch, grid)

                object_token, weights = mask_weighted_pool(
                    patch_tokens=patch_tokens,
                    patch_occupancy=occupancy,
                    gamma=gamma,
                    min_patch_occupancy=min_patch_occupancy,
                )

                embeddings = project_object_token(
                    self.model,
                    object_token,
                    normalize=False,
                )

                empty = weights.sum(dim=1) <= 0.0

                # 같은 forward 의 CLS 토큰이라 두 번 돌 필요가 없다.
                if empty_mask_fallback == "cls" and bool(empty.any()):
                    cls_embeddings = project_object_token(
                        self.model,
                        tokens[:, 0, :],
                        normalize=False,
                    )

                    embeddings = torch.where(
                        empty.unsqueeze(-1),
                        cls_embeddings,
                        embeddings,
                    )

            features.append(embeddings.to(dtype=torch.float32))
            occupancy_chunks.append(occupancy.float().cpu().numpy())
            weight_chunks.append(weights.float().cpu().numpy())
            empty_chunks.append(empty.cpu().numpy())

        stacked = torch.cat(features, dim=0).cpu().numpy().astype(np.float32)

        occupancy_all = np.concatenate(occupancy_chunks, axis=0)
        weights_all = np.concatenate(weight_chunks, axis=0)
        empty_all = np.concatenate(empty_chunks, axis=0)

        if empty_mask_fallback == "error" and bool(empty_all.any()):
            rows = np.nonzero(empty_all)[0].tolist()

            raise RuntimeError(
                f"patch occupancy is all zero for region rows {rows}. "
                "Lower --min-patch-occupancy or use "
                "--empty-mask-fallback cls."
            )

        keep = (
            ~empty_all
            if empty_mask_fallback == "skip"
            else np.ones_like(empty_all)
        )

        self.last_pooling_stats = PoolingStats(
            occupancy=occupancy_all,
            weight_sum=weights_all.sum(axis=1),
            active_patches=(weights_all > 0.0).sum(axis=1),
            fallback=empty_all if empty_mask_fallback == "cls" else (
                np.zeros_like(empty_all)
            ),
            keep=keep,
        )

        if not np.isfinite(stacked).all():
            raise RuntimeError("Embedding contains NaN or Inf")

        if normalize:
            stacked = l2_normalize(stacked)

        return stacked


class TensorRTBackend:
    """직렬화된 TensorRT 엔진으로 visual encoder 만 돌리는 경로.

    엔진에는 text encoder 가 없다. 노드는 encode_image 만 쓰므로 문제가 없다.
    export_onnx.py 가 L2 정규화를 그래프 안에 넣었으므로 출력은 이미 단위벡터다.

    엔진은 두 종류이며 입출력 이름으로 자동 판별한다.

        --part visual         : images -> embeddings
                                cls pooling 만 가능
        --part visual_pooled  : images, patch_weights -> embeddings, cls_embeddings
                                mask_weighted_patch + cls 둘 다 가능

    visual_pooled 엔진은 weighted mean 과 ln_post/proj 까지 그래프 안에 넣었다.
    노드는 마스크에서 점유율만 계산해 patch_weights 로 넘기면 되고, 그래서
    gamma / min_patch_occupancy 를 바꿔도 엔진을 다시 빌드할 필요가 없다.
    cls_embeddings 는 같은 forward 의 CLS 임베딩이라 empty_mask_fallback="cls"
    가 추가 추론 없이 동작한다.
    """

    name = "tensorrt"

    def __init__(
        self,
        engine_path: str,
        resolution: int = DEFAULT_RESOLUTION,
        crop_fit: str = DEFAULT_CROP_FIT,
    ) -> None:
        import tensorrt as trt

        if not torch.cuda.is_available():
            raise RuntimeError(
                "TensorRT 백엔드는 CUDA 가 필요합니다. "
                "--backend torch 를 쓰거나 GPU 를 확인하세요."
            )

        path = Path(engine_path).expanduser()

        if not path.is_file():
            raise FileNotFoundError(
                f"TensorRT 엔진이 없습니다: {path}\n"
                "python3 meridian_clip/build_engine.py 로 먼저 빌드하세요."
            )

        self.device = "cuda"
        self.trt = trt

        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)

        self.engine = runtime.deserialize_cuda_engine(path.read_bytes())

        if self.engine is None:
            raise RuntimeError(
                f"엔진 역직렬화 실패: {path}\n"
                "엔진은 빌드한 GPU/TensorRT 버전에서만 유효합니다. "
                "장비나 드라이버가 바뀌었다면 build_engine.py 로 재빌드하세요."
            )

        self.context = self.engine.create_execution_context()

        # 프로파일에서 배치 상한을, 출력 shape 에서 임베딩 차원을 읽어온다.
        _, _, max_shape = self.engine.get_tensor_profile_shape(
            ENGINE_INPUT_NAME,
            0,
        )

        self.max_batch = int(max_shape[0])
        self.resolution = int(max_shape[2])

        output_shape = self.engine.get_tensor_shape(ENGINE_OUTPUT_NAME)
        self.embedding_dim = int(output_shape[1])

        # 프레임마다 재할당하지 않도록 상한 크기로 한 번만 잡는다.
        self.input_buffer = torch.empty(
            (self.max_batch, 3, self.resolution, self.resolution),
            dtype=torch.float32,
            device="cuda",
        )
        self.output_buffer = torch.empty(
            (self.max_batch, self.embedding_dim),
            dtype=torch.float32,
            device="cuda",
        )

        self.crop_fit = crop_fit
        self.preprocess = build_preprocess(self.resolution, crop_fit)

        # 직전 가중평균 pooling 호출의 진단값. cls 경로에서는 None.
        self.last_pooling_stats: Optional[PoolingStats] = None

        # ------------------------------------------------------------
        # pooling 지원 여부는 엔진의 입출력 이름으로 판별한다.
        # ------------------------------------------------------------

        tensor_names = {
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
        }

        self.supports_pooling = (
            ENGINE_WEIGHTS_INPUT_NAME in tensor_names
            and ENGINE_CLS_OUTPUT_NAME in tensor_names
        )

        self.patch_count = 0
        self.patch_grid = 0

        if self.supports_pooling:
            _, _, weights_max = self.engine.get_tensor_profile_shape(
                ENGINE_WEIGHTS_INPUT_NAME,
                0,
            )

            self.patch_count = int(weights_max[1])

            grid = int(round(self.patch_count ** 0.5))

            if grid * grid != self.patch_count:
                raise RuntimeError(
                    f"patch 개수 {self.patch_count} 가 정사각 격자가 아닙니다. "
                    "엔진을 다시 export 하세요."
                )

            self.patch_grid = grid

            self.weights_buffer = torch.empty(
                (self.max_batch, self.patch_count),
                dtype=torch.float32,
                device="cuda",
            )
            self.cls_buffer = torch.empty(
                (self.max_batch, self.embedding_dim),
                dtype=torch.float32,
                device="cuda",
            )

            # RGB 전처리와 같은 기하를 쓰는 마스크 전처리.
            self.mask_preprocess = build_mask_preprocess(
                self.resolution,
                crop_fit,
            )

        self.description = (
            f"tensorrt engine: {path} "
            f"(max_batch={self.max_batch}, resolution={self.resolution}, "
            + (
                f"pooling={self.patch_grid}x{self.patch_grid})"
                if self.supports_pooling
                else "pooling=cls only)"
            )
        )

    def patch_geometry(self) -> Tuple[int, int, int]:
        """(patch grid, patch 개수, token 차원).

        엔진은 pooling 을 그래프 안에서 끝내므로 중간 token 차원(768)을
        밖으로 내보내지 않는다. 세 번째 값이 0인 것은 "알 수 없음"이라는 뜻이며
        노드는 로그에만 쓴다.
        """
        if not self.supports_pooling:
            raise TypeError(
                "이 TensorRT 엔진은 patch pooling 을 지원하지 않습니다. "
                "export_onnx.py --part visual_pooled 로 만든 엔진이 필요합니다."
            )

        return self.patch_grid, self.patch_count, 0

    def infer(
        self,
        image_batch: torch.Tensor,
        weight_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """엔진을 한 번 실행한다.

        반환은 (embeddings, cls_embeddings) 이며 pooling 미지원 엔진에서는
        두 번째가 None 이다. weight_batch 가 None 이면 가중치를 1로 채운다.
        (cls 만 필요한 경우로, pooled 출력은 쓰지 않는다.)
        """
        count = image_batch.shape[0]

        self.input_buffer[:count].copy_(image_batch)

        self.context.set_input_shape(
            ENGINE_INPUT_NAME,
            (count, 3, self.resolution, self.resolution),
        )
        self.context.set_tensor_address(
            ENGINE_INPUT_NAME,
            self.input_buffer.data_ptr(),
        )
        self.context.set_tensor_address(
            ENGINE_OUTPUT_NAME,
            self.output_buffer.data_ptr(),
        )

        # pooling 엔진은 입출력이 두 개씩이라 주소를 모두 지정해야 한다.
        if self.supports_pooling:
            if weight_batch is None:
                self.weights_buffer[:count].fill_(1.0)
            else:
                self.weights_buffer[:count].copy_(weight_batch)

            self.context.set_input_shape(
                ENGINE_WEIGHTS_INPUT_NAME,
                (count, self.patch_count),
            )
            self.context.set_tensor_address(
                ENGINE_WEIGHTS_INPUT_NAME,
                self.weights_buffer.data_ptr(),
            )
            self.context.set_tensor_address(
                ENGINE_CLS_OUTPUT_NAME,
                self.cls_buffer.data_ptr(),
            )

        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        pooled = self.output_buffer[:count].cpu().numpy().astype(np.float32)

        cls = (
            self.cls_buffer[:count].cpu().numpy().astype(np.float32)
            if self.supports_pooling
            else None
        )

        return pooled, cls

    def encode(
        self,
        regions: Sequence[PILImage.Image],
        batch_size: int,
        normalize: bool,
        masks: Optional[Sequence[PILImage.Image]] = None,
        pooling_mode: str = DEFAULT_POOLING_MODE,
        gamma: float = 1.0,
        min_patch_occupancy: float = 0.0,
        empty_mask_fallback: str = DEFAULT_EMPTY_MASK_FALLBACK,
    ) -> np.ndarray:
        if pooling_mode not in POOLING_MODES:
            raise ValueError(
                f"pooling_mode must be one of {POOLING_MODES}, "
                f"got '{pooling_mode}'"
            )

        # 조용히 CLS 로 되돌아가면 안 된다. 어떤 pooling 을 쓴 임베딩인지
        # 알 수 없게 되고, embedding_model_id 도 거짓이 된다.
        #
        # mask_weighted_patch 와 mask_weighted_value 는 입출력 이름이 같아서
        # 엔진만 보고는 구분할 수 없다. 어느 엔진을 넘길지는 노드가
        # pooling_mode 에 따라 고른다 (--pooled-engine-path / --value-engine-path).
        if pooling_mode != "cls" and not self.supports_pooling:
            part = (
                "visual_pooled_value"
                if pooling_mode == "mask_weighted_value"
                else "visual_pooled"
            )

            raise RuntimeError(
                f"{pooling_mode} pooling is not supported by this TensorRT "
                "visual engine (it has no patch_weights input).\n"
                "Build a pooling engine:\n"
                f"  python3 meridian_clip/export_onnx.py --part {part}\n"
                f"  python3 meridian_clip/build_engine.py --part {part}\n"
                "then point the matching --*-engine-path at it, "
                "or fall back to --backend torch."
            )

        if not regions:
            self.last_pooling_stats = None

            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        if pooling_mode == "cls":
            self.last_pooling_stats = None

            return self.encode_cls(
                regions=regions,
                batch_size=batch_size,
                normalize=normalize,
            )

        return self.encode_mask_weighted(
            regions=regions,
            masks=masks,
            batch_size=batch_size,
            normalize=normalize,
            gamma=gamma,
            min_patch_occupancy=min_patch_occupancy,
            empty_mask_fallback=empty_mask_fallback,
        )

    def encode_cls(
        self,
        regions: Sequence[PILImage.Image],
        batch_size: int,
        normalize: bool,
    ) -> np.ndarray:
        """CLS 임베딩만 뽑는 경로.

        pooling 엔진에서는 CLS 가 cls_embeddings 라는 별도 출력이므로 그것을
        읽는다. 두 엔진 모두 같은 ViT forward 를 재현하므로 값은 동일하다.
        """
        # 엔진 프로파일 상한을 넘는 배치는 넣을 수 없다.
        effective = min(batch_size, self.max_batch)

        tensors = [self.preprocess(region) for region in regions]

        chunks: List[np.ndarray] = []

        for start in range(0, len(tensors), effective):
            batch = torch.stack(
                tensors[start:start + effective]
            ).to("cuda", non_blocking=True)

            pooled, cls = self.infer(batch)

            chunks.append(cls if cls is not None else pooled)

        stacked = np.concatenate(chunks, axis=0)

        # 엔진 그래프에 이미 정규화가 들어 있어 멱등이지만,
        # --no-normalize 로 export 한 엔진도 받을 수 있게 그대로 적용한다.
        if normalize:
            stacked = l2_normalize(stacked)

        return stacked

    def encode_mask_weighted(
        self,
        regions: Sequence[PILImage.Image],
        masks: Optional[Sequence[PILImage.Image]],
        batch_size: int,
        normalize: bool,
        gamma: float,
        min_patch_occupancy: float,
        empty_mask_fallback: str,
    ) -> np.ndarray:
        """점유율 가중평균을 엔진 안에서 끝내는 경로.

        TorchBackend.encode_mask_weighted 와 같은 결과를 내야 한다. 점유율과
        가중치 계산은 같은 함수(compute_patch_occupancy / compute_patch_weights)를
        쓰고, weighted mean 이후는 엔진 그래프가 담당한다.
        """
        if empty_mask_fallback not in EMPTY_MASK_FALLBACKS:
            raise ValueError(
                f"empty_mask_fallback must be one of {EMPTY_MASK_FALLBACKS}, "
                f"got '{empty_mask_fallback}'"
            )

        if masks is None:
            raise ValueError(
                "mask-weighted pooling requires masks; "
                "build_regions() must return one mask per region. "
                "Use pooling_mode='cls' to encode without masks."
            )

        if len(masks) != len(regions):
            raise ValueError(
                f"region/mask count mismatch: regions={len(regions)}, "
                f"masks={len(masks)}"
            )

        effective = min(batch_size, self.max_batch)

        image_tensors = [self.preprocess(region) for region in regions]
        mask_tensors = [self.mask_preprocess(mask) for mask in masks]

        features: List[np.ndarray] = []
        occupancy_chunks: List[np.ndarray] = []
        weight_chunks: List[np.ndarray] = []
        empty_chunks: List[np.ndarray] = []

        for start in range(0, len(image_tensors), effective):
            stop = start + effective

            image_batch = torch.stack(
                image_tensors[start:stop]
            ).to("cuda", non_blocking=True)

            mask_batch = torch.stack(
                mask_tensors[start:stop]
            ).to("cuda", non_blocking=True)

            occupancy = compute_patch_occupancy(mask_batch, self.patch_grid)

            weights = compute_patch_weights(
                patch_occupancy=occupancy,
                gamma=gamma,
                min_patch_occupancy=min_patch_occupancy,
            )

            embeddings, cls_embeddings = self.infer(image_batch, weights)

            empty = (weights.sum(dim=1) <= 0.0).cpu().numpy()

            # 같은 forward 의 CLS 출력이라 두 번 돌 필요가 없다.
            if empty_mask_fallback == "cls" and bool(empty.any()):
                embeddings = np.where(
                    empty[:, None],
                    cls_embeddings,
                    embeddings,
                )

            features.append(embeddings)
            occupancy_chunks.append(occupancy.float().cpu().numpy())
            weight_chunks.append(weights.float().cpu().numpy())
            empty_chunks.append(empty)

        stacked = np.concatenate(features, axis=0).astype(np.float32)

        occupancy_all = np.concatenate(occupancy_chunks, axis=0)
        weights_all = np.concatenate(weight_chunks, axis=0)
        empty_all = np.concatenate(empty_chunks, axis=0)

        if empty_mask_fallback == "error" and bool(empty_all.any()):
            rows = np.nonzero(empty_all)[0].tolist()

            raise RuntimeError(
                f"patch occupancy is all zero for region rows {rows}. "
                "Lower --min-patch-occupancy or use "
                "--empty-mask-fallback cls."
            )

        keep = (
            ~empty_all
            if empty_mask_fallback == "skip"
            else np.ones_like(empty_all)
        )

        self.last_pooling_stats = PoolingStats(
            occupancy=occupancy_all,
            weight_sum=weights_all.sum(axis=1),
            active_patches=(weights_all > 0.0).sum(axis=1),
            fallback=empty_all if empty_mask_fallback == "cls" else (
                np.zeros_like(empty_all)
            ),
            keep=keep,
        )

        if not np.isfinite(stacked).all():
            raise RuntimeError("Embedding contains NaN or Inf")

        if normalize:
            stacked = l2_normalize(stacked)

        return stacked


# CLIP 논문의 logit scale (exp(learned) = 100). 확률이 필요하면 곱해서 softmax.
CLIP_LOGIT_SCALE = 100.0


def tokenize(prompts: Sequence[str], context_length: int) -> np.ndarray:
    """clip 패키지의 BPE tokenizer로 int32 [M, context_length] 를 만든다.

    tokenizer 는 가중치가 아니라 vocab 파일이라 .pt / .engine 어느 쪽과도
    무관하게 clip 패키지에서 온다.
    """
    import clip

    return (
        clip.tokenize(list(prompts), context_length=context_length)
        .numpy()
        .astype(np.int32)
    )


class TextEncoder:
    """프롬프트를 CLIP 텍스트 임베딩으로 바꾸는 인코더 (.pt 경로).

    프롬프트는 바뀌지 않으므로 생성자에서 한 번만 인코딩하고, 이후에는
    행렬곱만 한다. 즉 프레임마다 도는 경로에는 텍스트 인코딩이 없다.
    """

    name = "torch"

    def __init__(
        self,
        checkpoint: str,
        prompts: Sequence[str],
        use_cuda: bool = True,
    ) -> None:
        import clip

        if not prompts:
            raise ValueError("prompts must not be empty")

        self.prompts = list(prompts)
        self.device = (
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )

        # clip.load 는 "~" 를 확장하지 않는다 (TorchBackend 와 같은 이유).
        model, _ = clip.load(
            str(Path(checkpoint).expanduser()),
            device=self.device,
        )
        model.eval()

        tokens = clip.tokenize(self.prompts).to(self.device)

        with torch.inference_mode():
            text = model.encode_text(tokens).to(dtype=torch.float32)

        text = text / text.norm(dim=-1, keepdim=True)

        # [M, D]. 행렬만 남기고 모델은 버려 메모리를 돌려준다.
        self.matrix = text.cpu().numpy().astype(np.float32)

        self.embedding_dim = int(self.matrix.shape[1])
        self.logit_scale = float(model.logit_scale.exp().item())

        self.description = f"torch text checkpoint: {checkpoint}"

        del model

        if self.device == "cuda":
            torch.cuda.empty_cache()

    def similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """이미지 임베딩 [N, D] 와 프롬프트 [M, D] 의 코사인 유사도 [N, M]."""
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"embedding dim mismatch: image={embeddings.shape[1]}, "
                f"text={self.embedding_dim}"
            )

        return embeddings @ self.matrix.T


class TensorRTTextEncoder(TextEncoder):
    """텍스트 엔진(.engine)으로 프롬프트를 인코딩한다.

    .pt 를 전혀 읽지 않으므로 런타임에 checkpoint 파일이 필요 없다.
    tokenizer(BPE vocab)만 clip 패키지에서 가져온다.

    엔진은 export_onnx.py --part text 로 만든 그래프라 L2 정규화가 포함되어
    있고, 출력은 visual 엔진과 같은 512차원 공간에 놓인다.
    """

    name = "tensorrt"

    def __init__(
        self,
        engine_path: str,
        prompts: Sequence[str],
    ) -> None:
        import tensorrt as trt

        if not prompts:
            raise ValueError("prompts must not be empty")

        if not torch.cuda.is_available():
            raise RuntimeError(
                "TensorRT 텍스트 인코더는 CUDA 가 필요합니다. "
                "--text-backend torch 를 쓰거나 GPU 를 확인하세요."
            )

        path = Path(engine_path).expanduser()

        if not path.is_file():
            raise FileNotFoundError(
                f"텍스트 엔진이 없습니다: {path}\n"
                "python3 meridian_clip/export_onnx.py --part text 와\n"
                "python3 meridian_clip/build_engine.py --part text 를 실행하세요."
            )

        self.prompts = list(prompts)
        self.device = "cuda"

        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)

        engine = runtime.deserialize_cuda_engine(path.read_bytes())

        if engine is None:
            raise RuntimeError(
                f"텍스트 엔진 역직렬화 실패: {path}\n"
                "엔진은 빌드한 GPU/TensorRT 버전에서만 유효합니다."
            )

        context = engine.create_execution_context()

        _, _, max_shape = engine.get_tensor_profile_shape(
            ENGINE_TEXT_INPUT_NAME,
            0,
        )

        max_batch = int(max_shape[0])
        context_length = int(max_shape[1])

        if len(self.prompts) > max_batch:
            raise ValueError(
                f"프롬프트가 {len(self.prompts)}개인데 엔진 상한은 {max_batch}개입니다. "
                "build_engine.py --part text --max-batch N 으로 다시 빌드하세요."
            )

        self.embedding_dim = int(
            engine.get_tensor_shape(ENGINE_OUTPUT_NAME)[1]
        )
        self.logit_scale = CLIP_LOGIT_SCALE

        tokens = tokenize(self.prompts, context_length)

        device_input = torch.from_numpy(tokens).cuda()
        device_output = torch.empty(
            (len(self.prompts), self.embedding_dim),
            dtype=torch.float32,
            device="cuda",
        )

        context.set_input_shape(
            ENGINE_TEXT_INPUT_NAME,
            (len(self.prompts), context_length),
        )
        context.set_tensor_address(
            ENGINE_TEXT_INPUT_NAME,
            device_input.data_ptr(),
        )
        context.set_tensor_address(
            ENGINE_OUTPUT_NAME,
            device_output.data_ptr(),
        )

        stream = torch.cuda.current_stream()
        context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        # 프롬프트는 시작할 때 한 번만 인코딩하므로 엔진은 여기서 버린다.
        self.matrix = l2_normalize(
            device_output.cpu().numpy().astype(np.float32)
        )

        self.description = (
            f"tensorrt text engine: {path} "
            f"(max_prompts={max_batch}, context_length={context_length})"
        )

        del context
        del engine

        torch.cuda.empty_cache()


def create_text_encoder(
    backend: str,
    checkpoint: str,
    engine_path: str,
    prompts: Sequence[str],
    use_cuda: bool = True,
):
    """이름으로 텍스트 인코더를 만든다."""
    if backend == "torch":
        return TextEncoder(
            checkpoint=checkpoint,
            prompts=prompts,
            use_cuda=use_cuda,
        )

    if backend == "tensorrt":
        return TensorRTTextEncoder(
            engine_path=engine_path,
            prompts=prompts,
        )

    raise ValueError(
        f"text backend must be 'torch' or 'tensorrt', got '{backend}'"
    )


def create_backend(
    backend: str,
    checkpoint: str,
    engine_path: str,
    use_cuda: bool = True,
    crop_fit: str = DEFAULT_CROP_FIT,
):
    """이름으로 백엔드를 만든다."""
    if backend == "torch":
        return TorchBackend(
            checkpoint=checkpoint,
            use_cuda=use_cuda,
            crop_fit=crop_fit,
        )

    if backend == "tensorrt":
        return TensorRTBackend(
            engine_path=engine_path,
            crop_fit=crop_fit,
        )

    raise ValueError(
        f"backend must be 'torch' or 'tensorrt', got '{backend}'"
    )
