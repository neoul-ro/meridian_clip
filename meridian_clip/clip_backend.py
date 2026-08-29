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

import ctypes
import os
import re

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

import torch

import torch.nn.functional as F

from PIL import Image as PILImage


def limit_blas_threads(count: int = 1) -> List[Tuple[str, int]]:
    """numpy 뒤의 OpenBLAS 를 count 스레드로 제한한다.

    이 노드의 numpy 연산은 텍스트 유사도(32x512 @ 512x18) 정도가 전부라
    멀티스레드 BLAS 가 얻을 게 없다. 반대로 손해는 크다. 프레임 사이
    ~50ms 동안 BLAS 워커가 잠들었다가 **깨는 비용**이 행렬곱 본체보다
    100배 크게 나온다 (실측: 1회차 4.4ms, 2회차 0.05ms). 게다가 그 스레드는
    전처리 스레드풀(PREPROCESS_WORKERS)과 코어를 두고 싸운다.

    OPENBLAS_NUM_THREADS 환경변수는 numpy import 전에 있어야 먹으므로
    노드 안에서는 늦다. 대신 이미 로드된 libopenblas 를 직접 부른다.

    **로드된 것이 하나가 아니다.** numpy 는 자기 wheel 안의 64비트 정수
    빌드(심볼에 64_ 접미사)를 쓰고, scipy 는 시스템 libopenblasp 를 쓴다.
    둘 다 매핑되므로 전부 설정해야 한다. 하나만 건드리면 numpy 쪽이 그대로
    남아 아무 효과가 없다.

    반환값은 (라이브러리 이름, 설정 전 스레드 수) 목록이고, 하나도 못 찾으면
    빈 목록이다 (조용히 넘어간다 -- 성능 최적화일 뿐 동작에는 영향이 없다).
    """
    try:
        maps = Path(f"/proc/{os.getpid()}/maps").read_text()
    except OSError:
        return []

    changed = []

    for path in sorted(set(re.findall(r"\S+libopenblas\S*\.so\S*", maps))):
        try:
            library = ctypes.CDLL(path)
        except OSError:
            continue

        for suffix in ("64_", ""):
            setter = getattr(
                library, f"openblas_set_num_threads{suffix}", None)

            if setter is None:
                continue

            getter = getattr(
                library, f"openblas_get_num_threads{suffix}", None)

            previous = int(getter()) if getter is not None else -1

            setter(int(count))

            changed.append((os.path.basename(path), previous))

    return changed


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


# 224 를 만드는 구현. 기하와 의미는 같고 어디서 계산하는지만 다르다.
#
#   pil        crop 을 PIL 이미지로 만들고 PadToSquare -> Resize(BICUBIC) 를
#              스레드풀로 돌린다. **기본값이고 배포 경로다.**
#   interp_aa  프레임을 한 번 GPU 로 올리고 crop -> zero-pad ->
#              F.interpolate(bicubic, antialias) 로 224 를 만든다. crop 복사 /
#              PIL 생성 / 장별 H2D 가 전부 사라진다. 드리프트가 가장 작다.
#   roi_align  정사각 ROI 를 torchvision roi_align 배치 **한 번**으로 뽑고
#              bbox 밖 띠를 마스킹한다. 보간이 bilinear 고정이라 드리프트가
#              크지만, 배치 커널이라 pre 가 N 에 거의 무관하다.
#
# 셋 다 crop_fit="pad" 전용이다 (다른 fit 은 기하가 다르다).
#
# **경로 이름은 이미지 샘플링만 가리킨다.** 점유율은 세 경로 모두
# occupancy_from_frame() 으로 정확히 계산한다 (pil 경로와 비트 동일).
# 점유율까지 roi_align 으로 근사하면 얇은 세그먼트에서 빈 마스크 fallback 이
# 뒤집혀 임베딩이 종류째로 달라진다 -- occupancy_from_frame 주석 참고.
#
# 실측 (Jetson Orin, mask_weighted_value, uHumans2 N 평균 18.6):
#                     pre     enc    post   합계    FPS
#   pil (배포)       16.70   12.09   0.36   29.15   34.3
#   roi_align         8.56   11.84   0.34   20.74   48.2
#   interp_aa         9.21   11.86   0.33   21.40   46.7
# enc/post 는 경로와 무관하다 (같은 엔진, 같은 발행).
#
# pre 의 N 의존성이 갈린다 (N 6 -> 48):
#   pil        12.75 -> 28.98   (+127%)
#   roi_align   7.00 -> 10.66   (+52%)   배치 커널이라 N 을 흡수한다
#   interp_aa   6.52 -> 13.31   (+104%)  인스턴스별 커널 런치 0.17ms/개
# 교차점이 N 약 8~9 다. N 이 작은 장면에서는 interp_aa 가 더 빠르다.
#
# 임베딩 드리프트 (pil 대비 코사인):
#                    합성 노이즈        VOC2012 val         uHumans2 office
#   interp_aa      1.0000 / 0.9999   0.9995 / 0.9736    0.9979 / 0.9535
#   roi_align      0.9736 / 0.9425   0.9925 / 0.8768    0.9824 / 0.8490
#                  (평균 / 최소)
# zero-shot top-1 은 VOC 에서 pil 92.65% 대비 interp_aa 92.55% (-0.10pp),
# roi_align 92.48% (-0.17pp) 로 셋이 오차범위 안이다. 갈리는 것은 거리다.
#
# **어느 것을 고를까.** 임베딩 거리를 직접 쓰는 소비자(인스턴스 매칭,
# 재식별)가 있으면 interp_aa 다 -- 꼬리가 0.954 대 0.849 로 훨씬 짧다.
# 라벨링만 쓰고 N 이 크게 변하는 장면이면 roi_align 이 pre 를 0.65ms 아끼고
# N 에 덜 흔들린다.
#
# **기본값을 바꾸지 않는 이유.** 코사인 0.998 은 충분히 높지만 1.0 이 아니다.
# 이미 저장된 임베딩과 새로 만든 임베딩을 섞어 거리 비교를 하면 같은 물체가
# 0.95 대로 벌어질 수 있다 (인스턴스 매칭 / 재식별이 정확히 그것을 한다).
# 전환하려면 저장된 임베딩 전체를 재생성해야 하므로, 그 시점은 호출자가
# 명시적으로 골라야 한다.
PREPROCESS_PATHS = (
    "pil",
    "interp_aa",
    "roi_align",
)

DEFAULT_PREPROCESS_PATH = "pil"

# pil 이 아닌 경로들. 이 목록에 있으면 prepare_from_frame() 을 쓴다.
GPU_PREPROCESS_PATHS = tuple(
    name for name in PREPROCESS_PATHS if name != "pil"
)


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


# 전처리 스레드 수. PIL 은 resize 동안 GIL 을 놓으므로 파이썬 스레드로도
# 실제 병렬이 된다. 640x480 / crop 크기 혼합 실측에서 8개가 최적이었고
# 12개는 torch 내부 스레딩과 경합해 오히려 느렸다 (2.05배 -> 1.96배).
PREPROCESS_WORKERS = 8
# 청크 단위로 PIL 전처리와 엔진을 겹치려는 옵션. 기본은 끔.
#
# CPU 에 여유가 있을 때만 이득이다. 실측(Jetson Orin, 12코어, N=32):
#   workers=1  27.1 -> 32.1 FPS   (이득)
#   workers=2  38.1 -> 41.7 FPS   (이득)
#   workers=4  43.8 -> 43.0 FPS   (손해)
#   workers=8  44.8 -> 43.1 FPS   (손해)
# 워커가 CPU 를 채우고 나면 엔진 대기 중에 돌릴 여유가 없고, 노드의
# 2-stage pipeline 에서 Stage1/Stage2 균형만 깨진다 (cls 기준 Pre 17.6->5.3,
# Enc 17.9->29.2, queue 1.9->26.0, 처리량 50.1->45.2 FPS).
# GPU 가 훨씬 빠른 장비에서는 켜 볼 만하다.
DEFAULT_ASYNC_PREPROCESS = False


@dataclass
class PreparedBatch:
    """CPU 전처리를 끝낸 한 프레임. 여기서부터는 GPU 작업만 남는다.

    파이프라인 1단계(prepare)의 출력이자 2단계(run)의 입력이다. 두 스레드가
    나눠 들고 있어도 되도록 프레임에 딸린 것만 담는다 -- 백엔드의 엔진
    입출력 버퍼는 여기에 없고, 그것은 run() 을 부르는 쪽만 만진다.

        images     [N, 3, R, R] float32, device. pending 이 있으면 None 이다.
        occupancy  [N, grid*grid] float32, device (cls 모드에서는 None)
        pending    아직 안 끝난 224 기하 작업의 future 목록 (region 당 하나)

    **pending 이 핵심이다.** 이걸 미리 다 돌려 버리면 전처리와 엔진이 그대로
    직렬로 더해진다. future 로 들고 있다가 run() 이 청크마다 필요한 만큼만
    꺼내면, 엔진이 도는 동안(stream.synchronize 가 GIL 을 놓는다) 워커가
    다음 청크의 PIL 작업을 진행한다.
    """

    images: Optional[torch.Tensor]
    occupancy: Optional[torch.Tensor]
    pooling_mode: str
    pending: Optional[List] = None
    count: int = 0

    def __len__(self) -> int:
        # pending 상태에서는 아직 images 가 없으므로 count 로 센다.
        return self.count if self.images is None else int(self.images.shape[0])


def nearest_index_map(side: int, resolution: int) -> np.ndarray:
    """PIL 의 NEAREST resize 가 쓰는 출력->입력 행/열 인덱스를 그대로 만든다.

    Pillow 는 NEAREST 를 ImagingScaleAffine 으로 처리하고, 좌표를 곱셈이
    아니라 **누적 덧셈**으로 만든다 (Geometry.c).

        xo = a0 * 0.5;  for i: idx[i] = (int)xo;  xo += a0;   a0 = side/resolution

    (i + 0.5) * a0 을 직접 곱하면 부동소수 반올림이 달라져 side 가 64 의
    배수일 때 어긋난다. cumsum 은 같은 순서로 누적하므로 일치한다.
    1..256 과 300/400/480/512/640/1000 전부에서 실측 확인했다.
    """
    return nearest_index_maps(np.array([side]), resolution)[0]


def nearest_index_maps(sides: np.ndarray, resolution: int) -> np.ndarray:
    """nearest_index_map 을 여러 변에 대해 한 번에. 반환 [N, resolution].

    cumsum 은 축을 따라 순차 누적이라 1-D 로 하나씩 돌린 것과 부동소수
    반올림까지 같다 (실측 비트 동일). 세그먼트마다 파이썬으로 돌면
    32개에 0.23ms 인데 일괄로는 0.06ms 다.
    """
    steps = np.repeat(
        (np.asarray(sides, dtype=np.float64) / resolution)[:, None],
        resolution,
        axis=1,
    )
    steps[:, 0] *= 0.5

    return np.cumsum(steps, axis=1).astype(np.int64)


def stack_patch_counts(
    lengths: np.ndarray,
    offsets: np.ndarray,
    sides: np.ndarray,
    grid: int,
    resolution: int,
    patch_of_row: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """카운트 행렬 묶음과 이미지별 색인을 만든다.

    C[k, p, y] = "조합 k 에서 패치 p 에 속한 출력 행 중 원본 행 y 를
    집는 개수". 반환은 (C [K, grid, max(lengths)], inverse [N]) 이고
    이미지 i 의 행렬은 C[inverse[i]] 다.

    두 가지를 같이 한다.

    1. **중복 제거.** (length, offset, side) 가 같으면 카운트도 같다.
       합성 프레임처럼 세그먼트 크기가 다 같으면 조합이 하나로 줄어든다.
    2. **남은 조합은 bincount 한 번으로.** 조합마다 numpy 를 돌리면 크기가
       제각각인 실제 장면에서 32장에 1.60ms 가 드는데, 일괄로는 0.30ms 다.
       작은 배열에 numpy 를 수백 번 부르는 오버헤드가 계산보다 컸다.

    길이가 다르므로 최대 길이로 패딩한다. 카운트 행렬은 [grid, L] 로 작아
    (32장 수백 KB) 패딩 낭비가 없다. 마스크 본체는 패딩하지 않고 호출부가
    이미지별로 곱한다 -- 큰 세그먼트 하나 때문에 전체가 부풀지 않게.
    """
    # 세 값을 int64 하나로 접어 1-D unique 를 쓴다. np.unique(axis=0) 은
    # 행을 void 로 보고 정렬해서, 조합이 하나뿐인 경우(합성 프레임처럼
    # 크기가 다 같을 때) 오히려 손해다. 마스크 변은 프레임 크기를 넘지
    # 못하므로 8192 진법이면 충분하다.
    RADIX = 8192

    key = (lengths * RADIX + offsets) * RADIX + sides

    _, first, inverse = np.unique(
        key, return_index=True, return_inverse=True)

    count = len(first)
    limit = int(lengths.max())

    index = nearest_index_maps(sides[first], resolution)

    source = index - offsets[first][:, None]

    inside = (source >= 0) & (source < lengths[first][:, None])

    base = (
        np.arange(count)[:, None] * grid + patch_of_row[None, :]
    ) * limit

    tables = np.bincount(
        (base + source)[inside],
        minlength=count * grid * limit,
    ).reshape(count, grid, limit).astype(np.float32)

    return tables, inverse


def patch_occupancy_from_masks(
    masks: Sequence[np.ndarray],
    grid: int,
    resolution: int = DEFAULT_RESOLUTION,
) -> np.ndarray:
    """crop 크기 마스크에서 [N, grid*grid] 점유율을 바로 만든다.

    기존 경로는 마스크를 PIL 로 바꾸고 -> 정사각 패딩 -> 224 NEAREST resize
    -> GPU 업로드 -> float 변환 -> 32x32 average pooling 을 거쳤다. 결국
    필요한 것은 7x7 숫자 49개뿐인데 그걸 얻으려고 224x224 를 만들었다.

    NEAREST resize 는 보간이 아니라 **인덱스 gather** 이므로 접을 수 있다.
    출력 픽셀 (i, j) 는 원본 (idx[i], idx[j]) 를 그대로 집어 오고, 패치
    점유율은 32x32 블록의 평균이다. 따라서

        occupancy[p, q] = (1/1024) * sum_{i in p, j in q} mask[idx[i], idx[j]]
                        = (1/1024) * (Cy[p] . mask . Cx[q])

    이고, Cy[p, y] 는 "패치 p 에 속한 출력 행 중 원본 행 y 를 집는 개수"다.
    합이 정수(<= 1024)라 float32 에서 반올림이 없고, 기존 경로와 **비트
    단위로 같은 값**이 나온다 (랜덤 마스크 120회 전수 확인).

    crop_fit="pad" 전용이다. 다른 crop_fit 은 기하가 달라 호출부가 기존
    경로로 돌아가야 한다.
    """
    block = resolution // grid

    count = len(masks)

    occupancy = np.empty((count, grid * grid), dtype=np.float32)

    if not count:
        return occupancy

    patch_of_row = np.arange(resolution) // block

    heights = np.array([mask.shape[0] for mask in masks], dtype=np.int64)
    widths = np.array([mask.shape[1] for mask in masks], dtype=np.int64)

    sides = np.maximum(heights, widths)

    # PadToSquare 가 짧은 변을 가운데로 밀어 넣는다.
    rows, row_index = stack_patch_counts(
        heights, (sides - heights) // 2, sides,
        grid, resolution, patch_of_row)

    columns, column_index = stack_patch_counts(
        widths, (sides - widths) // 2, sides,
        grid, resolution, patch_of_row)

    for position, mask in enumerate(masks):
        height, width = mask.shape[:2]

        occupancy[position] = (
            rows[row_index[position], :, :height]
            @ mask.astype(np.float32)
            @ columns[column_index[position], :, :width].T
        ).ravel() / (block * block)

    return occupancy


class BatchPreprocessor:
    """224 전처리를 배치로 처리한다. build_preprocess() 와 같은 값을 낸다.

    나누는 지점이 중요하다.

        PadToSquare -> Resize -> convert("RGB")      <- PIL, 스레드풀로 병렬
        ------------------------------ uint8 ndarray
        ToTensor -> Normalize                        <- 배치 한 번에 device 에서

    **resize 를 PIL 에 남기는 것이 수치 보존의 조건이다.** torch 의 bicubic 은
    커널 상수가 PIL 과 달라(a=-0.75 vs -0.5) antialias=True 를 줘도 값이 맞지
    않는다 (실측 평균 절대오차 0.146 -- 임베딩이 바뀌는 크기다). 반면 이
    구성은 build_preprocess() 대비 최대 오차가 4.8e-7 로 float32 반올림
    수준이라 정확도 재측정이 필요 없다.

    부수 효과로 H2D 전송량이 4배 줄어든다. 224x224x3 을 float32(602KB)가
    아니라 uint8(150KB)로 올리고 나눗셈과 정규화를 device 에서 하기 때문이다.

    build_preprocess() / build_mask_preprocess() 는 그대로 남겨 둔다.
    tools/clip_selftest.py 와 tools/benchmark_imagenet.py 가 한 장짜리
    전처리로 직접 부르고 있다.
    """

    def __init__(
        self,
        resolution: int = DEFAULT_RESOLUTION,
        crop_fit: str = DEFAULT_CROP_FIT,
        device: str = "cuda",
        workers: int = PREPROCESS_WORKERS,
    ) -> None:
        from torchvision.transforms import Compose, InterpolationMode

        self.resolution = int(resolution)
        self.crop_fit = crop_fit
        self.device = device
        self.workers = max(1, int(workers))

        # 기하는 단일 경로(build_geometry)에서 그대로 가져온다. RGB/마스크/
        # 디버그가 갈라지면 patch occupancy 가 어긋나므로 여기서도 예외를
        # 두지 않는다.
        self.image_geometry = Compose(
            build_geometry(
                self.resolution, InterpolationMode.BICUBIC, crop_fit
            )
        )
        self.mask_geometry = Compose(
            build_geometry(
                self.resolution, InterpolationMode.NEAREST, crop_fit
            )
        )

        self.pool = ThreadPoolExecutor(
            max_workers=self.workers,
            thread_name_prefix="clip-preprocess",
        )

        # upload() 가 쓰는 pinned 스테이징 버퍼. 항목 shape 별로 하나씩
        # 잡아 두고 재사용한다. 값은 (버퍼, 마지막 H2D 완료 이벤트).
        self.staging: dict = {}

        self.mean = torch.tensor(
            CLIP_MEAN, device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)

        self.std = torch.tensor(
            CLIP_STD, device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)

        # 정규화를 한 커널로 접기 위한 미리 계산값 (normalize_from_frame 주석).
        #   ((x/255) - mean) / std  ==  x * (1/(255*std)) - mean/std
        self.norm_scale = 1.0 / (255.0 * self.std)
        self.norm_shift = self.mean / self.std

    def image_array(self, image: PILImage.Image) -> np.ndarray:
        """PIL 기하만 적용해 [H, W, 3] uint8 로 돌려준다."""
        resized = self.image_geometry(image)

        # convert("RGB") 는 이미 RGB 여도 224x224x3 을 통째로 복사한다
        # (32장 2.6ms). 노드가 넘기는 region 은 항상 RGB 라 매번 낭비였다.
        if resized.mode != "RGB":
            resized = resized.convert("RGB")

        return np.asarray(resized, dtype=np.uint8)

    def mask_array(self, mask: PILImage.Image) -> np.ndarray:
        """마스크에 같은 기하를 적용해 [H, W] uint8(0/255)로 돌려준다."""
        return np.asarray(self.mask_geometry(mask), dtype=np.uint8)

    def stack(
        self,
        items: Sequence[PILImage.Image],
        transform: Callable[[PILImage.Image], np.ndarray],
        parallel: bool = True,
    ) -> torch.Tensor:
        """기하를 돌리고 uint8 배치 하나로 device 에 올린다.

        parallel=False 는 스레드풀을 건너뛴다. 항목당 비용이 디스패치 비용
        보다 작을 때는 풀이 손해다 (masks() 주석 참고).
        """
        # 한 장이면 풀에 넘기는 비용이 이득보다 크다.
        if not parallel or len(items) == 1:
            arrays = [transform(item) for item in items]
        else:
            arrays = list(self.pool.map(transform, items))

        return self.upload(arrays)

    def submit(
        self,
        items: Sequence[PILImage.Image],
        transform: Callable[[PILImage.Image], np.ndarray],
    ) -> List:
        """기하 작업을 풀에 던지고 **기다리지 않고** future 를 돌려준다.

        pool.map 은 전부 끝날 때까지 블록해서 엔진과 겹칠 수 없다.
        submit 은 즉시 반환하므로 호출자가 필요한 시점에 result() 한다.
        """
        return [self.pool.submit(transform, item) for item in items]

    def collect(self, futures: Sequence) -> torch.Tensor:
        """future 들의 결과를 모아 device 로 올린다."""
        return self.upload([future.result() for future in futures])

    def upload(self, arrays: Sequence[np.ndarray]) -> torch.Tensor:
        """uint8 배치를 device 로 올린다.

        np.stack 이 만든 pageable 메모리에서 복사하면 드라이버가 내부
        임시 pinned 버퍼를 한 번 더 거친다. 미리 잡아 둔 pinned 버퍼에
        직접 써 넣으면 그 왕복이 사라진다 (32장 224x224x3 실측:
        1.28ms -> 0.83ms, 그중 장별 복사 0.58ms 는 양쪽 공통).

        버퍼를 재사용하므로 **다음 프레임이 덮어쓰기 전에 이전 H2D 가
        끝나야 한다.** 복사 직후 기록한 이벤트를 다음 호출 맨 앞에서
        기다린다. 이게 없으면 non_blocking 복사가 진행 중인 메모리를
        CPU 가 갈아엎어 조용히 값이 깨진다.
        """
        if self.device != "cuda" or not arrays:
            return torch.from_numpy(np.stack(arrays)).to(self.device)

        shape = arrays[0].shape
        staging, event = self.staging.get(shape, (None, None))

        if event is not None:
            event.synchronize()

        if staging is None or staging.shape[0] < len(arrays):
            staging = torch.empty(
                (max(len(arrays), 32), *shape),
                dtype=torch.uint8,
            ).pin_memory()

            event = torch.cuda.Event()

            self.staging[shape] = (staging, event)

        view = staging.numpy()

        for index, item in enumerate(arrays):
            view[index] = item

        batch = staging[:len(arrays)].to(self.device, non_blocking=True)

        event.record()

        return batch

    def normalize_images(self, batch: torch.Tensor) -> torch.Tensor:
        # ToTensor: HWC uint8 -> CHW float [0,1], 이어서 Normalize
        batch = batch.permute(0, 3, 1, 2).float().div_(255.0)

        return (batch - self.mean) / self.std

    def images(self, regions: Sequence[PILImage.Image]) -> torch.Tensor:
        """[N, 3, R, R] float32. build_preprocess() 를 N번 돈 것과 같다."""
        return self.normalize_images(self.stack(regions, self.image_array))

    def occupancy(self, masks: Sequence, grid: int) -> torch.Tensor:
        """[N, grid*grid] 패치 점유율. 가능하면 224 마스크를 만들지 않는다.

        crop_fit="pad" 이고 마스크가 crop 크기 ndarray 로 들어오면
        patch_occupancy_from_masks() 로 바로 계산한다 (32개 기준
        5.40ms -> 1.67ms). 그 밖의 경우 -- 다른 crop_fit, PIL 이미지,
        224 가 grid 로 안 나누어떨어지는 격자 -- 는 기존 경로를 쓴다.
        두 경로는 같은 값을 낸다.
        """
        usable = (
            self.crop_fit == "pad"
            and self.resolution % grid == 0
            and len(masks) > 0
            and all(isinstance(mask, np.ndarray) for mask in masks)
        )

        if usable:
            return torch.from_numpy(
                patch_occupancy_from_masks(masks, grid, self.resolution)
            ).to(self.device)

        images = [
            mask
            if isinstance(mask, PILImage.Image)
            else PILImage.fromarray(
                (np.ascontiguousarray(mask) != 0).view(np.uint8) * 255
            )
            for mask in masks
        ]

        return compute_patch_occupancy(self.masks(images), grid)

    def masks(self, masks: Sequence[PILImage.Image]) -> torch.Tensor:
        """[N, 1, R, R] float32 [0,1]. 마스크는 정규화하지 않는다.

        **스레드풀을 쓰지 않는다.** RGB 는 BICUBIC 3채널이라 장당 0.6ms 가
        들어 풀이 2.4배 이득이지만, 마스크는 NEAREST 1채널이라 장당 0.05ms
        뿐이라 디스패치 비용이 더 크다. N=1..64 전 구간에서 직렬이 빨랐다
        (N=32: 3.08ms 직렬 vs 4.88ms 풀8, N=64: 6.19 vs 8.55).
        """
        batch = self.stack(masks, self.mask_array, parallel=False)

        return batch.unsqueeze(1).float().div_(255.0)

    # ----------------------------------------------------------------
    # 전-GPU 경로 (preprocess_path="interp_aa")
    #
    # 위의 메서드들은 crop 이 이미 PIL 이미지로 만들어져 있는 것을 전제한다.
    # 아래 세 개는 그 앞단을 건너뛴다 -- 프레임을 **한 번** 올리고 crop 복사 /
    # PIL 생성 / 장별 H2D 없이 GPU 에서 224 를 만든다.
    # ----------------------------------------------------------------

    def frame_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        """프레임 한 장을 [1, 3, H, W] float 로 올린다. 프레임당 한 번이다.

        입력이 나중에 GPU 텐서로 들어오면 이 단계 자체가 사라진다
        (실측 0.5~0.9ms).
        """
        frame = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1)

        return frame.unsqueeze(0).float()

    @staticmethod
    def square_offsets(boxes: Sequence[Tuple[int, int, int, int]]):
        """PadToSquare 와 같은 **정수** offset 기하.

        여백을 (side - 변) // 2 로 밀어 넣는 것까지 PIL 쪽과 같아야 한다.
        반 픽셀만 어긋나도 보간 좌표계가 밀린다.
        """
        bounds = np.asarray(boxes, dtype=np.int64)

        x0, y0, x1, y1 = bounds.T

        widths = x1 - x0
        heights = y1 - y0

        sides = np.maximum(widths, heights)

        return (
            x0, y0, x1, y1,
            widths, heights, sides,
            (sides - widths) // 2, (sides - heights) // 2,
        )

    def images_from_frame(
        self,
        frame: torch.Tensor,
        boxes: Sequence[Tuple[int, int, int, int]],
    ) -> torch.Tensor:
        """[N, 3, R, R] float32. images() 와 같은 기하를 GPU 에서 재현한다.

        crop -> 정사각 zero-pad -> bicubic resize 순서와 커널 종류를 그대로
        유지한다. F.interpolate(antialias=True) 는 PIL 처럼 다운스케일에서
        커널을 배율만큼 넓히므로, plain grid_sample 과 달리 다운스케일에서도
        PIL 과 가깝다.

        **인스턴스별 루프가 남는다.** antialias 커널 폭이 세그먼트 배율마다
        다르므로 하나의 배치 호출로 묶을 수 없다. 대신 진짜 zero-pad 를 쓰기
        때문에 bbox 밖을 지우는 band 마스크가 아예 필요 없다 (roi_align 경로는
        그 마스크에 2.1ms 를 쓴다).

        실측 드리프트 (current 대비 코사인): 합성 노이즈 프레임 1.0000,
        VOC2012 val 0.9995 / p5 0.9982, uHumans2 office 0.9979 / 최소 0.9535.
        """
        resolution = self.resolution

        plane = frame[0]

        chunks = []

        for x0, y0, x1, y1 in boxes:
            crop = plane[:, y0:y1, x0:x1]

            width, height = x1 - x0, y1 - y0
            side = max(width, height)

            left = (side - width) // 2
            top = (side - height) // 2

            square = F.pad(
                crop,
                (left, side - width - left, top, side - height - top),
            )

            chunks.append(
                F.interpolate(
                    square.unsqueeze(0),
                    size=(resolution, resolution),
                    mode="bicubic",
                    antialias=True,
                    align_corners=False,
                )
            )

        images = torch.cat(chunks, dim=0)

        # PIL 은 uint8 을 거치므로 클램프가 들어간다. bicubic 은 오버슈트한다.
        return self.normalize_from_frame(images)

    def normalize_from_frame(self, images: torch.Tensor) -> torch.Tensor:
        """[0,255] float 배치를 CLIP 정규화한다. 패스 수를 줄인 형태다.

        원래 식은 텐서를 네 번 훑는다 -- clamp / div(255) / sub(mean) / div(std).
        같은 값을 두 번으로 줄인다.

            clamp  ->  addcmul(-mean/std, x, 1/(255*std))

        **재결합이므로 비트 동일은 아니다.** (x/255 - mean)/std 와
        x*(1/(255*std)) - mean/std 는 부동소수 마지막 비트에서 갈릴 수 있다
        (상대오차 ~1e-7). 임베딩 코사인에 미치는 영향은 1e-7 수준이라 측정
        한계 아래이고, verify_node_paths.py 가 실제로 확인한다. 이 경로는
        이미 보간 때문에 pil 과 비트 동일이 아니므로 (점유율과 달리) 비트
        동일이 기준이 아니다.
        """
        return torch.addcmul(
            -self.norm_shift, images.clamp_(0.0, 255.0), self.norm_scale)

    def images_from_frame_roi(
        self,
        frame: torch.Tensor,
        boxes: Sequence[Tuple[int, int, int, int]],
    ) -> torch.Tensor:
        """[N, 3, R, R] float32. 정사각 ROI 를 roi_align 배치 한 번으로 뽑는다.

        images_from_frame() 과 같은 화각을 보지만 세 가지가 다르다.

        1. **보간이 bilinear 고정이다.** roi_align 에 bicubic 옵션이 없다.
           그래서 pil 대비 드리프트가 크다 (PREPROCESS_PATHS 주석의 표).
        2. **파이썬 루프가 없다.** 커널 한 번에 N 개를 뽑으므로 pre 가 N 에
           거의 무관해진다 (N 6 -> 48 에서 +52%).
        3. **band 마스킹이 필요하다.** zero-pad 를 실제로 만들지 않고 프레임에서
           직접 샘플링하므로 bbox 밖을 사후에 지워야 한다. 그 대가로 경계
           몇 픽셀에는 bbox 밖 이웃이 스며든다 (interp_aa 는 진짜 zero-pad 라
           이 문제가 없다).

        ROI 는 **정수** offset 으로 잡는다 -- PadToSquare 가 여백을
        (side - 변) // 2 로 밀어 넣으므로, bbox 중심을 그대로 쓰면 최대 반
        픽셀이 어긋난다.
        """
        from torchvision.ops import roi_align

        resolution = self.resolution

        (
            x0, y0, _x1, _y1,
            widths, heights, sides, left, top,
        ) = self.square_offsets(boxes)

        origin_x = (x0 - left).astype(np.float32)
        origin_y = (y0 - top).astype(np.float32)

        sides_f = sides.astype(np.float32)

        rois = np.stack([
            np.zeros(len(boxes), dtype=np.float32),
            origin_x, origin_y,
            origin_x + sides_f, origin_y + sides_f,
        ], axis=1)

        device = self.device

        crops = roi_align(
            frame,
            torch.from_numpy(rois).to(device),
            (resolution, resolution),
            sampling_ratio=2,
            aligned=True,
        )

        # bbox 안에 해당하는 출력 픽셀만 남긴다. 판정 기준은 점유율 쪽과 같은
        # 최근접 정수 픽셀이다 (occupancy_from_frame 의 idx 와 같은 규약).
        steps = torch.from_numpy(
            sides_f / resolution).to(device).view(-1, 1)

        centers = (
            torch.arange(resolution, device=device).float().view(1, -1) + 0.5
        ) * steps          # 정사각 좌표 (픽셀 중심 단위 + 0.5)

        low_x = torch.from_numpy(left.astype(np.float32)).to(device).view(-1, 1)
        low_y = torch.from_numpy(top.astype(np.float32)).to(device).view(-1, 1)

        high_x = low_x + torch.from_numpy(
            widths.astype(np.float32)).to(device).view(-1, 1)
        high_y = low_y + torch.from_numpy(
            heights.astype(np.float32)).to(device).view(-1, 1)

        keep_x = (centers >= low_x) & (centers < high_x)
        keep_y = (centers >= low_y) & (centers < high_y)

        keep = (keep_y.unsqueeze(2) & keep_x.unsqueeze(1)).unsqueeze(1)

        return self.normalize_from_frame(crops * keep)

    def occupancy_from_frame(
        self,
        labels: np.ndarray,
        segment_ids: Sequence[int],
        boxes: Sequence[Tuple[int, int, int, int]],
        grid: int,
    ) -> torch.Tensor:
        """[N, grid*grid] 점유율. patch_occupancy_from_masks 와 **비트 동일**.

        crop 크기 마스크를 만들지 않고 프레임 라벨맵에서 직접 뽑는다.

        NEAREST resize 는 보간이 아니라 인덱스 gather 이므로
        (patch_occupancy_from_masks 주석 참고) 정사각 좌표를 프레임 좌표로
        옮기기만 하면 된다.

            정사각 행 s  <->  프레임 행 (y0 - top) + s
            출력   행 i  <->  정사각 행 idx[i]        (nearest_index_maps)

        정사각 ROI 의 offset 이 정수라서 이 대응이 정확히 성립한다. bbox 밖
        띠는 PadToSquare 가 0 으로 채우므로, 프레임 좌표가 bbox 안인지만 보면
        된다. 합이 정수(<= block^2)이고 분모가 2의 거듭제곱이라 float32 에서
        반올림이 없다.

        **근사하지 않는 것이 중요하다.** roi_align 7x7 로 점유율을 근사하면
        bin 하나가 프레임 수십 픽셀을 덮어 샘플 간격이 십수 픽셀이 된다. 벽
        모서리 / 문틀 같은 몇 픽셀짜리 세그먼트가 그 사이로 빠져 점유율이
        전부 0 이 되고, 그러면 빈 마스크 fallback 이 걸려 **같은 세그먼트가
        가중평균이 아니라 CLS 임베딩**으로 나온다 (실측: 얇은 세그먼트의
        58~98%에서 판정이 뒤집히고, 그때 코사인이 0.15~0.58 로 떨어진다).
        """
        block = self.resolution // grid

        (
            x0, y0, x1, y1,
            _widths, _heights, sides, left, top,
        ) = self.square_offsets(boxes)

        # PIL NEAREST 의 출력 -> 정사각 인덱스. x/y 가 같은 변을 쓴다.
        index = nearest_index_maps(sides, self.resolution)

        rows = (y0 - top)[:, None] + index
        columns = (x0 - left)[:, None] + index

        inside_rows = (rows >= y0[:, None]) & (rows < y1[:, None])
        inside_columns = (columns >= x0[:, None]) & (columns < x1[:, None])

        height, width = labels.shape[:2]

        device = self.device

        labels_device = torch.from_numpy(labels).to(device)

        rows_device = torch.from_numpy(
            np.clip(rows, 0, height - 1)).to(device).unsqueeze(2)
        columns_device = torch.from_numpy(
            np.clip(columns, 0, width - 1)).to(device).unsqueeze(1)

        gathered = labels_device[rows_device, columns_device]

        keep = (
            torch.from_numpy(inside_rows).to(device).unsqueeze(2)
            & torch.from_numpy(inside_columns).to(device).unsqueeze(1)
        )

        wanted = torch.tensor(
            list(segment_ids), device=device, dtype=labels_device.dtype)

        mask = (gathered == wanted.view(-1, 1, 1)) & keep

        # bool 을 float 로 펼치지 않고 바로 축약한다. avg_pool2d 를 쓰려면
        # [N,224,224] 를 float32 로 만들어야 해서 4.4MB 를 쓰고 다시 읽는데,
        # 여기서는 bool(1.1MB)만 읽는다 (실측 1.075 -> 0.262ms, 4배).
        #
        # 정확성: 블록 하나의 합은 최대 block^2 = 1024 라 int32 에서 정확하고,
        # float 변환도 정확하며, 1024 로 나누는 것은 2의 거듭제곱이라 정확하다.
        # avg_pool2d 버전과 **비트 동일**함을 torch.equal 로 확인했다.
        return (
            mask.view(len(segment_ids), grid, block, grid, block)
            .sum(dim=(2, 4), dtype=torch.int32)
            .float()
            .div_(float(block * block))
            .flatten(start_dim=1)
        )


def prepare_from_frame(
    backend,
    rgb_image: np.ndarray,
    labels: np.ndarray,
    segment_ids: Sequence[int],
    boxes: Sequence[Tuple[int, int, int, int]],
    pooling_mode: str = DEFAULT_POOLING_MODE,
    preprocess_path: str = "interp_aa",
) -> Optional["PreparedBatch"]:
    """전-GPU 전처리로 PreparedBatch 를 만든다 (interp_aa / roi_align).

    backend.prepare() 와 **같은 계약**을 지킨다 -- 같은 PreparedBatch 를
    돌려주므로 run() 이 어느 쪽에서 왔는지 알 필요가 없다. 다른 점은 입력이
    crop 목록이 아니라 프레임 한 장 + 라벨맵이라는 것이다. 그래서 crop 복사,
    PIL 이미지 생성, 장별 H2D 가 호출부에서도 사라진다.

    boxes 는 **실제로 crop 할 영역**이어야 한다. 노드는 bbox 에
    bbox_padding 을 더해 crop 하고 boxes 에는 tight bbox 를 담으므로
    (semantics 발행용), 호출부가 패딩이 들어간 박스를 넘겨야 한다. 이걸
    틀리면 pil 경로와 몇 픽셀 어긋난 crop 을 비교하게 된다.

    두 경로가 같은 값을 내는지는 tools/verify_occupancy.py (점유율 비트 동일)
    와 tools/benchmark_voc.py / benchmark_uhumans.py (임베딩 드리프트) 가
    확인한다.
    """
    if pooling_mode not in POOLING_MODES:
        raise ValueError(
            f"pooling_mode must be one of {POOLING_MODES}, "
            f"got '{pooling_mode}'"
        )

    if preprocess_path not in GPU_PREPROCESS_PATHS:
        raise ValueError(
            f"preprocess_path must be one of {GPU_PREPROCESS_PATHS}, "
            f"got '{preprocess_path}'. 'pil' 은 backend.prepare() 를 쓴다."
        )

    if backend.crop_fit != DEFAULT_CROP_FIT:
        raise ValueError(
            f"preprocess_path='{preprocess_path}' requires crop_fit='pad', "
            f"got '{backend.crop_fit}'. 다른 crop_fit 은 기하가 달라 "
            "pil 경로를 써야 한다."
        )

    if not len(segment_ids):
        return None

    if len(boxes) != len(segment_ids):
        raise ValueError(
            f"segment/box count mismatch: segments={len(segment_ids)}, "
            f"boxes={len(boxes)}"
        )

    preprocess = backend.batch_preprocess

    frame = preprocess.frame_tensor(rgb_image)

    images = (
        preprocess.images_from_frame(frame, boxes)
        if preprocess_path == "interp_aa"
        else preprocess.images_from_frame_roi(frame, boxes)
    )

    # cls 는 점유율을 쓰지 않는다. patch_geometry() 를 먼저 부르면 pooling 을
    # 지원하지 않는 visual 엔진에서 예외가 난다 (cls 전용 엔진이 그렇다).
    occupancy = None

    if pooling_mode != "cls":
        grid, _, _ = backend.patch_geometry()

        occupancy = preprocess.occupancy_from_frame(
            labels, segment_ids, boxes, grid)

    return PreparedBatch(
        images=images,
        occupancy=occupancy,
        pooling_mode=pooling_mode,
        count=len(segment_ids),
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

    기본 파라미터(gamma=1, min=0)에서는 항등이다. 점유율이 0..1 이라
    threshold 가 아무것도 거르지 않고 pow(1.0) 도 값을 바꾸지 않는다.
    그런데도 커널 두 개가 돌아 0.19ms 를 쓰고 있었다. 우회하면 0.011ms 이고
    torch.equal 로 비트 동일을 확인했다. 반환값은 호출부가 읽기만 하므로
    (weights.sum / copy_ / cpu) 입력과 같은 텐서를 돌려줘도 안전하다.
    """
    if gamma == 1.0 and min_patch_occupancy <= 0.0:
        return patch_occupancy.float()

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
        preprocess_workers: int = PREPROCESS_WORKERS,
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

        # encode 경로가 실제로 쓰는 배치 전처리. 위 두 개와 같은 값을 내며
        # (오차 4.8e-7) 스레드풀 + device 정규화로 5배 빠르다.
        self.batch_preprocess = BatchPreprocessor(
            resolution=self.resolution,
            crop_fit=crop_fit,
            device=self.device,
            workers=preprocess_workers,
        )

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
        prepared = self.prepare(
            regions=regions,
            masks=masks,
            pooling_mode=pooling_mode,
        )

        return self.run(
            prepared=prepared,
            batch_size=batch_size,
            normalize=normalize,
            gamma=gamma,
            min_patch_occupancy=min_patch_occupancy,
            empty_mask_fallback=empty_mask_fallback,
        )

    # ----------------------------------------------------------------
    # 2단계 분리. TensorRTBackend 와 같은 계약이다 (PreparedBatch 주석 참고).
    # ----------------------------------------------------------------

    def prepare(
        self,
        regions: Sequence[PILImage.Image],
        masks: Optional[Sequence[np.ndarray]] = None,
        pooling_mode: str = DEFAULT_POOLING_MODE,
    ) -> Optional[PreparedBatch]:
        """CPU 전처리만 끝낸다. region 이 없으면 None."""
        if pooling_mode not in POOLING_MODES:
            raise ValueError(
                f"pooling_mode must be one of {POOLING_MODES}, "
                f"got '{pooling_mode}'"
            )

        if not regions:
            return None

        if pooling_mode != "cls":
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

        images = self.batch_preprocess.images(regions)

        grid, _, _ = self.patch_geometry()

        occupancy = (
            None
            if pooling_mode == "cls"
            else self.batch_preprocess.occupancy(masks, grid)
        )

        return PreparedBatch(
            images=images,
            occupancy=occupancy,
            pooling_mode=pooling_mode,
        )

    def run(
        self,
        prepared: Optional[PreparedBatch],
        batch_size: int,
        normalize: bool,
        gamma: float = 1.0,
        min_patch_occupancy: float = 0.0,
        empty_mask_fallback: str = DEFAULT_EMPTY_MASK_FALLBACK,
    ) -> np.ndarray:
        """모델 forward 부터 끝까지."""
        if prepared is None:
            self.last_pooling_stats = None

            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        if prepared.pooling_mode == "cls":
            self.last_pooling_stats = None

            return self.run_cls(
                prepared=prepared,
                batch_size=batch_size,
                normalize=normalize,
            )

        return self.run_mask_weighted(
            prepared=prepared,
            batch_size=batch_size,
            normalize=normalize,
            use_value_tokens=(
                prepared.pooling_mode == "mask_weighted_value"
            ),
            gamma=gamma,
            min_patch_occupancy=min_patch_occupancy,
            empty_mask_fallback=empty_mask_fallback,
        )

    def run_cls(
        self,
        prepared: PreparedBatch,
        batch_size: int,
        normalize: bool,
    ) -> np.ndarray:
        """CLIP encode_image 를 그대로 쓰는 기존 경로."""
        prepared_images = prepared.images

        features: List[torch.Tensor] = []

        for start in range(0, len(prepared_images), batch_size):
            batch = prepared_images[start:start + batch_size]

            with torch.inference_mode():
                batch_features = self.model.encode_image(batch)

            # CUDA 에서 CLIP 은 fp16 으로 동작하므로 float32 로 맞춘다.
            features.append(batch_features.to(dtype=torch.float32))

        stacked = torch.cat(features, dim=0).cpu().numpy().astype(np.float32)

        if normalize:
            stacked = l2_normalize(stacked)

        return stacked

    def run_mask_weighted(
        self,
        prepared: PreparedBatch,
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

        prepared_images = prepared.images
        occupancy_all = prepared.occupancy

        features: List[torch.Tensor] = []
        occupancy_chunks: List[np.ndarray] = []
        weight_chunks: List[np.ndarray] = []
        empty_chunks: List[np.ndarray] = []

        for start in range(0, len(prepared_images), batch_size):
            stop = start + batch_size

            image_batch = prepared_images[start:stop]

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

                occupancy = occupancy_all[start:stop]

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
        preprocess_workers: int = PREPROCESS_WORKERS,
        async_preprocess: bool = DEFAULT_ASYNC_PREPROCESS,
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

        # encode 경로가 실제로 쓰는 배치 전처리. self.preprocess 와 같은 값을
        # 내며(오차 4.8e-7) 스레드풀 + GPU 정규화로 5배 빠르다. 자세한 내용은
        # BatchPreprocessor 문서 참고.
        self.batch_preprocess = BatchPreprocessor(
            resolution=self.resolution,
            crop_fit=crop_fit,
            device="cuda",
            workers=preprocess_workers,
        )
        self.async_preprocess = bool(async_preprocess)

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
            f"preprocess_workers={self.batch_preprocess.workers}, "
            f"async_preprocess={self.async_preprocess}, "
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
        as_tensor: bool = False,
    ) -> Tuple[Any, Optional[Any]]:
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

        # as_tensor=True 면 D2H 없이 device 텐서로 돌려준다. 출력 버퍼는
        # 다음 호출에서 덮어쓰이므로 **반드시 복사**해서 나간다.
        if as_tensor:
            pooled = self.output_buffer[:count].clone()

            cls = (
                self.cls_buffer[:count].clone()
                if self.supports_pooling
                else None
            )

            return pooled, cls

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
        prepared = self.prepare(
            regions=regions,
            masks=masks,
            pooling_mode=pooling_mode,
        )

        return self.run(
            prepared=prepared,
            batch_size=batch_size,
            normalize=normalize,
            gamma=gamma,
            min_patch_occupancy=min_patch_occupancy,
            empty_mask_fallback=empty_mask_fallback,
        )

    # ----------------------------------------------------------------
    # 2단계 분리: prepare() 는 CPU/PIL 전처리, run() 은 엔진 + 후처리.
    # encode() 는 둘을 이어 붙인 것이라 순차 경로와 결과가 같을 수밖에 없다.
    # ----------------------------------------------------------------

    def validate_pooling_mode(self, pooling_mode: str) -> None:
        """엔진이 이 pooling 을 지원하는지 본다.

        조용히 CLS 로 되돌아가면 안 된다. 어떤 pooling 을 쓴 임베딩인지 알 수
        없게 되고, embedding_model_id 도 거짓이 된다.

        mask_weighted_patch 와 mask_weighted_value 는 입출력 이름이 같아서
        엔진만 보고는 구분할 수 없다. 어느 엔진을 넘길지는 노드가
        pooling_mode 에 따라 고른다 (--pooled-engine-path / --value-engine-path).
        """
        if pooling_mode not in POOLING_MODES:
            raise ValueError(
                f"pooling_mode must be one of {POOLING_MODES}, "
                f"got '{pooling_mode}'"
            )

        if pooling_mode == "cls" or self.supports_pooling:
            return

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

    def prepare(
        self,
        regions: Sequence[PILImage.Image],
        masks: Optional[Sequence[np.ndarray]] = None,
        pooling_mode: str = DEFAULT_POOLING_MODE,
    ) -> Optional[PreparedBatch]:
        """파이프라인 1단계를 준비한다.

        async_preprocess=False면 224 기하/H2D/정규화까지 여기서 끝내고,
        True면 PIL 기하 future만 제출해 실제 수집은 run()에서 청크별로 한다.
        두 경로는 같은 기하/정규화 수식을 사용하며 실행 스케줄만 다르다.
        region 이 없으면 None 을 돌려준다.
        """
        self.validate_pooling_mode(pooling_mode)

        if not regions:
            return None

        if pooling_mode != "cls":
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

        # 플랫폼별로 전처리 스케줄만 바꾼다. 결과 수식/기하는 동일하다.
        # async=True  : PIL 기하를 future 로 제출하고 run() 에서 청크별 수집.
        #               CPU resize 와 TensorRT 를 겹칠 수 있다.
        # async=False : 여기서 전체 기하/H2D/정규화를 끝낸다. 기존 Jetson
        #               실행 순서를 그대로 재현하는 경로다.
        if self.async_preprocess:
            pending = self.batch_preprocess.submit(
                regions, self.batch_preprocess.image_array
            )
            images = None
        else:
            pending = None
            images = self.batch_preprocess.images(regions)

        # 224 마스크를 만들지 않고 7x7 점유율을 바로 얻는다 (occupancy 주석).
        # async 경로에서는 PIL worker 와 이 계산도 일부 겹칠 수 있다.
        occupancy = (
            None
            if pooling_mode == "cls"
            else self.batch_preprocess.occupancy(masks, self.patch_grid)
        )

        return PreparedBatch(
            images=images,
            occupancy=occupancy,
            pooling_mode=pooling_mode,
            pending=pending,
            count=len(regions),
        )

    def image_batch(
        self,
        prepared: PreparedBatch,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        """동기/비동기 전처리 차이를 한 곳에서 흡수한다."""
        if prepared.pending is not None:
            return self.batch_preprocess.normalize_images(
                self.batch_preprocess.collect(prepared.pending[start:stop])
            )

        if prepared.images is None:
            raise RuntimeError("PreparedBatch has neither images nor pending futures")

        return prepared.images[start:stop]

    def run(
        self,
        prepared: Optional[PreparedBatch],
        batch_size: int,
        normalize: bool,
        gamma: float = 1.0,
        min_patch_occupancy: float = 0.0,
        empty_mask_fallback: str = DEFAULT_EMPTY_MASK_FALLBACK,
    ) -> np.ndarray:
        """엔진 실행부터 끝까지. 파이프라인 2단계다."""
        if prepared is None:
            self.last_pooling_stats = None

            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        if prepared.pooling_mode == "cls":
            self.last_pooling_stats = None

            return self.run_cls(
                prepared=prepared,
                batch_size=batch_size,
                normalize=normalize,
            )

        return self.run_mask_weighted(
            prepared=prepared,
            batch_size=batch_size,
            normalize=normalize,
            gamma=gamma,
            min_patch_occupancy=min_patch_occupancy,
            empty_mask_fallback=empty_mask_fallback,
        )

    def run_cls(
        self,
        prepared: PreparedBatch,
        batch_size: int,
        normalize: bool,
    ) -> np.ndarray:
        """CLS 임베딩만 뽑는 경로.

        pooling 엔진에서는 CLS 가 cls_embeddings 라는 별도 출력이므로 그것을
        읽는다. 두 엔진 모두 같은 ViT forward 를 재현하므로 값은 동일하다.
        """
        # 엔진 프로파일 상한을 넘는 배치는 넣을 수 없다.
        effective = min(batch_size, self.max_batch)

        chunks: List[torch.Tensor] = []

        for start in range(0, len(prepared), effective):
            batch = self.image_batch(
                prepared, start, min(start + effective, len(prepared))
            )

            pooled, cls = self.infer(batch, as_tensor=True)

            chunks.append(cls if cls is not None else pooled)

        stacked = torch.cat(chunks, dim=0).cpu().numpy().astype(np.float32)

        # 엔진 그래프에 이미 정규화가 들어 있어 멱등이지만,
        # --no-normalize 로 export 한 엔진도 받을 수 있게 그대로 적용한다.
        if normalize:
            stacked = l2_normalize(stacked)

        return stacked

    def run_mask_weighted(
        self,
        prepared: PreparedBatch,
        batch_size: int,
        normalize: bool,
        gamma: float,
        min_patch_occupancy: float,
        empty_mask_fallback: str,
    ) -> np.ndarray:
        """점유율 가중평균을 엔진 안에서 끝내는 경로.

        TorchBackend.run_mask_weighted 와 같은 결과를 내야 한다. 점유율과
        가중치 계산은 같은 함수(BatchPreprocessor.occupancy /
        compute_patch_weights)를 쓰고, weighted mean 이후는 엔진 그래프가
        담당한다.
        """
        if empty_mask_fallback not in EMPTY_MASK_FALLBACKS:
            raise ValueError(
                f"empty_mask_fallback must be one of {EMPTY_MASK_FALLBACKS}, "
                f"got '{empty_mask_fallback}'"
            )

        effective = min(batch_size, self.max_batch)

        occupancy_all = prepared.occupancy

        # 루프 안에서는 **device 텐서만** 쌓는다. .cpu() 는 동기화이면서
        # GIL 을 쥐는 지점이라, 청크마다 부르면 엔진 대기 중에 워커가
        # 다음 청크를 만들지 못한다 (실측: 파이프라이닝이 1.02배로 무산됐던
        # 원인). D2H 는 루프가 끝난 뒤 한 번만 한다.
        feature_chunks: List[torch.Tensor] = []
        weight_gpu_chunks: List[torch.Tensor] = []
        empty_gpu_chunks: List[torch.Tensor] = []

        for start in range(0, len(prepared), effective):
            stop = start + effective

            # async=True면 이 청크 future만 기다리고, sync=False 경로에서는
            # 이미 준비된 tensor slice를 그대로 받는다.
            image_batch = self.image_batch(prepared, start, stop)

            occupancy = occupancy_all[start:stop]

            weights = compute_patch_weights(
                patch_occupancy=occupancy,
                gamma=gamma,
                min_patch_occupancy=min_patch_occupancy,
            )

            embeddings, cls_embeddings = self.infer(
                image_batch, weights, as_tensor=True
            )

            empty = weights.sum(dim=1) <= 0.0

            # 같은 forward 의 CLS 출력이라 두 번 돌 필요가 없다.
            # GPU 에서 고르면 동기화가 없다.
            if empty_mask_fallback == "cls":
                embeddings = torch.where(
                    empty.unsqueeze(1), cls_embeddings, embeddings
                )

            feature_chunks.append(embeddings)
            weight_gpu_chunks.append(weights)
            empty_gpu_chunks.append(empty)

        # ---- 여기서 처음이자 마지막으로 D2H ----
        stacked = torch.cat(feature_chunks, dim=0).cpu().numpy().astype(
            np.float32)

        weights_all = torch.cat(
            weight_gpu_chunks, dim=0).float().cpu().numpy()
        empty_all = torch.cat(empty_gpu_chunks, dim=0).cpu().numpy()
        occupancy_all = occupancy_all.float().cpu().numpy()

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
    preprocess_workers: int = PREPROCESS_WORKERS,
    async_preprocess: bool = DEFAULT_ASYNC_PREPROCESS,
):
    """이름으로 백엔드를 만든다."""
    if backend == "torch":
        return TorchBackend(
            checkpoint=checkpoint,
            use_cuda=use_cuda,
            crop_fit=crop_fit,
            preprocess_workers=preprocess_workers,
        )

    if backend == "tensorrt":
        return TensorRTBackend(
            engine_path=engine_path,
            crop_fit=crop_fit,
            preprocess_workers=preprocess_workers,
            async_preprocess=async_preprocess,
        )

    raise ValueError(
        f"backend must be 'torch' or 'tensorrt', got '{backend}'"
    )
