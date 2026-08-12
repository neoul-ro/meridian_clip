#!/usr/bin/env python3

"""
Meridian Perception Frontend - CLIP.

Input:
    /camera/rgb      sensor_msgs/Image (rgb8)   color frame
    /segment_image   sensor_msgs/Image (mono8)  label map, 픽셀 값 = segment_id

Processing:
    label map의 unique positive segment_id마다
    masked RGB region 또는 crop을 구성해 semantic embedding 생성

Output:
    /instance_embedding_set   meridian_msgs/InstanceEmbeddingSet / frame

Boundary:
    frame-local segment에 embedding만 부여하며
    geometry와 persistent identity는 생성하지 않는다.

계약 (meridian_msgs v0.0.2 / neoul-ro):
    `/segment_image`은 커스텀 메시지가 아니라 표준 sensor_msgs/Image (mono8)이며
    픽셀 값이 곧 segment_id다. 0=background, 1..255=segment.
    InstanceEmbeddingSet.segment_ids가 uint8[]이므로 세그먼트는 최대 255개다.

    두 입력 모두 std_msgs/Header를 가지므로 원리상 message_filters로 묶을 수
    있지만, 프레임 결합 정책(버퍼 크기, 짝 없는 프레임 폐기)을 직접 통제하려고
    capture timestamp를 키로 하는 버퍼를 그대로 유지한다.

설정값은 ROS 파라미터가 아니라 클래스 밖 DEFAULT_* 상수 + argparse 로 주고,
main()에서 생성자 인자로 넘긴다. (fastsam_node.py 와 동일한 방식)
"""

from __future__ import annotations

import argparse
import sys

from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rclpy

from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from meridian_msgs.msg import InstanceEmbeddingSet
from PIL import Image as PILImage
from sensor_msgs.msg import Image as RosImage
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rclpy.utilities import remove_ros_args

from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from meridian_clip.clip_backend import (
    CROP_FITS,
    DEFAULT_CROP_FIT,
    EMPTY_MASK_FALLBACKS,
    POOLING_MODES,
    build_debug_geometry,
    create_backend,
    create_text_encoder,
    l2_normalize,
)


# label map에서 0은 background/invalid 이므로 embedding 대상이 아니다.
BACKGROUND_SEGMENT_ID = 0

# InstanceEmbeddingSet.segment_ids가 uint8[]이므로 유효 범위는 1..255다.
MAX_SEGMENT_ID = 255

# crop/masking policy는 아직 TBD 이므로 실행 시 전환 가능하게 둔다.
#   bbox        : bbox로 자르기만 함 (기본값)
#   masked_bbox : bbox로 자르고 마스크 밖은 mask_fill로 채움
#   masked_full : 원본 크기 유지, 마스크 밖만 mask_fill로 채움
CROP_POLICIES = (
    "bbox",
    "masked_bbox",
    "masked_full",
)

# ---- 노드 설정 기본값 (클래스 밖 선언, argparse 로 덮어쓸 수 있음) ----

# 계약 토픽. RealSense를 직접 쓸 때는
#   --color-topic /camera/camera/color/image_raw
DEFAULT_COLOR_TOPIC = "/camera/rgb"
DEFAULT_SEGMENT_TOPIC = "/segment_image"
DEFAULT_EMBEDDING_TOPIC = "/instance_embedding_set"
DEFAULT_SEMANTICS_TOPIC = "/clip_semantics"

# 텍스트 인코더가 쓰는 기본 후보 단어. --prompts 또는 --prompt-file 로 바꾼다.
# 프롬프트는 시작할 때 한 번만 인코딩되므로 개수가 늘어도 프레임 비용은 그대로다.
DEFAULT_PROMPTS = (
    "a person",
    "a chair",
    "a desk",
    "a computer monitor",
    "a keyboard",
    "a laptop",
    "a cup",
    "a bottle",
    "a door",
    "a wall",
    "a floor",
    "a ceiling",
    "a potted plant",
    "a backpack",
    "a cardboard box",
    "a book",
    "a cable",
    "a window",
)

# 세그먼트마다 발행할 상위 후보 수
DEFAULT_TOP_K = 3

# 이보다 낮은 유사도의 세그먼트는 semantics 에서 제외한다 (0이면 전부 발행)
DEFAULT_MIN_SCORE = 0.0

# 프레임 로그: 몇 프레임마다 찍을지 (1=매 프레임, 0=끄기)
DEFAULT_LOG_EVERY = 1

# 임베딩 벡터에서 터미널에 보여줄 앞쪽 값 개수 (0이면 값은 생략하고 개수만)
DEFAULT_LOG_VALUES = 8

# 값까지 보여줄 세그먼트 수 상한
DEFAULT_LOG_MAX_SEGMENTS = 5

# 이미지 인코더 구현.
#   torch    : models/ViT-B-32.pt 를 clip.load 로 로드 (visual + text)
#   tensorrt : models/*.engine 로드 (visual only, fp16, 약 2.3배 빠름)
BACKENDS = (
    "torch",
    "tensorrt",
)

# 기본은 tensorrt. 엔진이 없거나 다른 GPU/TensorRT 버전이면 시작할 때 바로 실패하며,
# 그때는 --backend torch 로 돌리거나 build_engine.py 로 엔진을 다시 빌드한다.
DEFAULT_BACKEND = "tensorrt"

DEFAULT_ENGINE_PATH = (
    "~/meridian/src/meridian_clip/models/"
    "clip_vit_b32_visual_fp16.engine"
)

# mask_weighted_patch 용 엔진. weighted mean 과 ln_post/proj 까지 그래프에 들어
# 있고 (images, patch_weights) -> (embeddings, cls_embeddings) 형태다.
#   python3 meridian_clip/export_onnx.py  --part visual_pooled
#   python3 meridian_clip/build_engine.py --part visual_pooled
DEFAULT_POOLED_ENGINE_PATH = (
    "~/meridian/src/meridian_clip/models/"
    "clip_vit_b32_visual_pooled_fp16.engine"
)

# mask_weighted_value 용 엔진. 입출력은 위와 똑같고 마지막 블록에서 value
# 투영을 patch feature 로 쓰는 것만 다르다. 이름으로 구분할 수 없어 별도
# 경로로 둔다.
#   python3 meridian_clip/export_onnx.py  --part visual_pooled_value
#   python3 meridian_clip/build_engine.py --part visual_pooled_value
DEFAULT_VALUE_ENGINE_PATH = (
    "~/meridian/src/meridian_clip/models/"
    "clip_vit_b32_visual_pooled_value_fp16.engine"
)

# 텍스트 인코더도 같은 방식으로 고를 수 있다.
#   tensorrt : models/clip_vit_b32_text_fp16.engine  (.pt 불필요)
#   torch    : models/ViT-B-32.pt
DEFAULT_TEXT_BACKEND = "tensorrt"

DEFAULT_TEXT_ENGINE_PATH = (
    "~/meridian/src/meridian_clip/models/"
    "clip_vit_b32_text_fp16.engine"
)

# --backend torch 일 때 로드할 로컬 checkpoint(.pt) 경로.
# models/ 는 install 트리에 복사하지 않으므로 소스 경로를 가리킨다.
DEFAULT_MODEL_PATH = (
    "~/meridian/src/meridian_clip/models/ViT-B-32.pt"
)

# 49개 patch token을 하나로 합치는 방법. 자세한 설명은 clip_backend.py 참고.
#   cls                 : CLS token을 그대로 사용 (CLIP 원본 경로)
#   mask_weighted_patch : 패치별 객체 점유율로 768차원에서 가중평균
#   mask_weighted_value : 같은 가중평균이되 마지막 블록의 value 투영을
#                         patch feature로 사용 (MaskCLIP). 기본값.
#
# 가중평균을 쓰는 이유: crop 안에서도 마스크 밖 패치를 가중치로 배제하므로,
# 얇거나 비스듬한 물체처럼 bbox 안 배경 비율이 큰 세그먼트에서 임베딩이
# 배경에 덜 끌린다. 픽셀을 칠하지 않고 가중치로만 거르므로 기본 crop_policy가
# bbox인 이유이기도 하다. cls는 이 수단이 없어 crop 정책(검정 마스킹)에만
# 의존한다 -- cls로 돌릴 때는 --crop-policy masked_bbox 를 같이 주는 편이 낫다.
#
# VOC2012 val 실측 (GT 인스턴스 3420개, 프롬프트 "a photo of a {}" 20개,
# crop_policy=bbox / crop_fit=pad 즉 현재 기본값):
#   pooling                top-1     macro     AUC      분리도
#   cls                    83.10%   88.12%   0.9775   0.0662
#   mask_weighted_patch     7.89%   10.55%   0.4633   0.1158
#   mask_weighted_value    87.98%   90.20%   0.9897   0.1392   <- 기본값
#   value + W(정렬행렬)     90.61%   91.25%   0.9887   0.1392
#
# value가 기본값인 이유: 네 지표 전부에서 최고다. crop_policy가 masked_bbox
# 이던 시절에는 top-1만 cls에 졌었는데(81.67% vs 83.39%), bbox로 바꾸면서
# 그 맞바꿈이 사라졌다 -- 가중평균이 이미 마스크 밖을 배제하는데 픽셀까지
# 검게 칠하면 물체 경계와 맥락만 잃기 때문이다.
#
# 여기의 AUC는 열 방향(text->image)이다 -- 프롬프트를 고정하고 인스턴스를
# 줄 세운 값이라 맵에 말로 질의를 던지는 방향과 같다. 언어 쪽 측정은
# tools/benchmark_language.py 에 따로 있고 두 가지를 더 준다.
#   1. 반대 방향(image->text) AUC:
#      cls 0.9817 / patch 0.4057 / value 0.9816 / value+Wᵀ 0.9883
#      masked_bbox 시절에는 이 방향에서 value가 cls에 졌는데 지금은 동률이다.
#   2. 프롬프트 표현을 바꿨을 때의 강건성. value 가 가장 덜 흔들린다
#      (템플릿 10종 mAP: value 0.9003±0.0169, cls 0.7434±0.0662).
#
# patch가 무너지는 이유는 정보 손실이 아니라 축 어긋남이다. 선형 프로브는
# 89.12%로 cls(89.77%)와 동급인데, ln_post/proj가 CLS 전용으로 학습돼서
# 투영 결과가 텍스트와 직교하는 좌표계에 떨어진다. image->text AUC 0.4057,
# 즉 무작위(0.5)보다 나쁜 것이 그 증거다. 정렬행렬 하나로 0.9843까지
# 돌아온다 -- 정보는 남아 있고 좌표계만 틀어졌다는 뜻이다.
#
# 두 가중평균 모드 모두 torch 백엔드와 전용 TensorRT 엔진에서 된다.
# tensorrt 백엔드일 때 각각 --pooled-engine-path / --value-engine-path를
# 쓰며, 엔진이 없으면 시작할 때 바로 실패한다. 엔진을 못 만드는 환경에서는
# --pooling-mode cls 로 되돌린다.
DEFAULT_POOLING_MODE = "mask_weighted_value"

# 패치 가중치 지수. w_i = r_i ** gamma. 1.0이면 점유율을 그대로 쓴다.
DEFAULT_PATCH_WEIGHT_GAMMA = 1.0

# 이 값보다 낮은 점유율의 패치는 가중치를 0으로 만든다.
DEFAULT_MIN_PATCH_OCCUPANCY = 0.0

# 가중치 합이 0인 세그먼트(빈 마스크) 처리 방법. cls/skip/error.
DEFAULT_EMPTY_MASK_FALLBACK = "cls"

# pooling 방식이 다르면 임베딩 공간도 다르므로 downstream이 섞지 않도록
# 모델 ID를 구분한다. --embedding-model-id를 명시하면 그 값이 우선한다.
#
# cls ID 의 _masked_bbox 접미사는 DEFAULT_CROP_POLICY 가 masked_bbox 이던 시절에
# 붙은 것이다. 기본값이 bbox 로 바뀌었으니 이름이 더는 맞지 않지만, 저장된 맵이
# 이 문자열로 임베딩 공간을 식별하므로 마음대로 바꾸면 기존 데이터와의 대조가
# 깨진다. 바꿀 거면 downstream 과 함께 v2 로 올린다.
EMBEDDING_MODEL_IDS = {
    "cls": "openai_clip_vit_b32_cls_masked_bbox_v1",
    "mask_weighted_patch": "openai_clip_vit_b32_mask_weighted_patch_v1",
    "mask_weighted_value": "openai_clip_vit_b32_mask_weighted_value_v1",
}

# 모델, weight, 전처리, normalization을 식별하는 문자열.
# 빈 문자열이면 EMBEDDING_MODEL_IDS에서 pooling 방식에 맞춰 고른다.
DEFAULT_EMBEDDING_MODEL_ID = ""

# 텍스트 쪽 정렬. **이쪽이 기본 경로다.**
#
#   (e W)·t = e W tᵀ = e·(t Wᵀ)
#
# 즉 이미지를 W 로 옮기는 것과 텍스트를 Wᵀ 로 옮기는 것은 같은 내적을 준다
# (실측: 두 방식의 top-1 예측이 3420개 중 100.0000% 일치). 그런데 텍스트
# 쪽에 걸면 **이미지 임베딩이 원본 value 공간에 그대로 남는다.**
#
#   구성                           top-1     AUC      mAP     분리도
#   맵=value, 텍스트 그대로        87.98%   0.9897   0.9076   0.1392
#   맵=value@W (이미지 쪽)         90.61%   0.9879   0.9142   0.1081
#   맵=value, 텍스트@Wᵀ (이 경로)   90.61%   0.9887   0.8974   0.1392  <- 기본값
#
# top-1 은 이미지 쪽과 소수점까지 같다 (위 항등식의 실측 확인). 갈리는 곳은
# 재정규화가 개입하는 순위 지표뿐이고, 결정적인 칸은 분리도다 -- 텍스트 쪽은
# 이미지 임베딩을 건드리지 않으므로 0.1392 가 정확히 보존되는 반면 이미지
# 쪽은 0.1081 로 22% 깎인다. VLMap 처럼 임베딩을 저장해 두는 downstream 에서
# 이게 핵심이다: 맵은 분리도가 가장 좋은 공간에 남고, 변환 대상이 맵 전체가
# 아니라 프롬프트 몇 개뿐이며, 맵을 다시 만들지 않고 행렬만 갈아끼울 수 있다.
# 선형변환이라 프레임 간 평균과도 교환된다: mean(E)·W = mean(E·W).
#
# 검색 품질(mAP)만 보면 오히려 이미지 쪽이 낫다 (0.9142 vs 0.8974). 기본값을
# 텍스트 쪽으로 두는 이유는 이 노드의 산출물이 /instance_embedding_set,
# 즉 저장되는 임베딩이기 때문이다.
#
# 텍스트 쪽도 공짜는 아니다. 프롬프트 표현 강건성을 깎는다 -- Wᵀ 가 value 를
# cls 좌표계로 끌어오는 변환이라 cls 의 프롬프트 취약성까지 따라온다:
#   템플릿 10종 mAP   value 0.9003±0.0169   value+Wᵀ 0.8386±0.0549
#   최악의 템플릿      value 0.8519          value+Wᵀ 0.7161
# 즉 검색(mAP)은 어느 가족에서도 순수 value 가 낫고, 라벨(top-1)은 Wᵀ 가
# 낫다 -- 단 자연문 프롬프트에서는 top-1 조차 뒤집힌다 (문장 2종 평균
# 77.02% vs 순수 value 77.44%). Wᵀ 의 이득은 짧은 프롬프트에서 가장 크다.
# 맵에 텍스트 질의를 던지는 경로라면 --text-alignment-matrix none 을 검토한다.
# 측정: tools/benchmark_language.py --image-alignment
#
# **변환한 텍스트를 다시 정규화하면 안 된다.** Wᵀ 를 지나면 프롬프트마다
# 벡터 길이가 달라지는데 그 길이 차이가 정답 신호의 일부다. 정규화하면
# top-1 이 84.56% -> 40.56% 로 무너진다 (masked_bbox 시절 실측).
#
# ""  = pooling 모드에서 자동으로 고른다 (아래 표)
# "none" = 쓰지 않는다
# 그 외 = 그 경로의 .npy 를 쓴다
DEFAULT_TEXT_ALIGNMENT_MATRIX = ""

# pooling 모드별 기본 텍스트 정렬 행렬. 엔진 경로와 같은 방식으로 노드가
# 자동 선택한다. cls 는 이미 cls 좌표계라 변환이 필요 없다.
#
# 파일은 tools/fit_alignment.py --source-mode <모드> 로 만들며, 이때
# --crop-policy masked_bbox 를 준다. 노드 기본값(bbox)과 일부러 다르다 --
# 행렬 학습은 런타임을 흉내 내는 절차가 아니라 두 좌표계의 차이만 뽑아내는
# 캘리브레이션이고, 배경이 섞인 crop 은 양쪽 임베딩에 공통 잡음을 넣어
# 최소제곱이 그 잡음까지 맞추게 만든다. 런타임을 bbox 로 고정하고 학습
# 조건만 바꿔 잰 값 (VOC val top-1):
#   masked_bbox -> masked_bbox   90.61%   <- 기본
#   bbox        -> masked_bbox   89.88%
#   bbox        -> bbox          88.51%
#   정렬 없음                     87.98%
# 자세한 내용은 README "행렬은 masked_bbox crop 으로 학습한다" 절.
TEXT_ALIGNMENT_MATRICES = {
    "cls": "",
    "mask_weighted_patch": (
        "~/meridian/src/meridian_clip/models/align_patch_to_cls.npy"
    ),
    "mask_weighted_value": (
        "~/meridian/src/meridian_clip/models/align_value_to_cls.npy"
    ),
}

# 이미지 쪽 정렬 행렬 [D, D] 를 담은 .npy 경로. ""면 쓰지 않는다 (기본).
#
# 텍스트 쪽(위)이 분리도를 그대로 보존하므로 보통 이쪽은 쓸 필요가 없다.
# 임베딩을 저장하지 않고 그 자리에서 라벨만 뽑는 경우처럼, 정렬된 임베딩
# 자체가 필요할 때만 쓴다. 둘을 동시에 켜면 W 가 두 번 적용되므로 막아 둔다.
#
# 무엇을 하나: pooling 임베딩을 cls 좌표계로 돌려놓는 512x512 행렬이다.
# ln_post/proj는 학습 내내 CLS 토큰만 입력으로 받았으므로, 그 함수가 본 적
# 없는 입력(가중평균된 토큰)을 넣으면 출력이 텍스트와 어긋난 좌표계에
# 떨어진다. 어긋남이 고정된 선형변환이라 행렬 하나로 되돌아온다.
#
# 같은 crop을 두 방식으로 인코딩한 (source, cls) 쌍만 있으면 최소제곱으로
# 구해지며 **정답 라벨이 필요 없다** (tools/fit_alignment.py).
#
# VOC2012 val 실측 (crop_policy=bbox, models/ 의 두 행렬 모두 VOC train 을
# masked_bbox crop 으로 학습 -- TEXT_ALIGNMENT_MATRICES 주석 참고):
#   구성                          top-1     AUC      분리도
#   mask_weighted_value          87.98%   0.9897   0.1392   <- 행렬 없음
#   value + align_value_to_cls   90.61%   0.9879   0.1081
#   mask_weighted_patch           7.89%   0.4633   0.1158   <- 행렬 없음
#   patch + align_patch_to_cls   88.30%   0.9848   0.1123
#
# 기본 모드(value)에서는 선택 사항이다. 맞바꿈이 있기 때문이다 -- 라벨
# 정확도(top-1)는 오르지만 임베딩 자체의 품질(AUC, 분리도)은 내린다.
#   /clip_semantics 의 라벨이 목적       -> 쓴다
#   임베딩 매칭/다운스트림 학습이 목적    -> 쓰지 않는다 (기본)
#   맵에 텍스트 질의를 던지는 것이 목적   -> 쓰지 않는다 (DEFAULT_TEXT_ALIGNMENT_
#                                          MATRIX 의 mAP/강건성 표 참고)
# patch 모드에서는 사실상 필수다 (AUC 0.4633 = 무작위 수준).
#
# 행렬은 도메인에 딸린 물건이다. 두 파일 모두 VOC(일상 사물 사진)로 만들어
# 실내 로봇 장면 일반화는 미확인이며, 대상 도메인 이미지로 다시 뽑으면 된다.
#
# 정규화된 임베딩에 곱하고 다시 정규화한다. 엔진 밖 후처리라 재빌드가
# 필요 없고 backend(torch/tensorrt)와 무관하게 같은 값이 나온다.
DEFAULT_ALIGNMENT_MATRIX = ""

# crop/mask/occupancy PNG를 저장할 디렉터리 (""면 저장하지 않음).
# 이미지와 마스크의 공간 정렬은 임베딩 숫자로는 드러나지 않으므로
# (정렬이 깨져도 L2는 여전히 1이다) 눈으로 확인할 수단을 둔다.
DEFAULT_DEBUG_SAVE_DIR = ""

# 몇 프레임마다 디버그 이미지를 저장할지
DEFAULT_DEBUG_SAVE_EVERY = 30

DEFAULT_USE_CUDA = True

# crop 안에서 마스크 밖을 어떻게 할지.
#   bbox        : 그대로 둔다 (기본값)
#   masked_bbox : mask_fill(기본 검정)로 덮는다
#   masked_full : 자르지 않고 마스크 밖만 덮는다
#
# 기본값이 bbox 인 이유: 기본 pooling(mask_weighted_value)이 이미 패치 점유율로
# 마스크 밖을 배제하므로 픽셀을 검게 칠하는 일이 중복이다. 오히려 검정은 CLIP 이
# 학습 중 본 적 없는 분포라 patch feature 를 흔들고, 물체 경계와 주변 맥락
# (책상 위의 컵 같은)까지 지운다. cls pooling 은 마스크를 전혀 쓰지 않으므로
# 검정 마스킹이 유일한 배경 억제 수단이다 -- --pooling-mode cls 로 돌릴 때는
# --crop-policy masked_bbox 를 같이 주는 편이 낫다.
#
# 주의: README 와 DEFAULT_POOLING_MODE 주석의 VOC2012 실측값은 기본값이 바뀌기
# 전, 즉 masked_bbox 로 잰 것이다. 세 pooling 모드의 상대 순위가 뒤집힐 만한
# 변화는 아니지만 절대 수치를 재현하려면 --crop-policy masked_bbox 가 필요하다.
DEFAULT_CROP_POLICY = "bbox"

# 마스크 밖 픽셀을 채울 grayscale 값
DEFAULT_MASK_FILL = 0

# bbox 주변에 남길 여유 픽셀
DEFAULT_BBOX_PADDING = 4

# 이보다 작은 segment는 embedding을 만들지 않는다.
DEFAULT_MIN_SEGMENT_PIXELS = 16

# 한 번에 encode_image에 넣을 segment 수
DEFAULT_BATCH_SIZE = 16

# cosine similarity를 쓰는 downstream을 위한 L2 정규화
DEFAULT_NORMALIZE_EMBEDDINGS = True

# 짝을 기다리는 frame을 몇 개까지 들고 있을지
DEFAULT_SYNC_BUFFER_SIZE = 30

# positive segment가 없는 frame도 빈 set으로 발행할지
DEFAULT_PUBLISH_EMPTY_SETS = True

# QoS 기본값은 "어느 쪽과도 붙는" 조합으로 둔다.
#   구독 BEST_EFFORT : RELIABLE 발행자(meridian_sensor, fastsam_node)와
#                      BEST_EFFORT 발행자(RealSense 드라이버) 양쪽에서 받는다.
#   발행 RELIABLE    : RELIABLE 구독자와 BEST_EFFORT 구독자 양쪽에 전달된다.
# ROS2 규칙상 BEST_EFFORT 발행 -> RELIABLE 구독만 연결되지 않으므로,
# 이 조합이면 어떤 짝과도 QoS incompatible 이 나지 않는다.
DEFAULT_RELIABLE_INPUT = False
DEFAULT_RELIABLE_OUTPUT = True

# 발행 큐 깊이 (계약의 KEEP_LAST depth 10)
DEFAULT_QOS_DEPTH = 10


def build_qos(reliable: bool, depth: int = DEFAULT_QOS_DEPTH) -> QoSProfile:
    """KEEP_LAST/VOLATILE 고정, reliability 만 고르는 QoS 프로파일."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=(
            QoSReliabilityPolicy.RELIABLE
            if reliable
            else QoSReliabilityPolicy.BEST_EFFORT
        ),
        durability=DurabilityPolicy.VOLATILE,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Meridian CLIP inference ROS2 노드",
    )

    parser.add_argument(
        "--color-topic",
        default=DEFAULT_COLOR_TOPIC,
    )
    parser.add_argument(
        "--segment-topic",
        default=DEFAULT_SEGMENT_TOPIC,
    )
    parser.add_argument(
        "--embedding-topic",
        default=DEFAULT_EMBEDDING_TOPIC,
    )
    parser.add_argument(
        "--semantics-topic",
        default=DEFAULT_SEMANTICS_TOPIC,
        help="세그먼트별 zero-shot 결과(vision_msgs/Detection2DArray)",
    )

    parser.add_argument(
        "--publish-semantics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="텍스트 인코더를 로드해 semantics 를 함께 발행할지",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=list(DEFAULT_PROMPTS),
        help="zero-shot 후보 단어들",
    )
    parser.add_argument(
        "--prompt-file",
        default="",
        help="후보 단어를 한 줄에 하나씩 적은 파일 (--prompts 보다 우선)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="세그먼트마다 발행할 상위 후보 수",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="이 유사도 미만인 세그먼트는 semantics 에서 제외",
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help="몇 프레임마다 마스크 수/임베딩을 터미널에 찍을지 (0=끄기)",
    )
    parser.add_argument(
        "--log-values",
        type=int,
        default=DEFAULT_LOG_VALUES,
        help="임베딩 벡터에서 보여줄 앞쪽 값 개수 (0=값 생략)",
    )
    parser.add_argument(
        "--log-max-segments",
        type=int,
        default=DEFAULT_LOG_MAX_SEGMENTS,
        help="값까지 보여줄 세그먼트 수 상한",
    )

    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=DEFAULT_BACKEND,
        help="이미지 인코더 구현 (torch=.pt, tensorrt=.engine)",
    )
    parser.add_argument(
        "--engine-path",
        default=DEFAULT_ENGINE_PATH,
        help="--backend tensorrt + --pooling-mode cls 일 때 쓸 .engine 경로",
    )
    parser.add_argument(
        "--pooled-engine-path",
        default=DEFAULT_POOLED_ENGINE_PATH,
        help=(
            "--backend tensorrt + --pooling-mode mask_weighted_patch 일 때 쓸 "
            ".engine 경로 (export_onnx.py --part visual_pooled 로 생성)"
        ),
    )
    parser.add_argument(
        "--crop-fit",
        choices=CROP_FITS,
        default=DEFAULT_CROP_FIT,
        help=(
            "crop 을 224x224 로 만드는 방법. "
            "pad=긴 변을 224 에 맞추고 채움 (기본, 물체가 잘리지 않음), "
            "centercrop=CLIP 원본 (짧은 변 기준 + 가운데 오려내기), "
            "stretch=224x224 로 늘림"
        ),
    )
    parser.add_argument(
        "--value-engine-path",
        default=DEFAULT_VALUE_ENGINE_PATH,
        help=(
            "--backend tensorrt + --pooling-mode mask_weighted_value 일 때 쓸 "
            ".engine 경로 (export_onnx.py --part visual_pooled_value 로 생성)"
        ),
    )

    parser.add_argument(
        "--text-backend",
        choices=BACKENDS,
        default=DEFAULT_TEXT_BACKEND,
        help="텍스트 인코더 구현 (tensorrt=.engine, torch=.pt)",
    )
    parser.add_argument(
        "--text-engine-path",
        default=DEFAULT_TEXT_ENGINE_PATH,
        help="--text-backend tensorrt 일 때 쓸 텍스트 .engine 경로",
    )

    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="--backend torch 일 때 로드할 CLIP checkpoint(.pt) 경로",
    )
    parser.add_argument(
        "--embedding-model-id",
        default=DEFAULT_EMBEDDING_MODEL_ID,
        help="비워두면 --pooling-mode 에 맞춰 자동으로 정해진다",
    )

    parser.add_argument(
        "--pooling-mode",
        choices=POOLING_MODES,
        default=DEFAULT_POOLING_MODE,
        help=(
            "49개 patch token 을 합치는 방법. "
            "mask_weighted_value=마지막 블록의 value 투영을 마스크 점유율로 "
            "가중평균 (기본값. VOC2012 val 에서 AUC 0.9897 / top-1 87.98%), "
            "mask_weighted_patch=최종 patch token 으로 같은 가중평균 "
            "(텍스트 정렬이 깨져 AUC 0.4633, --alignment-matrix 없이는 "
            "zero-shot 라벨링이 동작하지 않는다), "
            "cls=CLIP 원본 CLS token. "
            "tensorrt 백엔드에서 앞의 두 모드는 각각 --value-engine-path / "
            "--pooled-engine-path 의 전용 엔진이 필요하다"
        ),
    )
    parser.add_argument(
        "--text-alignment-matrix",
        default=DEFAULT_TEXT_ALIGNMENT_MATRIX,
        help=(
            "텍스트 임베딩에 Wᵀ 를 곱해 프롬프트를 이미지 공간으로 끌어온다. "
            "빈 값이면 pooling 모드에 맞는 행렬을 자동으로 고르고(기본), "
            "'none' 이면 끈다. 이미지 임베딩은 원본 공간에 그대로 남으므로 "
            "저장해 두는 임베딩의 품질을 해치지 않는다 -- "
            "VOC2012 val 기준 top-1 87.98%%->90.61%%, 분리도는 0.1392 그대로 "
            "(이미지 쪽 --alignment-matrix 는 같은 top-1 에 분리도 0.1081)"
        ),
    )
    parser.add_argument(
        "--alignment-matrix",
        default=DEFAULT_ALIGNMENT_MATRIX,
        help=(
            "정렬 행렬 [D, D] .npy 경로. 임베딩에 곱하고 다시 정규화한다. "
            "기본 모드에는 models/align_value_to_cls.npy 를 쓴다 -- "
            "top-1 87.98%%->90.61%% 로 오르지만 분리도는 0.1392->0.1081 로 "
            "내린다 (라벨 정확도와 임베딩 품질의 맞바꿈). "
            "tools/fit_alignment.py 로 만든다 (정답 라벨 불필요). "
            "쓰면 embedding_model_id 에 _aligned 접미사가 붙는다"
        ),
    )
    parser.add_argument(
        "--patch-weight-gamma",
        type=float,
        default=DEFAULT_PATCH_WEIGHT_GAMMA,
        help="패치 가중치 지수 w=r**gamma (가중평균 모드 전용)",
    )
    parser.add_argument(
        "--min-patch-occupancy",
        type=float,
        default=DEFAULT_MIN_PATCH_OCCUPANCY,
        help="이 점유율 미만인 패치는 가중치 0 (가중평균 모드 전용)",
    )
    parser.add_argument(
        "--empty-mask-fallback",
        choices=EMPTY_MASK_FALLBACKS,
        default=DEFAULT_EMPTY_MASK_FALLBACK,
        help="패치 점유율이 전부 0인 세그먼트를 어떻게 할지",
    )

    parser.add_argument(
        "--debug-save-dir",
        default=DEFAULT_DEBUG_SAVE_DIR,
        help="crop/mask/occupancy PNG 를 저장할 디렉터리 (빈 값이면 끄기)",
    )
    parser.add_argument(
        "--debug-save-every",
        type=int,
        default=DEFAULT_DEBUG_SAVE_EVERY,
        help="몇 프레임마다 디버그 이미지를 저장할지",
    )

    parser.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_CUDA,
    )

    parser.add_argument(
        "--crop-policy",
        choices=CROP_POLICIES,
        default=DEFAULT_CROP_POLICY,
        help=(
            "crop 안에서 마스크 밖 픽셀 처리. "
            "bbox=그대로 둔다 (기본값. 가중평균 pooling 이 이미 점유율로 "
            "배제한다), "
            "masked_bbox=mask_fill 로 덮는다 (--pooling-mode cls 를 쓸 때 권장), "
            "masked_full=자르지 않고 마스크 밖만 덮는다"
        ),
    )
    parser.add_argument(
        "--mask-fill",
        type=int,
        default=DEFAULT_MASK_FILL,
    )
    parser.add_argument(
        "--bbox-padding",
        type=int,
        default=DEFAULT_BBOX_PADDING,
    )

    parser.add_argument(
        "--min-segment-pixels",
        type=int,
        default=DEFAULT_MIN_SEGMENT_PIXELS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--normalize-embeddings",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_NORMALIZE_EMBEDDINGS,
    )

    parser.add_argument(
        "--sync-buffer-size",
        type=int,
        default=DEFAULT_SYNC_BUFFER_SIZE,
    )
    parser.add_argument(
        "--publish-empty-sets",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_PUBLISH_EMPTY_SETS,
    )
    parser.add_argument(
        "--qos-depth",
        type=int,
        default=DEFAULT_QOS_DEPTH,
    )
    parser.add_argument(
        "--reliable-input",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RELIABLE_INPUT,
        help="color/segment 구독을 RELIABLE 로 (기본: BEST_EFFORT)",
    )
    parser.add_argument(
        "--reliable-output",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RELIABLE_OUTPUT,
        help="embedding/semantics 발행을 RELIABLE 로 (기본: RELIABLE, 계약 준수)",
    )

    return parser


class ClipInferenceNode(Node):
    """frame-local segment마다 CLIP semantic embedding을 부여하는 노드."""

    def __init__(
        self,
        color_topic: str = DEFAULT_COLOR_TOPIC,
        segment_topic: str = DEFAULT_SEGMENT_TOPIC,
        embedding_topic: str = DEFAULT_EMBEDDING_TOPIC,
        semantics_topic: str = DEFAULT_SEMANTICS_TOPIC,
        publish_semantics: bool = True,
        prompts: Optional[List[str]] = None,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        log_every: int = DEFAULT_LOG_EVERY,
        log_values: int = DEFAULT_LOG_VALUES,
        log_max_segments: int = DEFAULT_LOG_MAX_SEGMENTS,
        backend: str = DEFAULT_BACKEND,
        engine_path: str = DEFAULT_ENGINE_PATH,
        pooled_engine_path: str = DEFAULT_POOLED_ENGINE_PATH,
        value_engine_path: str = DEFAULT_VALUE_ENGINE_PATH,
        crop_fit: str = DEFAULT_CROP_FIT,
        text_backend: str = DEFAULT_TEXT_BACKEND,
        text_engine_path: str = DEFAULT_TEXT_ENGINE_PATH,
        model_path: str = DEFAULT_MODEL_PATH,
        embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
        alignment_matrix: str = DEFAULT_ALIGNMENT_MATRIX,
        text_alignment_matrix: str = DEFAULT_TEXT_ALIGNMENT_MATRIX,
        pooling_mode: str = DEFAULT_POOLING_MODE,
        patch_weight_gamma: float = DEFAULT_PATCH_WEIGHT_GAMMA,
        min_patch_occupancy: float = DEFAULT_MIN_PATCH_OCCUPANCY,
        empty_mask_fallback: str = DEFAULT_EMPTY_MASK_FALLBACK,
        debug_save_dir: str = DEFAULT_DEBUG_SAVE_DIR,
        debug_save_every: int = DEFAULT_DEBUG_SAVE_EVERY,
        use_cuda: bool = DEFAULT_USE_CUDA,
        crop_policy: str = DEFAULT_CROP_POLICY,
        mask_fill: int = DEFAULT_MASK_FILL,
        bbox_padding: int = DEFAULT_BBOX_PADDING,
        min_segment_pixels: int = DEFAULT_MIN_SEGMENT_PIXELS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize_embeddings: bool = DEFAULT_NORMALIZE_EMBEDDINGS,
        sync_buffer_size: int = DEFAULT_SYNC_BUFFER_SIZE,
        publish_empty_sets: bool = DEFAULT_PUBLISH_EMPTY_SETS,
        qos_depth: int = DEFAULT_QOS_DEPTH,
        reliable_input: bool = DEFAULT_RELIABLE_INPUT,
        reliable_output: bool = DEFAULT_RELIABLE_OUTPUT,
    ) -> None:
        super().__init__("clip_inference_node")

        # ============================================================
        # Configuration
        # ============================================================

        self.color_topic = color_topic
        self.segment_topic = segment_topic
        self.embedding_topic = embedding_topic
        self.semantics_topic = semantics_topic

        self.publish_semantics = publish_semantics
        self.prompts = list(prompts) if prompts else list(DEFAULT_PROMPTS)
        self.top_k = int(top_k)
        self.min_score = float(min_score)

        self.log_every = int(log_every)
        self.log_values = int(log_values)
        self.log_max_segments = int(log_max_segments)

        # 프레임 로그용 카운터. build_regions가 필터링 통계를 여기에 남긴다.
        self.frame_count = 0
        self.last_candidate_count = 0
        self.last_skipped_count = 0

        self.backend_name = backend
        self.engine_path = engine_path
        self.pooled_engine_path = pooled_engine_path
        self.value_engine_path = value_engine_path

        if crop_fit not in CROP_FITS:
            raise ValueError(
                f"crop_fit must be one of {CROP_FITS}, got '{crop_fit}'"
            )

        self.crop_fit = crop_fit

        self.text_backend_name = text_backend
        self.text_engine_path = text_engine_path

        self.model_path = model_path
        self.use_cuda = use_cuda

        self.pooling_mode = pooling_mode
        self.patch_weight_gamma = float(patch_weight_gamma)
        self.min_patch_occupancy = float(min_patch_occupancy)
        self.empty_mask_fallback = empty_mask_fallback

        self.debug_save_dir = debug_save_dir
        self.debug_save_every = int(debug_save_every)

        if self.pooling_mode not in POOLING_MODES:
            raise ValueError(
                f"pooling_mode must be one of {POOLING_MODES}, "
                f"got '{self.pooling_mode}'"
            )

        # 이미지 쪽 정렬 행렬은 임베딩 공간을 바꾸므로 embedding_model_id
        # 보다 먼저 읽어야 한다 (ID 에 접미사를 붙일지 여기서 결정된다).
        self.alignment_matrix = self.load_alignment_matrix(alignment_matrix)

        # 텍스트 쪽은 이미지 임베딩을 건드리지 않으므로 ID 가 바뀌지 않는다.
        # 저장된 임베딩은 여전히 순수 pooling 공간이고, 달라지는 것은
        # /clip_semantics 의 라벨뿐이다.
        self.text_alignment_matrix = self.load_text_alignment_matrix(
            text_alignment_matrix
        )

        if (
            self.alignment_matrix is not None
            and self.text_alignment_matrix is not None
        ):
            raise ValueError(
                "alignment_matrix 와 text_alignment_matrix 를 동시에 켜면 "
                "W 가 두 번 적용된다. 하나만 쓰거나 "
                "text_alignment_matrix='none' 으로 끈다"
            )

        # 명시하지 않으면 pooling 방식에서 유도한다.
        self.embedding_model_id = (
            embedding_model_id
            or EMBEDDING_MODEL_IDS[self.pooling_mode]
            + ("_aligned" if self.alignment_matrix is not None else "")
        )

        self.crop_policy = crop_policy
        self.mask_fill = int(mask_fill)
        self.bbox_padding = int(bbox_padding)

        self.min_segment_pixels = int(min_segment_pixels)
        self.batch_size = int(batch_size)
        self.normalize_embeddings = normalize_embeddings

        self.sync_buffer_size = int(sync_buffer_size)
        self.publish_empty_sets = publish_empty_sets

        if self.crop_policy not in CROP_POLICIES:
            raise ValueError(
                f"crop_policy must be one of {CROP_POLICIES}, "
                f"got '{self.crop_policy}'"
            )

        if not 0 <= self.mask_fill <= 255:
            raise ValueError(
                "mask_fill must be between 0 and 255"
            )

        if self.batch_size < 1:
            raise ValueError(
                "batch_size must be at least 1"
            )

        if self.backend_name not in BACKENDS:
            raise ValueError(
                f"backend must be one of {BACKENDS}, "
                f"got '{self.backend_name}'"
            )

        if self.text_backend_name not in BACKENDS:
            raise ValueError(
                f"text_backend must be one of {BACKENDS}, "
                f"got '{self.text_backend_name}'"
            )

        if self.empty_mask_fallback not in EMPTY_MASK_FALLBACKS:
            raise ValueError(
                f"empty_mask_fallback must be one of {EMPTY_MASK_FALLBACKS}, "
                f"got '{self.empty_mask_fallback}'"
            )

        if self.patch_weight_gamma <= 0.0:
            raise ValueError(
                "patch_weight_gamma must be positive, "
                f"got {self.patch_weight_gamma}"
            )

        if not 0.0 <= self.min_patch_occupancy <= 1.0:
            raise ValueError(
                "min_patch_occupancy must be between 0.0 and 1.0, "
                f"got {self.min_patch_occupancy}"
            )

        # pooling 방식마다 쓰는 엔진이 다르다.
        #   cls                 -> engine_path         images -> embeddings
        #   mask_weighted_patch -> pooled_engine_path  images, patch_weights
        #                                              -> embeddings, cls_embeddings
        #   mask_weighted_value -> value_engine_path   위와 입출력이 동일하나
        #                                              마지막 블록이 다르다
        # 뒤의 두 엔진은 입출력 이름이 같아 파일만 봐서는 구분할 수 없다.
        # 그래서 pooling_mode 로 여기서 고르고, 섞이지 않도록 경로를 분리한다.
        #
        # 기본 엔진에는 patch token 출력이 없으므로 pooling 을 할 수 없고,
        # 조용히 CLS 로 되돌리면 embedding_model_id 가 거짓이 되므로
        # 시작 시점에 알아채도록 여기서 확인한다.
        ENGINE_BY_MODE = {
            "cls": self.engine_path,
            "mask_weighted_patch": self.pooled_engine_path,
            "mask_weighted_value": self.value_engine_path,
        }

        ONNX_PART_BY_MODE = {
            "mask_weighted_patch": "visual_pooled",
            "mask_weighted_value": "visual_pooled_value",
        }

        self.active_engine_path = ENGINE_BY_MODE[self.pooling_mode]

        if self.backend_name == "tensorrt" and self.pooling_mode != "cls":
            if not Path(self.active_engine_path).expanduser().is_file():
                part = ONNX_PART_BY_MODE[self.pooling_mode]

                raise FileNotFoundError(
                    f"{self.pooling_mode} pooling 용 TensorRT 엔진이 없습니다: "
                    f"{self.active_engine_path}\n"
                    "다음으로 만들거나\n"
                    "  python3 meridian_clip/export_onnx.py "
                    f"--part {part}\n"
                    "  python3 meridian_clip/build_engine.py "
                    f"--part {part}\n"
                    "--backend torch 로 실행하세요."
                )

        # ============================================================
        # Image encoder backend
        # ============================================================

        # .pt 가 필요한 경우: torch 이미지 백엔드, 또는 torch 텍스트 인코더.
        # 둘 다 tensorrt 면 런타임에 checkpoint 파일이 아예 필요 없다.
        needs_checkpoint = self.backend_name == "torch" or (
            self.publish_semantics and self.text_backend_name == "torch"
        )

        checkpoint = (
            self.resolve_checkpoint()
            if needs_checkpoint
            else ""
        )

        self.get_logger().info(
            f"Loading image encoder backend: {self.backend_name}"
        )

        self.backend = create_backend(
            backend=self.backend_name,
            checkpoint=checkpoint,
            engine_path=self.active_engine_path,
            use_cuda=self.use_cuda,
            crop_fit=self.crop_fit,
        )

        self.device = self.backend.device

        # ViT-B/32 기준 512
        self.embedding_dim = int(self.backend.embedding_dim)

        # 정렬 행렬은 backend 보다 먼저 읽히므로 차원 검증은 여기서 한다.
        if (
            self.alignment_matrix is not None
            and self.alignment_matrix.shape[0] != self.embedding_dim
        ):
            raise ValueError(
                "alignment matrix dim mismatch: "
                f"matrix={self.alignment_matrix.shape[0]}, "
                f"embedding={self.embedding_dim}"
            )

        self.get_logger().info(self.backend.description)
        self.get_logger().info(f"Device: {self.device}")

        # 엔진은 optimization profile 의 max batch 를 넘길 수 없다.
        if self.backend.max_batch and self.batch_size > self.backend.max_batch:
            self.get_logger().warn(
                f"batch_size {self.batch_size} exceeds engine max batch "
                f"{self.backend.max_batch}; clamping. "
                "Rebuild with build_engine.py --max-batch to raise it."
            )
            self.batch_size = int(self.backend.max_batch)

        # ============================================================
        # Patch pooling
        # ============================================================

        # ViT-B/32 면 7. 모델에서 유도하므로 격자가 다른 ViT 도 그대로 된다.
        self.patch_grid = 0

        if self.pooling_mode != "cls":
            grid, patch_count, token_dim = self.backend.patch_geometry()

            self.patch_grid = grid

            # TensorRT 엔진은 pooling 을 그래프 안에서 끝내므로 중간 token
            # 차원을 내보내지 않는다. 그때 token_dim 은 0으로 온다.
            self.get_logger().info(
                f"Patch grid: {grid}x{grid} ({patch_count} tokens"
                + (f", dim={token_dim})" if token_dim else ")")
            )

        # crop/mask/occupancy 를 같은 좌표계로 저장하기 위한 기하.
        # 실제 전처리와 같은 build_geometry() 를 쓰므로 어긋날 수 없다.
        self.debug_rgb_geometry = None
        self.debug_mask_geometry = None

        if self.debug_save_dir:
            Path(self.debug_save_dir).expanduser().mkdir(
                parents=True,
                exist_ok=True,
            )

            (
                self.debug_rgb_geometry,
                self.debug_mask_geometry,
            ) = build_debug_geometry(
                int(self.backend.resolution),
                self.crop_fit,
            )

            self.get_logger().info(
                f"Debug images: {self.debug_save_dir} "
                f"(every {self.debug_save_every} frames)"
            )

        # ============================================================
        # Text encoder (zero-shot semantics)
        # ============================================================

        self.text_encoder = None

        if self.publish_semantics:
            self.get_logger().info(
                f"Loading text encoder backend: {self.text_backend_name}"
            )

            self.text_encoder = create_text_encoder(
                backend=self.text_backend_name,
                checkpoint=checkpoint,
                engine_path=self.text_engine_path,
                prompts=self.prompts,
                use_cuda=self.use_cuda,
            )

            self.get_logger().info(self.text_encoder.description)

            if self.text_encoder.embedding_dim != self.embedding_dim:
                raise ValueError(
                    "image/text embedding dim mismatch: "
                    f"image={self.embedding_dim}, "
                    f"text={self.text_encoder.embedding_dim}"
                )

            # 프롬프트는 생성자에서 한 번만 인코딩되므로 변환도 여기서
            # 한 번이면 된다. 프레임마다 도는 경로에는 아무 비용도 없다.
            self.apply_text_alignment()

            self.top_k = max(1, min(self.top_k, len(self.prompts)))

            self.get_logger().info(
                f"Prompts: {len(self.prompts)} "
                f"(top_k={self.top_k}, min_score={self.min_score})"
            )

        # ============================================================
        # Frame synchronization buffers
        # ============================================================

        # capture timestamp -> message
        self.color_buffer: "OrderedDict[Tuple[int, int], RosImage]" = (
            OrderedDict()
        )

        # label map도 color와 같은 sensor_msgs/Image (mono8)다.
        self.segment_buffer: "OrderedDict[Tuple[int, int], RosImage]" = (
            OrderedDict()
        )

        # ============================================================
        # ROS2 interfaces
        # ============================================================

        self.bridge = CvBridge()

        sub_qos = build_qos(reliable_input, qos_depth)
        pub_qos = build_qos(reliable_output, qos_depth)

        self.embedding_publisher = self.create_publisher(
            InstanceEmbeddingSet,
            self.embedding_topic,
            pub_qos,
        )

        # 시각화/디버깅이 CLIP 을 따로 로드하지 않아도 되도록,
        # 세그먼트별 zero-shot 결과를 노드가 직접 발행한다.
        self.semantics_publisher = (
            self.create_publisher(
                Detection2DArray,
                self.semantics_topic,
                pub_qos,
            )
            if self.publish_semantics
            else None
        )

        self.color_subscription = self.create_subscription(
            RosImage,
            self.color_topic,
            self.color_callback,
            sub_qos,
        )

        # 계약: label map은 sensor_msgs/Image (mono8), 픽셀 값 = segment_id
        self.segment_subscription = self.create_subscription(
            RosImage,
            self.segment_topic,
            self.segment_callback,
            sub_qos,
        )

        self.get_logger().info(
            f"Subscribed color image: {self.color_topic}"
        )

        self.get_logger().info(
            f"Subscribed segment label image (mono8): {self.segment_topic}"
        )

        self.get_logger().info(
            "QoS: sub="
            f"{'reliable' if reliable_input else 'best_effort'}, "
            f"pub={'reliable' if reliable_output else 'best_effort'}, "
            f"depth={qos_depth}"
        )

        self.get_logger().info(
            f"Publishing InstanceEmbeddingSet: {self.embedding_topic}"
        )

        if self.publish_semantics:
            self.get_logger().info(
                f"Publishing Detection2DArray: {self.semantics_topic}"
            )

        self.get_logger().info(
            f"Embedding model ID: {self.embedding_model_id}"
        )

        self.get_logger().info(
            f"Embedding dim: {self.embedding_dim}"
        )

        self.get_logger().info(
            f"Crop policy: {self.crop_policy} (fit={self.crop_fit})"
        )

        self.get_logger().info(
            f"Pooling mode: {self.pooling_mode}"
        )

        self.get_logger().info(
            "Alignment matrix (image side): "
            + (
                f"{self.alignment_matrix_path} "
                f"({self.alignment_matrix.shape[0]}x"
                f"{self.alignment_matrix.shape[1]})"
                if self.alignment_matrix is not None
                else "none"
            )
        )

        self.get_logger().info(
            "Alignment matrix (text side, Wᵀ): "
            + (
                f"{self.text_alignment_matrix_path} "
                f"({self.text_alignment_matrix.shape[0]}x"
                f"{self.text_alignment_matrix.shape[1]})"
                if self.text_alignment_matrix is not None
                else "none"
            )
        )

        if self.pooling_mode != "cls":
            self.get_logger().info(
                f"Patch gamma: {self.patch_weight_gamma}"
            )

            self.get_logger().info(
                f"Min patch occupancy: {self.min_patch_occupancy}"
            )

            self.get_logger().info(
                f"Empty mask fallback: {self.empty_mask_fallback}"
            )

    # ================================================================
    # Model
    # ================================================================

    def resolve_checkpoint(self) -> str:
        """clip.load에 넘길 로컬 checkpoint 경로를 확인한다.

        torch 이미지 백엔드와 텍스트 인코더가 이 파일을 쓴다.
        """
        if not self.model_path:
            raise ValueError(
                "--model-path 가 필요합니다 "
                "(--backend torch 또는 --publish-semantics 를 쓰는 경우)."
            )

        path = Path(self.model_path).expanduser()

        if not path.is_file():
            raise FileNotFoundError(
                f"CLIP weight file not found: {path}\n"
                "Run 'python3 meridian_clip/download_weights.py' first."
            )

        return str(path)

    # ================================================================
    # Frame synchronization
    # ================================================================

    @staticmethod
    def timestamp_key(
        stamp: Time,
    ) -> Tuple[int, int]:
        """Capture timestamp를 dict 키로 쓸 수 있는 형태로 바꾼다."""
        return (
            int(stamp.sec),
            int(stamp.nanosec),
        )

    def color_callback(
        self,
        msg: RosImage,
    ) -> None:
        key = self.timestamp_key(msg.header.stamp)

        self.color_buffer[key] = msg
        self.evict_stale(self.color_buffer)

        self.try_process(key)

    def segment_callback(
        self,
        msg: RosImage,
    ) -> None:
        # fastsam_node가 color frame의 header를 그대로 물려주므로
        # color와 같은 (sec, nanosec) 키가 나온다.
        key = self.timestamp_key(msg.header.stamp)

        self.segment_buffer[key] = msg
        self.evict_stale(self.segment_buffer)

        self.try_process(key)

    def evict_stale(
        self,
        buffer: OrderedDict,
    ) -> None:
        """짝을 찾지 못한 오래된 frame을 버린다."""
        while len(buffer) > self.sync_buffer_size:
            buffer.popitem(last=False)

    def try_process(
        self,
        key: Tuple[int, int],
    ) -> None:
        """같은 capture timestamp의 두 입력이 모두 도착하면 처리한다."""
        if key not in self.color_buffer:
            return

        if key not in self.segment_buffer:
            return

        color_msg = self.color_buffer.pop(key)
        segment_msg = self.segment_buffer.pop(key)

        try:
            self.process_frame(
                color_msg=color_msg,
                segment_msg=segment_msg,
            )

        except Exception as error:
            self.get_logger().error(
                f"CLIP frame processing failed: {error}"
            )

    # ================================================================
    # Processing
    # ================================================================

    def process_frame(
        self,
        color_msg: RosImage,
        segment_msg: RosImage,
    ) -> None:
        """한 frame의 모든 positive segment에 embedding을 부여한다."""
        rgb_image = self.bridge.imgmsg_to_cv2(
            color_msg,
            desired_encoding="rgb8",
        )

        labels = self.bridge.imgmsg_to_cv2(
            segment_msg,
            desired_encoding="mono8",
        )

        if rgb_image.shape[:2] != labels.shape[:2]:
            self.get_logger().error(
                "RGB and segment label image resolution mismatch: "
                f"rgb={rgb_image.shape[:2]}, "
                f"labels={labels.shape[:2]}"
            )
            return

        segment_ids, regions, masks, boxes = self.build_regions(
            rgb_image=rgb_image,
            labels=labels,
        )

        if not segment_ids:
            if self.publish_empty_sets:
                empty = np.zeros(
                    (0, self.embedding_dim),
                    dtype=np.float32,
                )

                self.publish_embeddings(
                    timestamp=color_msg.header.stamp,
                    segment_ids=[],
                    embeddings=empty,
                )

                self.publish_semantic_detections(
                    header=color_msg.header,
                    segment_ids=[],
                    boxes=[],
                    embeddings=empty,
                )

            return

        embeddings = self.encode_regions(regions, masks)

        # 가중평균 pooling 경로에서만 채워진다. cls 경로는 None.
        stats = self.backend.last_pooling_stats

        if stats is not None and not bool(stats.keep.all()):
            keep = stats.keep

            self.get_logger().warn(
                f"Dropping {int((~keep).sum())} segment(s) with empty patch "
                "occupancy (--empty-mask-fallback skip)"
            )

            segment_ids = [
                value
                for value, flag in zip(segment_ids, keep)
                if flag
            ]
            regions = [
                value
                for value, flag in zip(regions, keep)
                if flag
            ]
            masks = [
                value
                for value, flag in zip(masks, keep)
                if flag
            ]
            boxes = [
                value
                for value, flag in zip(boxes, keep)
                if flag
            ]

            embeddings = embeddings[keep]
            stats = stats.select(keep)

        self.log_frame(
            timestamp=color_msg.header.stamp,
            segment_ids=segment_ids,
            embeddings=embeddings,
            stats=stats,
        )

        self.save_debug_images(
            segment_ids=segment_ids,
            regions=regions,
            masks=masks,
            stats=stats,
        )

        self.publish_embeddings(
            timestamp=color_msg.header.stamp,
            segment_ids=segment_ids,
            embeddings=embeddings,
        )

        self.publish_semantic_detections(
            header=color_msg.header,
            segment_ids=segment_ids,
            boxes=boxes,
            embeddings=embeddings,
        )

    def build_regions(
        self,
        rgb_image: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[
        List[int],
        List[PILImage.Image],
        List[PILImage.Image],
        List[Tuple[int, int, int, int]],
    ]:
        """
        각 unique positive segment_id마다 CLIP 입력 region을 만든다.

        반환되는 segment_ids / regions / masks / boxes는 같은 순서를 유지하며,
        이 순서가 InstanceEmbeddingSet의 row 순서가 된다.
        masks[i]는 regions[i]와 정확히 같은 크기의 binary mask이며
        가중평균 pooling의 패치 점유율 계산에 쓴다.
        boxes는 마스크의 tight bbox (x0, y0, x1, y1)이며 semantics 발행에 쓴다.
        """
        unique_ids = np.unique(labels)

        segment_ids: List[int] = []
        regions: List[PILImage.Image] = []
        masks: List[PILImage.Image] = []
        boxes: List[Tuple[int, int, int, int]] = []

        candidates = 0
        skipped = 0

        for raw_id in unique_ids:
            segment_id = int(raw_id)

            # 0은 background/invalid
            if segment_id == BACKGROUND_SEGMENT_ID:
                continue

            candidates += 1

            mask = labels == segment_id

            if int(mask.sum()) < self.min_segment_pixels:
                skipped += 1
                continue

            bounds = self.mask_bounds(mask)

            if bounds is None:
                continue

            built = self.build_region(
                rgb_image=rgb_image,
                mask=mask,
                bounds=bounds,
            )

            if built is None:
                continue

            region, region_mask = built

            segment_ids.append(segment_id)
            regions.append(region)
            masks.append(region_mask)
            boxes.append(bounds)

        self.last_candidate_count = candidates
        self.last_skipped_count = skipped

        return segment_ids, regions, masks, boxes

    @staticmethod
    def mask_bounds(
        mask: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        """마스크를 감싸는 tight bbox (x0, y0, x1, y1)을 구한다."""
        rows = np.nonzero(mask.any(axis=1))[0]
        columns = np.nonzero(mask.any(axis=0))[0]

        if rows.size == 0 or columns.size == 0:
            return None

        return (
            int(columns[0]),
            int(rows[0]),
            int(columns[-1]) + 1,
            int(rows[-1]) + 1,
        )

    @staticmethod
    def mask_to_image(mask: np.ndarray) -> PILImage.Image:
        """마스크 배열을 백엔드가 받는 grayscale 이미지로 바꾼다."""
        return PILImage.fromarray(
            mask.astype(np.uint8) * 255
        )

    def build_region(
        self,
        rgb_image: np.ndarray,
        mask: np.ndarray,
        bounds: Tuple[int, int, int, int],
    ) -> Optional[Tuple[PILImage.Image, PILImage.Image]]:
        """crop_policy에 따라 segment 하나의 (RGB region, mask)를 구성한다.

        두 이미지는 항상 같은 크기다. 백엔드가 여기에 같은 resize/crop을
        적용하므로, 크기가 어긋나면 패치 점유율과 patch token의 위치 대응이
        깨진다.
        """
        if self.crop_policy == "masked_full":
            region = rgb_image.copy()
            region[~mask] = self.mask_fill

            return (
                PILImage.fromarray(region),
                self.mask_to_image(mask),
            )

        height, width = mask.shape[:2]

        tight_x0, tight_y0, tight_x1, tight_y1 = bounds

        y0 = max(tight_y0 - self.bbox_padding, 0)
        y1 = min(tight_y1 + self.bbox_padding, height)

        x0 = max(tight_x0 - self.bbox_padding, 0)
        x1 = min(tight_x1 + self.bbox_padding, width)

        region = rgb_image[y0:y1, x0:x1].copy()
        region_mask = mask[y0:y1, x0:x1]

        if self.crop_policy == "masked_bbox":
            # bbox 안에서도 다른 object의 픽셀은 배제한다.
            region[~region_mask] = self.mask_fill

        return (
            PILImage.fromarray(region),
            self.mask_to_image(region_mask),
        )

    def encode_regions(
        self,
        regions: List[PILImage.Image],
        masks: Optional[List[PILImage.Image]] = None,
    ) -> np.ndarray:
        """
        region들을 batch로 CLIP image encoder에 통과시킨다.

        반환 shape은 [N, embedding_dim] 이며 dtype은 float32이다.
        배치 분할과 정규화는 백엔드가 담당한다.
        pooling_mode가 cls면 masks는 쓰이지 않는다.

        alignment_matrix가 있으면 백엔드 출력에 곱하고 다시 정규화한다.
        백엔드 출력이 이미 단위벡터인 지점이라 torch/tensorrt 어느 쪽이든
        같은 값이 나온다.
        """
        embeddings = self.backend.encode(
            regions=regions,
            batch_size=self.batch_size,
            normalize=self.normalize_embeddings,
            masks=masks,
            pooling_mode=self.pooling_mode,
            gamma=self.patch_weight_gamma,
            min_patch_occupancy=self.min_patch_occupancy,
            empty_mask_fallback=self.empty_mask_fallback,
        )

        return self.apply_alignment(embeddings)

    def apply_alignment(self, embeddings: np.ndarray) -> np.ndarray:
        """정렬 행렬을 곱하고 L2 정규화한다. 행렬이 없으면 그대로 돌려준다."""
        if self.alignment_matrix is None or embeddings.shape[0] == 0:
            return embeddings

        aligned = embeddings.astype(np.float32) @ self.alignment_matrix

        return l2_normalize(aligned) if self.normalize_embeddings else aligned

    def load_text_alignment_matrix(self, path: str) -> Optional[np.ndarray]:
        """텍스트 쪽 정렬 행렬을 읽는다.

        ""     -> pooling 모드에서 자동 선택 (TEXT_ALIGNMENT_MATRICES)
        "none" -> 쓰지 않음
        그 외  -> 그 경로

        파일이 없을 때 동작이 두 갈래인 이유: 자동 선택은 '있으면 쓴다'는
        뜻이므로 경고만 하고 넘어간다. 그렇지 않으면 행렬을 만드는 도구
        (tools/fit_alignment.py)가 행렬이 없어서 못 도는 순환이 생기고,
        새로 받은 작업공간에서 노드가 아예 안 뜬다. 반대로 사용자가 경로를
        직접 지정했다면 그건 요구사항이므로 조용히 무시하지 않고 실패한다.
        """
        self.text_alignment_matrix_path = ""

        if path == "none":
            return None

        resolved = path or TEXT_ALIGNMENT_MATRICES.get(self.pooling_mode, "")

        if not resolved:
            return None

        target = Path(resolved).expanduser()

        if not target.exists():
            if path:
                raise FileNotFoundError(
                    f"text alignment matrix not found: {target}"
                )

            self.get_logger().warning(
                f"text alignment matrix not found: {target} -- "
                "정렬 없이 진행합니다. zero-shot 라벨 정확도가 낮아집니다 "
                "(VOC2012 val 기준 90.61% -> 87.98%). "
                "tools/fit_alignment.py 로 만들 수 있습니다"
            )

            return None

        matrix = np.load(target).astype(np.float32)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "text alignment matrix must be square [D, D], "
                f"got {tuple(matrix.shape)} from {target}"
            )

        self.text_alignment_matrix_path = str(target)

        return matrix

    def apply_text_alignment(self) -> None:
        """텍스트 임베딩 행렬 [M, D] 에 Wᵀ 를 곱한다.

        (e W)·t = e·(t Wᵀ) 이므로 이미지 쪽 변환과 같은 내적을 주면서
        이미지 임베딩은 원본 공간에 남는다.

        **재정규화하지 않는다.** Wᵀ 를 지나면 프롬프트마다 길이가 달라지는데
        그 차이가 정답 신호의 일부다 (정규화하면 top-1 84.56% -> 40.56%).
        """
        if self.text_alignment_matrix is None or self.text_encoder is None:
            return

        if self.text_alignment_matrix.shape[0] != self.text_encoder.embedding_dim:
            raise ValueError(
                "text alignment matrix dim mismatch: "
                f"matrix={self.text_alignment_matrix.shape[0]}, "
                f"text={self.text_encoder.embedding_dim}"
            )

        self.text_encoder.matrix = (
            self.text_encoder.matrix @ self.text_alignment_matrix.T
        ).astype(np.float32)

    def load_alignment_matrix(self, path: str) -> Optional[np.ndarray]:
        """정렬 행렬 .npy 를 읽는다. path 가 비면 None.

        모양이 틀리면 임베딩이 조용히 망가지는 대신 시작할 때 바로 실패한다.
        """
        self.alignment_matrix_path = ""

        if not path:
            return None

        resolved = Path(path).expanduser()

        if not resolved.exists():
            raise FileNotFoundError(f"alignment matrix not found: {resolved}")

        matrix = np.load(resolved).astype(np.float32)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "alignment matrix must be square [D, D], "
                f"got {tuple(matrix.shape)} from {resolved}"
            )

        self.alignment_matrix_path = str(resolved)

        return matrix

    # ================================================================
    # Logging
    # ================================================================

    def log_frame(
        self,
        timestamp: Time,
        segment_ids: List[int],
        embeddings: np.ndarray,
        stats=None,
    ) -> None:
        """한 프레임의 마스크 수와 임베딩을 터미널에 찍는다.

        --log-every 0 이면 아무것도 찍지 않는다. 발행과는 무관한 관찰용이다.
        stats가 있으면 세그먼트별 패치 점유율도 함께 찍는다.
        """
        self.frame_count += 1

        if self.log_every <= 0:
            return

        if self.frame_count % self.log_every != 0:
            return

        stamp = f"{timestamp.sec}.{timestamp.nanosec:09d}"

        lines = [
            f"frame={self.frame_count}  stamp={stamp}  "
            f"masks={len(segment_ids)}  dim={self.embedding_dim}  "
            f"(candidates={self.last_candidate_count}, "
            f"skipped_small={self.last_skipped_count})"
        ]

        if self.log_values > 0 and len(segment_ids):
            shown = min(len(segment_ids), self.log_max_segments)
            width = min(self.log_values, self.embedding_dim)

            norms = np.linalg.norm(embeddings, axis=1)

            for row in range(shown):
                values = " ".join(
                    f"{value:+.4f}"
                    for value in embeddings[row][:width]
                )

                tail = " ..." if width < self.embedding_dim else ""

                lines.append(
                    f"  segment {segment_ids[row]:<4} "
                    f"L2={norms[row]:.4f}  [{values}{tail}]"
                )

                if stats is not None:
                    occupancy = stats.occupancy[row]

                    note = (
                        "  (CLS fallback)"
                        if bool(stats.fallback[row])
                        else ""
                    )

                    lines.append(
                        f"    occupancy mean={occupancy.mean():.3f} "
                        f"max={occupancy.max():.3f}  "
                        f"active={int(stats.active_patches[row])}/"
                        f"{occupancy.size}  "
                        f"weights sum={stats.weight_sum[row]:.2f}{note}"
                    )

            if len(segment_ids) > shown:
                lines.append(
                    f"  ... 외 {len(segment_ids) - shown}개 세그먼트"
                )

        self.get_logger().info("\n".join(lines))

    def save_debug_images(
        self,
        segment_ids: List[int],
        regions: List[PILImage.Image],
        masks: List[PILImage.Image],
        stats=None,
    ) -> None:
        """이미지 / 마스크 / 점유율 세 장을 같은 좌표계로 저장한다.

        임베딩만 봐서는 RGB와 마스크의 공간 정렬이 깨졌는지 알 수 없다.
        정렬이 어긋나도 임베딩은 정상적으로 나오고 L2도 1이기 때문이다.
        세 장을 겹쳐보면 밝은 패치가 실제 객체 위치와 맞는지 바로 보인다.
        """
        if stats is None or self.debug_rgb_geometry is None:
            return

        if self.debug_save_every <= 0:
            return

        if self.frame_count % self.debug_save_every != 0:
            return

        directory = Path(self.debug_save_dir).expanduser()

        for row, segment_id in enumerate(segment_ids):
            prefix = (
                f"frame_{self.frame_count:06d}_"
                f"segment_{segment_id:03d}"
            )

            crop = self.debug_rgb_geometry(regions[row])
            crop.save(directory / f"{prefix}_crop.png")

            mask_image = self.debug_mask_geometry(masks[row])
            mask_image.save(directory / f"{prefix}_mask.png")

            # 7x7 점유율을 crop과 같은 크기로 확대해 겹쳐볼 수 있게 한다.
            occupancy = stats.occupancy[row].reshape(
                self.patch_grid,
                self.patch_grid,
            )

            occupancy_image = PILImage.fromarray(
                (occupancy * 255.0).clip(0.0, 255.0).astype(np.uint8)
            ).resize(
                mask_image.size,
                PILImage.Resampling.NEAREST,
            )

            occupancy_image.save(directory / f"{prefix}_occupancy.png")

    # ================================================================
    # Output
    # ================================================================

    def publish_embeddings(
        self,
        timestamp: Time,
        segment_ids: List[int],
        embeddings: np.ndarray,
    ) -> None:
        """InstanceEmbeddingSet을 frame 단위로 발행한다."""
        expected_shape = (
            len(segment_ids),
            self.embedding_dim,
        )

        if embeddings.shape != expected_shape:
            self.get_logger().error(
                "Invalid embedding matrix shape: "
                f"expected={expected_shape}, "
                f"actual={embeddings.shape}"
            )
            return

        # segment_ids는 uint8[]이다. 범위를 벗어난 값을 그냥 대입하면
        # rclpy 내부 assertion으로 떨어져 원인을 알기 어려우므로 여기서 막는다.
        out_of_range = [
            value
            for value in segment_ids
            if not 1 <= value <= MAX_SEGMENT_ID
        ]

        if out_of_range:
            self.get_logger().error(
                "segment_id out of uint8 range 1..255: "
                f"{out_of_range[:8]}"
            )
            return

        output_msg = InstanceEmbeddingSet()

        # 원본 color image의 capture time을 그대로 전달한다.
        # 이 값과 segment_id의 쌍이 downstream의 observation key가 된다.
        output_msg.timestamp = timestamp

        output_msg.embedding_model_id = self.embedding_model_id

        output_msg.segment_ids = segment_ids

        output_msg.embedding_dim = self.embedding_dim

        # row-major flattened [N, D]
        output_msg.embeddings = (
            embeddings.reshape(-1).tolist()
        )

        self.embedding_publisher.publish(
            output_msg
        )

        self.get_logger().debug(
            "Published InstanceEmbeddingSet: "
            f"segments={len(segment_ids)}, "
            f"dim={self.embedding_dim}"
        )

    def publish_semantic_detections(
        self,
        header,
        segment_ids: List[int],
        boxes: List[Tuple[int, int, int, int]],
        embeddings: np.ndarray,
    ) -> None:
        """세그먼트별 zero-shot 결과를 Detection2DArray로 발행한다.

        노드가 텍스트 인코더를 들고 있으므로 소비자(시각화 등)는 CLIP을
        로드할 필요가 없다. 프레임마다 도는 것은 행렬곱 하나뿐이다.
        """
        if self.semantics_publisher is None or self.text_encoder is None:
            return

        output_msg = Detection2DArray()

        # color frame의 header를 그대로 물려주어 stamp와 frame_id를 보존한다.
        output_msg.header = header

        if segment_ids:
            # 코사인 유사도이므로 단위벡터여야 한다.
            matrix = (
                embeddings
                if self.normalize_embeddings
                else l2_normalize(embeddings)
            )

            similarity = self.text_encoder.similarity(matrix)

            ranking = np.argsort(-similarity, axis=1)[:, :self.top_k]

            for row, segment_id in enumerate(segment_ids):
                best_score = float(similarity[row][ranking[row][0]])

                if best_score < self.min_score:
                    continue

                detection = Detection2D()
                detection.header = header
                detection.id = str(segment_id)

                x0, y0, x1, y1 = boxes[row]

                bbox = BoundingBox2D()
                bbox.center.position.x = float(x0 + x1) / 2.0
                bbox.center.position.y = float(y0 + y1) / 2.0
                bbox.center.theta = 0.0
                bbox.size_x = float(x1 - x0)
                bbox.size_y = float(y1 - y0)

                detection.bbox = bbox

                for index in ranking[row]:
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = self.prompts[int(index)]
                    hypothesis.hypothesis.score = float(
                        similarity[row][int(index)]
                    )

                    detection.results.append(hypothesis)

                output_msg.detections.append(detection)

        self.semantics_publisher.publish(output_msg)


def load_prompts(path: str) -> List[str]:
    """한 줄에 하나씩 적힌 프롬프트 파일을 읽는다. 빈 줄과 #주석은 건너뛴다."""
    lines = Path(path).expanduser().read_text(encoding="utf-8").splitlines()

    prompts = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]

    if not prompts:
        raise ValueError(f"prompt file is empty: {path}")

    return prompts


def main(args=None) -> None:
    rclpy.init(args=args)

    # ros2 run/launch 가 붙이는 --ros-args 부분을 걷어내고 우리 인자만 파싱
    argv = remove_ros_args(args=sys.argv)
    cli = build_parser().parse_args(argv[1:])

    prompts = (
        load_prompts(cli.prompt_file)
        if cli.prompt_file
        else cli.prompts
    )

    node = ClipInferenceNode(
        color_topic=cli.color_topic,
        segment_topic=cli.segment_topic,
        embedding_topic=cli.embedding_topic,
        semantics_topic=cli.semantics_topic,
        publish_semantics=cli.publish_semantics,
        prompts=prompts,
        top_k=cli.top_k,
        min_score=cli.min_score,
        log_every=cli.log_every,
        log_values=cli.log_values,
        log_max_segments=cli.log_max_segments,
        backend=cli.backend,
        engine_path=cli.engine_path,
        pooled_engine_path=cli.pooled_engine_path,
        value_engine_path=cli.value_engine_path,
        crop_fit=cli.crop_fit,
        text_backend=cli.text_backend,
        text_engine_path=cli.text_engine_path,
        model_path=cli.model_path,
        embedding_model_id=cli.embedding_model_id,
        alignment_matrix=cli.alignment_matrix,
        text_alignment_matrix=cli.text_alignment_matrix,
        pooling_mode=cli.pooling_mode,
        patch_weight_gamma=cli.patch_weight_gamma,
        min_patch_occupancy=cli.min_patch_occupancy,
        empty_mask_fallback=cli.empty_mask_fallback,
        debug_save_dir=cli.debug_save_dir,
        debug_save_every=cli.debug_save_every,
        use_cuda=cli.use_cuda,
        crop_policy=cli.crop_policy,
        mask_fill=cli.mask_fill,
        bbox_padding=cli.bbox_padding,
        min_segment_pixels=cli.min_segment_pixels,
        batch_size=cli.batch_size,
        normalize_embeddings=cli.normalize_embeddings,
        sync_buffer_size=cli.sync_buffer_size,
        publish_empty_sets=cli.publish_empty_sets,
        qos_depth=cli.qos_depth,
        reliable_input=cli.reliable_input,
        reliable_output=cli.reliable_output,
    )

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
