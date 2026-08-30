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
import queue
import sys
import threading
import time

from array import array
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import rclpy

from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from meridian_msgs.msg import InstanceEmbeddingSet
from PIL import Image as PILImage
from scipy import ndimage
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
    DEFAULT_ASYNC_PREPROCESS,
    DEFAULT_PREPROCESS_PATH,
    PREPROCESS_PATHS,
    PREPROCESS_WORKERS,
    EMPTY_MASK_FALLBACKS,
    POOLING_MODES,
    build_debug_geometry,
    create_backend,
    create_text_encoder,
    l2_normalize,
    limit_blas_threads,
    prepare_from_frame,
)
from meridian_clip.model_paths import (
    MODEL_DIR_ENV_VAR,
    default_model_dir,
    describe_candidates,
)


# CPU 전처리(1단계)와 GPU 추론(2단계)을 다른 스레드에서 겹쳐 돌린다.
# 처리량은 오르지만 프레임 하나의 지연은 줄지 않는다 (오히려 큐 대기만큼
# 늘어난다). 지연이 중요한 소비자가 있으면 꺼 둔다.
DEFAULT_PIPELINE_ENABLED = False

# 준비된 프레임을 담아 두는 큐 깊이. 가득 차면 가장 오래된 것을 버리고
# 최신 프레임을 넣는다 (최신 우선). 깊이를 키워도 처리량은 안 오르고
# 지연만 늘어난다.
DEFAULT_PIPELINE_QUEUE_DEPTH = 2

# N 프레임마다 단계별 소요시간을 한 줄로 찍는다. 0 이면 끈다.
DEFAULT_STATS_EVERY = 0


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

# zero-shot semantics(Detection2DArray)를 함께 발행할지.
#
# 기본값이 끔인 이유는 비용이다. 세그먼트 32개 기준 프레임당 5.1ms 로
# Postprocessing 의 88% 를 차지하는데, 그중 유사도 행렬곱은 0.07ms 뿐이고
# 나머지는 Detection2D 32개를 파이썬 객체로 만드는 값이다. 끄면 텍스트
# 인코더(126MB 엔진)도 로드하지 않는다.
#
# 실측 (N=32, mask_weighted_value):
#   순차     22.9 -> 26.0 FPS
#   파이프라인 36.1 -> 45.1 FPS
#
# 소비자가 있으면 켠다:
#   ros2 launch ... publish_semantics:=true
DEFAULT_PUBLISH_SEMANTICS = False

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

# 모델 바이너리(.pt/.onnx/.engine)가 있는 디렉터리.
#
# 절대경로를 소스에 박지 않는다 -- 머신마다 워크스페이스 이름도 사용자 이름도
# 다르다. 찾는 규칙은 model_paths.py 에 한 벌만 두고 launch 와 공유한다
# (환경변수 -> install share -> 소스 트리 순). meridian_seg 의
# engine_candidates() 와 같은 규약이다.
PACKAGE_MODEL_DIR = default_model_dir()

# 플랫폼마다 모델 파일의 디렉터리만 다르게 줄 수 있다. 빈 문자열이면 아래
# 개별 경로 기본값을 그대로 쓴다. --model-dir 를 주면 engine/checkpoint와
# 자동 선택되는 text alignment matrix의 디렉터리를 이 값으로 통일한다.
DEFAULT_MODEL_DIR = ""

# 하드웨어에 따라 달라지는 성능 파라미터. 알고리즘에는 영향을 주지 않는다.
DEFAULT_PREPROCESS_WORKERS = PREPROCESS_WORKERS
DEFAULT_ASYNC_PREPROCESS_ENABLED = DEFAULT_ASYNC_PREPROCESS

DEFAULT_ENGINE_PATH = str(
    Path(PACKAGE_MODEL_DIR) / "clip_vit_b32_visual_fp16.engine"
)

# mask_weighted_patch 용 엔진. weighted mean 과 ln_post/proj 까지 그래프에 들어
# 있고 (images, patch_weights) -> (embeddings, cls_embeddings) 형태다.
#   python3 meridian_clip/export_onnx.py  --part visual_pooled
#   python3 meridian_clip/build_engine.py --part visual_pooled
DEFAULT_POOLED_ENGINE_PATH = str(
    Path(PACKAGE_MODEL_DIR) / "clip_vit_b32_visual_pooled_fp16.engine"
)

# mask_weighted_value 용 엔진. 입출력은 위와 똑같고 마지막 블록에서 value
# 투영을 patch feature 로 쓰는 것만 다르다. 이름으로 구분할 수 없어 별도
# 경로로 둔다.
#   python3 meridian_clip/export_onnx.py  --part visual_pooled_value
#   python3 meridian_clip/build_engine.py --part visual_pooled_value
DEFAULT_VALUE_ENGINE_PATH = str(
    Path(PACKAGE_MODEL_DIR) / "clip_vit_b32_visual_pooled_value_fp16.engine"
)

# 텍스트 인코더도 같은 방식으로 고를 수 있다.
#   tensorrt : models/clip_vit_b32_text_fp16.engine  (.pt 불필요)
#   torch    : models/ViT-B-32.pt
DEFAULT_TEXT_BACKEND = "tensorrt"

DEFAULT_TEXT_ENGINE_PATH = str(
    Path(PACKAGE_MODEL_DIR) / "clip_vit_b32_text_fp16.engine"
)

# --backend torch 일 때 로드할 로컬 checkpoint(.pt) 경로.
# models/ 는 install 트리에 복사하지 않으므로 소스 경로를 가리킨다.
DEFAULT_MODEL_PATH = str(Path(PACKAGE_MODEL_DIR) / "ViT-B-32.pt")

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
    "mask_weighted_patch": str(
        Path(PACKAGE_MODEL_DIR) / "align_patch_to_cls.npy"
    ),
    "mask_weighted_value": str(
        Path(PACKAGE_MODEL_DIR) / "align_value_to_cls.npy"
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

# 한 번에 encode_image에 넣을 segment 수.
#
# **엔진 프로파일의 opt 와 같아야 한다.** TensorRT 는 opt 배치에 맞춰 커널을
# 고르므로, 노드가 다른 크기를 넣으면 튜닝되지 않은 지점에서 돌게 된다.
# models/ 의 엔진은 min=1 / opt=32 / max=64 로 빌드되어 있다:
#   python3 meridian_clip/build_engine.py --part visual_pooled_value \
#       --min-batch 1 --opt-batch 32 --max-batch 64
#
# 엔진의 max 를 넘기면 노드가 경고와 함께 잘라낸다. 성능을 잴 때는 이 값과
# 엔진 opt 가 일치하는지부터 확인한다 (README §6).
DEFAULT_BATCH_SIZE = 32

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


def parse_bool(value: str) -> bool:
    """launch 가 넘기는 "true"/"false" 문자열을 bool 로 바꾼다."""
    lowered = str(value).strip().lower()

    if lowered in ("1", "true", "yes", "on"):
        return True

    if lowered in ("0", "false", "no", "off"):
        return False

    raise argparse.ArgumentTypeError(
        f"expected true/false, got '{value}'"
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

    # 값을 받는 형태로 둔다. launch 가 인자를 항상 "--이름 값" 쌍으로
    # 넘기므로 BooleanOptionalAction 은 launch 에서 쓸 수 없다.
    # semantics 발행은 프레임당 5.1ms 라 끄고 쓰는 배포가 흔하다
    # (세그먼트 32개 기준 Postprocessing 의 88%).
    parser.add_argument(
        "--publish-semantics",
        type=parse_bool,
        default=DEFAULT_PUBLISH_SEMANTICS,
        metavar="true|false",
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
        "--min-segment-area",
        type=int,
        default=0,
        help=(
            "이 픽셀 수(라벨 이미지 = RGB 해상도 기준) 미만인 세그먼트는 "
            "인코딩하지 않는다. 0이면 끔. enc 시간이 세그먼트 수 N 에 거의 "
            "정비례하므로(실측 enc ~ 1.0ms x N) 잔챙이를 거르면 그만큼 준다. "
            "SAM 의 --area-min 과 달리 /segment_image 는 건드리지 않아서 "
            "geobuilder 는 작은 물체까지 그대로 3D 로 복원한다"
        ),
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=0,
        help=(
            "면적 큰 것부터 이 개수까지만 인코딩한다. 0이면 끔. enc 시간에 "
            "상한을 걸어 장면이 복잡해져도 프레임 시간이 튀지 않게 한다"
        ),
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
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=(
            "모델/엔진 디렉터리. 비우면 기존 개별 경로를 사용한다. "
            "지정하면 engine/model/text engine과 자동 alignment 파일의 "
            "디렉터리를 이 값으로 통일한다"
        ),
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
            "가중평균 (기본값. VOC2012 val 에서 AUC 0.9897 / top-1 87.98%%), "
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
        "--preprocess-path",
        choices=PREPROCESS_PATHS,
        default=DEFAULT_PREPROCESS_PATH,
        help=(
            "224 를 어디서 만들지. "
            "pil=CPU PIL BICUBIC (기본값, 배포 경로), "
            "interp_aa=GPU bicubic+antialias (드리프트 최소), "
            "roi_align=GPU bilinear 배치 1회 (N 이 커도 pre 가 안 늘어남). "
            "GPU 경로는 --crop-policy bbox --crop-fit pad 전용이며, "
            "임베딩이 미세하게 달라지므로(코사인 0.998 대) 바꾸면 저장된 "
            "임베딩을 재생성해야 한다"
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
        "--preprocess-workers",
        type=int,
        default=DEFAULT_PREPROCESS_WORKERS,
        help="PIL crop/resize ThreadPool worker 수 (플랫폼별 튜닝 값)",
    )
    parser.add_argument(
        "--async-preprocess",
        type=parse_bool,
        default=DEFAULT_ASYNC_PREPROCESS_ENABLED,
        metavar="true|false",
        help=(
            "TensorRT에서 PIL 전처리를 future로 제출해 엔진과 겹칠지 여부. "
            "false면 기존 동기 실행 순서를 사용한다"
        ),
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
    # BooleanOptionalAction 이 아니라 값을 받는 형태로 둔다. launch 가 인자를
    # 항상 "--이름 값" 쌍으로 넘기므로 플래그형은 launch 에서 못 쓴다.
    parser.add_argument(
        "--pipeline-enabled",
        type=parse_bool,
        default=DEFAULT_PIPELINE_ENABLED,
        metavar="true|false",
        help="CPU 전처리와 GPU 추론을 겹쳐 돌린다 (처리량↑, 지연은 그대로)",
    )
    parser.add_argument(
        "--pipeline-queue-depth",
        type=int,
        default=DEFAULT_PIPELINE_QUEUE_DEPTH,
        help="준비된 프레임 큐 깊이. 가득 차면 가장 오래된 것을 버린다",
    )
    parser.add_argument(
        "--stats-every",
        type=int,
        default=DEFAULT_STATS_EVERY,
        help="N 프레임마다 단계별 소요시간 한 줄 출력 (0=끔)",
    )

    return parser


@dataclass
class FrameWork:
    """1단계(preprocess_frame)가 만들고 2단계(infer_frame)가 소비하는 한 프레임.

    프레임에 딸린 것을 전부 들고 다니므로 두 단계가 다른 스레드에서 돌아도
    timestamp / segment_ids / boxes / 임베딩 행 순서가 섞일 수 없다. 큐는
    단일 생산자 - 단일 소비자 FIFO 라 순서도 보존된다.
    """

    stamp: Any
    header: Any
    segment_ids: List[int]
    regions: List[PILImage.Image]
    masks: List[Optional[np.ndarray]]
    boxes: List[Tuple[int, int, int, int]]
    # clip_backend.PreparedBatch. segment 가 없으면 None.
    prepared: Any
    # 두 입력의 짝이 맞춰진 시각 (perf_counter). latency 의 기준점이다.
    arrived_at: float
    preprocess_ms: float
    queued_at: float = 0.0


@dataclass
class StageTotals:
    """단계별 누적. 창(window) 단위로 평균을 내고 비운다."""

    frames: int = 0
    segments: int = 0
    preprocess_ms: float = 0.0
    queue_wait_ms: float = 0.0
    encoder_ms: float = 0.0
    postprocess_ms: float = 0.0
    latency_ms: float = 0.0
    latency_max_ms: float = 0.0
    dropped: int = 0
    started_at: float = field(default_factory=time.perf_counter)
    finished_at: float = 0.0


class ClipInferenceNode(Node):
    """frame-local segment마다 CLIP semantic embedding을 부여하는 노드."""

    def __init__(
        self,
        color_topic: str = DEFAULT_COLOR_TOPIC,
        segment_topic: str = DEFAULT_SEGMENT_TOPIC,
        embedding_topic: str = DEFAULT_EMBEDDING_TOPIC,
        semantics_topic: str = DEFAULT_SEMANTICS_TOPIC,
        publish_semantics: bool = DEFAULT_PUBLISH_SEMANTICS,
        prompts: Optional[List[str]] = None,
        top_k: int = DEFAULT_TOP_K,
        min_segment_area: int = 0,
        max_segments: int = 0,
        min_score: float = DEFAULT_MIN_SCORE,
        log_every: int = DEFAULT_LOG_EVERY,
        log_values: int = DEFAULT_LOG_VALUES,
        log_max_segments: int = DEFAULT_LOG_MAX_SEGMENTS,
        backend: str = DEFAULT_BACKEND,
        model_dir: str = DEFAULT_MODEL_DIR,
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
        preprocess_path: str = DEFAULT_PREPROCESS_PATH,
        mask_fill: int = DEFAULT_MASK_FILL,
        bbox_padding: int = DEFAULT_BBOX_PADDING,
        min_segment_pixels: int = DEFAULT_MIN_SEGMENT_PIXELS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        preprocess_workers: int = DEFAULT_PREPROCESS_WORKERS,
        async_preprocess: bool = DEFAULT_ASYNC_PREPROCESS_ENABLED,
        normalize_embeddings: bool = DEFAULT_NORMALIZE_EMBEDDINGS,
        sync_buffer_size: int = DEFAULT_SYNC_BUFFER_SIZE,
        publish_empty_sets: bool = DEFAULT_PUBLISH_EMPTY_SETS,
        qos_depth: int = DEFAULT_QOS_DEPTH,
        reliable_input: bool = DEFAULT_RELIABLE_INPUT,
        reliable_output: bool = DEFAULT_RELIABLE_OUTPUT,
        pipeline_enabled: bool = DEFAULT_PIPELINE_ENABLED,
        pipeline_queue_depth: int = DEFAULT_PIPELINE_QUEUE_DEPTH,
        stats_every: int = DEFAULT_STATS_EVERY,
    ) -> None:
        super().__init__("clip_inference_node")

        # BLAS 워커가 프레임마다 깨어나는 비용이 numpy 연산 본체보다 크다.
        # limit_blas_threads() 주석 참고. 백엔드보다 먼저 불러 둔다.
        limit_blas_threads(1)

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
        # select_segments() 가 쓰는 필터. 둘 다 0이면 기존 동작(전부 인코딩).
        self.min_segment_area = max(0, int(min_segment_area))
        self.max_segments = max(0, int(max_segments))
        self.min_score = float(min_score)

        self.log_every = int(log_every)
        self.log_values = int(log_values)
        self.log_max_segments = int(log_max_segments)

        # 프레임 로그용 카운터. build_regions가 필터링 통계를 여기에 남긴다.
        self.frame_count = 0
        # 라벨 확대 로그를 한 번만 찍기 위한 플래그 (match_labels_to_image)
        self.logged_label_resize = False
        self.last_candidate_count = 0
        self.last_skipped_count = 0

        self.backend_name = backend
        self.model_dir = str(Path(model_dir).expanduser()) if model_dir else ""

        def model_asset(path: str) -> str:
            if not path or not self.model_dir:
                return path
            return str(Path(self.model_dir) / Path(path).expanduser().name)

        # --model-dir를 주면 플랫폼별 workspace 차이를 여기서만 흡수한다.
        self.engine_path = model_asset(engine_path)
        self.pooled_engine_path = model_asset(pooled_engine_path)
        self.value_engine_path = model_asset(value_engine_path)

        if crop_fit not in CROP_FITS:
            raise ValueError(
                f"crop_fit must be one of {CROP_FITS}, got '{crop_fit}'"
            )

        self.crop_fit = crop_fit

        self.text_backend_name = text_backend
        self.text_engine_path = model_asset(text_engine_path)

        self.model_path = model_asset(model_path)
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
        # semantics 를 끄면 텍스트 인코더가 없어 곱할 대상 자체가 없다.
        # 그래도 읽으면 startup 로그가 "쓰고 있다"고 거짓말을 한다.
        self.text_alignment_matrix = (
            self.load_text_alignment_matrix(text_alignment_matrix)
            if self.publish_semantics
            else None
        )

        if not self.publish_semantics:
            self.text_alignment_matrix_path = ""

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

        # 224 를 어디서 만들지. 기본은 pil (기존 배포 경로) 이고, GPU 경로는
        # crop 복사 / PIL 생성 / 장별 H2D 를 건너뛴다. 선택 기준과 실측치는
        # clip_backend.PREPROCESS_PATHS 주석에 있다.
        #
        # **경로를 바꾸면 임베딩이 미세하게 달라진다** (코사인 0.998 대). 이미
        # 저장한 임베딩과 섞어 거리 비교를 하면 같은 물체가 0.95 대로 벌어지므로
        # (인스턴스 매칭 / 재식별이 정확히 그것을 한다), 전환하면 저장된
        # 임베딩을 전부 재생성해야 한다. 그래서 기본값은 건드리지 않는다.
        if preprocess_path not in PREPROCESS_PATHS:
            raise ValueError(
                f"preprocess_path must be one of {PREPROCESS_PATHS}, "
                f"got '{preprocess_path}'"
            )

        self.preprocess_path = preprocess_path

        # GPU 경로는 bbox crop + pad 기하 위에서만 성립한다. masked_bbox /
        # masked_full 은 crop 안을 mask_fill 로 칠하는데 그건 CPU 마스크가
        # 있어야 하고, centercrop / stretch 는 기하 자체가 다르다.
        if self.preprocess_path != "pil":
            if self.crop_policy != "bbox":
                raise ValueError(
                    f"preprocess_path='{self.preprocess_path}' requires "
                    f"crop_policy='bbox', got '{self.crop_policy}'"
                )

            if crop_fit != DEFAULT_CROP_FIT:
                raise ValueError(
                    f"preprocess_path='{self.preprocess_path}' requires "
                    f"crop_fit='{DEFAULT_CROP_FIT}', got '{crop_fit}'"
                )

        # cls 경로는 마스크를 쓰지 않는다. 백엔드가 무시하고, 디버그 저장은
        # stats(=None) 에서 먼저 빠져나간다. 그런데도 세그먼트마다 PIL 이미지를
        # 만들고 있었다 (N=32 에서 0.93ms). crop 안의 bool 배열은 masked_bbox /
        # masked_full 이 fill 에 쓰므로 그대로 두고, PIL 변환만 건너뛴다.
        self.needs_region_masks = pooling_mode != "cls"
        self.bbox_padding = int(bbox_padding)

        self.min_segment_pixels = int(min_segment_pixels)
        self.batch_size = int(batch_size)
        self.preprocess_workers = int(preprocess_workers)
        self.async_preprocess = bool(async_preprocess)
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

        if self.preprocess_workers < 1:
            raise ValueError(
                "preprocess_workers must be at least 1"
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
                    f"{self.active_engine_path}\n\n"
                    "모델 디렉터리로 찾아본 곳:\n  "
                    + describe_candidates()
                    + "\n\n다음으로 만들거나 (패키지 루트에서)\n"
                    "  python3 meridian_clip/export_onnx.py "
                    f"--part {part}\n"
                    "  python3 meridian_clip/build_engine.py "
                    f"--part {part}\n\n"
                    f"이미 만들었다면 {MODEL_DIR_ENV_VAR} 환경변수나 "
                    "--model-dir 로 그 위치를 알려주고,\n"
                    "엔진을 못 만드는 환경이면 --backend torch 로 실행하세요."
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
            preprocess_workers=self.preprocess_workers,
            async_preprocess=self.async_preprocess,
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
        self.get_logger().info(
            "Platform tuning: "
            f"model_dir={self.model_dir or '(individual paths)'}, "
            f"preprocess_workers={self.preprocess_workers}, "
            f"async_preprocess={self.async_preprocess}, "
            f"batch_size={self.batch_size}"
        )

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

        # ============================================================
        # 2-stage pipeline
        # ============================================================
        #
        # 1단계는 CPU (PIL 기하 + numpy), 2단계는 GPU (TensorRT) 라 서로
        # 겹칠 수 있다. 1단계는 구독 콜백 스레드가, 2단계는 아래 워커가
        # 맡는다. 두 단계가 부르는 함수는 순차 모드와 **완전히 같다**
        # (process_frame 이 둘을 이어 붙인 것뿐이다).
        self.pipeline_enabled = bool(pipeline_enabled)
        self.pipeline_queue_depth = max(1, int(pipeline_queue_depth))
        self.stats_every = int(stats_every)

        self.pipeline_queue: Optional[queue.Queue] = None
        self.pipeline_thread: Optional[threading.Thread] = None

        self.stats_lock = threading.Lock()
        self.stage_totals = StageTotals()

        if self.pipeline_enabled:
            self.pipeline_queue = queue.Queue(
                maxsize=self.pipeline_queue_depth
            )

            self.pipeline_thread = threading.Thread(
                target=self.pipeline_worker,
                name="clip-inference",
                daemon=True,
            )
            self.pipeline_thread.start()

        self.get_logger().info(
            "Pipeline: "
            + (
                f"enabled (queue depth {self.pipeline_queue_depth}, "
                "drop-oldest)"
                if self.pipeline_enabled
                else "disabled (sequential)"
            )
        )

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
            f"Crop policy: {self.crop_policy} (fit={self.crop_fit}, "
            f"preprocess={self.preprocess_path})"
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
            if self.pipeline_enabled:
                self.submit_frame(
                    color_msg=color_msg,
                    segment_msg=segment_msg,
                )
            else:
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
        """순차 경로. 두 단계를 그대로 이어 붙인 것이다.

        파이프라인 모드가 부르는 것과 **같은 두 함수**라, 결과가 달라질
        여지가 구조적으로 없다.
        """
        work = self.preprocess_frame(
            color_msg=color_msg,
            segment_msg=segment_msg,
        )

        if work is None:
            return

        self.infer_frame(work)

    def submit_frame(
        self,
        color_msg: RosImage,
        segment_msg: RosImage,
    ) -> None:
        """파이프라인 경로 1단계. 콜백 스레드에서 돈다."""
        work = self.preprocess_frame(
            color_msg=color_msg,
            segment_msg=segment_msg,
        )

        if work is None:
            return

        self.offer(work)

    def offer(self, work: FrameWork) -> None:
        """준비된 프레임을 큐에 넣는다. 가득 차면 **가장 오래된 것**을 버린다.

        입력이 처리 속도보다 빠를 때 오래된 프레임을 쌓아 두면 지연만 늘고
        내보내는 결과는 계속 과거가 된다. 최신 프레임을 우선한다.

        생산자와 소비자가 각각 하나뿐인 FIFO 라, 버리기는 맨 앞에서만
        일어나고 순서는 절대 뒤바뀌지 않는다.
        """
        work.queued_at = time.perf_counter()

        while True:
            try:
                self.pipeline_queue.put_nowait(work)
                return

            except queue.Full:
                try:
                    self.pipeline_queue.get_nowait()

                    with self.stats_lock:
                        self.stage_totals.dropped += 1

                except queue.Empty:
                    # 그 사이 소비자가 가져갔다. 다시 시도한다.
                    pass

    def pipeline_worker(self) -> None:
        """파이프라인 경로 2단계 전용 스레드."""
        while True:
            work = self.pipeline_queue.get()

            # 종료 신호
            if work is None:
                return

            try:
                self.infer_frame(work)

            except Exception as error:
                self.get_logger().error(
                    f"CLIP frame inference failed: {error}"
                )

    def stop_pipeline(self) -> None:
        """워커를 정리한다. destroy_node 에서 부른다."""
        if self.pipeline_thread is None:
            return

        self.pipeline_queue.put(None)
        self.pipeline_thread.join(timeout=5.0)
        self.pipeline_thread = None

    def destroy_node(self) -> bool:
        self.stop_pipeline()

        return super().destroy_node()

    # ----------------------------------------------------------------
    # Stage 1 : CPU 전처리
    # ----------------------------------------------------------------

    def match_labels_to_image(
        self,
        labels: np.ndarray,
        target_shape,
    ) -> Optional[np.ndarray]:
        """라벨 맵을 RGB 해상도로 최근접 확대한다.

        segmentor 는 마스크를 proto 격자로 낸다 -- FastSAM 계열은 letterbox
        패딩을 떼고 나면 640x480 입력에 대해 256x192 다. 이 노드의 나머지 경로는
        (bbox 계산, 패치 점유율, crop) 전부 라벨과 RGB 가 같은 격자라고 전제하고
        있어서, 크기가 다르면 예전에는 프레임을 통째로 버렸다. 그러면 파이프라인
        중간에 확대 전용 노드를 하나 더 두게 되는데, 그건 300KB 이미지를 매
        프레임 DDS 로 한 번 더 왕복시키는 값이다 (실측 코어 11%). 여기서 하면
        cv2.resize 한 번(0.29ms)으로 끝난다.

        **최근접이어야 한다.** 픽셀 값이 밝기가 아니라 segment_id 라서 보간하면
        원래 없던 id 가 경계에 생긴다.

        확대만 한다. 라벨에 letterbox 패딩이 남아 있으면 종횡비가 어긋나는데,
        그건 조용히 늘려서 될 일이 아니라 에러로 알린다.
        """
        target_h, target_w = target_shape
        source_h, source_w = labels.shape[:2]
        scale_y = target_h / source_h
        scale_x = target_w / source_w

        # 1% 문턱. FastSAM 은 letterbox 를 정수 proto 행으로 반올림하므로 16:9
        # 에서 배율이 0.07% 정도 어긋난다 -- 그건 정상이다. 패딩이 남은 경우는
        # 몇 십 % 씩 벌어지므로 이 문턱으로 갈린다.
        if abs(scale_y - scale_x) > 0.01 * max(scale_y, scale_x):
            self.get_logger().error(
                "RGB and segment label aspect ratio mismatch: "
                f"rgb={target_shape}, labels={labels.shape[:2]} "
                f"(scale y={scale_y:.4f} x={scale_x:.4f}). "
                "라벨에 letterbox 패딩이 남아 있으면 이렇게 된다.",
                throttle_duration_sec=5.0,
            )
            return None

        if not self.logged_label_resize:
            self.logged_label_resize = True
            self.get_logger().info(
                f"Segment labels {source_w}x{source_h} -> "
                f"{target_w}x{target_h} (nearest, scale {scale_x:.4f})"
            )
        return cv2.resize(
            labels, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    def preprocess_frame(
        self,
        color_msg: RosImage,
        segment_msg: RosImage,
    ) -> Optional[FrameWork]:
        """cv_bridge 변환 + region 만들기 + 224 전처리까지.

        GPU 를 쓰기는 하지만(H2D, 정규화) 엔진은 건드리지 않는다. 반환한
        FrameWork 는 다른 스레드의 infer_frame() 에 그대로 넘길 수 있다.
        """
        arrived_at = time.perf_counter()

        rgb_image = self.image_from_message(color_msg, "rgb8", 3)
        labels = self.image_from_message(segment_msg, "mono8", 1)

        if rgb_image.shape[:2] != labels.shape[:2]:
            labels = self.match_labels_to_image(labels, rgb_image.shape[:2])
            if labels is None:
                return None

        if self.preprocess_path == "pil":
            segment_ids, regions, masks, boxes = self.build_regions(
                rgb_image=rgb_image,
                labels=labels,
                with_masks=self.needs_region_masks,
            )

            prepared = (
                self.backend.prepare(
                    regions=regions,
                    masks=masks,
                    pooling_mode=self.pooling_mode,
                )
                if segment_ids
                else None
            )
        else:
            # GPU 경로. crop 복사 / 마스크 / PIL 을 만들지 않으므로 regions 와
            # masks 는 비어 있다 (디버그 저장이 그걸 보고 건너뛴다).
            segment_ids, crop_boxes, boxes = self.scan_segments(labels)

            regions = []
            masks = []

            prepared = (
                prepare_from_frame(
                    backend=self.backend,
                    rgb_image=rgb_image,
                    labels=labels,
                    segment_ids=segment_ids,
                    boxes=crop_boxes,
                    pooling_mode=self.pooling_mode,
                    preprocess_path=self.preprocess_path,
                )
                if segment_ids
                else None
            )

        return FrameWork(
            stamp=color_msg.header.stamp,
            header=color_msg.header,
            segment_ids=segment_ids,
            regions=regions,
            masks=masks,
            boxes=boxes,
            prepared=prepared,
            arrived_at=arrived_at,
            preprocess_ms=(time.perf_counter() - arrived_at) * 1000.0,
        )

    # ----------------------------------------------------------------
    # Stage 2 : 엔진 + 후처리 + 발행
    # ----------------------------------------------------------------

    def infer_frame(self, work: FrameWork) -> None:
        """한 frame의 모든 positive segment에 embedding을 부여한다."""
        dequeued_at = time.perf_counter()

        queue_wait_ms = (
            (dequeued_at - work.queued_at) * 1000.0
            if work.queued_at
            else 0.0
        )

        segment_ids = work.segment_ids
        regions = work.regions
        masks = work.masks
        boxes = work.boxes

        if not segment_ids:
            if self.publish_empty_sets:
                empty = np.zeros(
                    (0, self.embedding_dim),
                    dtype=np.float32,
                )

                self.publish_embeddings(
                    timestamp=work.stamp,
                    segment_ids=[],
                    embeddings=empty,
                )

                self.publish_semantic_detections(
                    header=work.header,
                    segment_ids=[],
                    boxes=[],
                    embeddings=empty,
                )

            self.record_stage_times(
                work=work,
                queue_wait_ms=queue_wait_ms,
                encoder_ms=0.0,
                postprocess_ms=(
                    time.perf_counter() - dequeued_at) * 1000.0,
            )

            return

        embeddings = self.backend.run(
            prepared=work.prepared,
            batch_size=self.batch_size,
            normalize=self.normalize_embeddings,
            gamma=self.patch_weight_gamma,
            min_patch_occupancy=self.min_patch_occupancy,
            empty_mask_fallback=self.empty_mask_fallback,
        )

        encoder_done_at = time.perf_counter()

        embeddings = self.apply_alignment(embeddings)

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
            timestamp=work.stamp,
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
            timestamp=work.stamp,
            segment_ids=segment_ids,
            embeddings=embeddings,
        )

        self.publish_semantic_detections(
            header=work.header,
            segment_ids=segment_ids,
            boxes=boxes,
            embeddings=embeddings,
        )

        self.record_stage_times(
            work=work,
            queue_wait_ms=queue_wait_ms,
            encoder_ms=(encoder_done_at - dequeued_at) * 1000.0,
            postprocess_ms=(
                time.perf_counter() - encoder_done_at) * 1000.0,
            segments=len(segment_ids),
        )

    # ----------------------------------------------------------------
    # 계측
    # ----------------------------------------------------------------

    def record_stage_times(
        self,
        work: FrameWork,
        queue_wait_ms: float,
        encoder_ms: float,
        postprocess_ms: float,
        segments: int = 0,
    ) -> None:
        """단계별 소요시간을 누적하고 필요하면 한 줄 요약을 찍는다.

        단계 구분:
            preprocess  cv_bridge + build_regions + backend.prepare
                        (224 기하, H2D, 정규화, 패치 점유율까지)
            queue wait  1단계가 큐에 넣은 뒤 2단계가 꺼낼 때까지
                        (순차 모드에서는 항상 0)
            encoder     backend.run  -- 엔진 실행, D2H, 빈마스크 fallback,
                        L2 정규화까지
            postprocess 정렬행렬 + keep 필터 + 로그/디버그 + 두 번의 publish
            latency     두 입력의 짝이 맞춰진 시각부터 발행이 끝날 때까지
        """
        finished_at = time.perf_counter()
        latency_ms = (finished_at - work.arrived_at) * 1000.0

        with self.stats_lock:
            totals = self.stage_totals

            totals.frames += 1
            totals.segments += segments
            totals.preprocess_ms += work.preprocess_ms
            totals.queue_wait_ms += queue_wait_ms
            totals.encoder_ms += encoder_ms
            totals.postprocess_ms += postprocess_ms
            totals.latency_ms += latency_ms
            totals.latency_max_ms = max(totals.latency_max_ms, latency_ms)
            totals.finished_at = finished_at

            if self.stats_every <= 0:
                return

            if totals.frames % self.stats_every != 0:
                return

            snapshot = totals
            self.stage_totals = StageTotals(started_at=finished_at)

        self.log_stage_times(snapshot)

    def log_stage_times(self, totals: StageTotals) -> None:
        """창 하나의 평균을 한 줄로 찍는다. 벤치마크가 이 줄을 읽는다."""
        frames = max(1, totals.frames)
        elapsed = max(1e-9, totals.finished_at - totals.started_at)

        self.get_logger().info(
            "STAGE "
            f"frames={totals.frames} "
            f"fps={frames / elapsed:.2f} "
            f"pre={totals.preprocess_ms / frames:.3f} "
            f"queue={totals.queue_wait_ms / frames:.3f} "
            f"enc={totals.encoder_ms / frames:.3f} "
            f"post={totals.postprocess_ms / frames:.3f} "
            f"latency={totals.latency_ms / frames:.3f} "
            f"latency_max={totals.latency_max_ms:.3f} "
            f"segments={totals.segments / frames:.1f} "
            f"dropped={totals.dropped}"
        )

    def image_from_message(self, message, encoding: str, channels: int):
        """sensor_msgs/Image -> ndarray. 인코딩이 맞으면 **복사하지 않는다**.

        cv_bridge 는 desired_encoding 을 주면 인코딩이 이미 같아도 배열을
        복사한다 (실측 720x480 rgb8 + mono8 두 장에 0.256ms, owndata=True).
        메시지 버퍼 위의 뷰로 충분하다 -- 0.002ms 이고 값은 같음을 확인했다.
        crop 은 어차피 .copy() 를 하고, 라벨은 읽기만 한다.

        인코딩이 다르면 cv_bridge 로 넘긴다 (변환이 실제로 필요한 경우다).
        """
        if message.encoding != encoding:
            return self.bridge.imgmsg_to_cv2(
                message, desired_encoding=encoding)

        height, width = int(message.height), int(message.width)

        step = int(message.step) or width * channels

        view = np.frombuffer(message.data, dtype=np.uint8)

        # step 이 width*channels 보다 크면 행 끝에 패딩이 있다.
        frame = view[:height * step].reshape(height, step)
        frame = frame[:, :width * channels]

        if channels == 1:
            return frame

        return frame.reshape(height, width, channels)

    def crop_bounds(
        self,
        bounds: Tuple[int, int, int, int],
        shape: Tuple[int, ...],
    ) -> Tuple[int, int, int, int]:
        """tight bbox -> 실제로 crop 할 영역. bbox_padding 을 붙이고 잘라낸다.

        **build_region() 과 GPU 경로가 같은 함수를 써야 한다.** 이 계산이
        갈리면 두 전처리 경로가 몇 픽셀 어긋난 crop 을 만들고, 그 상태로 잰
        임베딩 비교는 의미가 없어진다 (개발 중 실제로 겪었다 -- GPU 경로가
        패딩을 빼먹어 4px 어긋난 crop 끼리 비교하고 있었다).
        """
        height, width = shape[:2]

        return (
            max(int(bounds[0]) - self.bbox_padding, 0),
            max(int(bounds[1]) - self.bbox_padding, 0),
            min(int(bounds[2]) + self.bbox_padding, width),
            min(int(bounds[3]) + self.bbox_padding, height),
        )

    def select_segments(self, areas: np.ndarray) -> np.ndarray:
        """인코딩할 segment_id 를 고른다 (면적 순 내림차순).

        CLIP 은 프레임의 모든 세그먼트를 인코딩한다. 그런데 인코더 시간이
        세그먼트 수 N 에 거의 정비례한다 -- 이 플랫폼 실측으로 전체 스택에서
        ``enc ≈ 1.0 ms × N`` 이다 (N 7 -> 10 ms, 22 -> 21 ms, 42 -> 41 ms).
        그래서 잔챙이 마스크 몇 개가 프레임 시간을 그대로 밀어 올린다.

        SAM 쪽 ``area_min`` 을 올려도 N 은 줄지만, 그건 ``/segment_image`` 자체에서
        마스크를 빼기 때문에 **geobuilder 도 같이 잃는다.** geobuilder 는 N 에
        훨씬 둔감하므로(CPU 만 쓴다) 작은 물체까지 3D 로 복원하는 편이 낫다.
        그래서 필터를 여기, 비싼 소비자 쪽에만 둔다.

            min_segment_area   이 픽셀 수 미만인 세그먼트는 인코딩하지 않는다.
                               라벨 이미지 좌표 기준이다(= RGB 해상도).
            max_segments       면적 큰 것부터 이 개수까지만. enc 시간에 상한을
                               걸어 장면이 복잡해져도 프레임 시간이 안 튄다.

        둘 다 0 이면 아무것도 거르지 않는다(기존 동작). 빠진 세그먼트는
        InstanceEmbeddingSet.segment_ids 에 안 들어갈 뿐이라, id 로 짝짓는
        소비자에게는 그냥 "이번 프레임에 임베딩이 없는 인스턴스" 다.
        """
        present = np.nonzero(areas)[0]
        present = present[present != BACKGROUND_SEGMENT_ID]
        if not present.size:
            return present

        if self.min_segment_area > 0:
            present = present[areas[present] >= self.min_segment_area]
            if not present.size:
                return present

        if 0 < self.max_segments < present.size:
            # 면적 큰 것부터 max_segments 개. 남긴 뒤 id 순으로 되돌린다 --
            # 아래 로직이 present 가 오름차순인 것을 전제한다.
            keep = present[np.argsort(-areas[present], kind="stable")]
            present = np.sort(keep[: self.max_segments])
        return present

    def scan_segments(
        self,
        labels: np.ndarray,
    ) -> Tuple[
        List[int],
        List[Tuple[int, int, int, int]],
        List[Tuple[int, int, int, int]],
    ]:
        """라벨 프레임에서 (segment_ids, crop_boxes, tight_boxes) 만 뽑는다.

        build_regions() 의 앞부분과 같은 스캔이지만 crop 복사 / 마스크 생성 /
        PIL 변환을 하지 않는다. GPU 전처리 경로가 이걸 쓴다 -- 그 경로는
        프레임 한 장과 박스만 있으면 224 를 만들 수 있으므로 crop 을 만드는
        것이 순수한 낭비다 (실측 4.0ms).

        **GPU 로 옮기려다 되돌렸다.** 라벨맵이 점유율 때문에 어차피 GPU 로
        올라가니 여기서 scatter_reduce 로 라벨별 행/열 min·max 를 구하면 공짜일
        것 같았다. 값은 정확히 같았지만 **더 느렸다 (3.48ms vs 2.25ms)**.
        원인은 345,600개 원소를 256개 bin 에 atomic 으로 모으는 경합과, int64
        승격(2.7MB 할당) 그리고 결과를 읽으려면 D2H 동기화가 필요한 것이다.
        bin 수를 늘려 경합을 줄이는 변형(label*H+row 로 bincount)도 있지만
        CPU 2.25ms 를 이길 여지가 크지 않아 접었다.

        **두 종류의 박스를 구분해서 돌려준다.**
            crop_boxes   bbox_padding 이 들어간 실제 crop 영역. 전처리가 쓴다.
            tight_boxes  마스크의 tight bbox. semantics 발행이 쓴다.
        섞으면 pil 경로와 어긋난 crop 을 만든다 (crop_bounds 주석 참고).
        """
        areas = cv2.calcHist(
            [labels], [0], None,
            [MAX_SEGMENT_ID + 1], [0, MAX_SEGMENT_ID + 1],
        ).ravel()

        present = self.select_segments(areas)

        self.last_candidate_count = int(present.size)
        self.last_skipped_count = 0

        segment_ids: List[int] = []
        crop_boxes: List[Tuple[int, int, int, int]] = []
        tight_boxes: List[Tuple[int, int, int, int]] = []

        if not present.size:
            return segment_ids, crop_boxes, tight_boxes

        found = ndimage.find_objects(labels, max_label=int(present[-1]))

        for segment_id in present.tolist():
            if areas[segment_id] < self.min_segment_pixels:
                self.last_skipped_count += 1

                continue

            box = found[segment_id - 1]

            if box is None:
                continue

            row_slice, column_slice = box

            bounds = (
                int(column_slice.start),
                int(row_slice.start),
                int(column_slice.stop),
                int(row_slice.stop),
            )

            segment_ids.append(segment_id)
            tight_boxes.append(bounds)
            crop_boxes.append(self.crop_bounds(bounds, labels.shape))

        return segment_ids, crop_boxes, tight_boxes

    def build_regions(
        self,
        rgb_image: np.ndarray,
        labels: np.ndarray,
        with_masks: bool = True,
    ) -> Tuple[
        List[int],
        List[PILImage.Image],
        List[Optional[np.ndarray]],
        List[Tuple[int, int, int, int]],
    ]:
        """
        각 unique positive segment_id마다 CLIP 입력 region을 만든다.

        반환되는 segment_ids / regions / masks / boxes는 같은 순서를 유지하며,
        이 순서가 InstanceEmbeddingSet의 row 순서가 된다.
        masks[i]는 regions[i]와 정확히 같은 크기의 binary mask(bool ndarray)이며
        가중평균 pooling의 패치 점유율 계산에 쓴다. 백엔드가 224로 늘리지 않고
        바로 7x7 점유율로 접으므로 PIL로 바꾸지 않는다
        (clip_backend.patch_occupancy_from_masks 주석 참고).
        디버그 저장만 mask_to_image()로 PIL을 만든다.
        boxes는 마스크의 tight bbox (x0, y0, x1, y1)이며 semantics 발행에 쓴다.

        with_masks=False면 masks[i]는 None이다. cls pooling은 마스크를 쓰지
        않으므로 노드의 프레임 경로가 이걸로 마스크 생성을 건너뛴다.

        **기본값이 True인 것이 중요하다.** tools/의 벤치마크들은 첫 모드의
        노드로 crop을 한 번 만들어 나머지 모드에 그대로 넘긴다 -- 세 모드가
        같은 crop을 공유해야 pooling 차이만 분리되기 때문이다. 여기서
        pooling_mode를 보고 마스크를 생략하면 첫 모드가 cls일 때 뒤따르는
        마스크 모드들이 None을 받아 조용히 망가진다 (실측: value 87.92%
        -> 76.08%, patch 7.87% -> 35.26%). 그래서 생략 여부는 인스턴스
        상태가 아니라 **호출자가 명시**한다.
        """
        # 스캔은 scan_segments() 하나만 쓴다. 여기서 다시 구현하면 GPU 경로와
        # 갈라진다 (crop_bounds 주석 참고).
        candidates, crop_boxes, tight_boxes = self.scan_segments(labels)

        segment_ids: List[int] = []
        regions: List[PILImage.Image] = []
        masks: List[Optional[np.ndarray]] = []
        boxes: List[Tuple[int, int, int, int]] = []

        for segment_id, crop_box, bounds in zip(
                candidates, crop_boxes, tight_boxes):
            built = self.build_region(
                rgb_image=rgb_image,
                labels=labels,
                segment_id=segment_id,
                bounds=crop_box,
                with_mask=with_masks,
            )

            if built is None:
                continue

            region, region_mask = built

            segment_ids.append(segment_id)
            regions.append(region)
            masks.append(region_mask)
            boxes.append(bounds)

        return segment_ids, regions, masks, boxes

    @staticmethod
    def mask_to_image(mask: np.ndarray) -> PILImage.Image:
        """마스크 배열을 백엔드가 받는 grayscale 이미지로 바꾼다.

        bool 은 numpy 에서 1바이트라 view 로 uint8 을 공짜로 얻는다.
        astype 을 쓰면 배열을 한 번 더 복사한다. 호출부가 넘기는 마스크는
        모두 == 비교의 결과라 C-contiguous 이고, view 의 전제가 이것이다.
        """
        return PILImage.fromarray(
            mask.view(np.uint8) * 255
        )

    def build_region(
        self,
        rgb_image: np.ndarray,
        labels: np.ndarray,
        segment_id: int,
        bounds: Tuple[int, int, int, int],
        with_mask: bool = True,
    ) -> Optional[Tuple[PILImage.Image, Optional[np.ndarray]]]:
        """crop_policy에 따라 segment 하나의 (RGB region, mask)를 구성한다.

        두 이미지는 항상 같은 크기다. 백엔드가 여기에 같은 resize/crop을
        적용하므로, 크기가 어긋나면 패치 점유율과 patch token의 위치 대응이
        깨진다.

        마스크는 **필요한 범위에서만** 만든다. 프레임 전체 크기의 bool 배열이
        필요한 것은 masked_full 뿐이고, 나머지 두 정책은 bbox 안만 있으면
        된다. 세그먼트마다 640x480 배열을 만들던 비용이 여기서 사라진다.

        bounds 는 **이미 crop_bounds() 를 거친 crop 영역**이다 (tight bbox 가
        아니다). 패딩 계산이 scan_segments() 한 곳에만 있어야 GPU 경로와
        갈라지지 않는다 -- crop_bounds 주석 참고.
        """
        if self.crop_policy == "masked_full":
            mask = labels == segment_id

            region = rgb_image.copy()
            region[~mask] = self.mask_fill

            return (
                PILImage.fromarray(region),
                mask if with_mask else None,
            )

        # bounds 는 호출부(scan_segments)가 이미 crop_bounds() 로 만든 **crop
        # 영역**이다. 여기서 다시 패딩을 붙이면 이중 적용이 된다.
        x0, y0, x1, y1 = bounds

        region = rgb_image[y0:y1, x0:x1].copy()

        # crop 안에서만 마스크를 만든다. == 의 결과라 새 C-contiguous 배열이고,
        # mask_to_image 의 view 가 이것을 전제한다.
        region_mask = labels[y0:y1, x0:x1] == segment_id

        if self.crop_policy == "masked_bbox":
            # bbox 안에서도 다른 object의 픽셀은 배제한다.
            region[~region_mask] = self.mask_fill

        return (
            PILImage.fromarray(region),
            region_mask if with_mask else None,
        )

    def encode_regions(
        self,
        regions: List[PILImage.Image],
        masks: Optional[List[np.ndarray]] = None,
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
        if self.model_dir and not path and resolved:
            resolved = str(Path(self.model_dir) / Path(resolved).expanduser().name)

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
        masks: List[np.ndarray],
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

        # GPU 전처리 경로는 crop 과 마스크를 CPU 에 만들지 않는다. 디버그 저장은
        # 그 두 장을 겹쳐 보는 것이 목적이라, 없으면 저장할 것이 없다.
        # 정렬을 눈으로 확인해야 하면 --preprocess-path pil 로 한 번 돌린다.
        if len(regions) != len(segment_ids):
            if self.frame_count % self.debug_save_every == 0:
                self.get_logger().warn(
                    "디버그 이미지 저장을 건너뜁니다: "
                    f"preprocess_path='{self.preprocess_path}' 는 crop/마스크를 "
                    "만들지 않습니다. pil 경로로 돌리세요."
                )

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

            mask_image = self.debug_mask_geometry(
                self.mask_to_image(masks[row])
            )
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
        #
        # tolist() 로 넘기면 rclpy 가 원소 16384개를 하나씩 float 타입/범위
        # 검사한다 (실측 2.96ms, 그중 tolist 자체는 0.28ms 뿐이다).
        # array("f") 는 필드 타입과 정확히 같아서 그 검사를 건너뛴다
        # (0.007ms). 바이트가 그대로 들어가므로 값은 동일하다.
        output_msg.embeddings = array(
            "f",
            np.ascontiguousarray(embeddings, dtype=np.float32).tobytes(),
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
        min_segment_area=cli.min_segment_area,
        max_segments=cli.max_segments,
        min_score=cli.min_score,
        log_every=cli.log_every,
        log_values=cli.log_values,
        log_max_segments=cli.log_max_segments,
        backend=cli.backend,
        model_dir=cli.model_dir,
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
        preprocess_path=cli.preprocess_path,
        mask_fill=cli.mask_fill,
        bbox_padding=cli.bbox_padding,
        min_segment_pixels=cli.min_segment_pixels,
        batch_size=cli.batch_size,
        preprocess_workers=cli.preprocess_workers,
        async_preprocess=cli.async_preprocess,
        normalize_embeddings=cli.normalize_embeddings,
        sync_buffer_size=cli.sync_buffer_size,
        publish_empty_sets=cli.publish_empty_sets,
        qos_depth=cli.qos_depth,
        reliable_input=cli.reliable_input,
        reliable_output=cli.reliable_output,
        pipeline_enabled=cli.pipeline_enabled,
        pipeline_queue_depth=cli.pipeline_queue_depth,
        stats_every=cli.stats_every,
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
