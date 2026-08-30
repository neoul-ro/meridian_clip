"""clip_inference_node 실행용 launch.

노드 설정은 ROS 파라미터가 아니라 argparse CLI 인자로 받으므로
파라미터 파일(config/clip_params.yaml) 대신 arguments 로 넘긴다.
값을 바꾸려면 launch 인자를 쓴다:

    ros2 launch meridian_clip clip_inference.launch.py crop_policy:=masked_bbox

기본 color 토픽은 계약값인 /camera/rgb 다 (meridian_sensor 가 발행).
RealSense 를 직접 쓸 때는:

    ros2 launch meridian_clip clip_inference.launch.py \
        color_topic:=/camera/camera/color/image_raw

QoS 는 BooleanOptionalAction 플래그라 launch 의 "--이름 값" 형식으로 넘길 수
없다. 기본값(구독 best_effort / 발행 reliable)이 어느 짝과도 호환되므로 보통
건드릴 일이 없고, 필요하면 ros2 run 으로 --reliable-input 을 직접 준다.

모델 경로
--------
**절대경로를 이 파일에 박지 않는다.** 워크스페이스 이름도 사용자 이름도 머신마다
다르다. 엔진/체크포인트는 ``<패키지 루트>/models/`` 에 두는 것이 기본이고 --
download_weights.py / export_onnx.py / build_engine.py 세 스크립트의 기본 출력
위치와 같다 -- 찾는 순서는 노드(``clip_inference_node.model_dir_candidates``)와
글자 그대로 같다:

    1. 환경변수 MERIDIAN_CLIP_MODEL_DIR
    2. <share>/meridian_clip/models   (install 트리)
    3. <패키지 루트>/models           (소스 트리)

모델은 2GB 가 넘어서 install 트리로 복사시키지 않으므로, 평범한 colcon build
에서는 2번이 비어 있고 3번도 소스가 아닌 install 을 가리킨다. 그때는 1번을 쓴다:

    export MERIDIAN_CLIP_MODEL_DIR=<워크스페이스>/src/meridian/meridian_clip/models

한 번만 다르게 띄우고 싶으면 launch 인자로도 된다:

    ros2 launch meridian_clip clip_inference.launch.py model_dir:=/opt/clip_models

개별 경로 다섯 개(model_path / engine_path / pooled_engine_path /
value_engine_path / text_engine_path)는 ``model_dir`` 에서 조립하므로 그 한
줄만 바꾸면 전부 따라온다. 노드 쪽도 같은 규약이다 -- ``--model-dir`` 가 비어
있지 않으면 개별 경로의 파일 이름만 남기고 디렉터리를 갈아끼운다
(clip_inference_node.py 의 ``model_asset``).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

from meridian_clip.model_paths import default_model_dir


# 노드와 같은 함수를 부른다 -- 두 벌로 두면 조용히 어긋난다.
# model_paths 는 표준 라이브러리 + ament_index 만 쓰므로, torch/tensorrt 가 없는
# 시스템 파이썬으로 도는 ros2 launch 에서도 안전하게 import 된다.
# 여기서 미리 풀어 두는 이유: launch 로그와 --show-args 에 실제 경로가 찍히는
# 편이 경로 문제를 훨씬 빨리 잡는다.
DEFAULT_MODEL_DIR = default_model_dir()


def asset(filename):
    """model_dir 하위의 파일 하나. model_dir 을 바꾸면 같이 따라온다.

    치환이라 launch 인자 ``model_dir:=...`` 이 들어와도 그 값으로 풀린다.
    ``model_dir`` 은 아래 dict 에서 이 함수를 쓰는 항목들보다 먼저 나와야
    한다 -- launch 가 선언 액션을 순서대로 처리하기 때문이다.
    """
    return PathJoinSubstitution([LaunchConfiguration("model_dir"), filename])


LAUNCH_ARGUMENTS = {
    "color_topic": "/camera/rgb",
    # sensor_msgs/Image (mono8), 픽셀 값 = segment_id
    "segment_topic": "/segment_image",
    "embedding_topic": "/instance_embedding_set",
    # 구독 큐 깊이. 노드 기본값은 10인데, 입력이 이 노드보다 빠르면 큐가 꽉 차서
    # 지연이 그대로 쌓인다 -- 실측: SAM 26Hz 입력 / CLIP 16Hz 처리에서 라벨 도착
    # 부터 임베딩까지 중앙값 397ms(p95 479ms)였다. 1로 두면 항상 가장 최신 프레임만
    # 잡으므로 처리량은 그대로고 지연만 한 프레임 수준으로 떨어진다. 프레임을
    # 하나도 흘리면 안 되는 용도라면 키워라.
    "qos_depth": "1",
    # tensorrt = models/*.engine (기본), torch = models/ViT-B-32.pt
    # 이미지/텍스트 인코더를 따로 고른다.
    "backend": "tensorrt",
    "text_backend": "tensorrt",
    # 엔진/체크포인트/정렬행렬 디렉터리. 아래 개별 경로는 전부 여기서
    # 조립하므로, 모델을 옮겼으면 이 한 줄만 바꾸면 된다.
    "model_dir": DEFAULT_MODEL_DIR,
    "model_path": asset("ViT-B-32.pt"),
    "engine_path": asset("clip_vit_b32_visual_fp16.engine"),
    # pooling_mode:=mask_weighted_patch 일 때 쓰는 엔진.
    # export_onnx.py / build_engine.py 를 --part visual_pooled 로 돌려 만든다.
    "pooled_engine_path": asset("clip_vit_b32_visual_pooled_fp16.engine"),
    # pooling_mode:=mask_weighted_value 일 때 쓰는 엔진 (기본 풀링 모드).
    # export_onnx.py / build_engine.py 를 --part visual_pooled_value 로 만든다.
    "value_engine_path": asset("clip_vit_b32_visual_pooled_value_fp16.engine"),
    "text_engine_path": asset("clip_vit_b32_text_fp16.engine"),
    # crop 안에서 마스크 밖 픽셀 처리.
    #   bbox        : 그대로 둔다 (기본값). 기본 pooling 이 이미 패치 점유율로
    #                 마스크 밖을 배제하므로 검게 칠하는 것이 중복이고,
    #                 검정은 CLIP 이 학습 중 본 적 없는 분포다.
    #   masked_bbox : mask_fill 로 덮는다. pooling_mode:=cls 로 돌릴 때 권장.
    #   masked_full : 자르지 않고 마스크 밖만 덮는다.
    "crop_policy": "bbox",
    # 224 를 어디서 만들지. pil=CPU PIL BICUBIC, interp_aa=GPU bicubic+antialias,
    # roi_align=GPU bilinear 배치 1회. GPU 경로는 crop_policy=bbox / crop_fit=pad
    # 전용이다. 노드 자체 기본값은 pil 이지만 여기서는 roi_align 을 쓴다 --
    # clip_backend.py 의 실측표에서 value 모드 pre 가 16.15 -> 7.68ms 로 줄고,
    # 배치 커널이라 세그먼트 수 N 이 흔들려도 pre 가 거의 안 늘어난다.
    # 주의: 임베딩이 pil 과 비트 동일하지 않다 (코사인 평균 0.985, 최소 0.863).
    # 이미 저장해 둔 임베딩과 거리를 비교할 거면 그 임베딩들을 재생성해야 한다.
    "preprocess_path": "roi_align",
    # crop 을 224x224 로 만드는 방법. pad = 긴 변을 224 에 맞추고 채움(기본).
    # centercrop 은 CLIP 원본이지만 길쭉한 물체의 양끝을 잘라낸다.
    "crop_fit": "pad",
    # 엔진 프로파일의 opt 와 같아야 한다 (models/ 는 min=1/opt=32/max=64).
    # 어긋나면 TensorRT 가 튜닝하지 않은 배치 크기로 돌게 된다.
    "batch_size": "32",
    # 49개 patch token 을 합치는 방법.
    #   cls                 : CLS token (CLIP 원본 경로)
    #   mask_weighted_patch : 패치별 객체 점유율로 가중평균. 텍스트 정렬이
    #                         깨져 있어 alignment_matrix 없이는 못 쓴다.
    #   mask_weighted_value : 같은 가중평균이되 마지막 블록의 value 투영을
    #                         patch feature 로 사용 (MaskCLIP). 기본값.
    # tensorrt 백엔드는 모드에 맞는 엔진(pooled_engine_path /
    # value_engine_path)을 쓰고, torch 백엔드는 .pt 에서 직접 꺼낸다.
    # 바꾸려면 아래 한 줄이면 된다.
    #   ros2 launch ... pooling_mode:=cls
    "pooling_mode": "mask_weighted_value",
    # 정렬 행렬 .npy (선택). zero-shot 라벨 정확도가 우선이면 켠다 --
    # VOC2012 val 기준 top-1 81.67% -> 84.56%, 대신 AUC 0.9755 -> 0.9626.
    # 임베딩 매칭이 우선이면 비워 둔다. 빈 문자열은 launch 로 넘길 수 없으므로
    # 쓰려면 아래 줄의 주석을 풀거나 ros2 run 으로 직접 준다.
    #   ros2 launch ... alignment_matrix:=<경로>
    # "alignment_matrix": asset("align_value_to_cls.npy"),
    "patch_weight_gamma": "1.0",
    "min_patch_occupancy": "0.0",
    "empty_mask_fallback": "cls",
    # 프레임마다 터미널에 마스크 수/임베딩 값 찍기
    "log_every": "1",
    "log_values": "8",
    "log_max_segments": "5",
    # zero-shot semantics (Detection2DArray). 기본은 끔 -- 프레임당 5.1ms 로
    # 세그먼트 32개 기준 Postprocessing 의 88% 를 먹는다. 끄면 텍스트
    # 인코더도 로드하지 않는다. 소비자(clip_semantics 구독자)가 있으면 켠다.
    #   ros2 launch ... publish_semantics:=true
    "publish_semantics": "false",
    "top_k": "3",
    # 인코딩할 세그먼트를 고르는 필터. 둘 다 0이면 전부 인코딩(기본).
    # enc 시간이 세그먼트 수 N 에 거의 정비례한다 -- 이 플랫폼 실측으로 전체
    # 스택에서 enc ~ 1.0ms x N (N 7->10ms, 22->21ms, 42->41ms). 잔챙이 마스크가
    # 프레임 시간을 그대로 밀어 올리므로 여기서 거른다.
    #
    # SAM 쪽 area_min 을 올리는 것과 다르다. 그쪽은 /segment_image 에서 마스크를
    # 아예 빼서 geobuilder 도 같이 잃는다. geobuilder 는 CPU 만 쓰고 N 에 둔감해서
    # 작은 물체까지 3D 로 복원하는 편이 낫다. 그래서 비싼 소비자인 CLIP 쪽에만 둔다.
    "min_segment_area": "0",
    "max_segments": "0",
    "min_score": "0.0",
    # CPU 전처리(1단계)와 GPU 추론(2단계)을 다른 스레드에서 겹쳐 돌린다.
    # 처리량이 1.7배 안팎으로 오르지만 프레임 하나의 지연은 줄지 않는다
    # (큐 대기만큼 오히려 늘어난다). 임베딩 값은 완전히 동일하다.
    #   ros2 launch ... pipeline_enabled:=true
    "pipeline_enabled": "false",
    # 큐가 가득 차면 가장 오래된 프레임을 버리고 최신을 넣는다.
    # 깊이를 키워도 처리량은 안 오르고 지연만 늘어난다.
    "pipeline_queue_depth": "2",
    # PIL crop/resize 스레드 수. Jetson 은 4에서 이미 포화(4=43.8, 8=44.8 FPS)
    "preprocess_workers": "8",
    # 청크 단위 겹치기. Jetson 은 손해라 끈다 (clip_backend 주석 참고)
    "async_preprocess": "false",
    # N 프레임마다 단계별 소요시간 한 줄 출력 (0=끔)
    "stats_every": "0",
}

# 빈 문자열 기본값을 가진 인자는 위 dict 에 넣지 않는다. launch 가 항상
# "--이름 값" 쌍으로 넘기므로 빈 값이 섞이면 argparse 가 혼란스러워진다.
# 디버그 이미지 저장은 필요할 때 ros2 run 으로 직접 준다:
#   ros2 run meridian_clip clip_inference_node --backend torch \
#       --pooling-mode mask_weighted_patch --debug-save-dir /tmp/clip_debug


def generate_launch_description() -> LaunchDescription:
    declarations = [
        DeclareLaunchArgument(
            name,
            default_value=default,
        )
        for name, default in LAUNCH_ARGUMENTS.items()
    ]

    node_arguments = []

    for name in LAUNCH_ARGUMENTS:
        node_arguments.append(
            "--" + name.replace("_", "-")
        )
        node_arguments.append(
            LaunchConfiguration(name)
        )

    clip_node = Node(
        package="meridian_clip",
        executable="clip_inference_node",
        name="clip_inference_node",
        output="screen",
        arguments=node_arguments,
    )

    return LaunchDescription(
        declarations + [clip_node]
    )
