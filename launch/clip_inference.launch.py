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
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


LAUNCH_ARGUMENTS = {
    "color_topic": "/camera/rgb",
    # sensor_msgs/Image (mono8), 픽셀 값 = segment_id
    "segment_topic": "/segment_image",
    "embedding_topic": "/instance_embedding_set",
    # tensorrt = models/*.engine (기본), torch = models/ViT-B-32.pt
    # 이미지/텍스트 인코더를 따로 고른다.
    "backend": "tensorrt",
    "text_backend": "tensorrt",
    "model_path": (
        "~/meridian/src/meridian_clip/models/ViT-B-32.pt"
    ),
    "engine_path": (
        "~/meridian/src/meridian_clip/models/"
        "clip_vit_b32_visual_fp16.engine"
    ),
    # pooling_mode:=mask_weighted_patch 일 때 쓰는 엔진.
    # export_onnx.py / build_engine.py 를 --part visual_pooled 로 돌려 만든다.
    "pooled_engine_path": (
        "~/meridian/src/meridian_clip/models/"
        "clip_vit_b32_visual_pooled_fp16.engine"
    ),
    # pooling_mode:=mask_weighted_value 일 때 쓰는 엔진.
    # export_onnx.py / build_engine.py 를 --part visual_pooled_value 로 만든다.
    "value_engine_path": (
        "~/meridian/src/meridian_clip/models/"
        "clip_vit_b32_visual_pooled_value_fp16.engine"
    ),
    "text_engine_path": (
        "~/meridian/src/meridian_clip/models/"
        "clip_vit_b32_text_fp16.engine"
    ),
    # crop 안에서 마스크 밖 픽셀 처리.
    #   bbox        : 그대로 둔다 (기본값). 기본 pooling 이 이미 패치 점유율로
    #                 마스크 밖을 배제하므로 검게 칠하는 것이 중복이고,
    #                 검정은 CLIP 이 학습 중 본 적 없는 분포다.
    #   masked_bbox : mask_fill 로 덮는다. pooling_mode:=cls 로 돌릴 때 권장.
    #   masked_full : 자르지 않고 마스크 밖만 덮는다.
    "crop_policy": "bbox",
    # crop 을 224x224 로 만드는 방법. pad = 긴 변을 224 에 맞추고 채움(기본).
    # centercrop 은 CLIP 원본이지만 길쭉한 물체의 양끝을 잘라낸다.
    "crop_fit": "pad",
    # 엔진 프로파일의 opt 와 같아야 한다 (models/ 는 min=1/opt=32/max=64).
    # 어긋나면 TensorRT 가 튜닝하지 않은 배치 크기로 돌게 된다.
    "batch_size": "32",
    # 224 기하를 만드는 스레드 수. Jetson Orin(12코어)에서 8개가 최적이었다.
    "preprocess_workers": "8",
    # 전처리를 청크 단위로 엔진과 겹칠지. 워커가 CPU 를 채우고 나면 손해라
    # 기본은 false 다. GPU 가 훨씬 빠른 장비에서만 켜 볼 만하다.
    "async_preprocess": "false",
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
    # "alignment_matrix":
    #     "~/meridian/src/meridian_clip/models/align_value_to_cls.npy",
    "patch_weight_gamma": "1.0",
    "min_patch_occupancy": "0.0",
    "empty_mask_fallback": "cls",
    # 프레임마다 터미널에 마스크 수/임베딩 값 찍기
    "log_every": "1",
    "log_values": "8",
    "log_max_segments": "5",
    # zero-shot semantics
    "top_k": "3",
    "min_score": "0.0",
}

# 빈 문자열 기본값을 가진 인자는 위 dict 에 넣지 않는다. launch 가 항상
# "--이름 값" 쌍으로 넘기므로 빈 값이 섞이면 argparse 가 혼란스러워진다.
# 디버그 이미지 저장은 필요할 때 ros2 run 으로 직접 준다:
#   ros2 run meridian_clip clip_inference_node --backend torch \
#       --pooling-mode mask_weighted_patch --debug-save-dir /tmp/clip_debug
#
# --model-dir 도 같은 이유로 여기 없다. models/ 를 다른 곳에 복사해 두고
# 위 다섯 경로를 한 번에 그쪽으로 돌릴 때 쓴다:
#   ros2 run meridian_clip clip_inference_node --model-dir ~/clip_bench_code/models


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
