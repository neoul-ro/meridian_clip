import os
from glob import glob

from setuptools import find_packages, setup


package_name = "meridian_clip"


setup(
    name=package_name,
    version="0.0.2",
    packages=find_packages(
        exclude=["test", "test.*"],
    ),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [
                os.path.join(
                    "resource",
                    package_name,
                )
            ],
        ),
        (
            os.path.join(
                "share",
                package_name,
            ),
            ["package.xml"],
        ),
        (
            os.path.join(
                "share",
                package_name,
                "launch",
            ),
            glob("launch/*.launch.py"),
        ),
        # 노드 설정은 ROS 파라미터 파일이 아니라 argparse CLI 인자로 준다.
        # (config/clip_params.yaml 은 제거됨)
        # 모델 바이너리(.pt/.onnx/.engine)는 수백 MB이고 매 빌드마다 복사되므로
        # install 트리에는 설명만 두고, 실제 경로는 ROS 파라미터로 넘긴다.
        (
            os.path.join(
                "share",
                package_name,
                "models",
            ),
            glob("models/*.md"),
        ),
    ],
    install_requires=[
        "setuptools",
    ],
    zip_safe=True,
    maintainer="sojin",
    maintainer_email="sojin@example.com",
    description=(
        "Meridian Perception Frontend: CLIP ViT-B/32 semantic embedding"
    ),
    license="Apache-2.0",
    tests_require=[
        "pytest",
    ],
    entry_points={
        "console_scripts": [
            (
                "clip_inference_node = "
                "meridian_clip.clip_inference_node:main"
            ),
            # 관측용 노드들. CLIP/torch 를 로드하지 않고 발행된 토픽만 읽으므로
            # 시스템 파이썬으로 돌아간다 (shebang 재패치 대상이 아니다).
            (
                "embedding_monitor = "
                "meridian_clip.embedding_monitor:main"
            ),
            (
                "clip_label_viz = "
                "meridian_clip.clip_label_viz:main"
            ),
        ],
    },
)
