#!/usr/bin/env python3

"""clip_inference_node 의 출력을 터미널에서 확인하는 모니터.

이 툴도 CLIP 을 로드하지 않는다. 노드가 발행한 두 토픽을 읽어서 찍기만 한다.

구독:
    /instance_embedding_set   meridian_msgs/InstanceEmbeddingSet  (숫자)
    /clip_semantics           vision_msgs/Detection2DArray        (노드의 zero-shot 결과)

실행 (torch/clip 불필요):
    source /opt/ros/humble/setup.bash
    source ~/meridian/install/setup.bash
    ros2 run meridian_clip embedding_monitor
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import rclpy

from meridian_msgs.msg import InstanceEmbeddingSet
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from vision_msgs.msg import Detection2DArray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="clip_inference_node 출력 모니터 (CLIP 로드 없음)",
    )

    parser.add_argument("--embedding-topic", default="/instance_embedding_set")
    parser.add_argument("--semantics-topic", default="/clip_semantics")

    parser.add_argument(
        "--max-segments",
        type=int,
        default=8,
        help="한 프레임에서 표시할 세그먼트 수",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=1.0,
        help="화면 갱신 간격(초)",
    )

    return parser


class EmbeddingMonitor(Node):
    def __init__(self, arguments: argparse.Namespace) -> None:
        super().__init__("embedding_monitor")

        self.max_segments = arguments.max_segments

        self.embedding_count = 0
        self.semantics_count = 0

        self.last_embedding = None
        self.last_semantics = None

        self.create_subscription(
            InstanceEmbeddingSet,
            arguments.embedding_topic,
            self.embedding_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Detection2DArray,
            arguments.semantics_topic,
            self.semantics_callback,
            qos_profile_sensor_data,
        )

        self.create_timer(arguments.period, self.report)

        self.get_logger().info(
            f"Subscribed: {arguments.embedding_topic}, "
            f"{arguments.semantics_topic}"
        )

    def embedding_callback(self, msg: InstanceEmbeddingSet) -> None:
        self.embedding_count += 1
        self.last_embedding = msg

    def semantics_callback(self, msg: Detection2DArray) -> None:
        self.semantics_count += 1
        self.last_semantics = msg

    def report(self) -> None:
        msg = self.last_embedding

        if msg is None:
            print(
                "아직 메시지 없음 "
                "(clip_inference_node / fastsam / 카메라가 떠 있는지 확인)",
                flush=True,
            )
            return

        count = len(msg.segment_ids)
        dim = int(msg.embedding_dim)

        print(flush=True)
        print(
            f"embeddings #{self.embedding_count}  "
            f"stamp={msg.timestamp.sec}.{msg.timestamp.nanosec:09d}  "
            f"segments={count}  dim={dim}  model={msg.embedding_model_id}",
            flush=True,
        )

        if count:
            embeddings = np.asarray(
                msg.embeddings, dtype=np.float32
            ).reshape(count, dim)

            norms = np.linalg.norm(embeddings, axis=1)

            print(
                f"  L2 norm: min={norms.min():.4f} max={norms.max():.4f}"
                "   (정규화가 켜져 있으면 1.0)",
                flush=True,
            )
        else:
            print("  (positive segment 없음)", flush=True)

        semantics = self.last_semantics

        if semantics is None:
            print(
                "  semantics 없음 "
                "(노드를 --no-publish-semantics 로 띄웠는지 확인)",
                flush=True,
            )
            return

        print(
            f"  semantics #{self.semantics_count}  "
            f"detections={len(semantics.detections)}",
            flush=True,
        )

        for detection in semantics.detections[:self.max_segments]:
            best = "  |  ".join(
                f"{result.hypothesis.class_id} {result.hypothesis.score:.3f}"
                for result in detection.results
            )

            print(f"    segment {detection.id:<4} -> {best}", flush=True)

        if len(semantics.detections) > self.max_segments:
            print(
                f"    ... 외 {len(semantics.detections) - self.max_segments}개",
                flush=True,
            )


def main(args=None) -> int:
    rclpy.init(args=args)

    # ros2 run/launch 가 붙이는 --ros-args 부분을 걷어내고 우리 인자만 파싱
    argv = remove_ros_args(args=sys.argv)
    arguments = build_parser().parse_args(argv[1:])

    node = EmbeddingMonitor(arguments)

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
