#!/usr/bin/env python3

"""clip_inference_node 가 잘 도는지 눈으로 확인하는 시각화 relay.

이 툴은 CLIP 을 로드하지 않는다. 텍스트/이미지 인코딩은 전부 노드가 하고,
여기서는 노드가 발행한 결과를 색과 글자로 그리기만 한다.

구독:
    color                     sensor_msgs/Image                 (rgb8/bgr8)
    /segment_image            sensor_msgs/Image (mono8)         (마스크 색칠용)
    /clip_semantics           vision_msgs/Detection2DArray      (노드의 zero-shot 결과)
발행:
    /clip_label_overlay       sensor_msgs/Image

실행 (ROS 만 있으면 되고 torch/clip 은 필요 없다):
    source /opt/ros/humble/setup.bash
    source <워크스페이스>/install/setup.bash
    ros2 run meridian_clip clip_label_viz

보기:
    ros2 run rqt_image_view rqt_image_view /clip_label_overlay
"""

from __future__ import annotations

import argparse
import sys

from collections import OrderedDict

import cv2
import numpy as np

import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


def make_lut() -> np.ndarray:
    """segment_id -> BGR 색 LUT. segment_viz.py 와 같은 시드라 색이 일치한다."""
    rng = np.random.default_rng(42)
    lut = rng.integers(60, 256, size=(256, 3), dtype=np.uint8)
    lut[0] = (0, 0, 0)

    return lut


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="clip_inference_node 결과 시각화 (CLIP 로드 없음)",
    )

    parser.add_argument("--color-topic", default="/camera/rgb")
    parser.add_argument("--segment-topic", default="/segment_image")
    parser.add_argument("--semantics-topic", default="/clip_semantics")
    parser.add_argument("--output-topic", default="/clip_label_overlay")

    parser.add_argument(
        "--max-labels",
        type=int,
        default=10,
        help="큰 bbox 부터 이 개수까지만 이름표를 그린다",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="이 점수 미만이면 이름표를 생략한다",
    )
    parser.add_argument("--buffer-size", type=int, default=30)
    parser.add_argument(
        "--draw-box",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="노드가 보낸 bbox 사각형도 그릴지 (기본: 이름표만)",
    )

    return parser


class ClipLabelViz(Node):
    def __init__(self, arguments: argparse.Namespace) -> None:
        super().__init__("clip_label_viz")

        self.max_labels = arguments.max_labels
        self.min_score = arguments.min_score
        self.buffer_size = arguments.buffer_size
        self.draw_box = arguments.draw_box

        self.bridge = CvBridge()
        self.lut = make_lut()

        # timestamp -> 메시지. 세 입력을 capture time 으로 묶는다.
        self.color_buffer: OrderedDict = OrderedDict()
        self.segment_buffer: OrderedDict = OrderedDict()
        self.semantics_buffer: OrderedDict = OrderedDict()

        self.published_count = 0

        self.publisher = self.create_publisher(
            Image,
            arguments.output_topic,
            10,
        )

        self.create_subscription(
            Image,
            arguments.color_topic,
            self.color_callback,
            qos_profile_sensor_data,
        )
        # /segment_image 는 sensor_msgs/Image (mono8), 픽셀 값 = segment_id
        self.create_subscription(
            Image,
            arguments.segment_topic,
            self.segment_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Detection2DArray,
            arguments.semantics_topic,
            self.semantics_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"Publishing: {arguments.output_topic} "
            f"(semantics from {arguments.semantics_topic})"
        )

    # ----------------------------------------------------------------
    # 입력 결합
    # ----------------------------------------------------------------

    @staticmethod
    def key_of(stamp) -> tuple:
        return (int(stamp.sec), int(stamp.nanosec))

    def store(self, buffer: OrderedDict, key, value) -> None:
        buffer[key] = value

        while len(buffer) > self.buffer_size:
            buffer.popitem(last=False)

    def color_callback(self, msg: Image) -> None:
        key = self.key_of(msg.header.stamp)
        self.store(self.color_buffer, key, msg)
        self.try_render(key)

    def segment_callback(self, msg: Image) -> None:
        key = self.key_of(msg.header.stamp)
        self.store(self.segment_buffer, key, msg)
        self.try_render(key)

    def semantics_callback(self, msg: Detection2DArray) -> None:
        key = self.key_of(msg.header.stamp)
        self.store(self.semantics_buffer, key, msg)
        self.try_render(key)

    def try_render(self, key) -> None:
        if key not in self.color_buffer:
            return
        if key not in self.segment_buffer:
            return
        if key not in self.semantics_buffer:
            return

        color_msg = self.color_buffer.pop(key)
        segment_msg = self.segment_buffer.pop(key)
        semantics_msg = self.semantics_buffer.pop(key)

        try:
            self.render(color_msg, segment_msg, semantics_msg)

        except Exception as error:
            self.get_logger().error(f"render failed: {error}")

    # ----------------------------------------------------------------
    # 그리기
    # ----------------------------------------------------------------

    @staticmethod
    def draw_text(canvas, text, origin, color, scale=0.45) -> None:
        """검은 외곽선 + 색 글씨. 어떤 배경에서도 읽힌다."""
        for thickness, text_color in ((3, (0, 0, 0)), (1, color)):
            cv2.putText(
                canvas,
                text,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

    def render(
        self,
        color_msg: Image,
        segment_msg: Image,
        semantics_msg: Detection2DArray,
    ) -> None:
        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        labels = self.bridge.imgmsg_to_cv2(
            segment_msg, desired_encoding="mono8"
        )

        canvas = color.copy()

        # 세그먼트 마스크를 반투명하게 깐다
        mask = labels > 0

        if mask.any():
            colored = self.lut[labels]
            canvas[mask] = (
                0.65 * color[mask] + 0.35 * colored[mask]
            ).astype(np.uint8)

        # 노드가 보낸 detection 을 큰 것부터 그린다
        detections = sorted(
            semantics_msg.detections,
            key=lambda d: d.bbox.size_x * d.bbox.size_y,
            reverse=True,
        )

        drawn = 0

        for detection in detections:
            if drawn >= self.max_labels:
                break

            if not detection.results:
                continue

            best = detection.results[0]
            score = float(best.hypothesis.score)

            if score < self.min_score:
                continue

            center_x = float(detection.bbox.center.position.x)
            center_y = float(detection.bbox.center.position.y)
            half_width = float(detection.bbox.size_x) / 2.0
            half_height = float(detection.bbox.size_y) / 2.0

            if self.draw_box:
                cv2.rectangle(
                    canvas,
                    (int(center_x - half_width), int(center_y - half_height)),
                    (int(center_x + half_width), int(center_y + half_height)),
                    (255, 255, 255),
                    1,
                )

            # bbox 를 안 그릴 때는 세그먼트 중심에 이름표를 놓는다.
            origin = (
                int(center_x - half_width)
                if self.draw_box
                else int(center_x - half_width * 0.5)
            )

            self.draw_text(
                canvas,
                f"{best.hypothesis.class_id} {score:.2f}",
                (origin, max(int(center_y), 12)),
                (255, 255, 255),
            )

            drawn += 1

        header_text = (
            f"detections={len(semantics_msg.detections)}  labeled={drawn}  "
            f"stamp={semantics_msg.header.stamp.sec}"
        )

        self.draw_text(canvas, header_text, (10, 24), (0, 255, 0), scale=0.6)

        output = self.bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
        output.header = color_msg.header
        self.publisher.publish(output)

        self.published_count += 1

        if self.published_count % 30 == 1:
            self.get_logger().info(
                f"Published overlay: {self.published_count} frames "
                f"(detections={len(semantics_msg.detections)})"
            )


def main(args=None) -> int:
    rclpy.init(args=args)

    # ros2 run/launch 가 붙이는 --ros-args 부분을 걷어내고 우리 인자만 파싱
    argv = remove_ros_args(args=sys.argv)
    arguments = build_parser().parse_args(argv[1:])

    node = ClipLabelViz(arguments)

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
