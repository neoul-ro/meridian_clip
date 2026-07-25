import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge

from meridian_msgs.msg import RGBDFrame, SegmentImage, InstanceEmbeddingSet

# buffer keys are bounded to this many entries, dropping the oldest on overflow
BUFFER_LIMIT = 200


class MeridianClip(Node):

    def __init__(self):
        super().__init__('meridian_clip')

        self.declare_parameter('embedding_dim', 512)
        self.declare_parameter('embedding_model_id', 'stub/mean-rgb-v0')
        self.embedding_dim = self.get_parameter('embedding_dim').value
        self.embedding_model_id = self.get_parameter('embedding_model_id').value

        self.bridge = CvBridge()
        self.rgbd_buffer = {}
        self.segment_buffer = {}

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        self.rgbd_sub = self.create_subscription(
            RGBDFrame, '/rgbd_frame', self.rgbd_callback, qos)
        self.segment_sub = self.create_subscription(
            SegmentImage, '/segment_image', self.segment_callback, qos)
        self.embedding_pub = self.create_publisher(
            InstanceEmbeddingSet, '/instance_embedding_set', qos)

        self.get_logger().info(
            'meridian_clip started: embedding_dim=%d embedding_model_id=%s' %
            (self.embedding_dim, self.embedding_model_id))

    def rgbd_callback(self, msg):
        key = (msg.timestamp.sec, msg.timestamp.nanosec)
        self._buffer_put(self.rgbd_buffer, key, msg)
        if key in self.segment_buffer:
            self.process(key)

    def segment_callback(self, msg):
        key = (msg.timestamp.sec, msg.timestamp.nanosec)
        self._buffer_put(self.segment_buffer, key, msg)
        if key in self.rgbd_buffer:
            self.process(key)

    def _buffer_put(self, buffer, key, msg):
        buffer[key] = msg
        if len(buffer) > BUFFER_LIMIT:
            oldest_key = next(iter(buffer))
            del buffer[oldest_key]

    def process(self, key):
        rgbd_msg = self.rgbd_buffer.pop(key)
        segment_msg = self.segment_buffer.pop(key)

        rgb = self.bridge.imgmsg_to_cv2(rgbd_msg.rgb, desired_encoding='rgb8')
        labels = self.bridge.imgmsg_to_cv2(segment_msg.labels, desired_encoding='mono8')

        unique_ids = np.unique(labels)
        unique_ids = unique_ids[unique_ids > 0]

        segment_ids = []
        rows = []
        for seg_id in unique_ids:
            mask = labels == seg_id
            mean_rgb = rgb[mask].mean(axis=0)
            base = mean_rgb / 255.0
            embedding = np.resize(base, self.embedding_dim).astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            segment_ids.append(int(seg_id))
            rows.append(embedding)

        if rows:
            matrix = np.stack(rows, axis=0)
        else:
            matrix = np.zeros((0, self.embedding_dim), dtype=np.float32)

        out = InstanceEmbeddingSet()
        out.timestamp = rgbd_msg.timestamp
        out.embedding_model_id = self.embedding_model_id
        out.embedding_dim = self.embedding_dim
        out.segment_ids = segment_ids
        out.embeddings = matrix.flatten().astype(np.float32).tolist()
        self.embedding_pub.publish(out)

        self.get_logger().info(
            'published instance_embedding_set: %d segments (rgbd_buffer=%d segment_buffer=%d)' %
            (len(segment_ids), len(self.rgbd_buffer), len(self.segment_buffer)),
            throttle_duration_sec=5.0)


def main():
    rclpy.init()
    node = MeridianClip()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
