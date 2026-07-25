# meridian_clip

CLIP placeholder embedding node: joins RGB-D frames with segment labels by timestamp and publishes a deterministic mean-RGB stand-in embedding per instance. Real CLIP is TBD.

## I/O

| Topic | Type | Direction |
| --- | --- | --- |
| `/rgbd_frame` | `meridian_msgs/RGBDFrame` | sub |
| `/segment_image` | `meridian_msgs/SegmentImage` | sub |
| `/instance_embedding_set` | `meridian_msgs/InstanceEmbeddingSet` | pub |

## Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `embedding_dim` | int | `512` | Embedding vector length D |
| `embedding_model_id` | string | `stub/mean-rgb-v0` | Model/weights/preprocessing configuration ID |

## Run

```
ros2 run meridian_clip clip_node
```
