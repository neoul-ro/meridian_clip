# meridian_clip

CLIP placeholder embedding node: joins RGB frames with segment labels by `header.stamp` and publishes a deterministic mean-RGB stand-in embedding per instance. Real CLIP is TBD.

## I/O

| Topic | Type | Direction |
| --- | --- | --- |
| `/camera/rgb` | `sensor_msgs/Image` (rgb8) | sub |
| `/segment_image` | `sensor_msgs/Image` (mono8) | sub |
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
