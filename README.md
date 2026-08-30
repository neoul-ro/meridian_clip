# meridian_clip

세그먼트마다 **CLIP ViT-B/32 임베딩 512차원**을 붙여 발행하는 ROS 2 Humble 패키지.

```
/camera/rgb      (rgb8)  ──┐
                           ├──▶ clip_inference_node ──▶ /instance_embedding_set
/segment_image   (mono8) ──┘                        └──▶ /clip_semantics (기본 꺼짐)
```

clone 과 워크스페이스 구성은 [워크스페이스 README](../README.md) 를 먼저 본다.
아래는 clone 이 끝난 뒤부터다. `$MERIDIAN_WS` 는 워크스페이스 경로다
(`cd <워크스페이스> && export MERIDIAN_WS=$PWD`).

---

## 1. 실행 환경

이 노드는 **`torch` / `clip` / `tensorrt` 가 있는 python3.10** 에서만 돈다.
Jetson 은 JetPack 이 시스템 파이썬에 깔아 주므로 아래 A 는 건너뛴다.

```bash
# A. conda 를 쓰는 경우 (일반 PC). 환경 이름은 자유
conda create -y -n clip python=3.10 && conda activate clip
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/openai/CLIP.git tensorrt-cu11==10.13.0.35
pip install numpy==1.26.4 scipy pillow opencv-python onnx onnxruntime onnxsim ftfy regex tqdm

# B. Jetson / 시스템 파이썬
sudo apt install python3-scipy python3-opencv
/usr/bin/python3 -m pip install git+https://github.com/openai/CLIP.git ftfy regex tqdm onnx onnxruntime onnxsim
```

확인 (A 는 `conda activate clip` 한 터미널에서):

```bash
python3 -c "import torch, clip, tensorrt; print(torch.__version__, torch.cuda.is_available())"
```

> CUDA 12 면 `+cu121` / `tensorrt-cu12` 로 바꾼다.
> 자세한 설명과 플랫폼별 주의사항은 **[docs/setup.md](docs/setup.md)**.

---

## 2. 빌드

```bash
cd $MERIDIAN_WS
source /opt/ros/humble/setup.bash
colcon build --packages-select meridian_msgs meridian_clip \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
```

`--cmake-args` 두 개는 **conda 가 깔린 머신에서 필수다** (없으면 `meridian_msgs` 가
`No module named 'em'` 으로 죽는다). conda 가 없으면 생략해도 된다.

실행 파이썬(shebang)은 빌드가 알아서 찾아 이렇게 알려 준다:

```
[meridian_clip] 플랫폼=일반 PC · 런타임 파이썬 자동 감지: /home/you/miniconda3/envs/clip/bin/python
[meridian_clip]   있는 모듈: clip, tensorrt, torch
```

> **`--symlink-install` 은 쓰지 않는다** — 그 모드는 shebang 자동 감지를 건너뛴다.

---

## 3. 모델과 엔진 만들기

```bash
conda activate clip                       # A 만. B 는 생략
cd $MERIDIAN_WS/src/meridian/meridian_clip

python3 meridian_clip/download_weights.py                        # ViT-B-32.pt, 338MB
python3 meridian_clip/export_onnx.py  --part visual_pooled_value # 약 1분
python3 meridian_clip/build_engine.py --part visual_pooled_value \
    --min-batch 1 --opt-batch 32 --max-batch 64                  # 약 4분
```

`<패키지 루트>/models/` 에 만들어진다. `--part` 는 쓸 pooling 모드 것만:

| `--part` | 필요한 경우 |
|---|---|
| `visual_pooled_value` | **`mask_weighted_value` (기본 모드)** |
| `visual_pooled` | `mask_weighted_patch` |
| `visual` | `cls` |
| `text` | `--publish-semantics true` 로 zero-shot 라벨을 낼 때 |

엔진은 **GPU·TensorRT 버전에 종속**이라 머신이 바뀌면 다시 만든다.
엔진 없이 먼저 돌려 보려면 `--backend torch` (1.7배 느림).

---

## 4. 실행

```bash
cd $MERIDIAN_WS
source /opt/ros/humble/setup.bash && source install/setup.bash
export MERIDIAN_CLIP_MODEL_DIR=$MERIDIAN_WS/src/meridian/meridian_clip/models

ros2 launch meridian_clip clip_inference.launch.py
```

`MERIDIAN_CLIP_MODEL_DIR` 은 `~/.bashrc` 에 넣어 두면 매번 안 해도 된다. 모델은
2GB 가 넘어서 colcon install 트리로 복사하지 않기 때문에 이 한 줄이 필요하다.

값 바꾸기 / 인자 직접 주기:

```bash
ros2 launch meridian_clip clip_inference.launch.py pooling_mode:=cls publish_semantics:=true
ros2 run meridian_clip clip_inference_node --backend torch --debug-save-dir /tmp/clip_debug
```

전체 인자는 `ros2 run meridian_clip clip_inference_node --help`.

---

## 5. 동작 확인

```bash
ros2 run meridian_clip embedding_monitor
```

세그먼트 수 / `embedding_model_id` / L2 norm 을 찍는다. L2 가 1.0000 이면 정상이고,
**어떤 pooling 으로 돌고 있는지 확인하는 가장 빠른 방법**이다.

`아직 메시지 없음` 만 나오면 입력이 없는 것이다 —
`ros2 topic hz /camera/rgb` 와 `ros2 topic hz /segment_image` 를 먼저 본다.

---

## 6. 나오는 것

**`/instance_embedding_set`** — `meridian_msgs/InstanceEmbeddingSet`

```
timestamp          : 카메라 촬영 시각 (동기화 키)
embedding_model_id : openai_clip_vit_b32_mask_weighted_value_v1
embedding_dim      : 512
segment_ids        : [1, 2, 3, ...]            uint8[N]
embeddings         : float32[N*512], row-major  ← [N, 512] 로 reshape
```

`segment_ids[n]` 이 `n` 번째 행의 주인이고, 그 값은 같은 timestamp 의 `/segment_image`
픽셀 값과 일치한다.

**`/clip_semantics`** — `vision_msgs/Detection2DArray`, zero-shot 라벨.
**기본은 꺼져 있다** (프레임당 5.1ms, 텍스트 엔진도 필요).

| 노드 | 하는 일 |
|---|---|
| **`clip_inference_node`** | 본체. torch/clip/tensorrt 필요 |
| `embedding_monitor` | `/instance_embedding_set` 구독 → 요약 출력 |
| `clip_label_viz` | `/clip_semantics` → `/clip_label_overlay` |

---

## 7. 자주 막히는 것

| 증상 | 원인 |
|---|---|
| 스크립트에서 `No module named 'numpy'` | `python3` 가 conda **base**. `conda activate clip` |
| 빌드 중 `No module named 'em'` | §2 의 `--cmake-args` 두 개를 빠뜨림 |
| `ros2 run` 에서 `No module named 'clip'` | shebang 이 잘못된 파이썬. 빌드 로그의 자동 감지 결과 확인 |
| `FileNotFoundError: ... 엔진이 없습니다` | §3 미실행이거나 `MERIDIAN_CLIP_MODEL_DIR` 불일치. 에러가 찾아본 경로를 다 찍어 준다 |
| 노드는 떠 있는데 임베딩 0개 | `/camera/rgb` 와 `/segment_image` **해상도 불일치** — 조용히 프레임을 버린다 |
| `Alignment matrix ...: none` 경고 | 정상. 이 저장소에 `align_*.npy` 가 없다 |

각 항목의 재현·원인·확인 명령은 **[docs/troubleshooting.md](docs/troubleshooting.md)**.

---

## 더 읽을 것

| | |
|---|---|
| [docs/setup.md](docs/setup.md) | 환경 구성 상세, 경로 규약, 내부 동작, 테스트 |
| [docs/troubleshooting.md](docs/troubleshooting.md) | 에러별 원인과 확인 방법 |
| [docs/pooling.md](docs/pooling.md) | pooling 3종, 정렬 행렬, crop 정책 (측정 근거) |
| [docs/performance.md](docs/performance.md) | 단계별 소요시간, 최적화 기록 |

## 알려진 제약

- `models/align_*.npy` 와 그것을 만드는 `fit_alignment.py` 가 이 저장소에 없다.
  기본 실행에서 텍스트 정렬은 꺼진 상태로 동작한다 (zero-shot top-1 90.5% → 87.9%).
- 세그먼테이션 품질이 현재 병목이다. `--min-segment-area` / `--max-segments` 로
  거르는 것이 가장 값싼 개선이다.
- crop-and-encode 라 비용이 세그먼트 수 N 에 비례한다. 진짜 dense feature 가 아니다.
