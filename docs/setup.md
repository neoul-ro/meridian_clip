# 환경 구성 상세

[README](../README.md) 의 §1~§4 를 따라 하다 "왜?" 가 생겼을 때 보는 문서.
증상별 대처는 [troubleshooting.md](troubleshooting.md) 에 있다.

---

## 1. 왜 파이썬을 따로 신경 써야 하나

`clip_inference_node` 는 `torch` / `clip` / `tensorrt` 가 있는 파이썬에서만 돈다.
그게 어느 파이썬인지가 플랫폼마다 다르다:

| 플랫폼 | torch / tensorrt 가 있는 곳 |
|---|---|
| **Jetson** (JetPack) | 시스템 파이썬 `/usr/bin/python3` — conda 가 필요 없다 |
| **일반 PC** (x86 + dGPU) | 보통 conda 환경. 시스템 파이썬에 직접 깔아도 된다 |

**어느 쪽인지 직접 판단할 필요는 없다** — `colcon build` 가 판별해서 알려 준다(§3).
README 의 A/B 는 "무엇을 설치해야 하는가" 를 나눈 것일 뿐이다.

**파이썬 버전은 반드시 3.10 이어야 한다.** ROS 2 Humble 이 생성하는 `meridian_msgs`
파이썬 모듈이 `lib/python3.10/site-packages` 에 깔리기 때문이다. 다르면
`ModuleNotFoundError: No module named 'meridian_msgs'` 가 난다.

> **venv 로도 된다.** 다만 `rclpy` 등 ROS 2 파이썬 패키지를 보려면
> `python3 -m venv --system-site-packages` 로 만들어야 한다.

### conda 를 깔았다면: `python3` 가 어디를 가리키는지 확인한다

conda 를 설치하면 `~/.bashrc` 가 `~/miniconda3/bin` 을 **PATH 맨 앞**에 넣는다.
그래서 아무 환경도 activate 하지 않은 터미널의 `python3` 는 시스템 파이썬이 아니라
**conda base** 다. base 에는 torch 도 numpy 도 없다.
**`conda deactivate` 를 해도 PATH 는 그대로라 바뀌지 않는다.**

```bash
conda activate clip
which python3        # → ~/miniconda3/envs/clip/bin/python3   ← 이게 나와야 한다
```

이것이 이 패키지에서 가장 자주 막히는 지점이고, 아래 §2 의 `--cmake-args` 가
필요한 이유이기도 하다.

### Miniconda 설치 (없다면)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
exec bash
```

---

## 2. 빌드에 `--cmake-args` 두 개가 필요한 이유

```bash
colcon build --packages-select meridian_msgs meridian_clip \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
```

`meridian_msgs` 의 메시지 생성기는 CMake 가 찾은 `python3` 로 도는데, §1 의 PATH
문제 때문에 그게 conda base 로 잡힌다. 그러면

1. `rosidl_adapter` 가 `ModuleNotFoundError: No module named 'em'` 로 죽는다
2. `-DPython3_EXECUTABLE` 만 고치면, 이번엔 `rosidl_generator_py` 가 **다른 변수**를
   읽어서 `numpy` 로 죽는다

두 생성기가 각각 다른 변수를 보므로 **둘 다** 줘야 한다.
conda 가 아예 없는 머신에서는 생략해도 되고, 있어도 무해하다.

---

## 3. 실행 파이썬(shebang)은 빌드가 찾는다

`ros2 run` 이 실행할 콘솔 스크립트의 첫 줄 `#!...` 에 §1 에서 만든 파이썬이 박혀야
한다. **이건 손으로 설정하지 않는다** — `setup.py` 의
`BuildScriptsWithRuntimePython` 이 빌드할 때마다 판별한다.

```
[meridian_clip] 플랫폼=일반 PC · 런타임 파이썬 자동 감지: /home/you/miniconda3/envs/clip/bin/python
[meridian_clip]   있는 모듈: clip, tensorrt, torch
```

Jetson 이면 `플랫폼=Jetson · ... /usr/bin/python3` 이 나온다
(`/etc/nv_tegra_release` 또는 `/proc/device-tree/model` 로 판별).

**찾는 순서**

1. `MERIDIAN_CLIP_PYTHON` 환경변수 — 직접 지정. **검사를 통과하지 못해도 이걸 쓴다**
   (명시적 지정이 자동 감지를 이기는 편이 덜 놀랍다). 대신 무엇이 모자란지 경고한다
2. 지금 활성화된 conda / venv (`$CONDA_PREFIX`, `$VIRTUAL_ENV`)
3. 흔한 conda 설치 위치의 **모든** 환경 (`~/miniconda3/envs/*`, `~/anaconda3/envs/*`,
   `~/miniforge3/envs/*`, `~/mambaforge/envs/*`). 이름이 `clip` 일 필요 없다
4. `/usr/bin/python3` — Jetson/JetPack, 또는 시스템에 직접 설치한 경우
5. colcon 을 띄운 파이썬

**조건** — `torch` 와 `clip` 을 모두 가질 것, ROS 와 같은 파이썬 마이너 버전일 것.
`tensorrt` 까지 있으면 그것을 우선한다. 검사는 실제 import 가 아니라
`importlib.util.find_spec` 이라 후보당 0.01초다.

조건을 만족하는 파이썬이 하나도 없으면 **빌드를 실패시키지 않고 경고만** 한다
(모델 없이 문법만 확인하는 빌드를 막지 않기 위해).

직접 지정:

```bash
MERIDIAN_CLIP_PYTHON=/path/to/python colcon build --packages-select meridian_clip
```

### shebang 만은 절대경로일 수밖에 없다

나머지 경로(모델, 가중치, 워크스페이스)는 전부 유도해서 쓰지만 shebang 은 다르다 —
커널이 `#!` 뒤의 경로를 **그대로 exec** 하기 때문에 상대경로도 `$HOME` 같은 변수도
확장되지 않는다. 그리고 파이썬 인터프리터는 애초에 저장소 밖에 있는 물건이라 패키지
기준 상대경로라는 것이 존재하지 않는다.

그래서 이 값은 **소스에 적어 두지 않고 빌드 때 채운다.** 예전에는 `setup.cfg` 의
`[build_scripts] executable` 에 손으로 적게 했는데, clone 한 사람이 자기 플랫폼을
판단해야 했고 그 값을 커밋하면 남의 빌드가 깨졌다.

### `--symlink-install` 을 쓰면 안 되는 이유

그 모드는 `setup.py develop` 경로로 가서 `build_scripts` 명령을 **아예 거치지
않는다.** shebang 이 `/usr/bin/python3` 로 박히고 위 자동 감지가 통째로 무효가 된다.

---

## 4. 모델 디렉터리 탐색 규칙

**소스에 절대경로는 없다.** 노드와 launch 는 같은 함수
([`meridian_clip/model_paths.py`](../meridian_clip/model_paths.py))로 이 순서로 찾는다:

| | 후보 | 언제 맞나 |
|---|---|---|
| 1 | `--model-dir` / `model_dir:=` 인자 | 한 번만 다르게 띄울 때 |
| 2 | 환경변수 `MERIDIAN_CLIP_MODEL_DIR` | **평범한 `colcon build` — 보통 이것** |
| 3 | `<share>/meridian_clip/models` | install 트리 |
| 4 | `<패키지 루트>/models` | 소스에서 직접 실행 |

모델이 **실제로 들어 있는** 첫 후보를 고른다 — 디렉터리만 있고 비어 있으면 건너뛴다
(`setup.py` 가 `share/.../models` 를 비어 있는 채로 만들어 두기 때문).

3번과 4번은 `ros2 run` / `ros2 launch` 로는 잘 안 맞는다. 모델이 2GB 가 넘어서 colcon
install 트리로 복사시키지 않기 때문이다. 그래서 2번을 `~/.bashrc` 에 넣어 둔다:

```bash
export MERIDIAN_CLIP_MODEL_DIR=$MERIDIAN_WS/src/meridian/meridian_clip/models
```

`meridian_seg` 의 `MERIDIAN_SEG_ENGINE` 과 같은 규약이다.
못 찾으면 **어디를 봤는지 전부 찍고 죽는다:**

```
FileNotFoundError: mask_weighted_value pooling 용 TensorRT 엔진이 없습니다: ...

모델 디렉터리로 찾아본 곳:
  .../install/meridian_clip/share/meridian_clip/models  (디렉터리는 있으나 모델 파일이 없음)
  .../install/meridian_clip/lib/python3.10/site-packages/models  (없음)
```

### 경로 규약 요약

| | 어떻게 정해지나 | 커밋되나 |
|---|---|---|
| 워크스페이스 위치 | 아무 데나. 코드가 참조하지 않는다 | — |
| 패키지 소스 | `__file__` 에서 유도 | ✔ (상대) |
| 모델 디렉터리 | 위 4단계 탐색 (`model_paths.py`) | ✔ (상대) |
| install 트리 | `ament_index` 로 조회 | ✔ (상대) |
| **실행 파이썬 (shebang)** | 빌드가 자동 감지 (§3) | — (소스에 없음) |
| 모델을 다른 곳에 뒀을 때 | `MERIDIAN_CLIP_MODEL_DIR` | — (셸 설정) |

---

## 5. 엔진 빌드 확인값

```
[ok  ] .../clip_vit_b32_visual_pooled_value_fp16.engine  (172.0 MiB)
[parity] onnx fp32 vs tensorrt
  batch=32  embeddings      cos=0.999990  max_abs_diff=1.220e-03
[bench]
  tensorrt fp16 :   8.64 ms  (batch=32)
  torch fp16    :  14.91 ms  (batch=32)
  speedup       :   1.73x
```

`cos` 가 0.999 대면 정상이다. `--opt-batch` 는 노드의 `--batch-size` 와 **같은 값**
이어야 한다 — 다르면 TensorRT 가 튜닝하지 않은 배치로 돈다
([performance.md](performance.md)).

다른 곳에 만들려면 `--output-dir` (download_weights) / `--output` (export_onnx) /
`--engine` (build_engine).

---

## 6. 노드 안에서 벌어지는 일

```
1) 같은 timestamp 의 color / labels 를 짝짓는다   (버퍼 기반, message_filters 미사용)
2) 두 이미지의 해상도가 다르면 프레임을 버린다      ← 조용한 실패의 주범
3) segment_id 마다 bbox 로 crop                  (--crop-policy)
4) 224x224 로 맞춘다                             (--crop-fit)
5) ViT-B/32 → 49개 patch token 을 하나로 합침      (--pooling-mode)
6) ln_post → @ proj → 512차원 → L2 정규화
7) 발행
```

**2번**은 경고 없이 프레임 전체를 버린다. 노드는 멀쩡히 떠 있는데 임베딩이 0개 나온다.
세그먼테이션 노드가 color 해상도로 라벨맵을 내보내는지 확인한다.

3~4번은 **세그먼트마다** 돈다. 프레임당 한 번이 아니라 N번이고, 그래서 이 두 단계가
전체 시간의 절반을 넘는다.

`desired_encoding="rgb8"` 을 명시하는 이유: OpenCV 관례는 BGR 이지만 **CLIP 은 RGB
로 학습**됐다. 순서가 뒤집히면 임베딩이 조용히 망가진다.

> CLIP 코사인 유사도는 원래 **0.2–0.35 좁은 대역**에 분포한다. 절대값이 아니라
> 후보들 사이의 **순위**가 의미를 갖는다.

---

## 7. 주요 인자

설정은 ROS 파라미터가 아니라 **모듈 상수 + argparse** 다 (`remove_ros_args` 후 파싱).
전체 목록은 `ros2 run meridian_clip clip_inference_node --help`.

| 인자 | 기본값 | 뜻 |
|---|---|---|
| `--color-topic` / `--segment-topic` | `/camera/rgb` / `/segment_image` | 입력 |
| `--model-dir` | `""` (§4 의 탐색 순서) | 모델 5종의 디렉터리를 한 번에 지정 |
| `--backend` | `tensorrt` | `torch` = `.pt`, `tensorrt` = `.engine` |
| `--pooling-mode` | `mask_weighted_value` | `cls` / `mask_weighted_patch` 도 가능 |
| `--batch-size` | `32` | **엔진의 `--opt-batch` 와 같아야 한다** |
| `--crop-policy` | `bbox` | `masked_bbox` (cls 쓸 때 권장) / `masked_full` |
| `--crop-fit` | `pad` | `centercrop` / `stretch` |
| `--preprocess-path` | `pil` | `interp_aa` / `roi_align` 은 GPU 경로 |
| `--publish-semantics` | `false` | zero-shot 라벨. 켜면 텍스트 엔진 필요 |
| `--prompts` / `--prompt-file` | 18개 기본값 | zero-shot 후보 |
| `--text-alignment-matrix` | `""` (모드별 자동) | `none` 이면 끔 |
| `--min-segment-area` / `--max-segments` | `0` / `0` | 인코딩할 세그먼트 필터 (0 = 전부) |
| `--empty-mask-fallback` | `cls` | `skip` / `error` 도 있음 |
| `--qos-depth` | `10` (launch 는 `1`) | 구독 큐 깊이 |
| `--debug-save-dir` | `""` | crop/mask/occupancy PNG 저장 |
| `--stats-every` | `0` | N 프레임마다 단계별 소요시간 출력 |

QoS 는 구독 BEST_EFFORT / 발행 RELIABLE 이라 어느 짝과도 호환된다.
launch 는 `arguments=` 로 넘기므로 빈 문자열 인자나 `BooleanOptionalAction` 플래그
(`--debug-save-dir`, `--reliable-input`)를 전달할 수 없다 — `ros2 run` 으로 준다.

> `launch/clip_inference.launch.py` 는 노드 기본값과 달리
> `preprocess_path: roi_align` 을 쓴다. Jetson 에서 전처리가 2배 빠르지만 임베딩이
> 비트 동일하지 않으므로, 저장된 임베딩과 섞을 거면 `preprocess_path:=pil` 로 맞춘다.

---

## 8. 테스트

```bash
cd $MERIDIAN_WS/src/meridian/meridian_clip
source /opt/ros/humble/setup.bash          # ament_flake8 / ament_pep257 이 여기 있다
/usr/bin/python3 -m pytest test/ -q
```

현재 상태 (2026-08-30): **11 passed, 2 failed, 1 skipped.**

- `test_mask_pooling.py` — 통과. CLIP 가중치 없이 도는 순수 torch 연산 검증
- `test_flake8.py` — 실패 1건 (`clip_backend.py:703 E303 too many blank lines`)
- `test_pep257.py` — 실패 88건, 전부 `D213` (docstring 요약 위치). 프로젝트 전반의
  기존 스타일이라 새 breakage 가 아니다
