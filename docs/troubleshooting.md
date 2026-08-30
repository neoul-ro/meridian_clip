# 문제 해결

전부 실제로 재현된 것들이다. [README §7](../README.md#7-자주-막히는-것) 의 표를
항목별로 풀어 쓴 것이고, 배경 설명은 [setup.md](setup.md) 에 있다.

---

## `ModuleNotFoundError: No module named 'numpy'` — 스크립트를 돌릴 때

`python3` 가 conda **base** 를 가리키고 있다. base 에는 아무것도 없다.

```bash
which python3          # ~/miniconda3/bin/python3 이면 이 경우
conda activate clip    # → ~/miniconda3/envs/clip/bin/python3 로 바뀐다
```

**`conda deactivate` 로는 해결되지 않는다** — `~/.bashrc` 의 conda init 이
`~/miniconda3/bin` 을 PATH 에 남겨 두기 때문이다
([setup.md §1](setup.md#conda-를-깔았다면-python3-가-어디를-가리키는지-확인한다)).

---

## `ModuleNotFoundError: No module named 'em'` — `colcon build` 중

같은 PATH 문제가 `meridian_msgs` 의 메시지 생성기에서 터진 것이다.

```bash
colcon build --packages-select meridian_msgs meridian_clip \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
```

**`-DPython3_EXECUTABLE` 만 주면 이번엔 `numpy` 로 죽는다** — 두 생성기가 각각 다른
변수를 읽으므로 둘 다 줘야 한다 ([setup.md §2](setup.md#2-빌드에---cmake-args-두-개가-필요한-이유)).

---

## `ModuleNotFoundError: No module named 'clip'` (또는 `tensorrt`) — `ros2 run` 할 때

shebang 이 torch/clip 없는 파이썬을 가리킨다.

```bash
head -1 install/meridian_clip/lib/meridian_clip/clip_inference_node
```

빌드 로그에서 자동 감지가 무엇을 골랐는지 확인한다:

```
[meridian_clip] 플랫폼=... · 런타임 파이썬 자동 감지: ...
[meridian_clip]   있는 모듈: ...  ·  없는 모듈: ...
```

원인은 보통 셋 중 하나다:

- 조건을 만족하는 파이썬이 없다 → 환경을 만들고 다시 빌드 (README §1)
- `--symlink-install` 로 빌드했다 → 그 모드는 자동 감지를 건너뛴다. 빼고 다시 빌드
- 감지가 엉뚱한 것을 골랐다 → `MERIDIAN_CLIP_PYTHON=<경로>` 로 직접 지정

---

## `FileNotFoundError: ... pooling 용 TensorRT 엔진이 없습니다`

에러 메시지가 **찾아본 디렉터리를 전부 찍어 준다:**

```
모델 디렉터리로 찾아본 곳:
  .../install/meridian_clip/share/meridian_clip/models  (디렉터리는 있으나 모델 파일이 없음)
  .../install/meridian_clip/lib/python3.10/site-packages/models  (없음)
```

- 엔진을 안 만들었다 → README §3
- 만들었는데 못 찾는다 → `MERIDIAN_CLIP_MODEL_DIR` 이 그 위치를 가리키는지 확인
  ([setup.md §4](setup.md#4-모델-디렉터리-탐색-규칙))
- 급하면 `--backend torch` 로 우회 (`.pt` 만 있으면 된다)

---

## 모델 경로가 엉뚱한 곳(`/home/누군가/meridian/models/clip` 같은)으로 나온다

**옛날 빌드다.** 예전에는 launch 기본값이 `~/meridian/models/clip` 로 박혀 있었다.
지금은 소스 어디에도 절대경로가 없다.

```bash
cd $MERIDIAN_WS && rm -rf build install log
colcon build --packages-select meridian_msgs meridian_clip \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
```

`ros2 launch` 는 실행할 명령 전체를 로그에 찍으므로, `--model-dir` 값이 무엇으로
나오는지 거기서 바로 볼 수 있다.

---

## `ModuleNotFoundError: No module named 'meridian_msgs'`

둘 중 하나다:

- `source install/setup.bash` 를 안 했다
- 실행 파이썬이 3.10 이 아니다. ROS 2 Humble 이 메시지를
  `lib/python3.10/site-packages` 에 깔기 때문에 버전이 맞아야 한다

```bash
head -1 install/meridian_clip/lib/meridian_clip/clip_inference_node   # 이 파이썬으로
<그 경로> --version                                                    # 확인
```

---

## 노드는 떠 있는데 임베딩이 0개

`/camera/rgb` 와 `/segment_image` 의 **해상도가 다르면 프레임을 통째로 버린다.**
경고가 나오지 않아서 겉보기엔 멀쩡하다.

```bash
ros2 topic echo /camera/rgb --field height --once
ros2 topic echo /segment_image --field height --once
```

토픽 자체가 안 오는 경우도 있으니 `ros2 topic hz` 로 먼저 본다.
세그먼테이션 노드가 color 해상도로 라벨맵을 내보내는지 확인한다
([setup.md §6](setup.md#6-노드-안에서-벌어지는-일)).

---

## `embedding_monitor` 가 `아직 메시지 없음` 만 찍는다

입력이 없거나 위 해상도 문제다. 순서대로 확인한다:

```bash
ros2 topic hz /camera/rgb
ros2 topic hz /segment_image
ros2 topic hz /instance_embedding_set
```

---

## `ros2 node list` 가 비어 있다

`ROS_DOMAIN_ID` 가 셸마다 다르면 서로 안 보인다. 모든 터미널에서 같은 값을 쓴다.

---

## `Alignment matrix (text side, Wᵀ): none` 경고

`models/align_*.npy` 가 없다는 뜻이고 **정상 동작이다** — 이 저장소에는 그 파일도,
만드는 도구(`fit_alignment.py`)도 없다. zero-shot top-1 이 90.5% → 87.9% 로
내려가는 것이 전부이고 임베딩 자체는 영향받지 않는다 ([pooling.md](pooling.md)).

---

## 종료할 때 `ExternalShutdownException` 트레이스백

Ctrl-C / SIGTERM 을 받으면 나오는 **표시상의 문제**다. 데이터 손실이나 실패가 아니다.
