# meridian_clip

FastSAM 이 만든 세그먼트마다 **CLIP 임베딩 512차원**을 붙여 발행하는 ROS 2 Humble 패키지.

```
/camera/rgb            (rgb8)  ──┐
                                 ├──▶ clip_inference_node ──▶ /instance_embedding_set
/segment_image_resized (mono8) ──┘                        └──▶ /clip_semantics
```

frame-local 세그먼트에 **의미(semantic)만** 부여한다. 기하와 영속 identity 는 만들지 않는다.

---

## 0. 먼저 필요한 것

| | |
|---|---|
| [**`meridian_msgs`**](https://github.com/neoul-ro/meridian_msgs) | `InstanceEmbeddingSet` 메시지 정의. **빌드 의존이라 없으면 `colcon build` 가 실패한다.** 같은 워크스페이스 `src/` 에 두고 먼저 빌드한다 |
| 세그먼테이션 노드 | `/segment_image_resized` (mono8) 발행자. 이 저장소에는 없다 — §2 의 해상도 일치 조건을 맞춰 주는 것이 `segment_resize_node` 다 |
| conda `clip` 환경 | torch / clip / tensorrt. **빌드 파이썬과 다르다** — §8 을 먼저 읽을 것 |
| 모델 바이너리 | `.pt` / `.onnx` / `.engine` 는 커밋되지 않는다. §7 로 재생성한다 |

```bash
cd ~/meridian/src
git clone https://github.com/neoul-ro/meridian_msgs.git
git clone https://github.com/neoul-ro/meridian_clip.git

# setup.cfg 의 conda 경로를 본인 것으로 먼저 고친다 (§8)
cd ~/meridian && colcon build --packages-select meridian_msgs meridian_clip
```

`models/align_*.npy` 정렬 행렬은 저장소에 포함돼 있다 (각 1.1MB). 재생성에는
VOC2012 가 필요하지만(§3), 그대로 쓰면 기본 zero-shot 라벨링이 바로 동작한다.

---

## 1. 무엇이 나오나

**`/instance_embedding_set`** — `meridian_msgs/InstanceEmbeddingSet`

```
timestamp          : 카메라 촬영 시각 (동기화 키)
embedding_model_id : openai_clip_vit_b32_mask_weighted_value_v1
embedding_dim      : 512
segment_ids        : [1, 2, 3, ...]            uint8[N]
embeddings         : float32[N*512], row-major  ← [N, 512] 로 reshape
```

`segment_ids[n]` 이 `n` 번째 행의 주인이고, 그 값은 같은 timestamp 의
`/segment_image_resized` 픽셀 값과 일치한다. 즉 **임베딩 ↔ 이미지 위 영역**이 바로 연결된다.

**`/clip_semantics`** — `vision_msgs/Detection2DArray`.
512차원 벡터를 미리 준비한 텍스트 프롬프트와 코사인 비교한 zero-shot 라벨.
bbox 는 원본 해상도 기준.

> CLIP 코사인 유사도는 원래 **0.2–0.35 좁은 대역**에 분포한다. 절대값이 아니라
> 후보들 사이의 **순위**가 의미를 갖는다.

---

## 2. 파이프라인 안에서 벌어지는 일

```
1) 같은 timestamp 의 color / labels 를 짝짓는다   (버퍼 기반, message_filters 미사용)
2) 두 이미지의 해상도가 다르면 프레임을 버린다      ← segment_resize_node 가 필요한 이유
3) segment_id 마다 bbox 로 crop (마스크 밖은 그대로)  (crop_policy=bbox, §4)
4) 224x224 로 맞춘다                             (crop_fit=pad, §4)
5) ViT-B/32 → 49개 patch token 을 하나로 합침      (pooling_mode, §3)
6) ln_post → @ proj → 512차원 → L2 정규화
7) 발행
```

3~4번은 **세그먼트마다** 돈다. 프레임당 한 번이 아니라 N번이고, 그래서 이 두 단계가
전체 시간의 절반을 넘는다 — 단계별 실측은 §6.

`desired_encoding="rgb8"` 을 명시하는 이유: OpenCV 관례는 BGR 이지만 **CLIP 은 RGB 로
학습**됐다. 순서가 뒤집히면 색이 반전된 이미지를 넣는 셈이라 임베딩이 조용히 망가진다.

---

## 3. pooling 방식 3종

49개 patch token 을 하나로 합치는 방법. `--pooling-mode` 로 고른다.

| | `cls` | `mask_weighted_patch` | `mask_weighted_value` **(기본값)** |
|---|---|---|---|
| 쓰는 토큰 | CLS 1개 | 최종 patch 49개 | 마지막 블록의 **value** 49개 |
| 합치는 가중치 | attention 이 정함 | **마스크 점유율** | **마스크 점유율** |
| 마지막 블록 | 그대로 | 그대로 | attention/residual/MLP 건너뜀 |
| 필요한 엔진 | `..._visual_fp16` | `..._visual_pooled_fp16` | `..._visual_pooled_value_fp16` |
| `embedding_model_id` | `..._cls_masked_bbox_v1` | `..._mask_weighted_patch_v1` | `..._mask_weighted_value_v1` |

### 기본값이 `mask_weighted_value` 인 이유

**VOC2012 val, GT 인스턴스 3,420개, 프롬프트 `"a photo of a {}"` 20개,
`--crop-policy bbox`(기본값) / `--crop-fit pad`.**

| pooling | top-1 | macro | AUC | 분리도 ↑ |
|---|---|---|---|---|
| `cls` | 83.07% | 88.08% | 0.9776 | 0.0662 |
| `mask_weighted_patch` | 7.87% | 10.51% | **0.4633** | 0.1158 |
| **`mask_weighted_value`** | **87.92%** | **90.08%** | **0.9897** | **0.1392** |
| `mask_weighted_value` + W | 90.53% | 91.15% | 0.9887 | 0.1392 |

*top-1 = 20개 프롬프트 중 argmax 가 정답인 비율(행 방향), macro = 클래스별 top-1 의
평균. AUC = 정답 프롬프트 열의 점수만으로 그 클래스와 나머지를 갈라내는 능력의 클래스
평균(열 방향). 분리도 = 같은 클래스 쌍 평균 cos − 다른 클래스 쌍 평균 cos, 높을수록
인스턴스 매칭에 유리.*

> **이 표의 AUC 는 열 방향, 즉 text→image 다.** 프롬프트를 고정하고 인스턴스 3,420개를
> 줄 세운 값이라 "맵에 말로 질의를 던지는" 방향과 같다. 반대 방향(image→text)은
> §3 [언어 쪽 성능](#언어-쪽-성능) 참고.
>
> 재현: `python tools/benchmark_pooling.py --modes cls mask_weighted_patch
> mask_weighted_value` (top-1/macro) 와 `python tools/benchmark_language.py`
> (AUC/분리도). 두 도구 모두 `--crop-policy` 기본값이 `bbox` 라 인자 없이 나온다.

`value` 가 **네 지표 전부에서 최고**다. 임베딩 품질(AUC·분리도)뿐 아니라 라벨
정확도(top-1)까지 `cls` 를 앞서므로 기본값으로 둔다.

> **`masked_bbox` 시절과 결론이 달라진 지점.** 마스크 밖을 검게 칠하던 이전 기본값에서는
> `value` 의 top-1 이 81.67% 로 `cls` 의 83.39% 에 **졌고**, "AUC 는 최고지만 top-1 은
> 조금 손해"라는 맞바꿈이 있었다. crop 을 `bbox` 로 바꾸자 `value` 는 81.67% → 87.98% 로
> 오르고 `cls` 는 83.39% → 83.10% 로 제자리라, 맞바꿈이 사라졌다. 검정 마스킹이
> `value` 만 손해 보게 하고 있었던 셈이다 — 가중평균이 이미 마스크 밖을 배제하는데
> 픽셀까지 지우면 물체 경계와 맥락만 잃는다. `patch` 는 12.89% → 7.89% 로 더 나빠진다.

원리는 마지막 블록에서 갈린다. 1~11층은 세 모드가 완전히 동일하다.

- **`cls`** — CLS 를 그대로 쓴다. 안정적이지만 **마스크를 못 쓴다.** 배경을 억제할
  수단이 crop 정책뿐이라 `--crop-policy masked_bbox` 를 같이 줘야 하는데, 그러면
  crop 이 대부분 검정인 세그먼트에서 **CLS 가 검은 배경에 지배당한다.**
- **`mask_weighted_patch`** — 12층을 전부 통과한(attention 혼합 + residual + MLP) 최종
  patch token 을 점유율로 가중평균. **가장 많이 가공된 상태**라 CLS 와 거의 직교하고
  (실측 cos **0.1206**), CLS 전용 투영을 통과하면 텍스트 공간의 엉뚱한 곳에 떨어진다.
- **`mask_weighted_value`** — CLS 의 attention 출력이 곧 각 패치 **value 벡터의 가중합**
  이므로, value 는 이미 투영이 읽을 수 있는 형태다. 12층에서 value 투영만 꺼내 같은
  가중평균을 한다 (실측 cos **0.2744**, 투영 후 0.5804). 다르게 보면 **12층 attention 의
  softmax 분포를 마스크 점유율 분포로 갈아끼우고 residual/MLP 를 뺀 것**과 같다.
  **가중평균 식도 `ln_post`/`proj` 도 안 바꿨다.**

> **`mask_weighted_patch` 의 실패는 정보 손실이 아니라 축 어긋남이다.** 선형 프로브를
> 씌우면 **89.12%** 로 `cls`(89.77%)와 동급이다. 정보는 온전한데 정답 프롬프트가 나머지
> 19개보다 겨우 **+0.0009** 높고(`cls` 는 +0.0593), 프로브가 찾은 클래스 방향은 텍스트
> 프롬프트 방향과 **직교**한다(cos −0.0098, `cls` 는 +0.2126). AUC 0.4633 이 결정적인데,
> 열 안의 순위조차 무작위라는 뜻이라 점수 보정(centering/whitening/z-score, 전부 ≤18.22%)
> 으로는 못 살리고 좌표계를 되돌리는 행렬로만 살아난다. 실제로 `Wᵀ` 하나를 걸면
> image→text AUC 가 0.4057 → **0.9843** 으로 돌아온다. torch 와 TensorRT 가 일치하므로
> 엔진 문제도 아니다.
>
> *이 문단의 프로브 수치(89.12%, +0.0009, cos −0.0098, ≤18.22%)는 `masked_bbox` 시절에
> 잰 값이고 재측정하지 않았다. 리포에 프로브를 돌리는 도구가 없다. AUC 두 개만 `bbox`
> 기준으로 갱신했다.*

### 측정 (`~/pencil.png` 를 카메라 프레임으로 발행, 전체 파이프라인)

정답 "a pencil", 후보 14개. 마스크는 FastSAM 이 실제로 만든 것.

| pooling | 순위 | 점수 | 2등과의 격차 | 구별력 ↓ |
|---|---|---|---|---|
| `cls` | 1위 | 0.3106 | +0.0226 | 0.8321 |
| `mask_weighted_patch` | 12위 | 0.1709 | −0.0640 | 0.6959 |
| **`mask_weighted_value`** | **1위** | **0.3639** | **+0.0474** | 0.6039 |

*구별력 ↓ = 연필과 나머지 세그먼트의 평균 코사인, 낮을수록 잘 구별.*

실제 사무실 장면(세그먼트 15개)에서도 방향이 같다 — 1등 점수 평균
`cls` 0.2585 / `patch` 0.2627 / **`value` 0.2780**, 세그먼트 구별력
0.7929 / 0.8174 / **0.6775**.

> **`mask_weighted_patch` 의 실패는 구현 버그가 아니다.** 근거: ① 모든 가중치를 1로 둬도
> (마스킹 효과 0) 실패한다 ② `gamma` 0.5~8 × `min_patch_occupancy` 0~0.3 의 20조합에서
> 정답이 12위 아래를 못 벗어나고 1등이 20번 모두 "a wall" 이었다 ③ patch token 을 1개만
> 투영해도 꼴찌다(개수 문제가 아님) ④ torch 와 TensorRT 가 소수점 4자리까지 일치한다.
> 결정적으로, **투영을 한 글자도 안 바꾸고 입력만 value 로 교체하니 12위 → 1위가 됐다.**

**세 모드의 임베딩은 서로 다른 공간에 있다.** downstream 이 섞지 않도록
`EMBEDDING_MODEL_IDS` 가 자동으로 다른 ID 를 붙인다. 저장한 임베딩을 다시 읽을 때 확인할 것.

### 다른 모드로 바꾸기

기본값은 `mask_weighted_value` 이고, 나머지 둘은 인자 하나로 그대로 쓸 수 있다.
노드가 모드에 맞는 엔진 경로를 알아서 고르므로 엔진 인자는 따로 줄 필요가 없다.

```bash
# 기본값 (mask_weighted_value) — 인자 없이
ros2 run meridian_clip clip_inference_node

# CLIP 원본 CLS 경로. 전용 엔진이 필요 없어 엔진을 못 만드는 환경의 대비책이기도 하다
ros2 run meridian_clip clip_inference_node --pooling-mode cls

# 최종 patch token 가중평균. 정렬 행렬 없이는 zero-shot 라벨링이 동작하지 않는다
ros2 run meridian_clip clip_inference_node --pooling-mode mask_weighted_patch \
    --alignment-matrix ~/meridian/src/meridian_clip/models/align_patch_to_cls.npy
```

`tensorrt` 백엔드에서 뒤의 두 모드는 각각 `..._visual_fp16` / `..._visual_pooled_fp16`
엔진이 있어야 하고, 없으면 시작할 때 바로 실패한다 (조용히 되돌아가지 않는다).
`--backend torch` 는 엔진 없이 세 모드 모두 된다.
`tools/clip_selftest.py`, `tools/single_image_test.py` 도 같은 `--pooling-mode` 를 받으며
기본값도 노드와 같다. 셋을 한 번에 비교하려면 `tools/compare_pooling.py`.

### 정렬 행렬 — 기본 경로는 **텍스트 쪽**이다

`ln_post @ proj` 는 학습 내내 CLS 토큰만 입력으로 받았다. 그 함수가 본 적 없는 입력
(가중평균된 토큰)을 넣으면 출력이 텍스트와 어긋난 좌표계에 떨어진다. 어긋남이 **고정된
선형변환**이라 512x512 행렬 하나로 되돌아온다.

되돌리는 자리가 두 곳이고, **어느 쪽에 걸어도 내적은 같다**:

```
(e W)·t  =  e W tᵀ  =  e·(t Wᵀ)
   └ 이미지 쪽                 └ 텍스트 쪽 (기본값)
```

실측으로 두 방식의 top-1 예측이 3,420개 중 **100.0000% 일치**한다. 차이는 무엇이 원본으로
남느냐다 — 텍스트 쪽에 걸면 **이미지 임베딩이 원본 pooling 공간에 그대로 남는다.**

| 구성 | top-1 | macro | AUC | mAP | 분리도 ↑ |
|---|---|---|---|---|---|
| 정렬 없음 | 87.92% | 90.08% | **0.9897** | 0.9076 | **0.1392** |
| 이미지 쪽 (`--alignment-matrix`) | **90.53%** | **91.15%** | 0.9879 | **0.9143** | 0.1081 |
| **텍스트 쪽 (기본값)** | **90.53%** | **91.15%** | 0.9887 | 0.8972 | **0.1392** |

top-1 과 macro 는 **소수점까지 같다** — 예측이 바뀌지 않는다는 위 항등식의 실측 확인이다.
갈리는 곳은 재정규화가 개입하는 순위 지표뿐이다.

**결정적인 칸은 분리도다.** 텍스트 쪽은 이미지 임베딩을 건드리지 않으므로 분리도가
`0.1392` 로 **정확히 보존**되는 반면, 이미지 쪽은 `0.1081` 로 22% 깎인다. 임베딩을
저장해 두는 downstream(VLMap 등)에서 이게 핵심이다 — 맵은 분리도가 가장 좋은 공간에
남고, 변환 대상이 맵 전체가 아니라 프롬프트 몇 개뿐이며, **맵을 다시 만들지 않고 행렬만
갈아끼울 수 있다.** 선형이라 프레임 간 평균과도 교환된다(`mean(E)·W = mean(E·W)`).

> **AUC/mAP 는 이미지 쪽이 이기기도 한다.** 텍스트 쪽 정렬은 mAP 를 0.9076 → 0.8974 로
> 깎는데, 이미지 쪽은 오히려 0.9142 로 **올린다**. 순수 검색 품질만 놓고 고르면 이미지
> 쪽이고, 저장할 임베딩의 품질(분리도)까지 보면 텍스트 쪽이다. 기본값을 텍스트 쪽으로
> 두는 이유는 이 파이프라인의 산출물이 `/instance_embedding_set`, 즉 **저장되는 임베딩**
> 이기 때문이다.
>
> 텍스트 쪽도 공짜는 아니다. 프롬프트 표현이 바뀔 때의 강건성을 깎는다 —
> [언어 쪽 성능](#언어-쪽-성능) 에 숫자가 있다.
>
> 재현: `python tools/benchmark_language.py --image-alignment`

> **변환한 텍스트를 다시 정규화하면 안 된다.** `Wᵀ` 를 지나면 프롬프트마다 벡터 길이가
> 달라지는데 그 길이 차이가 정답 신호의 일부다. 정규화하면 top-1 이 84.56% → **40.56%** 로
> 무너진다 (실측). 노드는 정규화하지 않는다.
>
> **`inverse` 가 아니라 `transpose` 다.** 내적을 넘길 때 필요한 것은 수반(adjoint)이다.
> `W` 는 최소제곱으로 구한 일반 행렬이라 직교가 아니고(‖WᵀW − I‖ = 242.8, 조건수 457,695),
> `inv(W)` 를 쓰면 top-1 이 **5.73%** — 무작위(5%)로 무너진다.

#### 쓰는 법

노드가 pooling 모드에 맞는 행렬을 **엔진 경로와 같은 방식으로 자동 선택**한다.

| pooling | 자동 선택되는 행렬 |
|---|---|
| `mask_weighted_value` (기본값) | `models/align_value_to_cls.npy` |
| `mask_weighted_patch` | `models/align_patch_to_cls.npy` |
| `cls` | 없음 (이미 cls 좌표계) |

```bash
# 기본값 그대로 -- 텍스트 쪽 정렬이 켜져 있다
ros2 run meridian_clip clip_inference_node

# 끄기 (임베딩과 라벨을 모두 순수 pooling 공간에서 보고 싶을 때)
ros2 run meridian_clip clip_inference_node --text-alignment-matrix none

# 만들기 -- 정답 라벨이 필요 없다. 같은 crop 의 (source, cls) 임베딩 쌍만 쓴다
# crop-policy 는 노드 기본값(bbox)이 아니라 masked_bbox 다. 이유는 바로 아래.
~/miniconda3/envs/clip/bin/python tools/fit_alignment.py \
    --images datasets/VOCdevkit/VOC2012/JPEGImages \
    --labels-dir datasets/VOCdevkit/VOC2012/SegmentationObject \
    --id-list datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt \
    --source-mode mask_weighted_value \
    --crop-policy masked_bbox \
    --out src/meridian_clip/models/align_value_to_cls.npy
```

#### 행렬은 `masked_bbox` crop 으로 학습한다 (런타임과 다르게)

노드는 `bbox` 로 도는데 행렬은 `masked_bbox` 로 맞춘다. 직관에 어긋나므로 실측을 남긴다.
아래는 **런타임을 `bbox` 로 고정**한 채 행렬의 학습 조건만 바꿔가며 잰 값이다
(VOC train 3,494 crop 으로 학습 → VOC val 3,420 인스턴스로 평가, `mask_weighted_value`).

| 행렬 학습 조건 (source → target) | top-1 | macro | 평균 순위 | 학습 홀드아웃 cos |
|---|---|---|---|---|
| 정렬 없음 | 87.98% | 90.20% | 1.35 | — |
| **`masked_bbox` → `masked_bbox`** (기본) | **90.61%** | **91.25%** | **1.22** | — |
| `bbox` → `masked_bbox` | 89.88% | 90.16% | 1.33 | **0.9064** |
| `bbox` → `bbox` (런타임과 일치) | 88.51% | 90.06% | 1.25 | 0.8995 |

*이 네 줄은 `opt=8` 로 빌드된 이전 엔진에서 잰 값이라 §6 의 현재 표(정렬 없음 87.92%)와
0.1%p 안쪽에서 어긋난다. 네 조건을 같은 엔진으로 비교한 것이므로 상대 순위는 유효하다.*

**행렬 학습은 런타임을 흉내 내는 절차가 아니다.** `value` 와 `cls` 좌표계의 차이만 순수하게
뽑아내는 별도 캘리브레이션이고, 배경이 섞인 crop 은 두 임베딩에 **공통 잡음**을 넣어
최소제곱이 그 잡음까지 맞추게 만든다. target 쪽(`cls`)이 특히 그렇다 — `cls` 는 마스크를
못 쓰므로 `bbox` crop 에서는 목표값 자체가 배경에 오염된다. 그래서 target 만 되돌려도
88.51% → 89.88% 로 회복되고, source 까지 되돌리면 90.61% 가 된다.

> **`fit_alignment.py` 가 찍는 재구성 cos 를 모델 선택에 쓰면 안 된다.** 위 표에서
> 홀드아웃 cos 가 가장 높은 것은 혼합 조건(0.9064)인데 top-1 은 기본 조건이 이긴다.
> 그 cos 는 최소제곱의 목적함수일 뿐 zero-shot 정확도가 아니다. 학습 조건을 바꿨으면
> `benchmark_pooling.py` 로 실제로 재야 한다.

행렬은 도메인에 딸린 물건이다. 두 파일 모두 VOC(일상 사물 사진)로 만들어 실내 로봇 장면
일반화는 미확인이며, 대상 도메인 이미지로 다시 뽑으면 된다 — 그때도 `--crop-policy
masked_bbox` 를 쓴다.

행렬이 없으면 **경고만 하고 정렬 없이 진행한다** (자동 선택일 때). 그러지 않으면 행렬을
만드는 도구가 행렬이 없어서 못 도는 순환이 생긴다. 반대로 `--text-alignment-matrix` 로
경로를 직접 지정했는데 없으면 즉시 실패한다.

`--alignment-matrix`(이미지 쪽)와 동시에 켜면 `W` 가 두 번 적용되므로 생성자에서 막는다.

#### 임베딩 ID 가 바뀌지 않는 이유

텍스트 쪽 정렬은 **이미지 임베딩을 건드리지 않는다.** 그래서 `embedding_model_id` 는
`..._mask_weighted_value_v1` 그대로다. 저장된 임베딩은 여전히 순수 value 공간이고 달라지는
것은 `/clip_semantics` 의 라벨뿐이다. 이미지 쪽(`--alignment-matrix`)을 쓰면 공간이
바뀌므로 `_aligned` 접미사가 붙는다.

#### 학습 데이터와 한계

두 행렬 모두 VOC **train** 으로 학습하고 **val** 로 평가했다 (완전히 분리된 split).
홀드아웃 재구성 코사인: `align_value_to_cls` **0.9419** (변환 전 0.5791),
`align_patch_to_cls` **0.9235** (변환 전 0.3977).

사진 두 장(통짜 마스크, 프롬프트 `["a photo of a pencil", "a photo of a dog"]`)에서도
정답을 맞히지만 마진은 좁아진다 — pencil +0.1663 → +0.1118, dog +0.1857 → +0.1072.
VOC 의 AUC 하락과 같은 현상이다.

> **행렬은 도메인에 딸린 물건이다.** 둘 다 VOC(일상 사물 사진)로 만들었고 **실내 로봇 장면
> 일반화는 확인되지 않았다.** 대상 도메인 이미지로 `fit_alignment.py` 를 다시 돌리면 되고,
> 라벨이 없어도 되므로 이미지만 모으면 된다. 텍스트 쪽에 걸어 두면 맵을 다시 만들지 않고
> 행렬만 교체할 수 있다는 점이 여기서 실제로 값을 한다.

### 언어 쪽 성능

위의 모든 숫자는 프롬프트가 `"a photo of a {}"` 하나로 고정된 값이다. **표현을 바꾸면
어떻게 되는가**와 **image→text 방향은 어떤가**를 `tools/benchmark_language.py` 로 따로
쟀다. 같은 VOC2012 val 3,420개, 같은 crop/mask 다.

| 구성 | top-1 | i2t AUC | t2i AUC | mAP | P@10 | R@100 | 분리도 |
|---|---|---|---|---|---|---|---|
| `cls` | 83.07% | 0.9817 | 0.9776 | 0.8316 | 94.5% | 64.2% | 0.0662 |
| `mask_weighted_patch` | 7.87% | **0.4055** | 0.4633 | 0.1076 | 30.0% | 11.2% | 0.1158 |
| `patch` + Wᵀ | 88.30% | 0.9842 | 0.9834 | 0.8616 | 99.0% | 65.8% | 0.1158 |
| **`mask_weighted_value`** | 87.92% | 0.9816 | **0.9897** | **0.9076** | 99.0% | **68.9%** | **0.1392** |
| `value` + Wᵀ (기본값) | **90.53%** | **0.9883** | 0.9887 | 0.8972 | 99.0% | 67.6% | **0.1392** |

*i2t = 인스턴스 고정, 프롬프트 20개 줄 세우기. t2i = 프롬프트 고정, 인스턴스 3,420개
줄 세우기(위 표들의 AUC 와 같은 값). mAP/P@10/R@100 은 t2i 방향, 클래스 macro 평균.*

**`patch` 의 i2t AUC 0.4057 은 무작위(0.5)보다 나쁘다** — 축 어긋남 설명과 맞고, t2i AUC
0.4633 보다 더 직접적인 증거다. `Wᵀ` 하나로 0.9843 까지 돌아오는 것이 "정보는 남아
있고 좌표계만 틀어졌다"는 주장의 확인이다.

> **`masked_bbox` 시절에는 두 방향의 순위가 갈렸다.** 그때 `value` 는 t2i 최고(0.9755)면서
> i2t 에서는 `cls` 에 졌고(0.9582 vs 0.9670), 그게 "방향에 따라 순위가 뒤집힌다"는 이
> 절의 원래 요지였다. `bbox` 에서는 i2t 가 0.9816 vs 0.9817 로 사실상 동률이 되어
> **뒤집힘이 사라졌다.** 방향별로 다른 모드를 고를 이유가 없어졌다는 뜻이다.

#### 프롬프트를 바꿨을 때 (text→image mAP, 변형 간 mean ± std)

| 구성 | template (10개) | paraphrase (4개) | description (2개) |
|---|---|---|---|
| `cls` | 0.7434 ± 0.0662 (min 0.6373) | 0.7952 ± 0.0221 | 0.7001 ± 0.0168 |
| **`value`** | **0.9003 ± 0.0169** (min 0.8519) | **0.8922 ± 0.0113** | **0.8390 ± 0.0121** |
| `value` + Wᵀ | 0.8386 ± 0.0549 (min 0.7161) | 0.8747 ± 0.0135 | 0.8186 ± 0.0133 |

*template = 클래스명 고정, 템플릿만 교체(`"itap of a {}"` 등). paraphrase = 템플릿 고정,
클래스명을 동의어로 교체(`sofa`→`couch`/`settee`/`loveseat`). description = 템플릿 없이
자연문(`"a sofa in a living room"`).*

**`value` 공간이 언어 변화에 가장 덜 흔들린다.** 세 가족 모두에서 평균·최솟값 둘 다
최고이고, 표준편차는 template 에서 `cls` 의 1/4 다. `cls` 의 mAP 는 최악의 템플릿에서
0.8316 → **0.6373** 으로 무너지는데 `value` 는 0.9076 → **0.8519** 로 버틴다.

> **Wᵀ 는 그 강건성을 되돌려 놓는다.** `value` 를 cls 좌표계로 끌어오는 변환이라
> `cls` 의 프롬프트 취약성까지 같이 따라온다 (template std 0.0169 → 0.0549, 최솟값
> 0.8519 → 0.7161). 앞의 [정렬 행렬](#정렬-행렬--기본-경로는-텍스트-쪽이다) 절이
> 말한 맞바꿈이 언어 방향에서도 그대로 나타난다 — **검색 품질(mAP)은 세 가족 모두
> 순수 `value` 가 최고**이고, **top-1 은 Wᵀ 가 최고**다 (template 87.97% vs 86.09%,
> paraphrase 88.27% vs 86.31%). 맵에 질의를 던지는 경로면 끄고, `/clip_semantics`
> 라벨이 목적이면 켠다.
>
> 단 **자연문에서는 top-1 조차 Wᵀ 가 앞서지 못한다** (description 77.02% vs 순수
> `value` 77.44%). Wᵀ 는 VOC 이미지 쌍으로 맞춘 행렬이라 `"a photo of a {}"` 류의
> 짧은 프롬프트에서 이득이 가장 크고, 입력이 문장으로 길어지면 그 이득이 사라진다.

프롬프트 앙상블(정규화된 텍스트 임베딩 평균)은 모든 구성에 이득이고, `value` +
template 앙상블의 mAP **0.9112** 가 측정한 전체에서 가장 높다.

공통 약점은 여전히 자연문이다. `description` 가족에서 top-1 이 `value` 77.44%,
`cls` 70.19% 로 template 평균 대비 각각 −8.7%p, −8.8%p 떨어지고 변형 간 편차도
8~11%p 로 세 가족 중 가장 크다. 문장에 섞인 맥락 단어(`living room`, `on the street`)가
crop 하나짜리 인스턴스와 맞지 않기 때문으로 보인다. **로봇에 말로 시키는 형태가 이
파이프라인에서 가장 불리한 입력이다.**

> `bbox` 로 바꾸면서 세 가족 전부에서 `value` 의 mAP 가 크게 올랐다 (template
> 0.8358 → 0.9003, paraphrase 0.8254 → 0.8922, description 0.7689 → 0.8390).
> 배경 픽셀을 지우지 않는 편이 언어 질의에 유리하다는 뜻인데, 맥락 단어가 섞인
> 자연문에서 이득이 가장 큰 것과도 앞뒤가 맞는다.

---

## 4. crop 을 224 정사각형으로 만드는 방법 (`--crop-fit`)

CLIP 원본 전처리는 `Resize(짧은 변=224) + CenterCrop(224)` 다. **사진 한 장을 통째로
분류하는 전제**(주제가 가운데 있다)라 세그먼트 crop 에는 맞지 않는다.

실측: 연필 crop 1555×814 → `Resize(224)` 로 428×224 → CenterCrop 이 좌우 102px 씩 버려
**연필 길이의 48% 가 사라진다.** CLIP 에 실제로 들어간 것은 심도 지우개도 없는 노란 막대였다.
dogs.jpg 21개 세그먼트 중 12개(57%)가 종횡비 r>2 라 절반 이상을 잃었고, 평균 보존율 40.5%.

**셋 다 r 에 비례해 무언가를 잃는다. 잃는 대상이 다를 뿐이다.**

| | 잃는 것 | 남는 정도 |
|---|---|---|
| `centercrop` | 물체의 **범위** | 긴 축의 1/r 만 |
| **`pad`** (기본) | 물체의 **해상도** | 1/r 크기로 렌더링 |
| `stretch` | 물체의 **형태** | 종횡비 왜곡 (CLIP 학습 분포 밖) |

정답이 분명한 세그먼트로 잰 2등과의 격차 (`mask_weighted_value`):

| | r | `centercrop` | `pad` |
|---|---|---|---|
| 개 (dogs.jpg) | 1.24 | +0.0766 | +0.0764 |
| 연필 | 1.91 | +0.0147 | **+0.0596** |

정사각형에 가까우면 차이가 없고 길쭉하면 4배 벌어지므로 `pad` 가 기본값이다.

구현은 `build_geometry()` 한 곳이라 RGB·마스크·디버그 저장의 기하가 자동으로 같이 움직인다
— 마스크만 패딩되면 patch occupancy 가 조용히 거짓이 되므로 이게 중요하다
(검증: 마스크와 RGB 물체 영역 일치율 99.4%, 나머지는 BICUBIC vs NEAREST 경계 1픽셀).
**전처리는 엔진 밖이라 `crop_fit` 을 바꿔도 엔진 재빌드가 필요 없다.**

> `TorchBackend` 는 `clip.load()` 가 준 preprocess 를 **버리고** `build_preprocess` 로
> 덮는다. 그러지 않으면 마스크만 패딩되어 두 이미지가 어긋난다.
> `crop_fit="centercrop"` 이면 CLIP 의 `_transform` 과 완전히 같다.

### crop 안의 마스크 밖 픽셀 (`--crop-policy`)

`crop_fit` 이 crop **을** 224 로 만드는 방법이라면, `crop_policy` 는 crop **안**에서
마스크 밖 픽셀을 어떻게 할지다. 둘은 독립이다.

| | 하는 일 |
|---|---|
| **`bbox`** (기본) | bbox 로 자르기만 한다. 마스크 밖은 원본 픽셀 그대로 |
| `masked_bbox` | 자른 뒤 마스크 밖을 `--mask-fill`(기본 0=검정)로 덮는다 |
| `masked_full` | 자르지 않고 원본 크기 유지, 마스크 밖만 덮는다 |

기본값이 `bbox` 인 이유는 **기본 pooling 이 이미 마스크를 쓰기 때문**이다.
`mask_weighted_value` 는 패치 점유율로 마스크 밖을 가중치에서 배제하므로 픽셀까지
검게 칠하는 것은 중복이고, 검정은 CLIP 이 학습 중 본 적 없는 분포라 patch feature 를
흔들며 물체 경계와 주변 맥락(책상 위의 컵 같은)까지 지운다.

`--pooling-mode cls` 는 마스크를 전혀 못 쓰므로 배경 억제 수단이 crop 정책뿐이다.
**cls 로 돌릴 때는 `--crop-policy masked_bbox` 를 같이 준다.**

> **§3 의 VOC2012 표는 `bbox` 로 다시 잰 값이다.** 벤치마크 도구들도 노드와 함께
> `--crop-policy` 기본값이 `bbox` 로 바뀌었으므로 인자 없이 재현된다.
> 반면 아래 `crop_fit` 절의 연필/dogs 측정은 아직 `masked_bbox` 로 잰 값이라,
> 그 숫자를 재현하려면 `--crop-policy masked_bbox` 를 명시해야 한다.
>
> `benchmark_language.py` 와 `benchmark_imagenet.py` 는 임베딩을 `.npz` 로
> 캐시한다. 두 스크립트 모두 crop 설정을 캐시 식별에 포함하므로 (`benchmark_
> language.py` 는 불일치 시 중단, `benchmark_imagenet.py` 는 키가 달라 다시
> 인코딩) 예전 캐시가 새 기본값 실행에 조용히 섞이지는 않는다.
>
> **정렬 행렬(`models/align_*.npy`)은 예외로 `masked_bbox` crop 으로 학습한다.**
> 런타임 crop 정책과 맞추는 것이 맞아 보이지만, 실측하면 반대다 — 자세한 근거는
> [정렬 행렬](#정렬-행렬--기본-경로는-텍스트-쪽이다) 절 끝의 학습 조건 표를 참고.

---

## 5. 노드

| 노드 | 하는 일 |
|---|---|
| **`clip_inference_node`** | 본체 |
| `embedding_monitor` | `/instance_embedding_set` 구독 → 세그먼트 수 / `embedding_model_id` / L2 norm 출력. **어떤 pooling 인지 확인하는 가장 빠른 방법** |
| `clip_label_viz` | `/clip_semantics` 를 색과 글자로 그려 `/clip_label_overlay` 재발행 |

뒤의 둘은 CLIP/torch 를 로드하지 않고 토픽만 읽는다 — shebang 재패치 대상이 아니다.

### 실행

```bash
# 기본값 = mask_weighted_value + crop_fit pad
ros2 run meridian_clip clip_inference_node \
    --color-topic /camera/camera/color/image_raw \
    --segment-topic /segment_image_resized

ros2 run meridian_clip embedding_monitor
```

설정은 ROS 파라미터가 아니라 **모듈 상수 + argparse** 다 (`remove_ros_args` 후 파싱).
`config/clip_params.yaml` 은 죽은 파일이다.

### 주요 인자

| 인자 | 기본값 | 뜻 |
|---|---|---|
| `--color-topic` / `--segment-topic` | `/camera/rgb` / `/segment_image` | 입력 |
| `--backend` | `tensorrt` | `torch` = `.pt`, `tensorrt` = `.engine` |
| `--pooling-mode` | `mask_weighted_value` | §3 |
| `--text-alignment-matrix` | `""` (모드별 자동) | 텍스트 쪽 정렬. `none` 이면 끔, §3 |
| `--alignment-matrix` | `""` | 이미지 쪽 정렬. 보통 불필요, §3 |
| `--crop-fit` | `pad` | §4 |
| `--crop-policy` | `bbox` | `masked_bbox` / `masked_full` 도 있음, §4 |
| `--patch-weight-gamma` | 1.0 | `w = r^gamma`. 1.0 이면 점유율 그대로 |
| `--min-patch-occupancy` | 0.0 | 이보다 낮은 패치는 가중치 0 |
| `--empty-mask-fallback` | `cls` | 가중치 합이 0 일 때. `skip` / `error` 도 있음 |
| `--min-segment-pixels` | 16 | 이보다 작은 세그먼트는 건너뜀 |
| `--preprocess-workers` | 8 | 224 기하를 만드는 스레드 수 |
| `--async-preprocess` | `false` | 전처리를 청크 단위로 엔진과 겹칠지 |
| `--model-dir` | `""` | 모델 파일 경로들의 디렉터리를 한 번에 교체 |
| `--prompts` / `--prompt-file` | 18개 기본값 | zero-shot 후보 |
| `--debug-save-dir` | `""` | crop/mask/occupancy PNG 저장 |

`gamma` 와 `min_patch_occupancy` 는 **엔진 밖에서** 적용되므로 바꿔도 재빌드가 필요 없다.

> **`--async-preprocess` 는 기본이 꺼짐이다.** 켜면 PIL 기하를 future 로 던져 놓고
> 청크마다 꺼내 써서 CPU resize 와 엔진을 겹친다. 워커가 적을 때만 이득이다
> (Jetson Orin 12코어, N=32): workers=1 은 27.1→32.1 FPS, workers=2 는 38.1→41.7 FPS
> 로 오르지만 workers=4 는 43.8→43.0, workers=8 은 44.8→43.1 로 **내려간다**.
> 워커가 CPU 를 이미 채우고 나면 엔진 대기 중에 돌릴 여유가 없고, §6 의 2-stage
> pipeline 에서는 Stage1/Stage2 균형까지 깨진다 (cls 기준 Pre 17.6→5.3ms,
> Enc 17.9→29.2ms, queue 1.9→26.0ms, 처리량 50.1→45.2 FPS). GPU 가 훨씬 빠른
> 장비에서만 켜 볼 만하다. 켜고 끄고에 따라 **결과 임베딩은 바뀌지 않는다** —
> 기하/정규화 수식은 같고 실행 스케줄만 다르다.

> **`--model-dir`** 은 `--engine-path` / `--pooled-engine-path` / `--value-engine-path` /
> `--text-engine-path` / `--model-path` 와 정렬 행렬까지, 노드가 읽는 모델 파일의
> **디렉터리만** 한 번에 갈아끼운다 (파일 이름은 그대로). `models/` 를 다른 곳에
> 복사해 두고 돌릴 때 인자 하나로 끝난다.
>
> ```bash
> ros2 run meridian_clip clip_inference_node --model-dir ~/clip_bench_code/models
> ```

> **빈 마스크 처리** — crop 을 224 로 늘렸을 때 49개 패치 점유율이 전부 0 이 되는
> 아주 작은 세그먼트가 생긴다. 기본값 `cls` 는 같은 forward 의 CLS 임베딩으로 대체하는데,
> 그런 세그먼트끼리는 **완전히 같은 벡터**가 나온다(거의 검은 crop). 임베딩을 매칭에
> 쓴다면 `--empty-mask-fallback skip` 이 안전하다.

launch 는 `arguments=` 로 넘기므로 빈 문자열 인자나 `BooleanOptionalAction` 플래그를
전달할 수 없다 (`--debug-save-dir`, `--reliable-input` 은 `ros2 run` 으로).

---

## 6. 성능

### 측정 기준

성능을 말할 때는 아래 설정을 기준으로 한다. **이 세 가지가 맞아야 비교가 성립한다.**

| | 값 | 왜 |
|---|---|---|
| 엔진 프로파일 | **min=1 / opt=32 / max=64** | 운용 범위(세그먼트 1~64)를 다 덮고, TensorRT 가 32에 맞춰 커널을 고른다 |
| 노드 `--batch-size` | **32** | **엔진의 `opt` 와 같아야 한다.** 다르면 튜닝되지 않은 배치로 돌게 된다 |
| 보고 기준점 | **N=32** | 배치 32 = 엔진 호출 1회 = `opt` 지점 |

```bash
python3 meridian_clip/build_engine.py --part visual_pooled_value \
    --min-batch 1 --opt-batch 32 --max-batch 64
```

`--part visual` / `visual_pooled` 도 같은 인자로 빌드한다. 하드웨어는 RTX 2080 Ti /
12코어, 입력 640x480, TensorRT fp16, `--crop-fit pad` / `--crop-policy bbox`.

```bash
# 현재 코드
python3 tools/benchmark_stages.py --segments 32

# 최적화 전 코드와 나란히 (src/meridian_clip_backup 필요)
python3 tools/benchmark_stages.py --variant both --segments 32
```

데이터셋이 필요 없다 -- 640x480 프레임에 타원 블롭 N개를 깔아 라벨맵을 합성하므로
카메라도 FastSAM 도 없이 돌고, `RandomState(seed)` 라 어느 머신에서든 입력이 같다.

#### 속도와 정확도는 성질이 다르다

| | 실행 간 변동 |
|---|---|
| **정확도** | **없다.** 3회 반복이 소수점까지 일치한다 |
| **속도** | 도구 실행 단위로 ±3%, 구간별 이상치는 최대 2배 |

**값이 달라졌다면 정확도는 코드가 달라진 것이고, 속도는 그냥 노이즈일 수 있다.**

속도 변동의 출처는 두 가지다.

- **GPU 클럭.** 프레임의 44%만 GPU 가 일해서 드라이버가 계속 클럭을 올렸다 내렸다
  한다. 측정 시작 시점이 735MHz 일 때와 1485MHz 일 때가 섞인다. 엔진 시간이 7.4~8.0ms
  로 ±8% 흔들리는 이유다. 머신 간 비교나 회귀 측정을 할 거면
  `sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 1900` 으로 **양쪽 다 고정**해야 한다
- **웜업.** 도구를 새로 띄우면 엔진 로드와 CUDA 컨텍스트 초기화가 첫 측정에 섞인다.
  프레임 웜업 10장으로는 부족하다

그래서 **도구를 3회 이상 돌린 중앙값**으로 보고한다. 단발 측정값을 인용하면 4% 차이가
회귀처럼 보인다 (실제로 이 문서를 쓰는 동안 한 번 그랬다).

> 구간별로 재면 이상치가 커 보인다 -- 50프레임 기준 `build_regions` 는 중앙값 1.92ms
> 인데 최대 3.31ms, 점유율은 0.76ms 인데 최대 1.23ms 다. **마스크 유무와 무관하게
> 모든 구간이 똑같이 튄다** (GC·스레드 선점 등 시스템 지터). `benchmark_stages.py` 는
> 40프레임 중앙값을 내므로 이건 이미 걸러진다.

> **프로파일의 min/max 는 세그먼트 개수를 고정하는 설정이 아니다.** 엔진이 받을 수
> 있는 배치 크기의 범위일 뿐이다. 프레임 처리 시간이 세그먼트 수 N 에 비례하는 것은
> 설정 문제가 아니라 구조다 — 세그먼트마다 crop 을 만들고, 224 로 리사이즈하고,
> ViT 에 통과시킨다. N=64면 그 일이 두 배다.

### opt 기준 성능 (N=32)

| 단계 | `cls` | `mask_weighted_value` **(기본값)** | `mask_weighted_patch` |
|---|---|---|---|
| **Preprocessing** | **9.358** (55.6%) | **9.983** (54.9%) | **9.579** (53.5%) |
| **CLIP Encoder** | **7.243** (43.0%) | **7.984** (43.9%) | **8.092** (45.2%) |
| **Postprocessing** | **0.230** (1.4%) | **0.232** (1.3%) | **0.233** (1.3%) |
| **합계** | **16.832 ms** | **18.200 ms** | **17.904 ms** |
| **FPS** | **59.4** | **54.9** | **55.9** |
| 세그먼트 1개당 | 0.526 ms | 0.569 ms | 0.559 ms |

재현: `python3 tools/benchmark_stages.py --segments 32`

**세 모드가 이제 거의 같다.** 마스크 모드의 전처리 우위가 사라진 것은 점유율 직행
경로(아래 (3)) 덕분이다 — 마스크를 224 로 늘리지 않으니 `cls` 와 할 일이 거의 같아졌다.
**엔진은 7.24~8.09ms 로 세 모드가 동일**하고(엔진 파일만 다르고 코드 경로가 같다),
모드 선택은 **속도가 아니라 정확도 문제다** (§3).

각 단계에 무엇이 들어가는지:

| 단계 | 포함 |
|---|---|
| Preprocessing | 세그먼트 분해 + bbox crop → 224 리사이즈(RGB / 마스크) → uint8 H2D → GPU 정규화 → 패치 점유율·가중치 |
| CLIP Encoder | TensorRT 엔진 실행. 가중평균 pooling 은 엔진 그래프 안에서 끝난다 |
| Postprocessing | D2H → 빈마스크 fallback → L2 정규화 → 정렬 행렬 → zero-shot 유사도/top-k → 메시지 빌드 + publish |

### 세그먼트 개수(N)에 따른 변화 (참고)

FastSAM 은 프레임당 최대 255개를 낸다. N=32 가 기준점이지만 실제 N 은 장면마다 다르다.

**`mask_weighted_value`** (ms/frame, 중앙값)

| | N=1 | N=4 | N=8 | N=16 | **N=32** | N=64 |
|---|---|---|---|---|---|---|
| Preprocessing | 4.077 | 4.873 | 5.288 | 7.790 | **12.384** | 21.014 |
| CLIP Encoder | 1.863 | 2.126 | 2.790 | 4.199 | **8.351** | 16.538 |
| Postprocessing | 0.256 | 0.451 | 0.640 | 1.053 | **1.803** | 5.426 |
| **합계** | 6.197 | 7.449 | 8.718 | 13.042 | **22.537** | 42.978 |
| **FPS** | 161 | 134 | 115 | 77 | **44** | 23 |
| 세그먼트 1개당 | 6.197 | 1.862 | 1.090 | 0.815 | **0.704** | 0.672 |

**`mask_weighted_patch`**

| | N=1 | N=4 | N=8 | N=16 | **N=32** | N=64 |
|---|---|---|---|---|---|---|
| Preprocessing | 4.086 | 4.442 | 5.208 | 7.551 | **12.041** | 22.144 |
| CLIP Encoder | 1.935 | 2.178 | 2.729 | 3.849 | **8.335** | 16.021 |
| Postprocessing | 0.259 | 0.458 | 0.646 | 1.028 | **1.790** | 3.885 |
| **합계** | 6.281 | 7.078 | 8.582 | 12.427 | **22.165** | 42.049 |
| **FPS** | 159 | 141 | 117 | 81 | **45** | 24 |
| 세그먼트 1개당 | 6.281 | 1.770 | 1.073 | 0.777 | **0.693** | 0.657 |

**`cls`**

| | N=1 | N=4 | N=8 | N=16 | **N=32** | N=64 |
|---|---|---|---|---|---|---|
| Preprocessing | 3.819 | 3.895 | 4.236 | 6.103 | **9.611** | 16.767 |
| CLIP Encoder | 1.937 | 2.171 | 2.789 | 3.671 | **7.915** | 15.316 |
| Postprocessing | 0.190 | 0.430 | 0.575 | 0.982 | **1.697** | 3.256 |
| **합계** | 5.946 | 6.497 | 7.601 | 10.756 | **19.224** | 35.339 |
| **FPS** | 168 | 154 | 132 | 93 | **52** | 28 |
| 세그먼트 1개당 | 5.946 | 1.624 | 0.950 | 0.672 | **0.601** | 0.552 |

N=1 의 6ms 는 프레임 고정비(세그먼트 분해 1.2ms + 배치 1개짜리 엔진의 비효율)가 통째로
한 세그먼트에 얹힌 값이라 단가가 아니다. N≥16 부터 세그먼트 단가가 수렴한다.

### 224 전처리 경로 4종 비교 (uHumans2 office, 세 모드 전부)

`pre` 가 `enc` 보다 무거운 것이 §6 표의 일관된 결론이고, 그 대부분이 crop 을 224 로
만드는 구간이다. 그래서 같은 224 를 만드는 다른 경로 셋을 놓고 쟀다.

| 경로 | 방법 |
|---|---|
| **`current`** | **배포 경로.** crop → PadToSquare → PIL BICUBIC (CPU) |
| `roi_align` | 방법 ③. 정사각 ROI 한 번 + band 마스킹, bilinear 고정 |
| `interp_aa` | 방법 ②-a. 인스턴스별 zero-pad → `interpolate(bicubic, antialias)` |
| `grid_sample` | 방법 ②-b. 좌표 하나로 접어 batched `grid_sample(bicubic)` |

> **뒤의 셋은 `tools/` 안에만 있다.** `clip_backend.py` 는 여전히 `current` 만 쓴다.
> 교체 후보를 오프라인에서 재 본 것이지 노드 동작이 바뀐 것이 아니다.

**측정 조건** — `tools/benchmark_uhumans.py`, `uHumans2_office_s1_00h.bag`,
720x480, 프레임 297장 (`--stride 28 --max-frames 300 --warmup-frames 10`),
인스턴스 5536개(≥900px), 프롬프트 18종, ROS 노드 경로(post 에 DDS 발행 포함),
**RTX 2080 Ti / torch 2.2.0+cu118 / TensorRT 10.13.0.35**.
N 은 고정하지 않는다 — 프레임당 평균 18.6 (중앙값 18, 최소 5, 최대 40),
crop 배율 중앙값 0.54 로 **76% 가 업스케일**이다.
`cos` 는 같은 세그먼트를 `current` 로 낸 임베딩과의 코사인(드리프트)이다.

**`cls`** — Wᵀ 정렬 없음 (cls 용 행렬은 존재하지 않는다)

| 경로 | pre | enc | post | 합계 | FPS | top-1 전체 | top-1 thing | cos 평균 | cos p5 | cos 최소 |
|---|---|---|---|---|---|---|---|---|---|---|
| current | 9.20 | 5.71 | 0.15 | 15.06 | 66.4 | 27.76% | 49.13% | — | — | — |
| roi_align | **2.55** | 5.66 | 0.08 | 8.29 | 120.6 | **29.10%** | **50.13%** | 0.9780 | 0.9418 | 0.8274 |
| interp_aa | 2.64 | 5.69 | 0.07 | 8.39 | 119.1 | 27.57% | 48.26% | **0.9975** | **0.9918** | **0.9588** |
| grid_sample | 2.80 | 5.27 | 0.07 | **8.14** | **122.9** | 28.78% | 48.53% | 0.9858 | 0.9615 | 0.8974 |

**`mask_weighted_patch`** — Wᵀ = `align_patch_to_cls.npy`

| 경로 | pre | enc | post | 합계 | FPS | top-1 전체 | top-1 thing | cos 평균 | cos p5 | cos 최소 |
|---|---|---|---|---|---|---|---|---|---|---|
| current | 10.34 | 5.60 | 0.15 | 16.08 | 62.2 | 16.33% | 52.94% | — | — | — |
| roi_align | **2.74** | 5.53 | 0.09 | **8.36** | **119.6** | **16.91%** | **55.21%** | 0.9889 | 0.9712 | **0.8992** |
| interp_aa | 2.92 | 5.91 | 0.08 | 8.90 | 112.4 | 16.38% | 52.74% | **0.9987** | **0.9959** | 0.8984 |
| grid_sample | 3.05 | 5.55 | 0.08 | 8.68 | 115.3 | 16.62% | 53.68% | 0.9925 | 0.9798 | 0.8907 |

**`mask_weighted_value`** (배포 기본값) — Wᵀ = `align_value_to_cls.npy`

| 경로 | pre | enc | post | 합계 | FPS | top-1 전체 | top-1 thing | cos 평균 | cos p5 | cos 최소 |
|---|---|---|---|---|---|---|---|---|---|---|
| current | 10.05 | 5.56 | 0.15 | 15.76 | 63.4 | 31.25% | 55.08% | — | — | — |
| roi_align | **2.65** | 5.67 | 0.08 | 8.39 | 119.1 | **31.52%** | **56.28%** | 0.9819 | 0.9527 | 0.7645 |
| interp_aa | 2.77 | 5.86 | 0.07 | 8.70 | 115.0 | 30.82% | 54.55% | **0.9975** | **0.9917** | **0.9027** |
| grid_sample | 2.91 | 5.38 | 0.07 | **8.35** | **119.7** | 31.20% | 54.81% | 0.9881 | 0.9661 | 0.8825 |

읽는 법:

- **`pre` 가 3.3~3.8배 줄고 `enc` 는 그대로다.** 세 후보 모두 전처리만 바꾸므로 당연하지만,
  그래서 전체 FPS 가 모드와 무관하게 62~66 → 112~123 으로 **1.8~1.9배**가 된다.
  `current` 의 `pre` 가 `enc` 의 1.6~1.9배였던 것이 후보 경로에서는 0.5배 안팎으로 뒤집힌다.
- **경로 간 top-1 차이는 전체 기준 ±1.4%p, thing 기준 ±2.3%p 안쪽이다.** `roi_align` 이
  세 모드 모두에서 thing 기준 가장 높지만(+1.00 / +2.27 / +1.20%p) 드리프트는 평균·p5
  기준 셋 중 가장 크다(평균 0.9780~0.9889). `interp_aa` 는 정반대로 `current` 를 거의
  그대로 재현한다 (평균 0.9975 이상, 최소 0.8984 이상). **바꿔치기 후보로는 `interp_aa`,
  라벨 정확도만 보면 `roi_align`** 이다.
- **`cls` 의 `pre` 가 다른 두 모드보다 ~1ms 싸다** (9.20 vs 10.05/10.34). 마스크 PIL 을
  만들지 않기 때문이고, 후보 경로에서는 그 차이가 0.1~0.3ms 로 줄어든다.

> ⚠ **uHumans2 의 절대 top-1 은 VOC 와 비교할 수 없다.** semantic GT 만 주므로 인스턴스를
> 연결성분으로 만들고, 라벨 21종 중 여럿이 잡동사니 묶음이다(id 10 = 노트북+머그+키보드+종이…).
> `patch` 의 전체 16.33% 가 `cls` 의 27.76% 보다 낮은 것도 그 영향이 크다 — thing 기준으로는
> 52.94% 대 49.13% 로 뒤집힌다. **유효한 신호는 경로 간 Δ 와 드리프트 코사인이지
> 절대값이 아니다.** 절대 정확도는 바로 아래 VOC2012 표를 본다.

#### 절대 정확도 — VOC2012 val (같은 네 경로, 세 모드)

인스턴스 GT 가 있는 데이터셋이라 절대값이 의미를 갖는다. `tools/benchmark_voc.py`,
val 1449장 중 세그먼트 있는 것, **인스턴스 2845개**(≥900px), 프롬프트
`"a photo of a {class}"` 20종. Wᵀ 정렬은 uHumans2 와 같다
(`cls` 없음 / `patch` `align_patch_to_cls.npy` / `value` `align_value_to_cls.npy`).
**`grid_bicubic` 이 위 표 `interp_aa`(방법 ②-a)의 VOC 쪽 이름이다.**

| 모드 | 경로 | top-1 | Δ | 평균 마진 | cos 평균 | cos p5 | cos 최소 |
|---|---|---|---|---|---|---|---|
| **cls** | current | 84.64% | — | 0.0365 | — | — | — |
| | roi_align | **84.67%** | +0.03 | 0.0370 | 0.9808 | 0.9435 | 0.8132 |
| | grid_bicubic | 84.32% | -0.32 | 0.0364 | **0.9991** | **0.9971** | **0.9802** |
| | grid_sample | 84.36% | -0.28 | 0.0367 | 0.9841 | 0.9486 | 0.8186 |
| **patch** | current | 90.79% | — | 0.0333 | — | — | — |
| | roi_align | **90.86%** | +0.07 | 0.0330 | 0.9940 | 0.9814 | 0.9299 |
| | grid_bicubic | 90.79% | 0.00 | 0.0332 | **0.9996** | **0.9987** | **0.9858** |
| | grid_sample | 90.62% | -0.17 | 0.0331 | 0.9960 | 0.9864 | 0.9237 |
| **value** | current | **92.72%** | — | 0.0390 | — | — | — |
| | roi_align | 92.48% | -0.24 | 0.0390 | 0.9925 | 0.9731 | 0.8777 |
| | grid_bicubic | 92.58% | -0.14 | 0.0390 | **0.9995** | **0.9982** | **0.9740** |
| | grid_sample | 92.69% | -0.03 | 0.0389 | 0.9950 | 0.9817 | 0.9095 |

- **네 경로의 top-1 은 어느 모드에서도 0.35%p 안에 들어온다.** 2845개 기준 0.3%p 는
  10개 남짓이라 경로 선택이 정확도를 유의미하게 바꾸지 않는다. 모드 간 차이
  (84.6 → 90.8 → 92.7%) 가 경로 간 차이보다 **20배 이상 크다.**
- **드리프트 순서는 uHumans2 와 완전히 같다.** `grid_bicubic`(=`interp_aa`) 이 세 모드
  모두 평균 0.999 이상·최소 0.974 이상으로 `current` 를 가장 충실히 재현하고,
  `roi_align` 과 `grid_sample` 은 최소 0.81~0.93 까지 벌어진다. 두 데이터셋이 같은
  결론을 주므로 **경로를 바꾼다면 `interp_aa`** 다.
- 인코딩 시간(5.2~5.8s → 3.5~4.1s)도 같은 방향이지만, 여기 crop 은 VOC 원본 해상도라
  §6 앞 표의 ms 수치와 직접 비교하지 않는다.

재현:

```bash
python tools/benchmark_voc.py --package ~/meridian/src/meridian_clip \
    --voc-root ~/meridian/datasets/VOCdevkit/VOC2012 \
    --pooling-mode mask_weighted_value \
    --paths current roi_align grid_bicubic grid_sample
```

재현:

```bash
python tools/benchmark_uhumans.py --package ~/meridian/src/meridian_clip \
    --bag ~/Downloads/uHumans2_office_s1_00h.bag \
    --pooling-mode mask_weighted_value --stride 28 --max-frames 300
```

### 엔진 프로파일의 `opt` 가 실제로 성능을 바꾼다

처음에는 "배치 8부터 이미지당 비용이 평탄하니 배치를 키워도 소용없다"고 판단했는데,
**그 측정은 `opt=8` 로 빌드된 엔진 안에서 배치만 바꾼 것이었다.** `opt` 를 32로 올려
다시 빌드하니 TensorRT 가 큰 배치에 맞는 커널을 새로 골라 이미지당 비용 자체가 내려갔다.

| 엔진 시간 | opt=8 / batch 16 | **opt=32 / batch 32** | 변화 |
|---|---|---|---|
| `value` N=32 | 9.385 | **8.351** | −11.0% |
| `value` N=64 | 19.748 | **16.538** | −16.3% |
| `patch` N=32 | 10.065 | **8.335** | −17.2% |
| `patch` N=64 | 21.064 | **16.021** | −23.9% |
| `cls` N=32 | 10.710 | **7.915** | −26.1% |
| `cls` N=64 | 19.891 | **15.316** | −23.0% |

**배치 크기와 `opt` 는 별개 축이다.** 배치만 바꿔서 잰 곡선으로 `opt` 의 효과를 판단하면
안 된다. 엔진을 다시 빌드하면 fp16 커널 선택이 달라져 top-1 이 0.1%p 안쪽에서 움직이므로
(3,420개 중 3~4개), 정확도 표를 인용할 때는 어느 빌드인지 함께 봐야 한다.

---

### 최적화 기록

병목은 CLIP 이 아니라 **CLIP 을 먹이는 파이썬 코드**였다. 최적화 전에는 시간의 74~80%
가 CPU 전처리에 있었고 엔진은 20% 였다. 아래 두 가지를 적용했고, 둘 다
**임베딩을 바꾸지 않는다**(최대 오차 4.8e-7 = float32 반올림).

#### (1) 224 전처리 배치화 — `BatchPreprocessor`

전처리를 두 조각으로 나눴다.

```
PadToSquare → Resize(224, BICUBIC) → convert("RGB")   ← PIL, 스레드풀 8개로 병렬
──────────────────────────────────── uint8 ndarray
ToTensor → Normalize                                   ← 배치 한 번에 device 에서
```

- **스레드풀 8개.** PIL 은 `resize` 동안 GIL 을 놓으므로 파이썬 스레드로도 실제 병렬이
  된다. 실측에서 8개가 최적이고 12개는 torch 내부 스레딩과 경합해 더 느리다.
- **H2D 를 uint8 로.** 224×224×3 이 float32 602KB → uint8 150KB 로 4배 줄고, `/255` 와
  정규화는 GPU 에서 배치로 한다. 배치별 `torch.stack` + `.to(cuda)` 가 사라져
  `배치 스택 + H2D` 단계 자체가 없어졌다.
- **`Resize` 는 PIL 에 그대로 뒀다.** 이것이 수치 보존의 조건이다.

`build_preprocess` / `build_mask_preprocess` 는 한 장짜리 API 로 남겨 뒀다 —
`tools/clip_selftest.py` 와 `tools/benchmark_imagenet.py` 가 직접 호출한다.

#### (2) `build_regions` 단일 패스

세그먼트마다 640×480 전체를 4번 훑던 것(`labels == id`, `.sum()`, `.any(axis=1)`,
`.any(axis=0)`)을 **N 과 무관하게 2번**으로 고정했다.

- `np.bincount(labels.ravel(), minlength=256)` — 라벨별 면적. `np.unique` 의 정렬과
  세그먼트별 `mask.sum()` 을 **둘 다** 대체한다.
- `scipy.ndimage.find_objects` — 모든 라벨의 bbox 를 한 번의 C 패스로. 같은 라벨이 여러
  덩어리로 흩어져 있어도 전체를 감싸는 bbox 를 주므로 `labels == id` 와 의미가 같다.
- 마스크를 **bbox 안에서만** 만든다 (`labels[y0:y1, x0:x1] == id`). 세그먼트별 비용이
  O(H×W) → O(crop 면적) 으로 떨어진다. 프레임 전체 마스크가 필요한 것은 `masked_full`
  정책뿐이라 그 분기에서만 만든다.
- `mask_to_image` 의 `astype(np.uint8)` → `view(np.uint8)` (복사 1회 제거).

출력은 `segment_ids`/`boxes`/region 픽셀/mask 픽셀 모두 **비트 단위로 동일**하다.

#### (3) 패치 점유율 직행 — 224 마스크를 아예 만들지 않는다

마스크 경로는 **1024:1 낭비**였다. 224×224 = 50,176 픽셀을 만들어 GPU 에 올린 뒤
7×7 = **49개** 숫자로 줄이고 있었다.

NEAREST resize 는 보간이 아니라 **인덱스 gather** 라서 접을 수 있다. 출력 픽셀
`(i, j)` 는 원본 `(idx[i], idx[j])` 를 그대로 집어 오고 점유율은 32×32 블록의
평균이므로,

```
occupancy[p, q] = (1/1024) · Σ mask[idx[i], idx[j]]  =  (1/1024) · (Cy[p] · mask · Cx[q])
```

로 **계수행렬 두 개의 행렬곱**이 된다. `Cy[p, y]` 는 "패치 `p` 에 속한 출력 행 중
원본 행 `y` 를 집는 개수"다. 합이 정수(≤1024)라 float32 반올림이 없어 기존 경로와
**비트 단위로 같은 값**이 나온다 (`clip_backend.patch_occupancy_from_masks`).

- `(length, offset, side)` 가 같으면 계수행렬도 같으므로 **중복 제거**하고 색인으로 참조
- 남은 조합은 `bincount` 한 번으로 일괄 생성 — 조합마다 numpy 를 부르면 32장에 1.60ms
  인데 일괄로는 0.30ms 다. 작은 배열에 numpy 를 수백 번 부르는 오버헤드가 계산보다 컸다
- `crop_fit="pad"` 이고 마스크가 crop 크기 ndarray 로 들어올 때만 쓴다. 그 밖(다른
  `crop_fit`, PIL 입력, 224 가 격자로 안 나누어떨어짐)은 기존 224 경로로 돈다

**32개 기준 5.40ms → 1.67ms.** 이것 때문에 마스크 모드의 전처리가 `cls` 와 거의
같아졌다.

#### (4) ROS 메시지 고속 경로 — Postprocessing 7배

`InstanceEmbeddingSet.embeddings` setter 에는 `array.array` 고속 경로가 있는데
`.tolist()` 로 **Python 리스트**를 넘겨 느린 경로를 타고 있었다. 16,384개 원소를
**네 번** 훑는다:

```
numpy → .tolist()                     16,384개 Python float 객체 생성
      → all(isinstance(v, float) …)   16,384회 검사
      → all(범위 검사 …)                16,384회 검사
      → array.array('f', value)       다시 C 배열로
```

`array("f", arr.tobytes())` 로 넘기면 필드 타입과 정확히 같아 그 검사를 통째로
건너뛴다. 바이트가 그대로 들어가므로 **값은 동일**하다.

**1.327ms → 0.008ms (164배).** Postprocessing 전체가 1.6ms → 0.23ms 가 됐다.

#### (5) pinned 스테이징 버퍼

`np.stack` 이 만든 pageable 메모리에서 복사하면 드라이버가 내부 임시 pinned 버퍼를
한 번 더 거친다. 미리 잡아 둔 pinned 버퍼에 직접 써 넣으면 그 왕복이 사라진다
(32장 224×224×3: **1.28ms → 0.83ms**).

버퍼를 재사용하므로 **다음 프레임이 덮어쓰기 전에 이전 H2D 가 끝나야 한다.** 복사
직후 기록한 `torch.cuda.Event` 를 다음 호출 맨 앞에서 기다린다. 이게 없으면
`non_blocking` 복사가 진행 중인 메모리를 CPU 가 갈아엎어 **조용히 값이 깨진다.**

#### (6) 지연 futures + 루프 내 D2H 제거 — 성능 중립, 구조 개선

프레임 안에서 전처리와 엔진을 겹치려고 두 가지를 바꿨다.

- `prepare()` 가 224 기하를 `pool.map`(전부 끝날 때까지 블록) 대신 `pool.submit`
  으로 던지고 **future 만 들고 반환**한다. `run()` 이 청크마다 필요한 만큼만 `result()`
  한다
- 엔진 루프에서 **`.cpu()` 를 전부 걷어냈다.** 청크마다 3회씩 하던 D2H(`empty`,
  `occupancy`, `weights`)와 `np.where` fallback 을 device 텐서 누적으로 바꾸고,
  **D2H 는 루프가 끝난 뒤 한 번만** 한다. `infer(..., as_tensor=True)` 추가

**결과는 성능 중립이다** (18.20 → 18.20ms). 이유는 아래 "버린 것"에 적었다. 다만
동기화 지점이 청크당 3개에서 프레임당 1개로 줄었고, 이건 프레임 **간** 파이프라이닝의
전제 조건이라 남겨 두었다. `PreparedBatch` 가 프레임에 딸린 것만 담는 것도 같은 이유다.

#### ⚠ `build_regions` 의 마스크 생략은 호출자가 정한다

(3) 을 넣으면서 "cls 는 마스크를 안 쓰니 만들지 말자"를 `build_regions` 안에서
`self.pooling_mode` 로 판단하게 했더니 **`tools/` 의 벤치마크가 조용히 망가졌다.**

`tools/` 의 도구들은 **첫 모드의 노드로 crop 을 한 번 만들어 나머지 모드에 그대로
넘긴다** — 세 모드가 같은 crop 을 공유해야 pooling 차이만 분리되기 때문이다. 첫
모드가 `cls` 면 마스크가 전부 `None` 이 되고 뒤따르는 마스크 모드가 쓰레기를 받는다.

| `--modes` 순서 | `value` | `patch` |
|---|---|---|
| `value` 먼저 | 87.92% ✓ | 7.87% ✓ |
| **`cls` 먼저** | **76.08%** ✗ | **35.26%** ✗ |

알파벳순으로 주면 항상 `cls` 가 먼저라 두 모드가 말없이 틀어진다.

**고친 방식**: `build_regions(..., with_masks: bool = True)` 로 **기본값을 계약으로**
두고, 노드의 프레임 경로만 `with_masks=self.needs_region_masks` 로 명시해 생략한다.
생략 여부는 인스턴스 상태가 아니라 호출자가 정한다.

#### 효과 — 같은 엔진으로 코드만 바꿔 A/B

`src/meridian_clip_backup` 에 최적화 직전 코드가 남아 있어, **같은 엔진
(min=1/opt=32/max=64) 과 같은 `batch_size=32`** 로 코드만 갈아끼워 쟀다. 즉 아래 차이는
전부 코드에서 온 것이다. N=32, 40회 중앙값.

재현: `python3 tools/benchmark_stages.py --variant both --segments 32`

| | 단계 | 최적화 전 | **현재** | 배속 |
|---|---|---|---|---|
| **`cls`** | Preprocessing | 35.535 | **9.358** | **3.80x** |
| | CLIP Encoder | 7.650 | 7.243 | 1.06x (동일) |
| | Postprocessing | 1.612 | **0.230** | **7.01x** |
| | **합계** | **44.796 ms** (22.3 FPS) | **16.832 ms** (59.4 FPS) | **2.66x** |
| **`value`** | Preprocessing | 40.322 | **9.983** | **4.04x** |
| | CLIP Encoder | 7.825 | 7.984 | 0.98x (동일) |
| | Postprocessing | 1.617 | **0.232** | **6.95x** |
| | **합계** | **49.764 ms** (20.1 FPS) | **18.200 ms** (54.9 FPS) | **2.73x** |
| **`patch`** | Preprocessing | 40.440 | **9.579** | **4.22x** |
| | CLIP Encoder | 7.364 | 8.092 | 0.91x (동일) |
| | Postprocessing | 1.623 | **0.233** | **6.98x** |
| | **합계** | **49.427 ms** (20.2 FPS) | **17.904 ms** (55.9 FPS) | **2.76x** |

엔진은 손대지 않았으므로 그대로다(0.91~1.06x 는 클럭 변동). **전처리 3.8~4.2배와
Postprocessing 7배가 전부**이고, 그 결과 Preprocessing 비중이 79% → 54% 로 내려가면서
엔진이 최대 단일 항목이 됐다.

정확도는 그대로다.

**같은 엔진**으로 코드만 갈아끼워 잰 값이다 (top-1 micro / macro).

| pooling | 최적화 전 | 현재 |
|---|---|---|
| `cls` | 83.07% / 88.02% | 83.07% / **88.08%** |
| `mask_weighted_patch` | 7.89% / 10.55% | 7.87% / 10.51% |
| **`mask_weighted_value`** | **87.98%** / 90.14% | **87.92%** / 90.08% |

차이는 `value` 기준 **3,420개 중 2개**이고 `cls` macro 는 오히려 올랐다 -- 방향이
섞여 있는 것이 반올림 노이즈의 특징이다.

출처는 **전처리의 float32 반올림 하나뿐이다.** `ToTensor`+`Normalize`(CPU) 를
uint8 H2D + GPU 정규화로 바꾸면서 입력에 4.8e-7 오차가 생기고, fp16 엔진을 지나며
임베딩 코사인 0.999984 가 된다. 경계선에 있던 2개가 뒤집힌다. 점유율 직행 경로는
정수합이라 비트 동일이라 기여가 없다.

> **엔진 재빌드는 정확도를 바꾸지 않았다.** 백업 코드로 재면 옛 엔진(max=32)과
> 새 엔진(min=1/opt=32/max=64) 모두 `value` 87.98% 로 같다. 프로파일 변경은 커널
> 선택만 바꾸고 수치는 건드리지 않는다.
>
> 정확도 측정은 **완전히 결정적이다** -- 3회 반복이 소수점까지 일치한다. 속도와
> 달리 실행 간 변동이 없으므로, 값이 달라졌다면 코드가 달라진 것이다.

#### 측정해 보고 **버린** 것

같은 실수를 반복하지 않도록 남긴다. 전부 실측 후 폐기했다.

| 시도 | 결과 | 왜 안 됐나 |
|---|---|---|
| **프레임 내 CPU/GPU 파이프라이닝** | **1.00x** | 아래 참고 -- 겹칠 일감이 애초에 없다 |
| **마스크 → 7×7 BOX 직행** | 2.4% | 전송량은 1024:1 로 주는데 병목이 224 해상도가 아니라 마스크당 PIL 호출 오버헤드였다. NEAREST 양자화가 빠져 점유율도 달라진다(비트 비동일) |
| **마스크 기하를 numpy gather 로** | **6배 느림** | PIL 의 NEAREST 인덱스를 실측으로 재현해 비트 일치까지 확인했으나, PIL 의 C resize 가 numpy fancy indexing 보다 훨씬 빠르다 |
| **CUDA graph** (GPU tail) | **0.95x (역효과)** | replay 가 입력을 static 버퍼로 `copy_` 해야 하는데 그 비용이 커널 4개보다 크다. 대역폭 바운드라 런치 오버헤드가 지배적이지 않다 |
| **`torch.compile`** (GPU tail) | 2.96x → 프레임의 **0.8%** | GPU tail 이 전처리의 4.7% 뿐이라 상한이 낮다 |
| **`torch.jit.script`** (GPU tail) | 1.96x → 프레임의 0.6% | 위와 같음 |
| **`reducing_gap=2/3`** | 0.93x (역효과) | 큰 축소비에서만 이득인데 세그먼트 crop 은 확대되는 경우가 많다 |
| **pinned buffer + non_blocking** | ~1.00x | `np.stack` + H2D 가 이미 전처리의 8.8% 뿐 |
| **스레드 12/16/24개** | ~1.02x | 8개에서 이미 평평하다 |

**프레임 안에서는 겹칠 것이 없다 (실측).** (6) 을 넣고 청크별 대기 시간을 재 보면:

```
prepare(제출 + 점유율)   5.18 ms
청크   future 대기   업로드+정규화    엔진
  0        0.02          0.91      4.21
  1        0.01          0.47      5.63
```

`future 대기 = 0.02ms` -- 엔진이 시작될 때 PIL 은 **이미 끝나 있다.** `prepare()` 가
futures 를 던진 뒤 점유율(numpy, 메인 스레드)을 계산하는데, 그 사이에 워커 8개가
32장을 다 만들어 버린다. 즉 **중첩은 이미 일어나고 있고**, 상대가 엔진이 아니라
점유율 계산일 뿐이다.

청크를 쪼개도 소용없다. 겹칠 일감이 없는데 엔진만 비효율적이 된다.

| `batch_size` | 청크 수 | Preprocessing | Encoder | 합계 |
|---|---|---|---|---|
| 8 | 4 | 9.914 | 9.476 | 19.617 |
| 16 | 2 | 9.721 | 8.599 | 18.541 |
| **32** | **1** | 10.043 | **7.932** | **18.196** |

**남은 것은 프레임 *간* 파이프라이닝이다.** 프레임 k 의 엔진이 도는 동안 프레임 k+1 의
전처리를 시작하는 구조로, `prepare` 스레드와 `run` 스레드를 나누고 `PreparedBatch` 를
큐로 넘기면 된다. 기대치는 프레임당 `max(전처리 8.5, 엔진 7.9)` ≈ 10ms (**1.8배**)이고
처리량만 오른다(단일 프레임 지연은 그대로). 노드 구조 변경이라 아직 하지 않았다.

**vectorization** 은 두 갈래로 결론이 나 있다. GPU tail 은 이미 배치 연산이고, PIL 루프는
**crop 마다 크기가 달라 벡터화 자체가 불가능**하다.

**수치를 바꾸면 더 빨라지는 길은 남아 있다.** 셋 다 정확도 재측정이 필요하다.

| 후보 | 배속 (PIL 구간) | 픽셀 차이 (0~255) |
|---|---|---|
| GPU resize (`F.interpolate`) | 6.5x | 정규화 후 평균 0.146 |
| `cv2.INTER_CUBIC` | 2.25x | 최대 131, 평균 6.1 |
| resize 먼저 → pad 나중 | 1.24x | 최대 227, 평균 10.5 |
| `BILINEAR` | 1.18x | 최대 49, 평균 6.5 |

torch 의 bicubic 은 커널 상수가 PIL 과 달라(a=−0.75 vs −0.5) `antialias=True` 로도 값이
맞지 않는다.

#### 남은 병목 (N=32 `value`)

| | ms | 비중 |
|---|---|---|
| **TensorRT 엔진** | 7.98 | **43.9%** |
| Preprocessing (224 RGB + 분해/crop) | 9.98 | 54.9% |
| Postprocessing | 0.23 | 1.3% |

Preprocessing 안에서는 **224 RGB 리사이즈가 여전히 대부분**이고 그 다음이
`build_regions` 다. 마스크 경로는 (3) 으로 거의 사라졌고 Postprocessing 은 (4) 로
1.3% 까지 내려가 더 볼 것이 없다.

엔진이 사실상 최대 단일 항목이다. `opt` 를 32로 올려 한 번 더 얻었고(위 표),
`builder_optimization_level=5` 는 245초 빌드해서 **0.2%** 뿐이라 폐기했다. 남은 여지는
**INT8 양자화나 더 작은 백본**이다. 전처리는 위의 "수치를 바꾸는" 후보들뿐이다.

> **엔진 시간은 GPU 클럭에 크게 좌우된다.** 프레임의 44%만 GPU 가 일하므로 드라이버가
> 저부하로 판단해 클럭을 내린다. 부스트 상태(1965MHz)에서 재면 배치 32가 **5.76ms**
> (이미지당 0.180ms, fp16 피크의 91%) 로 여기가 천장이다. 머신 간 비교나 회귀 측정을
> 할 거면 `sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 1900` 으로 **양쪽 다 고정**해야
> 한다. 고정하지 않으면 같은 코드가 ±40% 흔들린다.

세그먼테이션 쪽에서 **N 자체를 줄이는 것**이 가장 값싼 지렛대다 (`sam_node` 의
`area_min` 상향, `--min-segment-pixels`) — 모든 단계가 N 에 선형이라 그대로 비례해서
줄어든다.

---

## 7. 모델과 엔진 (`models/`)

바이너리는 버전 관리 대상이 아니며 아래로 재생성한다. 모두 conda `clip` 환경에서 실행한다
(TensorRT 가 거기에만 있다).

```bash
python meridian_clip/download_weights.py                          # ViT-B-32.pt (SHA256 검증)

python meridian_clip/export_onnx.py  --part visual_pooled_value   # 노드 기본값
python meridian_clip/build_engine.py --part visual_pooled_value

python meridian_clip/export_onnx.py  --part text
python meridian_clip/build_engine.py --part text --max-batch 128 --opt-batch 32
```

`--part` 는 `visual`(cls) / `visual_pooled`(patch) / `visual_pooled_value`(value) / `text`.
쓸 pooling mode 것만 있으면 된다.

빌드 품질 손잡이는 두 개다. 둘 다 `trtexec` 의 같은 이름 옵션과 대응한다.

| 인자 | 기본값 | 뜻 (`trtexec`) |
|---|---|---|
| `--avg-timing` | 8 | tactic 하나를 몇 번 재서 평균낼지 (`--avgTiming`) |
| `--optimization-level` | 3 | tactic 탐색 범위 0~5, 5가 가장 넓다 (`--builderOptimizationLevel`) |

올리면 빌드가 느려지는 대신 tactic 선택이 노이즈에 덜 흔들린다. 장비를 바꿔
성능을 다시 잴 때는 두 값을 맞춰 두어야 비교가 성립한다.

fp16 은 커널 **내부**만이다. 네트워크 입출력은 ONNX 그대로 fp32 로 두므로
(`DIRECT_IO` / fp16 IO format 을 켜지 않는다) 노드 쪽 버퍼 dtype 은 바뀌지 않는다.

| 파일 | 용도 |
|---|---|
| `ViT-B-32.pt` | torch 백엔드 + torch 텍스트 인코더 |
| `clip_vit_b32_visual_fp16.engine` | `cls` 전용 |
| `clip_vit_b32_visual_pooled_fp16.engine` | `mask_weighted_patch` |
| `clip_vit_b32_visual_pooled_value_fp16.engine` | `mask_weighted_value` (기본값) |
| `clip_vit_b32_text_fp16.engine` | 텍스트 인코더 (`.pt` 불필요) |
| `align_value_to_cls.npy` | 기본 모드용 `--alignment-matrix` 512x512 정렬 행렬 (§3) |
| `align_patch_to_cls.npy` | `mask_weighted_patch` 용 정렬 행렬. 둘 다 엔진이 아니라 후처리 행렬이라 GPU/TensorRT 버전과 무관하다 |

**뒤의 두 pooled 엔진은 입출력 이름이 똑같아 파일만 봐서는 구분할 수 없다.**
어느 것을 쓸지는 노드가 `pooling_mode` 로 고른다
(`--pooled-engine-path` / `--value-engine-path`). 섞어 넣지 말 것.

엔진은 GPU·TensorRT 버전에 종속이다. 환경이 바뀌면 다시 만든다.
검증값: ONNX vs torch cos 0.99999988, TensorRT vs ONNX cos 0.999971,
속도 3.36 ms vs torch 6.94 ms (**2.07배**, batch 8, RTX 2080 Ti).

> **`POOLING_EPS` 는 `1e-4` 여야 하고 `clip_backend.py` 와 `export_onnx.py` 에서 같아야 한다.**
> fp16 최소 정규수가 약 6.1e-5 라 `1e-6` 같은 값은 TensorRT 에서 0 으로 flush 되어
> 빈 마스크에서 `0/0 = NaN` 이 된다. 실제로 재현된 버그다.

---

## 8. 빌드와 실행 환경

**빌드와 실행의 파이썬이 다르다.** 빌드는 시스템 파이썬(colcon)이 하고, 실행은
torch/cuda/tensorrt/clip 이 있는 conda `clip` 환경이 한다.

이어 주는 것이 콘솔 스크립트의 shebang 이고, `setup.cfg` 가 그것을 정한다.

```ini
[build_scripts]
executable=/usr/bin/env python3
```

> **clone 후 고칠 것이 없다.** 대신 **실행 전에 conda 환경을 활성화해야 한다.**
> `/usr/bin/env` 가 PATH 에서 `python3` 를 찾으므로, 활성화된 환경이 곧 노드가
> 도는 환경이다.
>
> 이 줄이 아예 없으면 colcon 은 **자신을 띄운 파이썬**(`/usr/bin/python3`)을
> shebang 에 박고 `ros2 launch` 가 `ModuleNotFoundError` 로 죽는다. colcon 이
> `sys.executable` 로 `setup.py` 를 돌리는데 `/usr/bin/colcon` 의 shebang 이
> 시스템 파이썬이라, **conda 를 켜 두고 빌드해도 결과가 같다.**
>
> 절대경로(`/home/<user>/miniconda3/...`)를 박으면 활성화 없이도 돌지만 파일이
> 머신마다 달라진다. 상대경로는 **쓸 수 없다** -- shebang 의 상대경로는 스크립트
> 위치가 아니라 실행하는 프로세스의 cwd 기준으로 풀려서 홈에서 띄울 때만
> 동작한다. 커널은 `~` 도 확장하지 않는다 (`#!~/miniconda3/...` 는 항상
> `bad interpreter` 다).

```bash
# 빌드 — 평범하게. conda 활성화 여부는 결과에 영향이 없다
cd ~/meridian && source /opt/ros/humble/setup.bash
colcon build --packages-select meridian_msgs meridian_clip

# 실행 — conda 환경을 켜고
conda activate clip
source ~/meridian/install/setup.bash
ros2 launch meridian_clip clip_inference.launch.py
```

활성화를 잊으면 base 파이썬을 잡아 `ModuleNotFoundError` 로 죽는다. 조용히 잘못된
파이썬으로 도는 경우는 없으므로 증상이 바로 드러난다.

> 환경 이름이 `clip` 이 아니면 그 이름으로 활성화하면 된다. `setup.cfg` 는
> 건드릴 필요가 없다.

> **scipy 가 필요하다.** `build_regions` 가 `scipy.ndimage.find_objects` 로 모든 라벨의
> bbox 를 한 번에 얻는다 (§6). `package.xml` 에 `python3-scipy` 로 선언돼 있고 conda
> `clip` 환경에도 있어야 한다.

---

## 9. 오프라인 도구 (`~/meridian/tools/`)

한 번 돌고 끝나는 검증용 스크립트. CLIP 가중치를 직접 로드한다.

| 도구 | 용도 |
|---|---|
| **`benchmark_stages.py`** | **Preprocessing / CLIP Encoder / Postprocessing 단계별 실행 시간. 데이터셋·카메라·FastSAM 불필요** (합성 프레임). `--variant both` 로 최적화 전후 A/B (§6) |
| `compare_pooling.py` | 사진 한 장으로 pooling 3종을 나란히 비교. crop/mask 를 한 번만 만들어 공유하므로 **pooling 만** 달라진다 |
| `benchmark_pooling.py` | VOC2012 로 pooling 별 zero-shot top-1 (§3 첫 표) |
| `benchmark_language.py` | VOC2012 로 **언어 쪽** — text→image 검색과 프롬프트 표현 강건성 ([언어 쪽 성능](#언어-쪽-성능)) |
| `fit_alignment.py` | 정렬 행렬 `W` 를 최소제곱으로 만든다. **정답 라벨 불필요** |
| `clip_selftest.py` | ROS 토픽 없이 노드를 직접 만들어 같은 코드 경로를 태움. `--check-parity` |
| `check_embedding_layout.py` | 메시지 평탄화 레이아웃 검증 |
| `single_image_test.py` | **현재 실행 불가** — 삭제된 `fastsam_ros.fastsam_segmenter` 를 import 한다 |

> **이 도구들은 첫 모드의 노드로 crop/mask 를 한 번 만들어 나머지 모드에 그대로
> 넘긴다.** 세 모드가 같은 crop 을 공유해야 pooling 차이만 분리되기 때문이다. 그래서
> `build_regions` 는 `pooling_mode` 를 보고 마스크를 생략하면 안 된다 — 생략 여부는
> `with_masks` 인자로 **호출자가 정한다** (§6 의 ⚠ 절).

```bash
python tools/compare_pooling.py --image ~/pencil.png --truth "a pencil" --crop-fit pad

# 세그먼테이션 품질을 변수에서 빼고 pooling 만 보고 싶을 때
python tools/compare_pooling.py --image x.png --labels mask.png --truth "a pencil"

# 언어 쪽. 이미지 임베딩을 --cache 에 받아 두면 두 번째부터는 텍스트만 다시 돈다
python tools/benchmark_language.py --cache /tmp/voc_embeddings.npz --per-variant
```

`--labels` 가 중요하다. FastSAM 품질이 나쁘면 pooling 비교가 아니라 세그먼테이션
비교가 되어버린다.

> **`--crop-policy` 기본값은 노드와 같은 `bbox` 다.** §3 의 표는 이 기본값으로 잰
> 값이라 인자 없이 재현된다 — §4 [crop 안의 마스크 밖 픽셀](#crop-안의-마스크-밖-픽셀---crop-policy) 참고.
> `--cache` 를 쓰는 `benchmark_language.py` / `benchmark_imagenet.py` 는 crop 설정을
> 캐시 식별에 포함하므로, 예전 `.npz` 가 새 기본값 실행에 섞이지 않는다.
>
> `fit_alignment.py` 만 예외다. 정렬 행렬은 `--crop-policy masked_bbox` 로 학습해야
> 하며(§3), `--target-crop-policy` 로 source/target 의 crop 정책을 따로 줄 수 있다.

---

## 10. 테스트

```bash
cd ~/meridian/src/meridian_clip
/usr/bin/python3 -m pytest test/ -q        # pytest 는 conda clip 환경에 없다
```

`test_mask_pooling.py` 는 CLIP 가중치 없이 도는 순수 torch 연산만 검증한다.
`test_pep257.py` 는 프로젝트 전반의 기존 D213 위반으로 실패한다 — 새 breakage 가 아니며
`flake8` 은 통과한다.

---

## 11. 남은 과제

- **세그먼테이션 품질이 현재 병목이다.** FastSAM everything 모드가 프레임당 50개 넘게
  뱉는데 대부분 물체가 아니라 벽·바닥 조각이다. 실제 장면에서 2등과의 격차가 0.02 를
  넘는 세그먼트가 15개 중 1개뿐이었고, 이건 어느 pooling 을 써도 마찬가지였다.
  `sam_node` 의 `area_min` 을 올리는 것이 첫 후보.
- **진짜 dense feature 는 아직 아니다.** crop-and-encode 라 세그먼트 수에 비례해
  비용이 든다. 프레임 전체를 한 번만 인코딩하는 방식은 검토하지 않았다.
  세그먼트당 단가가 곧 처리량 상한이라는 뜻이고, §6 의 N 별 표가 그 기울기다.
- **더 빠르게 하려면 정확도를 내주거나 백본을 바꿔야 한다.** 수치를 보존하는 최적화는
  §6 에서 소진했다. 남은 후보는 엔진 쪽 INT8 양자화 / 더 작은 백본, 또는 전처리
  resize 를 GPU·cv2 로 옮기는 것(픽셀이 달라지므로 `benchmark_pooling.py` 재측정 필수).
