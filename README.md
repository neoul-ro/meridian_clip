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
| `--preprocess-path` | `pil` | 224 를 어디서 만들지. `interp_aa` / `roi_align` 은 GPU 경로, §6 |
| `--crop-policy` | `bbox` | `masked_bbox` / `masked_full` 도 있음, §4 |
| `--patch-weight-gamma` | 1.0 | `w = r^gamma`. 1.0 이면 점유율 그대로 |
| `--min-patch-occupancy` | 0.0 | 이보다 낮은 패치는 가중치 0 |
| `--empty-mask-fallback` | `cls` | 가중치 합이 0 일 때. `skip` / `error` 도 있음 |
| `--min-segment-pixels` | 16 | 이보다 작은 세그먼트는 건너뜀 |
| `--prompts` / `--prompt-file` | 18개 기본값 | zero-shot 후보 |
| `--debug-save-dir` | `""` | crop/mask/occupancy PNG 저장 |

`gamma` 와 `min_patch_occupancy` 는 **엔진 밖에서** 적용되므로 바꿔도 재빌드가 필요 없다.

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
실행 간 변동은 ±5% 정도다.

> **프로파일의 min/max 는 세그먼트 개수를 고정하는 설정이 아니다.** 엔진이 받을 수
> 있는 배치 크기의 범위일 뿐이다. 프레임 처리 시간이 세그먼트 수 N 에 비례하는 것은
> 설정 문제가 아니라 구조다 — 세그먼트마다 crop 을 만들고, 224 로 리사이즈하고,
> ViT 에 통과시킨다. N=64면 그 일이 두 배다.

### opt 기준 성능 (N=32)

| 단계 | `cls` | `mask_weighted_value` **(기본값)** | `mask_weighted_patch` |
|---|---|---|---|
| 세그먼트 분해 + bbox crop | 2.404 | 2.373 | 2.326 |
| 224 전처리 — RGB | 7.208 | 6.879 | 6.825 |
| 224 전처리 — 마스크 | — | 2.909 | 2.662 |
| 패치 점유율 + 가중치 | — | 0.223 | 0.228 |
| **TensorRT 엔진** | **7.915** | **8.351** | **8.335** |
| D2H + 빈마스크 판정 | 0.088 | 0.218 | 0.211 |
| L2 정규화 + 정렬 | 0.060 | 0.049 | 0.050 |
| zero-shot 유사도 + top-k | 0.045 | 0.043 | 0.045 |
| 메시지 빌드 + publish | 1.503 | 1.492 | 1.484 |
| **Preprocessing** | **9.611** (50.0%) | **12.384** (54.9%) | **12.041** (54.3%) |
| **CLIP Encoder** | **7.915** (41.2%) | **8.351** (37.1%) | **8.335** (37.6%) |
| **Postprocessing** | **1.697** (8.8%) | **1.803** (8.0%) | **1.790** (8.1%) |
| **합계** | **19.224 ms** | **22.537 ms** | **22.165 ms** |
| **FPS** | **52.0** | **44.4** | **45.1** |
| 세그먼트 1개당 | 0.601 ms | 0.704 ms | 0.693 ms |

**세 모드의 차이는 마스크 경로가 전부다.** `cls` 가 3.3ms 빠른데 전부 Preprocessing
쪽이고(마스크 224 전처리 2.9ms + 점유율 0.22ms), **엔진은 7.92~8.35ms 로 사실상 같다.**
`value` 와 `patch` 는 22.5 vs 22.2ms 로 구분되지 않는다 — 엔진 파일만 다르고 코드
경로가 동일하니 당연하다. 모드 선택은 **속도가 아니라 정확도 문제다** (§3).

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

### 224 전처리 경로 3종 (`--preprocess-path`) — Jetson AGX Orin

`pil` 은 crop 을 PIL 로 만들어 CPU 에서 224 로 줄인다(**기본값, 배포 경로**).
GPU 경로 둘은 crop 을 만들지 않고 프레임 한 장에서 바로 224 를 뽑는다.

> **측정 조건.** uHumans2 office s1_00h, 720x480, `--stride 10`,
> **831프레임 / 15,544 인스턴스, N 평균 18.7**, batch 32, 점유율 `exact`,
> Jetson AGX Orin MAXN / torch 2.8.0 / TensorRT 10.3.0. **평균 ms**.
> 2026-08-24 측정. 위의 §6 표들은 RTX 2080 Ti / 640x480 이므로 **섞어 읽지 않는다.**

| 모드 | 경로 | pre | enc | post | 합계 | FPS | top-1 (things) | cos 평균 | cos 최소 |
|---|---|---|---|---|---|---|---|---|---|
| **cls** | `pil` | 14.30 | 10.46 | 0.34 | 25.11 | 39.8 | 48.87% | — | — |
| | `roi_align` | **6.56** | 10.26 | 0.29 | **17.11** | **58.4** | 51.13% | 0.9795 | 0.8395 |
| | `interp_aa` | 7.26 | 10.21 | 0.28 | 17.75 | 56.3 | 48.20% | **0.9975** | **0.9463** |
| **value** | `pil` | 16.15 | 11.55 | 0.34 | 28.05 | 35.7 | 55.56% | — | — |
| | `roi_align` | **7.68** | 11.18 | 0.31 | **19.17** | **52.2** | 56.08% | 0.9848 | 0.8628 |
| | `interp_aa` | 8.45 | 11.15 | 0.30 | 19.90 | 50.2 | 55.27% | **0.9979** | **0.9567** |
| **patch** | `pil` | 16.31 | 11.48 | 0.35 | 28.13 | 35.5 | 53.61% | — | — |
| | `roi_align` | **7.76** | 11.10 | 0.32 | **19.17** | **52.2** | 55.68% | 0.9906 | 0.8970 |
| | `interp_aa` | 8.59 | 11.07 | 0.31 | 19.96 | 50.1 | 53.01% | **0.9990** | **0.9636** |

**pre 가 1.9~2.2배, 처리량이 1.41~1.47배** 빨라지고 zero-shot top-1 은 오차범위
안이다(−0.67 ~ +2.26pp). 빈 마스크 fallback 판정 갈림은 **9칸 전부 0건**이다.

**그래도 기본값은 `pil` 이다.** 임베딩 코사인이 0.998 대로 높지만 1.0 이 아니라,
이미 저장된 임베딩과 새 임베딩을 섞어 거리 비교를 하면 같은 물체가 0.95 대로
벌어진다 — 인스턴스 매칭과 재식별이 정확히 그것을 한다. 바꾸려면 **저장된 임베딩
전체를 재생성해야 한다.**

| 소비자 | 권장 |
|---|---|
| 임베딩 **거리**를 직접 쓴다 (매칭·재식별) | `interp_aa` — 꼬리가 0.96 대 0.86 으로 짧다 |
| **라벨링만** 쓰고 N 이 크게 변한다 | `roi_align` — pre 가 N 에 거의 무관하다 (+35%) |
| 기존 임베딩과 섞어야 한다 | `pil` |

GPU 경로는 `--crop-policy bbox --crop-fit pad` 전용이고, 다른 조합을 주면 노드가
뜨지 않는다. 자세한 근거·드리프트·버그 기록은 `docs/preprocess_paths.md`,
최적화 전체 기록은 `docs/최적화.md` 에 있다.

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

#### 효과 — 같은 엔진으로 코드만 바꿔 A/B

`src/meridian_clip_backup` 에 최적화 직전 코드가 남아 있어, **같은 엔진
(min=1/opt=32/max=64) 과 같은 `batch_size=32`** 로 코드만 갈아끼워 쟀다. 즉 아래 차이는
전부 코드에서 온 것이다. N=32, 40회 중앙값.

**`cls`**

| 단계 | 최적화 전 | **현재** | 배속 |
|---|---|---|---|
| **Preprocessing** | 34.168 | **9.294** | **3.68x** |
| CLIP Encoder | 7.517 | 7.835 | 0.96x (동일) |
| Postprocessing | 1.597 | 1.530 | 1.04x |
| **합계** | **43.282 ms** (23.1 FPS) | **18.659 ms** (53.6 FPS) | **2.32x** |

**`mask_weighted_value`** (기본값)

| 단계 | 최적화 전 | **현재** | 배속 |
|---|---|---|---|
| **Preprocessing** | 38.783 | **12.542** | **3.09x** |
| CLIP Encoder | 7.739 | 8.363 | 0.93x (동일) |
| Postprocessing | 1.589 | 1.558 | 1.02x |
| **합계** | **48.111 ms** (20.8 FPS) | **22.464 ms** (44.5 FPS) | **2.14x** |

**`mask_weighted_patch`**

| 단계 | 최적화 전 | **현재** | 배속 |
|---|---|---|---|
| **Preprocessing** | 40.021 | **12.394** | **3.23x** |
| CLIP Encoder | 7.671 | 8.545 | 0.90x (동일) |
| Postprocessing | 1.589 | 1.566 | 1.01x |
| **합계** | **49.282 ms** (20.3 FPS) | **22.505 ms** (44.4 FPS) | **2.19x** |

**Preprocessing 만 3.1~3.7배 줄었고 나머지 둘은 그대로다.** 엔진과 후처리는 손대지
않았으므로 당연한 결과이고, 그쪽의 ±0.3~0.9ms 차이는 실행 간 변동이다. 그 결과
Preprocessing 비중이 **79~81% → 50~56%** 로 내려가면서 엔진이 최대 단일 항목이 됐다.

정확도는 그대로다 — `value` top-1 87.98% → 87.92% (엔진 재빌드분 포함, 3,420개 중 2개).

> **재현 방법.** `meridian_clip_backup` 은 코드만 남기고 `models/` 는 지웠다
> (2.2GB 중복이었고, 백업 코드도 엔진 경로를 `~/meridian/src/meridian_clip/models/`
> 절대경로로 갖고 있어 읽지 않는다). 그래서 `sys.path` 앞에 백업 경로를 넣어
> import 하면 같은 엔진을 쓰면서 옛 코드로 돌릴 수 있다:
>
> ```python
> import sys
> sys.path.insert(0, "/home/sojin/meridian/src/meridian_clip_backup")
> from meridian_clip.clip_inference_node import ClipInferenceNode  # 옛 코드
> ```
>
> 이때 `batch_size=32` 를 명시해야 한다 — 백업의 기본값은 16이다.

#### 측정해 보고 **버린** 것

같은 실수를 반복하지 않도록 남긴다. 전부 실측 후 폐기했다.

| 시도 | 결과 | 왜 안 됐나 |
|---|---|---|
| **CPU/GPU 파이프라이닝** | **1.02~1.04x** (N=8 은 0.87x) | 배치 사이 메인 스레드 작업(`.cpu().numpy()` ×3, `np.where`, 버퍼 복사)이 GIL 을 쥐어 엔진 대기 창에 PIL 워커가 못 돈다. 격리 테스트에서는 2.14배가 나오지만 실제 루프에서는 사라진다 |
| **마스크 → 7×7 BOX 직행** | 2.4% | 전송량은 1024:1 로 주는데 병목이 224 해상도가 아니라 마스크당 PIL 호출 오버헤드였다. NEAREST 양자화가 빠져 점유율도 달라진다(비트 비동일) |
| **마스크 기하를 numpy gather 로** | **6배 느림** | PIL 의 NEAREST 인덱스를 실측으로 재현해 비트 일치까지 확인했으나, PIL 의 C resize 가 numpy fancy indexing 보다 훨씬 빠르다 |
| **CUDA graph** (GPU tail) | **0.95x (역효과)** | replay 가 입력을 static 버퍼로 `copy_` 해야 하는데 그 비용이 커널 4개보다 크다. 대역폭 바운드라 런치 오버헤드가 지배적이지 않다 |
| **`torch.compile`** (GPU tail) | 2.96x → 프레임의 **0.8%** | GPU tail 이 전처리의 4.7% 뿐이라 상한이 낮다 |
| **`torch.jit.script`** (GPU tail) | 1.96x → 프레임의 0.6% | 위와 같음 |
| **`reducing_gap=2/3`** | 0.93x (역효과) | 큰 축소비에서만 이득인데 세그먼트 crop 은 확대되는 경우가 많다 |
| **pinned buffer + non_blocking** | ~1.00x | `np.stack` + H2D 가 이미 전처리의 8.8% 뿐 |
| **스레드 12/16/24개** | ~1.02x | 8개에서 이미 평평하다 |

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
| **TensorRT 엔진** | 8.35 | **37.1%** |
| 224 전처리 — RGB | 6.88 | 30.5% |
| 224 전처리 — 마스크 | 2.91 | 12.9% |
| 세그먼트 분해 + bbox crop | 2.37 | 10.5% |
| 나머지 5개 단계 | 2.03 | 9.0% |

엔진이 최대 항목이다. `opt` 를 32로 올려 한 번 더 얻었으므로(위 표) 남은 여지는
**INT8 양자화나 더 작은 백본**이다. 전처리는 위의 "수치를 바꾸는" 후보들뿐이다.

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

**빌드와 실행의 파이썬이 다르다.**

> **clone 했다면 `setup.cfg` 의 `[build_scripts] executable` 을 먼저 고친다.**
> 저장소에는 작성자 머신의 절대경로가 박혀 있다.
>
> ```bash
> sed -i "s|^executable=.*|executable=$HOME/miniconda3/envs/clip/bin/python|" \
>     ~/meridian/src/meridian_clip/setup.cfg
> ```
>
> conda 를 다른 곳에 깔았거나 환경 이름이 `clip` 이 아니면 그 경로로 바꾼다
> (`conda run -n <env> python -c 'import sys; print(sys.executable)'` 로 확인).
>
> 이 줄이 없으면 colcon 은 **자신을 띄운 파이썬**(`/usr/bin/python3`)을 콘솔 스크립트
> shebang 에 박고, `ros2 launch` 가 `ModuleNotFoundError: No module named 'clip'` 로
> 죽는다. 실행 환경을 shebang 에 고정하는 것이 목적이라 절대경로여야 하고,
> 그래서 머신마다 다르다. 래퍼 `.sh` 를 두지 않는 이유이기도 하다.

```bash
# 빌드 — conda 를 끄고
eval "$(conda shell.bash hook)" && conda deactivate && conda deactivate
cd ~/meridian && source /opt/ros/humble/setup.bash
colcon build --packages-select meridian_clip

# 재빌드 후 매번 — shebang 을 conda clip 으로 되돌린다
sed -i "1s|.*|#!$HOME/miniconda3/envs/clip/bin/python|" \
    ~/meridian/install/meridian_clip/lib/meridian_clip/clip_inference_node
```

실행은 conda `clip` 환경 (torch/cuda/tensorrt/clip 이 거기 있다).

> **scipy 가 필요하다.** `build_regions` 가 `scipy.ndimage.find_objects` 로 모든 라벨의
> bbox 를 한 번에 얻는다 (§6). `package.xml` 에 `python3-scipy` 로 선언돼 있고 conda
> `clip` 환경에도 있어야 한다.

---

## 9. 오프라인 도구 (`~/meridian/tools/`)

한 번 돌고 끝나는 검증용 스크립트. CLIP 가중치를 직접 로드한다.

| 도구 | 용도 |
|---|---|
| `compare_pooling.py` | 사진 한 장으로 pooling 3종을 나란히 비교. crop/mask 를 한 번만 만들어 공유하므로 **pooling 만** 달라진다 |
| `benchmark_pooling.py` | VOC2012 로 pooling 별 zero-shot top-1 (§3 첫 표) |
| `benchmark_language.py` | VOC2012 로 **언어 쪽** — text→image 검색과 프롬프트 표현 강건성 ([언어 쪽 성능](#언어-쪽-성능)) |
| `fit_alignment.py` | 정렬 행렬 `W` 를 최소제곱으로 만든다. **정답 라벨 불필요** |
| `clip_selftest.py` | ROS 토픽 없이 노드를 직접 만들어 같은 코드 경로를 태움. `--check-parity` |
| `check_embedding_layout.py` | 메시지 평탄화 레이아웃 검증 |
| `single_image_test.py` | **현재 실행 불가** — 삭제된 `fastsam_ros.fastsam_segmenter` 를 import 한다 |

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
