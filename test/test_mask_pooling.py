# Copyright 2026 Meridian
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""mask-weighted patch pooling 커널 테스트.

CLIP 가중치가 없어도 도는 순수 torch 연산만 검증한다. clip 패키지는
TorchBackend.__init__ 안에서만 import 되므로 colcon test 를 돌리는
시스템 파이썬에서도 실행된다.

ViT forward 재현(extract_final_visual_tokens)은 실제 checkpoint 가 필요하므로
여기서 다루지 않는다. tools/clip_selftest.py --check-parity 가 담당한다.
"""

import pytest

import torch

from meridian_clip.clip_backend import (
    compute_patch_occupancy,
    mask_weighted_pool,
    project_object_token,
)


RESOLUTION = 224
GRID = 7
PATCH_COUNT = GRID * GRID
PATCH_SIZE = RESOLUTION // GRID
WIDTH = 768
OUTPUT_DIM = 512


class FakeVisual(torch.nn.Module):
    """ln_post 와 proj 만 있는 최소 visual tower."""

    def __init__(self):
        super().__init__()
        self.ln_post = torch.nn.LayerNorm(WIDTH)
        self.proj = torch.nn.Parameter(
            torch.randn(WIDTH, OUTPUT_DIM) * (WIDTH ** -0.5)
        )


class FakeModel(torch.nn.Module):
    """project_object_token 이 필요한 것만 가진 가짜 CLIP."""

    def __init__(self):
        super().__init__()
        self.visual = FakeVisual()


@pytest.fixture
def patch_tokens():
    torch.manual_seed(0)

    return torch.randn(1, PATCH_COUNT, WIDTH)


def test_occupancy_shape():
    """[B,1,224,224] 마스크는 [B,49] 점유율이 된다."""
    masks = torch.zeros(2, 1, RESOLUTION, RESOLUTION)

    occupancy = compute_patch_occupancy(masks, GRID)

    assert occupancy.shape == (2, PATCH_COUNT)


def test_occupancy_rejects_bad_shape():
    """차원이나 채널 수가 틀리면 조용히 넘어가지 않는다."""
    with pytest.raises(ValueError):
        compute_patch_occupancy(torch.zeros(1, RESOLUTION, RESOLUTION), GRID)

    with pytest.raises(ValueError):
        compute_patch_occupancy(
            torch.zeros(1, 3, RESOLUTION, RESOLUTION),
            GRID,
        )


def test_occupancy_accepts_255_scale():
    """PIL 경유로 0..255 가 들어와도 0..1 로 해석한다."""
    masks = torch.full((1, 1, RESOLUTION, RESOLUTION), 255.0)

    occupancy = compute_patch_occupancy(masks, GRID)

    assert torch.allclose(occupancy, torch.ones_like(occupancy))


def test_empty_mask_yields_zero_weight(patch_tokens):
    """마스크가 비면 가중치 합이 0이고 pooled 토큰은 0 벡터다.

    노드는 이 상태를 보고 empty_mask_fallback 을 적용한다.
    """
    occupancy = compute_patch_occupancy(
        torch.zeros(1, 1, RESOLUTION, RESOLUTION),
        GRID,
    )

    pooled, weights = mask_weighted_pool(patch_tokens, occupancy)

    assert float(weights.sum()) == 0.0
    assert torch.allclose(pooled, torch.zeros_like(pooled))


def test_full_mask_equals_simple_mean(patch_tokens):
    """전부 1인 마스크의 가중평균은 49개 패치의 단순 평균과 같다."""
    occupancy = compute_patch_occupancy(
        torch.ones(1, 1, RESOLUTION, RESOLUTION),
        GRID,
    )

    pooled, _ = mask_weighted_pool(patch_tokens, occupancy)

    assert torch.allclose(
        pooled,
        patch_tokens.mean(dim=1),
        atol=1e-6,
    )


def test_single_patch_mask_selects_that_patch(patch_tokens):
    """한 패치만 1인 마스크는 그 패치 토큰만 남긴다.

    이 테스트가 통과해야 occupancy 의 인덱스 순서(row-major)가 patch token
    순서와 일치한다고 말할 수 있다. 격자를 잘못 펴면 다른 패치가 뽑힌다.
    """
    for index in (0, 1, GRID, PATCH_COUNT - 1):
        row = index // GRID
        column = index % GRID

        masks = torch.zeros(1, 1, RESOLUTION, RESOLUTION)
        masks[
            :,
            :,
            row * PATCH_SIZE:(row + 1) * PATCH_SIZE,
            column * PATCH_SIZE:(column + 1) * PATCH_SIZE,
        ] = 1.0

        occupancy = compute_patch_occupancy(masks, GRID)

        assert int((occupancy > 0.0).sum()) == 1
        assert float(occupancy[0, index]) == pytest.approx(1.0)

        pooled, _ = mask_weighted_pool(patch_tokens, occupancy)

        assert torch.allclose(
            pooled,
            patch_tokens[:, index, :],
            atol=1e-6,
        )


def test_gamma_sharpens_weights(patch_tokens):
    """지수 gamma 가 점유율에 적용된다."""
    occupancy = torch.full((1, PATCH_COUNT), 0.5)

    _, weights = mask_weighted_pool(patch_tokens, occupancy, gamma=2.0)

    assert float(weights[0, 0]) == pytest.approx(0.25)


def test_min_patch_occupancy_zeroes_low_patches(patch_tokens):
    """하한 미만인 패치는 가중치가 0이 된다."""
    occupancy = torch.zeros(1, PATCH_COUNT)
    occupancy[0, 0] = 0.9
    occupancy[0, 1] = 0.1

    _, weights = mask_weighted_pool(
        patch_tokens,
        occupancy,
        min_patch_occupancy=0.5,
    )

    assert float(weights[0, 0]) == pytest.approx(0.9)
    assert float(weights[0, 1]) == 0.0


def test_pool_rejects_mismatched_counts(patch_tokens):
    """패치 개수와 점유율 개수가 다르면 실패한다."""
    with pytest.raises(ValueError):
        mask_weighted_pool(patch_tokens, torch.ones(1, PATCH_COUNT - 1))

    with pytest.raises(ValueError):
        mask_weighted_pool(patch_tokens, torch.ones(PATCH_COUNT))


def test_projection_shape_and_norm():
    """[B,768] 객체 토큰은 [B,512] 단위벡터가 된다."""
    torch.manual_seed(0)

    model = FakeModel()
    object_token = torch.randn(3, WIDTH)

    embeddings = project_object_token(model, object_token, normalize=True)

    assert embeddings.shape == (3, OUTPUT_DIM)

    norms = embeddings.norm(dim=-1)

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_projection_without_normalize_is_not_unit():
    """normalize=False 면 정규화하지 않는다 (노드가 마지막에 담당)."""
    torch.manual_seed(0)

    model = FakeModel()

    embeddings = project_object_token(
        model,
        torch.randn(2, WIDTH),
        normalize=False,
    )

    assert embeddings.shape == (2, OUTPUT_DIM)
    assert not torch.allclose(
        embeddings.norm(dim=-1),
        torch.ones(2),
        atol=1e-3,
    )
