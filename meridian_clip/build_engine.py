#!/usr/bin/env python3

"""
Meridian Perception Frontend - TensorRT engine builder.

ONNX visual encoder를 TensorRT 엔진으로 빌드하고 정확도와 속도를 검증한다.

pip로 설치한 TensorRT wheel에는 trtexec가 포함되지 않으므로 Builder API를
직접 사용한다. 기능은 trtexec --fp16 --minShapes/--optShapes/--maxShapes와
동일하다.

배치 크기는 프레임의 segment 수에 따라 달라지므로 optimization profile로
min/opt/max 범위를 지정한다. max는 노드의 batch_size 파라미터와 맞춰야 한다.
그보다 큰 배치는 노드가 알아서 나누어 넣는다.

생성된 .engine은 빌드한 GPU 아키텍처와 TensorRT 버전에서만 유효하다.
다른 장비로 옮길 때는 .onnx를 옮기고 대상 장비에서 다시 빌드해야 한다.

사용 예:
    python3 meridian_clip/build_engine.py
    python3 meridian_clip/build_engine.py --max-batch 32 --no-fp16
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path
from typing import Tuple

import numpy as np
import tensorrt as trt


INPUT_NAME = "images"
TEXT_INPUT_NAME = "tokens"
OUTPUT_NAME = "embeddings"

# --part visual_pooled / visual_pooled_value 전용 입출력.
# 두 part 는 그래프 내부만 다르고 입출력 이름/모양이 같다. 어느 쪽인지는
# 엔진 파일만 봐서는 알 수 없으므로 노드가 pooling_mode 로 골라 넘긴다.
WEIGHTS_INPUT_NAME = "patch_weights"
CLS_OUTPUT_NAME = "cls_embeddings"

POOLED_PARTS = ("visual_pooled", "visual_pooled_value")

# ViT-B/32 기준. 두 인코더 모두 같은 512차원 공간으로 투영한다.
EMBEDDING_DIM = 512

# fp16 변환 후에도 downstream의 cosine similarity 비교가 유효하려면
# 이 정도 유사도는 유지되어야 한다.
PARITY_COSINE_THRESHOLD = 0.999

BENCHMARK_WARMUP = 10
BENCHMARK_ITERATIONS = 50


def read_patch_count(onnx_path: Path) -> int:
    """visual_pooled ONNX 에서 patch 개수를 읽는다 (patch_weights 의 1번 축)."""
    import onnx

    model = onnx.load(str(onnx_path))

    for graph_input in model.graph.input:
        if graph_input.name != WEIGHTS_INPUT_NAME:
            continue

        dimensions = graph_input.type.tensor_type.shape.dim

        if len(dimensions) != 2:
            raise RuntimeError(
                f"{WEIGHTS_INPUT_NAME} 는 [batch, patch] 여야 합니다."
            )

        return int(dimensions[1].dim_value)

    raise RuntimeError(
        f"{onnx_path} 에 {WEIGHTS_INPUT_NAME} 입력이 없습니다. "
        "export_onnx.py --part visual_pooled 로 만든 ONNX 인지 확인하세요."
    )


def build(
    onnx_path: Path,
    engine_path: Path,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    use_fp16: bool,
    workspace_gigabytes: int,
    inputs: Tuple[Tuple[str, Tuple[int, ...]], ...],
) -> None:
    """ONNX를 읽어 직렬화된 TensorRT 엔진을 만든다.

    inputs 는 (입력 이름, batch 를 제외한 shape) 쌍의 목록이다.
    visual_pooled 처럼 입력이 둘인 그래프는 optimization profile 에
    모든 입력의 범위를 등록해야 한다.
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # TensorRT 10부터 explicit batch가 기본이라 별도 플래그가 필요 없다.
    network = builder.create_network(0)

    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx_path)):
        messages = [
            parser.get_error(index)
            for index in range(parser.num_errors)
        ]
        raise RuntimeError(
            "ONNX 파싱 실패:\n"
            + "\n".join(str(message) for message in messages)
        )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        workspace_gigabytes << 30,
    )

    if use_fp16:
        if not builder.platform_has_fast_fp16:
            print("[warn] 이 GPU는 빠른 fp16을 지원하지 않습니다.")
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()

    for name, tail in inputs:
        profile.set_shape(
            name,
            (min_batch,) + tail,
            (opt_batch,) + tail,
            (max_batch,) + tail,
        )

    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)

    if serialized is None:
        raise RuntimeError("엔진 빌드에 실패했습니다.")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)


def load_engine(
    engine_path: Path,
) -> Tuple[trt.ICudaEngine, trt.IExecutionContext]:
    """직렬화된 엔진을 읽어 실행 컨텍스트를 만든다."""
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)

    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())

    if engine is None:
        raise RuntimeError("엔진 역직렬화에 실패했습니다.")

    return engine, engine.create_execution_context()


def infer(
    context: trt.IExecutionContext,
    feeds: dict,
    outputs: Tuple[str, ...] = (OUTPUT_NAME,),
) -> Tuple["np.ndarray", ...]:
    """엔진으로 한 배치를 추론한다. 버퍼는 torch 텐서를 그대로 쓴다.

    feeds 는 {입력 이름: numpy 배열}, outputs 는 읽어올 출력 이름들이다.
    입력이 둘인 pooled 엔진도 같은 함수로 돈다.
    """
    import torch

    batch = next(iter(feeds.values())).shape[0]

    # 텐서를 지역 변수로 붙잡아 둬야 실행 전에 해제되지 않는다.
    device_inputs = []

    for name, array in feeds.items():
        tensor = torch.from_numpy(np.ascontiguousarray(array)).cuda()
        device_inputs.append(tensor)

        context.set_input_shape(name, array.shape)
        context.set_tensor_address(name, tensor.data_ptr())

    device_outputs = []

    for name in outputs:
        tensor = torch.empty(
            (batch, EMBEDDING_DIM),
            dtype=torch.float32,
            device="cuda",
        )
        device_outputs.append(tensor)

        context.set_tensor_address(name, tensor.data_ptr())

    stream = torch.cuda.current_stream()
    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    return tuple(tensor.cpu().numpy() for tensor in device_outputs)


def make_tokens(count: int, context_length: int) -> "np.ndarray":
    """파리티 검증용 토큰 배치. 실제 tokenizer 출력을 그대로 쓴다."""
    import clip

    prompts = [
        f"a photo of object number {index}"
        for index in range(count)
    ]

    return clip.tokenize(
        prompts,
        context_length=context_length,
    ).numpy().astype(np.int32)


def output_names(part: str) -> Tuple[str, ...]:
    """엔진이 내보내는 출력 이름을 part 별로 고른다."""
    if part in POOLED_PARTS:
        return (OUTPUT_NAME, CLS_OUTPUT_NAME)

    return (OUTPUT_NAME,)


def make_feeds(
    part: str,
    batch: int,
    resolution: int,
    context_length: int,
    patch_count: int,
) -> dict:
    """검증용 입력 묶음을 part 에 맞춰 만든다."""
    if part == "text":
        return {TEXT_INPUT_NAME: make_tokens(batch, context_length)}

    images = np.random.randn(
        batch, 3, resolution, resolution
    ).astype(np.float32)

    if part not in POOLED_PARTS:
        return {INPUT_NAME: images}

    weights = np.random.rand(batch, patch_count).astype(np.float32)

    # 마스크가 빈 세그먼트(분모 clamp 경로)도 반드시 한 줄은 섞어서 검증한다.
    weights[0] = 0.0

    return {INPUT_NAME: images, WEIGHTS_INPUT_NAME: weights}


def check_parity(
    context: trt.IExecutionContext,
    onnx_path: Path,
    resolution: int,
    batches: Tuple[int, ...],
    part: str = "visual",
    context_length: int = 77,
    patch_count: int = 0,
) -> float:
    """ONNX fp32 출력과 TensorRT 출력을 비교한다."""
    import onnxruntime

    session = onnxruntime.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    outputs = output_names(part)

    worst = 1.0

    for batch in batches:
        feeds = make_feeds(
            part=part,
            batch=batch,
            resolution=resolution,
            context_length=context_length,
            patch_count=patch_count,
        )

        expected = session.run(list(outputs), feeds)
        actual = infer(context, feeds, outputs=outputs)

        for name, want, got in zip(outputs, expected, actual):
            cosine = float(
                np.min(
                    np.sum(want * got, axis=-1)
                    / (
                        np.linalg.norm(want, axis=-1)
                        * np.linalg.norm(got, axis=-1)
                    )
                )
            )
            max_diff = float(np.max(np.abs(want - got)))
            worst = min(worst, cosine)

            print(
                f"  batch={batch:<3d} {name:<15s} cos={cosine:.6f}  "
                f"max_abs_diff={max_diff:.3e}"
            )

    return worst


def torch_pooled_forward(model, images, weights):
    """clip_backend 의 mask_weighted_patch 경로를 torch 로 재현한다.

    벤치마크 비교 대상이 같은 계산이어야 하므로 여기에 그대로 적어 둔다.
    (export_onnx.CLIPVisualMaskPooledEncoder.forward 와 같은 식이다.)
    """
    import torch

    visual = model.visual

    x = visual.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

    cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )

    x = torch.cat([cls_token, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)

    patch_tokens = x[:, 1:, :].float()

    w = weights.unsqueeze(-1)
    object_token = (patch_tokens * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)

    return visual.ln_post(
        object_token.to(visual.proj.dtype)
    ) @ visual.proj


def benchmark(
    context: trt.IExecutionContext,
    checkpoint: Path,
    resolution: int,
    batch: int,
    part: str = "visual",
    patch_count: int = 0,
) -> None:
    """TensorRT와 기존 torch fp16 경로의 지연시간을 비교한다."""
    import time

    import torch

    feeds = make_feeds(
        part=part,
        batch=batch,
        resolution=resolution,
        context_length=77,
        patch_count=patch_count,
    )
    outputs = output_names(part)

    for _ in range(BENCHMARK_WARMUP):
        infer(context, feeds, outputs=outputs)

    start = time.perf_counter()
    for _ in range(BENCHMARK_ITERATIONS):
        infer(context, feeds, outputs=outputs)
    trt_milliseconds = (
        (time.perf_counter() - start) / BENCHMARK_ITERATIONS * 1000.0
    )

    print(f"  tensorrt fp16 : {trt_milliseconds:6.2f} ms  (batch={batch})")

    if not checkpoint.is_file():
        return

    import clip

    # 노드의 현재 경로와 동일한 조건: CUDA + fp16
    model, _ = clip.load(str(checkpoint), device="cuda")
    model.eval()

    tensor = torch.from_numpy(feeds[INPUT_NAME]).cuda().half()

    if part in POOLED_PARTS:
        weights = torch.from_numpy(feeds[WEIGHTS_INPUT_NAME]).cuda().float()

        def run():
            return torch_pooled_forward(model, tensor, weights)

    else:

        def run():
            return model.encode_image(tensor)

    with torch.inference_mode():
        for _ in range(BENCHMARK_WARMUP):
            run()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(BENCHMARK_ITERATIONS):
            run()
        torch.cuda.synchronize()

    torch_milliseconds = (
        (time.perf_counter() - start) / BENCHMARK_ITERATIONS * 1000.0
    )

    speedup = torch_milliseconds / trt_milliseconds

    print(f"  torch fp16    : {torch_milliseconds:6.2f} ms  (batch={batch})")
    print(f"  speedup       : {speedup:6.2f}x")


def main() -> int:
    """CLI 진입점."""
    package_root = Path(__file__).resolve().parent.parent
    models_dir = package_root / "models"

    parser = argparse.ArgumentParser(
        description="ONNX visual encoder를 TensorRT 엔진으로 빌드한다.",
    )
    parser.add_argument(
        "--part",
        choices=("visual", "visual_pooled", "visual_pooled_value", "text"),
        default="visual",
        help=(
            "빌드할 인코더. visual=이미지(CLS), "
            "visual_pooled=이미지(mask-weighted patch pooling), "
            "visual_pooled_value=같은 pooling 이되 마지막 블록의 value "
            "투영을 patch feature 로 사용 (MaskCLIP), "
            "text=프롬프트"
        ),
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="입력 ONNX (기본: models/clip_vit_b32_<part>.onnx)",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=None,
        help="출력 엔진 경로 (기본: models/clip_vit_b32_<part>_<precision>.engine)",
    )
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument(
        "--opt-batch",
        type=int,
        default=8,
        help="가장 자주 쓰이는 배치 크기. 여기에 맞춰 최적화된다.",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=32,
        help=(
            "visual 은 노드의 batch_size, text 는 프롬프트 개수 상한과 맞춘다."
        ),
    )
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument(
        "--context-length",
        type=int,
        default=77,
        help="text 전용. CLIP token 시퀀스 길이",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="fp32로 빌드한다",
    )
    parser.add_argument("--workspace", type=int, default=8, help="GiB")
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
    )

    arguments = parser.parse_args()

    is_text = arguments.part == "text"
    is_pooled = arguments.part in POOLED_PARTS

    onnx_path = arguments.onnx or (
        models_dir / f"clip_vit_b32_{arguments.part}.onnx"
    )

    if not onnx_path.is_file():
        print(
            f"[fail] ONNX가 없습니다: {onnx_path}\n"
            "       python3 meridian_clip/export_onnx.py "
            f"--part {arguments.part} 를 먼저 실행하세요.",
            file=sys.stderr,
        )
        return 1

    if not arguments.min_batch <= arguments.opt_batch <= arguments.max_batch:
        print(
            "[fail] min <= opt <= max 를 만족해야 합니다.",
            file=sys.stderr,
        )
        return 1

    use_fp16 = not arguments.no_fp16
    precision = "fp16" if use_fp16 else "fp32"

    # clip_vit_b32_<part>_<precision>.engine 규칙으로 통일한다.
    engine_path = arguments.engine or (
        models_dir / f"clip_vit_b32_{arguments.part}_{precision}.engine"
    )

    patch_count = 0

    if is_pooled:
        try:
            patch_count = read_patch_count(onnx_path)
        except Exception as error:
            print(f"[fail] {error}", file=sys.stderr)
            return 1

    if is_text:
        inputs = ((TEXT_INPUT_NAME, (arguments.context_length,)),)
    else:
        image_tail = (3, arguments.resolution, arguments.resolution)

        inputs = (
            ((INPUT_NAME, image_tail), (WEIGHTS_INPUT_NAME, (patch_count,)))
            if is_pooled
            else ((INPUT_NAME, image_tail),)
        )

    print(
        f"[trt ] version={trt.__version__} precision={precision} "
        f"part={arguments.part}"
    )
    print(
        f"[prof] batch min={arguments.min_batch} "
        f"opt={arguments.opt_batch} max={arguments.max_batch} "
        + (
            f"context_length={arguments.context_length}"
            if is_text
            else f"resolution={arguments.resolution}"
        )
        + (f" patches={patch_count}" if is_pooled else "")
    )
    print("[bld ] 빌드 중 (수 분 걸릴 수 있습니다)...")

    try:
        build(
            onnx_path=onnx_path,
            engine_path=engine_path,
            min_batch=arguments.min_batch,
            opt_batch=arguments.opt_batch,
            max_batch=arguments.max_batch,
            use_fp16=use_fp16,
            workspace_gigabytes=arguments.workspace,
            inputs=inputs,
        )
    except Exception as error:
        print(f"[fail] {error}", file=sys.stderr)
        return 1

    size_megabytes = engine_path.stat().st_size / (1 << 20)
    print(f"[ok  ] {engine_path}  ({size_megabytes:.1f} MiB)")

    engine, context = load_engine(engine_path)

    print("[parity] onnx fp32 vs tensorrt")
    batches = tuple(
        sorted(
            {
                arguments.min_batch,
                arguments.opt_batch,
                arguments.max_batch,
            }
        )
    )
    worst = check_parity(
        context=context,
        onnx_path=onnx_path,
        resolution=arguments.resolution,
        batches=batches,
        part=arguments.part,
        context_length=arguments.context_length,
        patch_count=patch_count,
    )

    if worst < PARITY_COSINE_THRESHOLD:
        print(
            f"[fail] 코사인 유사도 {worst:.6f} < {PARITY_COSINE_THRESHOLD}",
            file=sys.stderr,
        )
        return 1

    # 텍스트는 시작할 때 한 번만 도는 경로라 벤치마크에 의미가 없다.
    if not arguments.skip_benchmark and not is_text:
        print("[bench]")
        benchmark(
            context=context,
            checkpoint=models_dir / "ViT-B-32.pt",
            resolution=arguments.resolution,
            batch=arguments.opt_batch,
            part=arguments.part,
            patch_count=patch_count,
        )

    print()
    print(f"engine: {engine_path}")
    print(f"cosine: {worst:.6f}")

    del context
    del engine

    return 0


if __name__ == "__main__":
    sys.exit(main())
