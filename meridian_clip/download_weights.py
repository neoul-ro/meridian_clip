#!/usr/bin/env python3

"""
Meridian Perception Frontend - CLIP weight downloader.

OpenAI CLIP의 공식 checkpoint(.pt)를 패키지 로컬 models/ 아래로 내려받는다.

ONNX -> TensorRT 변환 파이프라인의 입력 아티팩트를 확보하는 것이 목적이며,
변환 결과물의 재현성을 위해 SHA256을 항상 검증한다.

URL 경로에 포함된 hex 문자열이 곧 해당 파일의 기대 SHA256이다.
이는 OpenAI CLIP 패키지가 사용하는 것과 동일한 규약이다.

동작 순서:
    1. 목적지에 이미 유효한 파일이 있으면 아무것도 하지 않는다.
    2. ~/.cache/clip 에 유효한 사본이 있으면 그것을 복사한다.
    3. 둘 다 없으면 원본 URL에서 내려받는다.

사용 예:
    python3 scripts/download_weights.py
    python3 scripts/download_weights.py --model ViT-B/32
    python3 scripts/download_weights.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request

from pathlib import Path
from typing import Dict, NamedTuple, Optional


class ModelSpec(NamedTuple):
    """checkpoint 하나의 다운로드 정보."""

    url: str
    filename: str
    sha256: str


# 노드가 쓰는 것은 ViT-B/32 뿐이지만, 나중에 모델을 바꿔 실험할 때를 위해
# 같은 규약의 다른 CLIP checkpoint도 함께 적어 둔다.
MODELS: Dict[str, ModelSpec] = {
    "ViT-B/32": ModelSpec(
        url=(
            "https://openaipublic.azureedge.net/clip/models/"
            "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/"
            "ViT-B-32.pt"
        ),
        filename="ViT-B-32.pt",
        sha256=(
            "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
        ),
    ),
    "ViT-B/16": ModelSpec(
        url=(
            "https://openaipublic.azureedge.net/clip/models/"
            "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/"
            "ViT-B-16.pt"
        ),
        filename="ViT-B-16.pt",
        sha256=(
            "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
        ),
    ),
    "ViT-L/14": ModelSpec(
        url=(
            "https://openaipublic.azureedge.net/clip/models/"
            "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/"
            "ViT-L-14.pt"
        ),
        filename="ViT-L-14.pt",
        sha256=(
            "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
        ),
    ),
}

# OpenAI CLIP 패키지가 쓰는 기본 캐시 위치
CLIP_CACHE_DIR = Path.home() / ".cache" / "clip"

CHUNK_SIZE = 1 << 20


def sha256_of(path: Path) -> str:
    """파일의 SHA256을 스트리밍으로 계산한다."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(CHUNK_SIZE)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def is_valid(path: Path, expected_sha256: str) -> bool:
    """경로에 기대 SHA256과 일치하는 파일이 있는지 확인한다."""
    if not path.is_file():
        return False

    return sha256_of(path) == expected_sha256


def report_progress(downloaded: int, total: int) -> None:
    """다운로드 진행률을 한 줄로 갱신한다."""
    megabytes = downloaded / (1 << 20)

    if total > 0:
        percent = 100.0 * downloaded / total
        total_megabytes = total / (1 << 20)

        line = (
            f"\r  {percent:5.1f}%  "
            f"{megabytes:7.1f} / {total_megabytes:.1f} MiB"
        )

    else:
        line = f"\r  {megabytes:7.1f} MiB"

    sys.stdout.write(line)
    sys.stdout.flush()


def download(url: str, destination: Path) -> None:
    """URL을 임시 파일로 내려받은 뒤 목적지로 옮긴다."""
    temporary = destination.with_suffix(destination.suffix + ".part")

    with urllib.request.urlopen(url) as response:
        total = int(
            response.headers.get("Content-Length", 0)
        )

        downloaded = 0

        with temporary.open("wb") as stream:
            while True:
                chunk = response.read(CHUNK_SIZE)

                if not chunk:
                    break

                stream.write(chunk)

                downloaded += len(chunk)
                report_progress(downloaded, total)

    sys.stdout.write("\n")

    temporary.replace(destination)


def find_cached_copy(spec: ModelSpec) -> Optional[Path]:
    """~/.cache/clip 에 있는 유효한 사본을 찾는다."""
    candidate = CLIP_CACHE_DIR / spec.filename

    if is_valid(candidate, spec.sha256):
        return candidate

    return None


def fetch(spec: ModelSpec, destination: Path, force: bool) -> Path:
    """checkpoint를 목적지에 확보하고 SHA256을 검증한다."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not force and is_valid(destination, spec.sha256):
        print(f"[skip] 이미 유효한 파일이 있습니다: {destination}")
        return destination

    cached = None if force else find_cached_copy(spec)

    if cached is not None:
        print(f"[copy] 검증된 캐시본을 복사합니다: {cached}")
        shutil.copyfile(cached, destination)

    else:
        print(f"[get ] 다운로드: {spec.url}")
        download(spec.url, destination)

    actual = sha256_of(destination)

    if actual != spec.sha256:
        destination.unlink(missing_ok=True)

        raise RuntimeError(
            "SHA256 불일치로 파일을 삭제했습니다.\n"
            f"  expected: {spec.sha256}\n"
            f"  actual:   {actual}"
        )

    print(f"[ok  ] SHA256 검증 통과: {destination}")

    return destination


def main() -> int:
    """CLI 진입점."""
    package_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="CLIP checkpoint(.pt)를 내려받고 SHA256을 검증한다.",
    )

    parser.add_argument(
        "--model",
        default="ViT-B/32",
        choices=sorted(MODELS),
        help="내려받을 CLIP 모델 (기본: ViT-B/32)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=package_root / "models",
        help="저장 디렉터리 (기본: <package>/models)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="캐시와 기존 파일을 무시하고 원본에서 다시 받는다",
    )

    arguments = parser.parse_args()

    spec = MODELS[arguments.model]
    destination = arguments.output_dir / spec.filename

    try:
        path = fetch(
            spec=spec,
            destination=destination,
            force=arguments.force,
        )

    except Exception as error:
        print(f"[fail] {error}", file=sys.stderr)
        return 1

    size_megabytes = path.stat().st_size / (1 << 20)

    print()
    print(f"model : {arguments.model}")
    print(f"path  : {path}")
    print(f"size  : {size_megabytes:.1f} MiB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
