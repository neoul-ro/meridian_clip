"""모델 디렉터리를 찾는 규칙. 노드와 launch 가 함께 쓴다.

**이 모듈은 무겁게 만들면 안 된다.** launch 파일이 import 하는데, ``ros2 launch``
는 시스템 파이썬으로 돌고 거기에는 torch/clip/tensorrt 가 없을 수 있다. 표준
라이브러리와 ament_index 만 쓴다.

왜 별도 모듈인가
----------------
찾는 규칙이 노드와 launch 두 곳에 필요한데, 두 벌로 두면 조용히 어긋난다.
실제로 어긋난 적이 있다 -- launch 는 ``~/meridian/models/clip`` 을, 노드는
``~/meridian/src/meridian_clip/models`` 를, 생성 스크립트는 ``<패키지>/models``
를 가리켜서 ``ros2 launch`` 가 기본값으로는 절대 뜨지 않았다.

절대경로를 소스에 박지 않는다
-----------------------------
워크스페이스 이름도 사용자 이름도 머신마다 다르다 (``~/meridian_ws``,
``~/yun/meridian_ws``, ``/opt/robot/ws`` ...). 경로는 항상 이 파일 위치나
ament share 디렉터리에서 **유도**하고, 그래도 안 되면 환경변수로 받는다.
"""

from __future__ import annotations

import os

from pathlib import Path
from typing import List

from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)


PACKAGE_NAME = "meridian_clip"

#: 모델을 다른 곳에 뒀을 때 알려주는 환경변수. meridian_seg 의
#: ``MERIDIAN_SEG_ENGINE`` 과 같은 규약이다.
MODEL_DIR_ENV_VAR = "MERIDIAN_CLIP_MODEL_DIR"

#: 이 디렉터리 안에서 기대하는 파일 이름들. 노드가 pooling 모드에 맞는 것을 고른다.
MODEL_FILENAMES = {
    "checkpoint": "ViT-B-32.pt",
    "visual": "clip_vit_b32_visual_fp16.engine",
    "visual_pooled": "clip_vit_b32_visual_pooled_fp16.engine",
    "visual_pooled_value": "clip_vit_b32_visual_pooled_value_fp16.engine",
    "text": "clip_vit_b32_text_fp16.engine",
    "align_patch": "align_patch_to_cls.npy",
    "align_value": "align_value_to_cls.npy",
}


def model_dir_candidates() -> List[Path]:
    """모델 디렉터리를 찾을 위치를 우선순위 순으로 돌려준다.

    엔진은 GPU 아키텍처와 TensorRT 버전에 묶여 이식되지 않고, .pt 까지 합치면
    2GB 가 넘어서 저장소에도 install 트리에도 넣지 않는다. 각 머신이
    download_weights.py / export_onnx.py / build_engine.py 로 직접 만든 것을
    여기서 찾는다 -- 세 스크립트 모두 ``<패키지 루트>/models`` 를 기본 출력으로
    쓰므로 3번이 그것과 같은 자리다.

    한 곳만 보면 안 되는 이유: ``--symlink-install`` 이면 ``__file__`` 이 소스
    트리를 가리키지만, 평범한 ``colcon build`` 면 install 트리를 가리킨다.
    후자에는 ``models/`` 가 없다.
    """
    found: List[Path] = []

    env = os.environ.get(MODEL_DIR_ENV_VAR, "").strip()
    if env:
        found.append(Path(env).expanduser())

    try:
        share = Path(get_package_share_directory(PACKAGE_NAME))
        found.append(share / "models")
    except (PackageNotFoundError, KeyError):
        pass

    # 소스 트리 (--symlink-install 또는 저장소에서 직접 실행)
    found.append(Path(__file__).resolve().parents[1] / "models")
    return found


def has_models(directory: Path) -> bool:
    """모델 파일이 실제로 하나라도 들어 있는 디렉터리인가.

    "디렉터리가 있는가" 로는 부족하다 -- setup.py 가 ``models/*.md`` 를 설치하려고
    ``share/meridian_clip/models`` 를 만들어 두므로, 비어 있는 그 디렉터리가
    소스 트리보다 먼저 뽑혀 버린다.
    """
    return any(
        (directory / filename).is_file()
        for filename in MODEL_FILENAMES.values()
    )


def default_model_dir() -> str:
    """모델이 실제로 들어 있는 첫 후보.

    하나도 없으면 존재하는 첫 디렉터리를, 그것도 없으면 마지막 후보를 돌려준다.
    없는 경로라도 돌려주는 편이 낫다 -- 그래야 엔진 로드 시점의
    ``FileNotFoundError`` 가 어디를 봤는지 그대로 보여준다. 여기서 raise 하면
    import 만 해도 죽어서 ``--backend torch`` 나 ``--model-dir`` 로 우회할
    기회가 사라진다.
    """
    tried = model_dir_candidates()
    return str(
        next(
            (p for p in tried if has_models(p)),
            next((p for p in tried if p.is_dir()), tried[-1]),
        )
    )


def describe_candidates() -> str:
    """에러 메시지에 붙일 "찾아본 곳" 목록."""
    lines = []

    for path in model_dir_candidates():
        if has_models(path):
            note = ""
        elif path.is_dir():
            note = "  (디렉터리는 있으나 모델 파일이 없음)"
        else:
            note = "  (없음)"
        lines.append(f"{path}{note}")

    return "\n  ".join(lines)
