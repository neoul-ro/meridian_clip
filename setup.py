"""meridian_clip 패키지 설치 스크립트.

여기서 하는 특별한 일이 하나 있다: **``ros2 run`` 이 쓸 파이썬을 빌드할 때
자동으로 찾아서 콘솔 스크립트 shebang 에 박는다.** 아래 RUNTIME PYTHON 절 참고.
"""

import os
import subprocess
import sys

from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

try:
    # setuptools 가 distutils 명령을 자기 네임스페이스로 흡수한 뒤
    from setuptools.command.build_scripts import build_scripts
except ImportError:
    # Ubuntu 22.04 / ROS Humble 기본값인 setuptools 58 에는 아직 없다
    from distutils.command.build_scripts import build_scripts


package_name = "meridian_clip"


# =====================================================================
# RUNTIME PYTHON 자동 감지
# =====================================================================
# clip_inference_node 는 torch / clip / tensorrt 가 있는 파이썬에서만 돈다.
# 그게 어느 파이썬인지가 플랫폼마다 다르다:
#
#   Jetson (JetPack)  : /usr/bin/python3 에 torch/tensorrt 가 이미 있다
#   x86 개발 박스     : conda 환경(예: ~/miniconda3/envs/clip)에만 있다
#
# 예전에는 이걸 setup.cfg 의 [build_scripts] executable 에 손으로 적게 했다.
# 그러면 (a) clone 한 사람이 자기 플랫폼을 판단해서 sed 를 돌려야 하고
# (b) 그 값이 작성자 머신 절대경로라 커밋하면 남의 빌드가 깨진다.
# 그래서 **빌드가 직접 후보들을 검사해서 고른다.**
#
# 검사는 import 가 아니라 importlib.util.find_spec 이라 torch 를 실제로
# 로드하지 않는다 -- 후보당 0.01초라 매 빌드마다 돌려도 부담이 없다.

#: 자동 감지를 무시하고 특정 파이썬을 강제할 때 쓰는 환경변수.
#: 커밋되지 않으므로 머신마다 다른 값을 안전하게 줄 수 있다.
RUNTIME_PYTHON_ENV_VAR = "MERIDIAN_CLIP_PYTHON"

#: 노드가 import 해야만 도는 모듈. 이게 전부 있는 파이썬을 고른다.
REQUIRED_MODULES = ("torch", "clip")

#: 있으면 좋지만 없어도 --backend torch 로 돌아간다.
OPTIONAL_MODULES = ("tensorrt",)

_PROBE = (
    "import importlib.util as u, sys;"
    "print(sys.version_info[0], sys.version_info[1],"
    "*[m for m in {names!r} if u.find_spec(m)])"
)


def is_jetson():
    """NVIDIA Jetson(Tegra) 보드인가.

    JetPack 이 torch/tensorrt 를 시스템 파이썬에 깔아 주므로 conda 가 없다.
    두 파일 중 하나만 있어도 Jetson 이다 -- L4T 는 nv_tegra_release 를 두고,
    device-tree model 은 "NVIDIA Jetson AGX Orin" 같은 문자열을 담는다.
    """
    if Path("/etc/nv_tegra_release").exists():
        return True

    try:
        model = Path("/proc/device-tree/model").read_bytes()
    except OSError:
        return False

    return b"jetson" in model.lower() or b"tegra" in model.lower()


def probe(interpreter):
    """(major, minor, 가진 모듈 집합). 못 돌리면 None."""
    names = REQUIRED_MODULES + OPTIONAL_MODULES

    try:
        out = subprocess.run(
            [str(interpreter), "-c", _PROBE.format(names=list(names))],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if out.returncode != 0:
        return None

    fields = out.stdout.split()

    if len(fields) < 2:
        return None

    return int(fields[0]), int(fields[1]), set(fields[2:])


def runtime_python_override():
    """``MERIDIAN_CLIP_PYTHON`` 으로 직접 지정한 인터프리터. 없으면 None."""
    value = os.environ.get(RUNTIME_PYTHON_ENV_VAR, "").strip()
    return Path(value).expanduser() if value else None


def runtime_python_candidates():
    """검사할 인터프리터를 우선순위 순으로.

    존재하지 않는 경로도 그냥 넣는다 -- probe() 가 걸러 준다.
    """
    found = []

    # 지금 활성화된 conda / venv 환경
    for var in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        prefix = os.environ.get(var, "").strip()

        if prefix:
            found.append(Path(prefix) / "bin" / "python")

    # 흔한 conda 설치 위치의 모든 환경. 이름을 clip 으로 강제하지 않는다 --
    # 사람마다 다르게 부른다.
    roots = [Path(os.environ.get("CONDA_PREFIX", "/nonexistent")).parent]
    roots += [
        Path.home() / name / "envs"
        for name in ("miniconda3", "anaconda3", "miniforge3", "mambaforge")
    ]

    for root in roots:
        try:
            entries = sorted(root.iterdir())
        except OSError:
            continue

        found.extend(entry / "bin" / "python" for entry in entries)

    # Jetson/JetPack, 그리고 시스템에 직접 설치한 경우
    found.append(Path("/usr/bin/python3"))

    # 최후: colcon 을 띄운 파이썬
    found.append(Path(sys.executable))

    seen = set()
    unique = []

    for path in found:
        key = str(path)

        if key not in seen:
            seen.add(key)
            unique.append(path)

    return unique


def detect_runtime_python():
    """노드를 돌릴 수 있는 파이썬을 고른다.

    조건 두 가지를 **모두** 만족해야 한다:

    1. ``REQUIRED_MODULES`` 를 전부 가지고 있을 것
    2. 빌드 파이썬과 같은 (major, minor) 일 것 -- ROS 2 가 생성한
       meridian_msgs 가 ``lib/pythonX.Y/site-packages`` 에 깔리므로,
       버전이 다르면 그 메시지를 import 하지 못한다

    ``MERIDIAN_CLIP_PYTHON`` 으로 직접 지정했으면 **검사를 통과하지 못해도 그것을
    쓴다.** 명시적 지정이 자동 감지를 이기는 것이 덜 놀랍고, 아직 환경을 다 만들지
    않은 상태에서 미리 경로를 박아 두는 용법도 막지 않는다. 대신 무엇이 모자란지는
    경고로 알려 준다.

    아무것도 못 찾으면 빌드 파이썬을 그대로 쓰고 경고만 한다. 여기서 실패시키면
    모델 없이 문법만 확인하려는 빌드까지 막힌다.
    """
    want = sys.version_info[:2]
    override = runtime_python_override()

    if override is not None:
        result = probe(override)

        if result is None:
            print(
                f"[meridian_clip] 경고: {RUNTIME_PYTHON_ENV_VAR}={override} 를 "
                "실행할 수 없습니다. 경로를 확인하세요."
            )
        else:
            major, minor, modules = result
            missing = sorted(set(REQUIRED_MODULES) - modules)

            if (major, minor) != want:
                print(
                    f"[meridian_clip] 경고: {RUNTIME_PYTHON_ENV_VAR} 이 "
                    f"python{major}.{minor} 인데 ROS 는 "
                    f"python{want[0]}.{want[1]} 입니다. "
                    "meridian_msgs 를 import 하지 못할 수 있습니다."
                )

            if missing:
                print(
                    f"[meridian_clip] 경고: {RUNTIME_PYTHON_ENV_VAR} 에 "
                    f"{', '.join(missing)} 이(가) 없습니다."
                )

        return override, (result[2] if result else set())

    best = None

    for candidate in runtime_python_candidates():
        result = probe(candidate)

        if result is None:
            continue

        major, minor, modules = result

        if (major, minor) != want:
            continue

        if not set(REQUIRED_MODULES).issubset(modules):
            continue

        # tensorrt 까지 있으면 즉시 채택. 없으면 후보로만 잡아 두고
        # 더 나은 것이 있는지 계속 본다.
        if set(OPTIONAL_MODULES).issubset(modules):
            return candidate, modules

        if best is None:
            best = (candidate, modules)

    if best is not None:
        return best

    return None, set()


class BuildScriptsWithRuntimePython(build_scripts):
    """콘솔 스크립트 shebang 을 감지된 런타임 파이썬으로 박는다.

    setuptools 는 기본적으로 ``sys.executable`` (= colcon 을 띄운 파이썬)을 쓰는데,
    거기에는 torch/clip 이 없어서 ``ros2 run`` 이 ModuleNotFoundError 로 죽는다.

    shebang 은 커널이 그대로 exec 하므로 상대경로도 ``$HOME`` 도 확장되지 않는다 --
    **여기만은 절대경로일 수밖에 없다.** 그래서 소스에 적어 두지 않고 빌드 때
    채운다.
    """

    def finalize_options(self):
        build_scripts.finalize_options(self)

        platform_name = "Jetson" if is_jetson() else "일반 PC"
        interpreter, modules = detect_runtime_python()

        if interpreter is None:
            print(
                f"[meridian_clip] 경고: 플랫폼={platform_name} — "
                f"{'/'.join(REQUIRED_MODULES)} 를 모두 갖춘 python"
                f"{sys.version_info[0]}.{sys.version_info[1]} 을 찾지 못했습니다.\n"
                f"[meridian_clip]   shebang 은 {sys.executable} 로 둡니다. "
                "clip_inference_node 는 이대로면 실행되지 않습니다.\n"
                f"[meridian_clip]   환경을 만든 뒤 다시 빌드하거나, "
                f"{RUNTIME_PYTHON_ENV_VAR}=<파이썬 경로> 로 직접 지정하세요."
            )
            return

        self.executable = str(interpreter)

        how = (
            f"{RUNTIME_PYTHON_ENV_VAR} 지정"
            if runtime_python_override() is not None
            else "자동 감지"
        )
        missing = sorted(
            (set(REQUIRED_MODULES) | set(OPTIONAL_MODULES)) - modules
        )

        print(
            f"[meridian_clip] 플랫폼={platform_name} · 런타임 파이썬 {how}: "
            f"{interpreter}\n"
            f"[meridian_clip]   있는 모듈: "
            f"{', '.join(sorted(modules)) or '없음'}"
            + (f"  ·  없는 모듈: {', '.join(missing)}" if missing else "")
        )


setup(
    cmdclass={
        "build_scripts": BuildScriptsWithRuntimePython,
    },
    name=package_name,
    version="0.0.2",
    packages=find_packages(
        exclude=["test", "test.*"],
    ),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [
                os.path.join(
                    "resource",
                    package_name,
                )
            ],
        ),
        (
            os.path.join(
                "share",
                package_name,
            ),
            ["package.xml"],
        ),
        (
            os.path.join(
                "share",
                package_name,
                "launch",
            ),
            glob("launch/*.launch.py"),
        ),
        # 노드 설정은 ROS 파라미터 파일이 아니라 argparse CLI 인자로 준다.
        # (config/clip_params.yaml 은 제거됨)
        # 모델 바이너리(.pt/.onnx/.engine)는 수백 MB이고 매 빌드마다 복사되므로
        # install 트리에는 설명만 두고, 실제 경로는 ROS 파라미터로 넘긴다.
        (
            os.path.join(
                "share",
                package_name,
                "models",
            ),
            glob("models/*.md"),
        ),
    ],
    install_requires=[
        "setuptools",
    ],
    zip_safe=True,
    maintainer="sojin",
    maintainer_email="sojin@example.com",
    description=(
        "Meridian Perception Frontend: CLIP ViT-B/32 semantic embedding"
    ),
    license="Apache-2.0",
    tests_require=[
        "pytest",
    ],
    entry_points={
        "console_scripts": [
            (
                "clip_inference_node = "
                "meridian_clip.clip_inference_node:main"
            ),
            # 관측용 노드들. CLIP/torch 를 로드하지 않고 발행된 토픽만 읽으므로
            # 시스템 파이썬으로 돌아간다 (shebang 재패치 대상이 아니다).
            (
                "embedding_monitor = "
                "meridian_clip.embedding_monitor:main"
            ),
            (
                "clip_label_viz = "
                "meridian_clip.clip_label_viz:main"
            ),
        ],
    },
)
