from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys


def _entry_script(package_dir, entry_script=None):
    if entry_script is not None:
        p = Path(entry_script)
        if not p.is_absolute():
            p = package_dir / p
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"entry script not found: {p}")
        return p

    candidates = sorted(package_dir.glob("*.py"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(f"no *.py found in: {package_dir}")

    names = "\n".join("  " + p.name for p in candidates)
    raise RuntimeError(
        "multiple *.py files found. Specify entry_script.\n" + names
    )


def _pyinstaller_version():
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--version"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "PyInstaller is not available in this Python environment."
        )
    return result.stdout.strip()


def _resources(package_dir, resources):
    normalized = []
    for item in resources:
        if isinstance(item, tuple):
            src, dst = item
        else:
            src, dst = item, item

        src = Path(src)
        if not src.is_absolute():
            src = package_dir / src
        src = src.resolve()

        dst = Path(dst)
        if dst.is_absolute():
            raise ValueError(f"resource destination must be relative: {dst}")
        if not src.exists():
            raise FileNotFoundError(f"resource not found: {src}")

        normalized.append((src, dst))
    return normalized


def build_exe(
    package_dir,
    exe_name,
    *,
    entry_script=None,
    resources=(),
    exclude_modules=(),
    required_pyinstaller_version=None,
    output_root=None,
    onefile=True,
    console=True,
    clean=True,
    noconfirm=True,
    extra_args=(),
):
    """Build an EXE from a prepared Python application directory.

    package_dir:
        Runnable application directory.
    exe_name:
        Output application name without ".exe".
    entry_script:
        Entry-point .py. If omitted, exactly one top-level .py must exist.
    resources:
        Relative resource paths, or (source, destination) pairs.
    exclude_modules:
        Modules passed to PyInstaller --exclude-module.
    required_pyinstaller_version:
        Exact version required, or None for no version check.
    output_root:
        Parent of build/, dist/, spec/. Default: package_dir.parent.
    """
    package_dir = Path(package_dir).resolve()
    if not package_dir.is_dir():
        raise FileNotFoundError(f"package directory not found: {package_dir}")

    entry = _entry_script(package_dir, entry_script)
    resource_specs = _resources(package_dir, resources)

    version = _pyinstaller_version()
    if (
        required_pyinstaller_version is not None
        and version != required_pyinstaller_version
    ):
        raise RuntimeError(
            "PyInstaller version mismatch.\n"
            f"  required = {required_pyinstaller_version}\n"
            f"  current  = {version}"
        )

    output_root = (
        package_dir.parent
        if output_root is None
        else Path(output_root).resolve()
    )
    build_dir = output_root / "build"
    dist_dir = output_root / "dist"
    spec_dir = output_root / "spec"
    build_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    command = [sys.executable, "-m", "PyInstaller"]
    if clean:
        command.append("--clean")
    if noconfirm:
        command.append("--noconfirm")

    command.append("--onefile" if onefile else "--onedir")
    command.append("--console" if console else "--windowed")

    command += [
        "--name", exe_name,
        "--distpath", str(dist_dir),
        "--workpath", str(build_dir),
        "--specpath", str(spec_dir),
    ]

    for module in exclude_modules:
        command += ["--exclude-module", module]

    for src, dst in resource_specs:
        command += ["--add-data", f"{src}:{dst}"]

    command += list(extra_args)
    command.append(str(entry))

    print("=== pyaino exe_builder ===")
    print("package     =", package_dir)
    print("entry       =", entry)
    print("PyInstaller =", version)
    print("output root =", output_root)
    if resource_specs:
        print("resources:")
        for src, dst in resource_specs:
            print(f"  {src} -> {dst}")
    if exclude_modules:
        print("exclude modules =", ", ".join(exclude_modules))
    print()

    subprocess.run(command, cwd=package_dir, check=True)

    result = dist_dir / (f"{exe_name}.exe" if onefile else exe_name)
    if not result.exists():
        raise RuntimeError(f"expected output not found: {result}")

    print()
    print("build completed =>", result)
    return result


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--entry", default=None)
    parser.add_argument("--resource", action="append", default=[])
    parser.add_argument("--exclude-module", action="append", default=[])
    parser.add_argument("--require-pyinstaller", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--onedir", action="store_true")
    parser.add_argument("--windowed", action="store_true")
    args = parser.parse_args()

    build_exe(
        args.package_dir,
        args.name,
        entry_script=args.entry,
        resources=args.resource,
        exclude_modules=args.exclude_module,
        required_pyinstaller_version=args.require_pyinstaller,
        output_root=args.output_root,
        onefile=not args.onedir,
        console=not args.windowed,
    )


if __name__ == "__main__":
    _main()
