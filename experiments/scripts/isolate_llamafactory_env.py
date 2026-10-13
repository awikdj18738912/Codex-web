"""Reuse the required base packages without exposing unrelated ASR packages.

Run with the project-local training venv after pip installation. All new files
are symlinks/metadata inside that venv; the base environment remains unchanged.
"""
from importlib import metadata
import json
from pathlib import Path
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    root = Path(__file__).resolve().parents[2]
    environment = Path(sys.prefix).resolve()
    if not environment.is_relative_to(root / ".runtime") or sys.prefix == sys.base_prefix:
        raise RuntimeError("Run only inside the project-local training venv")
    site = environment / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    base = Path(sys.base_prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    shared = environment / "shared_deps"
    if (environment / "shared_dependencies.json").exists() and "include-system-site-packages = false" in (environment / "pyvenv.cfg").read_text():
        saved = json.loads((environment / "shared_dependencies.json").read_text())
        for name, expected in saved["dependency_versions"].items():
            if metadata.version(name) != expected:
                raise RuntimeError("Shared dependency version changed: " + name)
        print("Existing isolated training environment verified")
        return
    selected, pending, seen = {}, [("llamafactory", frozenset())], set()
    while pending:
        name, extras = pending.pop()
        name = canonicalize_name(name)
        if (name, extras) in seen:
            continue
        seen.add((name, extras))
        distribution = metadata.distribution(name)
        selected[name] = distribution
        for spec in distribution.requires or []:
            requirement = Requirement(spec)
            if requirement.marker and not any(requirement.marker.evaluate({"extra": extra}) for extra in {"", *extras}):
                continue
            installed = metadata.distribution(requirement.name)
            if requirement.specifier and installed.version not in requirement.specifier:
                raise RuntimeError(f"{name}: {requirement} vs {installed.version}")
            pending.append((requirement.name, frozenset(requirement.extras)))
    shared.mkdir(exist_ok=True)
    linked = {}
    for name, distribution in selected.items():
        origin = Path(distribution.locate_file("")).resolve()
        if origin != base.resolve():
            continue
        for relative in distribution.files or []:
            first = Path(relative).parts[0]
            if first in {"..", "."}:
                continue
            source = base / first
            destination = shared / first
            if not source.exists():
                continue
            if destination.is_symlink():
                if destination.resolve() != source.resolve():
                    raise RuntimeError("Shared package origin changed: " + first)
            elif destination.exists():
                raise FileExistsError(destination)
            else:
                destination.symlink_to(source, target_is_directory=source.is_dir())
        linked[name] = distribution.version
    (site / "project_training_shared.pth").write_text(str(shared) + "\n")
    config = environment / "pyvenv.cfg"
    config.write_text(config.read_text().replace("include-system-site-packages = true", "include-system-site-packages = false"))
    record = dict(base_environment=str(Path(sys.base_prefix)), dependency_versions={n: d.version for n, d in selected.items()},
                  reused_packages=linked, unrelated_asr_packages_exposed=False)
    (environment / "shared_dependencies.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Isolated training dependency closure: {len(selected)} packages; {len(linked)} reused without copying")


if __name__ == "__main__":
    main()
