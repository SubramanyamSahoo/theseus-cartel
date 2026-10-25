"""Back up a stopped campaign, including response caches and the project source."""

import argparse
import fcntl
import hashlib
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


def export(run, destination=None):
    root = Path(__file__).resolve().parents[1]
    run = Path(run).resolve()
    if not (run / "campaign.json").is_file():
        raise ValueError("Provide a campaign directory containing campaign.json")
    destination = Path(destination).resolve() if destination else root.parent / "theseus_backups"
    if destination == run or run in destination.parents:
        raise ValueError("The backup destination must be outside the campaign directory")
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{run.name}_{stamp}_{uuid.uuid4().hex[:8]}.tar.gz"
    final = destination / name
    partial = destination / (name + ".partial")
    with (run / ".campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Campaign is still active. Export after it stops so the caches are consistent."
            ) from exc
        # Only declared project paths are added; no model cache, credentials, or environment directory.
        keep = [
            "src",
            "tests",
            "configs",
            "docs",
            "scripts",
            "phases",
            "README.md",
            "PHASES.md",
            "LICENSE",
            "pyproject.toml",
            "uv.lock",
            "SOURCES.lock.json",
            "UPSTREAM.lock.json",
            "BOOTSTRAP.sh",
            "START_TMUX.sh",
            "START_SUITE_TMUX.sh",
            "START_CAMPAIGN_TMUX.sh",
            "START_FULL_TMUX.sh",
        ]

        def clean(info):
            return None if "__pycache__" in Path(info.name).parts else info

        with partial.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
            archive.add(run, arcname=str(Path(root.name) / "results/campaigns" / run.name), filter=clean)
            for name in keep:
                path = root / name
                if path.exists():
                    archive.add(path, arcname=str(Path(root.name) / name), filter=clean)
        partial.replace(final)
    digest = hashlib.sha256()
    with final.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    (destination / (final.name + ".sha256")).write_text(f"{digest.hexdigest()}  {final.name}\n")
    print(f"Backup: {final}\nSHA256: {digest.hexdigest()}")
    print("Includes response caches and source. Download this archive before deleting the VM.")
    print("For resume, restore the original absolute project/run paths; deadlines do not reset.")
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_directory", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    export(args.campaign_directory, args.destination)
