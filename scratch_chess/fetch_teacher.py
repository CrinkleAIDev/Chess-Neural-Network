"""Download official Stockfish for optional teacher/opponent use, never inference."""
import hashlib
import json
from pathlib import Path
import zipfile

import requests


def main():
    root = Path("tools/stockfish").resolve()
    root.mkdir(parents=True, exist_ok=True)
    response = requests.get("https://api.github.com/repos/official-stockfish/Stockfish/releases/latest", timeout=30)
    response.raise_for_status()
    release = response.json()
    candidates = [a for a in release["assets"] if a["name"] in (
        "stockfish-windows-x86-64-universal.zip", "stockfish-windows-x86-64-avx2.zip")]
    if not candidates:
        raise RuntimeError("Official asset naming changed; select a Windows x86-64 build at https://stockfishchess.org/download/")
    asset = candidates[0]
    archive = root / asset["name"]
    digest = hashlib.sha256()
    with requests.get(asset["browser_download_url"], stream=True, timeout=(30, 120)) as download:
        download.raise_for_status()
        with archive.open("wb") as f:
            for chunk in download.iter_content(1024*1024):
                f.write(chunk)
                digest.update(chunk)
    actual = "sha256:" + digest.hexdigest()
    if asset.get("digest") and asset["digest"] != actual:
        raise RuntimeError("Official artifact digest mismatch; refusing to extract")
    destination = root / release["tag_name"]
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise RuntimeError("Unsafe archive member path")
        z.extractall(destination)
    executables = list(destination.rglob("*.exe"))
    provenance = {"release": release["tag_name"], "url": asset["browser_download_url"], "sha256": actual,
        "publisher_digest_verified": bool(asset.get("digest")), "executables": [str(p) for p in executables],
        "role": "optional training label provider and test opponent only"}
    (root / "source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()

