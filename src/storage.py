from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path


COLLECTIONS_DIR = Path("data/collections")


def save_uploaded_files(uploads: list[tuple[str, bytes]], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, content in uploads:
        safe_name = Path(name).name
        path = target_dir / safe_name
        path.write_bytes(content)
        manifest.append({"name": safe_name, "bytes": len(content)})
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def collection_id_for_uploads(uploads: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha1()
    for name, content in sorted(uploads, key=lambda item: item[0]):
        digest.update(name.encode("utf-8", errors="ignore"))
        digest.update(str(len(content)).encode("ascii"))
        digest.update(hashlib.sha1(content).digest())
    return digest.hexdigest()[:16]


def save_collection(uploads: list[tuple[str, bytes]], collection_name: str | None = None) -> Path:
    collection_id = collection_id_for_uploads(uploads)
    safe_name = _safe_name(collection_name or f"collection-{collection_id}")
    target_dir = COLLECTIONS_DIR / f"{safe_name}-{collection_id}"
    target_dir.mkdir(parents=True, exist_ok=True)
    save_uploaded_files(uploads, target_dir / "files")
    metadata = {
        "id": collection_id,
        "name": collection_name or safe_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": [{"name": Path(name).name, "bytes": len(content)} for name, content in uploads],
    }
    (target_dir / "collection.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target_dir


def list_collections() -> list[dict[str, str]]:
    if not COLLECTIONS_DIR.exists():
        return []
    collections = []
    for path in sorted(COLLECTIONS_DIR.iterdir(), reverse=True):
        metadata_path = path / "collection.json"
        if not metadata_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        collections.append(
            {
                "label": f"{metadata.get('name', path.name)} ({len(metadata.get('files', []))} files)",
                "path": str(path),
            }
        )
    return collections


def load_collection(path: str) -> list[tuple[str, bytes]]:
    files_dir = Path(path) / "files"
    uploads = []
    for file_path in files_dir.iterdir():
        if file_path.name == "manifest.json" or not file_path.is_file():
            continue
        uploads.append((file_path.name, file_path.read_bytes()))
    return uploads


def _safe_name(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in name.strip())
    return cleaned.strip("-")[:48] or "collection"
