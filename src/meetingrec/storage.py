"""
meetingrec.storage — where finished recordings go.

Hard rule enforced here: capture writes to a LOCAL staging directory, always.
Nothing streams straight into Drive or a network share. A sync client that
decides to lock, dedupe, or re-materialize a file mid-write will stall the
writer thread, and a stalled writer thread during a call means lost audio.
Archiving happens after the session ends, and it is allowed to fail without
taking the recording with it.

Three backends, one interface:

  local   — move into a directory. Simplest, no dependencies.
  drive   — move into Google Drive for Desktop's virtual drive. Still just a
            filesystem move; Google's client does the upload. No auth code.
  driveapi— real Drive API v3 resumable upload. Needs OAuth, gives you file
            IDs and metadata. Use when a machine has no Drive client.

Pick with `backend = "drive"` in config.toml.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import string
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

log = logging.getLogger(__name__)

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]
MIME_BY_SUFFIX = {
    ".wav": "audio/wav",
    ".opus": "audio/opus",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".json": "application/json",
    ".txt": "text/plain",
}


def guess_mime(path: Path) -> str:
    return MIME_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream")


def safe_name(text: str, maxlen: int = 80) -> str:
    """
    Make a string safe for a filename on Windows and Drive alike.

    Meeting titles arrive full of colons and slashes ("Q3 Review: Ops/L&D"), so
    separators collapse together with any whitespace around them rather than
    leaving "Review- Ops" behind.
    """
    cleaned = re.sub(r"[^A-Za-z0-9 ._()-]+", "-", text)
    cleaned = re.sub(r"\s*-+\s*", "-", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .-")
    return (cleaned[:maxlen].strip(" .-") or "untitled")


@dataclass
class ArchiveResult:
    ok: bool
    location: str
    backend: str
    error: Optional[str] = None


@runtime_checkable
class StorageBackend(Protocol):
    name: str

    def archive(self, path: Path, subdir: str, metadata: dict[str, Any]) -> ArchiveResult: ...


# --------------------------------------------------------------------------
# local / Drive for Desktop
# --------------------------------------------------------------------------


class LocalArchive:
    """Move finished files into a directory tree. Works for any mounted path."""

    name = "local"

    def __init__(self, root: Path, write_sidecar: bool = True):
        self.root = Path(root).expanduser()
        self.write_sidecar = write_sidecar

    def _target_dir(self, subdir: str) -> Path:
        d = self.root / subdir if subdir else self.root
        d.mkdir(parents=True, exist_ok=True)
        return d

    def archive(self, path: Path, subdir: str, metadata: dict[str, Any]) -> ArchiveResult:
        path = Path(path)
        if not path.exists():
            return ArchiveResult(False, "", self.name, f"source missing: {path}")
        try:
            dest_dir = self._target_dir(subdir)
            dest = self._unique(dest_dir / path.name)
            shutil.move(str(path), str(dest))
            if self.write_sidecar and metadata:
                dest.with_suffix(dest.suffix + ".json").write_text(
                    json.dumps(metadata, indent=2, default=str), encoding="utf-8"
                )
            log.info("archived -> %s", dest)
            return ArchiveResult(True, str(dest), self.name)
        except Exception as exc:
            log.exception("archive failed")
            return ArchiveResult(False, "", self.name, str(exc))

    @staticmethod
    def _unique(dest: Path) -> Path:
        if not dest.exists():
            return dest
        stem, suffix = dest.stem, dest.suffix
        for n in range(2, 1000):
            cand = dest.with_name(f"{stem}-{n}{suffix}")
            if not cand.exists():
                return cand
        return dest.with_name(f"{stem}-{int(time.time())}{suffix}")


def find_drive_desktop_root() -> Optional[Path]:
    """
    Locate Google Drive for Desktop's virtual drive.

    Checks every drive letter for a 'My Drive' folder, then falls back to the
    older %USERPROFILE%\\Google Drive layout. Returns None if not installed.
    """
    for letter in reversed(string.ascii_uppercase):
        root = Path(f"{letter}:/")
        try:
            if (root / "My Drive").is_dir():
                return root / "My Drive"
            if (root / "Shared drives").is_dir():
                return root
        except OSError:
            continue
    legacy = Path.home() / "Google Drive"
    return legacy if legacy.is_dir() else None


class DriveDesktopArchive(LocalArchive):
    """
    Google Drive for Desktop. Identical mechanics to LocalArchive — it IS the
    filesystem — but resolves the drive letter itself and refuses to run if
    the client is not mounted, rather than silently creating G:\\ as a folder.
    """

    name = "drive"

    def __init__(self, folder: str = "Meeting Recordings", root: Optional[Path] = None):
        resolved = Path(root).expanduser() if root else find_drive_desktop_root()
        if resolved is None:
            raise RuntimeError(
                "Google Drive for Desktop not found. Install it, or set "
                "backend = 'driveapi' / 'local' in config.toml."
            )
        super().__init__(resolved / folder)
        log.info("Drive for Desktop archive root: %s", self.root)

    def archive(self, path: Path, subdir: str, metadata: dict[str, Any]) -> ArchiveResult:
        # If the mount vanished (client quit, laptop offline), fail loudly
        # instead of writing into a directory that is no longer synced.
        if not self.root.drive or not Path(self.root.drive + "/").exists():
            return ArchiveResult(False, "", self.name, "Drive mount unavailable")
        return super().archive(path, subdir, metadata)


# --------------------------------------------------------------------------
# Drive API v3
# --------------------------------------------------------------------------


class DriveApiArchive:
    """
    Resumable upload via Drive API v3.

    Scope is drive.file — the app can only touch files it created itself. That
    keeps the consent screen tame and means this cannot wander through the rest
    of your Drive. The trade-off: it cannot upload into a pre-existing folder
    it did not make, so it creates and reuses its own root folder.

    Requires:  py -m pip install google-api-python-client google-auth-oauthlib
    """

    name = "driveapi"
    CHUNK = 8 * 1024 * 1024

    def __init__(
        self,
        credentials_file: Path,
        token_file: Path,
        folder_name: str = "Meeting Recordings",
        shared_drive_id: Optional[str] = None,
    ):
        self.credentials_file = Path(credentials_file).expanduser()
        self.token_file = Path(token_file).expanduser()
        self.folder_name = folder_name
        self.shared_drive_id = shared_drive_id
        self._service = None
        self._folder_cache: dict[str, str] = {}

    # -- auth ------------------------------------------------------------

    def _build(self):
        if self._service is not None:
            return self._service
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        creds = None
        if self.token_file.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_file), DRIVE_SCOPES)
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as exc:
                log.warning("token refresh failed (%s) — re-authorizing", exc)
                creds = None
        if not creds or not creds.valid:
            if not self.credentials_file.exists():
                raise RuntimeError(f"missing OAuth client secrets: {self.credentials_file}")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.credentials_file), DRIVE_SCOPES
            )
            creds = flow.run_local_server(port=0)
            self.token_file.parent.mkdir(parents=True, exist_ok=True)
            self.token_file.write_text(creds.to_json(), encoding="utf-8")
            os.chmod(self.token_file, 0o600)
        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return self._service

    # -- folders ---------------------------------------------------------

    def _shared_kwargs(self) -> dict[str, Any]:
        return {"supportsAllDrives": True} if self.shared_drive_id else {}

    def _folder_id(self, name: str, parent: Optional[str] = None) -> str:
        cache_key = f"{parent or 'root'}/{name}"
        if cache_key in self._folder_cache:
            return self._folder_cache[cache_key]

        svc = self._build()
        escaped = name.replace("'", "\\'")
        q = (
            f"name = '{escaped}' and mimeType = 'application/vnd.google-apps.folder' "
            f"and trashed = false"
        )
        if parent:
            q += f" and '{parent}' in parents"

        list_kwargs: dict[str, Any] = {"q": q, "fields": "files(id,name)", "pageSize": 10}
        if self.shared_drive_id:
            list_kwargs.update(
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                corpora="drive",
                driveId=self.shared_drive_id,
            )
        found = svc.files().list(**list_kwargs).execute().get("files", [])
        if found:
            self._folder_cache[cache_key] = found[0]["id"]
            return found[0]["id"]

        body: dict[str, Any] = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent:
            body["parents"] = [parent]
        elif self.shared_drive_id:
            body["parents"] = [self.shared_drive_id]
        created = svc.files().create(body=body, fields="id", **self._shared_kwargs()).execute()
        self._folder_cache[cache_key] = created["id"]
        log.info("created Drive folder %s (%s)", name, created["id"])
        return created["id"]

    def _resolve_path(self, subdir: str) -> str:
        parent = self._folder_id(self.folder_name)
        for part in [p for p in subdir.replace("\\", "/").split("/") if p]:
            parent = self._folder_id(part, parent)
        return parent

    # -- upload ----------------------------------------------------------

    def archive(self, path: Path, subdir: str, metadata: dict[str, Any]) -> ArchiveResult:
        from googleapiclient.http import MediaFileUpload

        path = Path(path)
        if not path.exists():
            return ArchiveResult(False, "", self.name, f"source missing: {path}")
        try:
            svc = self._build()
            parent = self._resolve_path(subdir)
            body: dict[str, Any] = {"name": path.name, "parents": [parent]}
            if metadata:
                # Drive caps description at 4096 chars.
                body["description"] = json.dumps(metadata, default=str)[:4000]

            media = MediaFileUpload(
                str(path), mimetype=guess_mime(path), resumable=True, chunksize=self.CHUNK
            )
            request = svc.files().create(
                body=body, media_body=media, fields="id,webViewLink", **self._shared_kwargs()
            )

            response = None
            while response is None:
                status, response = request.next_chunk(num_retries=5)
                if status:
                    log.info("uploading %s — %d%%", path.name, int(status.progress() * 100))

            link = response.get("webViewLink") or response["id"]
            log.info("uploaded %s -> %s", path.name, link)
            return ArchiveResult(True, link, self.name)
        except Exception as exc:
            log.exception("Drive upload failed")
            return ArchiveResult(False, "", self.name, str(exc))


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------


class DeferredDriveArchive(StorageBackend):
    """
    Drive for Desktop, resolved when it is actually needed.

    The recorder starts shortly after logon, and Drive for Desktop often has
    not finished mounting by then. Resolving the backend once at startup
    therefore found nothing, fell back to local, and stayed that way for the
    whole session — so every meeting that day was archived to the staging
    folder and silently never reached Drive.

    Nothing about that is an error worth giving up on: the mount appears a
    minute or two later and works fine. So resolution is retried on every
    archive call until it succeeds, and only then cached. If Drive never
    appears, files stay in staging and are picked up by a later run.
    """

    name = "drive"

    def __init__(self, folder: str = "Meeting Recordings",
                 root: Optional[Path] = None):
        self._folder = folder
        self._root = root
        self._real: Optional[DriveDesktopArchive] = None
        self._complained = False

    def _resolve(self) -> Optional[DriveDesktopArchive]:
        if self._real is not None:
            return self._real
        try:
            self._real = DriveDesktopArchive(folder=self._folder, root=self._root)
            if self._complained:
                log.info("Drive for Desktop is available now — archiving resumed")
            return self._real
        except Exception as exc:
            if not self._complained:
                # Once, not on every file: a 40-screenshot meeting would
                # otherwise write 40 identical errors.
                log.warning("Drive for Desktop not mounted yet (%s) — "
                            "keeping files in staging and retrying", exc)
                self._complained = True
            return None

    def archive(self, path: Path, subdir: str,
                metadata: dict[str, Any]) -> ArchiveResult:
        backend = self._resolve()
        if backend is None:
            return ArchiveResult(False, "", self.name,
                                 "Drive not mounted — file kept in staging")
        return backend.archive(path, subdir, metadata)

    def available(self) -> bool:
        return self._resolve() is not None


def build_backend(cfg) -> StorageBackend:
    """Construct the backend named in config, falling back to local on error."""
    choice = (getattr(cfg, "backend", None) or "local").lower()

    if choice == "drive":
        # Deferred rather than resolved here: at logon Drive is usually still
        # mounting, and a permanent fallback to local loses a whole day of
        # archiving to a few seconds of timing.
        backend = DeferredDriveArchive(
            folder=getattr(cfg, "drive_folder", "Meeting Recordings"),
            root=getattr(cfg, "drive_root", None),
        )
        if backend.available():
            log.info("Drive for Desktop archive ready")
        else:
            log.info("Drive for Desktop not mounted yet — will retry as "
                     "meetings finish")
        return backend

    if choice == "driveapi":
        return DriveApiArchive(
            credentials_file=getattr(cfg, "drive_credentials", "client_secret.json"),
            token_file=getattr(cfg, "drive_token", "token.json"),
            folder_name=getattr(cfg, "drive_folder", "Meeting Recordings"),
            shared_drive_id=getattr(cfg, "drive_shared_drive_id", None),
        )

    root = getattr(cfg, "archive_dir", None) or getattr(cfg, "staging_dir", Path("recordings"))
    return LocalArchive(Path(root))


def session_subdir(started_at: datetime, machine: str, group_by: str = "month",
                   slug: Optional[str] = None) -> str:
    """
    Folder layout inside the archive.

    Machine-first keeps several machines from colliding and makes it obvious
    where a recording came from. Passing a slug adds a folder per meeting,
    which matters once screenshots are on: a month of flat files is three
    audio tracks and a dozen images per call all jumbled together, and sorting
    by name interleaves meetings rather than separating them.
    """
    if group_by == "day":
        stamp = started_at.strftime("%Y/%m/%d")
    elif group_by == "flat":
        stamp = ""
    else:
        stamp = started_at.strftime("%Y/%m")

    parts = [safe_name(machine), stamp]
    if slug:
        parts.append(safe_name(slug, 60))
    return "/".join(p for p in parts if p)
