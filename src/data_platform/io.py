"""Crash-safe filesystem primitives shared by data-platform stores."""

from __future__ import annotations

import fcntl
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4


class ImmutableWriteConflict(RuntimeError):
    """An immutable address already exists with incompatible content."""


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def write_new_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_create(path: Path, data: bytes) -> None:
    """Create immutable bytes atomically and never replace an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid4().hex}"
    try:
        write_new_file(temporary, data)
        os.link(temporary, path)
        temporary.unlink()
        fsync_directory(path.parent)
    except FileExistsError as exc:
        temporary.unlink(missing_ok=True)
        raise ImmutableWriteConflict(f"immutable path already exists: {path}") from exc
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_replace(path: Path, data: bytes) -> None:
    """Atomically replace small mutable control state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid4().hex}"
    try:
        write_new_file(temporary, data)
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def publish_directory(temporary: Path, destination: Path) -> None:
    """Publish a fully fsynced immutable directory with same-filesystem rename."""
    try:
        for directory, _, _ in os.walk(temporary, topdown=False):
            fsync_directory(Path(directory))
        os.rename(temporary, destination)
        fsync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
