"""Persistence for enrolled face embeddings.

Backed by SQLite: it ships with Python, gives real transactions and DELETE
semantics, and survives concurrent writers — none of which a pickle file does.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from backend.app.core.exceptions import IdentityExistsError, IdentityNotFoundError
from backend.app.services.encoder import EMBEDDING_DIM, Embedding

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS identities (
    name        TEXT PRIMARY KEY,
    embedding   BLOB NOT NULL,
    created_at  TEXT NOT NULL
);
"""


@dataclass(frozen=True, slots=True)
class Identity:
    """An enrolled person and their reference embedding."""

    name: str
    embedding: Embedding
    created_at: datetime


class FaceGallery:
    """Stores and retrieves enrolled identities.

    A connection is opened per operation rather than held open, so instances
    are safe to use from the thread pool that runs blocking routes.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        # Serialises writers in-process; SQLite handles cross-process locking.
        self._lock = threading.Lock()
        self._initialise()

    def _initialise(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        logger.info("Gallery ready at %s", self._db_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.row_factory = sqlite3.Row
            # Concurrent reads while a writer holds the lock.
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def add(self, name: str, embedding: Embedding) -> Identity:
        """Enrol ``name``.

        Raises:
            IdentityExistsError: ``name`` is already enrolled.
            ValueError: the embedding has the wrong shape.
        """
        _validate_embedding(embedding)
        created_at = datetime.now(UTC)

        with self._lock, self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO identities (name, embedding, created_at) VALUES (?, ?, ?)",
                    (name, _to_blob(embedding), created_at.isoformat()),
                )
            except sqlite3.IntegrityError as exc:
                raise IdentityExistsError(f"'{name}' is already enrolled.") from exc

        logger.info("Enrolled identity %r", name)
        return Identity(name=name, embedding=embedding, created_at=created_at)

    def get(self, name: str) -> Identity:
        """Return one identity.

        Raises:
            IdentityNotFoundError: ``name`` is not enrolled.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT name, embedding, created_at FROM identities WHERE name = ?",
                (name,),
            ).fetchone()

        if row is None:
            raise IdentityNotFoundError(f"'{name}' is not enrolled.")
        return _to_identity(row)

    def list_all(self) -> list[Identity]:
        """Return every enrolled identity, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, embedding, created_at FROM identities ORDER BY created_at, name"
            ).fetchall()
        return [_to_identity(row) for row in rows]

    def delete(self, name: str) -> None:
        """Remove an identity.

        Raises:
            IdentityNotFoundError: ``name`` is not enrolled.
        """
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM identities WHERE name = ?", (name,))
            if cursor.rowcount == 0:
                raise IdentityNotFoundError(f"'{name}' is not enrolled.")

        logger.info("Removed identity %r", name)

    def count(self) -> int:
        """Number of enrolled identities. Cheap enough for readiness probes."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM identities").fetchone()
        return int(row["n"])

    def as_matrix(self) -> tuple[list[str], np.ndarray]:
        """Return all names and their embeddings stacked into one array.

        Matching compares one probe against every enrolment, so handing the
        recogniser a single (N, 128) array lets numpy do that in one
        vectorised pass instead of a Python loop.

        The array is empty with shape (0, 128) when nothing is enrolled, so
        callers can rely on its dimensionality.
        """
        identities = self.list_all()
        if not identities:
            return [], np.empty((0, EMBEDDING_DIM), dtype=np.float64)

        names = [identity.name for identity in identities]
        matrix = np.stack([identity.embedding for identity in identities])
        return names, matrix


def _validate_embedding(embedding: Embedding) -> None:
    if embedding.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"Expected an embedding of shape ({EMBEDDING_DIM},), got {embedding.shape}."
        )


def _to_blob(embedding: Embedding) -> bytes:
    # float64 is fixed-width, so the blob round-trips without a dtype header.
    return np.ascontiguousarray(embedding, dtype=np.float64).tobytes()


def _from_blob(blob: bytes) -> Embedding:
    return np.frombuffer(blob, dtype=np.float64)


def _to_identity(row: sqlite3.Row) -> Identity:
    return Identity(
        name=row["name"],
        embedding=_from_blob(row["embedding"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )
