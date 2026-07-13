"""Shared strict-schema and migration rules for evidence contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import json
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

CURRENT_SCHEMA_VERSION = "1.0.0"
Migration = Callable[[dict[str, Any]], dict[str, Any]]


class FrozenDict(dict[Any, Any]):
    """JSON-compatible mapping that rejects every ordinary mutation path."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("contract mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[Any, Any]:
        return {deepcopy(key, memo): deepcopy(value, memo) for key, value in self.items()}


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, StrictContract):
        return value
    if isinstance(value, Mapping):
        return FrozenDict({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


class StrictContract(BaseModel):
    """Immutable contract base with a fail-closed unknown-field policy."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )

    schema_version: Literal["1.0.0"] = CURRENT_SCHEMA_VERSION

    _migrations: ClassVar[dict[str, tuple[str, Migration]]] = {}

    def model_post_init(self, __context: Any) -> None:
        del __context
        for name in type(self).model_fields:
            object.__setattr__(self, name, _deep_freeze(getattr(self, name)))

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> "StrictContract":
        copied = super().model_copy(update=update, deep=deep)
        for name in type(copied).model_fields:
            object.__setattr__(copied, name, _deep_freeze(getattr(copied, name)))
        return copied

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("warnings", False)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("warnings", False)
        return super().model_dump_json(*args, **kwargs)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._migrations = {}

    @classmethod
    def register_migration(cls, source: str, target: str, migration: Migration) -> None:
        """Register one explicit, forward-only schema migration step."""
        if source == target:
            raise ValueError("schema migration must change the version")
        if source in cls._migrations:
            raise ValueError(f"migration already registered for {source!r}")
        cls._migrations[source] = (target, migration)

    @classmethod
    def migrate_payload(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Migrate a raw payload to the current schema or fail closed."""
        value = deepcopy(dict(payload))
        version = value.get("schema_version")
        if not isinstance(version, str):
            raise ValueError("schema_version is required before validation")

        visited: set[str] = set()
        while version != CURRENT_SCHEMA_VERSION:
            if version in visited:
                raise ValueError(f"schema migration cycle detected at {version!r}")
            visited.add(version)
            step = cls._migrations.get(version)
            if step is None:
                raise ValueError(
                    f"unsupported {cls.__name__} schema_version {version!r}; "
                    f"current={CURRENT_SCHEMA_VERSION!r}"
                )
            target, migration = step
            value = migration(deepcopy(value))
            if not isinstance(value, dict):
                raise TypeError("schema migration must return a dictionary")
            value["schema_version"] = target
            version = target
        return value

    @classmethod
    def from_versioned_payload(cls, payload: Mapping[str, Any]) -> "StrictContract":
        migrated = cls.migrate_payload(payload)
        return cls.model_validate_json(json.dumps(migrated, allow_nan=False), strict=True)
