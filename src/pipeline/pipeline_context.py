from collections.abc import Iterator
from typing import Any


class PipelineContext:
    def __init__(self, initial_state: dict[str, Any] | None = None):
        self._data: dict[str, Any] = initial_state or {}
        self._aborted: bool = False

    # - - - #

    @property
    def items(self) -> dict[str, Any]:
        return self._data

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    # - - - #

    def abort(self) -> None:
        self._aborted = True

    @property
    def is_aborted(self) -> bool:
        return self._aborted

    # - - - #

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    # - - - #
    
    def __getitem__(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    # - - - #
    
    def __contains__(self, key: str) -> bool:
        return key in self._data

    def has(self, key: str) -> bool:
        return key in self._data

    # - - - #
        
    def keys(self) -> list[str]:
        return list(self._data.keys())

    # - - - #

    def __len__(self) -> int:
        return len(self._data)

    # - - - #

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    # - - - #

    def __repr__(self) -> str:
        return f"PipelineContext(keys={list(self._data.keys())}, aborted={self._aborted})"
