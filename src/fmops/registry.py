from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar


T = TypeVar("T")


class RegistryError(KeyError):
    """Raised when a registry lookup or registration fails."""


@dataclass(frozen=True)
class RegistryEntry(Generic[T]):
    name: str
    factory: Callable[..., T]
    description: str = ""
    metadata: dict[str, Any] | None = None


class Registry(Generic[T]):
    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self._entries: dict[str, RegistryEntry[T]] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., T],
        description: str = "",
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        if not overwrite and name in self._entries:
            raise RegistryError(f"{self.namespace}: entry already registered: {name}")
        self._entries[name] = RegistryEntry(name, factory, description, metadata or {})

    def get(self, name: str) -> RegistryEntry[T]:
        try:
            return self._entries[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._entries)) or "<empty>"
            raise RegistryError(f"{self.namespace}: unknown entry '{name}', available: {available}") from exc

    def build(self, name: str, *args: Any, **kwargs: Any) -> T:
        return self.get(name).factory(*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": entry.name,
                "description": entry.description,
                "metadata": entry.metadata or {},
            }
            for entry in sorted(self._entries.values(), key=lambda item: item.name)
        ]


MODEL_REGISTRY: Registry[Any] = Registry("models")
DATASET_REGISTRY: Registry[Any] = Registry("datasets")
TRAINER_REGISTRY: Registry[Any] = Registry("trainers")
EVALUATOR_REGISTRY: Registry[Any] = Registry("evaluators")


def _build_architecture(family: str, config: Any, **kwargs: Any) -> Any:
    from .architecture_impl import build_reference_implementation

    return build_reference_implementation(family, config, **kwargs)


def register_builtin_models() -> None:
    from .architecture_impl import REFERENCE_IMPLEMENTATIONS

    for family in REFERENCE_IMPLEMENTATIONS:
        MODEL_REGISTRY.register(
            family,
            lambda config, _family=family, **kwargs: _build_architecture(_family, config, **kwargs),
            description=f"Reference implementation for {family}",
            metadata={"family": family, "backend": "torch"},
            overwrite=True,
        )


register_builtin_models()
