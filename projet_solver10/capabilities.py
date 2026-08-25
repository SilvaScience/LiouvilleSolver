"""Capability declarations for models and numerical backends.

The data structures in this module describe supported state types,
generators, numerical domains, exactness guarantees, and matrix-free
requirements. They let the solver select a compatible backend before any
expensive model construction begins.
"""

from dataclasses import dataclass, fields

from .exceptions import CapabilityError


@dataclass(frozen=True)
class Capabilities:
    """Numerical capabilities declared by a model or backend.

    Model-specific physical information belongs in ``metadata``.
    """

    pure_state: bool = True
    mixed_state: bool = False
    time_domain: bool = True
    frequency_domain: bool = False
    hilbert_resolvent: bool = False
    lindblad: bool = False
    finite_temperature: bool = False
    coupled_sectors: bool = True
    matrix_free: bool = True
    low_rank: bool = False
    metadata: tuple = ()

    @classmethod
    def from_value(cls, value):
        """Instance, mapping, or ``None`` -> normalized capabilities."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            known = {item.name for item in fields(cls)}
            unknown = sorted(set(value).difference(known))
            if unknown:
                raise ValueError(f"Unknown capabilities: {unknown}")
            data = dict(value)
            metadata = data.get("metadata", ())
            if isinstance(metadata, dict):
                metadata = tuple(sorted(metadata.items()))
            data["metadata"] = tuple(metadata)
            return cls(**data)
        raise TypeError("capabilities must be Capabilities, dict, or None.")

    def require(self, *names):
        """Capability names -> validation that all are supported."""
        missing = []
        for name in names:
            if not hasattr(self, name):
                raise ValueError(f"Unknown capability: {name!r}")
            if not getattr(self, name):
                missing.append(name)
        if missing:
            raise CapabilityError(
                "Unsupported capabilities: " + ", ".join(missing)
            )

    def as_dict(self):
        """Return a serializable representation."""
        result = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "metadata"
        }
        result["metadata"] = dict(self.metadata)
        return result


@dataclass(frozen=True)
class ModelRequirements:
    """Physical and numerical requirements declared by a model."""

    state_kind: str = "pure"
    generator_kind: str = "unitary"
    domains: tuple = ("time",)
    exactness: str = "exact"
    matrix_free_required: bool = False
    stationary_mode_deflation_required: bool = False
    metadata: tuple = ()

    def __post_init__(self):
        state_kind = str(self.state_kind).lower()
        generator_kind = str(self.generator_kind).lower()
        domains = tuple(str(item).lower() for item in self.domains)
        exactness = str(self.exactness).lower()
        if state_kind not in {"pure", "mixed"}:
            raise ValueError("state_kind must be 'pure' or 'mixed'.")
        if generator_kind not in {"unitary", "lindblad"}:
            raise ValueError(
                "generator_kind must be 'unitary' or 'lindblad'."
            )
        if not domains or not set(domains).issubset(
            {"time", "frequency"}
        ):
            raise ValueError(
                "domains must contain 'time' and/or 'frequency'."
            )
        if exactness not in {"exact", "approximate_allowed"}:
            raise ValueError(
                "exactness must be 'exact' or 'approximate_allowed'."
            )
        metadata = self.metadata
        if isinstance(metadata, dict):
            metadata = tuple(sorted(metadata.items()))
        object.__setattr__(self, "state_kind", state_kind)
        object.__setattr__(self, "generator_kind", generator_kind)
        object.__setattr__(self, "domains", domains)
        object.__setattr__(self, "exactness", exactness)
        object.__setattr__(self, "metadata", tuple(metadata))

    @classmethod
    def from_value(cls, value):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(
            "requirements must be ModelRequirements, dict, or None."
        )

    def as_dict(self):
        return {
            "state_kind": self.state_kind,
            "generator_kind": self.generator_kind,
            "domains": self.domains,
            "exactness": self.exactness,
            "matrix_free_required": self.matrix_free_required,
            "stationary_mode_deflation_required": (
                self.stationary_mode_deflation_required
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BackendCapabilities:
    """Effective backend capabilities, separate from model requirements."""

    state_kinds: tuple
    supported_pairs: tuple
    exactness: str
    matrix_free: bool
    low_rank: bool = False
    stationary_mode_deflation: bool = False

    def __post_init__(self):
        state_kinds = tuple(str(item).lower() for item in self.state_kinds)
        supported_pairs = tuple(
            (str(generator).lower(), str(domain).lower())
            for generator, domain in self.supported_pairs
        )
        exactness = str(self.exactness).lower()
        if not set(state_kinds).issubset({"pure", "mixed"}):
            raise ValueError("state_kinds contains an unknown value.")
        if exactness not in {"exact", "truncated_low_rank"}:
            raise ValueError("Unknown backend exactness.")
        object.__setattr__(self, "state_kinds", state_kinds)
        object.__setattr__(self, "supported_pairs", supported_pairs)
        object.__setattr__(self, "exactness", exactness)

    def missing_for(self, requirements):
        requirements = ModelRequirements.from_value(requirements)
        missing = []
        if requirements.state_kind not in self.state_kinds:
            missing.append(f"state_kind={requirements.state_kind}")
        for domain in requirements.domains:
            pair = (requirements.generator_kind, domain)
            if pair not in self.supported_pairs:
                missing.append(
                    f"generator/domain={requirements.generator_kind}/{domain}"
                )
        if (
            requirements.exactness == "exact"
            and self.exactness != "exact"
        ):
            missing.append("exactness=exact")
        if requirements.matrix_free_required and not self.matrix_free:
            missing.append("matrix_free")
        if (
            requirements.stationary_mode_deflation_required
            and not self.stationary_mode_deflation
        ):
            missing.append("stationary_mode_deflation")
        return tuple(missing)

    def supports(self, requirements):
        return not self.missing_for(requirements)

    def as_dict(self):
        return {
            "state_kinds": self.state_kinds,
            "supported_pairs": self.supported_pairs,
            "exactness": self.exactness,
            "matrix_free": self.matrix_free,
            "low_rank": self.low_rank,
            "stationary_mode_deflation": (
                self.stationary_mode_deflation
            ),
        }
