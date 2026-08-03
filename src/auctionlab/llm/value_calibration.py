"""Explicit, auditable calibration of LLM-inferred provisional valuations.

Frozen elicitation packs always store **raw** provisional values. Calibration
is deliberately a *replay-time* configuration so one frozen pack can be
evaluated under several calibrations without any further LLM call.

The transform is::

    calibrated = scale * raw * size_gamma ** max(0, bundle_size - size_threshold)
    calibrated = min(disclosed_budget, calibrated)      # when budget_cap

Three families select which parts apply:

``none``
    Identity. The raw value is returned byte-for-byte -- no scaling, no size
    adjustment, no budget cap, not even the non-negative clamp.
``uniform``
    ``scale`` only (plus the optional budget cap).
``exponential``
    ``scale`` and the bundle-size adjustment (plus the optional budget cap).

``scale`` may be **greater than one**: the current provisional-valuation
estimator generally *under*-estimates truth, so the required correction is
usually an inflation rather than a discount. This is the main reason the
legacy ``epsilon`` interface (restricted to ``(0, 1]`` and silently inert
unless ``discount_inferred`` was also set) is deprecated in favour of this
module.

Only initial LLM-inferred provisional valuations are calibrated. Exact
deterministic value-query answers and refinements are never touched -- see
:meth:`auctionlab.llm.proxies.LlmInferredXorProxy._revalue_and_upsert_atom`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping
import warnings


CALIBRATION_SCHEMA_VERSION = "1"

CALIBRATION_FAMILIES: tuple[str, ...] = ("none", "uniform", "exponential")

#: argparse dest-names of the deprecated calibration flags.
LEGACY_CALIBRATION_ARGS: tuple[str, ...] = (
    "discount_inferred",
    "epsilon",
    "size_discount_family",
    "size_discount_k0",
    "size_discount_gamma",
)

#: Legacy flags that carry a *value* (as opposed to the enabling gate).
LEGACY_CALIBRATION_VALUE_ARGS: tuple[str, ...] = (
    "epsilon",
    "size_discount_family",
    "size_discount_k0",
    "size_discount_gamma",
)


class CalibrationConfigError(ValueError):
    """Raised for an invalid, conflicting, or silently-inert calibration."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _positive_float(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationConfigError(
            f"{name} must be a number, got {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise CalibrationConfigError(f"{name} must be finite, got {value!r}")
    if number <= 0.0:
        raise CalibrationConfigError(f"{name} must be > 0, got {number!r}")
    return number


@dataclass(frozen=True)
class ValueCalibration:
    """One fully-resolved provisional-valuation calibration.

    Instances are immutable and validated on construction, so anything that
    holds one is holding a calibration that can actually be applied. The
    external JSON representation is :meth:`to_dict`.

    ``source_path`` and ``provenance`` are metadata: they are recorded in run
    artefacts but deliberately excluded from :meth:`config_hash`, so two runs
    that apply numerically identical calibrations hash identically regardless
    of where the config file happened to live.
    """

    family: str = "none"
    scale: float = 1.0
    size_gamma: float = 1.0
    size_threshold: int = 3
    budget_cap: bool = True
    provenance: Mapping[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    def __post_init__(self) -> None:
        if self.family not in CALIBRATION_FAMILIES:
            raise CalibrationConfigError(
                f"family must be one of {list(CALIBRATION_FAMILIES)}, "
                f"got {self.family!r}"
            )
        object.__setattr__(self, "scale", _positive_float("scale", self.scale))
        object.__setattr__(
            self, "size_gamma", _positive_float("size_gamma", self.size_gamma)
        )
        if isinstance(self.size_threshold, bool) or not isinstance(
            self.size_threshold, int
        ):
            raise CalibrationConfigError(
                "size_threshold must be an integer, got "
                f"{self.size_threshold!r}"
            )
        if self.size_threshold < 0:
            raise CalibrationConfigError(
                f"size_threshold must be >= 0, got {self.size_threshold}"
            )
        if not isinstance(self.budget_cap, bool):
            raise CalibrationConfigError(
                f"budget_cap must be a boolean, got {self.budget_cap!r}"
            )

        # Reject configurations whose parameters cannot possibly take effect.
        # Silently-inert settings are exactly the failure mode this module
        # exists to remove.
        if self.family == "none" and (
            self.scale != 1.0 or self.size_gamma != 1.0
        ):
            raise CalibrationConfigError(
                "family='none' ignores scale/size_gamma, but "
                f"scale={self.scale!r} size_gamma={self.size_gamma!r} were "
                "supplied. Use family='uniform' (scale only) or "
                "'exponential' (scale and size adjustment)."
            )
        if self.family == "uniform" and self.size_gamma != 1.0:
            raise CalibrationConfigError(
                "family='uniform' ignores size_gamma, but "
                f"size_gamma={self.size_gamma!r} was supplied. Use "
                "family='exponential' to apply a bundle-size adjustment."
            )
        object.__setattr__(self, "provenance", dict(self.provenance or {}))
        if self.source_path is not None:
            object.__setattr__(self, "source_path", str(self.source_path))

    # -- application ------------------------------------------------------

    @property
    def is_identity(self) -> bool:
        """True when this calibration provably cannot change any value."""
        return self.family == "none"

    def size_factor(self, bundle_size: int) -> float:
        if self.family != "exponential":
            return 1.0
        return self.size_gamma ** max(0, int(bundle_size) - self.size_threshold)

    def apply(
        self,
        raw_value: float,
        bundle_size: int,
        *,
        disclosed_budget: float | None = None,
    ) -> float:
        """Return the reported value for one raw provisional estimate.

        ``disclosed_budget`` must be a quantity the proxy legitimately
        observes (in practice the interest map's ``budget_hint``), never a
        hidden ground-truth value. It is ignored when ``budget_cap`` is off
        or when it is ``None``/non-positive.
        """
        if self.family == "none":
            return raw_value

        adjusted = float(raw_value) * self.scale * self.size_factor(bundle_size)
        adjusted = max(0.0, adjusted)
        if (
            self.budget_cap
            and disclosed_budget is not None
            and float(disclosed_budget) > 0.0
        ):
            adjusted = min(float(disclosed_budget), adjusted)
        return adjusted

    def invert(self, calibrated_value: float, bundle_size: int) -> float:
        """Best-effort recovery of the raw value behind a calibrated one.

        Used only to restore raw singleton *anchor* values before they are
        quoted back to the person simulator. The budget cap is a ``min`` and
        therefore not invertible; a value that was actually clipped by the cap
        cannot be recovered and is returned under-estimated. Anchors are
        singletons, which are essentially never budget-bound, so this is
        acceptable in practice.
        """
        if self.family == "none":
            return calibrated_value
        divisor = self.scale * self.size_factor(bundle_size)
        if divisor <= 0.0:
            return calibrated_value
        return calibrated_value / divisor

    # -- serialisation ----------------------------------------------------

    def effective_dict(self) -> dict[str, Any]:
        """Canonical numeric configuration, excluding all metadata."""
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "family": self.family,
            "scale": self.scale,
            "size_gamma": self.size_gamma,
            "size_threshold": self.size_threshold,
            "budget_cap": self.budget_cap,
        }

    def to_dict(self) -> dict[str, Any]:
        """Full external representation, including provenance."""
        return {**self.effective_dict(), "provenance": dict(self.provenance)}

    def config_hash(self) -> str:
        """Stable sha256 over the effective configuration (metadata-free)."""
        return hashlib.sha256(
            _canonical_json(self.effective_dict()).encode("utf-8")
        ).hexdigest()

    def short_hash(self) -> str:
        return self.config_hash()[:16]

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        source_path: str | None = None,
    ) -> "ValueCalibration":
        if not isinstance(data, Mapping):
            raise CalibrationConfigError(
                f"calibration config must be a JSON object, got {type(data).__name__}"
            )
        version = str(data.get("schema_version", CALIBRATION_SCHEMA_VERSION))
        if version != CALIBRATION_SCHEMA_VERSION:
            raise CalibrationConfigError(
                f"unsupported calibration schema_version {version!r}; "
                f"expected {CALIBRATION_SCHEMA_VERSION!r}"
            )
        unknown = set(data) - {
            "schema_version",
            "family",
            "scale",
            "size_gamma",
            "size_threshold",
            "budget_cap",
            "provenance",
        }
        if unknown:
            raise CalibrationConfigError(
                f"unknown calibration config keys: {sorted(unknown)}"
            )
        if "family" not in data:
            raise CalibrationConfigError("calibration config requires 'family'")
        threshold = data.get("size_threshold", 3)
        if isinstance(threshold, float) and threshold.is_integer():
            threshold = int(threshold)
        return cls(
            family=str(data["family"]),
            scale=data.get("scale", 1.0),
            size_gamma=data.get("size_gamma", 1.0),
            size_threshold=threshold,
            budget_cap=bool(data.get("budget_cap", True)),
            provenance=data.get("provenance") or {},
            source_path=source_path,
        )

    def with_provenance(self, **extra: Any) -> "ValueCalibration":
        return replace(self, provenance={**dict(self.provenance), **extra})

    # -- reporting --------------------------------------------------------

    def summary_fields(self, prefix: str = "pv_calibration") -> dict[str, Any]:
        """Flat key/value pairs for result CSV rows and run metadata."""
        return {
            f"{prefix}_family": self.family,
            f"{prefix}_scale": self.scale,
            f"{prefix}_size_gamma": self.size_gamma,
            f"{prefix}_size_threshold": self.size_threshold,
            f"{prefix}_budget_cap": self.budget_cap,
            f"{prefix}_config_path": self.source_path or "",
            f"{prefix}_config_hash": self.config_hash(),
        }

    def header_lines(self) -> list[str]:
        """Lines for the console run-configuration header."""
        lines = [
            f"  pv_calibration_family     {self.family}"
            + ("  (raw provisional values)" if self.is_identity else ""),
        ]
        if not self.is_identity:
            lines.append(f"  pv_calibration_scale      {self.scale:g}")
            if self.family == "exponential":
                lines.append(
                    f"  pv_calibration_size_gamma {self.size_gamma:g}"
                    f"  size_threshold={self.size_threshold}"
                )
            lines.append(
                f"  pv_calibration_budget_cap {'on' if self.budget_cap else 'off'}"
            )
        if self.source_path:
            lines.append(f"  pv_calibration_config     {self.source_path}")
        lines.append(f"  pv_calibration_hash       {self.short_hash()}")
        return lines


#: The default: report raw provisional values unchanged.
NO_CALIBRATION = ValueCalibration()


def load_calibration_config(path: str | Path) -> ValueCalibration:
    """Load and validate a ``--pv-calibration-config`` JSON file."""
    resolved = Path(path)
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CalibrationConfigError(
            f"cannot read calibration config {resolved}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CalibrationConfigError(
            f"calibration config {resolved} is not valid JSON: {exc}"
        ) from exc
    return ValueCalibration.from_dict(data, source_path=str(resolved))


def write_calibration_config(
    calibration: ValueCalibration,
    path: str | Path,
) -> Path:
    """Write a calibration directly consumable by ``--pv-calibration-config``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(calibration.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def legacy_calibration(
    *,
    discount_inferred: bool,
    epsilon: float = 1.0,
    size_discount_family: str | None = None,
    size_discount_k0: int = 3,
    size_discount_gamma: float = 1.0,
) -> ValueCalibration:
    """Translate the deprecated epsilon interface into a :class:`ValueCalibration`.

    Reproduces the historical transform exactly, including the fact that
    ``discount_inferred=False`` makes every other parameter inert. Used
    internally by :class:`~auctionlab.llm.proxies.LlmInferredXorProxy` so that
    old and new call sites share one implementation; the CLI refuses the inert
    combination outright rather than translating it.
    """
    if size_discount_family not in (None, "exponential"):
        raise CalibrationConfigError(
            "size_discount_family must be None or 'exponential', got "
            f"{size_discount_family!r}"
        )
    if not discount_inferred:
        return NO_CALIBRATION
    if size_discount_family == "exponential":
        return ValueCalibration(
            family="exponential",
            scale=epsilon,
            size_gamma=size_discount_gamma,
            size_threshold=size_discount_k0,
            budget_cap=False,
        )
    if epsilon == 1.0:
        # Numerically the identity; keep it a real 'uniform' so that a
        # deliberate --epsilon 1 --discount-inferred run still reports as
        # calibrated rather than masquerading as an uncalibrated baseline.
        return ValueCalibration(
            family="uniform", scale=1.0, budget_cap=False
        )
    return ValueCalibration(family="uniform", scale=epsilon, budget_cap=False)


def resolve_cli_calibration(
    args: Any,
    explicitly_set: Iterable[str],
    *,
    warn: Any = None,
) -> ValueCalibration:
    """Resolve the effective calibration for one CLI invocation.

    ``explicitly_set`` is the set of argparse dest-names actually present on
    the command line (see
    :func:`auctionlab.experiments.run_config.explicitly_set_args`) -- it is
    what distinguishes "the user asked for the legacy behaviour" from "these
    are just defaults".

    Raises :exc:`CalibrationConfigError` when a config file is combined with
    any legacy flag, and when legacy calibration *values* are supplied without
    ``--discount-inferred`` (which used to be a silent no-op).
    """
    explicit = set(explicitly_set)
    emit = warn if warn is not None else _default_warn

    config_path = getattr(args, "pv_calibration_config", None)
    legacy_used = sorted(explicit & set(LEGACY_CALIBRATION_ARGS))

    if config_path is not None and legacy_used:
        raise CalibrationConfigError(
            "--pv-calibration-config cannot be combined with the deprecated "
            "calibration flags "
            + ", ".join(f"--{name.replace('_', '-')}" for name in legacy_used)
            + ". Put every calibration parameter in the config file."
        )

    if config_path is not None:
        return load_calibration_config(config_path)

    if not legacy_used:
        return NO_CALIBRATION

    discount_inferred = bool(getattr(args, "discount_inferred", False))
    epsilon = getattr(args, "epsilon", 1.0)
    family = getattr(args, "size_discount_family", None)
    k0 = getattr(args, "size_discount_k0", 3)
    gamma = getattr(args, "size_discount_gamma", 1.0)

    emit(
        "The --epsilon/--discount-inferred/--size-discount-* flags are "
        "deprecated and will be removed. Use --pv-calibration-config PATH "
        "with an explicit calibration JSON instead (see README.md, "
        "'Optional calibration')."
    )

    # Validate every legacy parameter, whether or not it is reachable, so a
    # typo is reported instead of being quietly dropped.
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise CalibrationConfigError(
            f"--epsilon must be a number, got {epsilon!r}"
        ) from exc
    if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        raise CalibrationConfigError(
            f"--epsilon must be in (0, 1], got {epsilon!r}. The deprecated "
            "epsilon flag can only shrink values; use "
            "--pv-calibration-config with \"family\": \"uniform\" and a "
            "\"scale\" above 1 to inflate them."
        )
    if family not in (None, "exponential"):
        raise CalibrationConfigError(
            f"--size-discount-family must be 'exponential', got {family!r}"
        )
    gamma = _positive_float("--size-discount-gamma", gamma)
    if isinstance(k0, bool) or not isinstance(k0, int) or k0 < 0:
        raise CalibrationConfigError(
            f"--size-discount-k0 must be a non-negative integer, got {k0!r}"
        )

    value_flags = sorted(explicit & set(LEGACY_CALIBRATION_VALUE_ARGS))
    if value_flags and not discount_inferred:
        raise CalibrationConfigError(
            "legacy calibration values were supplied ("
            + ", ".join(f"--{name.replace('_', '-')}" for name in value_flags)
            + ") without --discount-inferred, which would silently do "
            "nothing. Add --discount-inferred, or (preferred) use "
            "--pv-calibration-config."
        )
    if family is None and "size_discount_gamma" in explicit and gamma != 1.0:
        raise CalibrationConfigError(
            "--size-discount-gamma has no effect without "
            "--size-discount-family exponential."
        )
    if family is None and "size_discount_k0" in explicit:
        raise CalibrationConfigError(
            "--size-discount-k0 has no effect without "
            "--size-discount-family exponential."
        )

    calibration = legacy_calibration(
        discount_inferred=discount_inferred,
        epsilon=epsilon,
        size_discount_family=family,
        size_discount_k0=k0,
        size_discount_gamma=gamma,
    )
    return calibration.with_provenance(
        source="deprecated_cli_flags",
        flags={
            "discount_inferred": discount_inferred,
            "epsilon": epsilon,
            "size_discount_family": family,
            "size_discount_k0": k0,
            "size_discount_gamma": gamma,
        },
    )


def _default_warn(message: str) -> None:
    warnings.warn(message, DeprecationWarning, stacklevel=3)
