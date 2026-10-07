"""Declarative, hashable specs for datasets and networks.

A spec is the *complete* description of an artifact: the same spec always
produces the same artifact, so its hash is the artifact's identity (cache
key, directory name, lineage link). Bump SCHEMA_VERSION when a field is
added/changed in a way that alters the produced artifact; old hashes then
stop matching instead of silently aliasing new behaviour.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1
MAX_RADICAL_SPACE = 10_000_000


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_payload(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()[:16]


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def hash(self) -> str:
        return hash_payload(self.model_dump(mode="json"))


class DatasetSpec(_Spec):
    schema_version: int = SCHEMA_VERSION

    # radical space
    D: int = Field(ge=1)
    alphabet: tuple[float, ...] = (-1.0, 0.0, 1.0)
    encoding: Literal["raw", "one_hot"] = "raw"

    # lexicon
    lexicon_size: int = Field(ge=2)
    p_irregular_types: float = Field(gt=0.0, lt=1.0)
    n_wug: int = Field(default=30, ge=1)

    # frequency
    p_token_irregular: float = Field(ge=0.0, le=1.0)
    freq_distribution: Literal["uniform", "zipf"] = "uniform"
    zipf_exponent: float = Field(default=1.0, gt=0.0)

    # classes / structure
    class_scheme: Literal["unique_per_irregular", "shared_k", "by_cluster"] = "unique_per_irregular"
    k_classes: int | None = Field(default=None, ge=1)
    irregular_placement: Literal["random", "clustered"] = "random"
    n_clusters: int = Field(default=1, ge=1)

    # wugs
    wug_strategy: Literal["random", "near_irregular", "far_from_lexicon"] = "random"

    # sampling
    sampling: Literal["stream", "finite"] = "stream"
    train_size: int | None = Field(default=None, ge=1)

    seed: int = 0

    @model_validator(mode="after")
    def _check(self) -> DatasetSpec:
        if len(set(self.alphabet)) != len(self.alphabet) or len(self.alphabet) < 2:
            raise ValueError("alphabet must contain at least 2 distinct values")
        if self.space_size > MAX_RADICAL_SPACE:
            raise ValueError(
                f"radical space {len(self.alphabet)}**{self.D}={self.space_size} "
                f"exceeds MAX_RADICAL_SPACE={MAX_RADICAL_SPACE}"
            )
        needed = self.lexicon_size + self.n_wug
        if needed > self.space_size:
            raise ValueError(
                f"D={self.D} with |alphabet|={len(self.alphabet)} gives only "
                f"{self.space_size} unique radicals, but lexicon_size+n_wug={needed} are "
                f"needed; duplicate inputs could carry different output classes."
            )
        if self.n_irregular < 1 or self.n_regular < 1:
            raise ValueError(
                f"lexicon_size={self.lexicon_size} with p_irregular_types="
                f"{self.p_irregular_types} leaves an empty regular/irregular group"
            )
        if self.class_scheme == "shared_k":
            if self.k_classes is None or self.k_classes > self.n_irregular:
                raise ValueError("shared_k needs 1 <= k_classes <= number of irregular types")
        elif self.k_classes is not None:
            raise ValueError("k_classes is only valid with class_scheme='shared_k'")
        if self.class_scheme == "by_cluster" and self.irregular_placement != "clustered":
            raise ValueError("class_scheme='by_cluster' requires irregular_placement='clustered'")
        if self.irregular_placement == "clustered" and self.n_clusters > self.n_irregular:
            raise ValueError("n_clusters cannot exceed the number of irregular types")
        if self.sampling == "finite" and self.train_size is None:
            raise ValueError("sampling='finite' requires train_size")
        if self.sampling == "stream" and self.train_size is not None:
            raise ValueError("train_size is only valid with sampling='finite'")
        return self

    @property
    def space_size(self) -> int:
        return len(self.alphabet) ** self.D

    @property
    def n_irregular(self) -> int:
        return round(self.lexicon_size * self.p_irregular_types)

    @property
    def n_regular(self) -> int:
        return self.lexicon_size - self.n_irregular

    @property
    def n_classes(self) -> int:
        if self.class_scheme == "unique_per_irregular":
            return self.n_irregular + 1
        if self.class_scheme == "shared_k":
            return int(self.k_classes or 0) + 1
        return self.n_clusters + 1

    @property
    def features_per_position(self) -> int:
        return len(self.alphabet) if self.encoding == "one_hot" else 1

    @property
    def input_dim(self) -> int:
        return self.D * self.features_per_position


class NetworkSpec(_Spec):
    schema_version: int = SCHEMA_VERSION

    arch: Literal["mlp", "transformer"] = "mlp"

    # mlp
    hidden: tuple[int, ...] = (16,)
    # shared
    activation: Literal["relu", "tanh", "gelu", "sigmoid", "leaky_relu", "silu"] = "relu"
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    norm: Literal["none", "layernorm", "batchnorm"] = "none"
    init: Literal["kaiming", "xavier", "normal"] = "kaiming"
    bias: bool = True
    # transformer (each of the D radical positions is one token)
    d_model: int = Field(default=32, ge=1)
    n_heads: int = Field(default=2, ge=1)
    n_layers: int = Field(default=1, ge=1)
    ff_dim: int = Field(default=64, ge=1)

    # optimization
    optimizer: Literal["sgd", "adam", "adamw"] = "adam"
    lr: float = Field(default=0.01, gt=0.0)
    momentum: float = Field(default=0.0, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    lr_schedule: Literal["const", "cosine", "linear_decay"] = "const"
    grad_clip: float | None = Field(default=None, gt=0.0)
    batch_size: int = Field(default=1, ge=1)

    # seeds; None = take the run seed
    init_seed: int | None = None
    data_order_seed: int | None = None

    # budget
    max_tokens: int = Field(default=5_000_000, ge=1)
    eval_every_tokens: int = Field(default=10_000, ge=1)
    early_stop: bool = True
    early_stop_patience: int = Field(default=5, ge=1)
    early_stop_min_delta: float = Field(default=1e-4, ge=0.0)

    # pruning (off unless both set)
    prune_interval_steps: int | None = Field(default=None, ge=1)
    max_prune_pct: float | None = Field(default=None, gt=0.0, le=100.0)

    @model_validator(mode="after")
    def _check(self) -> NetworkSpec:
        if self.arch == "transformer" and self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.norm == "batchnorm" and self.batch_size < 2:
            raise ValueError("batchnorm needs batch_size >= 2")
        if self.arch == "transformer" and self.norm == "batchnorm":
            raise ValueError("transformer supports norm in {'none','layernorm'}")
        if any(h < 1 for h in self.hidden):
            raise ValueError("hidden widths must be >= 1")
        if (self.prune_interval_steps is None) != (self.max_prune_pct is None):
            raise ValueError("set both prune_interval_steps and max_prune_pct, or neither")
        return self

    @property
    def pruning_enabled(self) -> bool:
        return self.prune_interval_steps is not None

    def with_seed(self, seed: int) -> NetworkSpec:
        """Fills unset seeds from the run seed, so the result (and its hash)
        fully determines the training run."""
        return self.model_copy(
            update={
                "init_seed": seed if self.init_seed is None else self.init_seed,
                "data_order_seed": seed if self.data_order_seed is None else self.data_order_seed,
            }
        )


def flatten(prefix: str, spec: BaseModel) -> dict[str, Any]:
    """Flat {prefix.field: value} view of a spec for MLflow / dataframes."""
    out: dict[str, Any] = {}
    for key, value in spec.model_dump(mode="json").items():
        out[f"{prefix}.{key}"] = json.dumps(value) if isinstance(value, list) else value
    return out
