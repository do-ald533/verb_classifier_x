from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from src.config import REGULAR_CLASS
from src.specs import DatasetSpec

WUG_POOL_SIZE = 5000
DATASET_FILES = ("spec.json", "data.pt", "stats.json")


@dataclass
class Dataset:
    """The lexicon (types) plus held-out wugs. Training tokens are not
    stored (stream mode): they are sampled on the fly by a TokenSource.

    X_types / X_wug are *model inputs* (after encoding); radicals_* keep the
    raw alphabet values for analysis. Types are ordered regular first, then
    irregular."""

    spec: DatasetSpec
    X_wug: torch.Tensor
    X_types: torch.Tensor
    y_types: torch.Tensor
    p_types: torch.Tensor  # token probability of each type under the sampler
    w_regular: torch.Tensor  # within-group sampling weights (sum to 1)
    w_irregular: torch.Tensor
    radicals_types: torch.Tensor
    radicals_wug: torch.Tensor
    train_indices: torch.Tensor | None = None  # only for sampling="finite"

    @property
    def n_regular(self) -> int:
        return self.spec.n_regular

    @property
    def n_irregular(self) -> int:
        return self.spec.n_irregular

    @property
    def n_classes(self) -> int:
        return self.spec.n_classes

    @property
    def p_token_irregular(self) -> float:
        return self.spec.p_token_irregular

    @property
    def input_dim(self) -> int:
        return self.spec.input_dim


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------
def _decode(indices: torch.Tensor, spec: DatasetSpec) -> torch.Tensor:
    """Radical index -> its D alphabet values (first position most
    significant, same order as itertools.product)."""
    k = len(spec.alphabet)
    alphabet = torch.tensor(spec.alphabet, dtype=torch.float32)
    powers = k ** torch.arange(spec.D - 1, -1, -1, dtype=torch.long)
    digits = (indices.unsqueeze(1) // powers.unsqueeze(0)) % k
    return alphabet[digits]


def _encode(radicals: torch.Tensor, spec: DatasetSpec) -> torch.Tensor:
    if spec.encoding == "raw":
        return radicals
    alphabet = torch.tensor(spec.alphabet, dtype=torch.float32)
    one_hot = (radicals.unsqueeze(-1) == alphabet).float()  # (n, D, K)
    return one_hot.reshape(radicals.shape[0], -1)


def _min_hamming(queries: torch.Tensor, reference: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    out = []
    for start in range(0, queries.shape[0], chunk):
        q = queries[start : start + chunk]
        out.append((q.unsqueeze(1) != reference.unsqueeze(0)).sum(-1).min(dim=1).values)
    return torch.cat(out).float()


def _choose_irregular(
    lexicon: torch.Tensor, spec: DatasetSpec, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Returns (order, irregular_clusters): `order` permutes the lexicon so
    regular verbs come first; irregular_clusters[i] is the cluster of the
    i-th irregular verb (None for random placement)."""
    n_total, n_irr = lexicon.shape[0], spec.n_irregular
    if spec.irregular_placement == "random":
        # lexicon rows are already in random order
        return torch.arange(n_total), None

    centers = torch.randperm(n_total, generator=generator)[: spec.n_clusters]
    per_cluster = [
        n_irr // spec.n_clusters + (c < n_irr % spec.n_clusters) for c in range(len(centers))
    ]
    taken = torch.zeros(n_total, dtype=torch.bool)
    chosen, cluster_ids = [], []
    for c, (center, quota) in enumerate(zip(centers, per_cluster, strict=True)):
        dist = (lexicon != lexicon[center]).sum(1).float()
        dist = dist + 0.5 * torch.rand(n_total, generator=generator)  # random tie-break
        dist[taken] = float("inf")
        sel = dist.topk(int(quota), largest=False).indices
        taken[sel] = True
        chosen.append(sel)
        cluster_ids.extend([c] * int(quota))
    irregular_pos = torch.cat(chosen)
    regular_pos = torch.nonzero(~taken).squeeze(1)
    return torch.cat([regular_pos, irregular_pos]), torch.tensor(cluster_ids, dtype=torch.long)


def _group_weights(n: int, spec: DatasetSpec, generator: torch.Generator) -> torch.Tensor:
    if spec.freq_distribution == "uniform":
        return torch.full((n,), 1.0 / n)
    ranks = torch.arange(1, n + 1, dtype=torch.float32)
    w = ranks.pow(-spec.zipf_exponent)[torch.randperm(n, generator=generator)]
    return w / w.sum()


def _choose_wugs(
    spec: DatasetSpec,
    perm: torch.Tensor,
    lexicon: torch.Tensor,
    irregular: torch.Tensor,
) -> torch.Tensor:
    n_wug, lex_size = spec.n_wug, spec.lexicon_size
    if spec.wug_strategy == "random":
        return _decode(perm[:n_wug], spec)
    pool_idx = torch.cat([perm[:n_wug], perm[n_wug + lex_size : n_wug + lex_size + WUG_POOL_SIZE]])
    pool = _decode(pool_idx, spec)
    if spec.wug_strategy == "near_irregular":
        score = _min_hamming(pool, irregular)
        pick = score.argsort(stable=True)[:n_wug]
    else:  # far_from_lexicon
        score = _min_hamming(pool, lexicon)
        pick = (-score).argsort(stable=True)[:n_wug]
    return pool[pick]


def build_dataset(spec: DatasetSpec) -> Dataset:
    generator = torch.Generator().manual_seed(spec.seed)
    n_wug, lex_size = spec.n_wug, spec.lexicon_size

    # unique radicals => no two verbs share an input with different labels
    perm = torch.randperm(spec.space_size, generator=generator)
    lexicon = _decode(perm[n_wug : n_wug + lex_size], spec)

    order, irr_clusters = _choose_irregular(lexicon, spec, generator)
    radicals_types = lexicon[order]
    irregular_radicals = radicals_types[spec.n_regular :]

    radicals_wug = _choose_wugs(spec, perm, radicals_types, irregular_radicals)

    n_irr = spec.n_irregular
    if spec.class_scheme == "unique_per_irregular":
        y_irr = torch.arange(1, n_irr + 1, dtype=torch.long)
    elif spec.class_scheme == "shared_k":
        k = int(spec.k_classes or 1)
        y_irr = (1 + torch.arange(n_irr) % k)[torch.randperm(n_irr, generator=generator)]
    else:  # by_cluster
        assert irr_clusters is not None
        y_irr = 1 + irr_clusters
    y_types = torch.cat([torch.full((spec.n_regular,), REGULAR_CLASS, dtype=torch.long), y_irr])

    w_regular = _group_weights(spec.n_regular, spec, generator)
    w_irregular = _group_weights(n_irr, spec, generator)
    p_types = torch.cat(
        [(1.0 - spec.p_token_irregular) * w_regular, spec.p_token_irregular * w_irregular]
    )

    dataset = Dataset(
        spec=spec,
        X_wug=_encode(radicals_wug, spec),
        X_types=_encode(radicals_types, spec),
        y_types=y_types,
        p_types=p_types,
        w_regular=w_regular,
        w_irregular=w_irregular,
        radicals_types=radicals_types,
        radicals_wug=radicals_wug,
    )
    if spec.sampling == "finite":
        dataset.train_indices = sample_token_indices(dataset, int(spec.train_size or 0), generator)
    return dataset


# --------------------------------------------------------------------------
# token sampling
# --------------------------------------------------------------------------
def sample_token_indices(dataset: Dataset, n: int, generator: torch.Generator) -> torch.Tensor:
    """Indices into dataset.X_types for n i.i.d. tokens: first draw whether
    each token is irregular (prob. p_token_irregular), then a verb within the
    chosen group (uniform, or by the group's Zipf weights)."""
    is_irregular = torch.rand(n, generator=generator) < dataset.p_token_irregular
    n_irr = int(is_irregular.sum())
    out = torch.empty(n, dtype=torch.long)
    if n_irr:
        out[is_irregular] = dataset.n_regular + torch.multinomial(
            dataset.w_irregular, n_irr, replacement=True, generator=generator
        )
    if n - n_irr:
        out[~is_irregular] = torch.multinomial(
            dataset.w_regular, n - n_irr, replacement=True, generator=generator
        )
    return out


class TokenSource:
    """Endless token stream for training. stream: fresh i.i.d. draws.
    finite: cycles through the dataset's stored train_indices, reshuffled
    each pass (its own generator, so data order is independent of the
    dataset seed)."""

    def __init__(self, dataset: Dataset, generator: torch.Generator):
        self.dataset = dataset
        self.generator = generator
        self._buffer = torch.empty(0, dtype=torch.long)
        if dataset.spec.sampling == "finite" and dataset.train_indices is None:
            raise ValueError("finite dataset is missing train_indices")

    def take(self, n: int) -> torch.Tensor:
        if self.dataset.spec.sampling == "stream":
            return sample_token_indices(self.dataset, n, self.generator)
        pieces = [self._buffer]
        have = self._buffer.numel()
        assert self.dataset.train_indices is not None
        while have < n:
            perm = torch.randperm(self.dataset.train_indices.numel(), generator=self.generator)
            fresh = self.dataset.train_indices[perm]
            pieces.append(fresh)
            have += fresh.numel()
        merged = torch.cat(pieces)
        self._buffer = merged[n:]
        return merged[:n]


# --------------------------------------------------------------------------
# stats + persistence
# --------------------------------------------------------------------------
def dataset_stats(dataset: Dataset) -> dict:
    spec = dataset.spec
    p = dataset.p_types
    entropy = float(-(p[p > 0] * p[p > 0].log()).sum())
    irr = dataset.radicals_types[spec.n_regular :]
    probe = irr[: min(len(irr), 300)]
    pairwise = (probe.unsqueeze(1) != probe.unsqueeze(0)).sum(-1).float()
    n = probe.shape[0]
    mean_pairwise = float(pairwise.sum() / (n * (n - 1))) if n > 1 else 0.0
    class_counts = torch.bincount(dataset.y_types, minlength=spec.n_classes)
    return {
        "dataset_hash": spec.hash,
        "n_types": spec.lexicon_size,
        "n_regular": spec.n_regular,
        "n_irregular": spec.n_irregular,
        "n_classes": spec.n_classes,
        "input_dim": spec.input_dim,
        "space_size": spec.space_size,
        "lexicon_density": spec.lexicon_size / spec.space_size,
        "token_entropy_nats": entropy,
        "max_type_token_prob": float(p.max()),
        "irregular_mean_pairwise_hamming": mean_pairwise,
        "wug_mean_min_hamming_to_lexicon": float(
            _min_hamming(dataset.radicals_wug, dataset.radicals_types).mean()
        ),
        "wug_mean_min_hamming_to_irregular": float(_min_hamming(dataset.radicals_wug, irr).mean()),
        "class_count_min": int(class_counts.min()),
        "class_count_max": int(class_counts.max()),
        "n_train_tokens_finite": (
            int(dataset.train_indices.numel()) if dataset.train_indices is not None else None
        ),
    }


def dataset_dir(root: str | Path, spec: DatasetSpec) -> Path:
    return Path(root) / "datasets" / spec.hash


def is_complete(directory: Path) -> bool:
    return all((directory / name).exists() for name in DATASET_FILES)


def save_dataset(dataset: Dataset, root: str | Path) -> Path:
    """Content-addressed, atomic: written to a temp dir then renamed, so a
    concurrent reader never sees a half-written dataset."""
    target = dataset_dir(root, dataset.spec)
    if is_complete(target):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".tmp-{dataset.spec.hash}-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    (tmp / "spec.json").write_text(dataset.spec.model_dump_json(indent=2))
    (tmp / "stats.json").write_text(json.dumps(dataset_stats(dataset), indent=2))
    payload = {
        "radicals_types": dataset.radicals_types,
        "radicals_wug": dataset.radicals_wug,
        "y_types": dataset.y_types,
        "w_regular": dataset.w_regular,
        "w_irregular": dataset.w_irregular,
    }
    if dataset.train_indices is not None:
        payload["train_indices"] = dataset.train_indices
    torch.save(payload, tmp / "data.pt")
    try:
        os.replace(tmp, target)
    except OSError:  # another process won the race; its copy is identical
        shutil.rmtree(tmp, ignore_errors=True)
    return target


def load_dataset(directory: str | Path) -> Dataset:
    directory = Path(directory)
    if not is_complete(directory):
        raise FileNotFoundError(f"{directory} is not a complete dataset (need {DATASET_FILES})")
    spec = DatasetSpec.model_validate_json((directory / "spec.json").read_text())
    data = torch.load(directory / "data.pt", weights_only=True)
    w_reg, w_irr = data["w_regular"], data["w_irregular"]
    p = spec.p_token_irregular
    return Dataset(
        spec=spec,
        X_wug=_encode(data["radicals_wug"], spec),
        X_types=_encode(data["radicals_types"], spec),
        y_types=data["y_types"],
        p_types=torch.cat([(1.0 - p) * w_reg, p * w_irr]),
        w_regular=w_reg,
        w_irregular=w_irr,
        radicals_types=data["radicals_types"],
        radicals_wug=data["radicals_wug"],
        train_indices=data.get("train_indices"),
    )
