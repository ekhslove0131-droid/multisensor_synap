from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator, SeedSequence


@dataclass
class SeedTree:
    root_seed: int
    _used: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def child_seed(self, namespace: str) -> int:
        if not namespace or namespace.startswith("/") or namespace.endswith("/"):
            raise ValueError("namespace must be a non-empty normalized path")
        digest = hashlib.sha256(namespace.encode("utf-8")).digest()
        digest_words = np.frombuffer(digest, dtype=">u4").astype(np.uint32)
        sequence = SeedSequence([self.root_seed, *[int(word) for word in digest_words]])
        seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
        self._used[namespace] = seed
        return seed

    def rng(self, namespace: str) -> Generator:
        return np.random.default_rng(self.child_seed(namespace))

    @property
    def used_namespaces(self) -> dict[str, int]:
        return dict(sorted(self._used.items()))
