"""Frozen Apps transport hashes accepted by the synthetic beta runtime."""

from __future__ import annotations

WIRE16: tuple[tuple[str, str, str], ...] = (
    (
        "synthetic_beta_v1", "ack.json",
        "f612f133443a8de989d03bc5b644b227db5ea11b7afcb3b577fd4b3ae352a023",
    ),
    (
        "synthetic_beta_v1", "admin_binding.json",
        "28a23de6f5d01c325bad5053f83e4d3e07855b99618b7d7cb1e86d15209fc687",
    ),
    (
        "synthetic_beta_v1", "native.json",
        "46d587fe49e9258b1b256ec8c39d36f6382c0263ef83846ab8ca80a5656370fa",
    ),
    (
        "synthetic_beta_v1", "observer.json",
        "7b9078042edde8638de8948485dd1cf01ab41af66bdbe166d8cbe4baef145cca",
    ),
    (
        "synthetic_beta_v1", "phone_binding.json",
        "cb2514507d34a9d5eea4bf12f51f13442ba669468f0ab4ca8a1ed936c735477f",
    ),
    (
        "synthetic_beta_v1", "projection.json",
        "8d951f7cb83067756843d88659e479169c4e71beeadaf901da29216fac504d1e",
    ),
    (
        "synthetic_beta_v1", "result.json",
        "c6a56562ff11bfe1adfa51805b53bc8d38e3936453d6de6b5f3e44746860d594",
    ),
    (
        "synthetic_beta_v1", "run.json",
        "35d92402eca13b3024a5c628695656b75c47d7cec6a434f862f8081f73cfde65",
    ),
    (
        "golden", "ack.json",
        "646f945f51bbbda8c23a1703df461a775e1d6ce9a9d3674642d65ba6ee016fe2",
    ),
    (
        "golden", "admin_binding.json",
        "ded9b5ff7d3b0393e64437a777b57cd4630e706b6e0853a693a9f0c25e6ea859",
    ),
    (
        "golden", "native.json",
        "880ae7a59a65a78f2fae21b7dc2801650af20d0e6f2d08188770aa0d57b2b44d",
    ),
    (
        "golden", "observer.json",
        "40cc85a2660c47399d0d74ec3f68733f3ee5d4f7ae36ccb04a2ae3630fde1001",
    ),
    (
        "golden", "phone_binding.json",
        "f766a737e540a19d81248f01dc44838832090c45874390ad9360ccacbee5ffa2",
    ),
    (
        "golden", "projection.json",
        "b970537ddcb2fe9f7d1f17c22b9be5b545d7f4322bb96ff1558e45b42b99f92d",
    ),
    (
        "golden", "result.json",
        "ed0e9df625cb7f5b1c225a9b3bb2d6229090451243bfb2fc82f008a739682323",
    ),
    (
        "golden", "run.json",
        "a501b3633eb9ba2e3e9525ce99310ee20a8c2ce304e28d1a87d5e81afed7895f",
    ),
)


def validate_wire16(value: object) -> None:
    expected = [
        {"kind": kind, "name": name, "sha256": digest}
        for kind, name, digest in WIRE16
    ]
    if value != expected:
        raise ValueError("beta_bundle_manifest_invalid")
