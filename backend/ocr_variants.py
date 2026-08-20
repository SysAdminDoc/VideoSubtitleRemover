"""Canonical PaddleOCR model-family names and aliases.

This is a dependency-free leaf module on purpose. The detector, the
PaddleOCR constructor helper, and both configuration mirrors all need the
same alias table, but the detection cascade must be able to normalize a
variant without importing anything that pulls in PaddleOCR: the cascade
tests block ``backend.paddle_compat`` to simulate the engine being absent.

Upstream identifiers are ``PP-OCRv<gen>_<family>_det`` / ``_rec``. The
PP-OCRv5 families are mobile and server; paddleocr 3.7.0 added the
PP-OCRv6 tiers tiny (1.5M), small (7.7M) and medium (34.5M), which
PaddleOCR reports at +4.6% detection and +5.1% recognition over
PP-OCRv5-server. Verified against the PaddlePaddle model repositories.

The default stays PP-OCRv5 mobile so upgrading paddleocr never silently
changes model weights, latency, or memory for an existing user.
"""

from __future__ import annotations

PADDLEOCR_DEFAULT_VARIANT = "mobile"
PADDLEOCR_V5_FAMILIES = ("mobile", "server")
PADDLEOCR_V6_TIERS = ("tiny", "small", "medium")


def _build_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for family in PADDLEOCR_V5_FAMILIES:
        aliases[family] = family
        for prefix in ("pp-ocrv5-", "ppocrv5-", "v5-"):
            aliases[f"{prefix}{family}"] = family
    for tier in PADDLEOCR_V6_TIERS:
        canonical = f"v6-{tier}"
        aliases[canonical] = canonical
        for prefix in ("pp-ocrv6-", "ppocrv6-"):
            aliases[f"{prefix}{tier}"] = canonical
        # A bare tier name is unambiguous: no PP-OCRv5 family shares it.
        aliases[tier] = canonical
    return aliases


PADDLEOCR_VARIANT_ALIASES = _build_aliases()
PADDLEOCR_CANONICAL_VARIANTS = tuple(
    sorted(set(PADDLEOCR_VARIANT_ALIASES.values()))
)


def normalize_paddleocr_variant(value: object) -> str:
    """Normalize a PaddleOCR family selection to its canonical name.

    Returns a PP-OCRv5 family ("mobile"/"server") or a PP-OCRv6 tier as
    "v6-<tier>". Unrecognized values fall back to the reviewed default.
    """
    text = str(
        value or PADDLEOCR_DEFAULT_VARIANT
    ).strip().lower().replace("_", "-")
    return PADDLEOCR_VARIANT_ALIASES.get(text, PADDLEOCR_DEFAULT_VARIANT)


def paddleocr_variant_generation(value: object = PADDLEOCR_DEFAULT_VARIANT) -> str:
    """Return "v5" or "v6" for a variant selection."""
    return "v6" if normalize_paddleocr_variant(value).startswith("v6-") else "v5"


def paddleocr_model_names(
    value: object = PADDLEOCR_DEFAULT_VARIANT,
) -> tuple[str, str]:
    """Return the explicit detection and recognition model names."""
    normalized = normalize_paddleocr_variant(value)
    if normalized.startswith("v6-"):
        prefix = f"PP-OCRv6_{normalized[3:]}"
    else:
        prefix = f"PP-OCRv5_{normalized}"
    return f"{prefix}_det", f"{prefix}_rec"
