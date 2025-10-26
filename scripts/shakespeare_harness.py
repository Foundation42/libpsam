#!/usr/bin/env python3
"""Shakespeare composite harness.

Builds tragedy/comedy overlays on top of a shared Shakespeare base model,
constructs layered composites, and prints prediction samples for a prompt.

Example:
    python scripts/shakespeare_harness.py --prompt "to be or not to be"

Requires libpsam shared library on the search path (set LIBPSAM_PATH if needed).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from psam import LayeredComposite, PSAM, load_composite, save_composite_manifest  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = REPO_ROOT / "corpora" / "text" / "tiny_shakespeare.psam"
DEFAULT_BASE_VOCAB = REPO_ROOT / "corpora" / "text" / "tiny_shakespeare.tsv"
FOLGER_DIR = REPO_ROOT / "corpora" / "text" / "Folger"

TRAGEDY_FILES = [
    "hamlet",
    "macbeth",
    "othello",
    "king-lear",
    "romeo-and-juliet",
    "julius-caesar",
]
COMEDY_FILES = [
    "a-midsummer-nights-dream",
    "twelfth-night",
    "as-you-like-it",
    "the-merchant-of-venice",
    "the-comedy-of-errors",
    "the-tempest",
]


def load_vocab(path: Path) -> Tuple[Dict[str, int], List[str]]:
    token_to_id: Dict[str, int] = {}
    id_to_token: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            idx = int(parts[0])
            token = parts[1]
            token_to_id[token] = idx
            if idx >= len(id_to_token):
                id_to_token.extend([""] * (idx - len(id_to_token) + 1))
            id_to_token[idx] = token
    return token_to_id, id_to_token


def tokenize_files(files: Sequence[Path], token_to_id: Dict[str, int], chunk_size: int = 4096):
    total = 0
    matched = 0
    chunk: List[int] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                for raw in line.strip().split():
                    total += 1
                    token_id = token_to_id.get(raw)
                    if token_id is None:
                        continue
                    matched += 1
                    chunk.append(token_id)
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
    if chunk:
        yield chunk
    yield (total, matched)  # sentinel to report counts


def train_overlay(
    name: str,
    files: Sequence[Path],
    token_to_id: Dict[str, int],
    vocab_size: int,
    window: int,
    top_k: int,
    out_path: Path,
    force: bool,
):
    if out_path.exists() and not force:
        print(f"[overlay] Reusing existing {name} overlay at {out_path}")
        return out_path

    print(f"[overlay] Training {name} overlay from {len(files)} files...")
    overlay = PSAM(vocab_size, window, top_k)
    total = matched = 0
    for batch in tokenize_files(files, token_to_id):
        if isinstance(batch, tuple):
            total, matched = batch
            break
        overlay.train_batch(batch)

    if matched == 0:
        overlay.destroy()
        raise RuntimeError(f"No tokens from {name} files matched the base vocabulary")
    overlay.finalize_training()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(str(out_path))
    ratio = matched / total if total else 0.0
    print(
        f"[overlay] Saved {name} overlay -> {out_path} | tokens used={matched}/{total} ({ratio:.2%})"
    )
    overlay.destroy()
    return out_path


def build_composite(
    out_path: Path,
    base_model: Path,
    overlays: List[Tuple[str, float, Path]],
    force: bool,
):
    if out_path.exists() and not force:
        print(f"[composite] Reusing {out_path}")
        return
    descs = [
        {"id": layer_id, "weight": weight, "path": str(path)}
        for layer_id, weight, path in overlays
    ]
    save_composite_manifest(
        out_path=str(out_path),
        base_model_path=str(base_model),
        overlays=descs,
        base_weight=1.0,
        created_by="shakespeare-harness",
    )
    print(f"[composite] Wrote {out_path}")


def load_text_tokens(text: str, token_to_id: Dict[str, int]) -> List[int]:
    tokens = []
    for raw in text.strip().split():
        token_id = token_to_id.get(raw)
        if token_id is not None:
            tokens.append(token_id)
    return tokens


def pretty_print_predictions(title: str, ids: List[int], scores: np.ndarray, id_to_token: List[str]):
    rows = []
    for idx, score in zip(ids, scores):
        token = id_to_token[idx] if 0 <= idx < len(id_to_token) else f"<{idx}>"
        rows.append({"token": token, "score": float(score)})
    print(json.dumps({"composite": title, "predictions": rows}, ensure_ascii=False))


def gather_file_paths(names: Sequence[str]) -> List[Path]:
    return [FOLGER_DIR / f"{name}.txt" for name in names]


def main():
    parser = argparse.ArgumentParser(description="Shakespeare layered composite harness")
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--base-vocab", type=Path, default=DEFAULT_BASE_VOCAB)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "shakespeare")
    parser.add_argument("--prompt", default="to be or not to be")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--model-top-k", type=int, default=32)
    parser.add_argument("--tragedy-weight", type=float, default=1.3)
    parser.add_argument("--comedy-weight", type=float, default=1.1)
    parser.add_argument("--force", action="store_true", help="Rebuild overlays and composites even if cached")
    args = parser.parse_args()

    token_to_id, id_to_token = load_vocab(args.base_vocab)
    vocab_size = len(id_to_token)
    if vocab_size == 0:
        raise RuntimeError("Base vocabulary is empty")

    tragedy_files = gather_file_paths(TRAGEDY_FILES)
    comedy_files = gather_file_paths(COMEDY_FILES)

    overlays_dir = args.out_dir / "overlays"
    tragedy_overlay = train_overlay(
        "tragedy",
        tragedy_files,
        token_to_id,
        vocab_size,
        args.window,
        args.model_top_k,
        overlays_dir / "tragedy.psam",
        args.force,
    )
    comedy_overlay = train_overlay(
        "comedy",
        comedy_files,
        token_to_id,
        vocab_size,
        args.window,
        args.model_top_k,
        overlays_dir / "comedy.psam",
        args.force,
    )

    composites_dir = args.out_dir / "composites"
    composites_dir.mkdir(parents=True, exist_ok=True)

    tragedy_comp = composites_dir / "tragedy_blend.psamc"
    comedy_comp = composites_dir / "comedy_blend.psamc"
    mixed_comp = composites_dir / "mixed_blend.psamc"

    build_composite(tragedy_comp, args.base_model, [("tragedy", args.tragedy_weight, tragedy_overlay)], args.force)
    build_composite(comedy_comp, args.base_model, [("comedy", args.comedy_weight, comedy_overlay)], args.force)
    build_composite(
        mixed_comp,
        args.base_model,
        [
            ("tragedy", args.tragedy_weight, tragedy_overlay),
            ("comedy", args.comedy_weight, comedy_overlay),
        ],
        args.force,
    )

    prompt_ids = load_text_tokens(args.prompt, token_to_id)
    if not prompt_ids:
        raise ValueError("Prompt did not map to any tokens in the base vocabulary")

    for comp_path in [tragedy_comp, comedy_comp, mixed_comp]:
        composite = load_composite(str(comp_path), verify_integrity=False)
        ids, scores = composite.predict(prompt_ids, args.top_k)
        pretty_print_predictions(comp_path.name, ids, scores, id_to_token)


if __name__ == "__main__":
    main()
