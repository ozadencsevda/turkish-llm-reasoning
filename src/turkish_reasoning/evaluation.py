"""
Evaluation utilities for the Turkish LLM reasoning benchmark.

The notebook stays readable; technical work lives here.

Main workflow:
1. Load saved outputs from notebooks 01-04 and verify dataset identity.
2. Apply conservative deterministic answer checks.
3. Validate semantic-similarity and NLI evaluator candidates against
   human-annotated development data.
4. Check the frozen choices on held-out human test data.
5. Apply the validated evaluators to the full saved-output set.

No benchmark LLM is queried in this module.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence
import gc
import hashlib
import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_SEMANTIC_EVALUATORS = {
    "jina_v3": {
        "model_id": "jinaai/jina-embeddings-v3",
        "adapter": "jina_text_matching",
        "trust_remote_code": True,
    },
    "multilingual_mpnet": {
        "model_id": (
            "sentence-transformers/"
            "paraphrase-multilingual-mpnet-base-v2"
        ),
        "adapter": "sentence_transformer",
        "trust_remote_code": False,
    },
}

DEFAULT_NLI_EVALUATORS = {
    "mdeberta_xnli": {
        "model_id": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    },
    "xlmr_xnli": {
        "model_id": "joeddav/xlm-roberta-large-xnli",
    },
}

EXPERIMENT_TYPES = {
    "closed_models_base": ("base", "closed"),
    "closed_models_variants": ("variants", "closed"),
    "open_models_base": ("base", "open"),
    "open_models_variants": ("variants", "open"),
}

ANSWER_HUMAN_TO_BINARY = {
    "equivalent": 1,
    "not_equivalent": 0,
}

REASONING_HUMAN_TO_NLI = {
    "supported": "entailment",
    "contradicted": "contradiction",
    "nei": "neutral",
}

EXPECTED_NLI_LABELS = {
    "entailment",
    "neutral",
    "contradiction",
}


class EvaluationError(RuntimeError):
    """Raised when the evaluation pipeline detects an unsafe state."""


class LongNLIInputError(EvaluationError):
    """Raised instead of silently truncating an NLI input."""


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def stable_id(*parts: Any, length: int = 24) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save_json(data: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(dict(data), file, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def atomic_save_csv(dataframe: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def _dataset_registry(project_root: Path) -> dict[str, dict[str, Path]]:
    return {
        "original": {
            "base": project_root / "data" / "base" / "base_questions.csv",
            "variants": project_root / "data" / "variants" / "variants.csv",
        },
        "current": {
            "base": (
                project_root / "data" / "base" / "base_questions_current.csv"
            ),
            "variants": (
                project_root / "data" / "variants" / "variants_current.csv"
            ),
        },
        "smoke": {
            "base": (
                project_root / "data" / "base" / "smoke_base_v1.csv"
            ),
            "variants": (
                project_root / "data" / "variants" / "smoke_variants_v1.csv"
            ),
        },
    }


# ---------------------------------------------------------------------
# Load and verify saved generation outputs
# ---------------------------------------------------------------------

def _run_matches(metadata: Mapping[str, Any], run_label: str) -> bool:
    if metadata.get("run_label") is not None:
        return str(metadata["run_label"]) == run_label
    return str(metadata.get("run_name", "")).endswith(f"__{run_label}")


def _dataset_hash_from_metadata(metadata: Mapping[str, Any]) -> str | None:
    for key in ("dataset_sha256", "dataset_hash", "data_sha256"):
        if metadata.get(key):
            return str(metadata[key])
    return None


def _canonical_columns(input_type: str) -> list[str]:
    if input_type == "base":
        return [
            "question_id", "category", "difficulty", "context", "question",
            "gold_evidence", "gold_answer",
        ]
    return [
        "variant_id", "base_question_id", "category", "difficulty",
        "variant_type", "morphological_feature", "context", "question",
        "gold_evidence", "gold_answer",
    ]


def _id_column(input_type: str) -> str:
    return "question_id" if input_type == "base" else "variant_id"


def _discover_generation_files(
    project_root: Path,
    run_label: str,
) -> list[tuple[Path, str]]:
    locations = {
        "base": project_root / "results" / "raw" / "base",
        "variants": project_root / "results" / "raw" / "variants",
    }
    files = []
    for input_type, directory in locations.items():
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".metadata.json"):
                continue
            files.append((path, input_type))
    return files


def load_generation_results(
    project_root: str | Path,
    source_run_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load final outputs from notebooks 01-04 and join them to canonical data.

    Safety checks:
    - requested run label,
    - completed status,
    - known experiment type,
    - dataset SHA-256,
    - record IDs,
    - duplicate IDs.
    """
    project_root = Path(project_root)
    registry = _dataset_registry(project_root)

    frames = []
    manifest = []

    for path, directory_input_type in _discover_generation_files(
        project_root, source_run_label
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(payload, dict):
            continue
        if not isinstance(payload.get("metadata"), dict):
            continue
        if not isinstance(payload.get("records"), list):
            continue

        metadata = payload["metadata"]
        if not _run_matches(metadata, source_run_label):
            continue

        experiment_type = metadata.get("experiment_type")
        if experiment_type not in EXPERIMENT_TYPES:
            raise EvaluationError(
                f"Unknown experiment_type in {path.name}: {experiment_type!r}"
            )

        input_type, model_access = EXPERIMENT_TYPES[experiment_type]
        if input_type != directory_input_type:
            raise EvaluationError(
                f"Directory/metadata mismatch in {path.name}."
            )
        if metadata.get("status") != "completed":
            raise EvaluationError(
                f"Run is not marked completed: {path.name}"
            )

        profile = str(metadata.get("dataset_profile", "original"))
        if profile not in registry:
            raise EvaluationError(f"Unknown dataset profile: {profile!r}")

        dataset_path = registry[profile][input_type]
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Canonical dataset not found: {dataset_path}"
            )

        dataset_hash = sha256_file(dataset_path)
        stored_hash = _dataset_hash_from_metadata(metadata)
        if stored_hash is None:
            raise EvaluationError(
                f"Dataset hash missing from generation metadata: {path.name}"
            )
        if stored_hash != dataset_hash:
            raise EvaluationError(
                f"Dataset hash mismatch: {path.name}"
            )

        canonical = pd.read_csv(dataset_path)
        required = _canonical_columns(input_type)
        missing = [c for c in required if c not in canonical.columns]
        if missing:
            raise EvaluationError(
                f"Canonical dataset missing columns: {missing}"
            )

        generated = pd.DataFrame(payload["records"])
        id_col = _id_column(input_type)
        if id_col not in generated.columns:
            raise EvaluationError(f"{id_col} missing in {path.name}")

        generated[id_col] = generated[id_col].astype(str)
        canonical[id_col] = canonical[id_col].astype(str)

        if generated[id_col].duplicated().any():
            raise EvaluationError(f"Duplicate {id_col} values in {path.name}")

        canonical_ids = set(canonical[id_col])
        unknown = set(generated[id_col]) - canonical_ids
        if unknown:
            raise EvaluationError(
                f"{path.name} contains unknown IDs: {sorted(unknown)[:5]}"
            )

        # A completed generation run must contain every benchmark item that is
        # actually evaluable. The original base dataset has one row with
        # missing gold fields, so completeness is checked against usable rows
        # rather than the raw CSV row count.
        required_content = ["context", "question", "gold_evidence", "gold_answer"]
        usable_canonical = canonical.dropna(subset=required_content).copy()
        usable_ids = set(usable_canonical[id_col])
        generated_ids = set(generated[id_col])

        missing_ids = usable_ids - generated_ids
        unexpected_unusable_ids = generated_ids - usable_ids

        if missing_ids:
            raise EvaluationError(
                f"Completed run {path.name} is missing evaluable benchmark "
                f"IDs: {sorted(missing_ids)[:5]}"
            )

        if unexpected_unusable_ids:
            raise EvaluationError(
                f"Completed run {path.name} contains IDs excluded from the "
                f"evaluable benchmark: {sorted(unexpected_unusable_ids)[:5]}"
            )

        generation_fields = [
            id_col, "model_label", "model_id", "provider", "model_reasoning",
            "model_answer", "model_confidence", "raw_response", "error",
        ]
        for column in generation_fields:
            if column not in generated.columns:
                generated[column] = None

        merged = canonical[required].merge(
            generated[generation_fields],
            on=id_col,
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(generated):
            raise EvaluationError(
                f"Some generated records could not be matched: {path.name}"
            )

        run_name = str(metadata.get("run_name", path.stem))
        merged["record_id"] = merged[id_col]
        merged["input_type"] = input_type
        merged["experiment_type"] = experiment_type
        merged["model_access"] = model_access
        merged["dataset_profile"] = profile
        merged["run_name"] = run_name
        merged["source_file"] = str(path.relative_to(project_root))
        merged["source_file_sha256"] = sha256_file(path)
        merged["record_uid"] = merged["record_id"].map(
            lambda rid: stable_id(run_name, input_type, rid)
        )
        frames.append(merged)

        manifest.append({
            "source_file": str(path.relative_to(project_root)),
            "source_file_sha256": sha256_file(path),
            "experiment_type": experiment_type,
            "input_type": input_type,
            "model_access": model_access,
            "dataset_profile": profile,
            "dataset_path": str(dataset_path.relative_to(project_root)),
            "dataset_sha256": dataset_hash,
            "run_name": run_name,
            "record_count": len(generated),
        })

    if not frames:
        raise FileNotFoundError(
            f"No completed outputs found for run {source_run_label!r}."
        )

    records = pd.concat(frames, ignore_index=True, sort=False)
    if records["record_uid"].duplicated().any():
        raise EvaluationError("Duplicate record_uid values detected.")

    return records, pd.DataFrame(manifest)


# ---------------------------------------------------------------------
# Conservative deterministic answer checks
# ---------------------------------------------------------------------

_SIMPLE_NUMBER_RE = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")


def _normalize_answer(value: Any) -> str:
    if is_blank(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\n\r\"'“”‘’")
    return re.sub(r"[.!?]+$", "", text).strip()


def _simple_decimal(value: Any) -> Decimal | None:
    text = _normalize_answer(value)
    if not _SIMPLE_NUMBER_RE.fullmatch(text):
        return None

    separator = "." if "." in text else "," if "," in text else None
    if separator:
        decimal_part = text.lstrip("+-").split(separator, 1)[1]
        # 1.000 can be locale-dependent; do not guess.
        if len(decimal_part) == 3:
            return None
        text = text.replace(",", ".")

    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _deterministic_answer_check(
    model_answer: Any,
    gold_answer: Any,
) -> dict[str, Any]:
    model_text = _normalize_answer(model_answer)
    gold_text = _normalize_answer(gold_answer)

    result = {
        "normalized_exact_match": False,
        "numeric_equivalent": False,
        "numeric_mismatch": False,
        "rule_based_positive": False,
        "rule_based_negative": False,
        "rule_based_method": None,
    }

    if not model_text or not gold_text:
        return result

    if model_text == gold_text:
        result.update({
            "normalized_exact_match": True,
            "rule_based_positive": True,
            "rule_based_method": "normalized_exact",
        })
        return result

    model_number = _simple_decimal(model_text)
    gold_number = _simple_decimal(gold_text)
    if model_number is not None and gold_number is not None:
        if model_number == gold_number:
            result.update({
                "numeric_equivalent": True,
                "rule_based_positive": True,
                "rule_based_method": "numeric",
            })
        else:
            # Two standalone, unambiguous numeric answers with different
            # values are safely incorrect; do not send them to an embedding
            # model, which can assign high similarity to nearby numbers.
            result.update({
                "numeric_mismatch": True,
                "rule_based_negative": True,
                "rule_based_method": "numeric_mismatch",
            })

    return result


def _normalize_confidence(value: Any) -> float:
    if is_blank(value):
        return float("nan")
    try:
        number = float(str(value).strip().replace("%", "").strip())
    except ValueError:
        return float("nan")
    return number / 100.0 if 0 <= number <= 100 else float("nan")


def prepare_evaluation_data(records_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare answer-check and confidence fields; no learned evaluator runs."""
    required = {
        "model_answer", "gold_answer", "model_confidence", "error",
    }
    missing = required - set(records_df.columns)
    if missing:
        raise EvaluationError(f"Missing columns: {sorted(missing)}")

    df = records_df.copy()
    df["generation_success"] = df.apply(
        lambda row: (
            is_blank(row["error"])
            and not is_blank(row["model_answer"])
        ),
        axis=1,
    )

    checks = df.apply(
        lambda row: pd.Series(
            _deterministic_answer_check(
                row["model_answer"], row["gold_answer"]
            )
        ),
        axis=1,
    )
    for column in checks.columns:
        df[column] = checks[column]

    df["needs_semantic_evaluation"] = (
        df["generation_success"]
        & ~df["rule_based_positive"]
        & ~df["rule_based_negative"]
    )
    df["confidence_normalized"] = df["model_confidence"].apply(
        _normalize_confidence
    )

    df["answer_correct"] = pd.Series(
        pd.NA, index=df.index, dtype="Int64"
    )
    deterministic_positive = (
        df["generation_success"] & df["rule_based_positive"]
    )
    deterministic_negative = (
        df["generation_success"] & df["rule_based_negative"]
    )

    df.loc[deterministic_positive, "answer_correct"] = 1
    df.loc[deterministic_negative, "answer_correct"] = 0

    df["accuracy_method"] = None
    deterministic = deterministic_positive | deterministic_negative
    df.loc[deterministic, "accuracy_method"] = df.loc[
        deterministic, "rule_based_method"
    ]
    return df


# ---------------------------------------------------------------------
# Evaluator-model helpers
# ---------------------------------------------------------------------

def _hf_revision(model_id: str) -> str:
    from huggingface_hub import HfApi
    return str(HfApi().model_info(repo_id=model_id, revision="main").sha)


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _clear_model_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------
# Semantic equivalence
# ---------------------------------------------------------------------

def _semantic_scores(
    evaluator_key: str,
    text_a: Sequence[str],
    text_b: Sequence[str],
    *,
    revision: str | None = None,
    batch_size: int = 32,
) -> tuple[np.ndarray, str]:
    if len(text_a) != len(text_b):
        raise ValueError("Semantic input lists must have equal length.")

    config = DEFAULT_SEMANTIC_EVALUATORS[evaluator_key]
    model_id = config["model_id"]
    revision = revision or _hf_revision(model_id)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        model_id,
        revision=revision,
        trust_remote_code=config["trust_remote_code"],
        device=_device(),
    )

    kwargs = dict(
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    if config["adapter"] == "jina_text_matching":
        emb_a = model.encode(
            list(map(str, text_a)),
            task="text-matching",
            prompt_name="text-matching",
            **kwargs,
        )
        emb_b = model.encode(
            list(map(str, text_b)),
            task="text-matching",
            prompt_name="text-matching",
            **kwargs,
        )
    else:
        emb_a = model.encode(list(map(str, text_a)), **kwargs)
        emb_b = model.encode(list(map(str, text_b)), **kwargs)

    scores = np.sum(emb_a * emb_b, axis=1).astype(float)
    del model
    _clear_model_memory()
    return scores, revision


def _binary_metrics(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def _best_threshold(
    y_true: Sequence[int],
    scores: Sequence[float],
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    unique = np.unique(scores)
    if len(unique) == 1:
        thresholds = [unique[0] - 1e-8, unique[0] + 1e-8]
    else:
        midpoints = (unique[:-1] + unique[1:]) / 2
        thresholds = np.concatenate(
            [[unique.min() - 1e-8], midpoints, [unique.max() + 1e-8]]
        )

    rows = []
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            **_binary_metrics(y_true, pred),
        })

    report = pd.DataFrame(rows).sort_values(
        ["macro_f1", "balanced_accuracy", "mcc", "threshold"],
        ascending=[False, False, False, False],
    )
    return float(report.iloc[0]["threshold"])


def validate_semantic_evaluators(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Select evaluator + threshold on human development data only.
    Test data is used only after the development-set choice is frozen.
    """
    required = {"model_answer", "gold_answer", "consensus_label"}
    for name, df in (("dev", dev_df), ("test", test_df)):
        missing = required - set(df.columns)
        if missing:
            raise EvaluationError(f"{name} missing columns: {sorted(missing)}")

    dev = dev_df[
        dev_df["consensus_label"].isin(ANSWER_HUMAN_TO_BINARY)
    ].copy()
    test = test_df[
        test_df["consensus_label"].isin(ANSWER_HUMAN_TO_BINARY)
    ].copy()
    if dev.empty or test.empty:
        raise EvaluationError(
            "Human answer dev/test data is empty after ambiguous items "
            "are excluded."
        )

    dev_y = dev["consensus_label"].map(ANSWER_HUMAN_TO_BINARY).astype(int)
    test_y = test["consensus_label"].map(ANSWER_HUMAN_TO_BINARY).astype(int)

    dev_rows = []
    state = {}

    for key, config in DEFAULT_SEMANTIC_EVALUATORS.items():
        scores, revision = _semantic_scores(
            key, dev["model_answer"], dev["gold_answer"]
        )
        threshold = _best_threshold(dev_y, scores)
        metrics = _binary_metrics(dev_y, (scores >= threshold).astype(int))
        state[key] = {"revision": revision, "threshold": threshold}
        dev_rows.append({
            "evaluator": key,
            "model_id": config["model_id"],
            "revision": revision,
            "threshold": threshold,
            **metrics,
        })

    dev_report = pd.DataFrame(dev_rows).sort_values(
        ["macro_f1", "balanced_accuracy", "mcc", "evaluator"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    selected_key = str(dev_report.iloc[0]["evaluator"])

    test_rows = []
    for key, config in DEFAULT_SEMANTIC_EVALUATORS.items():
        scores, _ = _semantic_scores(
            key,
            test["model_answer"],
            test["gold_answer"],
            revision=state[key]["revision"],
        )
        threshold = state[key]["threshold"]
        metrics = _binary_metrics(test_y, (scores >= threshold).astype(int))
        test_rows.append({
            "evaluator": key,
            "model_id": config["model_id"],
            "revision": state[key]["revision"],
            "threshold": threshold,
            "selected_on_dev": key == selected_key,
            **metrics,
        })

    selected = state[selected_key]
    selection = {
        "evaluator_key": selected_key,
        "model_id": DEFAULT_SEMANTIC_EVALUATORS[selected_key]["model_id"],
        "revision": selected["revision"],
        "threshold": float(selected["threshold"]),
        "selection_basis": "human_annotated_development_set",
    }
    return selection, dev_report, pd.DataFrame(test_rows)


# ---------------------------------------------------------------------
# Turkish sentence segmentation
# ---------------------------------------------------------------------

_STANZA = None


_LIST_MARKER_CANDIDATE_RE = re.compile(
    r"(?<!\S)(?P<number>[1-9]\d{0,2})(?P<delimiter>[.)])(?=\s+\S)"
)
_BARE_NUMBERED_FRAGMENT_RE = re.compile(r"^\s*\d{1,3}[.)]\s*$")
_PARENTHETICAL_TRAILING_MARKER_RE = re.compile(
    r"^\s*\(.*\)\s+\d{1,3}[.)]\s*$"
)


def _previous_nonspace_character(text: str, position: int) -> str | None:
    index = position - 1
    while index >= 0 and text[index].isspace():
        index -= 1
    return text[index] if index >= 0 else None


def _first_marker_has_safe_boundary(text: str, start: int) -> bool:
    """Require a plausible list start instead of deleting arbitrary numbers."""
    prefix = text[:start]
    if not prefix.strip():
        return True

    # A line break is a strong list boundary.
    line_start = prefix.rfind("\n") + 1
    if not prefix[line_start:].strip():
        return True

    previous = _previous_nonspace_character(text, start)
    return previous in {":", ";", ".", "!", "?", ")", "]", "}"}


def _candidate_is_not_arithmetic_result(text: str, start: int) -> bool:
    """Reject markers immediately following an arithmetic/operator context."""
    previous = _previous_nonspace_character(text, start)
    return previous not in {"=", "+", "-", "*", "/", "%", "^"}


def _find_list_marker_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for match in _LIST_MARKER_CANDIDATE_RE.finditer(text):
        start, end = match.span()
        if not _candidate_is_not_arithmetic_result(text, start):
            continue
        candidates.append({
            "start": start,
            "end": end,
            "number": int(match.group("number")),
            "text": match.group(0),
        })
    return candidates


def _find_confirmed_list_marker_spans(text: str) -> list[dict[str, Any]]:
    """Find high-confidence numbered-list markers in the original text.

    Markers are detected before sentence tokenization. A candidate is removed
    only when it belongs to a consecutive list beginning with 1 and containing
    at least two items. Numeric content that merely looks like ``500.`` or
    ``0.63`` is therefore preserved.
    """
    candidates = _find_list_marker_candidates(text)

    confirmed: list[dict[str, Any]] = []
    index = 0

    while index < len(candidates):
        candidate = candidates[index]

        if candidate["number"] != 1:
            index += 1
            continue

        if not _first_marker_has_safe_boundary(text, candidate["start"]):
            index += 1
            continue

        sequence = [candidate]
        expected = 2
        cursor = index + 1

        while cursor < len(candidates):
            current = candidates[cursor]

            if current["number"] == expected:
                # Every marker needs a boundary, not only the initial 1.
                # In "1. First step. Result 2. Check.", 2 is an answer.
                if not _first_marker_has_safe_boundary(text, current["start"]):
                    break
                between = text[sequence[-1]["end"] : current["start"]]
                if not between.strip():
                    break
                sequence.append(current)
                expected += 1
                cursor += 1
                continue

            # A fresh 1 can start another independent list. Any other number
            # breaks the current consecutive sequence.
            break

        if len(sequence) >= 2:
            confirmed.extend(sequence)
            index += len(sequence)
        else:
            index += 1

    return confirmed


def _remove_spans(text: str, spans: Sequence[Mapping[str, Any]]) -> str:
    """Remove only the explicitly confirmed character ranges."""
    cleaned = text
    for span in sorted(spans, key=lambda item: int(item["start"]), reverse=True):
        start = int(span["start"])
        end = int(span["end"])
        cleaned = cleaned[:start] + cleaned[end:]
    return cleaned


def _parenthesis_balance(text: str) -> tuple[int, int]:
    """Return final depth and minimum depth while scanning parentheses."""
    depth = 0
    minimum = 0
    for character in text:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            minimum = min(minimum, depth)
    return depth, minimum


def _is_fully_parenthetical(text: str) -> bool:
    stripped = text.strip()
    if not (stripped.startswith("(") and stripped.endswith(")")):
        return False

    depth = 0
    for index, character in enumerate(stripped):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return False
            if depth == 0 and index != len(stripped) - 1:
                return False
    return depth == 0


def _repair_reasoning_segments(segments: Sequence[str]) -> list[str]:
    """Repair sentence-tokenizer splits without deleting model content.

    Parenthetical fragments spanning multiple Stanza sentences are rejoined
    with a depth counter. A segment that is entirely parenthetical is attached
    to the preceding reasoning sentence, but no punctuation or numeric content
    is deleted.
    """
    repaired: list[str] = []
    pending: list[str] = []
    pending_depth = 0

    def append_complete(text: str) -> None:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return

        if _is_fully_parenthetical(text) and repaired:
            repaired[-1] = f"{repaired[-1]} {text}".strip()
        else:
            repaired.append(text)

    for raw_segment in segments:
        text = re.sub(r"\s+", " ", str(raw_segment)).strip()
        if not text:
            continue

        depth, minimum = _parenthesis_balance(text)

        if pending:
            pending.append(text)
            pending_depth += depth

            if pending_depth <= 0:
                append_complete(" ".join(pending))
                pending = []
                pending_depth = 0
            continue

        # Only start a pending buffer when this segment opens an unmatched
        # parenthesis. A stray closing parenthesis is preserved verbatim and
        # surfaced by the audit instead of being silently deleted.
        if depth > 0 and minimum >= 0:
            pending = [text]
            pending_depth = depth
            continue

        append_complete(text)

    # Never silently discard an unmatched fragment from the model output.
    if pending:
        append_complete(" ".join(pending))

    return repaired


def _non_whitespace_content(text: str) -> str:
    """Canonical form used only to verify character preservation."""
    return re.sub(r"\s+", "", text)


def _audit_reasoning_segmentation(
    *,
    original_text: str,
    cleaned_text: str,
    segments: Sequence[str],
    marker_spans: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    joined_segments = " ".join(map(str, segments))

    expected_content = _non_whitespace_content(cleaned_text)
    observed_content = _non_whitespace_content(joined_segments)
    content_preserved = observed_content == expected_content

    bare_numbered_fragments = [
        segment
        for segment in segments
        if _BARE_NUMBERED_FRAGMENT_RE.fullmatch(str(segment))
    ]
    parenthetical_trailing_markers = [
        segment
        for segment in segments
        if _PARENTHETICAL_TRAILING_MARKER_RE.fullmatch(str(segment))
    ]

    residual_confirmed_markers = _find_confirmed_list_marker_spans(cleaned_text)

    all_candidates = _find_list_marker_candidates(original_text)
    confirmed_positions = {
        (int(span["start"]), int(span["end"])) for span in marker_spans
    }
    unconfirmed_candidates = [
        candidate
        for candidate in all_candidates
        if (int(candidate["start"]), int(candidate["end"]))
        not in confirmed_positions
    ]

    marker_contexts = [
        original_text[
            max(0, int(span["start"]) - 35) :
            min(len(original_text), int(span["end"]) + 55)
        ]
        for span in marker_spans
    ]
    unconfirmed_contexts = [
        original_text[
            max(0, int(span["start"]) - 35) :
            min(len(original_text), int(span["end"]) + 55)
        ]
        for span in unconfirmed_candidates
    ]

    incomplete_equation_fragments = [
        str(segment)
        for segment in segments
        if str(segment).rstrip().endswith("=")
    ]

    # A segment ending in "=" is a segmentation blocker only when more
    # reasoning text follows it. If the cleaned source itself ends in "=",
    # the incompleteness came from the model output and is reported as a
    # source-quality warning rather than misclassified as a parser failure.
    segmentation_incomplete_equation_fragments = [
        str(segment)
        for index, segment in enumerate(segments)
        if str(segment).rstrip().endswith("=")
        and index < len(segments) - 1
    ]
    source_incomplete_equation = bool(
        str(cleaned_text).rstrip().endswith("=")
    )

    cleaned_depth, cleaned_minimum = _parenthesis_balance(cleaned_text)
    source_parentheses_balanced = (
        cleaned_depth == 0 and cleaned_minimum >= 0
    )

    unbalanced_segments = [
        str(segment)
        for segment in segments
        if _parenthesis_balance(str(segment)) != (0, 0)
    ]
    unbalanced_segment_count = len(unbalanced_segments)

    # If the source is balanced but the final segments are not, the
    # segmentation/repair step introduced an unsafe boundary. If the source
    # itself is already unbalanced, preserve that model output and surface it
    # separately as a source-quality warning instead of silently dropping it.
    segmentation_unbalanced_fragments = (
        unbalanced_segments if source_parentheses_balanced else []
    )
    source_unbalanced_parentheses = not source_parentheses_balanced

    blocking_reasons = []
    if not content_preserved:
        blocking_reasons.append("content_not_preserved")
    if bare_numbered_fragments:
        blocking_reasons.append("bare_numbered_fragments")
    if parenthetical_trailing_markers:
        blocking_reasons.append("parenthetical_trailing_markers")
    if residual_confirmed_markers:
        blocking_reasons.append("confirmed_list_markers_remaining")
    if segmentation_incomplete_equation_fragments:
        blocking_reasons.append("incomplete_equation_segment")
    if segmentation_unbalanced_fragments:
        blocking_reasons.append("unbalanced_segment_parentheses")

    source_warning_reasons = []
    if source_incomplete_equation:
        source_warning_reasons.append("source_ends_with_equals")
    if source_unbalanced_parentheses:
        source_warning_reasons.append("source_unbalanced_parentheses")

    return {
        # Compatibility alias: both fields check preservation AFTER the
        # explicitly recorded marker removals, not whether those removals
        # were semantically correct. Review marker contexts separately.
        "content_preserved": bool(content_preserved),
        "post_cleanup_content_preserved": bool(content_preserved),
        "content_check_scope": "cleaned_text_vs_segments_ignoring_whitespace",
        "original_non_whitespace_chars": len(_non_whitespace_content(original_text)),
        "cleaned_non_whitespace_chars": len(expected_content),
        "segmented_non_whitespace_chars": len(observed_content),
        "confirmed_list_marker_count": len(marker_spans),
        "confirmed_list_markers": [str(span["text"]) for span in marker_spans],
        "confirmed_list_marker_contexts": marker_contexts,
        "unconfirmed_list_candidate_count": len(unconfirmed_candidates),
        "unconfirmed_list_candidate_contexts": unconfirmed_contexts,
        "bare_numbered_fragment_count": len(bare_numbered_fragments),
        "bare_numbered_fragments": bare_numbered_fragments,
        "parenthetical_trailing_marker_count": len(parenthetical_trailing_markers),
        "parenthetical_trailing_markers": parenthetical_trailing_markers,
        "residual_confirmed_marker_count": len(residual_confirmed_markers),
        "incomplete_equation_fragment_count": len(incomplete_equation_fragments),
        "incomplete_equation_fragments": incomplete_equation_fragments,
        "segmentation_incomplete_equation_count": len(
            segmentation_incomplete_equation_fragments
        ),
        "segmentation_incomplete_equation_fragments": (
            segmentation_incomplete_equation_fragments
        ),
        "source_incomplete_equation": source_incomplete_equation,
        "cleaned_parenthesis_depth": cleaned_depth,
        "cleaned_parenthesis_minimum": cleaned_minimum,
        "source_parentheses_balanced": source_parentheses_balanced,
        "source_unbalanced_parentheses": source_unbalanced_parentheses,
        "unbalanced_segment_count": int(unbalanced_segment_count),
        "unbalanced_segments": unbalanced_segments,
        "segmentation_unbalanced_fragment_count": len(
            segmentation_unbalanced_fragments
        ),
        "segmentation_unbalanced_fragments": segmentation_unbalanced_fragments,
        "source_warning_count": len(source_warning_reasons),
        "source_warning_reasons": source_warning_reasons,
        "blocking_issue_count": len(blocking_reasons),
        "blocking_reasons": blocking_reasons,
    }


def _get_stanza_pipeline(*, download_if_missing: bool):
    global _STANZA
    import stanza

    if _STANZA is None:
        try:
            _STANZA = stanza.Pipeline(
                lang="tr",
                processors="tokenize",
                use_gpu=False,
                verbose=False,
            )
        except Exception as exc:
            if not download_if_missing:
                raise EvaluationError(
                    "Turkish Stanza tokenizer is missing. "
                    "Run with download_if_missing=True once."
                ) from exc
            stanza.download("tr", processors="tokenize", verbose=False)
            _STANZA = stanza.Pipeline(
                lang="tr",
                processors="tokenize",
                use_gpu=False,
                verbose=False,
            )

    return _STANZA


def segment_reasoning_with_audit(
    reasoning: Any,
    *,
    download_if_missing: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """Segment one model reasoning while auditing every transformation."""
    if is_blank(reasoning):
        return [], {
            "content_preserved": True,
            "post_cleanup_content_preserved": True,
            "content_check_scope": "cleaned_text_vs_segments_ignoring_whitespace",
            "confirmed_list_marker_count": 0,
            "bare_numbered_fragment_count": 0,
            "parenthetical_trailing_marker_count": 0,
            "residual_confirmed_marker_count": 0,
            "incomplete_equation_fragment_count": 0,
            "segmentation_incomplete_equation_count": 0,
            "source_incomplete_equation": False,
            "source_parentheses_balanced": True,
            "source_unbalanced_parentheses": False,
            "unbalanced_segment_count": 0,
            "segmentation_unbalanced_fragment_count": 0,
            "source_warning_count": 0,
            "source_warning_reasons": [],
            "blocking_issue_count": 0,
            "blocking_reasons": [],
        }

    original_text = str(reasoning)
    marker_spans = _find_confirmed_list_marker_spans(original_text)
    cleaned_text = _remove_spans(original_text, marker_spans)

    pipeline = _get_stanza_pipeline(
        download_if_missing=download_if_missing,
    )
    doc = pipeline(cleaned_text)

    raw_segments = [
        sentence.text.strip()
        for sentence in doc.sentences
        if sentence.text.strip()
    ]
    segments = _repair_reasoning_segments(raw_segments)

    audit = _audit_reasoning_segmentation(
        original_text=original_text,
        cleaned_text=cleaned_text,
        segments=segments,
        marker_spans=marker_spans,
    )
    return segments, audit


def segment_reasoning(
    reasoning: Any,
    *,
    download_if_missing: bool = False,
) -> list[str]:
    segments, audit = segment_reasoning_with_audit(
        reasoning,
        download_if_missing=download_if_missing,
    )
    if audit.get("blocking_issue_count", 0):
        raise EvaluationError(
            "Unsafe reasoning segmentation: "
            f"{audit.get('blocking_reasons', [])}. "
            "Use build_reasoning_segmentation_preview to inspect the audit."
        )
    return segments


def build_reasoning_segmentation_preview(
    records: pd.DataFrame,
    *,
    download_stanza_if_missing: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create claim and audit tables without running any learned evaluator."""
    required = {
        "record_uid", "record_id", "input_type", "model_access",
        "model_label", "model_id", "category", "difficulty",
        "generation_success", "model_reasoning",
    }
    missing = required - set(records.columns)
    if missing:
        raise EvaluationError(
            f"Segmentation preview is missing columns: {sorted(missing)}"
        )

    claim_rows = []
    audit_rows = []

    valid = records[
        records["generation_success"]
        & records["model_reasoning"].apply(lambda value: not is_blank(value))
    ]

    for _, row in valid.iterrows():
        segments, audit = segment_reasoning_with_audit(
            row["model_reasoning"],
            download_if_missing=download_stanza_if_missing,
        )

        for index, segment in enumerate(segments, start=1):
            claim_rows.append({
                "record_uid": row["record_uid"],
                "record_id": row["record_id"],
                "input_type": row["input_type"],
                "model_access": row["model_access"],
                "model_label": row["model_label"],
                "model_id": row["model_id"],
                "category": row["category"],
                "difficulty": row["difficulty"],
                "claim_index": index,
                "claim_text": segment,
            })

        audit_rows.append({
            "record_uid": row["record_uid"],
            "record_id": row["record_id"],
            "input_type": row["input_type"],
            "model_access": row["model_access"],
            "model_label": row["model_label"],
            "segment_count": len(segments),
            **audit,
        })

    return pd.DataFrame(claim_rows), pd.DataFrame(audit_rows)



# ---------------------------------------------------------------------
# True premise-hypothesis NLI
# ---------------------------------------------------------------------

def _load_nli_model(
    evaluator_key: str,
    *,
    revision: str | None = None,
):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_id = DEFAULT_NLI_EVALUATORS[evaluator_key]["model_id"]
    revision = revision or _hf_revision(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision
    ).to(_device())
    model.eval()

    id2label = {
        int(i): str(label).strip().casefold()
        for i, label in model.config.id2label.items()
    }
    if set(id2label.values()) != EXPECTED_NLI_LABELS:
        raise EvaluationError(
            f"Unsafe NLI label mapping for {model_id}: {id2label}"
        )
    return tokenizer, model, id2label, revision


def _nli_limit(tokenizer, model) -> int:
    values = []
    token_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(token_limit, int) and 0 < token_limit < 100_000:
        values.append(token_limit)
    model_limit = getattr(model.config, "max_position_embeddings", None)
    if isinstance(model_limit, int) and model_limit > 0:
        values.append(model_limit)
    if not values:
        raise EvaluationError("Could not determine NLI sequence limit.")
    return min(values)


def _nli_scores(
    evaluator_key: str,
    premises: Sequence[str],
    hypotheses: Sequence[str],
    *,
    revision: str | None = None,
    batch_size: int = 16,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    if len(premises) != len(hypotheses):
        raise ValueError("NLI input lists must have equal length.")

    tokenizer, model, id2label, revision = _load_nli_model(
        evaluator_key, revision=revision
    )
    premises = list(map(str, premises))
    hypotheses = list(map(str, hypotheses))
    max_length = _nli_limit(tokenizer, model)

    lengths = [
        len(tokenizer(p, h, add_special_tokens=True, truncation=False)["input_ids"])
        for p, h in zip(premises, hypotheses)
    ]
    over_limit = sum(length > max_length for length in lengths)
    audit = {
        "model_limit": max_length,
        "pair_count": len(lengths),
        "maximum_observed": max(lengths, default=0),
        "mean_observed": float(np.mean(lengths)) if lengths else 0.0,
        "over_limit_count": over_limit,
    }
    if over_limit:
        raise LongNLIInputError(
            f"{over_limit} NLI pair(s) exceed the model limit "
            f"({max_length}). Inputs are not silently truncated."
        )

    import torch
    rows = []
    device = _device()

    for start in range(0, len(premises), batch_size):
        batch = tokenizer(
            premises[start:start + batch_size],
            hypotheses[start:start + batch_size],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            probs = torch.softmax(model(**batch).logits, dim=-1).cpu().numpy()

        for row in probs:
            scores = {
                id2label[i]: float(row[i]) for i in range(len(row))
            }
            rows.append({
                "nli_label": max(scores, key=scores.get),
                "entailment_probability": scores["entailment"],
                "neutral_probability": scores["neutral"],
                "contradiction_probability": scores["contradiction"],
            })

    del model, tokenizer
    _clear_model_memory()
    return pd.DataFrame(rows), revision, audit


def _nli_metrics(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, classification_report

    labels = ["entailment", "neutral", "contradiction"]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "entailment_f1": float(report["entailment"]["f1-score"]),
        "neutral_f1": float(report["neutral"]["f1-score"]),
        "contradiction_f1": float(report["contradiction"]["f1-score"]),
    }


def validate_nli_evaluators(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    required = {"context", "claim_text", "consensus_label"}
    for name, df in (("dev", dev_df), ("test", test_df)):
        missing = required - set(df.columns)
        if missing:
            raise EvaluationError(f"{name} missing columns: {sorted(missing)}")

    dev = dev_df[
        dev_df["consensus_label"].isin(REASONING_HUMAN_TO_NLI)
    ].copy()
    test = test_df[
        test_df["consensus_label"].isin(REASONING_HUMAN_TO_NLI)
    ].copy()
    if dev.empty or test.empty:
        raise EvaluationError("Human reasoning dev/test data is empty.")

    dev_y = dev["consensus_label"].map(REASONING_HUMAN_TO_NLI)
    test_y = test["consensus_label"].map(REASONING_HUMAN_TO_NLI)

    dev_rows = []
    state = {}

    for key, config in DEFAULT_NLI_EVALUATORS.items():
        predictions, revision, audit = _nli_scores(
            key, dev["context"], dev["claim_text"]
        )
        metrics = _nli_metrics(dev_y, predictions["nli_label"])
        state[key] = {"revision": revision, "audit": audit}
        dev_rows.append({
            "evaluator": key,
            "model_id": config["model_id"],
            "revision": revision,
            **metrics,
        })

    dev_report = pd.DataFrame(dev_rows).sort_values(
        ["macro_f1", "accuracy", "evaluator"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    selected_key = str(dev_report.iloc[0]["evaluator"])

    test_rows = []
    for key, config in DEFAULT_NLI_EVALUATORS.items():
        predictions, _, audit = _nli_scores(
            key,
            test["context"],
            test["claim_text"],
            revision=state[key]["revision"],
        )
        metrics = _nli_metrics(test_y, predictions["nli_label"])
        test_rows.append({
            "evaluator": key,
            "model_id": config["model_id"],
            "revision": state[key]["revision"],
            "selected_on_dev": key == selected_key,
            "maximum_test_input_tokens": audit["maximum_observed"],
            **metrics,
        })

    selection = {
        "evaluator_key": selected_key,
        "model_id": DEFAULT_NLI_EVALUATORS[selected_key]["model_id"],
        "revision": state[selected_key]["revision"],
        "selection_basis": "human_annotated_development_set",
    }
    return selection, dev_report, pd.DataFrame(test_rows)


# ---------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------

def _build_claims(
    records: pd.DataFrame,
    *,
    download_stanza_if_missing: bool,
) -> pd.DataFrame:
    rows = []
    valid = records[
        records["generation_success"]
        & records["model_reasoning"].apply(lambda x: not is_blank(x))
    ]

    for _, row in valid.iterrows():
        claims, segmentation_audit = segment_reasoning_with_audit(
            row["model_reasoning"],
            download_if_missing=download_stanza_if_missing,
        )
        if segmentation_audit.get("blocking_issue_count", 0):
            raise EvaluationError(
                "Unsafe reasoning segmentation for "
                f"{row['record_id']}: "
                f"{segmentation_audit.get('blocking_reasons', [])}"
            )
        for index, claim in enumerate(claims, start=1):
            rows.append({
                "record_uid": row["record_uid"],
                "record_id": row["record_id"],
                "input_type": row["input_type"],
                "model_access": row["model_access"],
                "model_label": row["model_label"],
                "model_id": row["model_id"],
                "category": row["category"],
                "difficulty": row["difficulty"],
                "claim_index": index,
                "claim_text": claim,
                "context": row["context"],
            })
    return pd.DataFrame(rows)


def run_full_evaluation(
    prepared_records: pd.DataFrame,
    selection: Mapping[str, Any],
    *,
    download_stanza_if_missing: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply an already human-validated evaluator configuration."""
    if selection.get("selection_basis") != "human_annotated_development_set":
        raise EvaluationError(
            "Full evaluation requires a selection frozen on human dev data."
        )

    evaluated = prepared_records.copy()
    semantic = selection["semantic"]
    nli = selection["nli"]

    evaluated["semantic_similarity"] = np.nan
    unresolved = evaluated["needs_semantic_evaluation"]
    subset = evaluated.loc[unresolved].copy()

    if not subset.empty:
        scores, _ = _semantic_scores(
            semantic["evaluator_key"],
            subset["model_answer"],
            subset["gold_answer"],
            revision=semantic["revision"],
        )
        subset["semantic_similarity"] = scores
        subset["answer_correct"] = (
            scores >= float(semantic["threshold"])
        ).astype(int)

        evaluated.loc[subset.index, "semantic_similarity"] = scores
        evaluated.loc[subset.index, "answer_correct"] = (
            subset["answer_correct"].astype("Int64")
        )
        evaluated.loc[subset.index, "accuracy_method"] = "validated_semantic"

    evaluated["brier_score"] = np.nan
    valid_brier = (
        evaluated["answer_correct"].notna()
        & evaluated["confidence_normalized"].notna()
    )
    evaluated.loc[valid_brier, "brier_score"] = (
        evaluated.loc[valid_brier, "confidence_normalized"].astype(float)
        - evaluated.loc[valid_brier, "answer_correct"].astype(float)
    ) ** 2

    evaluated["end_to_end_correct"] = 0
    known = evaluated["answer_correct"].notna()
    evaluated.loc[known, "end_to_end_correct"] = (
        evaluated.loc[known, "answer_correct"].astype(int)
    )

    claims = _build_claims(
        evaluated,
        download_stanza_if_missing=download_stanza_if_missing,
    )
    length_audit = None

    if not claims.empty:
        scores, _, length_audit = _nli_scores(
            nli["evaluator_key"],
            claims["context"],
            claims["claim_text"],
            revision=nli["revision"],
        )
        scores = scores.rename(columns={
            "nli_label": "context_nli_label",
            "entailment_probability": "context_entailment_probability",
            "neutral_probability": "context_neutral_probability",
            "contradiction_probability": "context_contradiction_probability",
        })
        claims = pd.concat(
            [claims.reset_index(drop=True), scores.reset_index(drop=True)],
            axis=1,
        )

        claims["supported"] = (
            claims["context_nli_label"] == "entailment"
        ).astype(float)
        claims["contradicted"] = (
            claims["context_nli_label"] == "contradiction"
        ).astype(float)
        claims["nei"] = (
            claims["context_nli_label"] == "neutral"
        ).astype(float)

        aggregate = claims.groupby("record_uid").agg(
            reasoning_claim_count=("claim_index", "size"),
            context_support_rate=("supported", "mean"),
            context_contradiction_rate=("contradicted", "mean"),
            context_nei_rate=("nei", "mean"),
            mean_context_entailment_probability=(
                "context_entailment_probability", "mean"
            ),
        ).reset_index()

        evaluated = evaluated.merge(
            aggregate,
            on="record_uid",
            how="left",
            validate="one_to_one",
        )

    audit = {
        "selection": dict(selection),
        "answer_rule_version": "conservative_rules_v1",
        "reasoning_segmentation": "stanza_tr_sentence_v7_segmentation_safety_gate",
        "context_nli_length_audit": length_audit,
        "record_count": len(evaluated),
        "generation_success_count": int(
            evaluated["generation_success"].sum()
        ),
        "evaluated_answer_count": int(
            evaluated["answer_correct"].notna().sum()
        ),
        "claim_count": len(claims),
    }
    return evaluated, claims, audit


def save_full_evaluation(
    evaluated_records: pd.DataFrame,
    claim_level: pd.DataFrame,
    manifest_df: pd.DataFrame,
    audit: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "base": output_dir / "base_evaluated.csv",
        "variants": output_dir / "variants_evaluated.csv",
        "manifest": output_dir / "source_manifest.csv",
        "metadata": output_dir / "evaluation_metadata.json",
    }

    atomic_save_csv(
        evaluated_records[evaluated_records["input_type"] == "base"].copy(),
        paths["base"],
    )
    atomic_save_csv(
        evaluated_records[evaluated_records["input_type"] == "variants"].copy(),
        paths["variants"],
    )
    atomic_save_csv(manifest_df, paths["manifest"])
    atomic_save_json(audit, paths["metadata"])

    if not claim_level.empty:
        paths["claims"] = output_dir / "claim_level_evaluation.csv"
        atomic_save_csv(claim_level, paths["claims"])

    return paths
