"""
Human-validation utilities for the Turkish LLM reasoning benchmark.

This module prepares blinded annotation sets and reconstructs human consensus
labels. It does not choose automatic evaluators and does not run full
evaluation.

Answer labels:
    equivalent / not_equivalent / ambiguous

Reasoning labels:
    supported / contradicted / nei

`nei` means "not enough information".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .evaluation import (
    EvaluationError,
    atomic_save_csv,
    is_blank,
    segment_reasoning,
    stable_id,
)


ANSWER_LABELS = {"equivalent", "not_equivalent", "ambiguous"}
REASONING_LABELS = {"supported", "contradicted", "nei"}
ANNOTATION_SALT = "turkish_reasoning_human_validation_v1"


# ---------------------------------------------------------------------
# Folders and reproducible sampling
# ---------------------------------------------------------------------

def _paths(evaluation_root: str | Path) -> dict[str, Path]:
    root = Path(evaluation_root) / "human_validation"
    paths = {
        "templates": root / "templates",
        "keys": root / "private_keys",
        "annotations": root / "annotations",
        "adjudication": root / "adjudication",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _positive_int(value: int | None, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _balanced_sample(
    dataframe: pd.DataFrame,
    n: int,
    *,
    strata: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    n = _positive_int(n, "sample size")
    if dataframe.empty:
        raise EvaluationError("Cannot sample from an empty dataframe.")

    missing = [c for c in strata if c not in dataframe.columns]
    if missing:
        raise EvaluationError(f"Sampling columns missing: {missing}")

    if len(dataframe) <= n:
        return dataframe.sample(frac=1, random_state=seed).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    groups = list(
        dataframe.groupby(list(strata), dropna=False, sort=True)
    )

    selected = []
    if n >= len(groups):
        quota = max(1, n // len(groups))
        for _, group in groups:
            take = min(quota, len(group))
            selected.extend(
                rng.choice(group.index.to_numpy(), size=take, replace=False)
                .tolist()
            )
    else:
        for group_index in rng.choice(
            np.arange(len(groups)), size=n, replace=False
        ):
            group = groups[int(group_index)][1]
            selected.extend(
                rng.choice(group.index.to_numpy(), size=1, replace=False)
                .tolist()
            )

    selected = list(dict.fromkeys(selected))
    if len(selected) > n:
        selected = rng.choice(selected, size=n, replace=False).tolist()

    if len(selected) < n:
        remaining = dataframe[~dataframe.index.isin(selected)]
        extra = rng.choice(
            remaining.index.to_numpy(),
            size=n - len(selected),
            replace=False,
        )
        selected.extend(extra.tolist())

    return (
        dataframe.loc[selected]
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def _save_new(
    dataframe: pd.DataFrame,
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing validation file: {path}"
        )
    atomic_save_csv(dataframe, path)
    return path


# ---------------------------------------------------------------------
# Blinded templates
# ---------------------------------------------------------------------

def _annotation_id(
    record_uid: str,
    task: str,
    split: str,
    suffix: str = "",
) -> str:
    return stable_id(
        ANNOTATION_SALT, task, split, record_uid, suffix, length=20
    )


def _answer_template(
    sample: pd.DataFrame,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    blinded, key = [], []

    for _, row in sample.iterrows():
        aid = _annotation_id(str(row["record_uid"]), "answer", split)

        blinded.append({
            "annotation_id": aid,
            "context": row["context"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "model_answer": row["model_answer"],
            "human_label": "",
            "annotator_id": "",
            "notes": "",
        })

        info = {
            "annotation_id": aid,
            "record_uid": row["record_uid"],
            "input_type": row["input_type"],
            "model_access": row["model_access"],
            "model_label": row["model_label"],
            "model_id": row["model_id"],
            "run_name": row["run_name"],
            "category": row["category"],
            "difficulty": row["difficulty"],
        }
        for optional in (
            "record_id", "variant_type", "morphological_feature",
            "base_question_id",
        ):
            if optional in sample.columns:
                info[optional] = row.get(optional)
        key.append(info)

    return pd.DataFrame(blinded), pd.DataFrame(key)


def _reasoning_template(
    sample: pd.DataFrame,
    split: str,
    *,
    download_stanza_if_missing: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    blinded, key = [], []

    for _, row in sample.iterrows():
        claims = segment_reasoning(
            row["model_reasoning"],
            download_if_missing=download_stanza_if_missing,
        )

        for claim_index, claim in enumerate(claims, start=1):
            aid = _annotation_id(
                str(row["record_uid"]),
                "reasoning",
                split,
                str(claim_index),
            )

            blinded.append({
                "annotation_id": aid,
                "context": row["context"],
                "question": row["question"],
                "claim_text": claim,
                "human_label": "",
                "annotator_id": "",
                "notes": "",
            })

            info = {
                "annotation_id": aid,
                "record_uid": row["record_uid"],
                "claim_index": claim_index,
                "claim_text": claim,
                "input_type": row["input_type"],
                "model_access": row["model_access"],
                "model_label": row["model_label"],
                "model_id": row["model_id"],
                "run_name": row["run_name"],
                "category": row["category"],
                "difficulty": row["difficulty"],
            }
            for optional in (
                "record_id", "variant_type", "morphological_feature",
                "base_question_id",
            ):
                if optional in sample.columns:
                    info[optional] = row.get(optional)
            key.append(info)

    return pd.DataFrame(blinded), pd.DataFrame(key)


# ---------------------------------------------------------------------
# Pilot and formal dev/test preparation
# ---------------------------------------------------------------------

def prepare_pilot_sets(
    records_df: pd.DataFrame,
    evaluation_root: str | Path,
    *,
    answer_size: int = 50,
    reasoning_output_size: int = 20,
    random_seed: int = 20260926,
    overwrite: bool = False,
    download_stanza_if_missing: bool = False,
) -> dict[str, Any]:
    paths = _paths(evaluation_root)

    answer_pool = records_df[
        records_df["needs_semantic_evaluation"]
    ].copy()
    reasoning_pool = records_df[
        records_df["generation_success"]
        & records_df["model_reasoning"].apply(lambda x: not is_blank(x))
    ].copy()

    answer_sample = _balanced_sample(
        answer_pool,
        answer_size,
        strata=("input_type", "model_label"),
        seed=random_seed,
    )
    reasoning_sample = _balanced_sample(
        reasoning_pool,
        reasoning_output_size,
        strata=("input_type", "model_label"),
        seed=random_seed + 1,
    )

    answer_blinded, answer_key = _answer_template(answer_sample, "pilot")
    reasoning_blinded, reasoning_key = _reasoning_template(
        reasoning_sample,
        "pilot",
        download_stanza_if_missing=download_stanza_if_missing,
    )

    files = {
        "answer_template": _save_new(
            answer_blinded,
            paths["templates"] / "answer_pilot_blinded.csv",
            overwrite=overwrite,
        ),
        "answer_key": _save_new(
            answer_key,
            paths["keys"] / "answer_pilot_key.csv",
            overwrite=overwrite,
        ),
        "reasoning_template": _save_new(
            reasoning_blinded,
            paths["templates"] / "reasoning_pilot_blinded.csv",
            overwrite=overwrite,
        ),
        "reasoning_key": _save_new(
            reasoning_key,
            paths["keys"] / "reasoning_pilot_key.csv",
            overwrite=overwrite,
        ),
    }

    return {
        "files": files,
        "answer_item_count": len(answer_blinded),
        "reasoning_output_count": len(reasoning_sample),
        "reasoning_claim_count": len(reasoning_blinded),
    }


def _pilot_record_uids(paths: dict[str, Path]) -> set[str]:
    record_uids = set()
    for filename in ("answer_pilot_key.csv", "reasoning_pilot_key.csv"):
        path = paths["keys"] / filename
        if path.exists():
            key = pd.read_csv(path)
            record_uids.update(key["record_uid"].astype(str))
    return record_uids


def prepare_validation_sets(
    records_df: pd.DataFrame,
    evaluation_root: str | Path,
    *,
    answer_dev_size: int,
    answer_test_size: int,
    reasoning_dev_outputs: int,
    reasoning_test_outputs: int,
    random_seed: int = 20260926,
    overwrite: bool = False,
    download_stanza_if_missing: bool = False,
) -> dict[str, Any]:
    paths = _paths(evaluation_root)

    for name, value in {
        "answer_dev_size": answer_dev_size,
        "answer_test_size": answer_test_size,
        "reasoning_dev_outputs": reasoning_dev_outputs,
        "reasoning_test_outputs": reasoning_test_outputs,
    }.items():
        _positive_int(value, name)

    excluded = _pilot_record_uids(paths)

    answer_pool = records_df[
        records_df["needs_semantic_evaluation"]
        & ~records_df["record_uid"].astype(str).isin(excluded)
    ].copy()

    if len(answer_pool) < answer_dev_size + answer_test_size:
        raise EvaluationError(
            "Not enough unresolved answers for requested dev/test sizes."
        )

    answer_dev = _balanced_sample(
        answer_pool,
        answer_dev_size,
        strata=("input_type", "model_label"),
        seed=random_seed + 10,
    )
    answer_test_pool = answer_pool[
        ~answer_pool["record_uid"].isin(answer_dev["record_uid"])
    ]
    answer_test = _balanced_sample(
        answer_test_pool,
        answer_test_size,
        strata=("input_type", "model_label"),
        seed=random_seed + 11,
    )

    reasoning_pool = records_df[
        records_df["generation_success"]
        & records_df["model_reasoning"].apply(lambda x: not is_blank(x))
        & ~records_df["record_uid"].astype(str).isin(excluded)
    ].copy()

    if len(reasoning_pool) < reasoning_dev_outputs + reasoning_test_outputs:
        raise EvaluationError(
            "Not enough reasoning outputs for requested dev/test sizes."
        )

    reasoning_dev = _balanced_sample(
        reasoning_pool,
        reasoning_dev_outputs,
        strata=("input_type", "model_label"),
        seed=random_seed + 20,
    )
    reasoning_test_pool = reasoning_pool[
        ~reasoning_pool["record_uid"].isin(reasoning_dev["record_uid"])
    ]
    reasoning_test = _balanced_sample(
        reasoning_test_pool,
        reasoning_test_outputs,
        strata=("input_type", "model_label"),
        seed=random_seed + 21,
    )

    sets = {
        "answer_dev": (*_answer_template(answer_dev, "dev"), len(answer_dev)),
        "answer_test": (*_answer_template(answer_test, "test"), len(answer_test)),
        "reasoning_dev": (
            *_reasoning_template(
                reasoning_dev,
                "dev",
                download_stanza_if_missing=download_stanza_if_missing,
            ),
            len(reasoning_dev),
        ),
        "reasoning_test": (
            *_reasoning_template(
                reasoning_test,
                "test",
                download_stanza_if_missing=download_stanza_if_missing,
            ),
            len(reasoning_test),
        ),
    }

    result_files, counts = {}, {}
    for name, (blinded, key, output_count) in sets.items():
        result_files[name] = {
            "template": _save_new(
                blinded,
                paths["templates"] / f"{name}_blinded.csv",
                overwrite=overwrite,
            ),
            "key": _save_new(
                key,
                paths["keys"] / f"{name}_key.csv",
                overwrite=overwrite,
            ),
        }
        counts[name] = {
            "sampled_outputs": output_count,
            "annotation_items": len(blinded),
        }

    return {
        "files": result_files,
        "counts": counts,
        "pilot_record_uids_excluded": len(excluded),
    }


# ---------------------------------------------------------------------
# Independent annotator files
# ---------------------------------------------------------------------

def create_annotator_copy(
    template_path: str | Path,
    annotation_dir: str | Path,
    annotator_id: str,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Make one independent copy of a blinded template for one annotator.
    """
    template_path = Path(template_path)
    annotator_id = str(annotator_id).strip()
    if not annotator_id:
        raise ValueError("annotator_id cannot be blank.")

    dataframe = pd.read_csv(template_path)
    dataframe["annotator_id"] = annotator_id

    annotation_dir = Path(annotation_dir)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    destination = (
        annotation_dir / f"{template_path.stem}__{annotator_id}.csv"
    )
    return _save_new(dataframe, destination, overwrite=overwrite)


# ---------------------------------------------------------------------
# Load annotations, agreement and adjudication
# ---------------------------------------------------------------------

def _allowed_labels(task: str) -> set[str]:
    if task == "answer":
        return ANSWER_LABELS
    if task == "reasoning":
        return REASONING_LABELS
    raise ValueError("task must be 'answer' or 'reasoning'.")


def _load_annotations(
    annotation_dir: Path,
    template_path: Path,
    *,
    task: str,
) -> pd.DataFrame:
    files = sorted(annotation_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No completed annotation CSV files found in {annotation_dir}"
        )

    expected = set(
        pd.read_csv(template_path)["annotation_id"].astype(str)
    )
    allowed = _allowed_labels(task)
    frames = []

    for path in files:
        df = pd.read_csv(path)
        required = {"annotation_id", "annotator_id", "human_label"}
        missing = required - set(df.columns)
        if missing:
            raise EvaluationError(f"{path.name} missing: {sorted(missing)}")

        df = df[list(required)].copy()
        for column in required:
            df[column] = df[column].fillna("").astype(str).str.strip()
        df["human_label"] = df["human_label"].str.casefold()

        if (df["annotator_id"] == "").any():
            raise EvaluationError(f"Blank annotator_id in {path.name}.")
        if df["annotator_id"].nunique() != 1:
            raise EvaluationError(
                f"{path.name} must contain exactly one annotator_id."
            )
        if (df["human_label"] == "").any():
            raise EvaluationError(f"Unlabeled items remain in {path.name}.")
        if set(df["human_label"]) - allowed:
            raise EvaluationError(
                f"Invalid labels in {path.name}: "
                f"{sorted(set(df['human_label']) - allowed)}"
            )
        if set(df["annotation_id"]) != expected:
            raise EvaluationError(
                f"{path.name} does not match the expected annotation template."
            )
        if df["annotation_id"].duplicated().any():
            raise EvaluationError(f"Duplicate annotation_id in {path.name}.")

        frames.append(df)

    annotations = pd.concat(frames, ignore_index=True)
    if annotations.duplicated(
        ["annotation_id", "annotator_id"]
    ).any():
        raise EvaluationError(
            "An annotator labeled the same annotation_id more than once."
        )
    return annotations


def _agreement(annotations: pd.DataFrame) -> dict[str, Any]:
    annotators = sorted(annotations["annotator_id"].unique())
    if len(annotators) < 2:
        raise EvaluationError("At least two annotators are required.")

    pivot = annotations.pivot(
        index="annotation_id",
        columns="annotator_id",
        values="human_label",
    ).dropna()

    unanimous = pivot.apply(lambda row: row.nunique() == 1, axis=1)
    result = {
        "annotator_count": len(annotators),
        "items_labeled_by_all": len(pivot),
        "unanimous_agreement": float(unanimous.mean()),
        "disagreement_count": int((~unanimous).sum()),
        "cohen_kappa": None,
    }

    if len(annotators) == 2:
        from sklearn.metrics import cohen_kappa_score
        result["cohen_kappa"] = float(
            cohen_kappa_score(
                pivot[annotators[0]],
                pivot[annotators[1]],
            )
        )
    return result


def _consensus(
    annotations: pd.DataFrame,
    adjudication_path: Path,
    *,
    task: str,
    minimum_annotators: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts = annotations.groupby("annotation_id")["annotator_id"].nunique()
    if (counts < minimum_annotators).any():
        raise EvaluationError(
            f"Every item needs at least {minimum_annotators} annotators."
        )

    agreement = _agreement(annotations)
    rows = []

    for annotation_id, group in annotations.groupby("annotation_id"):
        labels = sorted(set(group["human_label"]))
        rows.append({
            "annotation_id": annotation_id,
            "consensus_label": labels[0] if len(labels) == 1 else None,
            "annotator_labels": "|".join(labels),
        })

    consensus = pd.DataFrame(rows)
    unresolved = consensus["consensus_label"].isna()

    if unresolved.any():
        adjudication_path.parent.mkdir(parents=True, exist_ok=True)

        if not adjudication_path.exists():
            template = consensus.loc[
                unresolved, ["annotation_id", "annotator_labels"]
            ].copy()
            template["adjudicated_label"] = ""
            template["adjudicator_id"] = ""
            template["notes"] = ""
            atomic_save_csv(template, adjudication_path)
            raise EvaluationError(
                "Annotator disagreements need adjudication. "
                f"Template created at {adjudication_path}"
            )

        adjudication = pd.read_csv(adjudication_path)
        adjudication["adjudicated_label"] = (
            adjudication["adjudicated_label"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
        )
        allowed = _allowed_labels(task)
        mapping = dict(zip(
            adjudication["annotation_id"].astype(str),
            adjudication["adjudicated_label"],
        ))

        for index in consensus.index[unresolved]:
            aid = str(consensus.at[index, "annotation_id"])
            label = mapping.get(aid, "")
            if not label:
                raise EvaluationError(
                    f"Adjudication incomplete for {aid}."
                )
            if label not in allowed:
                raise EvaluationError(
                    f"Invalid adjudicated label {label!r} for {aid}."
                )
            consensus.at[index, "consensus_label"] = label

    return consensus, agreement


# ---------------------------------------------------------------------
# Final human-labeled validation dataframe
# ---------------------------------------------------------------------

def load_labeled_validation_set(
    records_df: pd.DataFrame,
    evaluation_root: str | Path,
    *,
    task: str,
    split: str,
    minimum_annotators: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Reconnect completed blinded annotations to their private key and source
    model-output record.
    """
    if task not in {"answer", "reasoning"}:
        raise ValueError("task must be 'answer' or 'reasoning'.")
    if split not in {"pilot", "dev", "test"}:
        raise ValueError("split must be 'pilot', 'dev' or 'test'.")

    paths = _paths(evaluation_root)
    template = paths["templates"] / f"{task}_{split}_blinded.csv"
    key_path = paths["keys"] / f"{task}_{split}_key.csv"
    annotation_dir = paths["annotations"] / task / split
    adjudication = (
        paths["adjudication"] / f"{task}_{split}_adjudication.csv"
    )

    if not template.exists() or not key_path.exists():
        raise FileNotFoundError(
            f"Validation template/key missing for {task}/{split}."
        )

    annotations = _load_annotations(
        annotation_dir, template, task=task
    )
    consensus, agreement = _consensus(
        annotations,
        adjudication,
        task=task,
        minimum_annotators=minimum_annotators,
    )

    key = pd.read_csv(key_path)
    labeled = key.merge(
        consensus,
        on="annotation_id",
        how="inner",
        validate="one_to_one",
    )

    source_columns = [
        c for c in (
            "record_uid", "context", "question", "gold_answer",
            "model_answer", "model_reasoning", "gold_evidence",
        )
        if c in records_df.columns
    ]
    source = records_df[source_columns].drop_duplicates("record_uid")
    labeled = labeled.merge(
        source,
        on="record_uid",
        how="left",
        validate="many_to_one",
    )

    return labeled, agreement
