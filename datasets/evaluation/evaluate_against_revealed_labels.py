from pathlib import Path
import pandas as pd


def evaluate_against_revealed(
    prediction_file: str,
    output_detail_file: str | None = None,
    sep: str = ";",
    normalize_labels: bool = True,
) -> dict:
    """
    Evaluate a prediction CSV against all *_labels_revealed.csv files
    in the same folder by matching on ID.

    Expected columns in prediction and revealed files:
        ID;...;Label

    Only rows with IDs present in both prediction file and any revealed file
    are evaluated.
    """
    prediction_path = Path(prediction_file)

    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")

    folder = prediction_path.parent
    revealed_files = sorted(folder.glob("*_labels_revealed.csv"))

    if not revealed_files:
        raise FileNotFoundError(
            f"No revealed-label files found in folder: {folder}"
        )

    print(f"\nLoading prediction file: {prediction_path.name}")
    print("Using revealed files:")
    for rf in revealed_files:
        print(f" - {rf.name}")

    df_pred = pd.read_csv(prediction_path, sep=sep)

    if "ID" not in df_pred.columns or "Label" not in df_pred.columns:
        raise ValueError(
            "Prediction file must contain at least these columns: ID, Label"
        )

    df_pred = df_pred[["ID", "Label"]].copy()
    df_pred.rename(columns={"Label": "Label_pred"}, inplace=True)
    df_pred["ID"] = df_pred["ID"].astype(str).str.strip()

    if normalize_labels:
        df_pred["Label_pred"] = df_pred["Label_pred"].astype(str).str.strip().str.lower()

    # Load and combine all revealed files
    revealed_parts = []
    for rf in revealed_files:
        df_true_part = pd.read_csv(rf, sep=sep)

        if "ID" not in df_true_part.columns or "Label" not in df_true_part.columns:
            raise ValueError(
                f"Revealed file {rf.name} must contain at least these columns: ID, Label"
            )

        df_true_part = df_true_part[["ID", "Label"]].copy()
        df_true_part.rename(columns={"Label": "Label_true"}, inplace=True)
        df_true_part["ID"] = df_true_part["ID"].astype(str).str.strip()
        df_true_part["source_file"] = rf.name

        if normalize_labels:
            df_true_part["Label_true"] = (
                df_true_part["Label_true"].astype(str).str.strip().str.lower()
            )

        revealed_parts.append(df_true_part)

    df_true = pd.concat(revealed_parts, ignore_index=True)

    # Detect duplicate revealed IDs across files
    dup_ids = df_true[df_true["ID"].duplicated(keep=False)].sort_values("ID")
    if not dup_ids.empty:
        print("\nWarning: duplicate IDs found across revealed files.")
        print(dup_ids[["ID", "source_file"]].to_string(index=False))
        # Keep first occurrence to avoid exploding merge
        df_true = df_true.drop_duplicates(subset="ID", keep="first")

    # Detect duplicate prediction IDs
    pred_dups = df_pred[df_pred["ID"].duplicated(keep=False)].sort_values("ID")
    if not pred_dups.empty:
        print("\nWarning: duplicate IDs found in prediction file.")
        print(pred_dups[["ID"]].drop_duplicates().to_string(index=False))
        df_pred = df_pred.drop_duplicates(subset="ID", keep="first")

    pred_ids = set(df_pred["ID"])
    true_ids = set(df_true["ID"])

    matched_ids = pred_ids & true_ids
    missing_in_pred = true_ids - pred_ids
    extra_in_pred = pred_ids - true_ids

    print(f"\nMatched IDs:           {len(matched_ids)}")
    print(f"Missing predictions:   {len(missing_in_pred)}")
    print(f"Extra prediction IDs:  {len(extra_in_pred)}")

    df_merged = df_true.merge(df_pred, on="ID", how="inner")

    if df_merged.empty:
        raise ValueError("No matching IDs found between prediction file and revealed files.")

    df_merged["correct"] = df_merged["Label_true"] == df_merged["Label_pred"]

    accuracy = df_merged["correct"].mean()
    correct_count = int(df_merged["correct"].sum())
    wrong_count = int((~df_merged["correct"]).sum())

    print(f"\nOverall accuracy:      {accuracy:.4f}")
    print(f"Correct predictions:   {correct_count}")
    print(f"Wrong predictions:     {wrong_count}")
    print(f"Evaluated rows:        {len(df_merged)}")

    per_label_stats = (
        df_merged.groupby("Label_true")
        .agg(
            total=("correct", "size"),
            correct=("correct", "sum"),
        )
        .reset_index()
    )
    per_label_stats["wrong"] = per_label_stats["total"] - per_label_stats["correct"]
    per_label_stats["accuracy"] = per_label_stats["correct"] / per_label_stats["total"]

    print("\nPer-label stats:")
    print(per_label_stats.sort_values("Label_true").to_string(index=False))

    failures = df_merged[~df_merged["correct"]].copy()
    if not failures.empty:
        failure_breakdown = (
            failures.groupby(["Label_true", "Label_pred"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        print("\nFailure breakdown:")
        print(failure_breakdown.to_string(index=False))
    else:
        failure_breakdown = pd.DataFrame(columns=["Label_true", "Label_pred", "count"])
        print("\nFailure breakdown: no errors")

    confusion = pd.crosstab(
        df_merged["Label_true"],
        df_merged["Label_pred"],
        rownames=["True"],
        colnames=["Pred"],
    )

    print("\nConfusion matrix:")
    print(confusion)

    if output_detail_file is None:
        output_path = prediction_path.with_name(f"{prediction_path.stem}_evaluation_details.csv")
    else:
        output_path = Path(output_detail_file)

    df_merged.to_csv(output_path, sep=sep, index=False)
    print(f"\nSaved details to: {output_path}")

    return {
        "accuracy": float(accuracy),
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "evaluated_rows": len(df_merged),
        "matched_ids": len(matched_ids),
        "missing_predictions": len(missing_in_pred),
        "extra_prediction_ids": len(extra_in_pred),
        "per_label_stats": per_label_stats,
        "failure_breakdown": failure_breakdown,
        "confusion_matrix": confusion,
        "details": df_merged,
    }