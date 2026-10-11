#!/usr/bin/env python3
"""Entrena y registra un clasificador para UCI Bank Marketing."""

import os

# Evita variaciones por paralelismo de las bibliotecas numericas.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import io
import json
import random
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
DEFAULT_DATA_PATH = SCRIPT_DIR / "data" / "bank-marketing.zip"
DEFAULT_EXPERIMENT = "bank-marketing-logistic-comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena regresion logistica y registra la corrida en MLflow."
    )
    parser.add_argument("--c", type=float, default=1.0, help="Inverso de regularizacion.")
    parser.add_argument(
        "--class-weight",
        choices=("balanced", "none"),
        default="balanced",
        help="Ponderacion de clases.",
    )
    parser.add_argument(
        "--max-iter", type=int, default=2000, help="Maximo de iteraciones del modelo."
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Umbral para convertir probabilidades en clases.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraccion de prueba.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    if args.c <= 0:
        parser.error("--c debe ser mayor que cero")
    if args.max_iter <= 0:
        parser.error("--max-iter debe ser mayor que cero")
    if not 0 < args.decision_threshold < 1:
        parser.error("--decision-threshold debe estar entre cero y uno")
    if not 0 < args.test_size < 1:
        parser.error("--test-size debe estar entre cero y uno")
    return args


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_data(archive_path: Path) -> pd.DataFrame:
    archive_path = archive_path.expanduser().resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        print(f"Descargando datos desde {DATA_URL}")
        urlretrieve(DATA_URL, archive_path)

    with zipfile.ZipFile(archive_path) as outer_zip:
        with zipfile.ZipFile(io.BytesIO(outer_zip.read("bank.zip"))) as inner_zip:
            with inner_zip.open("bank-full.csv") as csv_file:
                return pd.read_csv(csv_file, sep=";")


def clean_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    duplicate_count = int(raw_df.duplicated().sum())
    clean_df = raw_df.drop_duplicates().copy()
    categorical_columns = clean_df.select_dtypes(include=["object", "string"]).columns
    for column in categorical_columns:
        clean_df[column] = clean_df[column].str.strip().str.lower()

    clean_df["previously_contacted"] = clean_df["pdays"].ne(-1).astype(int)
    clean_df["pdays"] = clean_df["pdays"].replace(-1, np.nan)
    clean_df["y"] = clean_df["y"].map({"no": 0, "yes": 1})

    if clean_df["y"].isna().any():
        raise ValueError("La variable objetivo contiene valores distintos de no/yes.")
    return clean_df, duplicate_count


def split_data(
    clean_df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x = clean_df.drop(columns=["y", "duration"])
    y = clean_df["y"].astype(int)
    return train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )


def build_model(
    x_train: pd.DataFrame,
    c_value: float,
    class_weight: str,
    max_iter: int,
    seed: int,
) -> Pipeline:
    numeric_features = x_train.select_dtypes(include="number").columns.tolist()
    categorical_features = x_train.select_dtypes(exclude="number").columns.tolist()
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", RobustScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    resolved_weight = None if class_weight == "none" else class_weight
    classifier = LogisticRegression(
        C=c_value,
        class_weight=resolved_weight,
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "average_precision": average_precision_score(y_test, probabilities),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "positive_prevalence": float(y_test.mean()),
    }
    return metrics, probabilities, predictions


def save_and_log_artifacts(
    metrics: dict[str, float],
    y_test: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
) -> None:
    with tempfile.TemporaryDirectory(dir=SCRIPT_DIR) as temporary_dir:
        artifact_dir = Path(temporary_dir)
        metrics_path = artifact_dir / "metrics.json"
        report_path = artifact_dir / "classification_report.txt"
        confusion_path = artifact_dir / "confusion_matrix.png"
        pr_curve_path = artifact_dir / "precision_recall_curve.png"

        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        report_path.write_text(
            classification_report(
                y_test, predictions, target_names=["no", "yes"], zero_division=0
            ),
            encoding="utf-8",
        )

        figure, axis = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            display_labels=["no", "yes"],
            cmap="Blues",
            colorbar=False,
            ax=axis,
        )
        axis.set_title(f"Matriz de confusion (umbral={threshold:.2f})")
        figure.tight_layout()
        figure.savefig(confusion_path, dpi=150)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(6, 4))
        PrecisionRecallDisplay.from_predictions(
            y_test, probabilities, plot_chance_level=True, ax=axis
        )
        axis.set_title("Curva precision-recall")
        figure.tight_layout()
        figure.savefig(pr_curve_path, dpi=150)
        plt.close(figure)

        for path in (metrics_path, report_path, confusion_path, pr_curve_path):
            mlflow.log_artifact(str(path), artifact_path="evaluation")


def configure_mlflow(experiment_name: str) -> str:
    tracking_database = (SCRIPT_DIR / "mlflow.db").resolve()
    artifact_root = (SCRIPT_DIR / "mlruns").resolve()
    artifact_root.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_database}")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name, artifact_location=artifact_root.as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    return experiment_id


def export_comparison(experiment_id: str) -> Path:
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id], filter_string="attributes.status = 'FINISHED'"
    )
    columns = [
        "run_id",
        "tags.mlflow.runName",
        "params.C",
        "params.class_weight",
        "params.max_iter",
        "params.decision_threshold",
        "params.seed",
        "metrics.average_precision",
        "metrics.roc_auc",
        "metrics.balanced_accuracy",
        "metrics.precision",
        "metrics.recall",
        "metrics.f1",
    ]
    comparison = runs.reindex(columns=columns).sort_values(
        "metrics.average_precision", ascending=False
    )
    output_path = SCRIPT_DIR / "resultados_corridas.csv"
    comparison.to_csv(output_path, index=False, float_format="%.8f")
    return output_path


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)
    raw_df = load_data(args.data_path)
    clean_df, duplicate_count = clean_data(raw_df)
    x_train, x_test, y_train, y_test = split_data(
        clean_df, args.test_size, args.seed
    )
    model = build_model(
        x_train, args.c, args.class_weight, args.max_iter, args.seed
    )
    experiment_id = configure_mlflow(args.experiment_name)
    run_name = args.run_name or (
        f"C={args.c:g}_weight={args.class_weight}_iter={args.max_iter}"
    )

    parameters = {
        "C": args.c,
        "class_weight": args.class_weight,
        "max_iter": args.max_iter,
        "solver": "lbfgs",
        "decision_threshold": args.decision_threshold,
        "seed": args.seed,
        "test_size": args.test_size,
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "duplicates_removed": duplicate_count,
        "dropped_leakage_feature": "duration",
    }

    with mlflow.start_run(run_name=run_name) as run:
        model.fit(x_train, y_train)
        metrics, probabilities, predictions = evaluate_model(
            model, x_test, y_test, args.decision_threshold
        )
        mlflow.log_params(parameters)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        mlflow.set_tags(
            {
                "dataset": "UCI Bank Marketing (ID 222)",
                "task": "binary_classification",
                "source_notebook": "TC5061_Notebook_Baseline_VargasAnuar.ipynb",
            }
        )
        save_and_log_artifacts(
            metrics, y_test, probabilities, predictions, args.decision_threshold
        )

        input_example = x_train.head(5)
        signature = infer_signature(input_example, model.predict(input_example))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        run_id = run.info.run_id

    comparison_path = export_comparison(experiment_id)
    print(f"Run ID: {run_id}")
    print(f"Experimento: {args.experiment_name}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Comparacion: {comparison_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
