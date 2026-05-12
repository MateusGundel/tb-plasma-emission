"""Classificadores baseline: Random Forest, SVM RBF, SVM Linear."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def get_baseline_models(config: dict) -> dict[str, Pipeline]:
    """Retorna dicionário de pipelines (scaler + classificador)."""
    baseline = config["baseline"]

    models = {
        "RF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=baseline["random_forest"]["n_estimators"],
                class_weight=baseline["random_forest"]["class_weight"],
                random_state=baseline["random_forest"]["random_state"],
                n_jobs=-1,
            )),
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel=baseline["svm_rbf"]["kernel"],
                class_weight=baseline["svm_rbf"]["class_weight"],
                random_state=baseline["svm_rbf"]["random_state"],
                probability=True,
            )),
        ]),
        "SVM_Linear": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel=baseline["svm_linear"]["kernel"],
                class_weight=baseline["svm_linear"]["class_weight"],
                random_state=baseline["svm_linear"]["random_state"],
                probability=True,
            )),
        ]),
    }
    return models
