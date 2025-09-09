import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numbers
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, \
    StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from catboost import CatBoostClassifier
from sklearn.utils.validation import check_X_y
from imblearn.base import BaseSampler

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['Times New Roman']

class LOFKNNCSENN(BaseSampler):
    _parameter_constraints = {
        "lof_threshold": [numbers.Real],
        "knn_threshold": [numbers.Real],
        "n_neighbors": [numbers.Integral],
        "min_majority_samples": [numbers.Integral],
        "min_minority_samples": [numbers.Integral],
        "smote_ratio": [numbers.Real],
        "random_state": [numbers.Integral, type(None)]
    }

    def __init__(self, lof_threshold=1.5, knn_threshold=0.7, n_neighbors=5,
                 min_majority_samples=10, min_minority_samples=5, smote_ratio=0.6, random_state=None):
        super().__init__()
        self.lof_threshold = lof_threshold
        self.knn_threshold = knn_threshold
        self.n_neighbors = n_neighbors
        self.min_majority_samples = min_majority_samples
        self.min_minority_samples = min_minority_samples
        self.smote_ratio = smote_ratio
        self.random_state = random_state
        self._sampling_type = 'bypass'

    def _fit_resample(self, X, y):
        X, y = check_X_y(
            X, y,
            accept_sparse=['csr', 'csc'],
            dtype=None,
            ensure_2d=True,
            allow_nd=False,
            ensure_all_finite=True,
            multi_output=False,
            y_numeric=False
        )

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(
                f"Warning: Only one class ({unique_classes[0]}) exists in the original data; sampling cannot be performed")
            return X, y

        class_counts = np.bincount(y)
        if len(class_counts) < 2:
            print(f"Warning: Only one class exists in the data; sampling cannot be performed")
            return X, y

        minority_class = np.argmin(class_counts)
        majority_class = 1 - minority_class if len(class_counts) == 2 else np.argmax(class_counts)

        X_minority = X[y == minority_class]
        X_majority = X[y == majority_class]

        print(f"Original data: Majority class samples={len(X_majority)}, Minority class samples={len(X_minority)}")

        if len(X_minority) < 2:
            print("Warning: Insufficient minority class samples; SMOTE oversampling cannot be performed")
            return X, y

        if len(X_majority) < self.min_majority_samples:
            print(
                f"Warning: Insufficient majority class samples; only {len(X_majority)} samples exist, requiring at least {self.min_majority_samples} samples")
            self.min_majority_samples = max(5, len(X_majority))

        if len(X_majority) > self.n_neighbors:
            print("Step 1: Remove outliers using LOF")
            lof = LocalOutlierFactor(
                n_neighbors=min(self.n_neighbors, len(X_majority) - 1),
                contamination='auto'
            )
            lof_scores = -lof.fit_predict(X_majority)

            majority_indices = np.argsort(lof_scores)[:max(self.min_majority_samples, len(X_majority) // 2)]
            majority_lof_filtered = X_majority[majority_indices]
            print(f"After LOF filtering: Majority class samples={len(majority_lof_filtered)}")
        else:
            majority_lof_filtered = X_majority
            print(f"LOF not executed (insufficient samples): Majority class samples={len(majority_lof_filtered)}")

        print("Step 2: SMOTE oversampling for minority class")

        target_minority_size = max(
            len(X_minority),
            int(len(majority_lof_filtered) * self.smote_ratio)
        )

        try:
            X_combined = np.vstack([X_minority, majority_lof_filtered])
            y_combined = np.hstack([
                np.full(len(X_minority), minority_class),
                np.full(len(majority_lof_filtered), majority_class)
            ])

            unique_classes = np.unique(y_combined)
            if len(unique_classes) < 2:
                raise ValueError(f"Only one class exists in the combined data: {unique_classes[0]}")

            smote = SMOTE(
                sampling_strategy={minority_class: target_minority_size},
                random_state=self.random_state
            )

            X_resampled, y_resampled = smote.fit_resample(X_combined, y_combined)

            X_minority_resampled = X_resampled[y_resampled == minority_class]
            y_minority_resampled = np.full(len(X_minority_resampled), minority_class)

            print(f"After SMOTE oversampling: Minority class samples={len(X_minority_resampled)}")
        except Exception as e:
            print(f"Warning: SMOTE oversampling failed: {e}; using original minority class samples")
            X_minority_resampled = X_minority
            y_minority_resampled = np.full(len(X_minority), minority_class)

        print("Step 3: KNN filtering for majority and minority classes separately")

        if len(majority_lof_filtered) > self.n_neighbors:
            print("Perform KNN filtering for majority class")
            knn_majority = KNeighborsClassifier(n_neighbors=min(self.n_neighbors, len(majority_lof_filtered) - 1))
            knn_majority.fit(
                np.vstack([X_minority_resampled, majority_lof_filtered]),
                np.hstack([y_minority_resampled, np.full(len(majority_lof_filtered), majority_class)])
            )

            majority_probs = knn_majority.predict_proba(majority_lof_filtered)
            majority_confidence = np.max(majority_probs, axis=1)

            confidence_indices = np.argsort(-majority_confidence)[
                                 :max(self.min_majority_samples, len(majority_lof_filtered) // 2)]
            majority_knn_filtered = majority_lof_filtered[confidence_indices]
            print(f"After majority class KNN filtering: Number of samples={len(majority_knn_filtered)}")
        else:
            majority_knn_filtered = majority_lof_filtered
            print(
                f"Majority class KNN not executed (insufficient samples): Number of samples={len(majority_knn_filtered)}")

        if len(X_minority_resampled) > self.n_neighbors:
            print("Perform KNN filtering for minority class")
            knn_minority = KNeighborsClassifier(n_neighbors=min(self.n_neighbors, len(X_minority_resampled) - 1))
            knn_minority.fit(
                np.vstack([X_minority_resampled, majority_knn_filtered]),
                np.hstack([y_minority_resampled, np.full(len(majority_knn_filtered), majority_class)])
            )

            minority_probs = knn_minority.predict_proba(X_minority_resampled)
            minority_confidence = np.max(minority_probs, axis=1)

            confidence_indices = np.argsort(-minority_confidence)[
                                 :max(self.min_minority_samples, len(X_minority_resampled) // 2)]
            minority_knn_filtered = X_minority_resampled[confidence_indices]
            print(f"After minority class KNN filtering: Number of samples={len(minority_knn_filtered)}")
        else:
            minority_knn_filtered = X_minority_resampled
            print(
                f"Minority class KNN not executed (insufficient samples): Number of samples={len(minority_knn_filtered)}")

        if len(majority_knn_filtered) > 0 and len(minority_knn_filtered) > 0:
            print("Step 4: ENN for boundary noise cleaning")
            enn_neighbors = min(self.n_neighbors, len(majority_knn_filtered) - 1, len(minority_knn_filtered) - 1)
            if enn_neighbors < 1:
                enn_neighbors = 1

            enn = EditedNearestNeighbours(n_neighbors=enn_neighbors)

            try:
                X_combined = np.vstack([minority_knn_filtered, majority_knn_filtered])
                y_combined = np.hstack([np.full(len(minority_knn_filtered), minority_class),
                                        np.full(len(majority_knn_filtered), majority_class)])

                X_resampled, y_resampled = enn.fit_resample(X_combined, y_combined)

                if len(np.unique(y_resampled)) < 2:
                    print("Warning: Only one class remains after ENN processing; returning previous data")
                    X_resampled = X_combined
                    y_resampled = y_combined
                else:
                    print(
                        f"After ENN cleaning: Majority class samples={np.sum(y_resampled == majority_class)}, Minority class samples={np.sum(y_resampled == minority_class)}")
            except Exception as e:
                print(f"Warning: ENN processing failed: {e}; returning previous data")
                X_resampled = X_combined
                y_resampled = y_combined
        else:
            print("Warning: Insufficient samples; ENN cleaning cannot be performed")
            X_resampled = np.vstack([minority_knn_filtered, majority_knn_filtered])
            y_resampled = np.hstack([np.full(len(minority_knn_filtered), minority_class),
                                     np.full(len(majority_knn_filtered), majority_class)])

        unique_classes = np.unique(y_resampled)
        if len(unique_classes) < 2:
            print(f"Warning: Only one class ({unique_classes[0]}) remains after processing; returning original data")
            return X, y

        print(
            f"Final result: Majority class samples={np.sum(y_resampled == majority_class)}, Minority class samples={np.sum(y_resampled == minority_class)}")

        return X_resampled, y_resampled

    def fit_resample(self, X, y):
        return self._fit_resample(X, y)


def select_best_base_models(X, y, candidate_models, n_best=5):
    print(f"Start adaptive model selection, selecting top {n_best} best models from {len(candidate_models)} candidate models...")

    model_scores = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in candidate_models:
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
            mean_score = scores.mean()
            model_scores.append((name, mean_score, model))
            print(f"{name}: f1 = {mean_score:.4f} ± {scores.std():.4f}")
        except Exception as e:
            print(f"Model {name} evaluation failed: {e}")
            continue

    model_scores.sort(key=lambda x: x[1], reverse=True)

    best_models = [(name, model) for name, score, model in model_scores[:n_best]]

    print("\nSelected best models:")
    for name, score, _ in model_scores[:n_best]:
        print(f"{name}: f1 = {score:.4f}")

    return best_models


def build_simplified_stacked_ensemble(best_base_models):
    return StackingClassifier(
        estimators=best_base_models,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        passthrough=True
    )


# Specify your Excel file path
file_path = 'your Excel file path.xlsx'

try:
    df = pd.read_excel(file_path)
    print(f"Successfully read data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Dataset column names: {list(df.columns)}")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if X.isnull().any().any() or y.isnull().any():
        print("Warning: Data contains missing values; data cleaning is recommended first")

    class_distribution = y.value_counts(normalize=True)
    print(f"Class distribution:\n{class_distribution}")

    minority_ratio = min(class_distribution)
    if minority_ratio < 0.1:
        print("Note: Severe class imbalance exists (minority class proportion < 10%)")

except Exception as e:
    print(f"Failed to read data: {e}")
    print("Please check if the file path is correct and the file format is a valid Excel file")

test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
)

train_classes, train_counts = np.unique(y_train, return_counts=True)
test_classes, test_counts = np.unique(y_test, return_counts=True)
print(f"Training set class distribution: {dict(zip(train_classes, train_counts))}")
print(f"Test set class distribution: {dict(zip(test_classes, test_counts))}")

min_train_samples = min(train_counts)
print(f"Number of minority class samples in training set: {min_train_samples}")

if min_train_samples < 2:
    print("Error: Insufficient minority class samples in training set; SMOTE oversampling cannot be performed")
    print("Please ensure there are at least two minority class samples in the dataset, or consider other methods")
    exit()

negative_count = np.sum(y == 0)
positive_count = np.sum(y == 1)
scale_pos_weight = negative_count / positive_count
class_weight = {0: 1, 1: negative_count / positive_count}
print(f"Class weights: {class_weight}")

# ---------------------- Base model library ----------------------
base_models = [
    ('rf',
     RandomForestClassifier(n_estimators=800, max_depth=6, max_features='sqrt', min_samples_split=5, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=300, max_depth=15, min_samples_split=5, random_state=42)),
    ('histgbm', HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)),
    ('lightgbm', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                    scale_pos_weight=scale_pos_weight, random_state=42)),
    ('catboost',
     CatBoostClassifier(n_estimators=1000, learning_rate=0.1, depth=8, random_state=42, l2_leaf_reg=0, verbose=0)),
    ('xgboost', xgb.XGBClassifier(n_estimators=1000, learning_rate=0.001, min_child_weight=1, max_depth=6, gamma=0.1,
                                  scale_pos_weight=scale_pos_weight, random_state=42)),
    ('ridge', RidgeClassifier(class_weight='balanced', random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='auto'))
]

# ---------------------- Model training and ROC curve plotting ----------------------
plt.figure(figsize=(12, 8))
plt.plot([0, 1], [0, 1], 'k--', label='Random guess (AUC=0.5)')
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']

# Train and evaluate all stacked models with n_best from 1 to 8
for n_best in range(1, 9):
    print(f"\n===== Training stacked model with n_best={n_best} =====")


    best_base_models = select_best_base_models(X_train, y_train, base_models, n_best=n_best)
    stacked_model = build_simplified_stacked_ensemble(best_base_models)
    pipelineSampler = ImbPipeline([
        ('sampler', LOFKNNCSENN(
            lof_threshold=1.5,
            knn_threshold=0.7,
            n_neighbors=5,
            min_majority_samples=10,
            min_minority_samples=5,
            smote_ratio=0.6,
            random_state=42
        )),
        ('classifier', stacked_model)
    ])

    print("Start model training...")
    pipelineSampler.fit(X_train, y_train)

    print("Model evaluation:")
    y_proba = pipelineSampler.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.plot(fpr, tpr, lw=2, color=colors[n_best - 1],
             label=f'meta{n_best} (AUC = {+auc:.4f})')

    # Output other evaluation metrics
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred = (y_proba > optimal_threshold).astype(int)
    print(classification_report(y_test, y_pred, digits=4))
    print(f"AUC: {auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    g_mean = np.sqrt(sensitivity * specificity)
    print(f"G-mean: {g_mean:.4f}")

plt.xlabel('False positive rate', fontsize=18)
plt.ylabel('True positive rate', fontsize=18)
plt.legend(loc="lower right", fontsize=18)
plt.grid(True)

plt.tight_layout()
plt.savefig('stacking_roc_comparison.png', dpi=500)
plt.show()