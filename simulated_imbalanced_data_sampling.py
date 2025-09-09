import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
import seaborn as sns

plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


class LOFKNNCSENN:
    def __init__(self, lof_threshold=1.5, knn_threshold=0.7, n_neighbors=5,
                 min_majority_samples=10, min_minority_samples=5, smote_ratio=0.6, random_state=None):
        self.lof_threshold = lof_threshold
        self.knn_threshold = knn_threshold
        self.n_neighbors = n_neighbors
        self.min_majority_samples = min_majority_samples
        self.min_minority_samples = min_minority_samples
        self.smote_ratio = smote_ratio
        self.random_state = random_state

    def fit_resample(self, X, y):
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"Warning: Only one class ({unique_classes[0]}) exists in the original data; sampling cannot be performed")
            return X, y

        minority_class = 1 if np.sum(y == 1) < np.sum(y == 0) else 0
        majority_class = 1 - minority_class

        X_minority = X[y == minority_class]
        X_majority = X[y == majority_class]

        print(f"Original data: Majority class samples={len(X_majority)}, Minority class samples={len(X_minority)}")

        if len(X_minority) < 2:
            print("Warning: Insufficient minority class samples; SMOTE oversampling cannot be performed")
            return X, y

        if len(X_majority) < self.min_majority_samples:
            print(f"Warning: Insufficient majority class samples; only {len(X_majority)} samples exist, requiring at least {self.min_majority_samples} samples")
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
            print(f"Majority class KNN not executed (insufficient samples): Number of samples={len(majority_knn_filtered)}")

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
            print(f"Minority class KNN not executed (insufficient samples): Number of samples={len(minority_knn_filtered)}")

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


def generate_imbalanced_data(n_samples=1000, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=2,
        n_redundant=10,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],
        random_state=random_state
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def visualize_sampling_methods(X, y):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    methods = {
        "Original data": (X, y),
        "SMOTE": SMOTE(random_state=42).fit_resample(X, y),
        "ADASYN": ADASYN(random_state=42).fit_resample(X, y),
        "SMOTEENN": SMOTEENN(random_state=42).fit_resample(X, y),
        "SMOTETomek": SMOTETomek(random_state=42).fit_resample(X, y),
        "LOF-KNN-CSENN": LOFKNNCSENN(random_state=42).fit_resample(X, y)
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (name, (X_resampled, y_resampled)) in enumerate(methods.items()):
        X_resampled_pca = pca.transform(X_resampled)

        class_ratio = np.bincount(y_resampled)
        ratio_text = f"Class ratio: {class_ratio[0]}:{class_ratio[1]}" if len(
            class_ratio) > 1 else f"Class: {class_ratio[0]}"

        scatter = sns.scatterplot(
            x=X_resampled_pca[:, 0],
            y=X_resampled_pca[:, 1],
            hue=y_resampled,
            palette=['blue', 'red'],
            alpha=0.6,
            s=30,
            ax=axes[i]
        )

        handles, labels = scatter.get_legend_handles_labels()
        if len(handles) == 2:
            axes[i].legend(handles, ['Majority class', 'Minority class'])
        else:
            axes[i].legend(handles, labels)

        axes[i].set_title(f"{name}")
        axes[i].set_xlabel("Dimension1")
        axes[i].set_ylabel("Dimension2")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y = generate_imbalanced_data(n_samples=3000, random_state=42)

    visualize_sampling_methods(X, y)