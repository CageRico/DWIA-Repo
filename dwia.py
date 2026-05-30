"""
Dynamic Weight Integration Algorithm (DWIA)
Corresponds to Section 3 of the paper: dynamic weight computation + ensemble prediction.

Core formulas:
  P(i,m) = alpha * Accuracy + beta * Recall + gamma * F1   (composite performance score)
  w_i = sum(w_i,m * |Dm|) / sum(|Dm|)                      (sample-weighted dynamic weight)
  y = sign(sum(w_i * y_i))                                  (ensemble prediction)

Base models: LR (i=1), DT (i=2), SVM (i=3)
"""

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from scipy.special import softmax


class DynamicWeightIntegrationAlgorithm:
    """
    Credit scoring algorithm based on dynamic weight integration (DWIA).

    Parameters
    ----------
    M : int
        Number of sub-dataset partitions.
    alpha, beta, gamma : float
        Weighting coefficients for the composite performance score;
        must satisfy alpha + beta + gamma = 1.
    """

    def __init__(self, M=5, alpha=0.4, beta=0.3, gamma=0.3):
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "alpha + beta + gamma must equal 1"
        self.M = M
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Base models: LR (i=1), DT (i=2), SVM (i=3)
        self.base_models_ = [
            LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, random_state=42),
            DecisionTreeClassifier(max_depth=8, criterion="gini", random_state=42),
            SVC(kernel="rbf", probability=True, random_state=42),
        ]
        self.weights_ = None   # final dynamic weights w_i, shape=(3,)

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        1. Partition the training set into M non-overlapping subsets.
        2. Evaluate each base model's composite score P(i,m) on each subset.
        3. Normalize scores within each subset via softmax to get w(i,m).
        4. Aggregate weights by subset size to obtain final weights w_i.
        5. Retrain base models on the full training set.
        """
        n = len(X)
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)

        subsets = np.array_split(indices, self.M)

        n_models = len(self.base_models_)
        # P_matrix[i, m] = composite score of model i on subset m
        P_matrix = np.zeros((n_models, self.M))

        for m, subset_idx in enumerate(subsets):
            # Train on all other subsets, evaluate on the current one
            train_idx = np.concatenate([subsets[j] for j in range(self.M) if j != m])
            X_train_sub, y_train_sub = X[train_idx], y[train_idx]
            X_eval, y_eval = X[subset_idx], y[subset_idx]

            for i, model in enumerate(self.base_models_):
                model.fit(X_train_sub, y_train_sub)
                y_pred = model.predict(X_eval)

                acc = accuracy_score(y_eval, y_pred)
                rec = recall_score(y_eval, y_pred, zero_division=0)
                f1  = f1_score(y_eval, y_pred, zero_division=0)

                # Paper formula: P(i,m) = alpha*Acc + beta*Recall + gamma*F1
                P_matrix[i, m] = self.alpha * acc + self.beta * rec + self.gamma * f1

        # Softmax normalization along the model axis (axis=0) -> w(i,m)
        W_matrix = softmax(P_matrix, axis=0)   # shape=(n_models, M)

        # Paper formula: w_i = sum(w_i,m * |Dm|) / sum(|Dm|)
        subset_sizes = np.array([len(s) for s in subsets], dtype=float)
        self.weights_ = (W_matrix * subset_sizes).sum(axis=1) / subset_sizes.sum()

        # Retrain base models on the full dataset
        for model in self.base_models_:
            model.fit(X, y)

        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Paper formula: y = sign(sum(w_i * y_i))
        Base model outputs {0,1} are converted to {-1,+1} before weighted summation.
        """
        weighted_sum = np.zeros(len(X))
        for i, model in enumerate(self.base_models_):
            y_pred = model.predict(X).astype(float)
            y_signed = 2 * y_pred - 1   # {0,1} -> {-1,+1}
            weighted_sum += self.weights_[i] * y_signed

        # sign: >0 -> 1 (high risk), <=0 -> 0 (low risk)
        return (weighted_sum > 0).astype(int)

    def predict_proba(self, X):
        """Weighted probability output for AUC computation."""
        prob_sum = np.zeros(len(X))
        for i, model in enumerate(self.base_models_):
            prob = model.predict_proba(X)[:, 1]
            prob_sum += self.weights_[i] * prob
        prob_pos = prob_sum
        return np.column_stack([1 - prob_pos, prob_pos])

    def get_weights(self):
        """Return the dynamic weights for each base model (LR, DT, SVM)."""
        if self.weights_ is None:
            raise RuntimeError("Call fit() before accessing weights.")
        return dict(zip(["LR", "DT", "SVM"], self.weights_))
