from sklearn.model_selection import cross_val_score
import numpy as np
class FeatureSelector2:
    def __init__(self, estimator, feature_importances, step=1):
        self.estimator = estimator
        self.feature_importances = feature_importances
        self.step = step

    def fit(self, X, y, method='sfs'):
        n_features = X.shape[1]

        if method == 'sfs':
            n_features_to_select = self._determine_n_features_to_select(X, y)
            selected_features = set()
            remaining_features = set(range(n_features))
            while len(selected_features) < n_features_to_select:
                best_feature = None
                best_score = -np.inf
                for feature in remaining_features:
                    current_features = list(selected_features) + [feature]
                    score = self._evaluate_subset(X[:, current_features], y)
                    if score > best_score:
                        best_score = score
                        best_feature = feature

                if best_feature is not None:
                    selected_features.add(best_feature)
                    remaining_features.remove(best_feature)
                else:
                    break

        elif method == 'sbs':
            n_features_to_select = self._determine_n_features_to_select(X, y)
            selected_features = set(range(n_features))
            while len(selected_features) > n_features_to_select:
                worst_feature = None
                worst_score = np.inf
                for feature in selected_features:
                    current_features = list(selected_features - {feature})
                    score = self._evaluate_subset(X[:, current_features], y)
                    if score < worst_score:
                        worst_score = score
                        worst_feature = feature

                if worst_feature is not None:
                    selected_features.remove(worst_feature)
                else:
                    break

        self.selected_features_ = list(selected_features)

    def _evaluate_subset(self, X_subset, y):
        scores = cross_val_score(self.estimator, X_subset, y, cv=5)
        return np.mean(scores)

    def _determine_n_features_to_select(self, X, y):
        feature_importance_indices = np.argsort(self.feature_importances)[::-1]
        cumulative_importance = np.cumsum(self.feature_importances[feature_importance_indices])
        threshold = 0.85 * cumulative_importance[-1]  # Select features contributing to 95% of importance
        n_features_to_select = np.argmax(cumulative_importance >= threshold) + 1
        return n_features_to_select