from DecisionTree import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=500, max_depth=50, min_samples_split=9, mtry=None): #jumlah tree dinaikan ada juga perbedaan. #100 sangat akurat mungkin ada data yang tak terlihat
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=None
        self.trees = []
        self.oob_score = None #oob score value
        self.feature_importances_ = None  # To store feature importances
        self.mtry = mtry

    def fit(self, X, y):
        # X, y = np.array(X), np.array(y)  # Convert to NumPy arrays
        self.trees = []
        n_features = X.shape[1]
        self.mtry = int(np.sqrt(n_features)) if self.mtry is None else self.mtry
        # self.mtry = int(np.log2(n_features)) if self.mtry is None else self.mtry
        self.feature_importances_ = np.zeros(n_features)
        oob_predictions = []
        oob_indexes = []  # Keep track of out-of-bag samples indexes

        for _ in range(self.n_trees): #bentuk trees sebanyak n_trees
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_features=self.mtry)
            X_sample, y_sample, oob_idx = self._bootstrap_samples(X, y) #sample hasil dari bootstrap sample yang terpilih
            # print("Y sample : ", y_sample)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            oob_indexes.append(oob_idx)
           
            # Predict for out-of-bag samples
            oob_pred = tree.predict(X[oob_idx])
            oob_predictions.append(oob_pred)
            self.feature_importances_ += tree.feature_importances_

        self.feature_importances_ /= self.n_trees
        # Check for zero sum and avoid division by zero
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
        oob_indexes_flat = np.concatenate(oob_indexes)
        oob_predictions_flat = np.concatenate(oob_predictions)
        y_oob = y[oob_indexes_flat]
        self.oob_score = np.mean(oob_predictions_flat == y_oob)
        
        # print('in')
        # # Prune each decision tree using Reduced Error Pruning
        # for tree in self.trees:
        #     tree.prune_rep(tree.root, X[oob_indexes_flat], y_oob)
        
    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0] #jumlah baris atau jumlah sampel
        idxs = np.random.choice(n_samples, n_samples, replace=True) #yang disini sampel yang dipilih bisa sama maka diset replace = True.
        oob_idxs = np.array(list(set(range(n_samples)) - set(idxs))) # keep track out of bag value
        return X[idxs], y[idxs], oob_idxs

    def _most_common_label(self, y):
        # print(y)
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    #predictions = [tree1[sample1, sample2, sample3, sample4],[sample1, sample2, sample3, sample4],[sample1, sample2, sample3, sample4]] 1 list per tree
    #tree_preds = [sample1[tree1,tree2,tree3,tree4], sample2[tree1,tree2,tree3,tree4], sample3[tree1,tree2,tree3,tree4]] dari sample yang sama hasil prediksi tree yang berbeda, makanya axes nya ditukar
    #predictions = hasil dari mean / modus atas prediction
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees]) #untuk setiap tree kita predict tree tersebut jadi kita akan mendapatkan list of list.
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
    
    def voteResult(self, X):
        vote = np.array([tree.predict(X) for tree in self.trees]) #untuk setiap tree kita predict tree tersebut jadi kita akan mendapatkan list of list.
        tree_vote = np.swapaxes(vote, 0, 1) #setiap prediksi ditukar dari awalnya [[1,0,0,1], [0,1,0,1]] hasil sample berbeda dalam satu tree menjadi hasil tiap prediksi satu sample terhadap seluruh tree
        return tree_vote

    def get_oob_score(self):
        if self.oob_score is None:
            raise ValueError("The model has not been trained yet.")
        return self.oob_score
    
    def print_tree(self):
        for tree in self.trees:
            tree._show_tree(tree.root)

def cross_val_score(estimator, X, y, cv=5): 
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    fold_sizes = np.full(cv, X.shape[0] // cv, dtype=int)
    fold_sizes[:X.shape[0] % cv] += 1
    current = 0
    scores = []

    for i, fold_size in enumerate(fold_sizes):
        start, stop = current, current + fold_size
        val_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        # print(f"Fitting model for fold {i+1}")
        estimator.fit(X_train, y_train)
        # print(f"Model fitted for fold {i+1} in {time() - start_time:.2f} seconds")
        predictions = estimator.predict(X_val)
        # print(f"Predictions completed for fold {i+1}")
        score = np.mean(predictions == y_val)
        scores.append(score)
        # print(f"Fold {i+1} score: {score}")
        current = stop


    return np.array(scores)