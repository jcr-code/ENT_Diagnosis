import numpy as np
from collections import Counter

#tujuan *,value=None = kalau membentuk Node class nanti harus di passing value = apa.
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None, samples=None):
        self.feature = feature #jumlah fitur
        self.threshold = threshold #
        self.left = left
        self.right = right
        self.value = value
        self.gain = None
        self.samples = samples
        
    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None #untuk mendapatkan root node nya.
        self.feature_importances_ = None  # To store feature importances
        self.best_feat=[]
        self.best_thres=[]

    # def prune_rep(self, node, X_val, y_val):
    #     if node is None or node.is_leaf_node():
    #         return

    #     if not node.left.is_leaf_node() or not node.right.is_leaf_node():
    #         self.prune_rep(node.left, X_val, y_val)
    #         self.prune_rep(node.right, X_val, y_val)

    #     # Pruning condition: check if merging the children improves error rate on validation set
    #     if node.left.is_leaf_node() and node.right.is_leaf_node():
    #         # Get the split indices for the current node
    #         left_idxs, right_idxs = self._split(X_val[:, node.feature], node.threshold)

    #         # Calculate error rate before pruning
    #         error_before = self._calculate_error_rate(X_val, y_val, left_idxs, right_idxs)

    #         # Merge the children
    #         merged_value = self._most_common_label(np.concatenate((y_val[left_idxs], y_val[right_idxs])))
    #         node.left, node.right = None, None
    #         node.value = merged_value

    #         # Calculate error rate after pruning
    #         error_after = self._calculate_error_rate(X_val, y_val, left_idxs, right_idxs)

    #         # If pruning improves error rate, keep the merge, otherwise revert
    #         if error_after <= error_before:
    #             return
    #         else:
    #             node.left = self._grow_tree(X_val[left_idxs], y_val[left_idxs])
    #             node.right = self._grow_tree(X_val[right_idxs], y_val[right_idxs])

    # def _calculate_error_rate(self, X_val, y_val, left_idxs, right_idxs):
    #     # Combine indices of left and right nodes
    #     all_idxs = np.concatenate((left_idxs, right_idxs))
    #     y_val_combined = np.concatenate((y_val[left_idxs], y_val[right_idxs]))

    #     # Predict on combined indices
    #     y_pred_combined = self.predict(X_val[all_idxs])

    #     # Calculate error rate
    #     error_rate = 1.0 - np.mean(y_pred_combined == y_val_combined)
    #     return error_rate

    def fit(self, X, y):
        n_features_total = X.shape[1]
        self.n_features = n_features_total if not self.n_features else min(n_features_total,self.n_features) #pengecekan error agar jumlah fitur tidak lebih dari jumlah fitur di X
        self.root = self._grow_tree(X, y)
        self.feature_importances_ = np.zeros(n_features_total)
        self._calculate_feature_importances(self.root, X.shape[0])

    def _grow_tree(self, X, y, depth=0):
        # print("Y = ", y, "depth = ", depth)
        n_samples, n_feats = X.shape #jumlah sampel(n_samples) dan feature(n_feats)
        n_labels = len(np.unique(y)) #jumlah target label (n_labels)

        # print("N samples : ", n_samples)
        # print("N features : ", n_feats)
        # print("N label : ", n_labels)
        # mengecek stopping criteria
        #ketika kriteria kedalaman lebih besar sama dengan, atau jumlah labels target sudah 1, atau jumlah samples kurang dari jumlah minimum
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y) #kalkulasi (kalau label tersisa hanya langsung kembalikan atau kelas yang paling banyak)
            return Node(value=leaf_value, samples=n_samples) #buat node baru sebagai leafnode dan dikembalikan

        #n_feats (jumlah fitur yang dimiliki contoh : [1, 2, 3, 4, 5])
        #n_features (jumlah fitur yang dipilih dari sebuah objek / size)
        #replace=False (by default True jika replace dapat memiliki feature yang sama, False agar tidak memiliki fitur yang sama)
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False) #jumlah fitur yang dipertimbangkan dilambangkan dengan indexnya.

        # mencari split terbaik
        best_feature, best_thresh, best_gain = self._best_split(X, y, feat_idxs)

        if (best_feature is None and best_thresh is None): #jikalau best_feature, best_tresh tidak ada gain dengan kata lain tidak menghasilkan gain hasil splitting
            leaf_value = self._most_common_label(y) #kalkulasi (kalau label tersisa hanya langsung kembalikan atau kelas yang paling banyak)
            return Node(value=leaf_value, samples=n_samples) #buat node baru sebagai leafnode dan dikembalikan

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh) #left index dan right index atas split hasil best feature dan best threshold kolom dari best threshold
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        node = Node(feature=best_feature, threshold=best_thresh, left=left, right=right, samples=n_samples)
        node.gain = best_gain
        return node

    #feat_idxs = feature indices atau index dari sebuah fitur.
    def _best_split(self, X, y, feat_idxs):
        # print("BEST SPLIT")
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column) #thresholds seluruh kolom fitur yang unik yang bisa didapat.

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain #jika gain terbesar best gain information masuk ke sini
                    split_idx = feat_idx #index dari fitur yang sedang dikerjakan
                    split_threshold = thr #thresshold saat ini
        # print("BEST GAIN:", best_gain)

        if best_gain <= 0:  # Mengecek apakah best gain tidak positive
            # print("Best gain is not positive, stopping split.")
            return None, None, None


        return split_idx, split_threshold, best_gain


    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, threshold) #lakukan proses split

        if len(left_idxs) == 0 or len(right_idxs) == 0: #kalau misal kosong kita return information gain yang didapat 0
            return 0
        
        # calculate the weighted avg. entropy of children
        n = len(y) #banyak samples yang dimiliki
        n_l, n_r = len(left_idxs), len(right_idxs) #banyak samples di node kiri dan kanan
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs]) #kalkulasi entropy kiri dan kanan
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r #weighted average dari kedua sample

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten() #index dimana leaf kiri dibuat argwhere ini mendapatkan index yang lebih kecil = threshold karena bentuknya list dalam list maka di flatten agar menjadi list saja
        right_idxs = np.argwhere(X_column > split_thresh).flatten() #index dimana leaf kanan dibuat argwhere ini mendapatkan index yang lebih besar = threshold karena bentuknya list dalam list maka di flatten agar menjadi list saja
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y) #bincount histogram untuk kemunculan fitur sebagai contoh [1,3,3,2,1] maka hasil bincount = [0,2,1,3] 0 = munucul 1x, 1 = muncul 2x, dst.
        ps = hist / len(y) #ps = p(X) dimana kemunculan fitur dibagi dengan jumlah fitur yang ada atau y
        return -np.sum([p * np.log2(p) for p in ps if p>0]) #kenapa lebih besar dari 0 karena kalau 0 ya apapun dikali 0 ya 0.

    #Contoh
    #labels_1 = ['cat', 'dog', 'cat', 'bird', 'dog', 'cat', 'dog']
    #Most common label in Example 1: cat
    #labels_2 = ['red', 'blue', 'red', 'green', 'green', 'blue', 'blue', 'red']
    #Most common label in Example 2: red karena merah muncul pertama.
    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        # print("Most Common : ",value)
        return value

    def _calculate_feature_importances(self, node, n_samples):
        if node.is_leaf_node():
            return

        left_weight = node.left.samples / n_samples
        right_weight = node.right.samples / n_samples

        self.feature_importances_[node.feature] += node.gain * (left_weight + right_weight)

        self._calculate_feature_importances(node.left, n_samples)
        self._calculate_feature_importances(node.right, n_samples)
    
    #self root karena root merupakan parent
    #lalu hasil akan dikembalikan kedalam bentuk numpy array
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def _show_tree(self, node, depth=0):
        if node is None:
            return

        indent = "  " * depth
        if node.is_leaf_node():
            print(indent + f"Leaf Node: Predicted Value = {node.value}, Samples = {node.samples}")
        print(indent + f"Split on Feature {node.feature} at Threshold {node.threshold}, Gain = {node.gain}, Samples = {node.samples}")
        self._show_tree(node.left, depth + 1)
        self._show_tree(node.right, depth + 1)
            