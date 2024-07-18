import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from RandomForest import RandomForest, cross_val_score
from sklearn.preprocessing import LabelEncoder
from functions import *
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_selection import SelectFromModel
import pandas as pd

#________________________________________________________________________
#FEATURE SELECTOR SFS & SBS
#________________________________________________________________________

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

#________________________________________________________________________
#FEATURE SELECTOR SFS & SBS
#________________________________________________________________________

#________________________________________________________________________
# DATA CONFIG
#________________________________________________________________________
# print(vote) 
# Load your dataset from the Excel file
# file_path = "../Progress_Proposal_THT/data_THT_transform_tanpa_mimisanFrek.xlsx" #non manipulative
# file_path = "../Progress_Proposal_THT/manipulasi_data_tht_transform.xlsx" #manipulative
# file_path = "../Progress_Proposal_THT/tht_transform_featureranking_sfs.xlsx" #SFS
# file_path = "../Progress_Proposal_THT/tht_transform_featureranking_sbs.xlsx" #SBS
file_path = "../Progress_Proposal_THT/tht_transform_featureranking_sklearn.xlsx" #feat importance
# file_path = "../Progress_Proposal_THT/tanpa_korpus_alenium.xlsx" #non korpus alenium
#________________________________________________________________________
# DATA CONFIG
#________________________________________________________________________

#________________________________________________________________________
# PRE-PROCESS
#________________________________________________________________________
df = pd.read_excel(file_path)

# Drop rows with missing values
df.dropna(axis=0, inplace=True)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode the target labels into numeric values
df['hasil_diagn_encoded'] = label_encoder.fit_transform(df['hasil_diagn'])

# Check the mapping between original labels and encoded values
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# hasil reverse value dengan key terhadap label_mapping
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# print("Label Mapping:", label_mapping)

# Now you have a new column 'hasil_diagn_encoded' containing numeric representations of the labels
# You can drop the original 'hasil_diagn' column if you don't need it anymore
df.drop(columns=['hasil_diagn'], inplace=True)

# Assuming the last column is the label/target column
X = df.drop("hasil_diagn_encoded", axis = 1).values  # Features + converting into numpy array
y = df['hasil_diagn_encoded'].values   # Labels + converting into numpy array

X_train, X_test, y_train, y_test = train_test_split( #stratified random split
    X, y, test_size=0.2, stratify=y, random_state=42 #1234
)

print(X_train)

#70:30
# X_train, X_test, y_train, y_test = train_test_split( #stratified random split
#     X, y, test_size=0.3, stratify=y, random_state=42 #1234 
# )
#60:40
# X_train, X_test, y_train, y_test = train_test_split( #stratified random split
#     X, y, test_size=0.4, stratify=y, random_state=42 #1234
# )

# for i in range(len(X_test)):
#     print("X_test" + "" + str(i), X_test[i])

# print("y Testing : ", y_test)
#________________________________________________________________________
# PRE-PROCESS
#________________________________________________________________________

#________________________________________________________________________
# TRAIN
#________________________________________________________________________
totalTree = 512
clf = RandomForest(n_trees=totalTree, min_samples_split=2)
clf.fit(X_train, y_train)

#VISUALISASI
vis_df = pd.read_excel(file_path)
# vis_df.columns = ['suhu', 'hidung tersumbat', 'pilek', 'suara serak', 'nyeri membuka mulut', 'nyeri kepala', 'vertigo', 'hidung nyeri', 
#                   'belakang hidung ganjal', 'nyeri telan tenggorokan', 'batuk', 'nyeri telinga', 'gangguan dengar', 'cairan telinga', 'leher bengkak',
#                   'mata gatal', 'telinga kemerahan', 'hidung kemerahan', 'telinga bengkak', 'hidung bengkak', 'telinga nmendengung', 'telinga gatal',
#                   'keringat dingin', 'tenggorokan kering', 'tenggorokan gatal', 'kepala berat', 'telinga berat', 'bersin', 'gendang telinga lubang',
#                   'telinga berair/kemasukan_air', 'telinga penuh/tertutup/tersumbat', 'pusing', 'mimisan', 'tenggorokan_ganjal', 
#                   'tenggorokan_panas', 'hidung_keluar_ingus', 'kekentalan_ingus', 'sesak_nafas', 'bersendawa', 'berdehem', 'kembung', 'mulut_pahit',
#                   'mulut_bau', 'mulut_kering', 'pandangan', 'mata_juling', 'nafsu_makan', 'pipi_bengkak', 'pipi_nyeri', 'badan_lemas', 'berat_badan', 
#                   'usia','mual', 'muntah', 'telinga', 'hidung', 'tenggorok', 'leher',
                #   'Diagnosis'] #Baseline
# vis_df.columns = ['pilek', 'suara serak', 'nyeri membuka mulut', 'vertigo', 'hidung nyeri', 
#                   'belakang hidung ganjal', 'nyeri telan tenggorokan', 'batuk', 'nyeri telinga', 'gangguan dengar', 'cairan telinga', 'leher bengkak',
#                   'mata gatal', 'telinga bengkak', 'hidung bengkak', 
#                   'tenggorokan kering', 'kepala berat', 'bersin',
#                   'mimisan',
#                   'hidung_keluar_ingus', 'kekentalan_ingus', 'sesak_nafas', 'berdehem',
#                   'mulut_kering', 'mata_juling', 'pipi_bengkak', 
#                   'usia','mual', 'muntah', 'telinga', 'hidung', 'tenggorok', 'leher',
#                   'Diagnosis'] #SFS
# vis_df.columns = ['nyeri membuka mulut', 'vertigo', 'hidung nyeri', 
#                   'belakang hidung ganjal', 'leher bengkak',
#                   'telinga kemerahan', 'hidung kemerahan', 'hidung bengkak', 'telinga gatal',
#                   'tenggorokan gatal', 'kepala berat', 'telinga berat', 'gendang telinga lubang',
#                   'telinga berair/kemasukan_air', 'mimisan',
#                   'tenggorokan_panas', 'kekentalan_ingus', 'bersendawa', 'berdehem', 'kembung', 'mulut_pahit',
#                   'mulut_bau', 'mulut_kering', 'pandangan', 'mata_juling', 'nafsu_makan', 'pipi_bengkak', 'pipi_nyeri', 'badan_lemas', 'berat_badan', 
#                   'mual', 'muntah',
#                   'Diagnosis'] #SBS
vis_df.columns = ['suhu', 'hidung tersumbat', 'pilek', 'suara serak', 'nyeri kepala', 'vertigo', 'hidung nyeri', 
                  'nyeri telan tenggorokan', 'batuk', 'nyeri telinga', 'gangguan dengar', 'cairan telinga',
                  'mata gatal', 'telinga bengkak', 'hidung bengkak', 'telinga nmendengung',
                  'tenggorokan kering', 'kepala berat', 'bersin',
                  'telinga penuh/tertutup/tersumbat', 'mimisan', 'hidung_keluar_ingus', 'sesak_nafas',
                  'usia','mual', 'telinga', 'hidung', 'tenggorok', 'leher',
                  'Diagnosis'] #feature_importance
feat_labels = vis_df.columns[:-1]

importances = clf.feature_importances_
# plt.figure(figsize=(12, 8))
# # Sort feature importances in descending order
# indices = np.argsort(importances)[::-1]  # Highest -> Last, Lowest -> First

# plt.xlabel('Feature importance')  # Change to xlabel as it's now the feature importance
# plt.barh(range(X_train.shape[1]),  # Change to barh for horizontal bar plot
#          importances[indices],
#          align='center')

# feat_labels = vis_df.columns[:-1]
# plt.yticks(range(X_train.shape[1]),  # Change to yticks
#            feat_labels[indices], rotation=0)  # Rotate labels horizontally

# plt.ylim([-1, X_train.shape[1]])  # Change to ylim for y-axis limit

# plt.tight_layout()
# plt.savefig('feature-importance.pdf', dpi=600)
# plt.show()

y_pred = clf.predict(X_test)
# print(y_pred)
acc =  accuracy(y_test, y_pred)
print(acc)

oob_score = clf.get_oob_score()
print("Out-of-Bag Score:", oob_score)

#________________________________________________________________________
# TRAIN
#________________________________________________________________________

#________________________________________________________________________
#EVALUATE
#________________________________________________________________________

# # Function to evaluate model performance
# def evaluate_model(X, y, params):
#     # Split the data
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Create and train the RandomForest model
#     model = RandomForest(
#         n_trees=params['n_trees'],
#         max_depth=params['max_depth'],
#         min_samples_split=params['min_samples_split']
#     )
#     model.fit(X_train, y_train)
    
#     # Predict on validation set
#     y_pred = model.predict(X_val)
    
#     # Calculate accuracy
#     acc = accuracy(y_val, y_pred)
    
#     return acc


# def random_search(X, y, param_grid, n_iter=50):
#     # Initialize variables to store the best parameters and accuracy
#     best_params = None
#     best_accuracy = 0
    
#     # Initialize a dictionary to store accuracy for each parameter combination
#     param_accuracies = {}
    
#     # Sample parameter combinations randomly
#     for _ in range(n_iter):
#         print(_)
#         params = {key: random.choice(value) for key, value in param_grid.items()}
#         acc = evaluate_model(X, y, params)
        
#         # Store the accuracy for the current parameter combination
#         param_accuracies[str(params)] = acc
        
#         if acc > best_accuracy:
#             best_accuracy = acc
#             best_params = params
    
#     # Print all parameter combinations and their corresponding accuracy
#     for param_comb, acc in param_accuracies.items():
#         print("Parameters:", param_comb)
#         print("Accuracy:", acc)
#         print()
    
#     return best_params, best_accuracy

# # Example usage
# param_grid = {
#     'n_trees': [100, 300, 500],
#     'max_depth': [10, 20, 30, 40, 50],
#     'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]
# }

# best_params, best_accuracy = random_search(X, y, param_grid, n_iter=5)

# print("Best Parameters:", best_params)
# print("Best Accuracy:", best_accuracy)
n_precision = []
n_recall = []
n_f1 = []
n_FPR = []
for key, values in label_mapping.items():
    n_precision.append(precision(y_test, y_pred, values))
    n_recall.append(recall(y_test, y_pred, values))
    n_f1.append(f1_score(y_test,y_pred, values))
    n_FPR.append(calcFPR(y_test, y_pred, values))

print("Number of Precision : ",np.mean(n_precision))
print("Number of Recall : ",np.mean(n_recall))
print("Number of F1 Scores : ",np.mean(n_f1))
# print("Macro Average F1 Score", macro_average_f1_score(y_test, y_pred))
# # print("Number of FPR : ", np.mean(n_FPR))
# print("Average Of F1 Score / Macro Score : ",macro_average_f1_score(y_test, y_pred))
# print("Average of Recall", np.mean([i for i in n_recall]))
# print("Average of FPR : ", np.mean([i for i in n_FPR]))


#________________________________________________________________________
#EVALUATE
#________________________________________________________________________
# clf.print_tree()

# print("69 : ",X_test[69])
# X_test_filt = np.array([X_test[69]])
# print(X_test_filt)
# xtest_pred = clf.predict(X_test_filt)
# print(xtest_pred)
#________________________________________________________________________
#FEATURE SELECTOR FEATURE IMPORTANCE
#________________________________________________________________________

# Create a feature selector based on feature importances
# feature_selector = SelectFromModel(clf, threshold='median')

# # Print selected feature indices
# selected_indices = feature_selector.get_support(indices=True)
# print("Selected feature indices:", selected_indices)

# # Optionally, print the selected feature names
# selected_feature_names = [vis_df.columns[i] for i in selected_indices]
# print("Selected feature names:", selected_feature_names)
#________________________________________________________________________
#FEATURE SELECTOR FEATURE IMPORTANCE
#________________________________________________________________________

#________________________________________________________________________
#FEATURE SELECTOR SFS
#________________________________________________________________________
# feature_selector = FeatureSelector2(estimator=clf , feature_importances=importances)

# feature_selector.fit(X_train, y_train, method='sfs') 

# selected_features = feature_selector.selected_features_
# print("Selected SFS features:", selected_features)

#________________________________________________________________________
#FEATURE SELECTOR SFS
#________________________________________________________________________
# feature_selector = FeatureSelector2(estimator=clf , feature_importances=importances)

# feature_selector.fit(X_train, y_train, method='sbs') 

# selected_features = feature_selector.selected_features_
# print("Selected SBS features:", selected_features)

#________________________________________________________________________
#FEATURE SELECTOR SBS
#________________________________________________________________________

#________________________________________________________________________
#CONFUSION MATRIX
#________________________________________________________________________

# matrix = confusion_matrix(y_test, y_pred, np.unique(y_test))
# plot_confusion_matrix(matrix, np.unique(y_test))
# plot_confusion_matrix_2(matrix, np.unique(y_test))

#________________________________________________________________________
#SAVEMODEL
#________________________________________________________________________  
# model_file_path = "random_forest_model_baseline.pkl"
# with open(model_file_path, 'wb') as f:
#         pickle.dump(clf, f)    

# print("Model saved successfully to:", model_file_path)
#________________________________________________________________________
#SAVE MODEL
#________________________________________________________________________
