import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("DataNew.xlsx")
print("Initial DataFrame:")
print(df.head())  # Print the first 6 lines
print(df.shape)

columns = ['ATACpeakScore', 'K562_POLR2A_ENCSR000BMR', 'K562_POLR2A_ENCSR000EHL', 'K562_POLR2A_ENCSR000FAJ', 'K562_POLR2A_ENCSR388QZF',
           'K562_POLR2A_ENCSR388QZF', 'K562_POLR2AphosphoS2_ENCSR000EGF', 'K562_POLR2AphosphoS2_ENCSR000EHF',
           'K562_POLR2AphosphoS5_ENCSR000BKR', 'K562_POLR2G_ENCSR283ZRI', 'K562_E2F1_ENCSR720HUL', 'K562_E2F4_ENCSR000EWL',
           'K562_E2F6_ENCSR000BLI', 'K562_E2F6_ENCSR000EWJ', 'K562_E2F8_ENCSR953DVM', 'K562_E4F1_ENCSR731LHZ', 'K562_H3K4me1_ENCSR000EWC',
           'K562_H3K4me2_ENCSR000AKT', 'K562_H3K4me3_ENCSR000AKU', 'K562_H3K4me3_ENCSR000DWD', 'K562_H3K4me3_ENCSR000EWA',
           'K562_H3K4me3_ENCSR668LDD']
# boxplot (without log2 transformation)
# Loop through the columns to create boxplots
# Create the 'images' directory if it doesn't exist
os.makedirs('boxplot_images', exist_ok=True)

# Filter rows by selected column values

filtered_df = df[((df['ATACpeakScore'] > 20) & (df['ATACpeakScore'] > 2000) | (df['Region'] == 'promoter')) |
                 (df['Region'] == 'enhancer') | (df['Region'] == 'geneBody')]

df = filtered_df.copy()


for i, col in enumerate(columns):

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Region', y=np.log2(df[col]), data=df, hue='Region', palette=None, legend=False)
    plt.title('Boxplot of log2transformed by Distinct Genome Region')
    plt.savefig(os.path.join('boxplot_images', f'{col}_box_plot.png'), dpi=300)

plt.tight_layout()  # Adjust layout
#plt.show()
print(df.info())
print(df.shape)


from sklearn.model_selection import train_test_split
# Split the data (70% training, 30% testing)
train_dat, test_dat = train_test_split(df, test_size=0.25, random_state=42, stratify=df['Region'])
# Check the dimensions of the training data
print("\nTraining Data:")
print(train_dat.head())
print("\nTesting Data:")
print(test_dat.head())


# Select columns data and scale them to standardize

from sklearn.preprocessing import StandardScaler
# Initialize the scaler
scaler = StandardScaler()
# Scale the selected columns in train and test data
train_scale = scaler.fit_transform(train_dat[['ATACpeakScore', 'K562_POLR2A_ENCSR000BMR', 'K562_E2F1_ENCSR720HUL', 'K562_H3K4me1_ENCSR000EWC']]) #
test_scale = scaler.transform(test_dat[['ATACpeakScore', 'K562_POLR2A_ENCSR000BMR', 'K562_E2F1_ENCSR720HUL', 'K562_H3K4me1_ENCSR000EWC']])       #
# View the scaled test data
print(test_scale[:5])

# To scale all datasets, uncomment and run the lines below

train_scale = scaler.fit_transform(train_dat.iloc[:, 5:29])  # Columns 6 to 29 (zero-indexed)
test_scale = scaler.transform(test_dat.iloc[:, 5:29])   # Columns 6 to 29 (zero-indexed)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Initialize the k-NN classifier with k=1
classifier_knn = KNeighborsClassifier(n_neighbors=1, weights="uniform", algorithm="auto", p=2, metric="minkowski")
# Fit the model with the scaled training data and the class labels
classifier_knn.fit(train_scale, train_dat['Region'])
# Predict the classes for the test data
predictions = classifier_knn.predict(test_scale)
cm = confusion_matrix(test_dat['Region'], predictions)
print(predictions[:5])
print(test_dat['Region'].head(10))


# Compute the confusion matrix
cm = confusion_matrix(test_dat['Region'], predictions)
print(cm)
actual_labels = test_dat['Region'].unique()
# Visualize the confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=actual_labels,  # Use actual labels for x-axis
            yticklabels=actual_labels)  # Use actual labels for y-axis
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# Calculate the misclassification error
misClassError = np.mean(predictions != test_dat['Region'])
# Calculate and print the accuracy
accuracy = 1 - misClassError
print(f'Accuracy = {accuracy:.4f}')

# labels for class
#labels = ['geneBody', 'untranscribed', 'TW', 'promoter', 'enhancer', 'CPS', 'divergent']

labels = range(cm.shape[0])  # Assuming labels are indexed from 0 to 6

# Initialize lists to hold precision and recall

precision = []
recall = []
print("Shape of confusion matrix:", cm.shape)
# calculate precision and recall for each class
for i in range(len(labels)):
    TP = cm[i, i]  # True Positives for class i
    FP = cm[:, i].sum() - TP  # False Positives for class i
    FN = cm[i, :].sum() - TP  # False Negatives for class i

    # Debugging: Print TP, FP, FN
    print(f"Class {i}: TP = {TP}, FP = {FP}, FN = {FN}")

    # Ensure TP and FP are scalars before division
    if isinstance(TP, (np.ndarray, list)):
        TP = TP.item()  # Convert to scalar if it's an array
    if isinstance(FP, (np.ndarray, list)):
        FP = FP.item()
    if isinstance(FN, (np.ndarray, list)):
        FN = FN.item()

    # Calculate precision and recall
    precision_value = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_value = TP / (TP + FN) if (TP + FN) > 0 else 0

    precision.append(precision_value)
    recall.append(recall_value)


print(precision)
print(recall)

# print precision and recall for each class
for i in range(len(labels)):

    print(f"class:{labels}[i] - Precision:{precision[i]:.4f}, Recall:{recall[i]:.4f}")