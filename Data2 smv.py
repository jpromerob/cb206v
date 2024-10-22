import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

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

for i, col in enumerate(columns):

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Region', y=np.log2(df[col]), data=df, hue='Region', palette=None, legend=False)
    plt.title('Boxplot of log2transformed by Distinct Genome Region')
    #plt.savefig(os.path.join('boxplot_images', f'{col}_box_plot.png'), dpi=300)

plt.tight_layout()  # Adjust layout
#plt.show()
print(df.info())
print(df.shape)

# Filter rows where Region is 'promoter' or 'enhancer'
df = df[df['Region'].isin(['promoter', 'enhancer'])]
print("\nFiltered DataFrame:")
print(df.head())
print(df.shape)


from sklearn.model_selection import train_test_split
# Split the data (70% training, 80% testing)
train_dat, test_dat = train_test_split(df, test_size=0.2, random_state=42)

# Select columns data and scale them to standardize
from sklearn.preprocessing import StandardScaler

# Scale the selected columns in train and test data
scaler = StandardScaler()

# Scale the selected columns in train and test data
train_scale = scaler.fit_transform(train_dat[['ATACpeakScore', 'K562_POLR2A_ENCSR000BMR', 'K562_POLR2A_ENCSR000EHL', 'K562_POLR2A_ENCSR000FAJ', 'K562_POLR2A_ENCSR388QZF',
                                              'K562_POLR2A_ENCSR388QZF', 'K562_POLR2AphosphoS2_ENCSR000EGF', 'K562_POLR2AphosphoS2_ENCSR000EHF',
                                              'K562_POLR2AphosphoS5_ENCSR000BKR', 'K562_POLR2G_ENCSR283ZRI', 'K562_E2F1_ENCSR720HUL', 'K562_E2F4_ENCSR000EWL',
                                              'K562_E2F6_ENCSR000BLI', 'K562_E2F6_ENCSR000EWJ', 'K562_E2F8_ENCSR953DVM', 'K562_E4F1_ENCSR731LHZ', 'K562_H3K4me1_ENCSR000EWC',
                                              'K562_H3K4me2_ENCSR000AKT', 'K562_H3K4me3_ENCSR000AKU', 'K562_H3K4me3_ENCSR000DWD', 'K562_H3K4me3_ENCSR000EWA',
                                              'K562_H3K4me3_ENCSR668LDD']])
test_scale = scaler.transform(test_dat[['ATACpeakScore', 'K562_POLR2A_ENCSR000BMR', 'K562_POLR2A_ENCSR000EHL', 'K562_POLR2A_ENCSR000FAJ', 'K562_POLR2A_ENCSR388QZF',
                                        'K562_POLR2A_ENCSR388QZF', 'K562_POLR2AphosphoS2_ENCSR000EGF', 'K562_POLR2AphosphoS2_ENCSR000EHF',
                                        'K562_POLR2AphosphoS5_ENCSR000BKR', 'K562_POLR2G_ENCSR283ZRI', 'K562_E2F1_ENCSR720HUL', 'K562_E2F4_ENCSR000EWL',
                                        'K562_E2F6_ENCSR000BLI', 'K562_E2F6_ENCSR000EWJ', 'K562_E2F8_ENCSR953DVM', 'K562_E4F1_ENCSR731LHZ', 'K562_H3K4me1_ENCSR000EWC',
                                        'K562_H3K4me2_ENCSR000AKT', 'K562_H3K4me3_ENCSR000AKU', 'K562_H3K4me3_ENCSR000DWD', 'K562_H3K4me3_ENCSR000EWA',
                                        'K562_H3K4me3_ENCSR668LDD']])
# View the scaled test data
print(test_scale[:5])

# To scale all datasets, uncomment and run the lines below

train_scale = scaler.fit_transform(train_dat.iloc[:, 5:27])  # Columns 6 to 29 (zero-indexed)
test_scale = scaler.transform(test_dat.iloc[:, 5:27])   # Columns 6 to 29 (zero-indexed)
# View the scaled test data
print(test_scale[:5])

# split the data
train_dat, test_dat = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Region'])
print("\nTraining Data:")
print(train_dat.head())
print("\nTesting Data:")
print(test_dat.head())

# separate features and target
X_train = train_dat.drop(columns=['Region'])
Y_train = train_dat['Region']
X_test = test_dat.drop(columns=['Region'])
Y_test = test_dat['Region']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(Y_train)
X_test_scaled = scaler.transform(Y_test)

# Train the SVM model
svm_model = SVC(kernel='linear', random_state=10000)
svm_model.fit(X_train, Y_train)

# Initialize SVM classifier with RBF kernel
svm = SVC(kernel='rbf', C=1.0, gamma='scale')

# Print the SVM model details
print("\nSVM Model:")
print(svm_model)


# Fit the model
svm.fit(X_train, Y_train)

# Make predictions
y_pred = svm.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(Y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


'''
# Convert categorical target to numeric
y_train_codes, y_train_labels = pd.factorize('Y_train')                                            )
y_test_codes, y_test_labels = pd.factorize('Y_test')

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
svm_model = SVC(kernel='linear', random_state=10000)
svm_model.fit(X_train_scaled, y_train_codes)


# Initialize the SVM classifier with RBF kernel
svm = SVC(kernel='rbf', C=1.0, gamma='scale')

# Print the SVM model details
print("\nSVM Model:")
print(svm_model)

# Fit the model
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

'''



