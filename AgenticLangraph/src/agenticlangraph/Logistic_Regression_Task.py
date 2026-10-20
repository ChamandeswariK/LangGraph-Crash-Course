import pandas as pd
from sklearn.linear_model import LogisticRegression
# Classification (Predicts a group/category)
# Probability between \(0\) and \(1\) (which is then rounded to \(0\) or \(1\))
# An "S-shaped" curve (Sigmoid function)
# Accuracy, Precision, Recall, F1-Score
# Predicting whether an email is "Spam" (\(1\)) or "Not Spam" (\(0\)).

# 1. Load the training and testing datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Split features (X) and target labels (y)
X_train = train_df.drop(columns=['label'])
y_train = train_df['label']

# Prepare test features (drop label if it exists in the test file)
if 'label' in test_df.columns:
    X_test = test_df.drop(columns=['label'])
else:
    X_test = test_df.copy()

# 3. Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000) # max_iter ensures convergence on the features
model.fit(X_train, y_train)

# 4. Generate predictions on the test set
predictions = model.predict(X_test)

# 5. Build the final submission format
submission = pd.DataFrame()

# Match the IDs if present, otherwise use row index matching
if 'id' in test_df.columns:
    submission['id'] = test_df['id']
else:
    submission['id'] = test_df.index

submission['label'] = predictions

# 6. Save the required output file
submission.to_csv('submissions.csv', index=False)
print("Logistic Regression submission saved as 'submissions.csv'!")
