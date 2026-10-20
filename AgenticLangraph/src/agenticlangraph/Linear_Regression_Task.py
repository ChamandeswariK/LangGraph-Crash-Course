import pandas as pd
from sklearn.linear_model import LinearRegression

# Regression (Predicts an absolute value)
# Continuous numerical value (e.g., \(15.4\), \(\$250,000\), \(-5\))
# Predicting how much a house will sell for based on square footage.

# 1. Load the data
# Make sure 'train.csv' and 'test.csv' are in the same folder as your notebook
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Separate features (X) and the target label (y)
# Drop the 'label' column to get features, keep it for y
X_train = train_df.drop(columns=['label'])
y_train = train_df['label']

# If your test set has an 'id' or 'label' column, remove it for prediction
# (Keep 'id' saved safely if needed for submission)
if 'label' in test_df.columns:
    X_test = test_df.drop(columns=['label'])
else:
    X_test = test_df.copy()

# 3. Initialize and train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predict on the test data
predictions = model.predict(X_test)

# 5. Create the submission file
submission = pd.DataFrame()

# Check if an 'id' column exists to match format, otherwise generate row index IDs
if 'id' in test_df.columns:
    submission['id'] = test_df['id']
else:
    submission['id'] = test_df.index

submission['label'] = predictions

# Save to csv exactly as requested in deliverables
submission.to_csv('submissions.csv', index=False)
print("Regression submission file saved successfully as 'submissions.csv'!")
