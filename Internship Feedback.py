import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Synthetic dataset
np.random.seed(42)  # For reproducibility
n = 100  # Number of interns

attendance = np.random.uniform(60, 100, n)          # Attendance %
tasks_submitted = np.random.randint(5, 20, n)       # Number of tasks submitted
feedback_score = np.random.uniform(1, 5, n)         # Feedback score (1-5 scale)


success_prob = (0.4 * (attendance / 100) + 
                0.4 * (tasks_submitted / 20) + 
                0.2 * (feedback_score / 5))
success = (success_prob + np.random.normal(0, 0.05, n)) > 0.6

# Create DataFrame
df = pd.DataFrame({
    'attendance': attendance,
    'tasks_submitted': tasks_submitted,
    'feedback_score': feedback_score,
    'success': success.astype(int)
})

print("Sample data:")
print(df.head())

#Classification
X = df[['attendance', 'tasks_submitted', 'feedback_score']]
y = df['success']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("\nFeature Importances:")
print(feature_importance_df)


def generate_guidance(row):
    guidance = []
    # Attendance guidance
    if row['attendance'] < 70:
        guidance.append('Improve attendance to increase success chances.')
    elif row['attendance'] < 85:
        guidance.append('Attendance is moderate; aim for higher consistency.')
    else:
        guidance.append('Attendance is strong; keep it up!')
    # Task submission guidance
    if row['tasks_submitted'] < 8:
        guidance.append('Submit more tasks to demonstrate engagement.')
    elif row['tasks_submitted'] < 15:
        guidance.append('Task submission is good; maintain or improve.')
    else:
        guidance.append('Excellent task submission rate!')
    # Feedback guidance
    if row['feedback_score'] < 2:
        guidance.append('Work on areas highlighted in feedback for improvement.')
    elif row['feedback_score'] < 4:
        guidance.append('Feedback is positive; continue improving.')
    else:
        guidance.append('Outstanding feedback; great job!')
    return ' '.join(guidance)

df['guidance'] = df.apply(generate_guidance, axis=1)

print("\nSample personalized guidance:")
print(df[['attendance', 'tasks_submitted', 'feedback_score', 'guidance']].head())
