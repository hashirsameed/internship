# 🎯 Intern Performance Prediction ML Project

A machine learning project that predicts intern performance using behavioral, feedback, and engagement data. This project aims to help HR and team leads identify potential high-performing interns and personalize mentorship for better results.

---

## 📌 Objective

To build a supervised machine learning model that:
- Predicts intern performance as "High Performer" or "Not High Performer"
- Analyzes key performance drivers such as task completion, attendance, mentor feedback, and engagement
- Deploys a user-friendly interface using **Streamlit**

---

## 📊 Dataset Overview

The dataset includes synthetic yet realistic intern performance records with the following features:

| Column | Description |
|--------|-------------|
| `intern_id` | Unique ID |
| `department` | Intern's department (e.g., Tech, HR) |
| `interaction_level` | Level of mentor interaction |
| `attendance_rate` | % attendance |
| `task_completion_rate` | Completed tasks vs. assigned |
| `avg_feedback_score` | Mentor feedback (scale 1–5) |
| `hours_per_week` | Weekly time investment |
| `final_assessment_score` | Evaluation score (0–100) |
| `performance_label` | Target variable: High / Not High |

---

## 🧹 Preprocessing

- Handled missing values using forward fill (`fillna`)
- Encoded categorical features:
  - `LabelEncoder` in `train_model_with_encoders.ipynb`
  - `One-Hot Encoding` in `train_model_onehot.ipynb`
- Feature Scaling with `StandardScaler`

---

## 🧠 Model Training

### Models Used:
- Logistic Regression
- Random Forest Classifier
- (Optional) XGBoost

### Features Used:
- Attendance, feedback, department, hours per week, and task ratios

### Target:
- `performance_label` (High vs. Not High)

### Evaluation Metrics:
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- Feature Importance Chart

---

## 📈 Feature Importance (Sample Output)

- **Mentor Feedback Score** 🟢  
- **Task Completion Rate** 🔵  
- **Attendance Rate** 🟣  
- **Engagement Level** 🔴  

These insights help mentors focus on what drives intern performance.

---

## 📷 Screenshots

<p float="left">
  <img src="screenshots/form_input.png" width="45%"/>
  <img src="screenshots/prediction_result.png" width="45%"/>
</p>

---

## 👨‍💻 Author

**MadadAllah Bhatti**  
_Data Analyst Intern @ [Internee.pk](https://www.internee.pk/)_  
🔗 [LinkedIn](https://www.linkedin.com/in/your-link)  
💻 [GitHub](https://github.com/your-github)

---

## Acknowledgment

This project is part of my internship at **[Internee.pk](https://www.internee.pk/)** — aiming to bridge the gap between learning and real-world data projects.

