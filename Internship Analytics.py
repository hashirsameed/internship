import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample data creation for tasks
tasks_data = {
    'intern_id': [101, 101, 102, 103],
    'task_start_time': [
        pd.Timestamp('2025-05-01 09:00'),
        pd.Timestamp('2025-05-03 10:00'),
        pd.Timestamp('2025-05-02 11:00'),
        pd.Timestamp('2025-05-04 09:00')
    ],
    'task_end_time': [
        pd.Timestamp('2025-05-01 17:00'),
        pd.Timestamp('2025-05-03 15:00'),
        pd.Timestamp('2025-05-02 18:00'),
        pd.Timestamp('2025-05-04 12:00')
    ]
}
tasks_df = pd.DataFrame(tasks_data)
tasks_df.to_csv('tasks.csv', index=False)

# Sample data creation for projects
projects_data = {
    'intern_id': [101, 102, 103],
    'project_completion_date': [
        pd.Timestamp('2025-05-10'),
        pd.Timestamp('2025-05-15'),
        pd.Timestamp('2025-05-20')
    ],
    'project_quality_rating': [4.5, 4.0, 4.8]
}
projects_df = pd.DataFrame(projects_data)
projects_df.to_csv('projects.csv', index=False)

# Sample data creation for mentor feedback
feedback_data = {
    'intern_id': [101, 101, 102, 103],
    'feedback_date': [
        pd.Timestamp('2025-05-05'),
        pd.Timestamp('2025-05-12'),
        pd.Timestamp('2025-05-18'),
        pd.Timestamp('2025-05-22')
    ],
    'feedback_score': [4, 5, 3, 5]
}
feedback_df = pd.DataFrame(feedback_data)
feedback_df.to_csv('mentor_feedback.csv', index=False)

# Load datasets
tasks = pd.read_csv('tasks.csv', parse_dates=['task_start
