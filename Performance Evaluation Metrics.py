import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tasks = pd.read_csv('tasks.csv', parse_dates=['task_start_time', 'task_end_time'])
projects = pd.read_csv('projects.csv', parse_dates=['project_completion_date'])
feedback = pd.read_csv('mentor_feedback.csv', parse_dates=['feedback_date'])

tasks['completion_time_hours'] = (tasks['task_end_time'] - tasks['task_start_time']).dt.total_seconds() / 3600
tasks['month'] = tasks['task_end_time'].dt.to_period('M')
task_metrics = tasks.groupby(['intern_id', 'month'])['completion_time_hours'].mean().reset_index()

projects['month'] = projects['project_completion_date'].dt.to_period('M')
project_metrics = projects.groupby(['intern_id', 'month'])['project_quality_rating'].mean().reset_index()

feedback['month'] = feedback['feedback_date'].dt.to_period('M')
feedback_metrics = feedback.groupby(['intern_id', 'month'])['feedback_score'].mean().reset_index()

monthly_report = pd.merge(task_metrics, project_metrics, on=['intern_id', 'month'], how='outer')
monthly_report = pd.merge(monthly_report, feedback_metrics, on=['intern_id', 'month'], how='outer')

monthly_report.fillna({'completion_time_hours': 0, 'project_quality_rating': 0, 'feedback_score': 0}, inplace=True)

monthly_report.rename(columns={
    'completion_time_hours': 'avg_task_completion_hours',
    'project_quality_rating': 'avg_project_quality',
    'feedback_score': 'avg_mentor_feedback'
}, inplace=True)

monthly_report.to_excel('intern_performance_monthly_report.xlsx', index=False)

intern_id = 101
intern_data = monthly_report[monthly_report['intern_id'] == intern_id]

plt.figure(figsize=(10,6))
plt.plot(intern_data['month'].astype(str), intern_data['avg_task_completion_hours'], marker='o', label='Task Completion Time (hrs)')
plt.plot(intern_data['month'].astype(str), intern_data['avg_project_quality'], marker='s', label='Project Quality')
plt.plot(intern_data['month'].astype(str), intern_data['avg_mentor_feedback'], marker='^', label='Mentor Feedback')
plt.xticks(rotation=45)
plt.title(f'Performance Metrics Over Time for Intern {intern_id}')
plt.xlabel('Month')
plt.ylabel('Metric Value')
plt.legend()
plt.tight_layout()
plt.show()
