import pandas as pd
import numpy as np

def clean_and_transform_data(df):
    # 1. Identify missing values
    print('Missing values before handling:')
    print(df.isnull().sum())

    # 2. Handle missing values
    median_age = df['age'].median()
    df['age'].fillna(median_age, inplace=True)
    df['skills'].fillna('Not Specified', inplace=True)

 
    df = df.drop_duplicates(subset=['applicant_id'], keep='first')

    Q1 = df['score'].quantile(0.25)
    Q3 = df['score'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df['score'] < lower_bound) | (df['score'] > upper_bound)]
    print(f'Outliers detected in score column: {len(outliers)}')

    df['application_date'] = pd.to_datetime(df['application_date'])
 
    def standardize_skills(skills):
        return [skill.strip().lower() for skill in skills.split(',')] if skills else []

    df['skills_list'] = df['skills'].apply(standardize_skills)

    df = df.sort_values(by='application_date').reset_index(drop=True)

    return df

# Sample data to demonstrate
sample_data = {
    'applicant_id': ['A001', 'A002', 'A003', 'A004', 'A005', 'A006', 'A002', 'A007', 'A008', 'A009'],
    'age': [22, 25, np.nan, 24, 23, 22, 25, 27, np.nan, 30],
    'skills': ['Python, SQL', 'Excel, PowerPoint', 'Python', 'SQL, Excel', np.nan, 'Python, Excel', 'Excel, PowerPoint', 'Python, SQL', 'SQL', 'Python'],
    'application_date': ['2025-05-01', '2025-05-02', '2025-05-02', '2025-05-03', '2025-05-04', '2025-05-05', '2025-05-02', '2025-05-06', '2025-05-07', '2025-05-08'],
    'score': [85, 90, 88, 92, 87, 85, 90, 91, 89, 93],
    'comments': ['Good', 'Excellent', 'Average', 'Good', 'Good', 'Average', 'Excellent', 'Good', 'Average', 'Excellent']
}

df = pd.DataFrame(sample_data)

# Run cleaning and transformation
cleaned_df = clean_and_transform_data(df)

# Preview cleaned and structured data
print(cleaned_df.head().to_dict(orient='records'))
