import pandas as pd

df = pd.read_csv("Dataset/job_postings_cleaned.csv")

print("Total Job Postings:", len(df))

print("\nAvailable Job Roles:")
print(df["role"].value_counts())

print("\nUnique Job Roles:")
print(df["role"].nunique())

print("\nSkills Sample:")
print(df["skills"].head(10))