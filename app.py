import pandas as pd
import os

# Create dataset folder
os.makedirs("Dataset", exist_ok=True)

# Resume Dataset
resumes = {
    "Name": ["Arun", "Priya", "Kumar", "Swathi", "Divya"],
    "Skills": [
        "Python Machine Learning SQL",
        "Python Data Analysis Excel",
        "Java HTML CSS",
        "Python Machine Learning Deep Learning",
        "HTML CSS JavaScript"
    ],
    "Education": ["BTech", "BSc", "BTech", "BTech", "BCA"]
}

resume_df = pd.DataFrame(resumes)

# Job Dataset
jobs = {
    "Job_Role": [
        "Machine Learning Engineer",
        "Data Analyst",
        "Python Developer",
        "Frontend Developer"
    ],
    "Required_Skills": [
        "Python Machine Learning Deep Learning",
        "Python SQL Data Analysis Excel",
        "Python SQL Programming",
        "HTML CSS JavaScript"
    ]
}

job_df = pd.DataFrame(jobs)

# Save CSV files
resume_df.to_csv("Dataset/resumes.csv", index=False)
job_df.to_csv("Dataset/jobs.csv", index=False)

print("Resume Dataset Created Successfully!")
print("Job Dataset Created Successfully!")

# Load Resume Dataset
resume_data = pd.read_csv("Dataset/resumes.csv")

# Load Job Dataset
job_data = pd.read_csv("Dataset/jobs.csv")

# Display Resume Data
print("\nRESUME DATASET")
print(resume_data)

# Display Job Data
print("\nJOB DATASET")
print(job_data)

# Dataset Information
print("\nDataset Shape:", resume_data.shape)
print("\nMissing Values:")
print(resume_data.isnull().sum())

# Data Cleaning

# Remove duplicate records
resume_data = resume_data.drop_duplicates()
job_data = job_data.drop_duplicates()

# Handle missing values
resume_data = resume_data.fillna("")
job_data = job_data.fillna("")

# Convert text into lowercase
resume_data["Skills"] = resume_data["Skills"].str.lower()
job_data["Required_Skills"] = job_data["Required_Skills"].str.lower()

print("Data Cleaning Completed Successfully!")

print("\nCleaned Resume Data:")
print(resume_data)

print("\nCleaned Job Data:")
print(job_data)

from sklearn.feature_extraction.text import TfidfVectorizer

# Extract resume skills
resume_skills = resume_data["Skills"].tolist()

# Extract job skills
job_skills = job_data["Required_Skills"].tolist()

# Combine both datasets
all_text = resume_skills + job_skills

# Apply TF-IDF
vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(all_text)

print("TF-IDF Vectorization Completed!")

print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity

# Separate Resume and Job vectors
resume_vectors = tfidf_matrix[:len(resume_data)]
job_vectors = tfidf_matrix[len(resume_data):]

# Calculate Similarity
similarity = cosine_similarity(resume_vectors, job_vectors)

# Display Matching Results
for i in range(len(resume_data)):

    best_job = similarity[i].argmax()
    score = similarity[i][best_job] * 100

    print("\nCandidate:", resume_data.iloc[i]["Name"])
    print("Recommended Job:", job_data.iloc[best_job]["Job_Role"])
    print("Matching Score:", round(score, 2), "%")


    from sklearn.neighbors import KNeighborsClassifier

# Training Dataset
training_skills = [
    "python machine learning deep learning",
    "python artificial intelligence",
    "python neural networks",
    "python sql data analysis",
    "excel sql data visualization",
    "python statistics analytics",
    "html css javascript",
    "html css web design",
    "javascript frontend development",
    "python django backend",
    "python flask programming",
    "python api development"
]

training_labels = [
    "Machine Learning Engineer",
    "Machine Learning Engineer",
    "Machine Learning Engineer",
    "Data Analyst",
    "Data Analyst",
    "Data Analyst",
    "Frontend Developer",
    "Frontend Developer",
    "Frontend Developer",
    "Python Developer",
    "Python Developer",
    "Python Developer"
]

# Convert Text into Numbers
knn_vectorizer = TfidfVectorizer()
X_train = knn_vectorizer.fit_transform(training_skills)

# Train KNN Model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, training_labels)

# Predict Job Roles
X_test = knn_vectorizer.transform(resume_data["Skills"])

predictions = knn_model.predict(X_test)

print("\nKNN PREDICTIONS")

for name, prediction in zip(resume_data["Name"], predictions):
    print(name, ":", prediction)

print("\nKNN Model Completed Successfully!")

from sklearn.naive_bayes import MultinomialNB

# Create Naive Bayes Model
nb_model = MultinomialNB()

# Train Model
nb_model.fit(X_train, training_labels)

# Predict Job Roles
nb_predictions = nb_model.predict(X_test)

# Display Results
print("\nNAIVE BAYES PREDICTIONS")

for name, prediction in zip(resume_data["Name"], nb_predictions):
    print(name, ":", prediction)

print("\nNaive Bayes Model Completed Successfully!")