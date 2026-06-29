# ----------------------------------------------------------------------------- EMAIL SPAM DETECTION ----------------------------------------------------------------------------------------
# This is a Python project which using Naive Bayes Supervised ML algorithm, checks a dataset containing many rows of Real and Spam emails.
# This also gives a rough calculation of Accuracy Score, Classification Report and Confusion Matrix.
# This was done originally in Google Colab with separate pieces of codes.
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1) Importing Dataset CSV file to Google Colab
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from google.colab import files
uploaded = files.upload()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2) Verification of Dataset loaded and reading the contents in the dataset using Pandas library.
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd

df = pd.read_csv('Email_Spam_Data.csv')
print(df.head())
print(df.info())

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3) A Python code to create features and target from the Dataset to train our AI Model
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
df["Text"] = df["Subject"] + " " + df["Full Content"]

X = df["Text"]
y = df["Email Label"]

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 4) Importing Required Modules and Libraries for our AI model to train and test our model on the dataset
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 5) Training and Testing our Model and returning Accuracy Score, Classification Report and Confusion Matrix as Output along with plotted graphs
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  #Here, on changing the values of test_size and random_state, the accuracy and the other values may vary slightly different.
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print('\n')

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

print('\n')

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 6) The Python Code to return various plotted graphs for our accuracy and classification report results.
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#  (a) Displaying the results of Confusion Matrix in the form of 'Real' or 'Spam'
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Real", "Spam"],
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.show()

#  (b) A Bar Graph to display the results of our Email Label column in the Dataset.
counts = df["Email Label"].value_counts()

plt.figure(figsize=(6,3))
plt.bar(counts.index, counts.values)

plt.title("Distribution of Email Labels")
plt.xlabel("Email Type")
plt.ylabel("Number of Emails")

plt.show()

#  (c) A Pie Chart to display the percentage of the number of 'Real' or 'Spam' emails.
plt.figure(figsize=(4,4))

plt.pie(
    counts,
    labels=counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Spam vs Real Emails")
plt.show()

#  (d) A ROC Curve Graph
y_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label="Spam")
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.show()

#  (e) A graph to show Top 10 Most Frequent Spam Words
spam = df[df["Email Label"]=="Spam"]["Full Content"]

cv = CountVectorizer(stop_words="english")
X_words = cv.fit_transform(spam)

word_freq = X_words.sum(axis=0)

words = pd.DataFrame({
    "Word": cv.get_feature_names_out(),
    "Count": word_freq.A1
})

words = words.sort_values(
    "Count",
    ascending=False
).head(10)

plt.figure(figsize=(8,5))
plt.bar(words["Word"], words["Count"])
plt.xticks(rotation=45)
plt.title("Top 10 Words in Spam Emails")
plt.show()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
