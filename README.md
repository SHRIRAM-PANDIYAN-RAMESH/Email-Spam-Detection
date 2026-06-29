# Email-Spam-Detection
This project is based on creating an AI Naive Bayes algorithm Python code that takes a dataset containing all set of emails classified as Real or Spam. This calculates the accuracy of the dataset, gives a classification report and creates a confusion matrix with a graph display for learning purposes.

# Introduction
Email Spam Detection is the project which with the help of an AI based model trained using Naive Bayes algorithm, which comes under Supervised Machine Learning, from a given dataset, checks the mail and decides whether the content inside the email is 'Real' or 'Spam'.
This was done using Python in Google Colab, using Scikit-Learn, Pandas and Matplotlib libraries and modules.

# Project Details
  # Name of Dataset: 
  Email_Spam_Data.csv
  # Modules Imported:
  Scikit-Learn: 
    - Metrics: [accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc]
    - Naive_Bayes: [MultinomialNB]
    - Model_Selection: [train_test_split]
    - feature_extraction.text: [TfidfVectorizer, CountVectorizer]
  Matplotlib:
    - matplotlib.pyplot as plt
  Pandas:
    - pandas as pd
  Google Colab
    - google.colab import files

# Output
This output is for the code with test_size = 0.2 and random_state = 42. Also on changing the values of the two factors gives a different result in the Accuracy Score, Confusion Matrix values and Classification Report.

Accuracy: 1.0

Classification Report:
              precision    recall  f1-score   support

        Real       1.00      1.00      1.00        53
        Spam       1.00      1.00      1.00        49

    accuracy                           1.00       102
   macro avg       1.00      1.00      1.00       102
weighted avg       1.00      1.00      1.00       102

Confusion Matrix:
[[53  0]
 [ 0 49]]

Our output also contains graphs for displaying confusion matrix, counting the number of 'Real' and 'Spam' emails using Bar Graph and Pie Chart, calculating the value of True Positives and False Positives on checking for spam emails using a ROC Curve and getting a rough estimation of the Top 10 words used in Spam emails through which the model can identify spam emails in future.

# Conclusion
This project thus creates a basic AI model that detects 'Real' or 'Spam' emails with the help of a given dataset on a given training and testing percentage of the dataset model.

# Project Created By
SHRIRAM PANDIYAN RAMESH
