import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Load the labeled dataset
data = pd.read_csv('filtered_processed5.csv', encoding='ISO-8859-1')
data.fillna(value='Unknown', inplace=True)
# Split the dataset into training and testing sets with a 80%-20% split
X_train, X_test, y_train, y_test = train_test_split(data["Tweet"], data["Sentiment_Labels"], test_size=0.2, random_state=42)
# Use RandomOverSampler to balance the training set
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train.to_frame(), y_train)
# Define the ngram ranges to loop through
ngram_ranges = [(1,4)]
for ngram_range in ngram_ranges:
 # Create a TF-IDF vectorizer and transform the training data
    tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['Tweet'])
 # Define the hyperparameter distribution
    param_dist = {'alpha': uniform(0.01, 10)}
 # Perform randomized search with cross-validation
    rs = RandomizedSearchCV(MultinomialNB(), param_distributions=param_dist, n_iter=100, cv=10, random_state=42)
    rs.fit(X_train_tfidf, y_train)
 # Train a Multinomial Naive Bayes classifier model with the best hyperparameters
    best_alpha = rs.best_params_['alpha']
    mnb_classifier = MultinomialNB(alpha=best_alpha)
    mnb_classifier.fit(X_train_tfidf, y_train)
 # Test the classifier on the testing set
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    y_pred = mnb_classifier.predict(X_test_tfidf)
 # Evaluate the performance of the classifier
    print('From filtered_procesed5: ')
    print("Evaluation metrics for ngram_range", ngram_range)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, pos_label='positive'))
    print("Recall:", recall_score(y_test, y_pred, pos_label='positive'))
    print("F1-score:", f1_score(y_test, y_pred, pos_label='positive'))
 # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
 # Print the confusion matrix and classification report
    print('Confusion matrix:')
    print(conf_matrix)
    print('Classification report:')
    print(classification_report(y_test, y_pred))

import pickle
with open("mnb_classifier.pkl", "wb") as f:
    pickle.dump(mnb_classifier, f)
    
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)
