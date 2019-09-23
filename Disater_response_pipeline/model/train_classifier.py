import sys, re, nltk, pickle
import pandas as pd

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')

def load_data(database_filepath):

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)

    X = df.message
    y = df.loc[:, 'related':'direct_report'].fillna(0)
    cat_names = y.columns

    return X, y, cat_names


def tokenize(text):

    text = re.sub(r"[^a-z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens]

    return tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42, verbose=3)))
    ])

    params = {
        'vect__ngram_range': ((1,1), (2,2)),
        'tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [10, 100],

        #'clf__estimator__criterion': ['gini', 'entropy'],
        #'clf__estimator__max_depth': [4, 5, 6, 7, 8],
        #'clf__estimator__max_features': ['auto', 'sqrt', 'log2']
    }

    model = GridSearchCV(pipeline, param_grid=params, cv=5)

    return model


def evaluate_model(model, X_test, Y_test, cat_names):
    pred = model.predict(X_test)

    for i, col in enumerate(Y_test.columns):
        print("Type of message: {}".format(cat_names[i]))
        print("Accuracy score: {:.2f}".format(accuracy_score(Y_test.values[i], pred[i])))
        print("Classification report: {}".format(classification_report(Y_test.values[i], pred[i])))

    return None


def save_model(model, model_filepath):

    pickle.dump(model, open(model_filepath, "wb"))

    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, cat_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, cat_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()