import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request

from sklearn.externals import joblib
from sqlalchemy import create_engine

import plotly.graph_objs as goj

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("model/classifier.pkl")

genre_counts = df.groupby('genre').count()['message']
genre_names = list(genre_counts.index)

category_names = df.drop(['id', 'message', 'original', 'genre'], axis=1).columns
category_melt = pd.melt(df, id_vars=['id'], value_vars=category_names, var_name='categories', value_name='flag')
category_counts = category_melt.groupby('categories')['flag'].sum().sort_values(ascending=False)

# corr matrix between each categories.
category_corr = df[category_names].corr().values


def graph_1(df):
    data = [goj.Bar(
        x=genre_names,
        y=genre_counts)]

    layout = goj.Layout(
        title='Distribution of Message Genres',
        xaxis=dict(title='Genre', tickangle=45),
        yaxis=dict(title="No. of messages"))

    return goj.Figure(data=data, layout=layout)


def graph_2(df):
    data = [goj.Bar(
        x=category_names,
        y=category_counts)]

    layout = goj.Layout(
        title='Message categories and occurance',
        xaxis=dict(title='Category', tickangle=45),
        yaxis=dict(title="No. of messages"))

    return goj.Figure(data=data, layout=layout)


def graph_3(df):
    data = [goj.Heatmap(
        x=category_names,
        y=category_counts,
        z=category_corr,
        colorscale='Viridis')]

    layout = goj.Layout(
        title='Chart Frequency of Categories of Messages',
        xaxis=dict(title='Category'),
        yaxis=dict(title="Frequency"))

    return goj.Figure(data=data, layout=layout)


graph_1 = graph_1(df)
graph_2 = graph_2(df)
graph_3 = graph_3(df)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # encode plotly graphs in JSON
    graphs = [graph_1, graph_2, graph_3]
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
