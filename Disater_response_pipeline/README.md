**BACKGROUND**

This model provides an classifier web app for disaster messages provided by [Figure Eight](https://www.figure-eight.com), a machine learning company. Message data contains real messages sent during disaster events. The web app provides an interface where aid workers can input any message and get classification results in several categories. 

Screenshots of the web app can be found [here](https://github.com/bobbleoxs/data_science/blob/master/Disater_response_pipeline/visuals/classify.png), [here](https://github.com/bobbleoxs/data_science/blob/master/Disater_response_pipeline/visuals/visual1.png), and [here](https://github.com/bobbleoxs/data_science/blob/master/Disater_response_pipeline/visuals/visual2.png).


**INSTRUCTIONS**

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
   - `python run.py`

3. Go to http://0.0.0.0:3001/


**FILE DIRECTORY**

`app`
```
run.py                       # runs Flask app

├── templates 

go.html                      # web app result page
master.html                  # web app main page  
```

`data`
```
disaster_categories.csv     # disaster categories  
disaster_messages.csv       # disaster messages
process_data.py             # pre-processing data
```

`models`
```
train_classifier.py         # ML model for classification           
```

`visuals`
```
classify.png                # web app page for classification   
visuals1,2                  # web app page screenshots
```

**Licensing, Author, Acknowledgements**

This work is licensed under a [Creative Commons  Attribution-NonCommercial-NoDerivatives 4.0 International License](http://creativecommons.org/licenses/by-nc-nd/4.0/). Please refer to [Udacity Terms of Service](https://www.udacity.com/legal) for further information.
