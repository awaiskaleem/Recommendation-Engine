# E-Commerce-Recommendation-Engine


.
├── README.md
├── __init__.py
├── app.py
├── artifacts (These will be loaded once trained)
│   ├── cate_enc_dict_feature.pickle
│   ├── cate_enc_dict_itemid.pickle
│   ├── cate_enc_dict_visitorid.pickle
│   ├── model_with_items.pickle
│   ├── model_without_items.pickle
│   ├── rate_matrix_all.npz
│   ├── rate_matrix_feature.npz
│   ├── rate_matrix_test.npz
│   └── rate_matrix_train.npz
├── data (This is where following data files should be placed)
│   ├── category_tree.csv
│   ├── events.csv
│   ├── item_properties_part1.csv
│   ├── item_properties_part2.csv
│   └── predictions.csv
├── docs (Exploratory data analysis and other documentations)
│   └── eda.ipynb
├── etl (extraction, transform, and load files)
│   ├── feature_extractor.py
│   └── preprocessing.py
├── output (results will be saved here provided input is a file)
│   └── results.csv
├── requirements.txt
└── src (this is where model code files are)
    ├── model.py

<h2>Introduction:</h2>
<b>Data Used:</b>https://www.kaggle.com/retailrocket/ecommerce-dataset
<br>The project build on e-commerce user visit data and creates a recommendation engine. Target predictions are items recommended for a particular use. Project also supports batch file as input and output in /output directory.

<h2>How To Guide:</h2>
<b>Data Used:</b>https://www.kaggle.com/retailrocket/ecommerce-dataset
<br>
To predict using prediction.csv placed in data directory, run following command
<br>
`<addr>` curl http://127.0.0.1:8080/predict_file
<br>
In case you would like to limit predictions to top N use following format:
<br>
`<addr>` curl http://127.0.0.1:8080/predict_file/N
`<addr>` curl http://127.0.0.1:8080/predict_file/5
<br>
