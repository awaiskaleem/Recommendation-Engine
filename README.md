# E-Commerce-Recommendation-Engine

The project build on e-commerce user visit data and creates a recommendation engine. Target predictions are items recommended for a particular use. Project also supports batch file as input and output in /output directory.

## Project Organization

```
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
```

> Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/)


## Requirements

Make sure you have the following requirements installed:

* `python`
* `docker`
* `docker-compose`

Also, make sure that you provide enough resources to the docker machine if you are on windows or MacOS.

## Data Setup

This project uses publically available data set on an e-commerce website. The data is download-able from following link https://www.kaggle.com/retailrocket/ecommerce-dataset

If the data is already downloaded then copy the *.csv files into `./data` folder. Following files are expected for this project to work:
```
├── data 
│   ├── category_tree.csv
│   ├── events.csv
│   ├── item_properties_part1.csv
│   ├── item_properties_part2.csv
│   └── predictions.csv [optional for batch input]
```


## Project Environment Setup

### Docker Compose

We have created a docker-compose environment based on the [jupyeter notebook stacks](https://github.com/jupyter/docker-stacks) and the [Big Data Europe Hive docker-compose](https://github.com/big-data-europe/docker-hive).

You can use the makefile to start the docker services in the background. The notebook url will be output in the logs:

```shell
$ make up
Creating network "big-data-mimic_default" with the default driver
Creating big-data-mimic_datanode_1                  ... done
Creating big-data-mimic_namenode_1                  ... done
Creating big-data-mimic_hive-metastore-postgresql_1 ... done
Creating big-data-mimic_hive-metastore_1            ... done
Creating big-data-mimic_hive-server_1               ... done
Creating big-data-mimic_notebook_1 ... done
Attaching to big-data-mimic_notebook_1
notebook_1                   |     To access the notebook, open this file in a browser:
notebook_1                   |         file:///home/jovyan/.local/share/jupyter/runtime/nbserver-14-open.html
notebook_1                   |     Or copy and paste one of these URLs:
notebook_1                   |         http://jupyter.dev.internal:8888/jupyter/?token=b66683c5b5adb35c5270061db6b7e90e0a0ae70754ed6623
notebook_1                   |      or http://127.0.0.1:8888/jupyter/?token=b66683c5b5adb35c5270061db6b7e90e0a0ae70754ed6623
```

The services will take around between 15 and 25 seconds to startup. To stop the services:

```shell
$ make stop
```

To stop only the notebook or the hive services:

```shell
$ make stop-hive
$ make stop-notebook
```

To remove all the containers:
```shell
make clean
```

Or you can use the docker-compose commands:

```shell
$ docker-compose up
```

This will start both of the services and output the logs in stdout. You can use the  `-d` flag to be started as a daemon. To run just one service run:

```shell
$ docker-compose up notebook
$ docker-compose up hive-server
```

And to stop them:

```shell
$ docker-compose stop
```

