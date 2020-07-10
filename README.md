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
* `google-cloud-sdk`

Also, make sure that you provide enough resources to the docker machine if you are on windows or MacOS.

## Data Setup

This project uses MIMIC III dataset. To download the data, after getting access from https://mimic.physionet.org/gettingstarted/cloud/, we suggest using the Google Cloud Storage. Install the google-cloud-sdk to and download the data with the following command:

```shell
$ export DATASET=mimiciii-1.4.physionet.org  # or mimiciii-demo-1.4.physionet.org
$ gsutil -m cp -r gs://${DATASET} ./data/raw/${DATASET}
```

If the data is already downloaded then copy the *.csv files into `data/raw/mimiciii-1.4.physionet.org/` folder.

Change the value of `DATASET` to the particular dataset do you want to download. This way you can control different datasets locally of different sizes to test the pipelines and model. In any case, this inmutable orginal data should be in `./data/raw/` directory.

> **IMPORTANT**: Remember to set the `DATASET` environment variable in any shell you are interacting with this project. Otherwise, it will default to mimiciii-1.4.physionet.org

Every output of a processing step, should be stored in `./data/processed/${DATASET}`. The convention is:

* Output of hive scripts: `./data/processed/${DATASET}/hive`
* Output of feature construction for length of stay: `./data/processed/${DATASET}/los`

You can also add nested directory structures to identify datasets. For example, if you are creating a subset dataset with half of the data for the lenght of stay features construction, you can use a path like `./data/processed/${DATASET}/los/subset=0.5/`. This way is easier to understand the differences between the datasets during experimentation. Some notebooks will use this convention when storing the datasets by default.

## Project Environment Setup

### Makefile

A Makefile is provided to automatize the execution targets and development tasks. This is not complete but you can fine some useful automation targets here.

```shell
$ make
Available rules:

clean               Delete all compiled Python files
data-hdfs-setup     Load data in HDFS. Set environemnt variable DATASET to change the dataset to load.
data-spark-setup    Load data in HDFS and create tables for Spark. Set environemnt variable DATASET to change the dataset to load.
hive-pipeline       Launch all the scripts in the hive pipeline
lint                Lint using flake8
requirements        Setup environment and install Python Dependencies
stop                Stop all docker-compose services
stop-hive           Stop hive infra
stop-notebook       Stop jupyter lab notebook
up                  Launch all the docker-compose stack
up-hive             Launch hive infra docker compose
up-notebook         Launch jupyter lab in docker compose
```

We recommend using the make commands through the project as it provides a standard way of autommating the project execution steps.

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

#### Running Hive ETL

First of all, make sure that the you follow the steps on the [Data Setup](#data-setup), and the hive services are running with the `make up-hive` command.

> The services could take almost a minute to finish all the initialization scripts. If there are some errors with the below steeps, try waiting a minute to let the initialization finish.

You can execute the hive pipeline with following targets:

```shell
$ make data-hdfs-setup hive-pipeline
[04-19-2020T13:53:09] hive-setup.sh | Using DATASET=mimiciii-1.4.physionet.org dataset located in /opt/scripts/../data/raw/mimiciii-1.4.physionet.org
[04-19-2020T13:53:09] hive-setup.sh | Set up root folders in HDFS
[04-19-2020T13:53:12] hive-setup.sh | Create the folders for raw data DATASET=mimiciii-1.4.physionet.org in HDFS
[04-19-2020T13:53:53] hive-setup.sh | Putting the files in HDFS
```

The `data-hdfs-setup` will load the dataset into HDFS to be used. This can take a lot of time depending of your system. The `hive-pipeline` can take up to 2 hours to finish. You can also run a single hive stage with the following command:

```shell
$ make hive-stage-1  # This will run the first stage in src/hive/1_staging.hql
```

Logging information will be placen in `data/logs/scripts.logs`.

At the end, you can find the following tables created:

| database |    tableName                    | isTemporary |
|----------|---------------------------------|-------------|
| default  |    admissions                   |    False    |
| default  |    chartevents                  |    False    |
| default  |    episode_timeseries           |    False    |
| default  |    episodic_data                |    False    |
| default  |    extract_subjects_all_stays   |    False    |
| default  |    icustays                     |    False    |
| default  |    itemid_to_variable_map       |    False    |
| default  |    labevents                    |    False    |
| default  |    num_seq                      |    False    |
| default  |    outputevents                 |    False    |
| default  |    patients                     |    False    |
| default  |    testset                      |    False    |
| default  |    validate_events              |    False    |
| default  |    validate_events_chartevents  |    False    |
| default  |    validate_events_cleaned      |    False    |
| default  |    validate_events_labevents    |    False    |
| default  |    validate_events_outputevents |    False    |
| default  |    valset                       |    False    |


The ones used for the construction of the features in spark will be `episode_timeseries` and `episodic_data`.

> **NOTE:** The remaining section is not required but is mentioned for troubleshooting purpose and as an alternative to `make data-hdfs-setup hive-pipeline`

Additionally, some scripts are provided to work with hive while developing. First connect to running `hive-server` service:

```shell
$ docker-compose exec hive-server bash
```

Now you can go to the project and launch the `./scripts/hive-setup.sh` script. This will load the necesary files in hdfs to process them with hive later on. You can pass the dataset to load as an argumment. This step is only needed once. Even if the container restart, the data is stored in a volume in your local machine.

```shell
root@hive-server:/opt# ./scripts/hive-setup.sh -h
hive-setup.sh [-dh]

Set up the folders structure in hdfs and upload the required datasets.

Flags:
    -h          Print this message and exits.
    -d DATASET  The dataset to use. The path must exist in data/raw/{DATASET}.
                Defaults to 'mimiciii-1.4.physionet.org'.
root@hive-server:/opt# ./scripts/hive-setup.sh
[04-19-2020T13:53:09] hive-setup.sh | Using DATASET=mimiciii-1.4.physionet.org dataset located in /opt/scripts/../data/raw/mimiciii-1.4.physionet.org
[04-19-2020T13:53:09] hive-setup.sh | Set up root folders in HDFS
[04-19-2020T13:53:12] hive-setup.sh | Create the folders for raw data DATASET=mimiciii-1.4.physionet.org in HDFS
[04-19-2020T13:53:53] hive-setup.sh | Putting the files in HDFS
[04-19-2020T13:54:58] hive-setup.sh | Success
```

To execute the hive transformations with the hive pipeline you can run the following script:

```shell
root@hive-server:/opt# ./scripts/hive-execute-queries-all.sh -h
hive-execute-queries-all.sh [-hds]

Execute a query stage with the proper parameters. Check the hive-queries.log
in the script folder to get information about the exeution.

Flags:
    -h          Print this message and exits.
    -d DATASET  The dataset to use. It must be loaded first to HDFS. Defaults
                to 'mimiciii-1.4.physionet.org'.
    -s SIZE     The size of the sample to use in the queries. Defaults to `50000`
root@hive-server:/opt#  ./scripts/hive-execute-queries-all.sh
```

Now you can extract any table from hive with the following command:

```shell
root@hive-server:/opt# ./scripts/hive-extract.sh -h
hive-extract.sh [-hfdpl] [TABLE]

Extract the hive datasets to a csv files.

Arguments:
    TABLE       The table to extract

Flags:
    -h          Print this message and exits.
    -d DATASET  The dataset to use. It must be loaded first to HDFS. Defaults
                to 'mimiciii-1.4.physionet.org'.
    -f FILE     The file name to extract. Defaults to the table name.
    -p DIR      Directory path to put the data. Defaults to .
    -l NUMBER   Limit in number of rows.
root@hive-server:/opt# ./scripts/hive-extract.sh episode_timeseries
Dataset: mimicii
Table to extract: episode_timeseries
Directory output: /home/mimiciii/project/scripts/../../data/processed/mimicii/hive
File output: episode_timeseries.csv
ls: cannot access /usr/lib/spark/lib/spark-assembly-*.jar: No such file or directory

Logging initialized using configuration in file:/etc/hive/conf.dist/hive-log4j.properties
OK
Time taken: 1.688 seconds, Fetched: 395 row(s)
```

### Spark Notebooks and Modeling

Once the notebook and the services is up and running, you can access to the notebook URL as mentioned in [Docker Compose](#docker-compose) section.

The notebooks will interact with Hive tables and Spark. The spark UI should be available at http://localhost:4040/jobs/ once the spark context is initialized in one of the notebooks.

Refer to the [notebooks/README.md](notebooks/README.md) docs to understand what is contained in each notebook and how to run them.


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
