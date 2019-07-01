# Yelp-Data-Challenge

This project showcases building a recommendendation engine using Yelp dataset.
The project covers following phases:
1. Extracting data from JSON files and storing it as Parquet files using PySpark.
2. Creating the necessary training dataset from the stored Parquet files using PySpark.
3. Training-Crossvalidation-Testing recommendation model using alternating least squares (collaborative filtering) algorithm available in PySpark.
4. Deploying the trained model as Flask API.

## Obtaining the Docker Image

The docker images can be found in the [Nandeesh's repository](https://hub.docker.com/r/nandee/yelp-data-challenge) on Docker Hub.

To get the docker image, the following `pull` command can be used.

    docker pull nandee/jupyter-pyspark

## Note: - Hardware requirement to run the Spark jobs in the container
Some jobs in the container need around 16Gb of RAM. So minimum requirement is to allocate more than 16Gb of RAM.
If possible allocate more than 4 CPUs.

To make these changes follow the instruction in this [Stack Overflow](https://stackoverflow.com/a/44533437/3323084) answer.

## Running the Image

### To run in detached mode

	docker run -it --rm -d -p 8888:8888 -p 9001:9001 -p 4040-4042:4040-4042 \
	                       -e GRANT_SUDO="yes" \
	                       --user root \
	                       --name jupyter \
	                       nandee/yelp-data-challenge

Enter the following command to get the URL to access the notebook.

	docker logs jupyter


### Or simply run the following command and copy-paste the URL shown in terminal into browser to access notebook

	docker run -it --rm -p 8888:8888 -p 9001:9001 -p 4040-4042:4040-4042 \
	                    -e GRANT_SUDO="yes" \
	                    --user root \
	                    --name jupyter \
	                    nandee/yelp-data-challenge

## Download data, extract and convert JSON to Parquet files for storage

After opening the Jupyter Notebook link in a browser, go to `yelp-data-challenge` directory and open `Extracting Yelp Dataset.ipynb` notebook.
Run the commands and Spark jobs in the notebook to fulfill the task.

## Generate training dataset and train a model

Open the `Recommender Model.ipynb` notebook and follow through the commands and Spark jobs to generate dataset, train a model and launch a Flask API to serve the model.

## Using the API to give recommendations to users

Open `Recommender System.ipynb` notebook which shows few examples on how to use the model.