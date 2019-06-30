import argparse
import json
import logging

from flask import Blueprint
from flask import Flask, request
from paste.translogger import TransLogger
from pyspark import SparkContext, SparkConf
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SQLContext
import cherrypy

main = Blueprint('main', __name__)
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

 
def init_spark_context():
    ''''Initialize spark context.'''
    conf = SparkConf().setAll((
        ('spark.task.maxFailures', '10'),
        ('spark.locality.wait', '20s'),
        ('spark.serializer', 'org.apache.spark.serializer.KryoSerializer'),
    ))
    
    _sc = SparkContext(
            appName='Recommendation-Server',
            conf=conf)
    _sqlc = SQLContext(sparkContext=_sc)

    return _sc, _sqlc


@main.route('/<user_id>/recommend/top/<int:count>', methods=['GET'])
def top_recommendations(user_id, count):
    logger.info('User %s top recommendations requested', user_id)
    user_id = user.filter(user.user_id == user_id)
    result = model.recommendForUserSubset(user_id, count).collect()
    return json.dumps(result.collect())


def create_app(model_path):
    _model = ALSModel.load(model_path)
    
    _app = Flask(__name__)
    _app.register_blueprint(main)
    return _app, _model


def run_server(_app):
 
    # Enable WSGI access logging via Paste
    app_logged = TransLogger(_app)
 
    # Mount the WSGI callable object (app) on the root directory
    cherrypy.tree.graft(app_logged, '/')
 
    # Set the configuration of the web server
    cherrypy.config.update({
        'engine.autoreload.on': True,
        'log.screen': True,
        'server.socket_port': 9001,
        'server.socket_host': '0.0.0.0'
    })
 
    # Start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_storage_path",
                            help="Path where the model is stored.")
    arg_parser.add_argument("--business_input",
                            help="Path to businesses parquet files.")
    arg_parser.add_argument("--user_input",
                            help="Path to users parquet files.")
    arg_parser.add_argument("--dataset_path",
                            help="Path to dataset parquet files.")
    args = arg_parser.parse_args()

    # Init spark context and load libraries
    sc, sqlc = init_spark_context()
    business = sqlc.read.parquet(args.business_input).cache()
    user = sqlc.read.parquet(args.user_input).cache()
    # to pre-load the data
    logger.info(business.count())
    logger.info(user.count())
#     dataset = sqlc.read.parquet(args.dataset_path).cache()
#     logger.info(dataset.count())

    app, model = create_app(args.model_storage_path)
    model.userFactors.cache()
    model.userFactors.count()
    model.itemFactors.cache()
    model.itemFactors.count()
    
    # start web server
    run_server(app)
