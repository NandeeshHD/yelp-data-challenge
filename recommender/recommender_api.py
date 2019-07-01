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


def get_recommendations(user_businesses, state=None):
    recomendations = []
    for _user in user_businesses:
        _businesses = _user[1]
        for _business in _businesses:
            recomendations.append(business.filter(
                business.business_id_numeric == _business[0]) \
                .select(['name', 'stars', 'review_count', 'address', 'city', 'state']) \
                .orderBy(['stars', 'review_count'], ascending=[0, 0]) \
                .collect()
            )

    if state:
        recomendations = [{'name': i[0][0], 'stars': i[0][1],
                           'review_count': i[0][2], 'address': i[0][3],
                           'city': i[0][4], 'state': i[0][5]} for i in recomendations if i[0][5] == state]
    else:
        recomendations = [{'name': i[0][0], 'stars': i[0][1],
                           'review_count': i[0][2], 'address': i[0][3],
                           'city': i[0][4], 'state': i[0][5]} for i in recomendations]
    print(recomendations)
    return recomendations


@main.route('/<user_id>/recommend/top/<int:count>', methods=['GET'])
def top_recommendations(user_id, count):
    logger.info('User %s top recommendations requested', user_id)
    state = request.args.get('state')
    user_id_df = dataset.filter(dataset.user_id == user_id).select(['user_id_numeric'])
    result = model.recommendForUserSubset(user_id_df, count).collect()
    recomendations = get_recommendations(result, state)
    if not result:
        result = model.recommendForUserSubset(user_id_df, 50000).collect()
        recomendations = get_recommendations(result, state)

    return json.dumps(recomendations[:count], indent=4)


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
    description = 'Recommender API'
    arg_parser = argparse.ArgumentParser(description=description)
    arg_parser.add_argument("--model_storage_path",
                            help="Path where the model is stored.")
    arg_parser.add_argument("--business_input",
                            help="Path to businesses parquet files.")
    arg_parser.add_argument("--dataset_path",
                            help="Path to dataset parquet files.")
    args = arg_parser.parse_args()

    # Init spark context and load libraries
    sc, sqlc = init_spark_context()
    # to pre-load the data
    business = sqlc.read.parquet(args.business_input).cache()
    logger.info(business.count())
    dataset = sqlc.read.parquet(args.dataset_path).cache()
    logger.info(dataset.count())

    app, model = create_app(args.model_storage_path)
    model.userFactors.cache()
    logger.info(model.userFactors.count())
    model.itemFactors.cache()
    logger.info(model.itemFactors.count())
    
    # start web server
    run_server(app)
