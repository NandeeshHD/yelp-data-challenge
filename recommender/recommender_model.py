import argparse
import logging

from pyspark import SparkContext, SparkConf
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SQLContext

LOGGING_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'


class RecommenderModel(object):
    '''Build and save recommender model'''
    
    name = 'RecommenderModel'

    args = None
    log_level = 'INFO'
    logger = None
    
    # hyper-parameters
    rank = [15]
    maxIter = [5]
    regParam = [0.01]

    def parse_arguments(self):
        '''Returns the parsed arguments from the command line.'''

        description = self.name
        if self.__doc__ is not None:
            description += " - "
            description += self.__doc__
        arg_parser = argparse.ArgumentParser(description=description)

        arg_parser.add_argument("dataset_path",
                                help="Path to dataset.")
        arg_parser.add_argument("model_storage_path",
                                help="Name of output table.")
        arg_parser.add_argument("--log_level", default=self.log_level,
                                help="Logging level")
        arg_parser.add_argument("--seed", type=int,
                                help="Set seed to get reproducible results.")
        arg_parser.add_argument("--numFolds", type=int, default=3,
                                help="Set number of folds for crossvalidation.")

        args = arg_parser.parse_args()
        self.init_logging(args.log_level)

        return args

    def init_logging(self, level=None):
        '''Initializes the logger
        :param level optional string setting logging level.
        '''
        if level is None:
            level = self.log_level
        else:
            self.log_level = level
        logging.basicConfig(level=level, format=LOGGING_FORMAT)

    def get_logger(self, spark_context=None):
        '''Get logger from SparkContext or (if None) from logging module
        :param spark_context: optional spark context object. If passed will return
        logger from that context.
        :return: logger object
        '''
        if spark_context is None:
            return logging.getLogger(self.name)
        return spark_context._jvm.org.apache.log4j.LogManager \
            .getLogger(self.name)

    def run(self):
        '''Entry method for the Spark job.'''
        self.args = self.parse_arguments()

        conf = SparkConf().setAll((
            ('spark.task.maxFailures', '10'),
            ('spark.locality.wait', '20s'),
            ('spark.serializer', 'org.apache.spark.serializer.KryoSerializer'),
#             ("spark.driver.maxResultSize", "8g"),
#             ("spark.driver.memory", "4g"),
#             ("spark.executor.memory", "4g"),
#             ("spark.hadoop.validateOutputSpecs", "false"),
        ))
        
        sc = SparkContext(
            appName=self.name,
            conf=conf)
        sqlc = SQLContext(sparkContext=sc)
        
        self.logger = self.get_logger(sc)

        dataset = sqlc.read.parquet(self.args.dataset_path)
        training, test = dataset.randomSplit([0.8, 0.2], seed=self.args.seed)
        model = self.train_validate(training)
        self.test(test)
        self.save_model(model)
        sc.stop()

    def train_validate(self, dataset):
        '''Build the recommendation model using ALS on the training data
        Set cold start strategy to 'drop' to ensure not to get NaN evaluation metrics.
        :param dataset: input dataset
        :return: bestModel trained by cross validator
        '''
        als = ALS(userCol='user_id_numeric', itemCol='business_id_numeric', ratingCol='stars',
                  coldStartStrategy="drop")
        
        # Construct a grid of parameters to search over.
        self.logger.info('Constructing parameter grid.')
        param_grid = ParamGridBuilder().addGrid(als.rank, self.rank) \
                                      .addGrid(als.maxIter, self.maxIter) \
                                      .addGrid(als.regParam, self.regParam) \
                                      .build()

        evaluator = RegressionEvaluator(labelCol='stars', metricName='rmse')

        # Perform k-fold validation
        self.logger.info('Performing training and crossvalidation.')
        cross_validator = CrossValidator(estimator=als,
                                         estimatorParamMaps=param_grid,
                                         evaluator=evaluator,
                                         numFolds=self.args.numFolds,
                                         seed=self.args.seed)

        # Run CrossValidator, and choose the best set of parameters.
        model = cross_validator.fit(dataset)
        bestModel = model.bestModel
        rank = bestModel.getRank()
        maxIter = bestModel.getMaxIter()
        regParam = bestModel.getRegParam()
        self.logger.info('Best model params:: Rank: {}, Max_iter: {}, Reg_param: {}'.format(
            rank, maxIter, regParam))
        return bestModel

    def save_model(self, model):
        '''Saves the trained model.
        :param model: the trained model to save'''
        model.bestModel.write().overwrite().save(self.args.model_storage_path)
        self.logger.info('Model saved at - {}'.format(self.args.model_storage_path))

    def test(self, test):
        '''Evaluate the model by computing the RMSE on the test data.
        :param test: test dataset
        '''
        model = ALSModel.load(self.args.model_storage_path)
        evaluator = RegressionEvaluator(labelCol='stars', metricName='rmse')
        predictions = model.transform(test)
        rmse = evaluator.evaluate(predictions)
        print("Root-mean-square error = " + str(rmse))


if __name__ == '__main__':
    recommender_model = RecommenderModel()
    recommender_model.run()
