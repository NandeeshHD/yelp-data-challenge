import argparse
import logging

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

NAME = 'GenerateDataset'
LOGGING_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'


def get_logger(name, spark_context=None):
    '''Get logger from SparkContext or (if None) from logging module
    :param spark_context: optional spark context object. If passed will return
    logger from that context.
    :return: logger object
    '''
    if spark_context is None:
        return logging.getLogger(name)
    return spark_context._jvm.org.apache.log4j.LogManager \
        .getLogger(name)


def generate_dataset(args, sc, sqlc):
    '''Generate training dataset by combining user, review & businesses.
    :param sc: spark context object
    :param sqlc: sql context object
    '''
    get_logger(NAME, sc).info('Loading review dataframe...')
    review = sqlc.read.parquet(args.review_input) \
             .select(['review_id_numeric', 'review_id', 'stars', 'business_id', 'user_id'])

    get_logger(NAME, sc).info('Loading user dataframe...')
    user = sqlc.read.parquet(args.user_input) \
           .select(['user_id', 'user_id_numeric'])

    get_logger(NAME, sc).info('Loading business dataframe...')
    business = sqlc.read.parquet(args.business_input) \
               .select(['business_id', 'business_id_numeric'])
    
    user_review_business = review.join(user, 'user_id').join(business, 'business_id').cache()

    user_review_business.coalesce(args.num_output_partitions) \
                        .write \
                        .parquet(args.output, mode='overwrite')
    
    get_logger(NAME, sc).info(user_review_business.show(5, truncate=False))
    get_logger(NAME, sc).info('Dataset saved at - {}'.format(args.output))


if __name__ == '__main__':
    description = '''Create dataset to train a Collaborative Filtering model.
                     Dataset is created by assinging numeric ids to users & businesses.'''

    arg_parser = argparse.ArgumentParser(description=NAME + ' - ' + description)

    arg_parser.add_argument("review_input",
                            help="Path to review parquet files.")
    arg_parser.add_argument("user_input",
                            help="Path to user parquet files.")
    arg_parser.add_argument("business_input",
                            help="Path to business parquet files.")
    arg_parser.add_argument("output",
                            help="Path to store the output dataset.")
    arg_parser.add_argument("--num_input_partitions", type=int,
                            default=100,
                            help="Number of input splits/partitions")
    arg_parser.add_argument("--num_output_partitions", type=int,
                            default=10,
                            help="Number of output partitions")
    arg_parser.add_argument("--log_level", default='INFO',
                            help="Logging level")
    
    args_ = arg_parser.parse_args()
    
    logging.basicConfig(level=args_.log_level, format=LOGGING_FORMAT)

    conf = SparkConf().setAll((
        ('spark.task.maxFailures', '10'),
        ('spark.locality.wait', '20s'),
        ('spark.serializer', 'org.apache.spark.serializer.KryoSerializer'),
    ))
    
    sc_ = SparkContext(
        appName=NAME,
        conf=conf)
    sqlc_ = SQLContext(sparkContext=sc_)

    generate_dataset(args_, sc_, sqlc_)

    sc_.stop()

