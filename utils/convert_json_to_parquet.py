import argparse
import logging

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.window import Window as W

LOGGING_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'


class JsonToParquet(object):
    '''Converts the input json file to parquet file.'''

    name = 'JsonToParquet'

    args = None
    log_level = 'INFO'
    logging.basicConfig(level=log_level, format=LOGGING_FORMAT)
    num_input_partitions = 100
    num_output_partitions = 10

    def parse_arguments(self):
        '''Returns the parsed arguments from the command line.'''

        description = self.name
        if self.__doc__ is not None:
            description += " - "
            description += self.__doc__
        arg_parser = argparse.ArgumentParser(description=description)

        arg_parser.add_argument("input",
                                help="Path to json file.")
        arg_parser.add_argument("output",
                                help="Name of output table.")
        arg_parser.add_argument("--num_input_partitions", type=int,
                                default=self.num_input_partitions,
                                help="Number of input splits/partitions")
        arg_parser.add_argument("--num_output_partitions", type=int,
                                default=self.num_output_partitions,
                                help="Number of output partitions")
        arg_parser.add_argument("--log_level", default=self.log_level,
                                help="Logging level")
        arg_parser.add_argument("--date_column",
                                help="Name of the date column if present.")
        arg_parser.add_argument("--date_format", default='yyyy-MM-dd',
                                help="Format of the date.")
        arg_parser.add_argument("--timestamp_column",
                                help="Name of the timestamp column if present.")
        arg_parser.add_argument("--timestamp_format", default='yyyy-MM-dd HH:mm:ss',
                                help="Format of the timestamp.")
        arg_parser.add_argument("--numeric_id_over_col",
                                help="Specify the column name to be used to create numeric ID.")

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
        '''Get logger from SparkContext or (if None) from logging module'''
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

        self.run_job(sc, sqlc)

        sc.stop()

    def run_job(self, sc, sqlc):
        input_rdd = sc.textFile(self.args.input,
                                minPartitions=self.args.num_input_partitions)

        output = sqlc.read.json(input_rdd).cache()
        n_records = output.count()
        
        if self.args.date_column:
            output = output.withColumn(self.args.date_column,
                                       F.to_date(self.args.date_column,
                                                 self.args.date_format))
        if self.args.timestamp_column:
            output = output.withColumn(self.args.timestamp_column,
                                       F.to_timestamp(self.args.timestamp_column,
                                                      self.args.timestamp_format))

        if self.args.numeric_id_over_col:
            column_name = self.args.numeric_id_over_col + '_numeric'
            # create a monotonically increasing id
            output = output.withColumn(column_name, F.monotonically_increasing_id())
            # then since the id is increasing but not consecutive
            # it means you can sort by it, so can use the `row_number`
            windowSpec = W.orderBy(column_name)
            output = output.withColumn(column_name, F.row_number().over(windowSpec))

        output.coalesce(self.args.num_output_partitions) \
              .write \
              .parquet(self.args.output, mode='overwrite')

        self.get_logger(sc).info('Number of records = {}'.format(n_records))


if __name__ == '__main__':
    job = JsonToParquet()
    job.run()

