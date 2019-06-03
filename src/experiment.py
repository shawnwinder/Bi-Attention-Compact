import logging
import time
import os
from shutil import copyfile
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, utils



class Experiment(object):
    def __init__(self, root_dir, tag, enable_log_file=True):
        """
        Args:
            root_dir: normally, this refers to 'MAMC_ROOT/experiments/'
            tag: denotes the specific experiment type, means what kind of
            experiment is carrying on
            enable_log_file: using file to record log or not
        """
        self.root_dir = root_dir
        self.tag = tag
        self.enable_log_file = enable_log_file
        self.experiment_dir = os.path.join(self.root_dir, '{}-{}'.format(
            self.tag, self.timestamp()))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)


    def timestamp(self):
        return time.strftime('%y%m%d-%H:%M:%S', time.localtime())


    def add_config_file(self, config_file):
        '''
        Args:
            config_file: absolute_path of the config_file
        '''
        config_file_basename = os.path.basename(config_file)
        dst_path = os.path.join(self.experiment_dir, config_file_basename)
        copyfile(config_file, dst_path)


    def add_chart(self, chart_name, xlabel='', ylabel=''):
        plt.figure(chart_name, figsize=(12, 8))
        plt.title(chart_name)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        return chart_name


    def add_plot(self, chart_name, epoch_results, curve_format='b.-',
                        curve_type='debug'):
        # switch figure
        plt.figure(chart_name)
        # plot
        plt.plot(range(1, 1+len(epoch_results)), epoch_results, curve_format,
                 label='{}_{}'.format(curve_type, chart_name))
        # label the curve type on chart
        if len(epoch_results) == 1:  # remove duplicated legends
            plt.legend()
        # save chart
        plt.draw()
        plt.savefig(os.path.join(self.experiment_dir, '{}.png'.format(chart_name)))


    def add_log(self, message):
        # logger test
        logger = logging.getLogger()
        # level
        logger.setLevel('DEBUG')
        # format
        # BASIC_FORMAT = "[%(asctime)s %(levelname)s] %(message)s"
        BASIC_FORMAT = "[%(levelname)s] %(message)s"
        DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
        # handler
        if not logger.handlers:  # deal with duplicated logging
            chlr = logging.StreamHandler() # console handler
            chlr.setFormatter(formatter)
            logger.addHandler(chlr)
            if self.enable_log_file:
                fhlr = logging.FileHandler(os.path.join(
                    self.experiment_dir, 'output.log')) # file handler
                fhlr.setFormatter(formatter)
                logger.addHandler(fhlr)

        # logging
        logger.info(message)


    def add_init_net_snapshot(self, params, workspace, name, tag, epoch):
        ''' save the model init_net as .pb file periodically
        Args:
            params: model params(weights and biases) to be stored
            workspace: caffe2 workspace
            name: the name specifies the experiment model and data
            tag: specifigc training type
            epoch: training epoch
        '''
        snapshot_dir = os.path.join(self.experiment_dir, 'snapshot')
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

        init_net_snapshot = os.path.join(
            snapshot_dir,
            '{}_{}_epoch-{}_init_net.pb'.format(
                name.lower(), tag.lower(), epoch),
        )

        init_net_proto = caffe2_pb2.NetDef()
        for param in params:
            blob = workspace.FetchBlob(param)
            shape = blob.shape
            op = core.CreateOperator(
                'GivenTensorFill',
                [],
                [param],
                arg=[
                    utils.MakeArgument('shape', shape),
                    utils.MakeArgument('values', blob)
                ]
            )
            init_net_proto.op.extend([op])
        with open(init_net_snapshot, 'wb') as f:
            f.write(init_net_proto.SerializeToString())


if __name__ == "__main__":
    ''' logger test
    logger = logging.getLogger()
    # level
    # logger.setLevel('INFO')
    logger.setLevel('DEBUG')

    # format
    BASIC_FORMAT = "[%(asctime)s %(levelname)s %(pathname)s] %(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    # handler
    chlr = logging.StreamHandler() # console handler
    chlr.setFormatter(formatter)
    # chlr.setLevel('INFO') # can be omitted, if so, same as logger's level
    fhlr = logging.FileHandler('tmp.log') # file handler
    fhlr.setFormatter(formatter)
    # fhlr.setLevel('INFO')
    logger.addHandler(chlr)
    logger.addHandler(fhlr)

    # logging
    logger.info('this is info')
    logger.debug('this is debug')
    '''

