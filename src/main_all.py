from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import argparse
import yaml
import os
import sys
import cv2
import time
from tqdm import tqdm
tqdm.monitor_interval = 0

from caffe2.python import (
    workspace,
    model_helper,
    core, brew,
    optimizer,
    net_drawer
)
from caffe2.proto import caffe2_pb2

from model_utils import load_init_net
from model_utils import add_input, add_softmax_loss, add_training_operators
from model_utils import add_model_all
from model_utils import add_accuracy
from experiment import Experiment



def function_log(func):
    ''' print basic running info of function '''
    def wrapper(*args, **kwargs):
        print("[INFO] start running {} ...".format(func.__name__))
        ret = func(*args, **kwargs)
        print("[INFO] finish running {} ...\n".format(func.__name__))
        return ret
    return wrapper


def file_log(run):
    ''' print running info of the script'''
    def wrapper(*args, **kwargs):
        print("[INFO] start running {} ...".format(__file__))
        beg_time = time.time()
        ret = run(*args, **kwargs)
        end_time = time.time()
        print("[INFO] finish running {}, total_time is {:.3f}s".format(
            __file__, end_time - beg_time))
        return ret
    return wrapper


def parse_args():
    # load config file
    config_parser = argparse.ArgumentParser(
        description='Imagenet model-finetune config parser',
    )
    config_parser.add_argument(
        '--config',
        type=str,
        required=True,
        help = 'config file'
    )
    args = config_parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
        config['config_path'] = os.path.join(os.getcwd(), args.config)
    return config


@function_log
def initialize(config):
    '''
    1. do the sanity check for path
    2. initialize workspace, e.g: add some CONST VALUE into config
    '''
    # 1. sanity check
    if not os.path.exists(config['root_dir']):
        raise ValueError("Root directory does not exist!")
    if not os.path.exists(config['network']['init_net']):
        raise ValueError("Pretrained init_net does not exist")

    # 2. initialze workspace
    if not config.has_key('consts'):
        # generate_and_dump_random_consts
        ### add rand_s & rand_h
        input_dim = config['model_arch']['feature_dim']
        output_dim = config['model_arch']['compact_dim']
        # randomly generate 1D index array containing indices in [0, output_dim)
        rand_h_1 = np.random.randint(output_dim, size=input_dim)
        rand_h_2 = np.random.randint(output_dim, size=input_dim)
        # randomly generate 1D index array of 1 and -1 with length of input_dim
        rand_s_1 = 2 * np.random.randint(2, size=input_dim) - 1
        rand_s_2 = 2 * np.random.randint(2, size=input_dim) - 1
        # store all the consts to config
        config['consts'] = dict()
        config['consts']['rand_h'] = [rand_h_1, rand_h_2]
        config['consts']['rand_s'] = [rand_s_1, rand_s_2]
        # dump all the consts
        tmp_consts = dict()
        tmp_consts['consts'] = dict()
        tmp_consts['consts']['rand_h'] = [rand_h_1, rand_h_2]
        tmp_consts['consts']['rand_s'] = [rand_s_1, rand_s_2]
        with open(config['config_path'], 'a') as f:
            f.write("\n#### this is for random const values\n")
            yaml.dump(tmp_consts, f, default_flow_style=False)

    # workspace reset
    workspace.ResetWorkspace(config['root_dir'])


@function_log
def build_training_model(config):
    # set device
    device_opt = caffe2_pb2.DeviceOption()
    if config['gpu_id'] is not None:
        device_opt.device_type = caffe2_pb2.CUDA
        device_opt.cuda_gpu_id = config['gpu_id']

    # build model
    with core.DeviceScope(device_opt):
        training_model = model_helper.ModelHelper(
            name = '{}_training_model'.format(config['name']),
        )
        data, label = add_input(training_model, config, is_test=False)
        pred = add_model_all(training_model, config, data, is_test=False)
        loss = add_softmax_loss(training_model, pred, label)
        add_training_operators(training_model, config, loss)
        add_accuracy(training_model)

    # init workspace for training net
    workspace.RunNetOnce(training_model.param_init_net)
    # if in finetune mode, we need to load pretrained weights and bias
    if config['finetune']:
        load_init_net(config['network']['init_net'], device_opt)
    workspace.CreateNet(training_model.net)
    return training_model


@function_log
def build_validation_model(config):
    # set device
    device_opt = caffe2_pb2.DeviceOption()
    if config['gpu_id'] is not None:
        device_opt.device_type = caffe2_pb2.CUDA
        device_opt.cuda_gpu_id = config['gpu_id']

    # build model
    with core.DeviceScope(device_opt):
        validation_model = model_helper.ModelHelper(
            name = '{}_validation_model'.format(config['name']),
            init_params=False,
        )
        data, label = add_input(validation_model, config, is_test=True)
        pred = add_model_all(validation_model, config, data, is_test=True)
        add_softmax_loss(validation_model, pred, label)
        add_accuracy(validation_model)

    # init workspace for validation net
    workspace.RunNetOnce(validation_model.param_init_net)
    workspace.CreateNet(validation_model.net)
    return validation_model


@file_log
def run_main(config):
    ''' running MAMC training & validation'''
    # init model
    initialize(config)
    training_model= build_training_model(config)
    validation_model= build_validation_model(config)

    # print network graph
    """
    # full-graph
    mamc_graph = net_drawer.GetPydotGraph(
        validation_model.net.Proto().op,
        "mamc_graph",
        rankdir="TB",
    )
    mamc_graph.write_svg("mamc_no_npairloss_graph.svg")
    print("write graph over...")
    sys.exit(0)

    # # mini-graph
    # mamc_graph_mini = net_drawer.GetPydotGraphMinimal(
    #     validation_model.net.Proto().op,
    #     "mamc_graph_minimal",
    #     rankdir="TB",
    #     minimal_dependency=True
    # )
    # mamc_graph_mini.write_svg("mamc_no_npairloss_graph_mini.svg")
    # print("write graph over...")
    # sys.exit(0)
    """

    # experiment params config
    # training mode
    # tag = "imagenet"
    tag = "cars196"
    if config['finetune']:
        tag = 'FINETUNE-{}'.format(tag)
    else:
        tag = 'RETRAIN-{}'.format(tag)

    root_experiments_dir = os.path.join(config['root_dir'], 'experiments')
    experiment = Experiment(root_experiments_dir, tag)
    experiment.add_config_file(config['config_path'])

    # add chart
    chart_acc = experiment.add_chart('accuracy', xlabel='epochs', ylabel='accuracy')
    chart_acc_5 = experiment.add_chart('accuracy_5', xlabel='epochs', ylabel='accuracy_5')
    chart_softmax_loss = experiment.add_chart('softmax_loss', xlabel='epochs', ylabel='softmax_loss')

    # plot params (should be added into 'experiment module'
    # TODO add 'variable' object to Experiment class
    training_acc_statistics = []
    training_acc5_statistics = []
    training_softmax_loss_statistics = []
    epoch_training_acc = 0
    epoch_training_acc5 = 0
    epoch_training_softmax_loss = 0
    training_accuracy = 0
    training_accuracy_5 = 0
    training_softmax_loss = 0

    validation_acc_statistics = []
    validation_acc5_statistics = []
    validation_softmax_loss_statistics = []

    # run the model
    experiment.add_log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for training_iter in tqdm(range(config['solver']['max_iterations'])):
        workspace.RunNet(training_model.net)
        accuracy = workspace.FetchBlob('accuracy')
        accuracy_5 = workspace.FetchBlob('accuracy_5')
        softmax_loss = workspace.FetchBlob('softmax_loss')

        epoch_training_acc += accuracy
        epoch_training_acc5 += accuracy_5
        epoch_training_softmax_loss += softmax_loss

        training_accuracy += accuracy
        training_accuracy_5 += accuracy_5
        training_softmax_loss += softmax_loss

        # display training result
        if training_iter != 0 and  (training_iter + 1) % config['solver']['display'] == 0:
            experiment.add_log("[TRAIN] epoch: {}   iteration: {}   accuracy: {:.4f}   "\
                  "accuracy_5: {:.4f}   softmax_loss: {:.4f}".format(
                      (training_iter // config['solver']['train_iterations'] + 1),
                      training_iter,
                      training_accuracy / config['solver']['display'],
                      training_accuracy_5 / config['solver']['display'],
                      training_softmax_loss / config['solver']['display'],
            ))
            experiment.add_log("Global learning rate: {}".format(
                workspace.FetchBlob('MultiPrecisionSgdOptimizer_0_lr_gpu{}'.format(config['gpu_id']))))

            # cleanup the counters
            training_accuracy = training_accuracy_5 = training_softmax_loss = 0

        # plot training statistics every epoch
        if training_iter != 0 and (training_iter + 1) % config['solver']['train_iterations'] == 0:
            training_acc_statistics.append(epoch_training_acc / config['solver']['train_iterations'])
            training_acc5_statistics.append(epoch_training_acc5 / config['solver']['train_iterations'])
            training_softmax_loss_statistics.append(epoch_training_softmax_loss / config['solver']['train_iterations'])

            epoch_training_acc = 0
            epoch_training_acc5 = 0
            epoch_training_softmax_loss = 0

            experiment.add_plot(chart_acc, training_acc_statistics, 'r.--', 'training')
            experiment.add_plot(chart_acc_5, training_acc5_statistics, 'r.--', 'training')
            experiment.add_plot(chart_softmax_loss, training_softmax_loss_statistics, 'b+--', 'training')

        # snapshot training model params
        if training_iter != 0 and  (training_iter + 1) % config['solver']['snapshot'] == 0:
            # save the model weights
            print("[INFO] snapshot the model..... ")
            experiment.add_init_net_snapshot(
                training_model.GetAllParams(),
                workspace,
                config['name'],
                tag,
                (training_iter // config['solver']['train_iterations'] + 1),
            )
            print("[INFO] snapshot the model. Done.....")

        # start to validate the model
        if training_iter != 0 and (training_iter + 1) % config['solver']['test_interval'] == 0:
            test_accuracy = 0
            test_accuracy_5 = 0
            test_softmax_loss = 0

            for test_iter in range(config['solver']['test_iterations']):
                workspace.RunNet(validation_model.net)
                accuracy = workspace.FetchBlob('accuracy')
                accuracy_5 = workspace.FetchBlob('accuracy_5')
                softmax_loss = workspace.FetchBlob('softmax_loss')

                # update counter
                test_accuracy += accuracy
                test_accuracy_5 += accuracy_5
                test_softmax_loss += softmax_loss
                experiment.add_log("[VALIDATION] accuracy: {:.4f}   accuracy_5: {:.4f}   "\
                      "softmax_loss: {:.4f}".format(
                    accuracy, accuracy_5, softmax_loss))

            # end validation
            experiment.add_log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            experiment.add_log("[VALIDATION] avg_acc: {:.4f}   avg_acc_5: {:.4f}   "\
                  "avg_softmax_loss: {:.4f}".format(
                      test_accuracy / config['solver']['test_iterations'],
                      test_accuracy_5 / config['solver']['test_iterations'],
                      test_softmax_loss / config['solver']['test_iterations'],
                  )
            )

            # plot validation statistics
            validation_acc_statistics.append(test_accuracy / config['solver']['test_iterations'])
            validation_acc5_statistics.append(test_accuracy_5 / config['solver']['test_iterations'])
            validation_softmax_loss_statistics.append(test_softmax_loss / config['solver']['test_iterations'])

            experiment.add_plot(chart_acc, validation_acc_statistics, 'c.--', 'validation')
            experiment.add_plot(chart_acc_5, validation_acc5_statistics, 'c.--', 'validation')
            experiment.add_plot(chart_softmax_loss, validation_softmax_loss_statistics, 'g+--', 'validation')

    experiment.add_log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


if __name__ == '__main__':
    config = parse_args()
    run_main(config)



