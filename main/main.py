
from dagster import get_dagster_logger, job, op, In

from my_mnist import *

@op
def setHyper():
    epoch = 20
    hyper = { "batch_size": 50, "num_classes": 10, "learning_rate": 0.001, "num_epochs": epoch }

    return hyper

@op(ins={'msg': In(int)})
def print_test(msg):
    logger = get_dagster_logger()
    logger.info(f"sunny dbg: {msg}")   


@job
def startTrainMnist():

    for i in range(2):
        epoch = i*2 + 10
        hyper = { "batch_size": 50, "num_classes": 10, "learning_rate": 0.001, "num_epochs": epoch }
        #hyper = [ 50, 10, 0.001, epoch ]
        hyper = setHyper()
        #print_test(100)
        #setHyperParam(hyper)
        doTrainMNIST(hyper)
