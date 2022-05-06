
from dagster import get_dagster_logger, job, op, In

from foodTrain import *
from myFoodTrain import *


@op
def setHyper1():
    epoch = 30
    hyper = { "batch_size": 50, "num_classes": 10, "learning_rate": 0.001, "num_epochs": epoch }

    return hyper

@op
def setHyper2():
    epoch = 10
    hyper = { "batch_size": 50, "num_classes": 10, "learning_rate": 0.001, "num_epochs": epoch }

    return hyper

@op
def setHyper3():
    epoch = 15
    hyper = { "batch_size": 50, "num_classes": 10, "learning_rate": 0.001, "num_epochs": epoch }

    return hyper

@op(ins={'msg': In(int)})
def print_test(msg):
    logger = get_dagster_logger()
    logger.info(f"sunny dbg: {msg}")   


@job
def startTrainFoodObjectDetection():

    hyper = setHyper1()
    doTrain(hyper)

    hyper = setHyper2()
    doTrain(hyper)
        

