#Bayes Sentiment Classifier
#By: Marc Baiza and Stephen Oh
import codecs
import math

######################
#   Pre-Processing   #
######################
def pre_proccessing():
    print("Start pre-processing data...\n")

    trainingSet = "trainingSet.txt"

    #Gets the training data from the provided text file
    trainingData = get_training_data(trainingSet)

def get_training_data(trainingFileName):
    print("Retrieving training data...")

######################
#   Classification   #
######################