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

    #Open file
    trainingFile = open(trainingFileName, encoding='utf-8')
    
    #Read data from file
    trainingData = trainingFile.readlines()

    #Print data for testing
    #print(trainingData)

    #Close file
    trainingFile.close()

    return trainingData

######################
#   Classification   #
######################



###################
#   Run program   #
###################
pre_proccessing()
