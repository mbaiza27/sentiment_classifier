#Bayes Sentiment Classifier
#By: Marc Baiza and Stephen Oh
import codecs
import math

######################
#   Pre-Processing   #
######################
#Some of the edgecase characters for words such as wont and won't
edgeCaseChars = ('~', '^', "'", '`' )

#The rest of the possible characters to be stripped
punctuationChars = ('!', '@', '#', '$', '%', '&', '*', '(', ')', '-', '_', '+', '=', '{', '}', '[', ']', '\'', '|', ';', ':', ',', '<', '>', '.', '/', '?')

#tab
tab = "\t"

#vocab
vocab = set()


def pre_proccessing():
    print("Start pre-processing data...\n")

    trainingSet = "trainingSet.txt"

    #Gets the training data from the provided text file in the form of a list
    trainingData = get_training_data(trainingSet)

    #Strip the necessary symbols from the lins for better accuracy
    clean_training_data(trainingData)
    #print(vocab) #For testing

########################################
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
##########################################
def clean_training_data(trainingData):
    print("Stripping unwanted symbols...")
    #Initialize Global variable for vocabulary
    global vocab

    for line in trainingData:
        vocabWord = str()
        
        for char in line:
            if char not in edgeCaseChars:
                #This means that there is no characters interupting words
                if char == tab:
                    #Hitting a tab means we have cycled through the line
                    break
                elif char in punctuationChars:
                    #Means that we should add the word if it hits any of these characters
                    wordLength = len(vocabWord)

                    if wordLength > 0:
                        vocab.add(vocabWord)
        
                    #Reset the word for the next word in the line
                    vocabWord = str()

                else: #append character to word
                    lowerChar = char.lower()
                    vocabWord += lowerChar
    
    #Print vocab for testing
    #print(vocab)
    #After stripping is finished make a sorted list
    vocab = list(vocab)
    #Now sort the list
    vocab.sort()
    #print(vocab)
##########################################


######################
#   Classification   #
######################



###################
#   Run program   #
###################
pre_proccessing()
