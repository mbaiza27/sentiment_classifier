#Bayes Sentiment Classifier
#By: Marc Baiza and Stephen Oh
import math
import numpy as np
######################
#   Pre-Processing   #
######################
#Some of the edgecase characters for words such as wont and won't
edgeCaseChars = ('~', '^', "'", '`' )

#The rest of the possible characters to be stripped
punctuationChars = ('!', '@', '#', '$', '%', '&', '*', '(', ')', '-', '_', '+', '=', '{', '}', '[', ']', '\\', '|', ';', ':', ',', '<', '>', '.', '/', '?', ' ', '"')

#tab
tab = "\t"

#vocab
vocab = set()


def pre_proccessing():
    print("Start pre-processing data...\n")

    trainingSet = "trainingSet.txt"
    testSet = "testSet.txt"
    trainingOutput = "training_out.txt"
    testOutput = "test_out.txt"

    #Gets the training data from the provided text file in the form of a list
    trainingData = get_training_data(trainingSet)
    #print(trainingData)
    #Strip the necessary symbols from the lins for better accuracy
    clean_training_data(trainingData)
    #print(vocab) #For testing

    #Now convert training/test data into a set of features
    featuredTrainingData = convert_to_features(trainingData)
    #print(featuredTrainingData)

    #Now do the same thing for test data
    testData = get_training_data(testSet)
    featuredTestData = convert_to_features(testData)

    #Now print output to files
    write_output(trainingOutput, featuredTrainingData)
    write_output(testOutput, featuredTestData)

    return featuredTrainingData, featuredTestData

########################################
def get_training_data(trainingFileName):
    print("Retrieving training data...")

    #Open file
    trainingFile = open(trainingFileName)

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

                else: #append char to word
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
def convert_to_features(data):
    print("converting to features...")
    vectors = list()
    for line in data:
        #Similar process to clean_training_data
        wordList = list()
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
                        wordList.append(vocabWord)

                    #Reset the word for the next word in the line
                    vocabWord = str()

                else: #append char to word
                    lowerChar = char.lower()
                    vocabWord += lowerChar

        #Now that we have the list of words we can vectorize them
        #print(wordList)
        vector = [0 for i in range(len(vocab))]

        for word in wordList:
            if word in vocab:
                wordIdx = vocab.index(word)
                vector[wordIdx] = 1
            else:
                continue

        sentimentIdx = len(line) - 3
        sentiment = int(line[sentimentIdx])

        vector.append(sentiment)
        vectors.append(vector)

    return vectors
###################################################################
def write_output(outFile, featuredVectors):
    file = open(outFile, "w")

    for word in vocab:
        file.write(word)
        file.write(',')

    file.write("classlabel\n")

    for i in featuredVectors:
        for string in i:
            file.write(str(string))
            file.write(",")
        file.write("\n")

    file.close()

######################
#   Classification   #
######################

def predict_labels(trainData, testData):
    trainData = np.array(trainData)
    testData = np.array(testData)
    testLabels = []
    trueTestLabels = testData[:,-1]
    trainLabels = trainData[:,-1]
    numFeatures = len(testData[0]) - 1
    probArray = np.zeros(shape=(4,numFeatures))
    print("First probArray:", probArray)
    for i in range(numFeatures):  #for every word in sentence
        totalzerosbad, totalzerosgood, totalonesbad,totalonesgood = 0,0,0,0
        #The array showing whether the word appears in the training sentences or not
        wordLabels = trainData[:,i]
        # print(len(wordLabels))
        # print(wordLabels)
        for j in range(len(trainLabels)):   #for each training set of the word
            if(trainLabels[j] == 0 and wordLabels[j] == 0):
                totalzerosbad += 1
            elif(trainLabels[j] == 0 and wordLabels[j] == 1):
                totalonesbad += 1
            elif(trainLabels[j] == 1 and wordLabels[j] == 0):
                totalzerosgood += 1
            elif(trainLabels[j] == 1 and wordLabels[j] == 1):
                totalonesgood += 1
        # print(totalzerosbad, totalonesbad, totalzerosgood, totalonesgood)
        zerobadprob = np.log((totalzerosbad + 1)/(len(trainLabels)-np.count_nonzero(trainLabels)+2))
        onebadprob = np.log((totalonesbad + 1)/(len(trainLabels)-np.count_nonzero(trainLabels)+2))
        zerogoodprob = np.log((totalzerosgood + 1)/(np.count_nonzero(trainLabels)+2))
        onegoodprob = np.log((totalonesgood + 1)/(np.count_nonzero(trainLabels)+2))
        probArray[0][i], probArray[1][i], probArray[2][i], probArray[3][i] = zerobadprob, onebadprob, zerogoodprob, onegoodprob
        # print("last probArray:", probArray)
        # break
        # print("oneprob", oneprob, "zeroprob", zeroprob)


    for sentence in testData:   #for each sentence in testData
        #Get initial probabilities of negative review vs positive review
        badprob = np.log((len(trainLabels)-np.count_nonzero(trainLabels))/len(trainLabels))
        goodprob = np.log(np.count_nonzero(trainLabels)/len(trainLabels))
        for i in range(len(sentence)-1):
            if (sentence[i] == 0):
                badprob +=probArray[0][i]
                goodprob += probArray[2][i]
            else:
                badprob += probArray[1][i]
                goodprob += probArray[3][i]
        # print("badprob: ", badprob, "goodprob", goodprob)


        if(badprob > goodprob):
            testLabels.append(0)
        else:
            testLabels.append(1)
    print(np.count_nonzero(np.equal(trueTestLabels, testLabels))/len(testLabels))
        # break
###################
#   Run program   #
###################
train, test = pre_proccessing()
predict_labels(train, test)
# print(train[0])
# print(test[0])
# print("Vocab: ", vocab)
