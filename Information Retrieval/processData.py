def processData(doc_name):
    """
    Process the text file from semEval task 6 training data to list

    :param doc_name:    String containing the document file name
    :return:            A list with lists of data separated by id, topic,tweet and stance
    """
    data = []
    f = open(doc_name,"r")
    line = f.readline()     #Firste line states id, year, title of the article
    line = f.readline()     #Starts reading from the second line
    while line:
        dataLine = []
        index = line.index("\t")                # Finding the index of the first tab \t
        id = line[0:index]                      # Storing the ID
        dataLine.append(id)                     # Adding the ID to a temporary list
        while index < len(line):                # Looping through the rest of the tabs: \t
            index2 = line.find("\t", index+1)
            if index == -1:
                #dataLine[3] = dataLine[3][:-1]  # Removing the last \r on each stance
                break
            attr = line[index+1:index2]         # Storing each attr in the line (topic, tweet and stance)
            dataLine.append(attr.decode("ISO-8859-1"))               # Adding the attr to the temp list
            index = index2                      # Setting prev index to newest

        data.append(dataLine)
        line = f.readline()
    f.close()
    return data


def getTopicData(topic):
    """
    Extracts the data from a given topic

    :param topic:   A string with topic name [All, Atheism, Climate Change is a Real Concern,
                    Feminist Movement, Hillary Clinton, Legalization of Abortion]
    :return:        A list with information about that topic
    """
    data = processData("semeval2016-task6-trainingdata.txt")
    if (topic == "All"):
        return data
    topicData = []
    for i in range(len(data)):
        if data[i][1] == topic:
            topicData.append(data[i])
    return topicData


def getAllTweets(data_file="All"):
    """
    Extracts all the tweets from the processed data

    :param data_file:   Either a list with data (from getTopicData(topic)) or the whole dataset
    :return tweets:     A list with all the tweets
    """
    if data_file == "All":
        data = processData("semeval2016-task6-trainingdata.txt")
    else:
        data = data_file
    tweets = []
    for i in range(len(data)):
        tweets.append(data[i][2])
    return tweets


def getAllStances(data_file="All"):
    """
    Extracts all the stances from the processed data

    :param data_file:   Either a list with data (from getTopicData(topic)) or the whole dataset 'All'
    :return stance:     A list with all the stances
    """
    if data_file == "All":
        data = processData("semeval2016-task6-trainingdata.txt")
    else:
        data = data_file
    stance = []
    for i in range(len(data)):
        stance.append(data[i][3])
    return stance


def convertStancesToNumbers(allStances):
    """
    Converts textual stance (FAVOR, NONE, AGAINST) to numbers (2, 1, 0)

    :param doc_name:    List of textual stances
    :return:            List of numeric stanecs
    """
    numberedStances = []
    for i in range(len(allStances)):
        if allStances[i] == u'FAVOR\r':
            numberedStances.append(2)
        elif allStances[i] == u'NONE\r':
            numberedStances.append(1)
        else:
            numberedStances.append(0)

    return numberedStances


def convertStancesToText(allNumberedStances):
    """
    Converts numeric stance (2, 1, 0) to textual stance (FAVOR, NONE, AGAINST)

    :param doc_name:    List of numeric stances
    :return:            List of textual stanecs
    """
    textStances = []
    for i in range(len(allNumberedStances)):
        if allNumberedStances[i] == 2:
            textStances.append("FAVOR")
        elif allNumberedStances[i] == 1:
            textStances.append("NONE")
        else:
            textStances.append("AGAINST")

    return textStances
