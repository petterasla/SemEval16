def processData(doc_name):
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
                dataLine[3] = dataLine[3][:-1]  # Removing the last \r on each stance
                break
            attr = line[index+1:index2]         # Storing each attr in the line (topic, tweet and stance)
            dataLine.append(attr)               # Adding the attr to the temp list
            index = index2                      # Setting prev index to newest

        data.append(dataLine)
        line = f.readline()
    f.close()
    return data

data = processData("semeval2016-task6-trainingdata.txt")

def getTopicData(topic):
    """
    Extracts the data from a given topic

    :param topic:   A string with topic name [Atheism, Climate Change is a Real Concern,
                    Feminist Movement, Hillary Clinton, Legalization of Abortion]
    :return:        A list with information about that topic
    """
    data = processData("semeval2016-task6-trainingdata.txt")
    topicData = []
    for i in range(len(data)):
        if data[i][1] == topic:
            topicData.append(data[i])
    return topicData

def getAllTweets(data_file="All"):
    """
    Extracts all the tweets from the processed data

    :param data_file:   Either a list with data (from getTopicData(topic)) or the whole dataset
    :return:            A list with all the tweets
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

    :param data_file:   Either a list with data (from getTopicData(topic)) or the whole dataset
    :return:            A list with all the stances
    """
    if data_file == "All":
        data = processData("semeval2016-task6-trainingdata.txt")
    else:
        data = data_file
    stance = []
    for i in range(len(data)):
        stance.append(data[i][3])
    return stance


# Example use
# Get all tweets and stances
allTweet = getAllTweets("All")
allStance = getAllStances("All")

#Gets all the tweets from Climate change
tweets = getAllTweets(getTopicData("Climate Change is a Real Concern"))

#Gets all the stance from Climate change
stance = getAllStances(getTopicData("Climate Change is a Real Concern"))

