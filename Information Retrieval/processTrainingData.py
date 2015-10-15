import re

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

def getAllHashtags(tweet_list):
    """
    Extracts all the hashtags from the tweets

    :param tweet_list:   A list with lists of all the tweets
    :return hashtags:    A list with lists of the hashtags for each tweet
    """
    data = tweet_list
    hashtags = []
    for line in data:
        hash = []
        hashStartIndex = line.find("#")
        stopIndex = line.find("#SemST")
        while hashStartIndex < stopIndex:
            hashStopIndex = line.find(" ", hashStartIndex)
            hash.append(line[hashStartIndex:hashStopIndex])
            hashStartIndex = line.find("#", hashStartIndex+1)
        hashtags.append(hash)
    return hashtags


def decryptHashtags(hashes):
    """
    Takes a single list of hashtags, removes the hashes and tries to split the tags into words

    :param hashes:      A list of hashes
    :return wordList:   A list of words with no duplicates nor hashtags
    """
    wordList = set()
    for tag in hashes:
        if tag[1:].isupper() or tag[1:].islower():
            wordList.add(tag[1:])
        else:
            words = re.findall('[A-Z][^A-Z]*',tag[1:])
            for word in words:
                wordList.add(word)

    return list(wordList)

def decryptAllHashtags(hashtag_list):
    """
    Takes a list with lists of hashtags and decrypts into words using decryptHashtags(hashes).

    :param hashtag_list:    A list with lists of hashtags
    :return wordList:       A list of words without hashtags and no duplicates
    """
    wordList = set()
    for hashes in hashtag_list:
        words = decryptHashtags(hashes)
        for word in words:
            wordList.add(word)
    return list(wordList)


# Example use
# Get all tweets and stances
#allTweet = getAllTweets("All")
#allStance = getAllStances("All")

#Get all the tweets from Climate change
#tweets = getAllTweets(getTopicData("Climate Change is a Real Concern"))

#Get all the stance from Climate change
#stance = getAllStances(getTopicData("Climate Change is a Real Concern"))

#Get all the hashtags from tweets (in this case, under the topic: Climate Change a real consern[0:10])
#hashtags = getAllHashtags(getAllTweets(getTopicData("Climate Change is a Real Concern"))[0:10])

#Decrypt all the hashtags from the topic: Climate Change is a real consern
#words = decryptAllHashtags(hashtags)
