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
                #dataLine[3] = dataLine[3][:-1]  # Removing the last \r on each stance
                break
            attr = line[index+1:index2]         # Storing each attr in the line (topic, tweet and stance)
            dataLine.append(attr.decode("ISO-8859-1"))               # Adding the attr to the temp list
            index = index2                      # Setting prev index to newest

        data.append(dataLine)
        line = f.readline()
    f.close()
    return data

#data = processData("semeval2016-task6-trainingdata.txt")

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
def getAllTweetsWithoutHashOrAlphaTag(tweet_list):
    """
    Extracts all the tweets and removes the hashtags and @ tags from the processed data

    :param tweet_list:   Either a list with data (from getTopicData(topic)) or the whole dataset
    :return tweets:     A list with all the tweets. No hashtags or @tags.
    """

    tweets = []
    for line in tweet_list:
        indices = []
        s = line
        hashStartIndex = line.find("#")
        stopIndex = line.find("#SemST")
        while hashStartIndex < stopIndex:
            hashStopIndex = line.find(" ", hashStartIndex)
            indices.append([hashStartIndex, hashStopIndex])
            hashStartIndex = line.find("#", hashStartIndex+1)
        for i in indices:
            hash = line[i[0]:i[1]]
            s = s.replace(hash,"")


        stopIndex = s.find("#SemST")
        tagStartIndex = s.find("@")
        indices = []
        while tagStartIndex < stopIndex and tagStartIndex != -1:
            tagStopIndex = s.find(" ", tagStartIndex)
            indices.append([tagStartIndex, tagStopIndex])
            tagStartIndex = s.find("@", tagStartIndex+1)
        s1 = s
        for i in indices:
            tag = s[i[0]:i[1]]
            s1 = s1.replace(tag, "")
        stopIndex = s1.find("#SemST")
        s1 = s1[:stopIndex]
        tweets.append(s1)
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
    :return wordList:   A list of words with no hashtags
    """
    wordList = []
    for tag in hashes:
        if tag[1:].isupper() or tag[1:].islower():
            wordList.append(tag[1:])
        else:
            words = re.findall('[A-Z][^A-Z]*',tag[1:])
            for word in words:
                wordList.append(word)

    return wordList

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
TOPIC = "All"
TOPIC2 = "Climate Change is a Real Concern"
TOPIC1 = "Atheism"
TOPIC3 = "Feminist Movement"
TOPIC4 = "Hillary Clinton"
TOPIC5 = "Legalization of Abortion"

# Get all tweets and stances
#allTweet = getAllTweets(getTopicData(TOPIC))
#allStance = getAllStances(getTopicData(TOPIC5))

#Get all the tweets from Climate change
#tweets = getAllTweets(getTopicData(TOPIC2))
#parsed_tweets = getAllTweetsWithoutHashOrAlphaTag(tweets[8:10])

#Get all the stance from Climate change
#stance = getAllStances(getTopicData(TOPC2))

#Get all the hashtags from tweets (within a topic)
#hashtags = getAllHashtags(getAllTweets(getTopicData(TOPIC5)))
#hashtags_in_one_list = [hash for hashes in hashtags for hash in hashes]

#Decrypt all the hashtags from the topic: Climate Change is a real consern
#words = decryptAllHashtags(hashtags)


