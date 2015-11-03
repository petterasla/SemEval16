import re
import random
import numpy as np
from nltk.stem.porter import PorterStemmer
from collections import Counter

TOPIC = "All"
TOPIC1 = "Atheism"
TOPIC2 = "Climate Change is a Real Concern"
TOPIC3 = "Feminist Movement"
TOPIC4 = "Hillary Clinton"
TOPIC5 = "Legalization of Abortion"

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

def train_test_split(data, percentage, test_topic):
    """
    This method takes a data list and splits into a training set and test set. It uses the percentage parameter
    to set the size of the test set. If the test_topic parameter is 'All', it will take the whole data set and
    take random samples from every topic. If not, it will only have

    :param data:            A list with data that want to be split in training and test set
    :param percentage:      How big you want the test set to be in percentage.
    :param test_topic:      Name of the topic you want to have in the test set
    :return:                Returns a list [train, test] which include a split of training and test data
    """
    if (test_topic == "All"):
        k = int(percentage*len(data))
        random.shuffle(data)
        test = data[:k]
        train = data[k:]
        return [train, test]
    else:
        topic_data = getTopicData(test_topic)
        k = int(percentage*len(topic_data))
        test = topic_data[:k]
        test_ids = [test[x][0] for x in range(len(test))]
        train = [data[x] for x in range(len(data)) if data[x][0] not in test_ids]
        return [train, test]


def stemming(data):
    pt = PorterStemmer()
    new_data = []
    for tweet in data:
        words = tweet.split(" ")
        stemmed_words = []
        for word in words:
            stemmed_words.append(pt.stem(word))
        new_tweet = ""
        for word in stemmed_words:
            new_tweet = new_tweet + " " + word + " "
        new_data.append(new_tweet)

    return new_data

#data = getAllTweets(getTopicData(TOPIC2)[0:3])
#print data
#k = stemming(data)
#print k

def count_hashtags(topic):

    tweets = getAllTweets(getTopicData(topic))
    stance = getAllStances(getTopicData(topic))
    hashtags = getAllHashtags(tweets)
    hashtag_list_favor = []
    hashtag_list_against = []
    hashtag_list_none = []

    for i in range(len(hashtags)):
        if len(hashtags[i]) > 0:
            if (stance[i] == "FAVOR\r"):
                for hash in hashtags[i]:
                    hashtag_list_favor.append(hash.lower())
            elif (stance[i] == "AGAINST\r"):
                for hash in hashtags[i]:
                    hashtag_list_against.append(hash.lower())
            else:
                for hash in hashtags[i]:
                    hashtag_list_none.append(hash.lower())
    favor_count = Counter(hashtag_list_favor)
    against_count = Counter(hashtag_list_against)
    none_count = Counter(hashtag_list_none)
    print "Favor: "
    print favor_count
    print "Against: "
    print against_count
    print "None: "
    print none_count

count_hashtags(TOPIC5)