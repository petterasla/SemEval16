import re
import random
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment import vader as vader
# When using vader.SentimentIntensityAnalyzer() sentiment methods, you might have to download and store this in
# /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/nltk/sentiment/vader_lexicon.txt'
# https://github.com/nltk/nltk/blob/develop/nltk/sentiment/vader_lexicon.txt
"""
If you use the VADER sentiment analysis tools, please cite:

Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
Sentiment Analysis of Social Media Text. Eighth International Conference on
Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
"""
from nltk.tokenize import word_tokenize
from collections import Counter
import os
import json
import unicodedata
import string


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

def getTopicTestData(topic):
    """
    Extract test data from a given topic

    :param topic:   A string with the topic. [All, Atheism, Climate Change is a Real Concern,
                    Feminist Movement, Hillary Clinton, Legalization of Abortion].
    :return:        Returns a list with the information with the related topic
    """
    data = processData("../eval_semeval16_task6/SemEval2016-Task6-subtaskA-testdata.txt")
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

# Get all the tweets from Climate change
#tweets = getAllTweets(getTopicData(TOPIC2))
#parsed_tweets = getAllTweetsWithoutHashOrAlphaTag(tweets[8:10])

# Get all the stance from Climate change
#stance = getAllStances(getTopicData(TOPC2))

# Get all the hashtags from tweets (within a topic)
#hashtags = getAllHashtags(getAllTweets(getTopicData(TOPIC5)))
#hashtags_in_one_list = [hash for hashes in hashtags for hash in hashes]

# Decrypt all the hashtags from the topic: Climate Change is a real consern
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


def train_test_split_on_stance(data, test_data, favor_p, against_p, none_p):
    """
    This method takes a data list and splits into a training set and test set. It uses the percentage parameter
    to set the size of each set based on their stance. i.e: FAVOR = 30, AGAINST = 15 and NONE = 45 and percentage
    is set to 0.33, then the test set will consist of 10 FAVOR, 5 AGAINST and 15 NONE.

    :param data:            A list with data that want to be split in training and test set
    :param test_data:       A list with the test data want to split up.
    :param favor_p:         How big you want the FAVOR test set to be in percentage.
    :param against_p:       How big you want the AGAINST test set to be in percentage.
    :param none_p:          How big you want the NONE test set to be in percentage.
    :return:                Returns a list [train, test] which include a split of training and test data
    """
    favor_list, against_list, none_list = [], [], []
    for t in test_data:
        if (t[3] == 'FAVOR'):
            favor_list.append(t)
        elif (t[3] == 'AGAINST'):
            against_list.append(t)
        else:
            none_list.append(t)
    k_favor, k_against, k_none = int(favor_p*len(favor_list)), int(against_p*len(against_list)), int(none_p*len(none_list))
    favor_test, against_test, none_test = favor_list[:k_favor], against_list[:k_against], none_list[:k_none]
    test = against_test + favor_test + none_test
    test_ids = [test[x][0] for x in range(len(test))]
    train = [data[x] for x in range(len(data)) if data[x][0] not in test_ids]
    print "len of data: "+ str(len(data)) + "\t len of favor: " + str(len(favor_list)) + "\t len of against: " \
          + str(len(against_list))+ "\t len of none: " + str(len(none_list)) + \
          "\t Total: " + str(len(favor_list) + len(against_list) + len(none_list))
    print "len of train: "+ str(len(train)) + "\t len of test: " + str(len(test)) + "\t total: " \
          + str(len(train) + len(test))
    return [train, test]
#train_test_split_on_stance(getTopicData(TOPIC2), TOPIC2, 0.2, 0.5, 0.3)


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


def lemmatizing(data):
    lem = WordNetLemmatizer()
    new_data = []
    for tweet in data:
        tweet = tweet.lower()
        words = tweet.split(" ")
        lemmatized = []
        for word in words:
            lemmatized.append(lem.lemmatize(word))
        new_tweet = ""
        for word in lemmatized:
            new_tweet = new_tweet + " " + word
        new_data.append(new_tweet)
    return new_data
#data = getAllTweets(getTopicData(TOPIC2))
#print data[5]
#k = lemmatizing(data)
#print k[5]


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
#count_hashtags(TOPIC5)


def processAbstracts():
    category_approval = 8       # Including the number: 2, 3, 4, 5, 6, 7. See TCP paper for more info.
    favor_endorsement = 2       # Including the number: 1, 2 or 3
    against_endorsement = 5     # Including the number: 5, 6 or 7
    #path = os.path.abspath('')
    with open("../BaselineSystem/abstracts_with_meta.txt", "r") as articleJson:
        articleInfo = json.load(articleJson)
        articleJson.close()
    favor_abstracts = []
    against_abstracts = []
    for article in articleInfo:
        #print article["Status"]["isMetaTitleEqualToOriginal"]
        if (article["Status"]["isMetaTitleEqualToOriginal"] == "False"):
            continue
        else:
            if (int(article["Category"]) >= category_approval):
                continue
            else:
                if (int(article["Endorsement"]) <= favor_endorsement):
                    favor_abstracts.append(article["Abstract"])
                elif (int(article["Endorsement"]) >= against_endorsement):
                    against_abstracts.append(article["Abstract"])

    print "favor abstracts: " + str(len(favor_abstracts)) + "\t against abastracts: " + str(len(against_abstracts))
    return [favor_abstracts, against_abstracts]
#processAbstracts()

def createClimateLexicon(topXwords = 100):
    """
    This method uses (for now) abstracts pulled from The Consensus Project as data to create a lexicon
    based on the most frequent words

    :param topXwords:   Number of words that should be included in the lexicon
    :return:            Returns a lexicon (list) containing with x most frequent words
    """
    with open("../BaselineSystem/abstracts_with_meta.txt", "r") as abstractInfo:
        abstracts = json.load(abstractInfo)
        abstractInfo.close()



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


def determineNegationFeature(text):
    """
    Creates feature (0 or 1) for whether the tweet contains negated segments or not

    :param doc_name:    Tweet as string
    :return:            Binary value (0 or 1)
    """
    feature = vader.negated(text, include_nt=True)
    if feature:
        return 1
    else:
        return 0


def lengthOfTweetFeature(text):
    """
    Creates a normalized feature between 0 and 1 for the length of the tweet

    :param doc_name:    Tweet as string
    :return:            Float between 0 and 1
    """
    return len(text)/140.0  #Maximunm length of a tweet


def numberOfTokensFeature(text):
    """
    Finds the number of tokens in a tweet

    :param doc_name:    Tweet as string
    :return:            Integer
    """
    return len(word_tokenize(text))


def numberOfCapitalWords(text):
    """
    Finds the number of capital words in a tweet

    :param text:        Tweet as string
    :return:            Integer number of capital words
    """
    capitalized = []
    if vader.allcap_differential(text.split(" ")):
        for word in text.split(" "):
            if word.isupper():
                capitalized.append(word)
        return len(capitalized)
    else:
        return 0
#print numberOfCapitalWords("HEI hvordan Gaar DET!")

def numberOfNonSinglePunctMarks(text):
    """
    Finds the number of non-single punctuation marks in a tweet

    :param text:        Tweet as a string
    :return:            A list where first element is an integer number of punctuation marks and last element
                        is whether the last punctuation mark is ! or ? (either 0 or 1)
    """
    counter = 0
    isQuestionMarkOrExclamationMarkLast = 0
    for word in text.split(" "):
        for i in range(len(word)):
            # string.punctuation contains: !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
            if (word[i] in string.punctuation) and not (i == len(word)-1) and (word[i+1] in string.punctuation):
                counter += 1
                if word[len(word)-1] in "!?":
                    isQuestionMarkOrExclamationMarkLast = 1
                break           # Break here otherwise a word like: "hello!!!!" will count as 3

    return [counter, isQuestionMarkOrExclamationMarkLast]
#print numberOfNonSinglePunctMarks("hei hei!!#% Dette er en test !#!")


def numberOfLengtheningWords(text):
    """
    Counts the number of words that are longer than usual like: cooool.
    It will count every word that has (at least) three of the same consecutive letters

    :param text:        Tweet as a string
    :return:            Integer number of lengthening words.
    """
    counter = 0
    for word in text.split(" "):
        for i in range(len(word)):
            letter = word[i]
            if (len(word) > 2) and (i < len(word)-2) and (word[i+1] == letter) and (word[i+2] == letter):
                counter +=1
                break
    return counter
#print numberOfLengtheningWords("cooolll balll hey how arre you")


def determineSentiment(text):
    """
    Determines the sentiment of a tweet based on the vader module

    :param text:        Tweet as a string
    :return:            Returns a dict of {neg:x, neu:y, pos:z, compound:w}
    """
    return vader.SentimentIntensityAnalyzer().polarity_scores(text)

