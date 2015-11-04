import json

def readTweets(filename):
    tweets = []
    with open(filename + '.json') as f:
        for line in f:
            tweet = json.loads(line)
            tweets.append(tweet['text'])
            #print tweet['text'] + "\n"
            #print "##################################################################################################"

    return tweets

#print readTweets('stream__climate')