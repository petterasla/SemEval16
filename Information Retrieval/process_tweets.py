import json

def readTweets(filename):
    tweets = []
    with open(filename + '.json') as f:
        for line in f:
            tweet = json.loads(line)
            tweets.append(tweet['text'])
            #print tweet['text'] + "\n"
            #print "##################################################################################################"

    f.close()
    return tweets

def readTweetsAndIDs(filename):
    tweets = []
    with open(filename + '.json') as f:
        for line in f:
            tweet = json.loads(line)
            info = []
            info.append(tweet['id'])
            info.append(tweet['text'])
            tweets.append(info)
            #print tweet['text'] + "\n"
            #print "##################################################################################################"

    f.close()
    return tweets


def removeDuplicates(filename):
    with open(filename + '.json') as f:
        # load json objects to dictionaries
        jsons = map(json.loads, f)

    uniques = {x['text']: x for x in jsons}

    # write to new json file
    with open(filename + '_clean.json' ,'w') as nf:
        for js in uniques.values():
            nf.write(json.dumps(js))
            nf.write('\n')

    f.close()
    nf.close()
    print "Removed " + str(len(readTweets(filename))-len(readTweets(filename+'_clean'))) + " tweets"

# Examples:
#print readTweets('stream__climate')
#removeDuplicates('stream__chemtrails')
#removeDuplicates('stream__climate')
#removeDuplicates('stream__climatechange')
#removeDuplicates('stream__environment')


# Tweets with id that are 'against' in: stream__cliatechange_clean.json:

# [662022121904840704, 662020009183551488, 662016401067155456, 662020009783373824, 662016718303195136]

# Tweets with id that are 'favor' in: stream__cliatechange_clean.json:

# [662015471261245441, 662015310367752192, 662018031292256256, 662018963916722176, 662015805635325952, 662020824115896320, 662022433453572097 -
"""
tweets = readTweetsAndIDs("stream__climatechange_clean")
for t in range(130,150):
    try:
        print tweets[t][0]
        print tweets[t][1]
    except:
        print "error"
        """