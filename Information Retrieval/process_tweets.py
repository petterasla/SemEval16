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