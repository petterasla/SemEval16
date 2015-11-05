# To run this code, first edit config.py with your configuration, then:
#
# mkdir data
# python twitter_stream.py -q '#climate'
# 
# It will produce the list of tweets for the query "apple" 
# in the file data/stream_apple.json
#
# Modified version of:
# https://gist.github.com/bonzanini/af0463b927433c73784d

import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import argparse
import string
import config
import json

def get_parser():
    """Get parser for command line arguments."""
    parser = argparse.ArgumentParser(description="Twitter Scraper")
    parser.add_argument("-q",
                        "--query",
                        dest="query",
                        help="Query/Filter",
                        default='-')
    parser.add_argument("-d",
                        "--data-dir",
                        dest="data_dir",
                        help="Output/Data Directory")
    return parser


class MyListener(StreamListener):
    """Custom StreamListener for streaming data."""
    def __init__(self, data_dir, query):
        query_fname = format_filename(query)
        self.outfile = "stream_%s.json" % (query_fname)

    def on_data(self, data):
        tweet = json.loads(data)
        # Exclude retweets
        if (not tweet['retweeted'] and "RT @" not in tweet['text'])  \
            and "RT@" not in tweet['text']  \
                and "RT " not in tweet['text']:
            try:
                with open(self.outfile, 'a') as f:
                    f.write(data)
                    print(data)
                    return True
            except BaseException as e:
                print("Error on_data: %s" % str(e))
                time.sleep(5)
            return True

    def on_error(self, status):
        print(status)
        return True


def format_filename(fname):
    """Convert file name into a safe string.
    Arguments:
        fname -- the file name to convert
    Return:
        String -- converted file name
    """
    return ''.join(convert_valid(one_char) for one_char in fname)


def convert_valid(one_char):
    """Convert a character into '_' if invalid.
    Arguments:
        one_char -- the char to convert
    Return:
        Character -- converted char
    """
    valid_chars = "-_.%s%s" % (string.ascii_letters, string.digits)
    if one_char in valid_chars:
        return one_char
    else:
        return '_'

@classmethod
def parse(cls, api, raw):
    status = cls.first_parse(api, raw)
    setattr(status, 'json', json.dumps(raw))
    return status

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    auth = OAuthHandler(config.consumer_key, config.consumer_secret)
    auth.set_access_token(config.access_token, config.access_secret)
    api = tweepy.API(auth)

    twitter_stream = Stream(auth, MyListener(args.data_dir, args.query))
    twitter_stream.filter(track=[args.query])