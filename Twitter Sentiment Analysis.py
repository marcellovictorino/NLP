

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from twitterAPI import *
# import sentiment_mod as s # replace with the actual module name

# consumer key, consumer secret, access token, access secret. All imported from twitterAPI module

class listener(StreamListener):
    
    def on_data(self, data):
        try:
            all_data = json.loads(data)
            tweet = all_data["text"]
            # Testing: filtering specific words of interest
            # if 'traffic' in tweet.lower():
            #     print(tweet)
            # else:
            #     print('Nothing about traffic ...')
            # sentiment_value, confidence = s.sentiment(tweet)
            # print(tweet, sentiment_value, confidence)

            # saving to file for future analysis
            with open('twitterData_COSL.txt','a') as fh:
                fh.write(tweet+'\n')

            print(tweet)

            # if confidence*100 >= 80:
            # 	output = open("twitter-out.txt","a")
            # 	output.write(sentiment_value)
            # 	output.write('\n')
            # 	output.close()
            return True
        except:
            pass

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
twitterStream = Stream(auth, listener())

#########################################
# Can only use one of the following filters

### 1) Look for specific words or Hashtags
# twitterStream.filter(track=["traffic"])
# twitterStream.filter(track=['#SugarLand', '#COSL'])

### 2) Only look for specific user tweets
# twitterStream.filter(follow=[''])

### 3) Only look for tweet from a specific location (bounding box)
twitterStream.filter(locations=[-95.696438,29.540022,-95.576973,29.654943]) # Sugar Land Area. Verified
