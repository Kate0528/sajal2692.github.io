---
layout: notebook
title: Finding Donors for Charity using Machine Learning
skills: Python, Scikit-learn, Decision Trees, SVM, Adaboost
external_type: Github
external_url: https://github.com/Kate0528/Twitter-Sentiment-Analysis
description: An Adaboost classifier to accurately predict whether an individual makes more than $50,000, and identify likely donors for a non-profit organisation.
---
---

Twitter represents a fundamentally new instrument to make social measurements. Millions of people voluntarily express opinions across any topic imaginable --- this data source is incredibly valuable for both research and business.

For example, researchers have shown that the "mood" of communication on twitter reflects biological rhythms and can even used to predict the stock market. A student here at UW used geocoded tweets to plot a map of locations where "thunder" was mentioned in the context of a storm system in Summer 2012.Researchers from Northeastern University and Harvard University studying the characteristics and dynamics of Twitter have [an excellent resource](http://www.ccs.neu.edu/home/amislove/twittermood/) for learning more about how Twitter can be used to analyze moods at national scale.

In this project, I accessed Twitter API with Python to fetch raw data. Then, I runed several algorithms to the following obejctives.

1. Calculated the sentiment score for each tweet based on a pre-computed sentiment dictionary.

2. Derived the sentiment for new terms.

3. Computed the term frequency histogram of the livestream data I fetched before.

4. Analyzed the relationship between location and mood based on a sample of twitter data.

5. Get ten most frequently occurring hashtags from the data I gathered before.

Understanding the sentiments of a randomly sampled data can help us better understand different happiness level of US states. This project provides a descriptive approach to the popular words and hashtags in a given period. This sort of task can arise in either a non-profit or a for-profit setting, where organizations construct a model that reflects the emotions at different time points in a large dataset. We can also use these code to estimate the public's perception (the sentiment) of a particular term or phrase.

The dataset for this project originates from the [Twitter developer platform](https://developer.twitter.com/en/docs/basics/getting-started). 

----
## Fetch the livestream data from Twitter API

The JSON data I fetched looks like this. I fetched around 100,000 tweets from the Twitter API.
```json
{"created_at":"Wed Oct 10 21:40:22 +0000 2018","id":1050138998998532104,"id_str":"1050138998998532104","text":"RT @morenawer82: Si escuchas un disco entero de Maluma, se te olvida la tabla del 3.","source":"\u003ca href=\"https:\/\/mobile.twitter.com\" rel=\"nofollow\"\u003eTwitter Lite\u003c\/a\u003e","truncated":false,"in_reply_to_status_id":null,"in_reply_to_status_id_str":null,"in_reply_to_user_id":null,"in_reply_to_user_id_str":null,"in_reply_to_screen_name":null,"user":{"id":334983966,"id_str":"334983966","name":"eLena Sanper","screen_name":"UnAnimaMundi","location":"Madrid","url":null,"description":"Insertar sexo. Insertar profesi\u00f3n. Insertar frase pseudofilos\u00f3fica.","translator_type":"none","protected":false,"verified":false,"followers_count":28,"friends_count":55,"listed_count":0,"favourites_count":85,"statuses_count":91,"created_at":"Wed Jul 13 23:52:55 +0000 2011","utc_offset":null,"time_zone":null,"geo_enabled":false,"lang":"es","contributors_enabled":false,"is_translator":false,"profile_background_color":"1A1B1F","profile_background_image_url":"http:\/\/abs.twimg.com\/images\/themes\/theme1\/bg.png","profile_background_image_url_https":"https:\/\/abs.twimg.com\/images\/themes\/theme1\/bg.png","profile_background_tile":true,"profile_link_color":"19CF86","profile_sidebar_border_color":"181A1E","profile_sidebar_fill_color":"252429","profile_text_color":"666666","profile_use_background_image":true,"profile_image_url":"http:\/\/pbs.twimg.com\/profile_images\/1031302538908323840\/ayDcAqJG_normal.jpg","profile_image_url_https":"https:\/\/pbs.twimg.com\/profile_images\/1031302538908323840\/ayDcAqJG_normal.jpg","default_profile":false,"default_profile_image":false,"following":null,"follow_request_sent":null,"notifications":null},"geo":null,"coordinates":null,"place":null,"contributors":null,"retweeted_status":{"created_at":"Wed Oct 03 19:15:34 +0000 2018","id":1047565841204695040,"id_str":"1047565841204695040","text":"Si escuchas un disco entero de Maluma, se te olvida la tabla del 3.","source":"\u003ca href=\"http:\/\/twitter.com\/download\/android\" rel=\"nofollow\"\u003eTwitter for Android\u003c\/a\u003e","truncated":false,"in_reply_to_status_id":null,"in_reply_to_status_id_str":null,"in_reply_to_user_id":null,"in_reply_to_user_id_str":null,"in_reply_to_screen_name":null,"user":{"id":438018513,"id_str":"438018513","name":"Morenawer","screen_name":"morenawer82","location":"en la parra,como buena riojana","url":null,"description":"Me seduce una cara y un cuerpo cuando hay una mente que los mueve. Yo hago el amor con las mentes. Hay que follarse a las mentes","translator_type":"none","protected":false,"verified":false,"followers_count":13619,"friends_count":642,"listed_count":145,"favourites_count":141554,"statuses_count":69371,"created_at":"Fri Dec 16 03:03:43 +0000 2011","utc_offset":null,"time_zone":null,"geo_enabled":true,"lang":"es","contributors_enabled":false,"is_translator":false,"profile_background_color":"7FDBB6","profile_background_image_url":"http:\/\/abs.twimg.com\/images\/themes\/theme18\/bg.gif","profile_background_image_url_https":"https:\/\/abs.twimg.com\/images\/themes\/theme18\/bg.gif","profile_background_tile":false,"profile_link_color":"F58EA8","profile_sidebar_border_color":"FFFFFF","profile_sidebar_fill_color":"F6FFD1","profile_text_color":"333333","profile_use_background_image":true,"profile_image_url":"http:\/\/pbs.twimg.com\/profile_images\/1049290656995315718\/l2eaQRxn_normal.jpg","profile_image_url_https":"https:\/\/pbs.twimg.com\/profile_images\/1049290656995315718\/l2eaQRxn_normal.jpg","profile_banner_url":"https:\/\/pbs.twimg.com\/profile_banners\/438018513\/1537396200","default_profile":false,"default_profile_image":false,"following":null,"follow_request_sent":null,"notifications":null},"geo":null,"coordinates":null,"place":null,"contributors":null,"is_quote_status":false,"quote_count":20,"reply_count":53,"retweet_count":886,"favorite_count":2600,"entities":{"hashtags":[],"urls":[],"user_mentions":[],"symbols":[]},"favorited":false,"retweeted":false,"filter_level":"low","lang":"es"},"is_quote_status":false,"quote_count":0,"reply_count":0,"retweet_count":0,"favorite_count":0,"entities":{"hashtags":[],"urls":[],"user_mentions":[{"screen_name":"morenawer82","name":"Morenawer","id":438018513,"id_str":"438018513","indices":[3,15]}],"symbols":[]},"favorited":false,"retweeted":false,"filter_level":"low","lang":"es","timestamp_ms":"1539207622657"}
```

```python
import oauth2 as oauth
import urllib.request as urllib

api_key = "khGGAbUjx3QOgwmJV8iBL8n5R"
api_secret = "r5ZwqVPaypAGjmXhzXLcCIUdBTkh7Fi7wOza4MzWPCV6o9NgHJ"
access_token_key = "1048089827470532609-rubHXdzWX0pzvIRqXKeKbU5E4Za4tW"
access_token_secret = "XEfwyba1DC8Q05ytVv7raYpQKRklPjJJOGJMEFtomzSgT"

_debug = 0

oauth_token    = oauth.Token(key=access_token_key, secret=access_token_secret)
oauth_consumer = oauth.Consumer(key=api_key, secret=api_secret)

signature_method_hmac_sha1 = oauth.SignatureMethod_HMAC_SHA1()

http_method = "GET"


http_handler  = urllib.HTTPHandler(debuglevel=_debug)
https_handler = urllib.HTTPSHandler(debuglevel=_debug)

'''
Construct, sign, and open a twitter request
using the hard-coded credentials above.
'''
def twitterreq(url, method, parameters):
  req = oauth.Request.from_consumer_and_token(oauth_consumer,
                                             token=oauth_token,
                                             http_method=http_method,
                                             http_url=url, 
                                             parameters=parameters)

  req.sign_request(signature_method_hmac_sha1, oauth_consumer, oauth_token)

  headers = req.to_header()

  if http_method == "POST":
    encoded_post_data = req.to_postdata()
  else:
    encoded_post_data = None
    url = req.to_url()

  opener = urllib.OpenerDirector()
  opener.add_handler(http_handler)
  opener.add_handler(https_handler)

  response = opener.open(url, encoded_post_data)

  return response

def fetchsamples():
  url = "https://stream.twitter.com/1.1/statuses/sample.json"
  parameters = []
  response = twitterreq(url, "GET", parameters)
  for line in response:
    print(line.decode('utf-8').strip())

if __name__ == '__main__':
  fetchsamples()

```

If we wish, we can also modify in the fetchsamples function above to use [twitter search API](https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets.html) to search for specific terms.For example, if we want to search for the term "Microsoft", we can pass "https://api.twitter.com/1.1/search/tweets.json?q=microsoft" as the url to the fetchsamples function above. 

## Derive the sentiment of each tweet.

Here we prepared an AFINN file which contains a list of pre-computed sentiment scores. 

AFINN is a list of English words rated for valence with an integerbetween minus five (negative) and plus five (positive). The words havebeen manually labeled by Finn Årup Nielsen in 2009-2011. The fileis tab-separated. There are two versions:AFINN-111: Newest version with 2477 words and phrases.AFINN-96: 1468 unique words and phrases on 1480 lines. Note that thereare 1480 lines, as some words are listed twice. The word list in notentirely in alphabetic ordering.An evaluation of the word list is available in:The list was used in:Lars Kai Hansen, Adam Arvidsson, Finn Årup Nielsen, Elanor Colleoni,Michael Etter, "Good Friends, Bad News - Affect and Virality inTwitter", The 2011 International Workshop on Social Computing,Network, and Services (SocialComNet 2011).This database of words is copyright protected and distributed under"Open Database License (ODbL) v1.0"http://www.opendatacommons.org/licenses/odbl/1.0/ or a similarcopyleft license.

We used AFINN-111 file for this step. Each line in this file contains a word or phrase followed by a sentiment score. Each word or phrase that is found in a tweet but not found in AFINN-111.txt was given a sentiment score of 0. Please note that the AFINN-111.txt file format is tab-delimited, meaning that the term and the score are separated by a tab character. A tab character can be identified a "\t".  

```python
import sys
import json
import re
def lines(fp):
    print (str(len(fp.readlines())))
def get_sents(f):
    """This function is used to generate the sentiments dictionary"""
    dic = {}
    with open(f, "r") as f:
        for line in f:
            s = line.strip().split("\t")
            dic[s[0]] = s[1]
    return dic
```
```python
def get_tweet_sents(file,dictionary):
    """This function is used to calculate the sentiments for each tweet. 
    Each line would output a tweet's sentiments."""
    with open(sys.argv[2], "r") as ins:
      #  i=1
        for line in ins:     
            data = json.loads(line)
            sum = 0
            if 'delete' in data.keys():
                sum  = 0
            else:
                text = data["text"].lower()

                
                words = text.split(" ")
                for word in words:
                    if word in dictionary.keys():
                        sum += int(dictionary[word]) 
            print(sum)
def main():
    sent_file = open(sys.argv[1])
    tweet_file = open(sys.argv[2])
    lines(sent_file)
    lines(tweet_file)
    #Start to calculate the sentiment score for each tweet.
    get_tweet_sents(sys.argv[2],get_sents(sys.argv[1]))
if __name__ == '__main__':
    main()
```

Here we built a get_sents function to generate a sentiment dictionary from AFINN_111 file. Then we converted the JSON file into a Python data structure, using json package. We mapped each term in each tweet to the sentiment dictionary and aggregated these scores together to compute a sentiment score for each tweet. The nth output line of the python script contains only a single number that represents the score of the nth tweet in the JSON file.

## Derive the sentiment of new terms.

Because some of the terms could not be found in our pre-defined sentiment dictionary, in this step, we would develop a sentiment metric based on the following research. [O'Connor, B., Balasubramanyan, R., Routedge, B., & Smith, N. From Tweets to Polls: Linking Text Sentiment to Public Opinion Time Series. (ICWSM), May 2010. (Links to an external site.)Links to an external site.](http://www.cs.cmu.edu/~nasmith/papers/oconnor+balasubramanyan+routledge+smith.icwsm10.pdf). Since we have already deduced the overall sentiment of a tweet in the section before, we can work backwards to deduce the sentiment of the non-sentiment carrying words that do not appear in AFINN-111 file. For example, if the word soccer always appears in proximity with positive words like great and fun, then we can deduce that the term soccer itself carries a positive sentiment.

```python
import sys
import json
import re
import string
def hw():
    print ('Hello, world!')

def lines(fp):
    print (str(len(fp.readlines())))
def get_sents(f):
    """This function is used to generate the sentiments dictionary"""
    dic = {}
    with open(f, "r") as f:
        for line in f:
            s = line.strip().split("\t")
            dic[s[0]] = s[1]
    return dic
def get_terms(f,dictionary):
    """This function gets the terms of file f based on the sentiment dictionary."""
    word_dic = {}
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    with open(f, "r") as ins:
      #  i=1
        for line in ins:     
            data = json.loads(line)
            sum = 0
            if 'delete' in data.keys():
                sum  = 0
            elif data["lang"]=="en":
                text = data["text"].lower()
                text = regex.sub("", text)
                words = text.split(" ")
                for word in words:
                    if word in dictionary.keys():
                        sum += int(dictionary[word]) 
                for word in words:
                    if word not in dictionary.keys():
                        if word not in word_dic.keys():
                            word_dic[word]=[1,1]
                        if word in word_dic.keys():
                            if sum > 0:
                                word_dic[word][0]+=1
                            elif sum < 0:
                                word_dic[word][1]+=1
    return word_dic
def print_out(word_dic):
    """This function print out the terms related to sentiments."""
    for word in word_dic.keys():
        word_dic[word]=word_dic[word][0]/word_dic[word][1]
        try:
            print(word, word_dic[word])
        except UnicodeEncodeError:
            print("UnicodeEncodeError: Character is either emoji or misc ",
                "symbol/pictograph and unsupported by Windows/OS/prompt. ",
                "Will not print word sentiment.")
def main():
    sent_file = open(sys.argv[1])
    tweet_file = open(sys.argv[2])
    hw()
    lines(sent_file)
    lines(tweet_file)
    print_out(get_terms(sys.argv[2],get_sents(sys.argv[1])))
if __name__ == '__main__':
    main()
```
Each line of the output contains a term and a sentiment score for that term.
----

## Compute term frequency.
Here we assume the tweet file contains data formatted the same way as the livestream data.The frequency of a term can be calculated as [# of occurrences of the term in all tweets]/[# of occurrences of all terms in all tweets].Each line of output contains a term, followed by the frequency of that term in the entire JSON file.

```python
import sys
import json
import re
import string
def get_fre(f):
    """This function gets the frequency of the tweets' terms."""
    fre_dic={}
    word_total = 0
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    with open(f, "r") as ins:
      #  i=1
        for line in ins:     
            data = json.loads(line)
            if 'text' in data.keys() and data["lang"]=="en":
                text = data["text"].lower()
                text = regex.sub("", text)
                words = text.split(" ")
                for word in words:
                    word_total += 1
                    if word not in fre_dic.keys():
                         fre_dic[word]=1
                    else:
                        fre_dic[word]+=1
        for word in fre_dic.keys():
            try:
                print(word,fre_dic[word])
            except UnicodeEncodeError:
                print("UnicodeEncodeError: Character is either emoji or misc ",
                  "symbol/pictograph and unsupported by Windows/OS/prompt. ",
                  "Will not print word frequency.")
def main():
    tweet_file = open(sys.argv[1])
    get_fre(sys.argv[1])
if __name__ == '__main__':
    main()
```
## Find out which state is happiest.

Here we assume the tweet file contains data formatted the same way as the livestream data. The way we are assigning a location to a tweet is to use the user field to determine the twitter user's home city and state. Since real data is dirty, we do not expect that every tweet would have a text field. We can even rank the U.S. states based on the aggregated sentiment scores, but just as an example, we will only output the happiest state using the following code.
```python
import sys
import json
import re
def get_sents(f):
    """This function is used to generate the sentiments dictionary"""
    dic = {}
    with open(f, "r") as f:
        for line in f:
            s = line.strip().split("\t")
            dic[s[0]] = s[1]
    return dic
def get_happy_state(f,dictionary):
    """This function is used to get the happiest state among these tweets based on the user location."""
    states = {'AL':'Alabama', "AK":'Alaska', "AZ":'Arizona', "AR":'Arkansas', "CA":'California', "CO":'Colorado', "CT":'Connecticut', "DC":'Delaware', 
    "DE":'Delaware', "FL":'Florida', "GA":'Georgia',"HI":'Hawaii', "ID":'Idaho', "IL":'Illinois', "IN": 'Indiana', "IA":'Iowa', "KS":'Kansas', 
    "KY": 'Kentucky',"LA":'Louisiana', "ME":'Maine', "MD":'Maryland',"MA":'Massachusetts', "MI":'Michigan', "MN":'Minnesota', "MS":'Mississippi',
     "MO":'Missouri', "MT":'Montana', "NE":'Nebraska', "NV":'Nevada', "NH":'New Hampshire', "NJ":'New Jersey',"NM":'New Mexico', "NY":'New York', 
     "NC":'North Carolina', "ND":'North Dakota', "OH":'Ohio', "OK":'Oklahoma', "OR":'Oregon', "PA":'Pennsylvania', "RI":'Rhode Island', "SC":'South Carolina',
     "SD":'South Dakota', "TN":'Tennessee', "TX": 'Texas', "UT":'Utah', "VT":'Vermont', "VA":'Virginia', "WA":'Washington', "WV":'West Virginia', "WI":'Wisconsin', 
     "WY":'Wyoming'}
    #Start to calculate the sentiment score for each tweet.
    happy_dic = {}
    with open(f, "r") as ins:
      #  i=1
        for line in ins:     
            data = json.loads(line)
            sum = 0
            if 'delete' in data.keys():
                sum  = 0
            elif data["user"]["location"] is not None and "USA" in data["user"]["location"]:
                text = data["text"].lower()
                location = data["user"]["location"].split(',')[0]
                words = text.split(" ")
                for word in words:
                    if word in dictionary.keys():
                        sum += int(dictionary[word]) 
                for abb,state in states.items():
                    if location == state and abb not in happy_dic.keys():
                        happy_dic[abb]=0   
                    elif location == state and abb in happy_dic.keys():
                        happy_dic[abb] += sum
    print(max(happy_dic.keys(), key=(lambda k: happy_dic[k])))
def main():
    get_happy_state(sys.argv[2],get_sents(sys.argv[1]))
if __name__ == '__main__':
    main()
```
The result shows that California is the happiest U.S. state in terms of our tweets file.
## Top ten hash tags.

Here we also assume that the JSON file contains data formatted the same way as the livestream data. Since the hashtags have already been extracted by twitter, our task is much simpler now. Each line of output contains a hashtag, followed by the frequency of that hashtag in the entire file.

```python
import sys
import json
import re
import string
import operator
from collections import Counter
def get_top_ten(f):
    hash_dic = {}
    with open(f, "r") as ins:
      #  i=1
        for line in ins:     
            data = json.loads(line)
            sum = 0
            if 'delete' in data.keys():
                sum  = 0
            elif data["lang"]=="en":
                if len(data["entities"]["hashtags"])>0:
                    for ele in data["entities"]["hashtags"]:
                        if ele["text"] not in hash_dic.keys():
                            hash_dic[str(ele["text"])]=0
                        else:
                            hash_dic[str(ele["text"])]+=1
        d=Counter(hash_dic)
        for k,v in d.most_common(10):
            print (k,v)
def main():
    get_top_ten(sys.argv[1])
if __name__ == '__main__':
    main()
```

----

#### Results:

|       Hash Tag       |       Frequency     |
|     :------------:   | :-----------------: |
| WorldMentalHealthDay |         39          |
|      BTSinLondon     |         27          |
|   TOPTRENCHPARTY     |         25          |
|           BTS        |         22          |
|        PCAs          |         20          |
|        AMAs          |         17          |
|      BTSxLondon      |         13          |
|       PSAT           |         12          |
|       AlexaPlay      |         10          |
|     ShareOpition     |         7           |


