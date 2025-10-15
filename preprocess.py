import re
import json
from ftfy import fix_text
from unidecode import unidecode
import inflection

# Cleans text in tweets
def clean_tweet(text):
    # Fixing incorrect encoding issues
    text = fix_text(text)

    # Converting text Ascii
    text = unidecode(text)
    
    # May want to store hashtags and mentions after looking through them. 
    # Majority are just #goldenglobes, but there are some of each that relate to celebrity names
    hashtags = re.findall(r"#\w+", text)
    mentions = re.findall(r"@\w+", text)

    # Removing hashtags and mentions
    text = re.sub(r"[@#]\w+", "", text)
    # Removing URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Removing extra spaces
    text = re.sub(r"\s+", " ", text).strip()


    # Current normalizing method is generating a lot of bad data, it can't separate incorrectly capitalized words/names
    # Normalizing hashtags
    cleaned_tags = []
    for tag in hashtags: 
        tag = tag.lstrip("#") 
        tag = inflection.humanize(inflection.underscore(tag))
        cleaned_tags.append(tag)

    # Normalizing mentions
    cleaned_mentions = []
    for mention in mentions: 
        mention = mention.lstrip("@") 
        mention = inflection.humanize(inflection.underscore(mention))
        cleaned_mentions.append(mention)

    return text, cleaned_tags, cleaned_mentions

# Cleans tweets in input_file and stores result in input_file+_clean.json 
def store_cleaned_tweets(input_file):
    cleaned_tweets = []

    with open(input_file, "r") as f:
        tweets = json.load(f)

    for tweet in tweets:
        text = tweet.get("text", "")

        # Clean the text
        # cleaned_text = clean_tweet(text) #If we choose to not use hashtags and mentions
        cleaned_text, hashtags, mentions = clean_tweet(text) #If we choose to keep hastags and mentions

        # Construct cleaned tweet object
        new_tweet = {
            "id": tweet.get("id"),
            "user": tweet.get("user", {}), # May remove because the tweet should be identifiable by id (extraneous)
            "timestamp_ms": tweet.get("timestamp_ms"), # Might remove if we decide timestamps are not helpful
            "text": cleaned_text,
            "hashtags": hashtags,
            "mentions": mentions
        }
        cleaned_tweets.append(new_tweet)

    # Making output filename
    output_file = input_file.replace(".json", "_clean.json")

    # Saving output
    with open(output_file, "w") as f:
        json.dump(cleaned_tweets, f, indent=2)


store_cleaned_tweets("gg2013.json")