import os
import glob
from bs4 import BeautifulSoup
import pandas as pd

def parse_tweet(post, source_file):
    p_tag = post.find("p")     # extrait des balises p
    text = p_tag.get_text().strip() if p_tag else None
    
    tweet_id = post.get("{http://www.w3.org/XML/1998/namespace}id")
    who = post.get("who")
    when = post.get("when")
    lang = post.get("{http://www.w3.org/XML/1998/namespace}lang")
    
    trailer = post.find("trailer")
    medium = None
    retweet_count = None
    is_retweet = None
    
    if trailer:
        medium_tag = trailer.find("f", {"name": "medium"})
        if medium_tag and medium_tag.find("string"):
            medium = medium_tag.find("string").get_text()
            
        retweet_tag = trailer.find("f", {"name": "retweetcount"})
        if retweet_tag and retweet_tag.find("numeric"):
            retweet_count = retweet_tag.find("numeric").get("value")
            
        is_retweet_tag = trailer.find("f", {"name": "isRetweet"})
        if is_retweet_tag and is_retweet_tag.find("binary"):
            is_retweet = is_retweet_tag.find("binary").get("value")
    
    return {
        "source_file": source_file,
        "tweet_id": tweet_id,
        "user_id": who,
        "creation_date": when,
        "text": text,
        "language": lang,
        "medium": medium,
        "is_retweet": is_retweet,
        "retweet_count": retweet_count,
    }

def main():
    all_data = []

    for xml_file in glob.glob("../../data/raw/polititweet/*.xml"):
        if "olac-cmr" in xml_file:  # passe fichier de métadonnées
            continue
            
        with open(xml_file, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "xml")
            tweets = soup.find_all("post")

            for tweet in tweets:
                row = parse_tweet(tweet, os.path.basename(xml_file))
                all_data.append(row)

    df = pd.DataFrame(all_data)
    df.to_csv("polititweets_full.csv", index=False, encoding="utf-8")
    print("extraction ok")
    print(f"{len(df)} tweets exportés.")

if __name__ == "__main__":
    main()
