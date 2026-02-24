import os
import tweepy
from datetime import datetime, time, timedelta
import pytz
from typing import Any, List, Dict, Tuple, cast
from dotenv import load_dotenv

load_dotenv()

class XScraper:
    def __init__(self):
        self.bearer_token = os.getenv("X_BEARER_TOKEN")
        self.client = tweepy.Client(bearer_token=self.bearer_token, wait_on_rate_limit=True)
        self.timezone = pytz.timezone("US/Eastern")
        self.target_user_id = os.getenv("X_USER_ID")

    def get_market_times_utc(self) -> Tuple[datetime, datetime]:
        """Returns today's 9:30 AM and 4:00 PM ET as UTC objects."""
        now_et = datetime.now(self.timezone)
        today = now_et.date()
        
        # 9:30 AM ET
        start_et = self.timezone.localize(datetime.combine(today, time(9, 30)))
        # 4:00 PM ET
        end_et = self.timezone.localize(datetime.combine(today, time(16, 0)))

        # X API Requirement: end_time must be at least 10-30 seconds in the past.
        # If running exactly at 4:00 PM, we subtract a small buffer.
        now_utc = datetime.now(pytz.utc)
        start_utc = start_et.astimezone(pytz.utc)
        end_utc = end_et.astimezone(pytz.utc)

        # Safety: Ensure end_time is not in the future relative to the API call
        if end_utc > (now_utc - timedelta(seconds=30)):
            end_utc = now_utc - timedelta(seconds=30)

        return start_utc, end_utc

    def fetch_recent_activity(self) -> List[Dict[str, Any]]:
        """Fetches tweets strictly between 9:30 AM and 4:00 PM ET today."""
        if not self.target_user_id:
            raise ValueError("X_USER_ID not found in .env file.")

        start_time, end_time = self.get_market_times_utc()

        # get_users_tweets supports both start_time and end_time
        response = self.client.get_users_tweets(
            id=self.target_user_id,
            start_time=start_time,
            end_time=end_time,
            tweet_fields=["created_at", "text", "referenced_tweets", "conversation_id", "attachments"],
            max_results=50 
        )

        tweets_data: Any = response.data
        if not tweets_data:
            print(f"No tweets found between {start_time} and {end_time} UTC.")
            return []

        processed_tweets = []
        for tweet in tweets_data:
            # Skip image/media tweets
            if tweet.attachments and 'media_keys' in tweet.attachments:
                continue

            ref_info = {"type": "original", "parent_id": None}
            if tweet.referenced_tweets:
                ref_info["type"] = tweet.referenced_tweets[0].type
                ref_info["parent_id"] = tweet.referenced_tweets[0].id

            processed_tweets.append({
                "tweet_id": tweet.id,
                "conversation_id": tweet.conversation_id,
                "text": tweet.text,
                "created_at": tweet.created_at.isoformat(),
                "ref_type": ref_info["type"],
                "parent_id": ref_info["parent_id"]
            })

        # Sort chronologically for the LLM
        processed_tweets.sort(key=lambda x: x['created_at'])
        return processed_tweets