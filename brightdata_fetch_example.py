import requests
import os
from dotenv import load_dotenv

def get_amazon_competitor_prices(keywords, limit=3):
    """
    Query the Bright Data Amazon products dataset for competitor prices using keywords.
    Returns the JSON response or None on error.
    """
    load_dotenv()
    api_key = os.getenv('BRIGHTDATA_API_KEY')
    dataset_id = "gd_l7q7dkf244hwjntr0"  # Amazon products dataset
    url = "https://api.brightdata.com/datasets/marketplace/filter"
    payload = {
        "dataset_id": dataset_id,
        "limit": limit,
        "search_term": " ".join(keywords)
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200 and response.headers.get('Content-Type', '').startswith('application/json'):
        try:
            return response.json()
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return None
    else:
        print("Bright Data API error:", response.status_code, response.text)
        return None

# Example usage:
if __name__ == "__main__":
    ad_keywords = ["laptop", "dell", "xps"]
    results = get_amazon_competitor_prices(ad_keywords, limit=3)
    print("Results:", results)
