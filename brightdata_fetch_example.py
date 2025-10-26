from brightdata import bdclient
import os
from dotenv import load_dotenv

def get_google_results(keywords):
    """
    Query the Bright Data Google search dataset for results using keywords.
    Returns the parsed content or None on error.
    """
    load_dotenv()
    api_key = os.getenv('BRIGHTDATA_API_KEY')
    client = bdclient(api_token=api_key)
    query = " ".join(keywords)
    try:
        results = client.search(query=query, search_engine="google", parse=True)
        return client.parse_content(results)
    except Exception as e:
        print(f"Bright Data SDK error: {e}")
        return None

# Example usage:
if __name__ == "__main__":
    ad_keywords = ["laptop", "dell", "xps"]
    results = get_google_results(ad_keywords)
    print("Results:", results)
