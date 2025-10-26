# from brightdata import client as bdclient
import os
from dotenv import load_dotenv

# Reka integration
from openai import OpenAI


def get_reka_research(keywords):
    """
    Use Reka API to perform deep research on the given keywords.
    Returns the assistant's response as a string, or None on error.
    """
    load_dotenv()
    api_key = os.getenv("REKA_API_KEY")
    if not api_key:
        print("No REKA_API_KEY found in environment.")
        return None
    client = OpenAI(base_url="https://api.reka.ai/v1", api_key=api_key)
    query = f"Perform a deep research on: {' '.join(keywords)}. Summarize key findings, trends, and cite sources if possible."
    try:
        completion = client.chat.completions.create(
            model="reka-flash-research", messages=[{"role": "user", "content": query}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Reka API error: {e}")
        return None


def get_google_results(keywords):
    """
    Query the Bright Data Google search dataset for results using keywords.
    Returns the parsed content or None on error.
    """
    load_dotenv()
    api_key = os.getenv("BRIGHTDATA_API_KEY")
    if not api_key:
        print("No BRIGHTDATA_API_KEY found in environment.")
        return None
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

    print("\nReka Deep Research Results:")
    reka_results = get_reka_research(ad_keywords)
    print(reka_results)

    print("Bright Data Google Results:")
    bd_results = get_google_results(ad_keywords)
    print(bd_results)
