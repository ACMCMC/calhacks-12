"""
Elasticsearch integration for PrivAds
Handles natural language search over ad metadata using Elastic Agent Builder.
"""

import requests
import os
from typing import List, Dict, Any
from elasticsearch import Elasticsearch

class ElasticSearchClient:
    """Client for Elasticsearch operations."""

    def __init__(self, es_url: str = None, agent_builder_url: str = None):
        self.es_url = es_url or os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        self.agent_builder_url = agent_builder_url or os.getenv("AGENT_BUILDER_URL", "http://localhost:5601")

        # Initialize Elasticsearch client
        self.es_client = Elasticsearch(self.es_url)

        # Agent Builder endpoint (this might need adjustment based on actual API)
        self.agent_endpoint = f"{self.agent_builder_url}/api/agent_builder/converse"

    def search_with_agent_builder(self, query: str) -> Dict[str, Any]:
        """
        Use Elastic Agent Builder to convert natural language query to Elasticsearch query.
        """
        try:
            payload = {
                "query": query,
                "agent_id": "privads_ad_search_agent"  # Would be configured in Kibana
            }

            response = requests.post(
                self.agent_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Agent Builder error: {response.status_code}")
                return self._fallback_search(query)

        except Exception as e:
            print(f"Agent Builder request failed: {e}")
            return self._fallback_search(query)

    def _fallback_search(self, query: str) -> Dict[str, Any]:
        """
        Fallback search using simple Elasticsearch query.
        """
        try:
            # Simple multi-match query across metadata fields
            es_query = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["tags", "type", "description", "sentiment"]
                    }
                },
                "size": 20
            }

            response = self.es_client.search(
                index="ads_metadata",
                body=es_query
            )

            return {
                "results": [
                    {
                        "ad_id": hit["_source"]["ad_id"],
                        "metadata": hit["_source"],
                        "score": hit["_score"]
                    }
                    for hit in response["hits"]["hits"]
                ]
            }

        except Exception as e:
            print(f"Elasticsearch fallback search failed: {e}")
            return {"results": []}

    def index_ad_metadata(self, ad_id: str, metadata: Dict[str, Any]):
        """
        Index ad metadata in Elasticsearch.
        """
        try:
            doc = {
                "ad_id": ad_id,
                **metadata
            }

            self.es_client.index(
                index="ads_metadata",
                id=ad_id,
                document=doc
            )

        except Exception as e:
            print(f"Failed to index ad {ad_id}: {e}")

    def create_index_mapping(self):
        """
        Create Elasticsearch index with proper mapping for ad metadata.
        """
        mapping = {
            "mappings": {
                "properties": {
                    "ad_id": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "type": {"type": "keyword"},
                    "alignment": {"type": "keyword"},
                    "sentiment": {"type": "float"},
                    "has_cta": {"type": "boolean"},
                    "dominant_color": {"type": "keyword"},
                    "avg_motion": {"type": "float"},
                    "description": {"type": "text"},
                    "keywords": {"type": "keyword"}
                }
            }
        }

        try:
            self.es_client.indices.create(
                index="ads_metadata",
                body=mapping,
                ignore=400  # Ignore if index already exists
            )
            print("✓ Created Elasticsearch index 'ads_metadata'")
        except Exception as e:
            print(f"Failed to create index: {e}")

def search_ads_elastic(query: str) -> List[Dict[str, Any]]:
    """
    Main function to search ads using natural language.
    Used by the /search_ads API endpoint.
    """
    client = ElasticSearchClient()
    response = client.search_with_agent_builder(query)

    # Extract and format results
    results = []
    if "results" in response:
        for result in response["results"]:
            results.append({
                "ad_id": result.get("ad_id", ""),
                "thumbnail_url": result.get("thumbnail_url", ""),
                "metadata": result.get("metadata", {})
            })

    return results

# For testing/development
if __name__ == "__main__":
    # Test search
    results = search_ads_elastic("Show me car ads that are videos")
    print(f"Found {len(results)} results")

    for result in results[:3]:
        print(f"- {result['ad_id']}: {result['metadata']}")