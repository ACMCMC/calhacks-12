#!/usr/bin/env python3
"""
Mock Data Generator for PrivAds Testing
Creates realistic test data for testing components without ML pipeline.
"""

import json
import random
from typing import Dict, List, Any

class MockDataGenerator:
    """Generates mock data for testing PrivAds components."""
    
    def __init__(self):
        self.ad_templates = [
            {
                "id": "mock_ad_001",
                "description": "Revolutionary smartphone with AI-powered camera and all-day battery life",
                "source_url": "https://example.com/ads/mock_ad_001.jpg",
                "features": {
                    "category": "technology",
                    "product_type": "smartphone",
                    "price_range": "premium",
                    "target_audience": "tech_enthusiasts"
                }
            },
            {
                "id": "mock_ad_002", 
                "description": "Sustainable fashion collection made from recycled materials",
                "source_url": "https://example.com/ads/mock_ad_002.jpg",
                "features": {
                    "category": "fashion",
                    "product_type": "clothing",
                    "price_range": "mid_range",
                    "target_audience": "eco_conscious"
                }
            },
            {
                "id": "mock_ad_003",
                "description": "Professional development course for data science and machine learning",
                "source_url": "https://example.com/ads/mock_ad_003.jpg",
                "features": {
                    "category": "education",
                    "product_type": "online_course",
                    "price_range": "affordable",
                    "target_audience": "professionals"
                }
            },
            {
                "id": "mock_ad_004",
                "description": "Premium coffee beans sourced from sustainable farms worldwide",
                "source_url": "https://example.com/ads/mock_ad_004.jpg",
                "features": {
                    "category": "food",
                    "product_type": "beverage",
                    "price_range": "premium",
                    "target_audience": "coffee_lovers"
                }
            },
            {
                "id": "mock_ad_005",
                "description": "Luxury electric vehicle with autonomous driving capabilities",
                "source_url": "https://example.com/ads/mock_ad_005.jpg",
                "features": {
                    "category": "automotive",
                    "product_type": "electric_car",
                    "price_range": "luxury",
                    "target_audience": "early_adopters"
                }
            }
        ]
        
        self.web_contexts = [
            {
                "url": "https://technews.com/smartphone-review",
                "title": "Latest Smartphone Reviews: AI Camera Technology",
                "page_type": "news",
                "keywords": ["smartphone", "tech", "review", "camera", "ai", "battery"],
                "summary_text": "Latest tech news and smartphone reviews | Topics: smartphone, tech, review, camera, ai",
                "domain": "technews.com"
            },
            {
                "url": "https://fashionblog.com/sustainable-fashion",
                "title": "Sustainable Fashion Trends 2024",
                "page_type": "blog",
                "keywords": ["fashion", "sustainable", "eco", "recycled", "trends"],
                "summary_text": "Sustainable fashion blog | Topics: fashion, sustainable, eco, recycled, trends",
                "domain": "fashionblog.com"
            },
            {
                "url": "https://education.com/courses",
                "title": "Online Learning: Data Science Courses",
                "page_type": "education",
                "keywords": ["education", "courses", "data science", "machine learning", "online"],
                "summary_text": "Online education platform | Topics: education, courses, data science, machine learning",
                "domain": "education.com"
            },
            {
                "url": "https://coffeeblog.com/premium-coffee",
                "title": "Premium Coffee Beans Guide",
                "page_type": "blog",
                "keywords": ["coffee", "premium", "beans", "sustainable", "farms"],
                "summary_text": "Coffee blog about premium beans | Topics: coffee, premium, beans, sustainable",
                "domain": "coffeeblog.com"
            },
            {
                "url": "https://autonews.com/electric-cars",
                "title": "Electric Vehicle News and Reviews",
                "page_type": "news",
                "keywords": ["electric", "car", "vehicle", "autonomous", "luxury"],
                "summary_text": "Automotive news about electric vehicles | Topics: electric, car, vehicle, autonomous",
                "domain": "autonews.com"
            }
        ]
    
    def generate_user_embedding(self, size: int = 128) -> List[float]:
        """Generate a random user embedding."""
        return [random.random() - 0.5 for _ in range(size)]
    
    def get_random_ad(self) -> Dict[str, Any]:
        """Get a random ad from templates."""
        return random.choice(self.ad_templates)
    
    def get_random_web_context(self) -> Dict[str, Any]:
        """Get a random web context from templates."""
        return random.choice(self.web_contexts)
    
    def create_mock_request(self, ad: Dict[str, Any] = None, web_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a mock request for testing."""
        if ad is None:
            ad = self.get_random_ad()
        if web_context is None:
            web_context = self.get_random_web_context()
        
        return {
            "ad_context": {
                "ad_id": ad["id"],
                "description": ad["description"],
                "ad_features": ad["features"],
                "content_signals": {
                    "page_type": web_context["page_type"],
                    "top_keywords": web_context["keywords"][:5],
                    "has_tech_signals": any(word in web_context["keywords"] for word in ["tech", "ai", "smartphone"]),
                    "has_fashion_signals": any(word in web_context["keywords"] for word in ["fashion", "style", "clothing"]),
                    "has_education_signals": any(word in web_context["keywords"] for word in ["education", "course", "learning"])
                }
            },
            "web_context": web_context,
            "customization_preferences": {
                "tone": "professional",
                "length": "medium",
                "include_cta": True
            }
        }
    
    def create_test_scenarios(self) -> List[Dict[str, Any]]:
        """Create various test scenarios for comprehensive testing."""
        scenarios = []
        
        # Scenario 1: Tech news + smartphone ad
        scenarios.append({
            "name": "Tech News + Smartphone Ad",
            "ad": self.ad_templates[0],  # smartphone
            "web_context": self.web_contexts[0],  # tech news
            "expected_match": "high"
        })
        
        # Scenario 2: Fashion blog + fashion ad
        scenarios.append({
            "name": "Fashion Blog + Fashion Ad", 
            "ad": self.ad_templates[1],  # fashion
            "web_context": self.web_contexts[1],  # fashion blog
            "expected_match": "high"
        })
        
        # Scenario 3: Education site + course ad
        scenarios.append({
            "name": "Education Site + Course Ad",
            "ad": self.ad_templates[2],  # course
            "web_context": self.web_contexts[2],  # education
            "expected_match": "high"
        })
        
        # Scenario 4: Mismatch scenario
        scenarios.append({
            "name": "Tech News + Fashion Ad (Mismatch)",
            "ad": self.ad_templates[1],  # fashion
            "web_context": self.web_contexts[0],  # tech news
            "expected_match": "low"
        })
        
        # Scenario 5: Coffee blog + coffee ad
        scenarios.append({
            "name": "Coffee Blog + Coffee Ad",
            "ad": self.ad_templates[3],  # coffee
            "web_context": self.web_contexts[3],  # coffee blog
            "expected_match": "high"
        })
        
        return scenarios
    
    def save_mock_data(self, filename: str = "mock_data.json"):
        """Save mock data to a JSON file."""
        data = {
            "ads": self.ad_templates,
            "web_contexts": self.web_contexts,
            "test_scenarios": self.create_test_scenarios(),
            "sample_user_embeddings": {
                "tech_enthusiast": self.generate_user_embedding(),
                "fashion_lover": self.generate_user_embedding(),
                "student": self.generate_user_embedding(),
                "coffee_lover": self.generate_user_embedding()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Mock data saved to {filename}")
        return data

def main():
    """Generate and save mock data."""
    print("🎭 PrivAds Mock Data Generator")
    print("=" * 40)
    
    generator = MockDataGenerator()
    
    # Generate mock data
    data = generator.save_mock_data()
    
    print(f"\n📊 Generated:")
    print(f"   {len(data['ads'])} ad templates")
    print(f"   {len(data['web_contexts'])} web contexts")
    print(f"   {len(data['test_scenarios'])} test scenarios")
    print(f"   {len(data['sample_user_embeddings'])} user embeddings")
    
    print(f"\n🧪 Test Scenarios:")
    for i, scenario in enumerate(data['test_scenarios'], 1):
        print(f"   {i}. {scenario['name']} (Expected: {scenario['expected_match']} match)")
    
    print(f"\n🚀 Usage:")
    print(f"   from mock_data_generator import MockDataGenerator")
    print(f"   generator = MockDataGenerator()")
    print(f"   request = generator.create_mock_request()")
    print(f"   # Use request for testing Gemini customization")

if __name__ == "__main__":
    main()

