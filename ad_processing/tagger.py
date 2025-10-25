"""
Contextual Tagger for Project Aura
Generates contextual tags for ad creatives based on content and features.
"""

import re
from typing import Dict, Any, List
from pathlib import Path

class AdTagger:
    """Generates contextual tags for ad classification."""

    def __init__(self):
        # Define tag categories and keywords
        self.tag_mappings = {
            'political': [
                'election', 'vote', 'candidate', 'political', 'government', 'policy',
                'democrat', 'republican', 'conservative', 'liberal', 'progressive'
            ],
            'patriotic': [
                'america', 'american', 'flag', 'freedom', 'patriot', 'national',
                'usa', 'united states', 'stars and stripes'
            ],
            'chess': [
                'chess', 'checkmate', 'pawn', 'rook', 'bishop', 'knight', 'queen', 'king',
                'strategy', 'tournament', 'grandmaster'
            ],
            'logic': [
                'logic', 'reasoning', 'puzzle', 'brain', 'intelligence', 'smart',
                'thinking', 'cognitive', 'analytical'
            ],
            'social_good': [
                'charity', 'donate', 'help', 'community', 'volunteer', 'nonprofit',
                'giving', 'philanthropy', 'social impact'
            ],
            'luxury': [
                'luxury', 'premium', 'expensive', 'high-end', 'exclusive', 'elite',
                'designer', 'boutique', 'luxurious'
            ],
            'eco_friendly': [
                'eco', 'green', 'sustainable', 'environment', 'organic', 'recycle',
                'renewable', 'carbon neutral', 'earth friendly'
            ],
            'technology': [
                'tech', 'innovation', 'digital', 'software', 'app', 'ai', 'machine learning',
                'automation', 'smart', 'connected'
            ],
            'health': [
                'health', 'fitness', 'wellness', 'medical', 'nutrition', 'exercise',
                'healthy', 'diet', 'medicine'
            ],
            'education': [
                'learn', 'education', 'school', 'university', 'student', 'teacher',
                'knowledge', 'study', 'academic'
            ]
        }

        # Political alignment mappings
        self.alignment_keywords = {
            'conservative': ['conservative', 'republican', 'right', 'traditional', 'patriot'],
            'liberal': ['liberal', 'democrat', 'left', 'progressive', 'equality'],
            'centrist': ['moderate', 'centrist', 'balanced', 'middle']
        }

    def generate_tags(self, features: Dict[str, Any], text_content: str = "") -> Dict[str, Any]:
        """
        Generate contextual tags based on extracted features and text content.

        Args:
            features: Dictionary of extracted features
            text_content: Raw text content from ad

        Returns:
            Dictionary with tags, type, alignment, etc.
        """
        tags = []
        ad_type = "general"
        alignment = None
        theme = None

        # Combine text from various sources
        full_text = text_content.lower()
        if 'ocr_text' in features:
            full_text += " " + features['ocr_text'].lower()
        if 'keywords' in features:
            full_text += " " + " ".join(features['keywords'])

        # Check for specific categories
        for category, keywords in self.tag_mappings.items():
            if any(keyword in full_text for keyword in keywords):
                tags.append(category)
                if category in ['political', 'patriotic']:
                    ad_type = "political"
                elif category in ['chess', 'logic']:
                    ad_type = "educational"
                    theme = category
                elif category in ['social_good', 'eco_friendly']:
                    ad_type = "social_impact"
                elif category in ['luxury', 'technology', 'health']:
                    ad_type = "commercial"
                    theme = category

        # Determine political alignment if political
        if ad_type == "political":
            for align, keywords in self.alignment_keywords.items():
                if any(keyword in full_text for keyword in keywords):
                    alignment = align
                    break

        # Add sentiment-based tags
        if 'sentiment' in features:
            sentiment = features['sentiment']
            if sentiment == 'positive':
                tags.append('positive')
            elif sentiment == 'negative':
                tags.append('negative')

        # Add visual tags based on features
        if features.get('has_cta', False):
            tags.append('has_cta')

        if features.get('avg_motion', 0) > 0.1:
            tags.append('dynamic')
        else:
            tags.append('static')

        # Color-based tags
        dominant_color = features.get('dominant_color', '')
        if 'red' in dominant_color or 'blue' in dominant_color:
            tags.append('patriotic_colors')

        return {
            'tags': list(set(tags)),  # Remove duplicates
            'type': ad_type,
            'alignment': alignment,
            'theme': theme,
            'specific': self._get_specific_tags(full_text)
        }

    def _get_specific_tags(self, text: str) -> str:
        """Extract more specific contextual information."""
        # Look for specific entities or contexts
        if any(word in text for word in ['chess', 'grandmaster', 'tournament']):
            return 'competitive_chess'
        elif any(word in text for word in ['election', '2024', 'vote']):
            return 'election_2024'
        elif any(word in text for word in ['luxury', 'premium', 'expensive']):
            return 'high_end_lifestyle'
        elif any(word in text for word in ['eco', 'sustainable', 'green']):
            return 'environmental'
        else:
            return 'general'

    def get_relevant_contexts(self, tags: List[str]) -> List[str]:
        """
        Get relevant context keywords for Chroma filtering.
        """
        context_keywords = []

        # Map tags to context keywords
        context_mapping = {
            'political': ['politics', 'government', 'election'],
            'patriotic': ['patriotism', 'national', 'american'],
            'chess': ['strategy', 'logic', 'competition'],
            'logic': ['reasoning', 'puzzles', 'intelligence'],
            'social_good': ['charity', 'community', 'helping'],
            'luxury': ['premium', 'expensive', 'elite'],
            'eco_friendly': ['environment', 'sustainable', 'green'],
            'technology': ['innovation', 'digital', 'tech'],
            'health': ['wellness', 'fitness', 'medical'],
            'education': ['learning', 'knowledge', 'academic']
        }

        for tag in tags:
            if tag in context_mapping:
                context_keywords.extend(context_mapping[tag])

        return list(set(context_keywords))

# Convenience function
def tag_ad(features: Dict[str, Any], text_content: str = "") -> Dict[str, Any]:
    """Convenience function to tag an ad."""
    tagger = AdTagger()
    return tagger.generate_tags(features, text_content)

if __name__ == "__main__":
    # Test tagging
    tagger = AdTagger()

    # Test political ad
    features = {
        'sentiment': 'positive',
        'has_cta': True,
        'keywords': ['vote', 'election', 'america']
    }
    tags = tagger.generate_tags(features, "Vote for America! Make America Great Again!")
    print("Political ad tags:", tags)

    # Test luxury ad
    features2 = {
        'sentiment': 'positive',
        'has_cta': True,
        'keywords': ['luxury', 'premium', 'expensive']
    }
    tags2 = tagger.generate_tags(features2, "Experience luxury like never before!")
    print("Luxury ad tags:", tags2)