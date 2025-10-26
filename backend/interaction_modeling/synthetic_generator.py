"""
Synthetic User Interaction Data Generator
Generates realistic user behavior patterns for training click prediction models.
"""

import numpy as np
from scipy.stats import beta
from typing import List, Dict, Any, Optional
import json
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class InteractionAction:
    """Represents a single user interaction action"""
    action: str
    timestamp: float
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class SyntheticInteractionGenerator:
    """Generates synthetic user interaction sequences using probabilistic models"""

    # Action space including the rare "click_ad" action
    ACTIONS = [
        'scroll_down', 'scroll_up', 'click', 'hover', 'wait',
        'focus_window', 'blur_window', 'close_tab', 'click_ad'
    ]

    # User behavior profiles with Beta distribution parameters
    PROFILES = {
        'rushed_reader': {
            'session_length': {'alpha': 1, 'beta': 3},      # Short sessions (30-60s)
            'interaction_freq': {'alpha': 2, 'beta': 1},   # Frequent but quick actions
            'pause_prob': {'alpha': 1, 'beta': 4},          # Rare pauses
            'click_prob': {'alpha': 1, 'beta': 3},          # Low clicking
            'scroll_pattern': 'rapid',
            'ad_click_multiplier': 0.1,                     # Very unlikely to click ads
        },
        'careful_reader': {
            'session_length': {'alpha': 3, 'beta': 1},      # Long sessions (2-3min)
            'interaction_freq': {'alpha': 1, 'beta': 2},   # Slow, deliberate actions
            'pause_prob': {'alpha': 2, 'beta': 1},          # Frequent thoughtful pauses
            'click_prob': {'alpha': 2, 'beta': 2},          # Moderate clicking
            'scroll_pattern': 'gradual',
            'ad_click_multiplier': 5.0,                     # Much more likely to click ads
        },
        'distracted_user': {
            'session_length': {'alpha': 2, 'beta': 2},      # Medium sessions (1-2min)
            'interaction_freq': {'alpha': 1, 'beta': 1},   # Variable frequency
            'pause_prob': {'alpha': 3, 'beta': 1},          # Very frequent pauses
            'click_prob': {'alpha': 1, 'beta': 2},          # Low clicking
            'blur_prob': {'alpha': 2, 'beta': 1},           # Frequent window switches
            'scroll_pattern': 'erratic',
            'ad_click_multiplier': 0.2,                     # Slightly more likely when focused
        },
        'focused_reader': {
            'session_length': {'alpha': 4, 'beta': 1},      # Very long sessions (3-4min)
            'interaction_freq': {'alpha': 1.5, 'beta': 1.5}, # Steady interaction
            'pause_prob': {'alpha': 1.5, 'beta': 2},        # Occasional focused pauses
            'click_prob': {'alpha': 2.5, 'beta': 1.5},      # High clicking
            'scroll_pattern': 'methodical',
            'ad_click_multiplier': 8.0,                     # Most likely to click ads
        },
        'click_happy': {
            'session_length': {'alpha': 2, 'beta': 2},      # Medium sessions
            'interaction_freq': {'alpha': 3, 'beta': 1},   # Very frequent actions
            'pause_prob': {'alpha': 1, 'beta': 3},          # Few pauses
            'click_prob': {'alpha': 4, 'beta': 1},          # Very high clicking
            'scroll_pattern': 'rapid',
            'ad_click_multiplier': 3.0,                     # Moderately likely to click ads
        },
        'mobile_user': {
            'session_length': {'alpha': 1.5, 'beta': 2},    # Shorter sessions on mobile
            'interaction_freq': {'alpha': 2, 'beta': 1},   # Frequent but different patterns
            'pause_prob': {'alpha': 1.5, 'beta': 2},        # Moderate pauses
            'click_prob': {'alpha': 2, 'beta': 1.5},        # Higher clicking on mobile
            'scroll_pattern': 'rapid',
            'ad_click_multiplier': 2.0,                     # Mobile users more likely to click ads
        }
    }

    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        self.base_ad_click_prob = 0.01  # 1% base probability (increase from 0.1%)

    def sample_beta(self, params: Dict[str, float]) -> float:
        """Sample from Beta distribution"""
        return beta.rvs(params['alpha'], params['beta'])

    def get_action_probabilities(self, profile: str, time_since_last_action: float = 0) -> np.ndarray:
        """Get action probabilities for a user profile, modified by current state"""
        profile_config = self.PROFILES[profile]

        # Base probabilities for each action
        base_probs = {
            'scroll_down': 0.25,
            'scroll_up': 0.15,
            'click': profile_config['click_prob']['alpha'] / (profile_config['click_prob']['alpha'] + profile_config['click_prob']['beta']),
            'hover': 0.20,
            'wait': profile_config['pause_prob']['alpha'] / (profile_config['pause_prob']['alpha'] + profile_config['pause_prob']['beta']),
            'focus_window': 0.05,
            'blur_window': profile_config.get('blur_prob', {'alpha': 1, 'beta': 3})['alpha'] / 10,  # Low probability
            'close_tab': 0.01,  # Very rare
            'click_ad': self.base_ad_click_prob * profile_config['ad_click_multiplier']
        }

        # Modify probabilities based on time since last action
        # Longer pauses increase likelihood of blur or close, decrease scroll
        if time_since_last_action > 10:  # More than 10 seconds
            base_probs['blur_window'] *= 2
            base_probs['close_tab'] *= 3
            base_probs['scroll_down'] *= 0.5
            base_probs['scroll_up'] *= 0.5
        elif time_since_last_action > 30:  # More than 30 seconds
            base_probs['blur_window'] *= 5
            base_probs['close_tab'] *= 10
            base_probs['scroll_down'] *= 0.2
            base_probs['scroll_up'] *= 0.2

        # Convert to numpy array in correct order
        probs = np.array([base_probs[action] for action in self.ACTIONS])

        # Normalize to ensure they sum to 1
        return probs / probs.sum()

    def generate_interaction_sequence(self, profile: str, max_length: int = 300) -> List[InteractionAction]:
        """Generate a sequence of user interactions for a given profile"""
        sequence = []
        current_time = 0.0

        # Sample session length (in seconds)
        session_length = self.sample_beta(self.PROFILES[profile]['session_length']) * 240  # Scale to 0-4 minutes
        session_length = max(10, min(session_length, 300))  # Clamp to reasonable range

        while current_time < session_length and len(sequence) < max_length:
            # Get action probabilities based on current state
            time_since_last = current_time - sequence[-1].timestamp if sequence else 0
            action_probs = self.get_action_probabilities(profile, time_since_last)

            # Sample next action
            action_idx = np.random.choice(len(self.ACTIONS), p=action_probs)
            action = self.ACTIONS[action_idx]

            # Sample time to next action based on profile
            if action == 'wait':
                # Wait actions are longer
                time_increment = self.sample_beta({'alpha': 2, 'beta': 1}) * 15  # 0-30 seconds
            else:
                # Regular actions
                interaction_freq = self.sample_beta(self.PROFILES[profile]['interaction_freq'])
                time_increment = np.random.exponential(1 / interaction_freq)  # Exponential distribution

            time_increment = max(0.1, min(time_increment, 30))  # Clamp to reasonable range
            current_time += time_increment

            # Create action details based on action type
            details = {}
            if action in ['scroll_down', 'scroll_up']:
                if self.PROFILES[profile]['scroll_pattern'] == 'rapid':
                    details['scroll_amount'] = np.random.uniform(0.1, 0.5)
                elif self.PROFILES[profile]['scroll_pattern'] == 'gradual':
                    details['scroll_amount'] = np.random.uniform(0.05, 0.2)
                else:  # erratic
                    details['scroll_amount'] = np.random.uniform(0.02, 0.8)
            elif action == 'click':
                details['element_type'] = np.random.choice(['link', 'button', 'image', 'text'])
            elif action == 'hover':
                details['hover_duration'] = np.random.exponential(2)  # 0-10 seconds

            sequence.append(InteractionAction(
                action=action,
                timestamp=current_time,
                details=details
            ))

        return sequence

    def generate_training_samples(self, n_samples: int = 10000) -> List[Dict[str, Any]]:
        """Generate training samples with sequences and labels"""
        training_samples = []

        for i in range(n_samples):
            if i % 1000 == 0:
                print(f"Generated {i}/{n_samples} samples...")

            # Sample user profile
            profile = np.random.choice(list(self.PROFILES.keys()))

            # Generate interaction sequence
            sequence = self.generate_interaction_sequence(profile)

            # Check if sequence contains ad click
            has_ad_click = any(action.action == 'click_ad' for action in sequence)

            training_samples.append({
                'profile': profile,
                'sequence': sequence,
                'has_ad_click': has_ad_click,
                'session_length': sequence[-1].timestamp if sequence else 0
            })

        return training_samples


if __name__ == "__main__":
    # Test the generator
    generator = SyntheticInteractionGenerator()

    # Generate a few sample sequences
    for profile in ['rushed_reader', 'careful_reader', 'distracted_user']:
        sequence = generator.generate_interaction_sequence(profile, max_length=20)
        print(f"\n{profile.upper()}:")
        for action in sequence[:10]:  # Show first 10 actions
            print(".1f")
        if len(sequence) > 10:
            print(f"  ... and {len(sequence) - 10} more actions")

    # Generate training samples
    print("\nGenerating training samples...")
    samples = generator.generate_training_samples(100)
    ad_click_rate = sum(1 for s in samples if s['has_ad_click']) / len(samples)
    print(".3f")