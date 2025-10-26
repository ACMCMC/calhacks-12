"""
Sliding Window Feature Extractor
Converts interaction sequences into ML-ready feature vectors using 2-minute sliding windows.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class WindowFeatures:
    """Features extracted from a sliding window of interactions"""
    # Temporal features
    time_since_last_action: float
    avg_time_between_actions: float
    session_duration: float

    # Action counts (normalized by window size)
    scroll_down_count: float
    scroll_up_count: float
    click_count: float
    hover_count: float
    blur_count: float
    focus_count: float
    wait_count: float
    close_tab_count: float

    # Derived features
    scroll_depth_max: float
    interaction_density: float
    attention_score: float
    scroll_velocity_avg: float
    click_to_hover_ratio: float
    blur_frequency: float

    # Pattern features
    action_entropy: float
    burstiness_score: float
    engagement_rhythm: float

    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for ML"""
        return np.array([
            self.time_since_last_action,
            self.avg_time_between_actions,
            self.session_duration,
            self.scroll_down_count,
            self.scroll_up_count,
            self.click_count,
            self.hover_count,
            self.blur_count,
            self.focus_count,
            self.wait_count,
            self.close_tab_count,
            self.scroll_depth_max,
            self.interaction_density,
            self.attention_score,
            self.scroll_velocity_avg,
            self.click_to_hover_ratio,
            self.blur_frequency,
            self.action_entropy,
            self.burstiness_score,
            self.engagement_rhythm
        ])


class SlidingWindowExtractor:
    """Extracts features from sliding windows of user interactions"""

    WINDOW_SIZE = 120  # 2 minutes
    WINDOW_STEP = 10   # 10 seconds between extractions
    MIN_ACTIONS = 5    # Minimum actions needed for valid window

    def __init__(self):
        self.feature_names = [
            'time_since_last_action', 'avg_time_between_actions', 'session_duration',
            'scroll_down_count', 'scroll_up_count', 'click_count', 'hover_count',
            'blur_count', 'focus_count', 'wait_count', 'close_tab_count',
            'scroll_depth_max', 'interaction_density', 'attention_score',
            'scroll_velocity_avg', 'click_to_hover_ratio', 'blur_frequency',
            'action_entropy', 'burstiness_score', 'engagement_rhythm'
        ]

    def extract_window_features(self, action_history: List[Any], window_start: float, window_end: float) -> Optional[WindowFeatures]:
        """Extract features from a specific time window of interactions"""
        # Filter actions in this window
        window_actions = [a for a in action_history if window_start <= a.timestamp <= window_end]

        if len(window_actions) < self.MIN_ACTIONS:
            return None  # Not enough data for reliable features

        # Calculate temporal features
        timestamps = [a.timestamp for a in window_actions]
        time_since_last_action = window_end - timestamps[-1]

        # Average time between actions
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            avg_time_between_actions = np.mean(intervals)
        else:
            avg_time_between_actions = window_end - window_start

        session_duration = window_end - action_history[0].timestamp if action_history else window_end

        # Count actions (normalized by window size)
        action_counts = defaultdict(int)
        for action in window_actions:
            action_counts[action.action] += 1

        window_duration = window_end - window_start
        scroll_down_count = action_counts['scroll_down'] / window_duration
        scroll_up_count = action_counts['scroll_up'] / window_duration
        click_count = action_counts['click'] / window_duration
        hover_count = action_counts['hover'] / window_duration
        blur_count = action_counts['blur_window'] / window_duration
        focus_count = action_counts['focus_window'] / window_duration
        wait_count = action_counts['wait'] / window_duration
        close_tab_count = action_counts['close_tab'] / window_duration

        # Calculate scroll depth (cumulative scroll amount)
        scroll_depth = 0
        scroll_amounts = []
        for action in window_actions:
            if action.action in ['scroll_down', 'scroll_up']:
                amount = action.details.get('scroll_amount', 0.1)
                if action.action == 'scroll_down':
                    scroll_depth += amount
                else:
                    scroll_depth -= amount
                scroll_amounts.append(amount)

        scroll_depth_max = max(0, scroll_depth)  # Only positive depth
        scroll_velocity_avg = np.mean(scroll_amounts) if scroll_amounts else 0

        # Interaction density
        interaction_density = len(window_actions) / window_duration

        # Attention score (based on focus/blur balance and interaction patterns)
        focus_balance = focus_count - blur_count
        attention_score = max(0, min(1, 0.5 + focus_balance * 10 + interaction_density * 0.1))

        # Click to hover ratio (engagement quality indicator)
        click_to_hover_ratio = click_count / (hover_count + 0.001)  # Avoid division by zero

        # Blur frequency (distraction indicator)
        blur_frequency = blur_count

        # Action entropy (diversity of actions)
        total_actions = sum(action_counts.values())
        if total_actions > 0:
            action_probs = [count / total_actions for count in action_counts.values()]
            action_entropy = -sum(p * np.log(p + 1e-10) for p in action_probs)
        else:
            action_entropy = 0

        # Burstiness score (how clustered actions are)
        if len(timestamps) > 2:
            intervals = np.diff(timestamps)
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            burstiness_score = std_interval / (mean_interval + 1e-10)
        else:
            burstiness_score = 0

        # Engagement rhythm (regularity of interactions)
        if len(timestamps) > 3:
            # Calculate autocorrelation-like measure
            intervals = np.diff(timestamps)
            if len(intervals) > 1:
                rhythm_score = np.corrcoef(intervals[:-1], intervals[1:])[0, 1]
                engagement_rhythm = max(0, rhythm_score)  # Only positive correlation
            else:
                engagement_rhythm = 0
        else:
            engagement_rhythm = 0

        return WindowFeatures(
            time_since_last_action=time_since_last_action,
            avg_time_between_actions=avg_time_between_actions,
            session_duration=session_duration,
            scroll_down_count=scroll_down_count,
            scroll_up_count=scroll_up_count,
            click_count=click_count,
            hover_count=hover_count,
            blur_count=blur_count,
            focus_count=focus_count,
            wait_count=wait_count,
            close_tab_count=close_tab_count,
            scroll_depth_max=scroll_depth_max,
            interaction_density=interaction_density,
            attention_score=attention_score,
            scroll_velocity_avg=scroll_velocity_avg,
            click_to_hover_ratio=click_to_hover_ratio,
            blur_frequency=blur_frequency,
            action_entropy=action_entropy,
            burstiness_score=burstiness_score,
            engagement_rhythm=engagement_rhythm
        )

    def extract_all_windows(self, action_history: List[Any]) -> List[Tuple[WindowFeatures, bool]]:
        """Extract features from all sliding windows in a sequence"""
        if not action_history:
            return []

        windows = []
        max_time = action_history[-1].timestamp

        # Slide window across the entire sequence
        for window_start in np.arange(0, max_time - self.WINDOW_SIZE + self.WINDOW_STEP, self.WINDOW_STEP):
            window_end = window_start + self.WINDOW_SIZE

            # Only process windows that have data
            if window_end > max_time:
                break

            features = self.extract_window_features(action_history, window_start, window_end)
            if features is None:
                continue

            # Check if ad was clicked in this window
            clicked_ad = any(a.action == 'click_ad' and window_start <= a.timestamp <= window_end
                           for a in action_history)

            windows.append((features, clicked_ad))

        return windows

    def save_feature_names(self, filepath: str):
        """Save feature names for later use"""
        with open(filepath, 'w') as f:
            json.dump(self.feature_names, f)

    @staticmethod
    def load_feature_names(filepath: str) -> List[str]:
        """Load feature names"""
        with open(filepath, 'r') as f:
            return json.load(f)


if __name__ == "__main__":
    from synthetic_generator import SyntheticInteractionGenerator

    # Test the extractor
    generator = SyntheticInteractionGenerator()
    extractor = SlidingWindowExtractor()

    # Generate a test sequence
    sequence = generator.generate_interaction_sequence('careful_reader', max_length=50)
    print(f"Generated sequence with {len(sequence)} actions over {sequence[-1].timestamp:.1f} seconds")

    # Extract windows
    windows = extractor.extract_all_windows(sequence)
    print(f"Extracted {len(windows)} valid windows")

    if windows:
        features, clicked_ad = windows[0]
        print(f"First window features shape: {features.to_array().shape}")
        print(f"Ad clicked in window: {clicked_ad}")
        print("Sample features:")
        print(f"  Time since last action: {features.time_since_last_action:.2f}s")
        print(f"  Interaction density: {features.interaction_density:.3f} actions/s")
        print(f"  Attention score: {features.attention_score:.3f}")
        print(f"  Scroll depth max: {features.scroll_depth_max:.3f}")

    # Save feature names
    extractor.save_feature_names('/home/acreomarino/privads/models/feature_names.json')
    print("Feature names saved")