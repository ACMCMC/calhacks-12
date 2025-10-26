"""
Iframe Injection System
Handles the injection of customized ads into web pages via iframe or web components.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import base64
from datetime import datetime
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdInjectionRequest(BaseModel):
    """Request model for ad injection."""
    customized_ad: Dict[str, Any]
    target_url: str
    injection_method: str = "iframe"  # "iframe", "web_component", "native"
    position: str = "bottom"  # "top", "bottom", "sidebar", "inline"
    styling: Optional[Dict[str, Any]] = None

class AdInjectionResponse(BaseModel):
    """Response model for ad injection."""
    injection_code: str
    injection_method: str
    ad_container_id: str
    success: bool
    error_message: Optional[str] = None

class AdInjectionService:
    """Service for injecting customized ads into web pages."""
    
    def __init__(self):
        self.injection_templates = self._load_injection_templates()
        self.default_styling = self._get_default_styling()
    
    def inject_ad(self, request: AdInjectionRequest) -> AdInjectionResponse:
        """
        Generate injection code for a customized ad.
        
        Args:
            request: AdInjectionRequest containing ad and injection preferences
            
        Returns:
            AdInjectionResponse with injection code
        """
        try:
            # Generate unique container ID
            container_id = f"privads-container-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Get injection method
            injection_method = request.injection_method.lower()
            
            if injection_method == "iframe":
                injection_code = self._generate_iframe_injection(request, container_id)
            elif injection_method == "web_component":
                injection_code = self._generate_web_component_injection(request, container_id)
            elif injection_method == "native":
                injection_code = self._generate_native_injection(request, container_id)
            else:
                raise ValueError(f"Unsupported injection method: {injection_method}")
            
            return AdInjectionResponse(
                injection_code=injection_code,
                injection_method=injection_method,
                ad_container_id=container_id,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in ad injection: {e}")
            return AdInjectionResponse(
                injection_code="",
                injection_method=request.injection_method,
                ad_container_id="",
                success=False,
                error_message=str(e)
            )
    
    def _generate_iframe_injection(self, request: AdInjectionRequest, container_id: str) -> str:
        """Generate iframe-based injection code."""
        customized_ad = request.customized_ad
        styling = request.styling or self.default_styling
        
        # Create ad content HTML
        ad_html = self._create_ad_html(customized_ad, styling)
        
        # Encode ad content for iframe
        encoded_content = base64.b64encode(ad_html.encode('utf-8')).decode('utf-8')
        
        # Generate iframe injection code
        injection_code = f"""
<!-- PrivAds Customized Ad Injection -->
<div id="{container_id}" style="margin: 20px 0; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden;">
    <iframe 
        src="data:text/html;base64,{encoded_content}"
        width="100%" 
        height="200"
        frameborder="0"
        scrolling="no"
        style="border: none;"
        title="Customized Advertisement">
    </iframe>
</div>

<script>
// Add interaction tracking
document.getElementById('{container_id}').addEventListener('click', function(e) {{
    // Track ad interaction
    if (typeof gtag !== 'undefined') {{
        gtag('event', 'ad_click', {{
            'ad_id': '{customized_ad.get('original_ad_id', 'unknown')}',
            'target_url': '{request.target_url}',
            'injection_method': 'iframe'
        }});
    }}
    
    // Track custom event
    if (typeof window.privadsTracker !== 'undefined') {{
        window.privadsTracker.trackAdClick('{customized_ad.get('original_ad_id', 'unknown')}');
    }}
}});

// Add visibility tracking
const observer = new IntersectionObserver((entries) => {{
    entries.forEach(entry => {{
        if (entry.isIntersecting) {{
            // Track ad view
            if (typeof gtag !== 'undefined') {{
                gtag('event', 'ad_view', {{
                    'ad_id': '{customized_ad.get('original_ad_id', 'unknown')}',
                    'target_url': '{request.target_url}'
                }});
            }}
        }}
    }});
}}, {{ threshold: 0.5 }});

observer.observe(document.getElementById('{container_id}'));
</script>
"""
        
        return injection_code.strip()
    
    def _generate_web_component_injection(self, request: AdInjectionRequest, container_id: str) -> str:
        """Generate web component-based injection code."""
        customized_ad = request.customized_ad
        styling = request.styling or self.default_styling
        
        injection_code = f"""
<!-- PrivAds Customized Ad Web Component -->
<div id="{container_id}" class="privads-ad-container">
    <style>
        .privads-ad-container {{
            {self._css_dict_to_string(styling)}
        }}
        
        .privads-ad-content {{
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        
        .privads-ad-content::before {{
            content: 'Ad';
            position: absolute;
            top: 5px;
            right: 10px;
            font-size: 10px;
            opacity: 0.7;
        }}
        
        .privads-ad-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .privads-ad-description {{
            font-size: 14px;
            margin-bottom: 15px;
            line-height: 1.4;
        }}
        
        .privads-ad-cta {{
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
        }}
        
        .privads-ad-cta:hover {{
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
        }}
    </style>
    
    <div class="privads-ad-content">
        <div class="privads-ad-title">🎯 Personalized for You</div>
        <div class="privads-ad-description">{customized_ad.get('customized_ad_text', '')}</div>
        <a href="#" class="privads-ad-cta" onclick="privadsHandleClick(event)">Learn More</a>
    </div>
</div>

<script>
function privadsHandleClick(event) {{
    event.preventDefault();
    
    // Track click
    if (typeof gtag !== 'undefined') {{
        gtag('event', 'ad_click', {{
            'ad_id': '{customized_ad.get('original_ad_id', 'unknown')}',
            'target_url': '{request.target_url}',
            'injection_method': 'web_component'
        }});
    }}
    
    // Open ad destination (placeholder)
    window.open('https://example.com/ad/{customized_ad.get('original_ad_id', 'unknown')}', '_blank');
}}

// Add visibility tracking
const observer = new IntersectionObserver((entries) => {{
    entries.forEach(entry => {{
        if (entry.isIntersecting) {{
            if (typeof gtag !== 'undefined') {{
                gtag('event', 'ad_view', {{
                    'ad_id': '{customized_ad.get('original_ad_id', 'unknown')}',
                    'target_url': '{request.target_url}'
                }});
            }}
        }}
    }});
}}, {{ threshold: 0.5 }});

observer.observe(document.getElementById('{container_id}'));
</script>
"""
        
        return injection_code.strip()
    
    def _generate_native_injection(self, request: AdInjectionRequest, container_id: str) -> str:
        """Generate native HTML injection code."""
        customized_ad = request.customized_ad
        styling = request.styling or self.default_styling
        
        injection_code = f"""
<!-- PrivAds Native Ad Injection -->
<div id="{container_id}" class="privads-native-ad" style="{self._css_dict_to_string(styling)}">
    <div class="ad-content">
        <div class="ad-label">Sponsored</div>
        <div class="ad-text">{customized_ad.get('customized_ad_text', '')}</div>
        <div class="ad-actions">
            <button class="ad-button" onclick="privadsNativeClick(event)">Learn More</button>
        </div>
    </div>
</div>

<style>
.privads-native-ad {{
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 15px;
    margin: 20px 0;
    background: #f9f9f9;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}}

.ad-content {{
    position: relative;
}}

.ad-label {{
    font-size: 11px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}}

.ad-text {{
    font-size: 14px;
    line-height: 1.4;
    color: #333;
    margin-bottom: 12px;
}}

.ad-button {{
    background: #007bff;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    transition: background 0.2s;
}}

.ad-button:hover {{
    background: #0056b3;
}}
</style>

<script>
function privadsNativeClick(event) {{
    event.preventDefault();
    
    // Track click
    if (typeof gtag !== 'undefined') {{
        gtag('event', 'ad_click', {{
            'ad_id': '{customized_ad.get('original_ad_id', 'unknown')}',
            'target_url': '{request.target_url}',
            'injection_method': 'native'
        }});
    }}
    
    // Open ad destination
    window.open('https://example.com/ad/{customized_ad.get('original_ad_id', 'unknown')}', '_blank');
}}

// Add visibility tracking
const observer = new IntersectionObserver((entries) => {{
    entries.forEach(entry => {{
        if (entry.isIntersecting) {{
            if (typeof gtag !== 'undefined') {{
                gtag('event', 'ad_view', {{
                    'ad_id': '{customized_ad.get('original_ad_id', 'unknown')}',
                    'target_url': '{request.target_url}'
                }});
            }}
        }}
    }});
}}, {{ threshold: 0.5 }});

observer.observe(document.getElementById('{container_id}'));
</script>
"""
        
        return injection_code.strip()
    
    def _create_ad_html(self, customized_ad: Dict[str, Any], styling: Dict[str, Any]) -> str:
        """Create HTML content for the ad."""
        ad_text = customized_ad.get('customized_ad_text', '')
        ad_id = customized_ad.get('original_ad_id', 'unknown')
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customized Ad</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
        }}
        
        .ad-container {{
            max-width: 100%;
            margin: 0 auto;
        }}
        
        .ad-label {{
            font-size: 10px;
            opacity: 0.7;
            margin-bottom: 10px;
        }}
        
        .ad-content {{
            font-size: 14px;
            line-height: 1.4;
            margin-bottom: 15px;
        }}
        
        .ad-cta {{
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            transition: all 0.3s ease;
        }}
        
        .ad-cta:hover {{
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
        }}
    </style>
</head>
<body>
    <div class="ad-container">
        <div class="ad-label">🎯 Personalized Advertisement</div>
        <div class="ad-content">{ad_text}</div>
        <a href="#" class="ad-cta" onclick="handleClick()">Learn More</a>
    </div>
    
    <script>
        function handleClick() {{
            // Track click and open destination
            if (window.parent && window.parent !== window) {{
                window.parent.postMessage({{
                    type: 'ad_click',
                    ad_id: '{ad_id}'
                }}, '*');
            }}
            
            // Open destination URL
            window.open('https://example.com/ad/{ad_id}', '_blank');
        }}
    </script>
</body>
</html>
"""
        
        return html
    
    def _css_dict_to_string(self, css_dict: Dict[str, Any]) -> str:
        """Convert CSS dictionary to string."""
        if not css_dict:
            return ""
        
        css_parts = []
        for property, value in css_dict.items():
            # Convert camelCase to kebab-case
            css_property = ''.join(['-' + c.lower() if c.isupper() else c for c in property])
            css_parts.append(f"{css_property}: {value};")
        
        return ' '.join(css_parts)
    
    def _load_injection_templates(self) -> Dict[str, str]:
        """Load injection templates from files."""
        templates = {}
        template_dir = Path("templates")
        
        if template_dir.exists():
            for template_file in template_dir.glob("*.html"):
                with open(template_file, 'r') as f:
                    templates[template_file.stem] = f.read()
        
        return templates
    
    def _get_default_styling(self) -> Dict[str, Any]:
        """Get default styling for ads."""
        return {
            "width": "100%",
            "maxWidth": "600px",
            "margin": "20px auto",
            "borderRadius": "8px",
            "boxShadow": "0 2px 10px rgba(0,0,0,0.1)",
            "overflow": "hidden"
        }

# Initialize the injection service
ad_injection_service = AdInjectionService()

# Example usage and testing
if __name__ == "__main__":
    # Test the injection service
    test_request = AdInjectionRequest(
        customized_ad={
            'customized_ad_text': 'Revolutionary smartphone with AI-powered camera. Perfect for tech enthusiasts!',
            'original_ad_id': 'test_ad_001',
            'confidence_score': 0.85
        },
        target_url="https://example.com",
        injection_method="web_component",
        position="bottom"
    )
    
    result = ad_injection_service.inject_ad(test_request)
    print("Injection Code:")
    print(result.injection_code)
    print(f"\nSuccess: {result.success}")
    print(f"Container ID: {result.ad_container_id}")

