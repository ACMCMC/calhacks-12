# PrivAds Web Ad Service

A complete web ad customization pipeline that extracts web content, finds the best matching ad, customizes it with AI, and injects it into web pages.

## Features

- **Web Content Extraction**: Intelligently extracts important text, keywords, and context from any web page
- **Smart Ad Matching**: Uses ML models to find the best ad based on user preferences and web content
- **AI-Powered Customization**: Leverages Google's Gemini API to generate contextually relevant ad text
- **Multiple Injection Methods**: Supports iframe, web component, and native HTML injection
- **Browser Extension**: Seamless integration with a Chrome extension for easy testing and deployment
- **Privacy-First**: No user data stored, only embeddings and contextual signals

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Page      │ -> │  Text Extractor  │ -> │  Content       │
│   (URL)         │    │  (BeautifulSoup) │    │  Analysis      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ↓                           ↓                   ↓
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Best Ad       │ <- │  ML Matching     │ <- │  User Embedding │
│   Selection     │    │  (ChromaDB)      │    │  (128D Vector)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ↓                           ↓                   ↓
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gemini        │ -> │  Customized      │ -> │  Injection      │
│   Customization │    │  Ad Text         │    │  Code           │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export GOOGLE_AI_API_KEY="your_gemini_api_key"
export ELASTICSEARCH_URL="http://localhost:9200"
export CHROMA_URL="http://localhost:8001"
```

### 3. Start the Service

```bash
cd web_ad_service
python main.py
```

The service will be available at `http://localhost:8002`

### 4. Install Browser Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked" and select the `browser_extension` folder
4. The PrivAds extension will appear in your toolbar

## API Endpoints

### POST `/web_ad/complete`

Complete web ad customization pipeline.

**Request:**
```json
{
  "url": "https://example.com",
  "user_embedding": [0.1, -0.2, 0.3, ...],
  "user_id": "user123",
  "injection_method": "web_component",
  "position": "bottom"
}
```

**Response:**
```json
{
  "success": true,
  "ad_id": "ad_001",
  "customized_ad_text": "Revolutionary smartphone with AI-powered camera. Perfect for tech enthusiasts!",
  "injection_code": "<div id='privads-container-...'>...</div>",
  "web_context": {
    "page_type": "news",
    "keywords": ["tech", "smartphone", "ai"],
    "summary_text": "Latest tech news..."
  },
  "ad_context": {
    "ad_id": "ad_001",
    "similarity_score": 0.85,
    "ad_features": {...}
  },
  "customization_metadata": {
    "confidence_score": 0.92,
    "customization_applied": {...}
  }
}
```

### POST `/web_ad/quick`

Quick ad injection for testing (bypasses ML pipeline).

**Request:**
```json
{
  "url": "https://example.com",
  "ad_text": "Test advertisement"
}
```

### GET `/web_ad/extract/{url}`

Extract web content from a URL.

### GET `/web_ad/health`

Health check endpoint.

## Browser Extension Usage

1. **Enable Extension**: Click the PrivAds icon and toggle "Enable Ads"
2. **Quick Test**: Click "Quick Test Ad" to inject a test advertisement
3. **Auto-injection**: The extension automatically injects ads on page load
4. **Settings**: Configure injection method, delay, and other preferences

## Components

### Web Text Extractor (`web_text_extractor.py`)

Extracts important content from web pages:
- Page title and meta description
- Headings and paragraph text
- Keywords and topics
- Page type classification
- Content summarization

### Best Ad Service (`best_ad_service.py`)

Finds the best matching ad:
- ML-based ad matching using user embeddings
- Content relevance scoring
- Fallback keyword matching
- Ad feature analysis

### Gemini Customizer (`gemini_customizer.py`)

Customizes ads with AI:
- Context-aware text generation
- Web content integration
- Tone adaptation
- Confidence scoring

### Ad Injection Service (`ad_injection.py`)

Generates injection code:
- Multiple injection methods (iframe, web component, native)
- Responsive styling
- Click tracking
- Visibility monitoring

## Configuration

### Environment Variables

- `GOOGLE_AI_API_KEY`: Google AI API key for Gemini
- `ELASTICSEARCH_URL`: Elasticsearch instance URL
- `CHROMA_URL`: ChromaDB instance URL

### Extension Settings

- **Injection Method**: Choose between iframe, web component, or native HTML
- **Auto-inject Delay**: Delay before injecting ads (default: 2000ms)
- **Domain Restrictions**: Enable/disable ads for specific domains

## Testing

### Manual Testing

```bash
# Test web content extraction
curl "http://localhost:8002/web_ad/extract/https://example.com"

# Test quick ad injection
curl -X POST "http://localhost:8002/web_ad/quick" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "ad_text": "Test ad"}'

# Test complete pipeline
curl -X POST "http://localhost:8002/web_ad/complete" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

### Browser Extension Testing

1. Install the extension
2. Navigate to any website
3. Click the PrivAds icon
4. Use "Quick Test Ad" to inject a test advertisement
5. Verify the ad appears and is clickable

## Integration Examples

### JavaScript Integration

```javascript
// Inject ad programmatically
fetch('http://localhost:8002/web_ad/complete', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    url: window.location.href,
    user_embedding: userEmbedding
  })
})
.then(response => response.json())
.then(data => {
  if (data.success) {
    // Inject the ad code
    document.body.insertAdjacentHTML('beforeend', data.injection_code);
  }
});
```

### React Component

```jsx
import { useState, useEffect } from 'react';

function PrivAdsComponent({ url, userEmbedding }) {
  const [adCode, setAdCode] = useState('');
  
  useEffect(() => {
    fetch('http://localhost:8002/web_ad/complete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, user_embedding })
    })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        setAdCode(data.injection_code);
      }
    });
  }, [url, user_embedding]);
  
  return <div dangerouslySetInnerHTML={{ __html: adCode }} />;
}
```

## Privacy & Security

- **No Personal Data**: Only user embeddings (128D vectors) are used
- **Local Processing**: Web content extraction happens locally
- **Secure API**: All API calls use HTTPS in production
- **GDPR Compliant**: No persistent user data storage
- **Transparent**: All ad injections are clearly marked

## Troubleshooting

### Common Issues

1. **API Not Available**: Ensure the service is running on port 8002
2. **Gemini API Error**: Check your Google AI API key
3. **Extension Not Working**: Verify extension permissions and API connection
4. **Ads Not Injecting**: Check browser console for errors

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Check

```bash
curl http://localhost:8002/web_ad/health
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

