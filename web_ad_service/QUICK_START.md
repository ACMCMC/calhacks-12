# PrivAds Web Ad Service - Quick Start Guide

## 🚀 Getting Started (No ML Pipeline Required)

You can test the core functionality without needing the full ML pipeline! Here's how:

### 1. Install Dependencies

```bash
cd web_ad_service
pip install -r requirements.txt
```

### 2. Test Individual Components

#### Test Gemini Customization (Works without API key!)
```bash
python test_gemini.py
```
This will:
- Test Gemini fallback mode (works without API key)
- Test with API key if you have one
- Create HTML test files with customized ads
- Show you exactly how the customization works

#### Test All Components
```bash
python test_components.py
```
This will:
- Test web content extraction
- Test Gemini customization with mock data
- Test ad injection
- Test the quick ad endpoint
- Generate HTML test files

### 3. Start the Service

```bash
python main.py
```
The service will start on `http://localhost:8002`

### 4. Test the API Endpoints

#### Quick Ad Test (Bypasses ML pipeline)
```bash
curl -X POST "http://localhost:8002/web_ad/quick" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "ad_text": "Test advertisement!"}'
```

#### Web Content Extraction Test
```bash
curl "http://localhost:8002/web_ad/extract/https://example.com"
```

#### Health Check
```bash
curl "http://localhost:8002/web_ad/health"
```

### 5. Install Browser Extension

1. Open Chrome → `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked" → Select `browser_extension/` folder
4. Click PrivAds icon in toolbar
5. Click "Quick Test Ad" on any website!

## 🎯 What You Can Test Right Now

### ✅ Working Without ML Pipeline:
- **Web content extraction** from any URL
- **Gemini ad customization** (fallback mode works without API key)
- **Ad injection** with multiple methods
- **Browser extension** functionality
- **Quick ad endpoint** for testing

### ⚠️ Requires ML Pipeline:
- **Best ad matching** (needs your trained models)
- **Complete pipeline** endpoint
- **User embedding** integration

## 🔧 Testing Scenarios

### Scenario 1: Test Gemini Fallback
```bash
python test_gemini.py
```
- Creates mock ad and web context
- Tests customization without API key
- Generates HTML test files
- Shows confidence scoring

### Scenario 2: Test Web Extraction
```bash
python test_components.py
```
- Tests extraction from example.com
- Shows page type classification
- Displays extracted keywords
- Generates summary text

### Scenario 3: Test Browser Extension
1. Install extension
2. Go to any website
3. Click PrivAds icon
4. Click "Quick Test Ad"
5. Watch ad appear!

## 📁 Generated Test Files

After running tests, you'll get:
- `gemini_test_*.html` - Customized ads with different scenarios
- `quick_ad_test.html` - Quick ad injection test
- `mock_ad_test.html` - Mock ad injection test

Open these in your browser to see the ads in action!

## 🔑 Optional: Enable Full Gemini AI

To get AI-powered customization instead of fallback templates:

1. Get Google AI API key: https://makersuite.google.com/app/apikey
2. Set environment variable:
   ```bash
   export GOOGLE_AI_API_KEY="your_key_here"
   ```
3. Run tests again:
   ```bash
   python test_gemini.py
   ```

## 🐛 Troubleshooting

### Service Won't Start
```bash
# Check if port 8002 is available
lsof -i :8002

# Install missing dependencies
pip install -r requirements.txt
```

### Extension Not Working
- Check if service is running on localhost:8002
- Open Chrome DevTools → Console for errors
- Verify extension permissions

### Gemini API Errors
- Check API key: `echo $GOOGLE_AI_API_KEY`
- Test API key: https://makersuite.google.com/app/apikey
- Fallback mode works without API key

## 🎉 Success Indicators

You'll know it's working when:
- ✅ `python test_gemini.py` shows "Fallback mode working!"
- ✅ `python test_components.py` shows all green checkmarks
- ✅ Browser extension injects ads on websites
- ✅ HTML test files show customized ads

## 🚀 Next Steps

Once basic testing works:
1. **Integrate with your ML models** (when ready)
2. **Add real ad inventory**
3. **Deploy to production**
4. **Add advanced features**

The foundation is solid - you can build on it incrementally!

