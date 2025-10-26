# Self-Paced Workshop Tutorial

**Time to complete**: 45-60 minutes
**Prerequisites**: Node.js 18+, Bright Data account (free signup)

## Learning Path

This tutorial is designed for independent learning. Follow the steps in order for the best experience.

---

## Part 1: Setup & First Run (15 min)

### Step 1.1: Clone & Install
```bash
git clone https://github.com/ScrapeAlchemist/The_Missing_Link_in_AI.git
cd The_Missing_Link_in_AI
npm install
```

### Step 1.2: Get API Credentials

**Why**: You need API access for the intelligent MCP agent to search and fetch pages without getting blocked.

1. **Sign up** at [brightdata.com](https://brightdata.com) (free tier available)
2. **Get Bright Data API key** at [brightdata.com/cp/settings](https://brightdata.com/cp/settings):
   - Navigate to "API tokens" → Generate new token → Copy it
3. **Get Anthropic API key** at [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys):
   - Optional but recommended for higher-quality synthesis (heuristic fallback used if omitted)

### Step 1.3: Configure Environment
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```bash
BRIGHTDATA_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=sk-ant-...  # Optional but recommended
```

Zones for API demo (optional): If you want to run the direct API demo (`npm run demo:api`) or validate all checks in `npm run verify`, also add your zone names:
```bash
SERP_ZONE=your_serp_zone_name       # Create a SERP API zone
UNLOCKER_ZONE=your_unlocker_zone    # Create a Web Unlocker zone
```

Note: Zones are not required for the main MCP agent, but are required for the legacy API demo and full verification.

### Step 1.4: Test Your Setup
```bash
npm start "latest AI developments 2025"
```

**Note**: `npm start` requires a search query as an argument. You can search for any topic:
```bash
npm start "your topic here"
```

**Expected output**: You should see:
- MCP agent initializing
- AI agent researching your query
- Intelligent analysis of up to 10 sources
- Comprehensive research report with citations
- Tool usage statistics

**Troubleshooting**:
- "Missing: BRIGHTDATA_API_KEY" → Check your `.env` file
- "ANTHROPIC_API_KEY not set" → This is OK! The workflow will use heuristic summarization
- "Authentication failed" → Verify BRIGHTDATA_API_KEY is correct

✅ **Checkpoint**: You've successfully run an intelligent MCP-powered research agent!

---

## Part 2: Understanding the MCP Agent (15 min)

**What is MCP?** Model Context Protocol allows AI agents (or simple workflows) to access tools as capabilities. The main workflow uses Bright Data MCP tools directly to search, scrape, and then synthesize results.

### Step 2.1: How the Agent Works

**Open the code**: [src/main_workflow.js](src/main_workflow.js)

**Key components**:
1. **MultiServerMCPClient**: Connects to Bright Data MCP server
2. **ChatAnthropic** (optional): LLM for higher-quality synthesis if `ANTHROPIC_API_KEY` is set
3. **Heuristic fallback**: Offline summarizer (headings, first sentences, keywords) when no LLM is configured

**The workflow autonomously**:
- Searches for relevant sources (up to 10)
- Selects URLs with domain diversity
- Scrapes each page as markdown
- Synthesizes comprehensive reports

### Step 2.2: Try Different Queries

```bash
# Academic research
npm start "CRISPR gene editing latest developments"

# E-commerce
npm start "best noise cancelling headphones 2025"

# Trend analysis
npm start "AI startup funding trends"
```

**Observe**:
- How the workflow selects diverse sources
- Which pages it scrapes
- How it synthesizes information
- Tool usage statistics at the end

### Step 2.3: Alternative Methods (Optional)

**Alternative: Direct API approach**:
```bash
npm run demo:api
```
Shows raw HTTP requests. Requires `SERP_ZONE` and `UNLOCKER_ZONE` in your `.env`.

**Simple MCP demo**:
```bash
npm run demo:mcp
```
Basic MCP integration with ReAct agent. Requires `ANTHROPIC_API_KEY`.

---

### Note on Python SDK

If you prefer working in Python, Bright Data offers a [Python SDK](https://github.com/brightdata/bright-data-sdk-python) that provides high-level abstractions and type safety for production applications.

✅ **Checkpoint**: You understand how the intelligent MCP agent works and have seen it analyze multiple sources!

---

## Part 3: Real-World Use Cases (15 min)

Now see how to apply this to real problems.

### Use Case 1: Research Assistant

**Problem**: Need to quickly research a topic with citations

```bash
npm start "quantum computing breakthroughs 2025"
```

**What it does**:
- Analyzes up to 10 sources
- Intelligently selects which pages to scrape
- Synthesizes comprehensive research report
- Includes citations automatically

**Applications**: Building RAG systems, academic research, fact-checking

---

### Use Case 2: Price Monitoring

**Problem**: Monitor product prices across retailers

```bash
npm start "MacBook Pro M4 price comparison"
```

**What it does**:
- Searches multiple retailers
- Extracts pricing information
- Compares across sources
- Provides organized comparison

**Applications**: E-commerce automation, deal alerts, competitive analysis

---

### Use Case 3: Trend Analysis

**Problem**: Track what's happening in an industry

```bash
npm start "electric vehicle market trends 2025"
```

**What it does**:
- Analyzes up to 10 industry sources
- Identifies key trends and patterns
- Aggregates insights from multiple perspectives
- Provides comprehensive market intelligence

**Applications**: Market intelligence, competitive analysis, trend monitoring

✅ **Checkpoint**: You've seen three practical applications!

---

## Part 4: Build Your Own (10 min)

Time to combine what you've learned.

### Challenge: Custom Workflow

Create a new file `my_workflow.js`:

```javascript
import { runWorkflow } from './src/main_workflow.js';

async function main() {
  const query = 'YOUR_QUERY';

  const result = await runWorkflow(query, {
    maxResults: 10   // Analyzes up to 10 sources
  });

  // The workflow returns a comprehensive report
  console.log('\n=== RESEARCH REPORT ===');
  console.log(result.result);

  console.log('\n=== STATISTICS ===');
  console.log('Execution time:', result.executionTime, 'seconds');
  console.log('Tools used:', result.toolUsage);
}

main().catch(console.error);
```

Run it:
```bash
node my_workflow.js
```

### Ideas to Try

**News Aggregator**: Get latest news on a topic
```javascript
const result = await runWorkflow('AI breakthroughs 2025', { maxResults: 10 });
```

**Product Comparison**: Compare multiple products
```javascript
const result = await runWorkflow('best noise cancelling headphones', { maxResults: 10 });
```

**Learning Assistant**: Research a new topic
```javascript
const result = await runWorkflow('how does CRISPR work', { maxResults: 10 });
```

---

## Part 5: Integrate with Your AI (Optional)

The workflow can be integrated with other systems:

```javascript
import { runWorkflow } from './src/main_workflow.js';

async function enhancedRAG(userQuestion) {
  // Get comprehensive web research
  const webResearch = await runWorkflow(userQuestion, { maxResults: 10 });

  // The workflow has already synthesized information from multiple sources
  const researchReport = webResearch.result;

  // Use this as context for your downstream LLM or application
  // For example, you could:
  // 1. Store in a vector database for RAG
  // 2. Pass to a specialized domain LLM
  // 3. Use in a multi-agent system

  return {
    research: researchReport,
    sources: webResearch.toolUsage,
    executionTime: webResearch.executionTime
  };
}

// Use it
const data = await enhancedRAG('What are the latest AI developments?');
console.log(data.research);
```

**What this achieves**: The workflow provides comprehensive, pre-synthesized research that you can integrate into larger AI systems.

---

## What You've Learned

✅ How to build intelligent MCP-powered research workflows
✅ Using Bright Data MCP tools for web search and scraping
✅ Workflow analyzes up to 10 sources with domain diversity
✅ Real-world use cases (research, pricing, trends)
✅ How to bypass blocks, CAPTCHAs automatically
✅ Building RAG pipelines with intelligent synthesis
✅ Python SDK availability for Python developers

---

## Next Steps

### Level Up Your Skills

1. **Customize the workflow**: Modify the system prompt in [src/main_workflow.js:182](src/main_workflow.js#L182) for specialized research
2. **Build a chatbot**: Use the workflow as a research backend for conversational AI
3. **Monitor continuously**: Set up scheduled jobs to track topic changes over time
4. **Multi-workflow systems**: Combine with other specialized workflows for complex tasks

### Resources

- **Bright Data Docs**: [docs.brightdata.com](https://docs.brightdata.com)
- **SERP API Guide**: [docs.brightdata.com/scraping-automation/serp-api](https://docs.brightdata.com/scraping-automation/serp-api)
- **Web Unlocker Guide**: [docs.brightdata.com/scraping-automation/web-unlocker](https://docs.brightdata.com/scraping-automation/web-unlocker)

### Share Your Project

Built something cool? We'd love to see it!
- Tag [@Bright_Data](https://twitter.com/Bright_Data) on Twitter
- Star this repo ⭐
- Open a PR with your custom workflow

---

## Responsible Use

✅ **Do**:
- Access public data
- Respect rate limits (Bright Data handles this)
- Follow website Terms of Service
- Build helpful tools

❌ **Don't**:
- Spam requests
- Bypass paywalls
- Scrape private/authenticated content
- Create fake engagement

---

## Troubleshooting

**Problem**: Workflow takes a long time
**Solution**: The workflow analyzes up to 10 sources intelligently. This is normal for comprehensive research. Reduce `maxResults` for faster testing.

**Problem**: Output is too brief
**Solution**: The workflow may need a more specific query. Try being more detailed in your search.

**Problem**: "Missing: ANTHROPIC_API_KEY"
**Solution**: The workflow will run with heuristic summarization when Claude is not configured. For higher-quality output, add your key to `.env`.

**Problem**: "Authentication failed"
**Solution**: Verify BRIGHTDATA_API_KEY is correct in `.env`

**Problem**: Most pages failed to scrape
**Solution**: Check your internet connection. The workflow will warn you if too many scrapes fail.

---

**Questions?** Open an issue on GitHub or check [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)

**Ready to build?** Start experimenting with different queries or create your own workflow!

---

## Appendix: No-LLM Offline Mode

- Don't have an Anthropic key? You can still complete the workshop.
- The main workflow (`npm start`) will run heuristic summarization when `ANTHROPIC_API_KEY` is not set:
  - Uses headings and first sentences from scraped pages
  - Highlights common keywords
  - Always includes source URLs (citations)
- Quality will be lower than using an LLM, but it's reliable and sufficient for learning the MCP flow.
- The `demo:mcp` script requires Claude (Anthropic) for the ReAct agent demo.
