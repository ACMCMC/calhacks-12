# Live Web Access Workshop: Supercharge Your AI with Real-Time Data

Learn how to give your AI agents access to real time web data - the missing link between static LLMs and dynamic, credible applications.

## Why This Matters

AI applications are only as good as their data. While LLMs have vast knowledge, they're limited by their training cutoff dates and can't access real time information. This creates fundamental problems:

- Outdated information: Can't answer questions about recent events, current prices, or latest developments
- No source verification: Can't cite or verify information from authoritative sources
- Limited context: Missing the dynamic, constantly-updating nature of the web

The solution: Give your AI agents the ability to search, fetch, and process live web data in real time. This transforms static models into dynamic systems that can:
- Research current topics with citations
- Monitor prices and trends
- Access the latest information
- Build RAG systems with fresh data
- Power autonomous agents with real-world context

## What You'll Build

An intelligent AI research agent powered by MCP (Model Context Protocol) that:
- Autonomously searches and analyzes web sources
- Makes smart decisions about which pages to scrape
- Synthesizes comprehensive research reports with citations
- Uses LangChain's ReAct framework for intelligent reasoning

All while bypassing blocks, CAPTCHAs, and anti-bot measures automatically.

## Quick Start

### Prerequisites
- Node.js 18+
- Bright Data account (sign up at https://brightdata.com)

### Installation

```bash
git clone https://github.com/ScrapeAlchemist/The_Missing_Link_in_AI.git
cd The_Missing_Link_in_AI
npm install
```

### Configuration

1. Get API Key from https://brightdata.com/cp/settings
   - Navigate to "API tokens" → Generate new token

2. (Optional) Get Anthropic API Key from https://console.anthropic.com/settings/keys
   - Recommended for higher-quality synthesis (heuristic fallback used if omitted)
   
3. Set up environment:
```bash
cp .env.example .env
# Edit .env with your credentials
```

Required in `.env`:
```bash
BRIGHTDATA_API_KEY=your_api_key_here
```

Optional:
```bash
ANTHROPIC_API_KEY=your_anthropic_key_here   # Enables LLM synthesis (recommended)
SERP_ZONE=your_serp_zone_name             # Create a SERP API zone in Bright Data
UNLOCKER_ZONE=your_unlocker_zone          # Create a Web Unlocker zone in Bright Data
```

4. Verify your setup:
```bash
npm run verify
```

### Run

```bash
# Main workflow - Intelligent MCP agent analyzes 10 sources
npm start "your search query here"

# Examples:
npm start "latest AI news"
npm start "quantum computing breakthroughs 2025"

# Or run directly:
node src/main_workflow.js "your search query"
```

### Self-Paced Learning

**New to this?** Follow the step-by-step tutorial:
- **[TUTORIAL.md](TUTORIAL.md)** - Complete self-paced workshop (45-60 min)
- Includes exercises, troubleshooting, and challenges
- Perfect for independent learning after the live workshop

## Workshop Content

### Why Live Web Access Matters
- LLMs need fresh, verifiable data
- Use cases: research assistants, price tracking, RAG systems, trend analysis

### How Websites Block Bots
- IP blocking, CAPTCHAs, JavaScript challenges, fingerprinting
- Good news: Bright Data handles this automatically

### MCP Integration

The main workflow uses Bright Data MCP tools:
- Search relevant sources with domain diversity
- Scrape selected pages as clean markdown
- Synthesize a research report with citations
  - Uses Claude (Anthropic) if configured
  - Falls back to a heuristic, offline summarizer otherwise

**Alternative Access Methods** (for learning/comparison):
- `npm run demo:api` - Basic HTTP API approach (requires `SERP_ZONE` + `UNLOCKER_ZONE`)
- `npm run demo:mcp` - ReAct agent demo (requires `ANTHROPIC_API_KEY`)

**Note**: Bright Data also provides a [Python SDK](https://github.com/brightdata/bright-data-sdk-python) for high-level abstractions and type safety in Python applications.

### Responsible Use

- Do: Access public data, respect rate limits, follow TOS
- Don't: Spam, fake engagement, bypass paywalls, access private content

## Project Structure

```
.env                       # Environment variables (local, not committed)
src/
  main_workflow.js          # Main MCP-powered intelligent agent (10 sources)
  brightdata_api_demo.js    # Direct HTTP approach
  brightdata_mcp_demo.js    # Simple MCP demo
  verify_setup.js           # Setup verification tool
docs/
  GETTING_STARTED.md        # Detailed setup guide
  API_REFERENCE.md          # API documentation
```

## How the Workflow Works

The main workflow performs a simple, reliable pipeline:

1. Initialize Bright Data MCP client and tools
2. Run a search and select URLs with domain diversity
3. Scrape each page into clean markdown
4. Synthesize a report with citations
   - With Claude: higher-quality analysis and synthesis
   - Without Claude: heuristic summary highlighting headings, first sentences, keywords, and sources

```javascript
import { runWorkflow } from './src/main_workflow.js';

const result = await runWorkflow('quantum computing 2025', {
  maxResults: 10
});

console.log(result.result);        // Final research report (LLM or heuristic)
console.log(result.toolUsage);     // Which tools were used
console.log(result.executionTime); // Time taken
```

The workflow:
- Searches for relevant sources
- Scrapes pages as markdown
- Synthesizes findings with citations (LLM or heuristic)

## Documentation

- Getting Started: docs/GETTING_STARTED.md
- API Reference: docs/API_REFERENCE.md
- Bright Data Docs: https://docs.brightdata.com

## Common Use Cases

Research Assistant: Comprehensive analysis with citations
```javascript
import { runWorkflow } from './src/main_workflow.js';
const result = await runWorkflow('quantum computing 2025', { maxResults: 10 });
console.log(result.result); // Full research report
```

Price Monitoring: Compare prices across retailers
```javascript
const result = await runWorkflow('best laptop deals RTX 4090', { maxResults: 10 });
// Agent finds and analyzes multiple retailers
```

RAG Pipeline: Enrich LLM with live web data
```javascript
const webResearch = await runWorkflow(userQuery, { maxResults: 10 });
// webResearch.result contains synthesized information with sources
// Pass to your downstream LLM or use directly
```

---

## Show Your Support

Give a star if this workshop helped you!

## No LLM? Offline Mode

- Claude is optional for the main workflow. If `ANTHROPIC_API_KEY` is not set, the workflow still runs and produces a heuristic summary with citations (headings, first sentences, keywords, and sources).
- For higher quality synthesis, set `ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_MODEL`, default `claude-sonnet-4-5`).

## Demo Requirements

- `npm run demo:api` requires `SERP_ZONE` and `UNLOCKER_ZONE` in `.env` (Bright Data zones).
- `npm run demo:mcp` requires `ANTHROPIC_API_KEY` (uses a ReAct agent demo).
