/**
 * Main Workflow: MCP-Powered Research Agent
 * Uses Bright Data MCP tools + LangChain to research and synthesize results
 */

import 'dotenv/config';
import { ChatAnthropic } from '@langchain/anthropic';
import { MultiServerMCPClient } from '@langchain/mcp-adapters';
import { fileURLToPath } from 'url';

// Configuration constants
const CONFIG = {
  apiKey: process.env.BRIGHTDATA_API_KEY,
  anthropicKey: process.env.ANTHROPIC_API_KEY,
  anthropicModel: process.env.ANTHROPIC_MODEL || 'claude-sonnet-4-5',
};

// Content processing constants
const CONTENT_LIMITS = {
  SCRAPE_PREVIEW: 3000,  // Characters to include per scraped page in synthesis
  MAX_RESULTS: 10,       // Default maximum URLs to analyze
  MAX_DOMAINS_PER_SOURCE: 2, // Maximum pages per domain for diversity
};

// Stopwords for heuristic keyword extraction
const STOPWORDS = new Set([
  'the','and','a','to','of','in','for','on','with','at','by','from','as','that','this','it','is','are','was','were','be','or','an','if','than','then','so','but','we','you','your','they','their','our','us','not','can','will','may','more','most','about','into','over','also','new','latest'
]);

function validateConfig() {
  const missing = [];
  if (!CONFIG.apiKey) missing.push('BRIGHTDATA_API_KEY');
  if (missing.length > 0) throw new Error(`Missing: ${missing.join(', ')}. Check your .env file.`);
}

function validateInput(query, maxResults) {
  if (!query || typeof query !== 'string' || query.trim().length === 0) {
    throw new Error('Query must be a non-empty string');
  }
  if (maxResults !== undefined && (typeof maxResults !== 'number' || maxResults <= 0 || maxResults > 50)) {
    throw new Error('maxResults must be a positive number between 1 and 50');
  }
}


/**
 * Selects diverse URLs from search results
 * Strategy: Limit pages per domain to ensure variety of sources
 */
function selectDiverseUrls(searchData, maxResults) {
  const urls = [];
  const perDomain = {};

  if (searchData.organic && Array.isArray(searchData.organic)) {
    for (const result of searchData.organic) {
      if (result.link && urls.length < maxResults) {
        try {
          const domain = new URL(result.link).hostname.replace('www.', '');

          // Limit pages per domain for diversity
          if ((perDomain[domain] || 0) < CONTENT_LIMITS.MAX_DOMAINS_PER_SOURCE) {
            urls.push(result.link);
            perDomain[domain] = (perDomain[domain] || 0) + 1;
          }
        } catch {
          // Skip invalid URLs
        }
      }
    }
  }

  return { urls, domainCount: Object.keys(perDomain).length };
}

async function runWorkflow(query, options = {}) {
  const { maxResults = CONTENT_LIMITS.MAX_RESULTS } = options;

  validateInput(query, maxResults);

  console.log(`\n${'='.repeat(60)}`);
  console.log('STEP 1: Initializing MCP Client');
  console.log('='.repeat(60));
  console.log(`Query: "${query}"`);
  console.log(`Max results to analyze: ${maxResults}`);

  if (!CONFIG.anthropicKey) {
    console.log('\nNote: ANTHROPIC_API_KEY not set - using heuristic summarization');
    console.log('Add ANTHROPIC_API_KEY to .env for LLM-powered synthesis\n');
  }

  const client = new MultiServerMCPClient({
    bright_data: {
      url: `https://mcp.brightdata.com/sse?token=${CONFIG.apiKey}&pro=1`,
      transport: 'sse',
    },
  });

  console.log(`\n${'='.repeat(60)}`);
  console.log('STEP 2: Loading Bright Data Tools');
  console.log('='.repeat(60));
  const allTools = await client.getTools();
  const searchTool = allTools.find(t => t.name === 'search_engine');
  const scrapeTool = allTools.find(t => t.name === 'scrape_as_markdown');

  if (!searchTool || !scrapeTool) {
    throw new Error('Required tools not found: search_engine and scrape_as_markdown');
  }
  console.log('Loaded tools: search_engine, scrape_as_markdown');

  console.log(`\n${'='.repeat(60)}`);
  console.log('STEP 3: Searching for Results');
  console.log('='.repeat(60));
  const searchResult = await searchTool.invoke({ query, engine: 'google' });

  console.log(`\n${'='.repeat(60)}`);
  console.log('STEP 4: Selecting URLs with Domain Diversity');
  console.log('='.repeat(60));
  const llm = CONFIG.anthropicKey
    ? new ChatAnthropic({ apiKey: CONFIG.anthropicKey, model: CONFIG.anthropicModel, temperature: 0 })
    : null;

  let urls = [];
  let domainCount = 0;

  try {
    const searchData = typeof searchResult === 'string' ? JSON.parse(searchResult) : searchResult;
    const selection = selectDiverseUrls(searchData, maxResults);
    urls = selection.urls;
    domainCount = selection.domainCount;

    console.log(`Selected ${urls.length} URLs from ${domainCount} domains`);
  } catch (e) {
    console.error(`Failed to parse search results: ${e.message}`);
  }

  if (urls.length === 0) {
    throw new Error('No URLs found in search results');
  }

  console.log(`\n${'='.repeat(60)}`);
  console.log('STEP 5: Scraping and Analyzing Pages');
  console.log('='.repeat(60));
  const scrapedContent = [];

  for (let i = 0; i < urls.length; i++) {
    const url = urls[i];
    try {
      const domain = new URL(url).hostname.replace('www.', '');
      console.log(`Scraping ${i + 1}/${urls.length}: ${domain}`);
      const content = await scrapeTool.invoke({ url });
      scrapedContent.push({
        url,
        domain,
        content: typeof content === 'string' ? content : JSON.stringify(content)
      });
    } catch (error) {
      console.log(`Failed to scrape ${url}: ${error.message}`);
    }
  }

  if (scrapedContent.length < urls.length * 0.3) {
    console.log(`\nWarning: Only ${scrapedContent.length}/${urls.length} pages scraped successfully`);
    console.log('Results may be incomplete\n');
  }

  console.log(`\n${'='.repeat(60)}`);
  console.log('STEP 6: Synthesizing Research Report');
  console.log('='.repeat(60));
  const started = Date.now();

  const combinedContent = scrapedContent.map((item, i) =>
    `\n=== Source ${i + 1} [${item.domain}]: ${item.url} ===\n${item.content.substring(0, CONTENT_LIMITS.SCRAPE_PREVIEW)}`
  ).join('\n\n');

  const uniqueDomains = new Set(scrapedContent.map(s => s.domain));
  console.log(`Synthesizing from ${scrapedContent.length} pages across ${uniqueDomains.size} domains`);

  let result;
  if (llm) {
    const synthesisResponse = await llm.invoke([
      { role: 'system', content: 'Analyze web sources and create a comprehensive research report. Include key findings and cite sources with URLs.' },
      { role: 'user', content: `Query: ${query}\n\nSources:\n${combinedContent}\n\nCreate a comprehensive research report with citations.` }
    ]);
    result = synthesisResponse.content;
  } else {
    result = heuristicSummarize(query, scrapedContent);
  }

  const executionTime = Math.round((Date.now() - started) / 1000);

  console.log(`\n${'='.repeat(60)}`);
  console.log('STEP 7: Research Complete');
  console.log('='.repeat(60));
  console.log(`Completed in ${executionTime} seconds`);

  // Display results
  console.log('\n' + '='.repeat(80));
  console.log('RESEARCH RESULTS');
  console.log('='.repeat(80));
  console.log(result);
  console.log('='.repeat(80));

  console.log('\nResearch Statistics:');
  console.log(`   - Execution time: ${executionTime}s`);
  console.log(`   - Sources analyzed: ${scrapedContent.length}`);
  console.log('   - Tools used:');
  console.log('     - search_engine: 1x');
  console.log(`     - scrape_as_markdown: ${scrapedContent.length}x`);
  console.log('\n');

  try { await client.close(); } catch {}

  return {
    query,
    result,
    toolUsage: { search_engine: 1, scrape_as_markdown: scrapedContent.length },
    executionTime,
    sourcesAnalyzed: scrapedContent.length,
  };
}

async function main() {
  try {
    validateConfig();
    const args = process.argv.slice(2);
    const query = args.length > 0 ? args.join(' ') : 'latest AI developments 2025';
    await runWorkflow(query, { maxResults: CONTENT_LIMITS.MAX_RESULTS });
  } catch (error) {
    console.error(`Error: ${error.message}`);
    process.exit(1);
  }
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  main();
}

export { runWorkflow };

// Heuristic Fallback Logic

/**
 * Extract top keywords from text, filtering stopwords
 */
function topKeywords(text, n = 15) {
  const counts = new Map();
  for (const raw of text.toLowerCase().split(/[^a-z0-9]+/g)) {
    const w = raw.trim();
    if (!w || STOPWORDS.has(w) || w.length < 3) continue;
    counts.set(w, (counts.get(w) || 0) + 1);
  }
  return [...counts.entries()].sort((a,b)=>b[1]-a[1]).slice(0,n).map(([w])=>w);
}

/**
 * Extract first sentences from markdown content
 */
function firstSentences(md, max = 3) {
  const text = md.replace(/[#>*`_\-]+/g, ' ').replace(/\s+/g, ' ').trim();
  const sentences = text.split(/(?<=[.!?])\s+/).filter(Boolean);
  return sentences.slice(0, max);
}

/**
 * Extract headings from markdown content
 */
function extractHeadings(md, max = 5) {
  const lines = md.split(/\r?\n/);
  const heads = [];
  for (const line of lines) {
    const m = line.match(/^(#{1,3})\s+(.*)/);
    if (m) heads.push(m[2].trim());
    if (heads.length >= max) break;
  }
  return heads;
}

/**
 * Generate heuristic summary when LLM is not available
 * Uses headings, first sentences, and keyword extraction
 */
function heuristicSummarize(query, scrapedContent) {
  const pieces = scrapedContent.map((s, i) => ({
    url: s.url,
    domain: s.domain,
    headings: extractHeadings(s.content, 2),
    sentences: firstSentences(s.content, 2)
  }));

  const combinedText = scrapedContent.map(s => s.content).join('\n');
  const keywords = topKeywords(combinedText, 12);

  const lines = [];
  lines.push(`Heuristic Summary for: ${query}`);
  lines.push('');
  lines.push('Overview:');
  for (const p of pieces.slice(0, 5)) {
    if (p.headings.length > 0) {
      lines.push(`- ${p.headings[0]} (${p.domain})`);
    } else if (p.sentences.length > 0) {
      lines.push(`- ${p.sentences[0]} (${p.domain})`);
    }
  }
  if (pieces.length > 5) lines.push(`- ...and ${pieces.length - 5} more sources`);

  lines.push('');
  lines.push('Key Topics:');
  lines.push(`- ${keywords.join(', ')}`);

  lines.push('');
  lines.push('Sources:');
  for (const p of pieces) {
    lines.push(`- ${p.domain}: ${p.url}`);
  }

  lines.push('');
  lines.push('(Rendered without an LLM. Set ANTHROPIC_API_KEY for higher-quality synthesis.)');

  return lines.join('\n');
}
