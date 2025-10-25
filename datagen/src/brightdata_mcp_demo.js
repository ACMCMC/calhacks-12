/**
 * Bright Data MCP Demo with LangChain
 * Demonstrates actual MCP integration using LangChain's MultiServerMCPClient
 * Creates a ReAct agent with Bright Data tools for web search and scraping
 */

import 'dotenv/config';
import { ChatAnthropic } from '@langchain/anthropic';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { MultiServerMCPClient } from '@langchain/mcp-adapters';
import { fileURLToPath } from 'url';

// Content limit to match main workflow
const CONTENT_LIMITS = {
  SCRAPE_PREVIEW: 3000,
};


async function runMcpStyleDemo(query = 'AI news 2025') {
  try {
    if (!process.env.ANTHROPIC_API_KEY) {
      console.log('ANTHROPIC_API_KEY not set. This demo requires an LLM.');
      console.log('Run the main workflow (npm start) for heuristic fallback without an LLM.');
      return;
    }
    console.log(`\n${'='.repeat(60)}`);
    console.log('STEP 1: Initializing MCP Client');
    console.log('='.repeat(60));

    console.log('Configuring MCP client with Bright Data');
    const client = new MultiServerMCPClient({
      bright_data: {
        url: `https://mcp.brightdata.com/sse?token=${process.env.BRIGHTDATA_API_KEY}&pro=1`,
        transport: 'sse',
      },
    });

    console.log(`\n${'='.repeat(60)}`);
    console.log('STEP 2: Retrieving Tools from Bright Data MCP');
    console.log('='.repeat(60));

    const allTools = await client.getTools();
    console.log(`Retrieved ${allTools.length} tools from Bright Data MCP`);

    // Filter to commonly used tools for demo simplicity
    // All 61 tools use ~3.3K tokens per request, 2 tools use ~183 tokens
    const COMMON_TOOLS = ['search_engine', 'scrape_as_markdown'];
    const tools = allTools.filter(t => COMMON_TOOLS.includes(t.name));

    // Wrap scrape_as_markdown to limit content length
    const toolUsageStats = { search_engine: 0, scrape_as_markdown: 0 };

    const wrappedTools = tools.map(t => {
      if (t.name === 'scrape_as_markdown') {
        const originalInvoke = t.invoke.bind(t);
        return {
          ...t,
          invoke: async (input) => {
            toolUsageStats.scrape_as_markdown++;

            // Extract URL from different input formats
            let url = 'unknown';
            if (typeof input === 'string') {
              url = input;
            } else if (input && typeof input === 'object') {
              const keys = Object.keys(input);
              if (keys.length > 0) {
                url = input.url || input[keys[0]] || JSON.stringify(input).substring(0, 100);
              }
            }

            console.log(`[Agent] Scraping: ${url}`);
            const result = await originalInvoke(input);
            const content = typeof result === 'string' ? result : JSON.stringify(result);
            const truncated = content.substring(0, CONTENT_LIMITS.SCRAPE_PREVIEW);
            console.log(`[Agent] Scraped ${truncated.length} chars (limit: ${CONTENT_LIMITS.SCRAPE_PREVIEW})`);
            return truncated;
          }
        };
      } else if (t.name === 'search_engine') {
        const originalInvoke = t.invoke.bind(t);
        return {
          ...t,
          invoke: async (input) => {
            toolUsageStats.search_engine++;

            // Extract query and engine from different input formats
            let query = 'unknown';
            let engine = 'google';
            if (typeof input === 'string') {
              query = input;
            } else if (input && typeof input === 'object') {
              const keys = Object.keys(input);
              if (keys.length > 0) {
                query = input.query || input[keys[0]] || JSON.stringify(input).substring(0, 100);
                engine = input.engine || 'google';
              }
            }

            console.log(`[Agent] Searching: "${query}" on ${engine}`);
            const result = await originalInvoke(input);
            console.log(`[Agent] Search completed`);
            return result;
          }
        };
      }
      return t;
    });

    console.log(`Using ${wrappedTools.length} tools for demo: ${wrappedTools.map(t => t.name).join(', ')}`);
    console.log(`Content limit per scrape: ${CONTENT_LIMITS.SCRAPE_PREVIEW} characters`);

    console.log(`\n${'='.repeat(60)}`);
    console.log('STEP 3: Configuring LLM');
    console.log('='.repeat(60));

    const model = process.env.ANTHROPIC_MODEL || 'claude-sonnet-4-5';
    console.log(`Model: ${model}`);
    console.log(`Temperature: 0 (deterministic)`);

    const llm = new ChatAnthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
      model,
      temperature: 0,
    });

    console.log(`\n${'='.repeat(60)}`);
    console.log('STEP 4: Creating ReAct Agent');
    console.log('='.repeat(60));

    console.log('Agent uses Reasoning + Acting pattern to autonomously plan and execute tool calls');

    const systemPrompt = `
You are a web research agent with access to Bright Data's tools:
- search_engine: Search Google/Bing/Yandex
- scrape_as_markdown: Extract page content as markdown

Process queries step by step:
1. Search for relevant information
2. Scrape the most relevant pages (up to ${Math.floor(CONTENT_LIMITS.SCRAPE_PREVIEW / 1000)}k chars each)
3. Synthesize the information into a clear answer
`.trim();

    const agent = createReactAgent({
      llm,
      tools: wrappedTools,
      prompt: systemPrompt,
    });

    console.log('Agent created successfully');

    console.log(`\n${'='.repeat(60)}`);
    console.log('STEP 5: Executing Agent Query');
    console.log('='.repeat(60));

    console.log(`Query: "${query}"\n`);

    const startTime = Date.now();
    const result = await agent.invoke({
      messages: [{ role: 'user', content: query }],
    });
    const executionTime = Math.round((Date.now() - startTime) / 1000);

    console.log(`\n${'='.repeat(60)}`);
    console.log('STEP 6: Agent Execution Complete');
    console.log('='.repeat(60));

    const finalMsg = result.messages[result.messages.length - 1];
    console.log('\n' + '='.repeat(80));
    console.log('AGENT RESPONSE:');
    console.log('='.repeat(80));
    console.log(finalMsg?.content ?? '[No content]');
    console.log('='.repeat(80));

    console.log('\nAgent Statistics:');
    console.log(`   - Execution time: ${executionTime}s`);
    console.log(`   - Total messages: ${result.messages.length}`);
    console.log('   - Tool usage:');
    console.log(`     - search_engine: ${toolUsageStats.search_engine}x`);
    console.log(`     - scrape_as_markdown: ${toolUsageStats.scrape_as_markdown}x`);
    console.log('\n');

    try {
      await client.close();
      console.log('MCP client closed');
    } catch {}

  } catch (error) {
    console.error('MCP demo failed:', error.message);
    throw error;
  }
}

// Run if this is the main module
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const q = process.argv.slice(2).join(' ') || 'AI news 2025';
  runMcpStyleDemo(q).catch(err => {
    console.error('Error:', err.message);
    process.exit(1);
  });
}

export { runMcpStyleDemo };
