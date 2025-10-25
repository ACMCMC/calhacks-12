/**
 * Setup Verification Script
 * Checks Node version, environment variables, Bright Data API, and optional Anthropic
 */

import 'dotenv/config';
import fetch from 'node-fetch';

const REQUIRED_ENV = ['BRIGHTDATA_API_KEY'];

console.log('Bright Data Workshop - Setup Verification');
console.log('============================================================');

let allChecks = true;

console.log('\n1) Checking Node.js version...');
const nodeVersion = process.version;
const majorVersion = parseInt(nodeVersion.slice(1).split('.')[0]);
if (majorVersion >= 18) {
  console.log(`   OK: Node.js ${nodeVersion} (>=18 required)`);
} else {
  console.log(`   ERROR: Node.js ${nodeVersion} is too old (>=18 required)`);
  console.log('   Hint: Install Node.js 18 or higher from https://nodejs.org');
  allChecks = false;
}

console.log('\n2) Checking environment variables...');
for (const key of REQUIRED_ENV) {
  if (process.env[key]) {
    console.log(`   OK: ${key} is set`);
  } else {
    console.log(`   MISSING: ${key}`);
    console.log(`   Hint: Add ${key} to your .env file`);
    allChecks = false;
  }
}

console.log('\n   Optional variables:');
if (process.env.ANTHROPIC_API_KEY) {
  console.log('   OK: ANTHROPIC_API_KEY is set (enables LLM summaries)');
} else {
  console.log('   Note: ANTHROPIC_API_KEY not set (heuristic summaries will be used)');
}
if (process.env.SERP_ZONE) {
  console.log('   OK: SERP_ZONE is set (enables SERP API demo)');
} else {
  console.log('   Note: SERP_ZONE not set (demo:api search test will be skipped)');
}
if (process.env.UNLOCKER_ZONE) {
  console.log('   OK: UNLOCKER_ZONE is set (enables Web Unlocker demo)');
} else {
  console.log('   Note: UNLOCKER_ZONE not set (demo:api fetch test will be skipped)');
}

if (process.env.BRIGHTDATA_API_KEY && process.env.SERP_ZONE) {
  console.log('\n3) Testing Bright Data SERP API...');
  try {
    const testUrl = 'https://www.google.com/search?q=test&num=1&brd_json=1';

    const response = await fetch('https://api.brightdata.com/request', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.BRIGHTDATA_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        zone: process.env.SERP_ZONE,
        url: testUrl,
        format: 'raw'
      })
    });

    if (response.ok) {
      console.log('   OK: SERP API connection successful');
      const data = await response.json();
      console.log(`   OK: Received ${data.organic?.length || 0} search results`);
    } else {
      console.log(`   ERROR: SERP API connection failed: ${response.status}`);
      if (response.status === 401) {
        console.log('   Hint: Check BRIGHTDATA_API_KEY at https://brightdata.com/cp/settings');
      } else if (response.status === 403) {
        console.log('   Hint: Check SERP_ZONE name at https://brightdata.com/cp/zones');
      } else if (response.status === 402) {
        console.log('   Hint: Add credits at https://brightdata.com/cp/billing');
      }
      allChecks = false;
    }
  } catch (error) {
    console.log(`   ERROR: Network error: ${error.message}`);
    allChecks = false;
  }
} else {
  console.log('\n3) Skipping SERP API test (credentials not configured)');
}

if (process.env.BRIGHTDATA_API_KEY && process.env.UNLOCKER_ZONE) {
  console.log('\n4) Testing Web Unlocker...');
  try {
    const response = await fetch('https://api.brightdata.com/request', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.BRIGHTDATA_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        zone: process.env.UNLOCKER_ZONE,
        url: 'https://example.com',
        format: 'raw',
        data_format: 'markdown'
      })
    });

    if (response.ok) {
      console.log('   OK: Web Unlocker connection successful');
      const content = await response.text();
      console.log(`   OK: Fetched ${content.length} characters`);
    } else {
      console.log(`   ERROR: Web Unlocker connection failed: ${response.status}`);
      if (response.status === 403) {
        console.log('   Hint: Check UNLOCKER_ZONE name at https://brightdata.com/cp/zones');
      }
      allChecks = false;
    }
  } catch (error) {
    console.log(`   ERROR: Network error: ${error.message}`);
    allChecks = false;
  }
} else {
  console.log('\n4) Skipping Web Unlocker test (credentials not configured)');
}

if (process.env.ANTHROPIC_API_KEY) {
  console.log('\n5) Testing Anthropic API (optional)...');
  try {
    const { default: Anthropic } = await import('@anthropic-ai/sdk');
    const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    const response = await client.messages.create({
      model: process.env.ANTHROPIC_MODEL || 'claude-sonnet-4-5',
      max_tokens: 10,
      messages: [{ role: 'user', content: 'Hi' }],
    });
    if (response.content?.[0]?.text) {
      console.log('   OK: Anthropic API connection successful');
      console.log(`   OK: Using model: ${process.env.ANTHROPIC_MODEL || 'claude-sonnet-4-5'}`);
    }
  } catch (error) {
    console.log(`   ERROR: Anthropic API error: ${error.message}`);
    console.log('   Note: This is optional; workshop works without it');
  }
} else {
  console.log('\n5) Anthropic not configured (optional)');
  console.log('   Note: Summaries will use heuristic instead of LLM');
}

console.log('\n============================================================');
if (allChecks) {
  console.log('\nALL CHECKS PASSED. You are ready to go.\n');
  console.log('Next steps:');
  console.log('  npm run verify            # Re-run verification');
  console.log('  npm start                 # Run main workflow');
  console.log('  npm run demo:api          # Try direct API approach');
  console.log('  npm run demo:mcp          # Try MCP demo with ReAct agent');
  console.log('\nOr follow the tutorial: TUTORIAL.md\n');
} else {
  console.log('\nSOME CHECKS FAILED. Please fix the issues above.\n');
  console.log('Common fixes:');
  console.log('  1. Copy .env.example to .env');
  console.log('  2. Add your credentials from https://brightdata.com/cp');
  console.log('  3. Make sure zone names match exactly');
  console.log('\nNeed help? Check docs/GETTING_STARTED.md\n');
  process.exit(1);
}
