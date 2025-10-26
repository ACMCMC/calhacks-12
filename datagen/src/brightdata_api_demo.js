/**
 * Bright Data API Demo
 * Direct HTTP requests - maximum control and flexibility
 */

import 'dotenv/config';
import fetch from 'node-fetch';
import fs from 'fs';
import path from 'path';

const CONFIG = {
  apiKey: process.env.BRIGHTDATA_API_KEY,
  serpZone: process.env.SERP_ZONE,
  unlockerZone: process.env.UNLOCKER_ZONE,
  endpoint: 'https://api.brightdata.com/request'
};

console.log('METHOD 1: Direct API Requests\n');

// Example 1: Search
async function demoSearch(maxResults = 10) {
  console.log('Example 1: Searching Google...');
  
  const searchUrl = 'https://www.google.com/search?q=web+scraping+2025&brd_json=1';
  
  const response = await fetch(CONFIG.endpoint, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${CONFIG.apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      zone: CONFIG.serpZone,
      url: searchUrl,
      format: 'raw'
    })
  });

  const data = await response.json();
  const results = data.organic || [];
  
  console.log(`Found ${results.length} results:`);
  results.forEach((r, i) => console.log(`  ${i + 1}. ${r.title}`));
  console.log();

  // Collect up to maxResults URLs
  const urls = [];
  for (const r of results) {
    const u = r?.link || r?.url || r?.href;
    if (u) urls.push(u);
    if (urls.length >= maxResults) break;
  }
  return urls;
}

// Example 2: Fetch multiple pages concurrently and save as HTML
async function demoFetchAll(urls) {
  console.log('Example 2: Fetching page content (concurrent)...');

  if (!urls || urls.length === 0) {
    console.log('  No URLs to fetch; skipping.');
    return [];
  }

  const outDir = path.join(process.cwd(), 'out');
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }

  function sanitizeFilename(str) {
    return str.replace(/[^a-z0-9._-]/gi, '-').slice(0, 120);
  }

  const jobs = urls.map((url, idx) => (async () => {
    try {
      const res = await fetch(CONFIG.endpoint, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${CONFIG.apiKey}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          zone: CONFIG.unlockerZone,
          url,
          format: 'raw',
          data_format: 'html'
        })
      });

      const html = await res.text();
      const u = new URL(url);
      const baseName = `${String(idx + 1).padStart(2, '0')}-${sanitizeFilename(u.hostname + u.pathname) || 'page'}.html`;
      const filePath = path.join(outDir, baseName);
      fs.writeFileSync(filePath, html, 'utf8');
      console.log(`  Saved: ${filePath} (${html.length} chars)`);
      return { url, filePath, length: html.length };
    } catch (err) {
      console.log(`  Failed: ${url} (${err.message})`);
      return { url, error: err.message };
    }
  })());

  const results = await Promise.all(jobs);
  console.log();
  return results;
}

// Run demos
async function main() {
  try {
    const urls = await demoSearch(10);
    await demoFetchAll(urls);
    
    console.log('API demos complete');
    console.log('\nKey Points:');
    console.log('  - Simple HTTP POST requests');
    console.log('  - Works with any HTTP client');
    console.log('  - Bright Data handles blocking/CAPTCHAs');
    
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

main();
