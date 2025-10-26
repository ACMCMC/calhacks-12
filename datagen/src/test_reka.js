/**
 * Test Reka API Integration
 * Simple test to verify Reka API is working
 */

import 'dotenv/config';
import fetch from 'node-fetch';

const CONFIG = {
  rekaApiKey: process.env.REKA_API_KEY,
  baseUrl: 'https://api.reka.ai/v1',
  model: 'reka-flash-research'
};

async function testRekaAPI() {
  console.log('🧪 Testing Reka API Integration');
  console.log('='.repeat(50));

  try {
    const response = await fetch(`${CONFIG.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${CONFIG.rekaApiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: CONFIG.model,
        messages: [
          {
            role: 'user',
            content: 'Hello! Can you tell me what you are and what you can do?'
          }
        ],
        max_tokens: 100,
        temperature: 0.1
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.log(`❌ API Error: ${response.status} ${response.statusText}`);
      console.log(`Error details: ${errorText}`);
      return;
    }

    const data = await response.json();
    console.log('✅ Reka API is working!');
    console.log('Response:', data.choices[0]?.message?.content);

  } catch (error) {
    console.log(`❌ Test failed: ${error.message}`);
  }
}

// Run the test
testRekaAPI();
