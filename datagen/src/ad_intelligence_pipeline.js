/**
 * Ad Intelligence Pipeline
 * Bright Data + Reka-powered feature extraction for ad creatives
 */

import 'dotenv/config';
import { MultiServerMCPClient } from '@langchain/mcp-adapters';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const CONFIG = {
  brightdataApiKey: process.env.BRIGHTDATA_API_KEY,
  rekaApiKey: process.env.REKA_API_KEY,
  targetImages: 10000,
  targetVideos: 1000,
  batchSize: 50,
  maxProcessingTime: 300, // 5 minutes
};

// Feature extraction categories
const FEATURE_CATEGORIES = {
  VISUAL: [
    'object_detection',
    'scene_classification', 
    'color_analysis',
    'composition_analysis',
    'text_extraction',
    'brand_recognition'
  ],
  VIDEO: [
    'motion_analysis',
    'scene_transitions',
    'audio_features',
    'temporal_patterns',
    'face_analysis'
  ],
  CREATIVE: [
    'emotional_sentiment',
    'product_category',
    'ad_type_classification',
    'call_to_action_strength',
    'visual_complexity'
  ]
};

class AdIntelligencePipeline {
  constructor() {
    this.mcpClient = null;
    this.collectedAssets = [];
    this.features = new Map();
    this.processingStats = {
      imagesProcessed: 0,
      videosProcessed: 0,
      featuresExtracted: 0,
      startTime: Date.now()
    };
  }

  async initialize() {
    console.log('🚀 Initializing Ad Intelligence Pipeline');
    
    // Initialize Bright Data MCP client
    this.mcpClient = new MultiServerMCPClient({
      bright_data: {
        url: `https://mcp.brightdata.com/sse?token=${CONFIG.brightdataApiKey}&pro=1`,
        transport: 'sse',
      },
    });

    console.log('✅ Bright Data MCP client initialized');
    return this;
  }

  /**
   * Phase 1: Collect ad creatives using Bright Data
   */
  async collectAdCreatives() {
    console.log('\n📊 PHASE 1: Collecting Ad Creatives');
    console.log('='.repeat(60));

    const collectionSources = [
      {
        name: 'Facebook Ads Library',
        searchQueries: [
          'automotive ads',
          'fashion ads', 
          'tech ads',
          'food ads',
          'travel ads'
        ],
        platform: 'facebook'
      },
      {
        name: 'Google Ads Transparency',
        searchQueries: [
          'commercial ads',
          'product ads',
          'brand ads'
        ],
        platform: 'google'
      },
      {
        name: 'TikTok Ads',
        searchQueries: [
          'tiktok ads',
          'video ads',
          'mobile ads'
        ],
        platform: 'tiktok'
      }
    ];

    const tools = await this.mcpClient.getTools();
    const searchTool = tools.find(t => t.name === 'search_engine');
    const scrapeTool = tools.find(t => t.name === 'scrape_as_markdown');

    if (!searchTool || !scrapeTool) {
      throw new Error('Required Bright Data tools not available');
    }

    for (const source of collectionSources) {
      console.log(`\n🔍 Collecting from ${source.name}`);
      
      for (const query of source.searchQueries) {
        try {
          // Search for ad libraries
          const searchResults = await searchTool.invoke({
            query: `${query} ${source.name} ads library`,
            engine: 'google'
          });

          // Extract ad URLs and metadata
          const adUrls = await this.extractAdUrls(searchResults, scrapeTool);
          
          // Download creative assets
          const assets = await this.downloadAssets(adUrls, scrapeTool);
          
          this.collectedAssets.push(...assets);
          
          console.log(`   ✅ Collected ${assets.length} assets for "${query}"`);
          
          // Check if we have enough assets
          if (this.collectedAssets.length >= CONFIG.targetImages + CONFIG.targetVideos) {
            console.log('🎯 Target collection size reached!');
            break;
          }
          
        } catch (error) {
          console.log(`   ❌ Failed to collect from ${source.name}: ${error.message}`);
        }
      }
    }

    console.log(`\n📈 Collection Summary:`);
    console.log(`   - Total assets collected: ${this.collectedAssets.length}`);
    console.log(`   - Images: ${this.collectedAssets.filter(a => a.type === 'image').length}`);
    console.log(`   - Videos: ${this.collectedAssets.filter(a => a.type === 'video').length}`);
    
    return this.collectedAssets;
  }

  /**
   * Phase 2: Extract features using Reka
   */
  async extractFeatures() {
    console.log('\n🧠 PHASE 2: Feature Extraction with Reka');
    console.log('='.repeat(60));

    const assets = this.collectedAssets.slice(0, CONFIG.targetImages + CONFIG.targetVideos);
    const batches = this.createBatches(assets, CONFIG.batchSize);

    console.log(`Processing ${assets.length} assets in ${batches.length} batches`);

    for (let i = 0; i < batches.length; i++) {
      const batch = batches[i];
      console.log(`\n📦 Processing batch ${i + 1}/${batches.length} (${batch.length} assets)`);

      // Process batch in parallel
      const batchPromises = batch.map(asset => this.processAsset(asset));
      const batchResults = await Promise.allSettled(batchPromises);

      // Collect successful results
      batchResults.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          this.features.set(batch[index].id, result.value);
          this.processingStats.featuresExtracted++;
        } else {
          console.log(`   ❌ Failed to process ${batch[index].id}: ${result.reason}`);
        }
      });

      // Check time constraint
      const elapsed = (Date.now() - this.processingStats.startTime) / 1000;
      if (elapsed > CONFIG.maxProcessingTime) {
        console.log(`⏰ Time limit reached (${elapsed}s). Stopping processing.`);
        break;
      }
    }

    console.log(`\n📊 Feature Extraction Summary:`);
    console.log(`   - Assets processed: ${this.features.size}`);
    console.log(`   - Features extracted: ${this.processingStats.featuresExtracted}`);
    console.log(`   - Processing time: ${((Date.now() - this.processingStats.startTime) / 1000).toFixed(2)}s`);

    return this.features;
  }

  /**
   * Process individual asset with Reka
   */
  async processAsset(asset) {
    const features = {
      asset_id: asset.id,
      type: asset.type,
      url: asset.url,
      extracted_at: new Date().toISOString(),
      features: {}
    };

    try {
      // Visual features (for both images and videos)
      if (asset.type === 'image' || asset.type === 'video') {
        features.features.visual = await this.extractVisualFeatures(asset);
      }

      // Video-specific features
      if (asset.type === 'video') {
        features.features.video = await this.extractVideoFeatures(asset);
      }

      // Creative intelligence features
      features.features.creative = await this.extractCreativeFeatures(asset);

      return features;

    } catch (error) {
      console.log(`   ❌ Failed to extract features for ${asset.id}: ${error.message}`);
      throw error;
    }
  }

  /**
   * Extract visual features using Reka
   */
  async extractVisualFeatures(asset) {
    const visualFeatures = {};

    // Object detection
    visualFeatures.objects = await this.callReka('reka-core-20241211', {
      task: 'object_detection',
      input: asset.url,
      prompt: 'Identify all objects, products, logos, and people in this ad creative'
    });

    // Scene classification
    visualFeatures.scene = await this.callReka('reka-core-20241211', {
      task: 'scene_classification',
      input: asset.url,
      prompt: 'Classify the scene: indoor/outdoor, setting type, lighting conditions'
    });

    // Color analysis
    visualFeatures.colors = await this.callReka('reka-core-20241211', {
      task: 'color_analysis',
      input: asset.url,
      prompt: 'Analyze dominant colors, color harmony, and emotional impact of colors'
    });

    // Text extraction (OCR)
    visualFeatures.text = await this.callReka('reka-core-20241211', {
      task: 'text_extraction',
      input: asset.url,
      prompt: 'Extract all text including headlines, CTAs, and product names'
    });

    return visualFeatures;
  }

  /**
   * Extract video-specific features
   */
  async extractVideoFeatures(asset) {
    const videoFeatures = {};

    // Motion analysis
    videoFeatures.motion = await this.callReka('reka-video-20241211', {
      task: 'motion_analysis',
      input: asset.url,
      prompt: 'Analyze motion intensity, direction, and patterns throughout the video'
    });

    // Audio features
    videoFeatures.audio = await this.callReka('reka-audio-20241211', {
      task: 'audio_analysis',
      input: asset.url,
      prompt: 'Extract audio features: music type, voice characteristics, sound effects'
    });

    // Temporal patterns
    videoFeatures.temporal = await this.callReka('reka-video-20241211', {
      task: 'temporal_analysis',
      input: asset.url,
      prompt: 'Analyze pacing, rhythm, and scene transitions'
    });

    return videoFeatures;
  }

  /**
   * Extract creative intelligence features
   */
  async extractCreativeFeatures(asset) {
    const creativeFeatures = {};

    // Emotional sentiment
    creativeFeatures.emotion = await this.callReka('reka-core-20241211', {
      task: 'emotional_analysis',
      input: asset.url,
      prompt: 'Analyze emotional tone, mood, and sentiment of the ad'
    });

    // Product category
    creativeFeatures.category = await this.callReka('reka-core-20241211', {
      task: 'product_classification',
      input: asset.url,
      prompt: 'Classify the product category: automotive, fashion, tech, food, etc.'
    });

    // Ad type classification
    creativeFeatures.adType = await this.callReka('reka-core-20241211', {
      task: 'ad_type_classification',
      input: asset.url,
      prompt: 'Classify ad type: brand awareness, product promotion, app install, etc.'
    });

    // CTA strength
    creativeFeatures.ctaStrength = await this.callReka('reka-core-20241211', {
      task: 'cta_analysis',
      input: asset.url,
      prompt: 'Analyze call-to-action prominence, clarity, and urgency'
    });

    return creativeFeatures;
  }

  /**
   * Call Reka API for feature extraction
   */
  async callReka(model, params) {
    // This would integrate with Reka's API
    // For now, returning mock data structure
    return {
      model: model,
      task: params.task,
      confidence: Math.random() * 0.5 + 0.5,
      results: `Mock ${params.task} results for ${params.input}`,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Helper methods
   */
  extractAdUrls(searchResults, scrapeTool) {
    // Implementation to extract ad URLs from search results
    return [];
  }

  async downloadAssets(urls, scrapeTool) {
    // Implementation to download and categorize assets
    return [];
  }

  createBatches(items, batchSize) {
    const batches = [];
    for (let i = 0; i < items.length; i += batchSize) {
      batches.push(items.slice(i, i + batchSize));
    }
    return batches;
  }

  /**
   * Export results
   */
  async exportResults() {
    const outputDir = path.join(__dirname, '..', 'output');
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    // Export features as JSON
    const featuresArray = Array.from(this.features.values());
    const featuresFile = path.join(outputDir, 'ad_features.json');
    fs.writeFileSync(featuresFile, JSON.stringify(featuresArray, null, 2));

    // Export summary statistics
    const summary = {
      total_assets: this.features.size,
      processing_time: (Date.now() - this.processingStats.startTime) / 1000,
      features_per_asset: this.calculateFeatureStats(),
      timestamp: new Date().toISOString()
    };

    const summaryFile = path.join(outputDir, 'processing_summary.json');
    fs.writeFileSync(summaryFile, JSON.stringify(summary, null, 2));

    console.log(`\n📁 Results exported to ${outputDir}`);
    console.log(`   - Features: ${featuresFile}`);
    console.log(`   - Summary: ${summaryFile}`);

    return { featuresFile, summaryFile };
  }

  calculateFeatureStats() {
    const stats = {};
    for (const [assetId, features] of this.features) {
      const featureCount = Object.keys(features.features).length;
      stats[assetId] = featureCount;
    }
    return stats;
  }

  async cleanup() {
    if (this.mcpClient) {
      try {
        await this.mcpClient.close();
      } catch (error) {
        console.log('Error closing MCP client:', error.message);
      }
    }
  }
}

// Main execution
async function main() {
  const pipeline = new AdIntelligencePipeline();
  
  try {
    await pipeline.initialize();
    await pipeline.collectAdCreatives();
    await pipeline.extractFeatures();
    await pipeline.exportResults();
    
    console.log('\n🎉 Ad Intelligence Pipeline completed successfully!');
    
  } catch (error) {
    console.error('❌ Pipeline failed:', error.message);
    process.exit(1);
  } finally {
    await pipeline.cleanup();
  }
}

// Run if this is the main module
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  main();
}

export { AdIntelligencePipeline };
