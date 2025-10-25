/**
 * Focused Dataset Collection for Ad Intelligence Challenge
 * Optimized for collecting 10k images + 1k videos
 */

import 'dotenv/config';
import { MultiServerMCPClient } from '@langchain/mcp-adapters';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CONFIG = {
  brightdataApiKey: process.env.BRIGHTDATA_API_KEY,
  targetImages: 10000,
  targetVideos: 1000,
  outputDir: path.join(__dirname, '..', 'collected_assets'),
  batchSize: 50,
  maxRetries: 3
};

class DatasetCollector {
  constructor() {
    this.mcpClient = null;
    this.collectedAssets = [];
    this.collectionStats = {
      imagesCollected: 0,
      videosCollected: 0,
      platformsScraped: 0,
      startTime: Date.now(),
      totalRequests: 0,
      successfulRequests: 0
    };
  }

  async initialize() {
    console.log('🚀 Initializing Dataset Collector');
    console.log(`Target: ${CONFIG.targetImages} images + ${CONFIG.targetVideos} videos`);
    
    this.mcpClient = new MultiServerMCPClient({
      bright_data: {
        url: `https://mcp.brightdata.com/sse?token=${CONFIG.brightdataApiKey}&pro=1`,
        transport: 'sse',
      },
    });

    // Create output directory
    if (!fs.existsSync(CONFIG.outputDir)) {
      fs.mkdirSync(CONFIG.outputDir, { recursive: true });
    }

    console.log('✅ Bright Data MCP client initialized');
    return this;
  }

  /**
   * Main collection strategy - focused on high-yield sources
   */
  async collectDatasets() {
    console.log('\n📊 FOCUSED DATASET COLLECTION');
    console.log('='.repeat(60));

    const tools = await this.mcpClient.getTools();
    const searchTool = tools.find(t => t.name === 'search_engine');
    const scrapeTool = tools.find(t => t.name === 'scrape_as_markdown');

    if (!searchTool || !scrapeTool) {
      throw new Error('Required Bright Data tools not available');
    }

    this.searchTool = searchTool;
    this.scrapeTool = scrapeTool;

    // High-yield collection strategies
    const strategies = [
      {
        name: 'Instagram Business Examples',
        queries: [
          'instagram business examples ads',
          'instagram sponsored content examples',
          'instagram brand ads examples',
          'instagram product ads examples',
          'instagram marketing examples'
        ],
        priority: 'high'
      },
      {
        name: 'Facebook Ads Library',
        queries: [
          'facebook ads library automotive',
          'facebook ads library fashion',
          'facebook ads library technology',
          'facebook ads library food',
          'facebook ads library travel'
        ],
        priority: 'high'
      },
      {
        name: 'YouTube Video Ads',
        queries: [
          'youtube video ads examples',
          'youtube commercial ads',
          'youtube brand ads',
          'youtube product ads',
          'youtube app ads'
        ],
        priority: 'high'
      },
      {
        name: 'TikTok Ads',
        queries: [
          'tiktok ads examples',
          'tiktok video ads',
          'tiktok brand ads',
          'tiktok product ads'
        ],
        priority: 'medium'
      },
      {
        name: 'Google Ads Examples',
        queries: [
          'google ads examples',
          'google display ads examples',
          'google video ads examples',
          'google shopping ads examples'
        ],
        priority: 'medium'
      }
    ];

    // Execute collection strategies
    for (const strategy of strategies) {
      if (this.isCollectionComplete()) {
        console.log('🎯 Collection targets reached!');
        break;
      }

      try {
        console.log(`\n🔍 Collecting from ${strategy.name}`);
        const assets = await this.collectFromStrategy(strategy);
        this.collectedAssets.push(...assets);
        this.collectionStats.platformsScraped++;
        
        console.log(`   ✅ Collected ${assets.length} assets from ${strategy.name}`);
        console.log(`   📊 Progress: ${this.collectionStats.imagesCollected}/${CONFIG.targetImages} images, ${this.collectionStats.videosCollected}/${CONFIG.targetVideos} videos`);
        
      } catch (error) {
        console.log(`   ❌ Failed to collect from ${strategy.name}: ${error.message}`);
      }
    }

    this.updateStats();
    await this.saveCollectedAssets();
    
    return this.collectedAssets;
  }

  /**
   * Collect from a specific strategy
   */
  async collectFromStrategy(strategy) {
    const assets = [];
    
    for (const query of strategy.queries) {
      if (this.isCollectionComplete()) break;

      try {
        console.log(`   🔍 Searching: "${query}"`);
        
        // Search for ad examples
        const searchResults = await this.searchTool.invoke({
          query: query,
          engine: 'google'
        });

        this.collectionStats.totalRequests++;

        // Extract URLs from search results
        const urls = await this.extractUrlsFromSearch(searchResults);
        console.log(`   📋 Found ${urls.length} URLs`);

        // Process URLs in batches
        const batches = this.createBatches(urls, CONFIG.batchSize);
        for (const batch of batches) {
          if (this.isCollectionComplete()) break;

          const batchAssets = await this.processBatch(batch, strategy.name);
          assets.push(...batchAssets);
          
          // Small delay between batches
          await new Promise(resolve => setTimeout(resolve, 1000));
        }

        this.collectionStats.successfulRequests++;

      } catch (error) {
        console.log(`   ⚠️  Failed to process "${query}": ${error.message}`);
      }
    }

    return assets;
  }

  /**
   * Extract URLs from search results
   */
  async extractUrlsFromSearch(searchResults) {
    const urls = [];
    
    try {
      const data = typeof searchResults === 'string' ? JSON.parse(searchResults) : searchResults;
      
      if (data.organic && Array.isArray(data.organic)) {
        for (const result of data.organic) {
          if (result.link && this.isValidAdUrl(result.link)) {
            urls.push(result.link);
          }
        }
      }
    } catch (error) {
      console.log(`   ⚠️  Failed to parse search results: ${error.message}`);
    }

    return urls;
  }

  /**
   * Process a batch of URLs
   */
  async processBatch(urls, platform) {
    const assets = [];
    
    for (const url of urls) {
      if (this.isCollectionComplete()) break;

      try {
        // Scrape the page
        const pageContent = await this.scrapeTool.invoke({ url });
        
        // Extract assets from page content
        const extractedAssets = await this.extractAssetsFromPage(pageContent, url, platform);
        assets.push(...extractedAssets);

      } catch (error) {
        console.log(`   ⚠️  Failed to process ${url}: ${error.message}`);
      }
    }

    return assets;
  }

  /**
   * Extract assets from scraped page content
   */
  async extractAssetsFromPage(pageContent, sourceUrl, platform) {
    const assets = [];
    const content = typeof pageContent === 'string' ? pageContent : JSON.stringify(pageContent);
    
    // Extract image URLs
    const imageUrls = this.extractImageUrls(content);
    for (const imageUrl of imageUrls) {
      if (this.isValidAssetUrl(imageUrl) && this.collectionStats.imagesCollected < CONFIG.targetImages) {
        assets.push({
          id: this.generateAssetId('image'),
          type: 'image',
          url: imageUrl,
          source_url: sourceUrl,
          platform: platform,
          collected_at: new Date().toISOString()
        });
        this.collectionStats.imagesCollected++;
      }
    }

    // Extract video URLs
    const videoUrls = this.extractVideoUrls(content);
    for (const videoUrl of videoUrls) {
      if (this.isValidAssetUrl(videoUrl) && this.collectionStats.videosCollected < CONFIG.targetVideos) {
        assets.push({
          id: this.generateAssetId('video'),
          type: 'video',
          url: videoUrl,
          source_url: sourceUrl,
          platform: platform,
          collected_at: new Date().toISOString()
        });
        this.collectionStats.videosCollected++;
      }
    }

    return assets;
  }

  /**
   * Extract image URLs from content
   */
  extractImageUrls(content) {
    const imageUrls = [];
    
    // Multiple patterns for image URLs
    const patterns = [
      /https?:\/\/[^\s"']+\.(jpg|jpeg|png|gif|webp)(\?[^\s"']*)?/gi,
      /https?:\/\/[^\s"']+\.(jpg|jpeg|png|gif|webp)/gi,
      /"([^"]*\.(jpg|jpeg|png|gif|webp)[^"]*)"/gi,
      /'([^']*\.(jpg|jpeg|png|gif|webp)[^']*)'/gi
    ];

    for (const pattern of patterns) {
      const matches = content.match(pattern);
      if (matches) {
        imageUrls.push(...matches);
      }
    }

    return [...new Set(imageUrls)]; // Remove duplicates
  }

  /**
   * Extract video URLs from content
   */
  extractVideoUrls(content) {
    const videoUrls = [];
    
    // Multiple patterns for video URLs
    const patterns = [
      /https?:\/\/[^\s"']+\.(mp4|webm|mov|avi)(\?[^\s"']*)?/gi,
      /https?:\/\/[^\s"']+\.(mp4|webm|mov|avi)/gi,
      /"([^"]*\.(mp4|webm|mov|avi)[^"]*)"/gi,
      /'([^']*\.(mp4|webm|mov|avi)[^']*)'/gi
    ];

    for (const pattern of patterns) {
      const matches = content.match(pattern);
      if (matches) {
        videoUrls.push(...matches);
      }
    }

    return [...new Set(videoUrls)]; // Remove duplicates
  }

  /**
   * Validate ad URL
   */
  isValidAdUrl(url) {
    const adPlatforms = [
      'facebook.com/ads',
      'instagram.com',
      'youtube.com',
      'tiktok.com',
      'linkedin.com/ads',
      'google.com/ads',
      'business.instagram.com',
      'ads.google.com'
    ];
    
    return adPlatforms.some(platform => url.includes(platform));
  }

  /**
   * Validate asset URL
   */
  isValidAssetUrl(url) {
    const validExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.mp4', '.webm', '.mov', '.avi'];
    return validExtensions.some(ext => url.toLowerCase().includes(ext));
  }

  /**
   * Generate unique asset ID
   */
  generateAssetId(type) {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substr(2, 5);
    return `${type === 'image' ? 'i' : 'v'}${timestamp}${random}`;
  }

  /**
   * Check if collection is complete
   */
  isCollectionComplete() {
    return this.collectionStats.imagesCollected >= CONFIG.targetImages && 
           this.collectionStats.videosCollected >= CONFIG.targetVideos;
  }

  /**
   * Create batches
   */
  createBatches(items, batchSize) {
    const batches = [];
    for (let i = 0; i < items.length; i += batchSize) {
      batches.push(items.slice(i, i + batchSize));
    }
    return batches;
  }

  /**
   * Update collection statistics
   */
  updateStats() {
    this.collectionStats.totalAssets = this.collectedAssets.length;
    this.collectionStats.processingTime = (Date.now() - this.collectionStats.startTime) / 1000;
    this.collectionStats.successRate = this.collectionStats.successfulRequests / this.collectionStats.totalRequests;
  }

  /**
   * Save collected assets to files
   */
  async saveCollectedAssets() {
    const assetsFile = path.join(CONFIG.outputDir, 'collected_assets.json');
    const statsFile = path.join(CONFIG.outputDir, 'collection_stats.json');

    // Save assets
    fs.writeFileSync(assetsFile, JSON.stringify(this.collectedAssets, null, 2));
    
    // Save statistics
    fs.writeFileSync(statsFile, JSON.stringify(this.collectionStats, null, 2));

    console.log(`\n📁 Assets saved to ${CONFIG.outputDir}`);
    console.log(`   - Assets: ${assetsFile}`);
    console.log(`   - Stats: ${statsFile}`);

    return { assetsFile, statsFile };
  }

  /**
   * Get collection summary
   */
  getCollectionSummary() {
    return {
      totalAssets: this.collectedAssets.length,
      images: this.collectionStats.imagesCollected,
      videos: this.collectionStats.videosCollected,
      platforms: this.collectionStats.platformsScraped,
      processingTime: this.collectionStats.processingTime,
      successRate: this.collectionStats.successRate,
      targets: {
        images: CONFIG.targetImages,
        videos: CONFIG.targetVideos
      }
    };
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
  const collector = new DatasetCollector();
  
  try {
    await collector.initialize();
    await collector.collectDatasets();
    
    const summary = collector.getCollectionSummary();
    console.log('\n🎉 Dataset collection completed!');
    console.log('📊 Summary:', summary);
    
  } catch (error) {
    console.error('❌ Collection failed:', error.message);
    process.exit(1);
  } finally {
    await collector.cleanup();
  }
}

// Run if this is the main module
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  main();
}

export { DatasetCollector };
