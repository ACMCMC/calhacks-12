/**
 * Bright Data Ad Collection Strategy
 * Multi-platform ad creative collection using Bright Data MCP tools
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
  batchSize: 20
};

class BrightDataAdCollector {
  constructor() {
    this.mcpClient = null;
    this.collectedAssets = [];
    this.collectionStats = {
      imagesCollected: 0,
      videosCollected: 0,
      platformsScraped: 0,
      startTime: Date.now()
    };
  }

  async initialize() {
    console.log('🚀 Initializing Bright Data Ad Collector');
    
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
   * Main collection strategy across multiple platforms
   */
  async collectAdCreatives() {
    console.log('\n📊 COLLECTING AD CREATIVES');
    console.log('='.repeat(60));

    const collectionStrategies = [
      {
        name: 'Facebook Ads Library',
        method: this.collectFromFacebookAds.bind(this),
        priority: 'high'
      },
      {
        name: 'Google Ads Transparency',
        method: this.collectFromGoogleAds.bind(this),
        priority: 'high'
      },
      {
        name: 'TikTok Ads',
        method: this.collectFromTikTok.bind(this),
        priority: 'medium'
      },
      {
        name: 'Instagram Ads',
        method: this.collectFromInstagram.bind(this),
        priority: 'medium'
      },
      {
        name: 'YouTube Ads',
        method: this.collectFromYouTube.bind(this),
        priority: 'medium'
      },
      {
        name: 'LinkedIn Ads',
        method: this.collectFromLinkedIn.bind(this),
        priority: 'low'
      }
    ];

    const tools = await this.mcpClient.getTools();
    const searchTool = tools.find(t => t.name === 'search_engine');
    const scrapeTool = tools.find(t => t.name === 'scrape_as_markdown');

    if (!searchTool || !scrapeTool) {
      throw new Error('Required Bright Data tools not available');
    }

    this.searchTool = searchTool;
    this.scrapeTool = scrapeTool;

    // Execute collection strategies
    for (const strategy of collectionStrategies) {
      if (this.isCollectionComplete()) {
        console.log('🎯 Collection targets reached!');
        break;
      }

      try {
        console.log(`\n🔍 Collecting from ${strategy.name}`);
        const assets = await strategy.method();
        this.collectedAssets.push(...assets);
        this.collectionStats.platformsScraped++;
        
        console.log(`   ✅ Collected ${assets.length} assets from ${strategy.name}`);
        
      } catch (error) {
        console.log(`   ❌ Failed to collect from ${strategy.name}: ${error.message}`);
      }
    }

    this.updateStats();
    await this.saveCollectedAssets();
    
    return this.collectedAssets;
  }

  /**
   * Facebook Ads Library Collection
   */
  async collectFromFacebookAds() {
    const assets = [];
    const searchQueries = [
      'automotive ads facebook',
      'fashion ads facebook',
      'tech ads facebook',
      'food ads facebook',
      'travel ads facebook',
      'finance ads facebook',
      'health ads facebook',
      'entertainment ads facebook'
    ];

    for (const query of searchQueries) {
      if (this.isCollectionComplete()) break;

      try {
        // Search for Facebook ads
        const searchResults = await this.searchTool.invoke({
          query: `${query} ads library`,
          engine: 'google'
        });

        // Extract ad URLs from search results
        const adUrls = await this.extractAdUrlsFromSearch(searchResults);
        
        // Download and process assets
        const batchAssets = await this.processAdUrls(adUrls, 'facebook');
        assets.push(...batchAssets);

        console.log(`   📱 Found ${batchAssets.length} Facebook ads for "${query}"`);

      } catch (error) {
        console.log(`   ⚠️  Failed to collect Facebook ads for "${query}": ${error.message}`);
      }
    }

    return assets;
  }

  /**
   * Google Ads Transparency Collection
   */
  async collectFromGoogleAds() {
    const assets = [];
    const searchQueries = [
      'google ads transparency center',
      'political ads google',
      'commercial ads google',
      'product ads google',
      'brand ads google'
    ];

    for (const query of searchQueries) {
      if (this.isCollectionComplete()) break;

      try {
        const searchResults = await this.searchTool.invoke({
          query: query,
          engine: 'google'
        });

        const adUrls = await this.extractAdUrlsFromSearch(searchResults);
        const batchAssets = await this.processAdUrls(adUrls, 'google');
        assets.push(...batchAssets);

        console.log(`   🔍 Found ${batchAssets.length} Google ads for "${query}"`);

      } catch (error) {
        console.log(`   ⚠️  Failed to collect Google ads for "${query}": ${error.message}`);
      }
    }

    return assets;
  }

  /**
   * TikTok Ads Collection
   */
  async collectFromTikTok() {
    const assets = [];
    const searchQueries = [
      'tiktok ads examples',
      'tiktok video ads',
      'tiktok brand ads',
      'tiktok product ads',
      'tiktok app ads'
    ];

    for (const query of searchQueries) {
      if (this.isCollectionComplete()) break;

      try {
        const searchResults = await this.searchTool.invoke({
          query: query,
          engine: 'google'
        });

        const adUrls = await this.extractAdUrlsFromSearch(searchResults);
        const batchAssets = await this.processAdUrls(adUrls, 'tiktok');
        assets.push(...batchAssets);

        console.log(`   🎵 Found ${batchAssets.length} TikTok ads for "${query}"`);

      } catch (error) {
        console.log(`   ⚠️  Failed to collect TikTok ads for "${query}": ${error.message}`);
      }
    }

    return assets;
  }

  /**
   * Instagram Ads Collection
   */
  async collectFromInstagram() {
    const assets = [];
    const searchQueries = [
      'instagram ads examples',
      'instagram sponsored posts',
      'instagram brand ads',
      'instagram product ads',
      'instagram story ads'
    ];

    for (const query of searchQueries) {
      if (this.isCollectionComplete()) break;

      try {
        const searchResults = await this.searchTool.invoke({
          query: query,
          engine: 'google'
        });

        const adUrls = await this.extractAdUrlsFromSearch(searchResults);
        const batchAssets = await this.processAdUrls(adUrls, 'instagram');
        assets.push(...batchAssets);

        console.log(`   📸 Found ${batchAssets.length} Instagram ads for "${query}"`);

      } catch (error) {
        console.log(`   ⚠️  Failed to collect Instagram ads for "${query}": ${error.message}`);
      }
    }

    return assets;
  }

  /**
   * YouTube Ads Collection
   */
  async collectFromYouTube() {
    const assets = [];
    const searchQueries = [
      'youtube ads examples',
      'youtube video ads',
      'youtube brand ads',
      'youtube product ads',
      'youtube app ads'
    ];

    for (const query of searchQueries) {
      if (this.isCollectionComplete()) break;

      try {
        const searchResults = await this.searchTool.invoke({
          query: query,
          engine: 'google'
        });

        const adUrls = await this.extractAdUrlsFromSearch(searchResults);
        const batchAssets = await this.processAdUrls(adUrls, 'youtube');
        assets.push(...batchAssets);

        console.log(`   📺 Found ${batchAssets.length} YouTube ads for "${query}"`);

      } catch (error) {
        console.log(`   ⚠️  Failed to collect YouTube ads for "${query}": ${error.message}`);
      }
    }

    return assets;
  }

  /**
   * LinkedIn Ads Collection
   */
  async collectFromLinkedIn() {
    const assets = [];
    const searchQueries = [
      'linkedin ads examples',
      'linkedin sponsored content',
      'linkedin b2b ads',
      'linkedin professional ads'
    ];

    for (const query of searchQueries) {
      if (this.isCollectionComplete()) break;

      try {
        const searchResults = await this.searchTool.invoke({
          query: query,
          engine: 'google'
        });

        const adUrls = await this.extractAdUrlsFromSearch(searchResults);
        const batchAssets = await this.processAdUrls(adUrls, 'linkedin');
        assets.push(...batchAssets);

        console.log(`   💼 Found ${batchAssets.length} LinkedIn ads for "${query}"`);

      } catch (error) {
        console.log(`   ⚠️  Failed to collect LinkedIn ads for "${query}": ${error.message}`);
      }
    }

    return assets;
  }

  /**
   * Extract ad URLs from search results
   */
  async extractAdUrlsFromSearch(searchResults) {
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
   * Process ad URLs and download assets
   */
  async processAdUrls(urls, platform) {
    const assets = [];
    
    for (const url of urls.slice(0, CONFIG.batchSize)) {
      if (this.isCollectionComplete()) break;

      try {
        // Scrape the ad page to extract creative assets
        const pageContent = await this.scrapeTool.invoke({ url });
        
        // Extract image and video URLs from page content
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
      if (this.isValidAssetUrl(imageUrl)) {
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
      if (this.isValidAssetUrl(videoUrl)) {
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
    const imageRegex = /https?:\/\/[^\s]+\.(jpg|jpeg|png|gif|webp)(\?[^\s]*)?/gi;
    const matches = content.match(imageRegex);
    
    if (matches) {
      imageUrls.push(...matches);
    }

    return [...new Set(imageUrls)]; // Remove duplicates
  }

  /**
   * Extract video URLs from content
   */
  extractVideoUrls(content) {
    const videoUrls = [];
    const videoRegex = /https?:\/\/[^\s]+\.(mp4|webm|mov|avi)(\?[^\s]*)?/gi;
    const matches = content.match(videoRegex);
    
    if (matches) {
      videoUrls.push(...matches);
    }

    return [...new Set(videoUrls)]; // Remove duplicates
  }

  /**
   * Validate ad URL
   */
  isValidAdUrl(url) {
    const adPlatforms = [
      'facebook.com/ads',
      'google.com/ads',
      'tiktok.com/business',
      'instagram.com',
      'youtube.com',
      'linkedin.com/ads'
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
   * Update collection statistics
   */
  updateStats() {
    this.collectionStats.totalAssets = this.collectedAssets.length;
    this.collectionStats.processingTime = (Date.now() - this.collectionStats.startTime) / 1000;
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
  const collector = new BrightDataAdCollector();
  
  try {
    await collector.initialize();
    await collector.collectAdCreatives();
    
    const summary = collector.getCollectionSummary();
    console.log('\n🎉 Collection completed successfully!');
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

export { BrightDataAdCollector };
