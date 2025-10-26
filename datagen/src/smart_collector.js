/**
 * Smart Ad Collector
 * Uses feature fingerprint to find and collect similar ads
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
  fingerprintDir: path.join(__dirname, '..', 'feature_fingerprints'),
  targetImages: 10000,
  targetVideos: 1000,
  outputDir: path.join(__dirname, '..', 'smart_collected_assets'),
  batchSize: 20,
  maxRetries: 3,
  maxProcessingTime: 300 // 5 minutes
};

class SmartCollector {
  constructor() {
    this.mcpClient = null;
    this.fingerprint = null;
    this.searchQueries = [];
    this.collectedAssets = [];
    this.collectionStats = {
      imagesCollected: 0,
      videosCollected: 0,
      queriesExecuted: 0,
      successfulQueries: 0,
      startTime: Date.now(),
      totalRequests: 0
    };
  }

  async initialize() {
    console.log('🧠 Initializing Smart Ad Collector');
    console.log(`Target: ${CONFIG.targetImages} images + ${CONFIG.targetVideos} videos`);
    
    // Load fingerprint and search queries
    await this.loadFingerprint();
    
    // Initialize Bright Data MCP client
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

    console.log('✅ Smart collector initialized');
    return this;
  }

  /**
   * Load feature fingerprint and search queries
   */
  async loadFingerprint() {
    const fingerprintFile = path.join(CONFIG.fingerprintDir, 'feature_fingerprint.json');
    const searchQueriesFile = path.join(CONFIG.fingerprintDir, 'search_queries.json');

    if (!fs.existsSync(fingerprintFile)) {
      throw new Error(`Fingerprint file not found: ${fingerprintFile}`);
    }

    this.fingerprint = JSON.parse(fs.readFileSync(fingerprintFile, 'utf8'));
    this.searchQueries = JSON.parse(fs.readFileSync(searchQueriesFile, 'utf8'));

    console.log(`✅ Loaded fingerprint with ${this.searchQueries.length} search strategies`);
    console.log(`   - Visual patterns: ${Object.keys(this.fingerprint.visual_patterns).length}`);
    console.log(`   - Creative patterns: ${Object.keys(this.fingerprint.creative_patterns).length}`);
  }

  /**
   * Main smart collection process
   */
  async collectSimilarAds() {
    console.log('\n🎯 SMART AD COLLECTION');
    console.log('='.repeat(60));
    console.log('Using feature fingerprint to find similar ads...');

    const tools = await this.mcpClient.getTools();
    const searchTool = tools.find(t => t.name === 'search_engine');
    const scrapeTool = tools.find(t => t.name === 'scrape_as_markdown');

    if (!searchTool || !scrapeTool) {
      throw new Error('Required Bright Data tools not available');
    }

    this.searchTool = searchTool;
    this.scrapeTool = scrapeTool;

    // Execute search strategies in priority order
    const highPriorityQueries = this.searchQueries.filter(q => q.priority === 'high');
    const mediumPriorityQueries = this.searchQueries.filter(q => q.priority === 'medium');
    const lowPriorityQueries = this.searchQueries.filter(q => q.priority === 'low');

    console.log(`\n🔍 Executing ${highPriorityQueries.length} high-priority searches`);
    await this.executeSearchBatch(highPriorityQueries);

    if (!this.isCollectionComplete()) {
      console.log(`\n🔍 Executing ${mediumPriorityQueries.length} medium-priority searches`);
      await this.executeSearchBatch(mediumPriorityQueries);
    }

    if (!this.isCollectionComplete()) {
      console.log(`\n🔍 Executing ${lowPriorityQueries.length} low-priority searches`);
      await this.executeSearchBatch(lowPriorityQueries);
    }

    // Save collected assets
    await this.saveCollectedAssets();

    return this.collectedAssets;
  }

  /**
   * Execute a batch of search queries
   */
  async executeSearchBatch(queries) {
    for (const query of queries) {
      if (this.isCollectionComplete()) {
        console.log('🎯 Collection targets reached!');
        break;
      }

      try {
        console.log(`\n🔍 Searching: "${query.query}"`);
        console.log(`   Type: ${query.type}, Priority: ${query.priority}`);
        
        const searchResults = await this.searchTool.invoke({
          query: query.query,
          engine: 'google'
        });

        this.collectionStats.totalRequests++;
        this.collectionStats.queriesExecuted++;

        // Extract URLs from search results
        const urls = await this.extractUrlsFromSearch(searchResults);
        console.log(`   📋 Found ${urls.length} URLs`);

        if (urls.length > 0) {
          // Process URLs to find ad assets
          const assets = await this.processUrlsForAssets(urls, query);
          this.collectedAssets.push(...assets);
          
          console.log(`   ✅ Collected ${assets.length} assets from this query`);
          console.log(`   📊 Progress: ${this.collectionStats.imagesCollected}/${CONFIG.targetImages} images, ${this.collectionStats.videosCollected}/${CONFIG.targetVideos} videos`);
        }

        this.collectionStats.successfulQueries++;

        // Small delay between queries
        await new Promise(resolve => setTimeout(resolve, 2000));

      } catch (error) {
        console.log(`   ❌ Failed to process "${query.query}": ${error.message}`);
      }
    }
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
   * Process URLs to find ad assets
   */
  async processUrlsForAssets(urls, query) {
    const assets = [];
    
    for (const url of urls.slice(0, 5)) { // Limit to top 5 URLs per query
      if (this.isCollectionComplete()) break;

      try {
        // Scrape the page
        const pageContent = await this.scrapeTool.invoke({ url });
        
        // Extract assets from page content
        const extractedAssets = await this.extractAssetsFromPage(pageContent, url, query);
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
  async extractAssetsFromPage(pageContent, sourceUrl, query) {
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
          search_query: query.query,
          query_type: query.type,
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
          search_query: query.query,
          query_type: query.type,
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
      'ads.google.com',
      'pinterest.com',
      'snapchat.com'
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
   * Save collected assets
   */
  async saveCollectedAssets() {
    const assetsFile = path.join(CONFIG.outputDir, 'smart_collected_assets.json');
    const statsFile = path.join(CONFIG.outputDir, 'collection_stats.json');

    // Save assets
    fs.writeFileSync(assetsFile, JSON.stringify(this.collectedAssets, null, 2));
    
    // Update stats
    this.collectionStats.totalAssets = this.collectedAssets.length;
    this.collectionStats.processingTime = (Date.now() - this.collectionStats.startTime) / 1000;
    this.collectionStats.successRate = this.collectionStats.successfulQueries / this.collectionStats.queriesExecuted;

    // Save statistics
    fs.writeFileSync(statsFile, JSON.stringify(this.collectionStats, null, 2));

    console.log(`\n📁 Smart collection results saved to ${CONFIG.outputDir}`);
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
      queriesExecuted: this.collectionStats.queriesExecuted,
      successRate: this.collectionStats.successRate,
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
  const collector = new SmartCollector();
  
  try {
    await collector.initialize();
    await collector.collectSimilarAds();
    
    const summary = collector.getCollectionSummary();
    console.log('\n🎉 Smart collection completed!');
    console.log('📊 Summary:', summary);
    
  } catch (error) {
    console.error('❌ Smart collection failed:', error.message);
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

export { SmartCollector };
