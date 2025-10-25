/**
 * Scaled Ad Collector
 * Scales up smart collection to reach 10k images + 1k videos target
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
  outputDir: path.join(__dirname, '..', 'scaled_collected_assets'),
  batchSize: 50,
  maxRetries: 3,
  maxProcessingTime: 1800, // 30 minutes
  parallelBatches: 3
};

class ScaledCollector {
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
      totalRequests: 0,
      batchesCompleted: 0
    };
  }

  async initialize() {
    console.log('🚀 Initializing Scaled Ad Collector');
    console.log(`Target: ${CONFIG.targetImages} images + ${CONFIG.targetVideos} videos`);
    console.log(`Time limit: ${CONFIG.maxProcessingTime} seconds`);
    
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

    console.log('✅ Scaled collector initialized');
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

    // Expand search queries for scaling
    this.searchQueries = this.expandSearchQueries();

    console.log(`✅ Loaded fingerprint with ${this.searchQueries.length} search strategies`);
  }

  /**
   * Expand search queries for scaling
   */
  expandSearchQueries() {
    const expandedQueries = [...this.searchQueries];
    
    // Add more specific queries based on patterns
    const visualPatterns = this.fingerprint.visual_patterns;
    const creativePatterns = this.fingerprint.creative_patterns;
    
    // Add more object-specific queries
    for (const object of visualPatterns.objects.slice(0, 10)) {
      expandedQueries.push({
        query: `${object} ads examples`,
        type: 'visual_object',
        priority: 'high',
        expected_yield: 'high'
      });
    }
    
    // Add more industry-specific queries
    for (const industry of creativePatterns.industries.slice(0, 5)) {
      expandedQueries.push({
        query: `${industry} advertising examples`,
        type: 'industry',
        priority: 'high',
        expected_yield: 'medium'
      });
    }
    
    // Add more platform-specific queries
    const platforms = ['instagram', 'facebook', 'youtube', 'tiktok', 'linkedin', 'pinterest', 'snapchat'];
    for (const platform of platforms) {
      expandedQueries.push({
        query: `${platform} ads examples`,
        type: 'platform',
        priority: 'high',
        expected_yield: 'high'
      });
      expandedQueries.push({
        query: `${platform} advertising examples`,
        type: 'platform',
        priority: 'medium',
        expected_yield: 'medium'
      });
    }
    
    // Add style-specific queries
    for (const style of creativePatterns.styles.slice(0, 5)) {
      expandedQueries.push({
        query: `${style} ads examples`,
        type: 'style',
        priority: 'medium',
        expected_yield: 'medium'
      });
    }
    
    // Add emotion-specific queries
    for (const emotion of creativePatterns.emotions.slice(0, 5)) {
      expandedQueries.push({
        query: `${emotion} ads examples`,
        type: 'emotion',
        priority: 'medium',
        expected_yield: 'low'
      });
    }
    
    return expandedQueries;
  }

  /**
   * Main scaled collection process
   */
  async collectAtScale() {
    console.log('\n🎯 SCALED AD COLLECTION');
    console.log('='.repeat(60));
    console.log('Scaling up to reach target collection...');

    const tools = await this.mcpClient.getTools();
    const searchTool = tools.find(t => t.name === 'search_engine');
    const scrapeTool = tools.find(t => t.name === 'scrape_as_markdown');

    if (!searchTool || !scrapeTool) {
      throw new Error('Required Bright Data tools not available');
    }

    this.searchTool = searchTool;
    this.scrapeTool = scrapeTool;

    // Execute collection in batches
    const batches = this.createBatches(this.searchQueries, CONFIG.batchSize);
    console.log(`\n📦 Executing ${batches.length} batches of ${CONFIG.batchSize} queries each`);

    for (let i = 0; i < batches.length; i++) {
      if (this.isCollectionComplete() || this.isTimeLimitReached()) {
        console.log('🎯 Collection targets reached or time limit reached!');
        break;
      }

      const batch = batches[i];
      console.log(`\n📦 Processing batch ${i + 1}/${batches.length} (${batch.length} queries)`);
      
      try {
        await this.executeBatch(batch);
        this.collectionStats.batchesCompleted++;
        
        console.log(`   ✅ Batch ${i + 1} completed`);
        console.log(`   📊 Progress: ${this.collectionStats.imagesCollected}/${CONFIG.targetImages} images, ${this.collectionStats.videosCollected}/${CONFIG.targetVideos} videos`);
        
        // Small delay between batches
        await new Promise(resolve => setTimeout(resolve, 5000));
        
      } catch (error) {
        console.log(`   ❌ Batch ${i + 1} failed: ${error.message}`);
      }
    }

    // Save collected assets
    await this.saveCollectedAssets();

    return this.collectedAssets;
  }

  /**
   * Create batches from queries
   */
  createBatches(queries, batchSize) {
    const batches = [];
    for (let i = 0; i < queries.length; i += batchSize) {
      batches.push(queries.slice(i, i + batchSize));
    }
    return batches;
  }

  /**
   * Execute a batch of queries
   */
  async executeBatch(queries) {
    for (const query of queries) {
      if (this.isCollectionComplete() || this.isTimeLimitReached()) {
        break;
      }

      try {
        console.log(`   🔍 Searching: "${query.query}"`);
        
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
          
          if (assets.length > 0) {
            console.log(`   ✅ Collected ${assets.length} assets`);
          }
        }

        this.collectionStats.successfulQueries++;

        // Small delay between queries
        await new Promise(resolve => setTimeout(resolve, 1000));

      } catch (error) {
        console.log(`   ⚠️  Failed to process "${query.query}": ${error.message}`);
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
    
    for (const url of urls.slice(0, 3)) { // Limit to top 3 URLs per query for speed
      if (this.isCollectionComplete() || this.isTimeLimitReached()) break;

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
    
    const patterns = [
      /https?:\/\/[^\s"']+\.(jpg|jpeg|png|gif|webp)(\?[^\s"']*)?/gi,
      /https?:\/\/[^\s"']+\.(jpg|jpeg|png|gif|webp)/gi
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
    
    const patterns = [
      /https?:\/\/[^\s"']+\.(mp4|webm|mov|avi)(\?[^\s"']*)?/gi,
      /https?:\/\/[^\s"']+\.(mp4|webm|mov|avi)/gi
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
      'facebook.com/ads', 'instagram.com', 'youtube.com', 'tiktok.com',
      'linkedin.com/ads', 'google.com/ads', 'business.instagram.com',
      'ads.google.com', 'pinterest.com', 'snapchat.com'
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
   * Check if time limit is reached
   */
  isTimeLimitReached() {
    const elapsed = (Date.now() - this.collectionStats.startTime) / 1000;
    return elapsed >= CONFIG.maxProcessingTime;
  }

  /**
   * Save collected assets
   */
  async saveCollectedAssets() {
    const assetsFile = path.join(CONFIG.outputDir, 'scaled_collected_assets.json');
    const statsFile = path.join(CONFIG.outputDir, 'collection_stats.json');

    // Save assets
    fs.writeFileSync(assetsFile, JSON.stringify(this.collectedAssets, null, 2));
    
    // Update stats
    this.collectionStats.totalAssets = this.collectedAssets.length;
    this.collectionStats.processingTime = (Date.now() - this.collectionStats.startTime) / 1000;
    this.collectionStats.successRate = this.collectionStats.successfulQueries / this.collectionStats.queriesExecuted;

    // Save statistics
    fs.writeFileSync(statsFile, JSON.stringify(this.collectionStats, null, 2));

    console.log(`\n📁 Scaled collection results saved to ${CONFIG.outputDir}`);
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
      batchesCompleted: this.collectionStats.batchesCompleted,
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
  const collector = new ScaledCollector();
  
  try {
    await collector.initialize();
    await collector.collectAtScale();
    
    const summary = collector.getCollectionSummary();
    console.log('\n🎉 Scaled collection completed!');
    console.log('📊 Summary:', summary);
    
  } catch (error) {
    console.error('❌ Scaled collection failed:', error.message);
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

export { ScaledCollector };
