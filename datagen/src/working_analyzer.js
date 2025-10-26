/**
 * Working Asset Analyzer
 * Fixed Reka API integration for asset analysis
 */

import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CONFIG = {
  rekaApiKey: process.env.REKA_API_KEY,
  baseUrl: 'https://api.reka.ai/v1',
  model: 'reka-flash-research',
  timeout: 30000,
  outputDir: path.join(__dirname, '..', 'analysis_results')
};

class WorkingAnalyzer {
  constructor() {
    this.apiKey = CONFIG.rekaApiKey;
    this.baseUrl = CONFIG.baseUrl;
    this.model = CONFIG.model;
    this.analysisResults = [];
  }

  async initialize() {
    console.log('🧠 Initializing Working Asset Analyzer');
    
    if (!this.apiKey) {
      throw new Error('REKA_API_KEY not set. Please add your Reka API key to .env file');
    }

    // Create output directory
    if (!fs.existsSync(CONFIG.outputDir)) {
      fs.mkdirSync(CONFIG.outputDir, { recursive: true });
    }

    console.log('✅ Reka API client initialized');
    return this;
  }

  /**
   * Analyze all existing assets
   */
  async analyzeExistingAssets() {
    console.log('\n📊 ANALYZING EXISTING ASSETS');
    console.log('='.repeat(60));

    // Get existing assets
    const assets = this.getExistingAssets();
    console.log(`Found ${assets.length} assets to analyze`);
    console.log(`   - Images: ${assets.filter(a => a.type === 'image').length}`);
    console.log(`   - Videos: ${assets.filter(a => a.type === 'video').length}`);

    // Analyze each asset
    for (let i = 0; i < assets.length; i++) {
      const asset = assets[i];
      console.log(`\n🔍 Analyzing ${i + 1}/${assets.length}: ${asset.id}`);
      
      try {
        const analysis = await this.analyzeAsset(asset);
        this.analysisResults.push(analysis);
        console.log(`   ✅ Completed analysis for ${asset.id}`);
        
        // Small delay to avoid rate limiting
        await new Promise(resolve => setTimeout(resolve, 2000));
        
      } catch (error) {
        console.log(`   ❌ Failed to analyze ${asset.id}: ${error.message}`);
        this.analysisResults.push({
          asset_id: asset.id,
          type: asset.type,
          error: error.message,
          analyzed_at: new Date().toISOString()
        });
      }
    }

    // Save results
    await this.saveAnalysisResults();

    return this.analysisResults;
  }

  /**
   * Get existing assets from images and videos folders
   */
  getExistingAssets() {
    const assets = [];
    
    // Add images
    const imagesDir = path.join(__dirname, '..', 'images');
    if (fs.existsSync(imagesDir)) {
      const imageFiles = fs.readdirSync(imagesDir).filter(f => f.match(/\.(png|jpg|jpeg|gif|webp)$/i));
      for (const file of imageFiles) {
        assets.push({
          id: file.replace(/\.[^/.]+$/, ''), // Remove extension
          type: 'image',
          path: path.join(imagesDir, file),
          filename: file
        });
      }
    }

    // Add videos
    const videosDir = path.join(__dirname, '..', 'videos');
    if (fs.existsSync(videosDir)) {
      const videoFiles = fs.readdirSync(videosDir).filter(f => f.match(/\.(mp4|webm|mov|avi)$/i));
      for (const file of videoFiles) {
        assets.push({
          id: file.replace(/\.[^/.]+$/, ''), // Remove extension
          type: 'video',
          path: path.join(videosDir, file),
          filename: file
        });
      }
    }

    return assets;
  }

  /**
   * Analyze individual asset
   */
  async analyzeAsset(asset) {
    const analysis = {
      asset_id: asset.id,
      type: asset.type,
      filename: asset.filename,
      analyzed_at: new Date().toISOString(),
      features: {}
    };

    try {
      // For now, let's just do a basic text analysis
      // We'll add image analysis later once we get the basic API working
      const description = await this.getBasicDescription(asset);
      analysis.features.description = description;

      // Extract basic features
      const stats = fs.statSync(asset.path);
      analysis.features.basic = {
        file_size: stats.size,
        type: asset.type,
        filename: asset.filename,
        created: stats.birthtime,
        modified: stats.mtime
      };

      return analysis;

    } catch (error) {
      console.log(`   ⚠️  Failed to analyze ${asset.id}: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get basic description using Reka (text-only for now)
   */
  async getBasicDescription(asset) {
    try {
      const response = await fetch(`${this.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: this.model,
          messages: [
            {
              role: 'user',
              content: `Analyze this ${asset.type} file named "${asset.filename}" and provide a detailed description of what you would expect to see in an advertisement. Include objects, people, text, colors, and overall composition that would be typical for this type of ad.`
            }
          ],
          max_tokens: 300,
          temperature: 0.1
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Reka API error: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const data = await response.json();
      return data.choices[0]?.message?.content || 'No description available';

    } catch (error) {
      console.log(`   ⚠️  Reka API call failed: ${error.message}`);
      return `Analysis failed: ${error.message}`;
    }
  }

  /**
   * Save analysis results
   */
  async saveAnalysisResults() {
    const resultsFile = path.join(CONFIG.outputDir, 'working_analysis.json');
    const summaryFile = path.join(CONFIG.outputDir, 'analysis_summary.json');

    // Save detailed analysis
    fs.writeFileSync(resultsFile, JSON.stringify(this.analysisResults, null, 2));

    // Save summary
    const summary = {
      total_assets: this.analysisResults.length,
      successful_analyses: this.analysisResults.filter(r => !r.error).length,
      failed_analyses: this.analysisResults.filter(r => r.error).length,
      analysis_time: new Date().toISOString()
    };
    
    fs.writeFileSync(summaryFile, JSON.stringify(summary, null, 2));

    console.log(`\n📁 Analysis results saved to ${CONFIG.outputDir}`);
    console.log(`   - Detailed analysis: ${resultsFile}`);
    console.log(`   - Summary: ${summaryFile}`);

    return { resultsFile, summaryFile };
  }
}

// Main execution
async function main() {
  const analyzer = new WorkingAnalyzer();
  
  try {
    await analyzer.initialize();
    await analyzer.analyzeExistingAssets();
    
    console.log('\n🎉 Asset analysis completed successfully!');
    
  } catch (error) {
    console.error('❌ Analysis failed:', error.message);
    process.exit(1);
  }
}

// Run if this is the main module
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  main();
}

export { WorkingAnalyzer };
