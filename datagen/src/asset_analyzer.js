/**
 * Asset Feature Analyzer
 * Analyzes existing assets (images/videos) to create feature fingerprints
 * Uses Reka AI for comprehensive feature extraction
 */

import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CONFIG = {
  rekaApiKey: process.env.REKA_API_KEY,
  baseUrl: 'https://api.reka.ai',
  models: {
    core: 'reka-core-20241211',
    video: 'reka-video-20241211', 
    audio: 'reka-audio-20241211'
  },
  timeout: 30000,
  maxRetries: 3,
  outputDir: path.join(__dirname, '..', 'analysis_results')
};

class AssetAnalyzer {
  constructor() {
    this.apiKey = CONFIG.rekaApiKey;
    this.baseUrl = CONFIG.baseUrl;
    this.models = CONFIG.models;
    this.analysisResults = [];
    this.featureFingerprint = null;
  }

  async initialize() {
    console.log('🧠 Initializing Asset Feature Analyzer');
    
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
        await new Promise(resolve => setTimeout(resolve, 1000));
        
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

    // Create feature fingerprint
    await this.createFeatureFingerprint();
    
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
          url: `file://${path.join(imagesDir, file)}`
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
          url: `file://${path.join(videosDir, file)}`
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
      path: asset.path,
      analyzed_at: new Date().toISOString(),
      features: {}
    };

    try {
      // Visual features (for both images and videos)
      analysis.features.visual = await this.extractVisualFeatures(asset);
      
      // Video-specific features
      if (asset.type === 'video') {
        analysis.features.video = await this.extractVideoFeatures(asset);
      }

      // Creative intelligence features
      analysis.features.creative = await this.extractCreativeFeatures(asset);

      return analysis;

    } catch (error) {
      console.log(`   ⚠️  Failed to extract features for ${asset.id}: ${error.message}`);
      throw error;
    }
  }

  /**
   * Extract visual features using Reka
   */
  async extractVisualFeatures(asset) {
    const visualFeatures = {};

    // Object Detection & Recognition
    visualFeatures.objects = await this.callReka(this.models.core, {
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Analyze this advertisement and identify all objects, products, logos, people, and brand elements. Provide detailed descriptions and confidence scores.'
            },
            {
              type: 'image_url',
              image_url: { url: asset.url }
            }
          ]
        }
      ],
      max_tokens: 1000,
      temperature: 0.1
    });

    // Scene Classification & Context
    visualFeatures.scene = await this.callReka(this.models.core, {
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Classify the scene and context of this ad: indoor/outdoor, setting type, lighting conditions, time of day, and overall atmosphere.'
            },
            {
              type: 'image_url',
              image_url: { url: asset.url }
            }
          ]
        }
      ],
      max_tokens: 500,
      temperature: 0.1
    });

    // Color Analysis & Mood
    visualFeatures.colors = await this.callReka(this.models.core, {
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Analyze the color palette, dominant colors, color harmony, and emotional impact of colors in this advertisement.'
            },
            {
              type: 'image_url',
              image_url: { url: asset.url }
            }
          ]
        }
      ],
      max_tokens: 500,
      temperature: 0.1
    });

    // Text Extraction & OCR
    visualFeatures.text = await this.callReka(this.models.core, {
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Extract all text from this advertisement including headlines, call-to-action buttons, product names, brand names, and any other written content.'
            },
            {
              type: 'image_url',
              image_url: { url: asset.url }
            }
          ]
        }
      ],
      max_tokens: 800,
      temperature: 0.1
    });

    // Composition & Design Analysis
    visualFeatures.composition = await this.callReka(this.models.core, {
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Analyze the visual composition: rule of thirds, symmetry, focal points, visual hierarchy, and overall design principles used in this ad.'
            },
            {
              type: 'image_url',
              image_url: { url: asset.url }
            }
          ]
        }
      ],
      max_tokens: 600,
      temperature: 0.1
    });

    return visualFeatures;
  }

  /**
   * Extract video-specific features
   */
  async extractVideoFeatures(asset) {
    const videoFeatures = {};

    // Motion Analysis
    videoFeatures.motion = await this.callReka(this.models.video, {
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Analyze the motion and movement in this video advertisement: motion intensity, direction, patterns, camera movements, and overall kinetic energy.'
            },
            {
              type: 'video_url',
              video_url: { url: asset.url }
            }
          ]
        }
      ],
      max_tokens: 800,
      temperature: 0.1
    });

    // Audio Features
    videoFeatures.audio = await this.callReka(this.models.audio, {
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Analyze the audio in this video advertisement: music type, voice characteristics, sound effects, audio quality, and overall audio design.'
            },
            {
              type: 'video_url',
              video_url: { url: asset.url }
            }
          ]
        }
      ],
      max_tokens: 700,
      temperature: 0.1
    });

    // Temporal Patterns
    videoFeatures.temporal = await this.callReka(this.models.video, {
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Analyze temporal patterns: pacing, rhythm, timing of key moments, and overall video structure in this advertisement.'
            },
            {
              type: 'video_url',
              video_url: { url: asset.url }
            }
          ]
        }
      ],
      max_tokens: 600,
      temperature: 0.1
    });

    return videoFeatures;
  }

  /**
   * Extract creative intelligence features
   */
  async extractCreativeFeatures(asset) {
    const creativeFeatures = {};

    // Emotional Sentiment Analysis
    creativeFeatures.emotion = await this.callReka(this.models.core, {
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Analyze the emotional tone, mood, and sentiment of this advertisement. What emotions does it evoke? What feeling is it trying to convey?'
            },
            {
              type: asset.type === 'video' ? 'video_url' : 'image_url',
              [asset.type === 'video' ? 'video_url' : 'image_url']: { url: asset.url }
            }
          ]
        }
      ],
      max_tokens: 600,
      temperature: 0.1
    });

    // Product Category Classification
    creativeFeatures.category = await this.callReka(this.models.core, {
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Classify the product category and industry of this advertisement: automotive, fashion, technology, food, travel, finance, etc.'
            },
            {
              type: asset.type === 'video' ? 'video_url' : 'image_url',
              [asset.type === 'video' ? 'video_url' : 'image_url']: { url: asset.url }
            }
          ]
        }
      ],
      max_tokens: 400,
      temperature: 0.1
    });

    // Ad Type Classification
    creativeFeatures.adType = await this.callReka(this.models.core, {
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Classify the type of advertisement: brand awareness, product promotion, app install, lead generation, e-commerce, etc.'
            },
            {
              type: asset.type === 'video' ? 'video_url' : 'image_url',
              [asset.type === 'video' ? 'video_url' : 'image_url']: { url: asset.url }
            }
          ]
        }
      ],
      max_tokens: 400,
      temperature: 0.1
    });

    return creativeFeatures;
  }

  /**
   * Call Reka API with retry logic
   */
  async callReka(model, payload) {
    const maxRetries = CONFIG.maxRetries;
    let lastError;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: model,
            ...payload
          }),
          timeout: CONFIG.timeout
        });

        if (!response.ok) {
          throw new Error(`Reka API error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        return {
          model: model,
          content: data.choices[0]?.message?.content || 'No content returned',
          usage: data.usage,
          timestamp: new Date().toISOString(),
          attempt: attempt
        };

      } catch (error) {
        lastError = error;
        console.log(`   ⚠️  Attempt ${attempt}/${maxRetries} failed: ${error.message}`);
        
        if (attempt < maxRetries) {
          const delay = Math.pow(2, attempt) * 1000; // Exponential backoff
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }

    throw new Error(`Reka API failed after ${maxRetries} attempts: ${lastError.message}`);
  }

  /**
   * Create feature fingerprint from analysis results
   */
  async createFeatureFingerprint() {
    console.log('\n🔍 CREATING FEATURE FINGERPRINT');
    console.log('='.repeat(60));

    const successfulAnalyses = this.analysisResults.filter(r => !r.error);
    console.log(`Creating fingerprint from ${successfulAnalyses.length} successful analyses`);

    if (successfulAnalyses.length === 0) {
      console.log('❌ No successful analyses to create fingerprint');
      return;
    }

    // Extract common patterns
    const fingerprint = {
      created_at: new Date().toISOString(),
      total_assets: successfulAnalyses.length,
      images: successfulAnalyses.filter(a => a.type === 'image').length,
      videos: successfulAnalyses.filter(a => a.type === 'video').length,
      common_features: {
        visual: this.extractCommonVisualFeatures(successfulAnalyses),
        creative: this.extractCommonCreativeFeatures(successfulAnalyses),
        video: this.extractCommonVideoFeatures(successfulAnalyses)
      },
      search_queries: this.generateSearchQueries(successfulAnalyses)
    };

    this.featureFingerprint = fingerprint;
    console.log('✅ Feature fingerprint created');
    console.log(`   - Common visual features: ${Object.keys(fingerprint.common_features.visual).length}`);
    console.log(`   - Common creative features: ${Object.keys(fingerprint.common_features.creative).length}`);
    console.log(`   - Search queries generated: ${fingerprint.search_queries.length}`);

    return fingerprint;
  }

  /**
   * Extract common visual features
   */
  extractCommonVisualFeatures(analyses) {
    const features = {
      common_objects: [],
      common_colors: [],
      common_scenes: [],
      common_compositions: []
    };

    // Analyze patterns across all assets
    for (const analysis of analyses) {
      if (analysis.features?.visual) {
        // Extract common objects, colors, scenes, etc.
        // This would analyze the content and find patterns
      }
    }

    return features;
  }

  /**
   * Extract common creative features
   */
  extractCommonCreativeFeatures(analyses) {
    const features = {
      common_emotions: [],
      common_categories: [],
      common_ad_types: []
    };

    // Analyze patterns across all assets
    for (const analysis of analyses) {
      if (analysis.features?.creative) {
        // Extract common emotions, categories, ad types
      }
    }

    return features;
  }

  /**
   * Extract common video features
   */
  extractCommonVideoFeatures(analyses) {
    const features = {
      common_motion_patterns: [],
      common_audio_features: [],
      common_temporal_patterns: []
    };

    // Analyze patterns across video assets
    for (const analysis of analyses) {
      if (analysis.type === 'video' && analysis.features?.video) {
        // Extract common motion, audio, temporal patterns
      }
    }

    return features;
  }

  /**
   * Generate search queries based on analysis
   */
  generateSearchQueries(analyses) {
    const queries = [];
    
    // Generate queries based on common features
    queries.push('advertisement examples');
    queries.push('marketing creative examples');
    queries.push('brand advertising examples');
    
    return queries;
  }

  /**
   * Save analysis results
   */
  async saveAnalysisResults() {
    const resultsFile = path.join(CONFIG.outputDir, 'asset_analysis.json');
    const fingerprintFile = path.join(CONFIG.outputDir, 'feature_fingerprint.json');
    const summaryFile = path.join(CONFIG.outputDir, 'analysis_summary.json');

    // Save detailed analysis
    fs.writeFileSync(resultsFile, JSON.stringify(this.analysisResults, null, 2));
    
    // Save feature fingerprint
    if (this.featureFingerprint) {
      fs.writeFileSync(fingerprintFile, JSON.stringify(this.featureFingerprint, null, 2));
    }

    // Save summary
    const summary = {
      total_assets: this.analysisResults.length,
      successful_analyses: this.analysisResults.filter(r => !r.error).length,
      failed_analyses: this.analysisResults.filter(r => r.error).length,
      analysis_time: new Date().toISOString(),
      fingerprint_created: !!this.featureFingerprint
    };
    
    fs.writeFileSync(summaryFile, JSON.stringify(summary, null, 2));

    console.log(`\n📁 Analysis results saved to ${CONFIG.outputDir}`);
    console.log(`   - Detailed analysis: ${resultsFile}`);
    console.log(`   - Feature fingerprint: ${fingerprintFile}`);
    console.log(`   - Summary: ${summaryFile}`);

    return { resultsFile, fingerprintFile, summaryFile };
  }
}

// Main execution
async function main() {
  const analyzer = new AssetAnalyzer();
  
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

export { AssetAnalyzer };
