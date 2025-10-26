/**
 * Reka Integration for Ad Intelligence
 * Multi-modal feature extraction using Reka's vision, video, and audio models
 */

import 'dotenv/config';
import fetch from 'node-fetch';

const REKA_CONFIG = {
  apiKey: process.env.REKA_API_KEY,
  baseUrl: 'https://api.reka.ai',
  models: {
    core: 'reka-core-20241211',
    video: 'reka-video-20241211', 
    audio: 'reka-audio-20241211'
  },
  timeout: 30000, // 30 seconds per request
  maxRetries: 3
};

class RekaFeatureExtractor {
  constructor() {
    this.apiKey = REKA_CONFIG.apiKey;
    this.baseUrl = REKA_CONFIG.baseUrl;
    this.models = REKA_CONFIG.models;
  }

  /**
   * Extract comprehensive visual features from ad images
   */
  async extractVisualFeatures(imageUrl, assetId) {
    const features = {
      asset_id: assetId,
      type: 'visual',
      extracted_at: new Date().toISOString(),
      features: {}
    };

    try {
      // 1. Object Detection & Recognition
      features.features.objects = await this.callReka(this.models.core, {
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: 'Analyze this advertisement image and identify all objects, products, logos, people, and brand elements. Provide detailed descriptions and confidence scores.'
              },
              {
                type: 'image_url',
                image_url: { url: imageUrl }
              }
            ]
          }
        ],
        max_tokens: 1000,
        temperature: 0.1
      });

      // 2. Scene Classification & Context
      features.features.scene = await this.callReka(this.models.core, {
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
                image_url: { url: imageUrl }
              }
            ]
          }
        ],
        max_tokens: 500,
        temperature: 0.1
      });

      // 3. Color Analysis & Mood
      features.features.colors = await this.callReka(this.models.core, {
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
                image_url: { url: imageUrl }
              }
            ]
          }
        ],
        max_tokens: 500,
        temperature: 0.1
      });

      // 4. Text Extraction & OCR
      features.features.text = await this.callReka(this.models.core, {
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
                image_url: { url: imageUrl }
              }
            ]
          }
        ],
        max_tokens: 800,
        temperature: 0.1
      });

      // 5. Composition & Design Analysis
      features.features.composition = await this.callReka(this.models.core, {
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
                image_url: { url: imageUrl }
              }
            ]
          }
        ],
        max_tokens: 600,
        temperature: 0.1
      });

      // 6. Brand Recognition
      features.features.brand = await this.callReka(this.models.core, {
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: 'Identify any visible brand logos, company names, or brand elements in this advertisement.'
              },
              {
                type: 'image_url',
                image_url: { url: imageUrl }
              }
            ]
          }
        ],
        max_tokens: 400,
        temperature: 0.1
      });

      return features;

    } catch (error) {
      console.error(`Failed to extract visual features for ${assetId}:`, error.message);
      throw error;
    }
  }

  /**
   * Extract video-specific features from ad videos
   */
  async extractVideoFeatures(videoUrl, assetId) {
    const features = {
      asset_id: assetId,
      type: 'video',
      extracted_at: new Date().toISOString(),
      features: {}
    };

    try {
      // 1. Motion Analysis
      features.features.motion = await this.callReka(this.models.video, {
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
                video_url: { url: videoUrl }
              }
            ]
          }
        ],
        max_tokens: 800,
        temperature: 0.1
      });

      // 2. Scene Transitions & Editing
      features.features.transitions = await this.callReka(this.models.video, {
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: 'Analyze scene transitions, editing patterns, cut frequency, and pacing in this video advertisement.'
              },
              {
                type: 'video_url',
                video_url: { url: videoUrl }
              }
            ]
          }
        ],
        max_tokens: 600,
        temperature: 0.1
      });

      // 3. Audio Features
      features.features.audio = await this.callReka(this.models.audio, {
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
                video_url: { url: videoUrl }
              }
            ]
          }
        ],
        max_tokens: 700,
        temperature: 0.1
      });

      // 4. Temporal Patterns
      features.features.temporal = await this.callReka(this.models.video, {
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
                video_url: { url: videoUrl }
              }
            ]
          }
        ],
        max_tokens: 600,
        temperature: 0.1
      });

      // 5. Face & Person Analysis
      features.features.people = await this.callReka(this.models.video, {
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: 'Analyze people in this video: demographics, emotions, expressions, interactions, and human elements.'
              },
              {
                type: 'video_url',
                video_url: { url: videoUrl }
              }
            ]
          }
        ],
        max_tokens: 600,
        temperature: 0.1
      });

      return features;

    } catch (error) {
      console.error(`Failed to extract video features for ${assetId}:`, error.message);
      throw error;
    }
  }

  /**
   * Extract creative intelligence features
   */
  async extractCreativeFeatures(assetUrl, assetId, assetType) {
    const features = {
      asset_id: assetId,
      type: 'creative',
      extracted_at: new Date().toISOString(),
      features: {}
    };

    try {
      // 1. Emotional Sentiment Analysis
      features.features.emotion = await this.callReka(this.models.core, {
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: 'Analyze the emotional tone, mood, and sentiment of this advertisement. What emotions does it evoke? What feeling is it trying to convey?'
              },
              {
                type: assetType === 'video' ? 'video_url' : 'image_url',
                [assetType === 'video' ? 'video_url' : 'image_url']: { url: assetUrl }
              }
            ]
          }
        ],
        max_tokens: 600,
        temperature: 0.1
      });

      // 2. Product Category Classification
      features.features.category = await this.callReka(this.models.core, {
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: 'Classify the product category and industry of this advertisement: automotive, fashion, technology, food, travel, finance, etc.'
              },
              {
                type: assetType === 'video' ? 'video_url' : 'image_url',
                [assetType === 'video' ? 'video_url' : 'image_url']: { url: assetUrl }
              }
            ]
          }
        ],
        max_tokens: 400,
        temperature: 0.1
      });

      // 3. Ad Type Classification
      features.features.adType = await this.callReka(this.models.core, {
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: 'Classify the type of advertisement: brand awareness, product promotion, app install, lead generation, e-commerce, etc.'
              },
              {
                type: assetType === 'video' ? 'video_url' : 'image_url',
                [assetType === 'video' ? 'video_url' : 'image_url']: { url: assetUrl }
              }
            ]
          }
        ],
        max_tokens: 400,
        temperature: 0.1
      });

      // 4. Call-to-Action Analysis
      features.features.cta = await this.callReka(this.models.core, {
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: 'Analyze the call-to-action in this advertisement: prominence, clarity, urgency, and effectiveness of the CTA.'
              },
              {
                type: assetType === 'video' ? 'video_url' : 'image_url',
                [assetType === 'video' ? 'video_url' : 'image_url']: { url: assetUrl }
              }
            ]
          }
        ],
        max_tokens: 500,
        temperature: 0.1
      });

      // 5. Visual Complexity Analysis
      features.features.complexity = await this.callReka(this.models.core, {
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: 'Analyze the visual complexity of this advertisement: information density, visual clutter, simplicity vs complexity, and overall design sophistication.'
              },
              {
                type: assetType === 'video' ? 'video_url' : 'image_url',
                [assetType === 'video' ? 'video_url' : 'image_url']: { url: assetUrl }
              }
            ]
          }
        ],
        max_tokens: 500,
        temperature: 0.1
      });

      return features;

    } catch (error) {
      console.error(`Failed to extract creative features for ${assetId}:`, error.message);
      throw error;
    }
  }

  /**
   * Call Reka API with retry logic
   */
  async callReka(model, payload) {
    const maxRetries = REKA_CONFIG.maxRetries;
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
          timeout: REKA_CONFIG.timeout
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
   * Batch process multiple assets
   */
  async batchExtractFeatures(assets) {
    const results = [];
    const batchSize = 5; // Process 5 assets concurrently
    
    for (let i = 0; i < assets.length; i += batchSize) {
      const batch = assets.slice(i, i + batchSize);
      console.log(`\n📦 Processing batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(assets.length / batchSize)}`);
      
      const batchPromises = batch.map(async (asset) => {
        try {
          const features = {
            asset_id: asset.id,
            type: asset.type,
            url: asset.url,
            extracted_at: new Date().toISOString(),
            features: {}
          };

          // Extract features based on asset type
          if (asset.type === 'image') {
            features.features.visual = await this.extractVisualFeatures(asset.url, asset.id);
            features.features.creative = await this.extractCreativeFeatures(asset.url, asset.id, 'image');
          } else if (asset.type === 'video') {
            features.features.video = await this.extractVideoFeatures(asset.url, asset.id);
            features.features.creative = await this.extractCreativeFeatures(asset.url, asset.id, 'video');
          }

          return features;

        } catch (error) {
          console.error(`❌ Failed to process ${asset.id}: ${error.message}`);
          return {
            asset_id: asset.id,
            error: error.message,
            extracted_at: new Date().toISOString()
          };
        }
      });

      const batchResults = await Promise.allSettled(batchPromises);
      results.push(...batchResults.map(r => r.status === 'fulfilled' ? r.value : r.reason));
      
      // Small delay between batches to avoid rate limiting
      if (i + batchSize < assets.length) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    return results;
  }
}

export { RekaFeatureExtractor };
