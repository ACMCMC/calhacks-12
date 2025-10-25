/**
 * Feature Fingerprint Creator
 * Analyzes existing asset analysis to create searchable feature patterns
 */

import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CONFIG = {
  analysisFile: path.join(__dirname, '..', 'analysis_results', 'working_analysis.json'),
  outputDir: path.join(__dirname, '..', 'feature_fingerprints')
};

class FeatureFingerprint {
  constructor() {
    this.analysisData = null;
    this.fingerprint = null;
    this.searchQueries = [];
  }

  async initialize() {
    console.log('🔍 Initializing Feature Fingerprint Creator');
    
    // Create output directory
    if (!fs.existsSync(CONFIG.outputDir)) {
      fs.mkdirSync(CONFIG.outputDir, { recursive: true });
    }

    // Load analysis data
    if (!fs.existsSync(CONFIG.analysisFile)) {
      throw new Error(`Analysis file not found: ${CONFIG.analysisFile}`);
    }

    this.analysisData = JSON.parse(fs.readFileSync(CONFIG.analysisFile, 'utf8'));
    console.log(`✅ Loaded analysis data for ${this.analysisData.length} assets`);
    
    return this;
  }

  /**
   * Create comprehensive feature fingerprint
   */
  async createFingerprint() {
    console.log('\n🧬 CREATING FEATURE FINGERPRINT');
    console.log('='.repeat(60));

    const successfulAnalyses = this.analysisData.filter(a => !a.error);
    console.log(`Creating fingerprint from ${successfulAnalyses.length} successful analyses`);

    if (successfulAnalyses.length === 0) {
      throw new Error('No successful analyses to create fingerprint');
    }

    // Extract feature patterns
    const fingerprint = {
      created_at: new Date().toISOString(),
      total_assets: successfulAnalyses.length,
      images: successfulAnalyses.filter(a => a.type === 'image').length,
      videos: successfulAnalyses.filter(a => a.type === 'video').length,
      
      // Visual patterns
      visual_patterns: this.extractVisualPatterns(successfulAnalyses),
      
      // Creative patterns  
      creative_patterns: this.extractCreativePatterns(successfulAnalyses),
      
      // Content patterns
      content_patterns: this.extractContentPatterns(successfulAnalyses),
      
      // Search strategies
      search_strategies: this.generateSearchStrategies(successfulAnalyses),
      
      // Quality indicators
      quality_indicators: this.extractQualityIndicators(successfulAnalyses)
    };

    this.fingerprint = fingerprint;
    console.log('✅ Feature fingerprint created');
    console.log(`   - Visual patterns: ${Object.keys(fingerprint.visual_patterns).length}`);
    console.log(`   - Creative patterns: ${Object.keys(fingerprint.creative_patterns).length}`);
    console.log(`   - Search strategies: ${fingerprint.search_strategies.length}`);

    return fingerprint;
  }

  /**
   * Extract visual patterns from analyses
   */
  extractVisualPatterns(analyses) {
    const patterns = {
      common_objects: new Set(),
      common_colors: new Set(),
      common_compositions: new Set(),
      common_scenes: new Set(),
      common_text_elements: new Set()
    };

    // Analyze each description for patterns
    for (const analysis of analyses) {
      if (analysis.features?.description) {
        const description = analysis.features.description.toLowerCase();
        
        // Extract objects
        const objects = this.extractObjects(description);
        objects.forEach(obj => patterns.common_objects.add(obj));
        
        // Extract colors
        const colors = this.extractColors(description);
        colors.forEach(color => patterns.common_colors.add(color));
        
        // Extract compositions
        const compositions = this.extractCompositions(description);
        compositions.forEach(comp => patterns.common_compositions.add(comp));
        
        // Extract scenes
        const scenes = this.extractScenes(description);
        scenes.forEach(scene => patterns.common_scenes.add(scene));
        
        // Extract text elements
        const textElements = this.extractTextElements(description);
        textElements.forEach(text => patterns.common_text_elements.add(text));
      }
    }

    return {
      objects: Array.from(patterns.common_objects),
      colors: Array.from(patterns.common_colors),
      compositions: Array.from(patterns.common_compositions),
      scenes: Array.from(patterns.common_scenes),
      text_elements: Array.from(patterns.common_text_elements)
    };
  }

  /**
   * Extract creative patterns
   */
  extractCreativePatterns(analyses) {
    const patterns = {
      emotions: new Set(),
      ad_types: new Set(),
      industries: new Set(),
      styles: new Set()
    };

    for (const analysis of analyses) {
      if (analysis.features?.description) {
        const description = analysis.features.description.toLowerCase();
        
        // Extract emotions
        const emotions = this.extractEmotions(description);
        emotions.forEach(emotion => patterns.emotions.add(emotion));
        
        // Extract ad types
        const adTypes = this.extractAdTypes(description);
        adTypes.forEach(type => patterns.ad_types.add(type));
        
        // Extract industries
        const industries = this.extractIndustries(description);
        industries.forEach(industry => patterns.industries.add(industry));
        
        // Extract styles
        const styles = this.extractStyles(description);
        styles.forEach(style => patterns.styles.add(style));
      }
    }

    return {
      emotions: Array.from(patterns.emotions),
      ad_types: Array.from(patterns.ad_types),
      industries: Array.from(patterns.industries),
      styles: Array.from(patterns.styles)
    };
  }

  /**
   * Extract content patterns
   */
  extractContentPatterns(analyses) {
    const patterns = {
      common_themes: new Set(),
      common_keywords: new Set(),
      common_phrases: new Set(),
      common_concepts: new Set()
    };

    for (const analysis of analyses) {
      if (analysis.features?.description) {
        const description = analysis.features.description.toLowerCase();
        
        // Extract themes
        const themes = this.extractThemes(description);
        themes.forEach(theme => patterns.common_themes.add(theme));
        
        // Extract keywords
        const keywords = this.extractKeywords(description);
        keywords.forEach(keyword => patterns.common_keywords.add(keyword));
        
        // Extract phrases
        const phrases = this.extractPhrases(description);
        phrases.forEach(phrase => patterns.common_phrases.add(phrase));
        
        // Extract concepts
        const concepts = this.extractConcepts(description);
        concepts.forEach(concept => patterns.common_concepts.add(concept));
      }
    }

    return {
      themes: Array.from(patterns.common_themes),
      keywords: Array.from(patterns.common_keywords),
      phrases: Array.from(patterns.common_phrases),
      concepts: Array.from(patterns.common_concepts)
    };
  }

  /**
   * Generate search strategies based on patterns
   */
  generateSearchStrategies(analyses) {
    const strategies = [];
    
    // Visual-based searches
    const visualPatterns = this.extractVisualPatterns(analyses);
    for (const object of visualPatterns.objects.slice(0, 5)) {
      strategies.push({
        type: 'visual_object',
        query: `${object} advertisement examples`,
        priority: 'high',
        expected_yield: 'high'
      });
    }
    
    // Industry-based searches
    const creativePatterns = this.extractCreativePatterns(analyses);
    for (const industry of creativePatterns.industries.slice(0, 3)) {
      strategies.push({
        type: 'industry',
        query: `${industry} ads examples`,
        priority: 'high',
        expected_yield: 'medium'
      });
    }
    
    // Style-based searches
    for (const style of creativePatterns.styles.slice(0, 3)) {
      strategies.push({
        type: 'style',
        query: `${style} advertising examples`,
        priority: 'medium',
        expected_yield: 'medium'
      });
    }
    
    // Platform-specific searches
    const platforms = ['instagram', 'facebook', 'youtube', 'tiktok', 'linkedin'];
    for (const platform of platforms) {
      strategies.push({
        type: 'platform',
        query: `${platform} ads examples`,
        priority: 'high',
        expected_yield: 'high'
      });
    }
    
    return strategies;
  }

  /**
   * Extract quality indicators
   */
  extractQualityIndicators(analyses) {
    const indicators = {
      file_sizes: analyses.map(a => a.features?.basic?.file_size || 0),
      common_extensions: new Set(),
      quality_tiers: {
        high: 0,
        medium: 0,
        low: 0
      }
    };

    // Analyze file sizes for quality indicators
    for (const analysis of analyses) {
      if (analysis.features?.basic?.file_size) {
        const size = analysis.features.basic.file_size;
        if (size > 500000) indicators.quality_tiers.high++;
        else if (size > 100000) indicators.quality_tiers.medium++;
        else indicators.quality_tiers.low++;
      }
      
      if (analysis.filename) {
        const ext = analysis.filename.split('.').pop();
        indicators.common_extensions.add(ext);
      }
    }

    indicators.common_extensions = Array.from(indicators.common_extensions);
    indicators.avg_file_size = indicators.file_sizes.reduce((a, b) => a + b, 0) / indicators.file_sizes.length;
    
    return indicators;
  }

  // Helper methods for pattern extraction
  extractObjects(description) {
    const objects = [];
    const objectKeywords = ['product', 'logo', 'brand', 'person', 'model', 'car', 'phone', 'food', 'clothing', 'shoes', 'watch', 'bag', 'jewelry'];
    for (const keyword of objectKeywords) {
      if (description.includes(keyword)) {
        objects.push(keyword);
      }
    }
    return objects;
  }

  extractColors(description) {
    const colors = [];
    const colorKeywords = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black', 'white', 'gray', 'brown', 'gold', 'silver'];
    for (const color of colorKeywords) {
      if (description.includes(color)) {
        colors.push(color);
      }
    }
    return colors;
  }

  extractCompositions(description) {
    const compositions = [];
    const compKeywords = ['centered', 'rule of thirds', 'symmetrical', 'asymmetrical', 'diagonal', 'vertical', 'horizontal', 'focal point'];
    for (const comp of compKeywords) {
      if (description.includes(comp)) {
        compositions.push(comp);
      }
    }
    return compositions;
  }

  extractScenes(description) {
    const scenes = [];
    const sceneKeywords = ['indoor', 'outdoor', 'studio', 'natural', 'urban', 'home', 'office', 'restaurant', 'beach', 'mountain'];
    for (const scene of sceneKeywords) {
      if (description.includes(scene)) {
        scenes.push(scene);
      }
    }
    return scenes;
  }

  extractTextElements(description) {
    const textElements = [];
    const textKeywords = ['headline', 'tagline', 'cta', 'button', 'call to action', 'slogan', 'brand name', 'price'];
    for (const text of textKeywords) {
      if (description.includes(text)) {
        textElements.push(text);
      }
    }
    return textElements;
  }

  extractEmotions(description) {
    const emotions = [];
    const emotionKeywords = ['happy', 'confident', 'excited', 'calm', 'energetic', 'professional', 'friendly', 'luxury', 'affordable'];
    for (const emotion of emotionKeywords) {
      if (description.includes(emotion)) {
        emotions.push(emotion);
      }
    }
    return emotions;
  }

  extractAdTypes(description) {
    const adTypes = [];
    const typeKeywords = ['brand awareness', 'product promotion', 'app install', 'lead generation', 'e-commerce', 'retail', 'service'];
    for (const type of typeKeywords) {
      if (description.includes(type)) {
        adTypes.push(type);
      }
    }
    return adTypes;
  }

  extractIndustries(description) {
    const industries = [];
    const industryKeywords = ['automotive', 'fashion', 'technology', 'food', 'travel', 'finance', 'health', 'beauty', 'fitness', 'entertainment'];
    for (const industry of industryKeywords) {
      if (description.includes(industry)) {
        industries.push(industry);
      }
    }
    return industries;
  }

  extractStyles(description) {
    const styles = [];
    const styleKeywords = ['modern', 'classic', 'minimalist', 'bold', 'elegant', 'playful', 'professional', 'creative', 'artistic'];
    for (const style of styleKeywords) {
      if (description.includes(style)) {
        styles.push(style);
      }
    }
    return styles;
  }

  extractThemes(description) {
    const themes = [];
    const themeKeywords = ['lifestyle', 'luxury', 'affordability', 'innovation', 'tradition', 'sustainability', 'convenience', 'quality'];
    for (const theme of themeKeywords) {
      if (description.includes(theme)) {
        themes.push(theme);
      }
    }
    return themes;
  }

  extractKeywords(description) {
    // Simple keyword extraction
    const words = description.split(/\s+/);
    const keywords = words.filter(word => 
      word.length > 4 && 
      !['this', 'that', 'with', 'from', 'they', 'have', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'will', 'about', 'there', 'could', 'other', 'after', 'first', 'well', 'also', 'where', 'much', 'some', 'very', 'when', 'here', 'just', 'into', 'over', 'think', 'more', 'these', 'your', 'work', 'life', 'only', 'know', 'years', 'most', 'people', 'good', 'make', 'through', 'back', 'much', 'before', 'right', 'means', 'old', 'any', 'same', 'tell', 'boy', 'follow', 'came', 'want', 'show', 'every', 'great', 'put', 'end', 'why', 'let', 'help', 'put', 'turn', 'here', 'why', 'ask', 'went', 'men', 'read', 'need', 'land', 'different', 'home', 'move', 'try', 'kind', 'hand', 'picture', 'again', 'change', 'off', 'play', 'spell', 'air', 'away', 'animal', 'house', 'point', 'page', 'letter', 'mother', 'answer', 'found', 'study', 'still', 'learn', 'should', 'america', 'world', 'high', 'every', 'near', 'add', 'food', 'between', 'own', 'below', 'country', 'plant', 'last', 'school', 'father', 'keep', 'tree', 'never', 'start', 'city', 'earth', 'eyes', 'light', 'thought', 'head', 'under', 'story', 'saw', 'left', 'dont', 'few', 'while', 'along', 'might', 'close', 'something', 'seemed', 'next', 'hard', 'open', 'example', 'begin', 'life', 'always', 'those', 'both', 'paper', 'together', 'got', 'group', 'often', 'run', 'important', 'until', 'children', 'side', 'feet', 'car', 'miles', 'night', 'walked', 'white', 'sea', 'began', 'grew', 'took', 'river', 'four', 'carry', 'state', 'once', 'book', 'hear', 'stop', 'without', 'second', 'later', 'miss', 'idea', 'enough', 'eat', 'face', 'watch', 'far', 'indian', 'real', 'almost', 'let', 'above', 'girl', 'sometimes', 'mountain', 'cut', 'young', 'talk', 'soon', 'list', 'song', 'leave', 'family', 'it\'s'].includes(word.toLowerCase())
    );
    return keywords.slice(0, 10); // Top 10 keywords
  }

  extractPhrases(description) {
    const phrases = [];
    const phrasePatterns = ['call to action', 'brand awareness', 'product promotion', 'visual hierarchy', 'color palette', 'focal point', 'white space', 'brand identity'];
    for (const phrase of phrasePatterns) {
      if (description.includes(phrase)) {
        phrases.push(phrase);
      }
    }
    return phrases;
  }

  extractConcepts(description) {
    const concepts = [];
    const conceptKeywords = ['marketing', 'advertising', 'branding', 'design', 'visual', 'creative', 'promotion', 'engagement', 'conversion'];
    for (const concept of conceptKeywords) {
      if (description.includes(concept)) {
        concepts.push(concept);
      }
    }
    return concepts;
  }

  /**
   * Save fingerprint to files
   */
  async saveFingerprint() {
    const fingerprintFile = path.join(CONFIG.outputDir, 'feature_fingerprint.json');
    const searchQueriesFile = path.join(CONFIG.outputDir, 'search_queries.json');
    const summaryFile = path.join(CONFIG.outputDir, 'fingerprint_summary.json');

    // Save detailed fingerprint
    fs.writeFileSync(fingerprintFile, JSON.stringify(this.fingerprint, null, 2));
    
    // Save search queries
    const searchQueries = this.fingerprint.search_strategies.map(s => ({
      query: s.query,
      type: s.type,
      priority: s.priority,
      expected_yield: s.expected_yield
    }));
    fs.writeFileSync(searchQueriesFile, JSON.stringify(searchQueries, null, 2));

    // Save summary
    const summary = {
      total_assets: this.fingerprint.total_assets,
      images: this.fingerprint.images,
      videos: this.fingerprint.videos,
      visual_patterns_count: Object.keys(this.fingerprint.visual_patterns).length,
      creative_patterns_count: Object.keys(this.fingerprint.creative_patterns).length,
      search_strategies_count: this.fingerprint.search_strategies.length,
      created_at: this.fingerprint.created_at
    };
    fs.writeFileSync(summaryFile, JSON.stringify(summary, null, 2));

    console.log(`\n📁 Feature fingerprint saved to ${CONFIG.outputDir}`);
    console.log(`   - Fingerprint: ${fingerprintFile}`);
    console.log(`   - Search queries: ${searchQueriesFile}`);
    console.log(`   - Summary: ${summaryFile}`);

    return { fingerprintFile, searchQueriesFile, summaryFile };
  }
}

// Main execution
async function main() {
  const fingerprint = new FeatureFingerprint();
  
  try {
    await fingerprint.initialize();
    await fingerprint.createFingerprint();
    await fingerprint.saveFingerprint();
    
    console.log('\n🎉 Feature fingerprint created successfully!');
    
  } catch (error) {
    console.error('❌ Fingerprint creation failed:', error.message);
    process.exit(1);
  }
}

// Run if this is the main module
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  main();
}

export { FeatureFingerprint };
