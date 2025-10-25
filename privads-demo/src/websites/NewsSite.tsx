import React, { useState, useEffect } from 'react';
import { usePrivAds } from '../components/PrivAdsProvider';
import { Clock, User, TrendingUp } from 'lucide-react';
import './NewsSite.css';

interface Article {
  id: string;
  title: string;
  excerpt: string;
  author: string;
  publishedAt: string;
  category: string;
  readTime: number;
}

const NewsSite: React.FC = () => {
  const { updateSiteContext, updatePrediction } = usePrivAds();
  const [currentArticle, setCurrentArticle] = useState<Article | null>(null);
  const [readingProgress, setReadingProgress] = useState(0);

  // Mock news articles
  const articles: Article[] = [
    {
      id: '1',
      title: 'AI Revolution in Healthcare: New Breakthroughs Save Lives',
      excerpt: 'Researchers at leading medical institutions have developed AI-powered diagnostic tools that can detect early-stage cancers with unprecedented accuracy, potentially saving millions of lives annually.',
      author: 'Dr. Sarah Chen',
      publishedAt: '2025-10-25T10:30:00Z',
      category: 'Health',
      readTime: 5
    },
    {
      id: '2',
      title: 'Sustainable Energy Solutions Power Global Transition',
      excerpt: 'Countries worldwide are accelerating their shift to renewable energy sources, with solar and wind power now accounting for over 40% of new electricity generation capacity.',
      author: 'Michael Rodriguez',
      publishedAt: '2025-10-25T09:15:00Z',
      category: 'Environment',
      readTime: 4
    },
    {
      id: '3',
      title: 'Tech Giants Invest Billions in Privacy-First Computing',
      excerpt: 'Major technology companies are committing unprecedented resources to develop privacy-preserving technologies, marking a fundamental shift in how personal data is handled online.',
      author: 'Emma Thompson',
      publishedAt: '2025-10-25T08:45:00Z',
      category: 'Technology',
      readTime: 6
    },
    {
      id: '4',
      title: 'Global Education Initiative Reaches One Billion Students',
      excerpt: 'An international coalition has successfully expanded access to quality education, with innovative online platforms connecting students worldwide with expert instructors.',
      author: 'Dr. James Park',
      publishedAt: '2025-10-25T07:20:00Z',
      category: 'Education',
      readTime: 7
    }
  ];

  useEffect(() => {
    // Update site context for news consumption
    updateSiteContext({
      siteType: 'news',
      currentCategory: currentArticle?.category || 'general',
      readingProgress,
      timeSpent: readingProgress * 2, // Mock time spent
      contentType: 'article'
    });

    // Update prediction based on reading engagement
    const basePrediction = 25; // Base receptiveness for news sites
    const engagementBonus = readingProgress * 0.5; // More engaged = more receptive
    const categoryBonus = currentArticle?.category === 'Technology' ? 15 : 0; // Tech readers more receptive to tech ads

    updatePrediction({
      probability: Math.min(basePrediction + engagementBonus + categoryBonus, 85),
      factors: {
        contextRelevance: 70 + (readingProgress * 0.3),
        userPreferences: 60 + categoryBonus,
        contentType: 75, // News content generally receptive
        interactionHistory: readingProgress * 0.8
      }
    });
  }, [currentArticle, readingProgress, updateSiteContext, updatePrediction]);

  const handleArticleClick = (article: Article) => {
    setCurrentArticle(article);
    setReadingProgress(0);
  };

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const element = e.currentTarget;
    const scrollTop = element.scrollTop;
    const scrollHeight = element.scrollHeight - element.clientHeight;
    const progress = scrollHeight > 0 ? (scrollTop / scrollHeight) * 100 : 0;
    setReadingProgress(Math.round(progress));
  };

  return (
    <div className="news-site">
      <div className="site-header">
        <h1>TruthLens News</h1>
        <p>Your trusted source for privacy-first journalism</p>
      </div>

      <div className="content-layout">
        <div className="articles-list">
          <h2>Latest News</h2>
          {articles.map(article => (
            <div
              key={article.id}
              className={`article-card ${currentArticle?.id === article.id ? 'active' : ''}`}
              onClick={() => handleArticleClick(article)}
            >
              <div className="article-meta">
                <span className="category">{article.category}</span>
                <span className="read-time">
                  <Clock size={14} />
                  {article.readTime} min read
                </span>
              </div>
              <h3>{article.title}</h3>
              <p>{article.excerpt}</p>
              <div className="article-footer">
                <span className="author">
                  <User size={14} />
                  {article.author}
                </span>
                <span className="date">
                  {new Date(article.publishedAt).toLocaleDateString()}
                </span>
              </div>
            </div>
          ))}
        </div>

        <div className="article-viewer">
          {currentArticle ? (
            <div className="article-content" onScroll={handleScroll}>
              <div className="article-header">
                <span className="category-badge">{currentArticle.category}</span>
                <h1>{currentArticle.title}</h1>
                <div className="article-meta">
                  <span className="author">
                    <User size={16} />
                    {currentArticle.author}
                  </span>
                  <span className="date">
                    {new Date(currentArticle.publishedAt).toLocaleDateString()}
                  </span>
                  <span className="read-time">
                    <Clock size={16} />
                    {currentArticle.readTime} min read
                  </span>
                </div>
              </div>

              <div className="article-body">
                <p>{currentArticle.excerpt}</p>

                {/* Mock article content */}
                <p>In a groundbreaking development that promises to reshape the landscape of modern healthcare, researchers have unveiled a new artificial intelligence system capable of detecting medical conditions with remarkable precision.</p>

                <p>The technology, developed through collaboration between leading medical institutions and AI researchers, represents a significant leap forward in diagnostic capabilities. Early testing shows detection rates exceeding 95% for several critical conditions.</p>

                <p>"This breakthrough has the potential to save countless lives by enabling earlier intervention and more accurate diagnoses," said Dr. Sarah Chen, lead researcher on the project.</p>

                <div className="ad-placeholder">
                  <div className="ad-content">
                    <h4>Advertisement</h4>
                    <p>Experience personalized healthcare recommendations powered by AI. Learn more about preventive care solutions.</p>
                    <button className="ad-button">Explore Health Solutions</button>
                  </div>
                </div>

                <p>The implications extend beyond individual patient care. Healthcare systems worldwide are already exploring ways to integrate this technology into their standard diagnostic workflows.</p>

                <p>Privacy advocates have praised the approach, noting that the system processes data locally without requiring sensitive medical information to be transmitted to external servers.</p>

                <p>As the technology continues to evolve, experts predict it will become an essential tool in the medical professional's arsenal, complementing rather than replacing human expertise.</p>
              </div>

              <div className="ad-placeholder">
                <div className="ad-content">
                  <h4>Premium News Analysis</h4>
                  <p>Get deeper insights and exclusive analysis with our premium subscription. Access expert commentary and in-depth reporting.</p>
                  <button className="ad-button">Upgrade to Premium</button>
                </div>
              </div>

              <div className="reading-progress">
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${readingProgress}%` }}
                  />
                </div>
                <span className="progress-text">{readingProgress}% complete</span>
              </div>
            </div>
          ) : (
            <div className="no-article">
              <TrendingUp size={48} />
              <h3>Select an article to start reading</h3>
              <p>Choose from the latest news stories on the left to begin your reading experience.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default NewsSite;