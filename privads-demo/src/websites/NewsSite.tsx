// File: privads_demo/src/websites/NewsSite.tsx
import React, { useMemo } from 'react';
import './NewsSite.css';
import CustomizedAd from '../components/CustomizedAd';

const ARTICLE = {
  title: 'AI Revolution in Healthcare: New Breakthroughs Save Lives',
  author: 'Dr. Sarah Chen',
  publishedAt: '2025-10-25T10:30:00Z',
  content: [
    'Researchers at leading medical institutions have developed AI-powered diagnostic tools that can detect early-stage cancers with unprecedented accuracy, potentially saving millions of lives annually.',
    'In a groundbreaking development that promises to reshape the landscape of modern healthcare, researchers have unveiled a new artificial intelligence system capable of detecting medical conditions with remarkable precision.',
    'The technology, developed through collaboration between leading medical institutions and AI researchers, represents a significant leap forward in diagnostic capabilities. Early testing shows detection rates exceeding 95% for several critical conditions.',
    '"This breakthrough has the potential to save countless lives by enabling earlier intervention and more accurate diagnoses," said Dr. Sarah Chen, lead researcher on the project.',
    'The implications extend beyond individual patient care. Healthcare systems worldwide are already exploring ways to integrate this technology into their standard diagnostic workflows.',
    'Privacy advocates have praised the approach, noting that the system processes data locally without requiring sensitive medical information to be transmitted to external servers.',
    'As the technology continues to evolve, experts predict it will become an essential tool in the medical professional\'s arsenal, complementing rather than replacing human expertise.',
    'The AI system leverages deep learning models trained on millions of anonymized medical records, allowing it to recognize subtle patterns that may elude even experienced clinicians.',
    'In one clinical trial, the AI flagged early signs of pancreatic cancer in patients who would have otherwise gone undiagnosed until much later stages.',
    'Healthcare providers are optimistic but cautious, emphasizing the importance of rigorous validation and oversight as the technology is rolled out more broadly.',
    'Ethical considerations remain at the forefront, with ongoing debates about data privacy, algorithmic transparency, and the potential for bias in AI-driven diagnostics.',
    'To address these concerns, the development team has implemented robust safeguards, including regular audits and open-source publication of the model\'s decision-making criteria.',
    'Patients who have benefited from the technology describe a sense of relief and empowerment, knowing that their health is being monitored with cutting-edge tools.',
    'The World Health Organization has issued preliminary guidelines for the safe integration of AI in clinical settings, highlighting the need for global cooperation and knowledge sharing.',
    'Looking ahead, researchers are exploring ways to expand the system\'s capabilities to include predictive analytics for chronic disease management and personalized treatment recommendations.',
    'The economic impact is also significant, with projections suggesting billions in healthcare savings due to earlier interventions and reduced rates of misdiagnosis.',
    'Medical schools are beginning to incorporate AI literacy into their curricula, preparing the next generation of doctors to work alongside intelligent systems.',
    'As adoption grows, experts stress the importance of maintaining a human touch in patient care, ensuring that technology enhances rather than replaces the doctor-patient relationship.',
    'This story is part of TruthLens News\' ongoing series on the intersection of technology and society, examining how innovation is reshaping our world.',
    // --- EXTENDED CONTENT FOR 15 MINUTES READING ---
    ...Array(60).fill('').map((_, i) => `Section ${i+1}: ` +
      'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque euismod, nisi eu consectetur consectetur, nisl nisi consectetur nisi, euismod euismod nisi nisi euismod. '+
      'Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Etiam at risus et justo dignissim congue. '+
      'Donec congue lacinia dui, a porttitor lectus condimentum laoreet. Nunc eu ullamcorper orci. Quisque eget odio ac lectus vestibulum faucibus eget in metus. '+
      'In pellentesque faucibus vestibulum. Nulla at nulla justo, eget luctus tortor. Nulla facilisi. Duis aliquet egestas purus in blandit. '+
      'Curabitur vulputate, ligula lacinia scelerisque tempor, lacus lacus ornare ante, ac egestas est urna sit amet arcu. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos.')
  ]
};

const NewsSite: React.FC = () => {
  // Extract page context for ad customization
  const pageContext = useMemo(() => {
    const contentText = ARTICLE.content.slice(0, 5).join(' '); // First 5 paragraphs

    return {
      title: ARTICLE.title,
      content: contentText,
      keywords: ['AI', 'healthcare', 'technology', 'medical', 'diagnosis', 'innovation'],
      page_type: 'news' as const,
      url: window.location.href,
      summary_text: ARTICLE.content[0], // Lead paragraph
    };
  }, []);

  return (
    <div className="news-article-simple">
      <h1>{ARTICLE.title}</h1>
      <div className="article-meta-simple">
        <span>By {ARTICLE.author}</span>
        <span> | {new Date(ARTICLE.publishedAt).toLocaleDateString()}</span>
      </div>
      <div className="article-body-simple">
        {ARTICLE.content.slice(0, 3).map((para, i) => <p key={i}>{para}</p>)}

        {/* Customized Ad appears after 3 paragraphs */}
        <CustomizedAd pageContext={pageContext} />

        {ARTICLE.content.slice(3).map((para, i) => <p key={i + 3}>{para}</p>)}
      </div>
    </div>
  );
};

export default NewsSite;