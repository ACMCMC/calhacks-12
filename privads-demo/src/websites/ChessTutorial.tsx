import React, { useState, useEffect } from 'react';
import { usePrivAds } from '../components/PrivAdsProvider';
import { Crown, Target, BookOpen, Trophy } from 'lucide-react';
import './ChessTutorial.css';

interface ChessLesson {
  id: string;
  title: string;
  description: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  estimatedTime: number;
  completed: boolean;
}

const ChessTutorial: React.FC = () => {
  const { updateSiteContext, updatePrediction } = usePrivAds();
  const [currentLesson, setCurrentLesson] = useState<ChessLesson | null>(null);
  const [lessonProgress, setLessonProgress] = useState(0);
  const [movesPlayed, setMovesPlayed] = useState(0);

  const lessons: ChessLesson[] = [
    {
      id: '1',
      title: 'Basic Pawn Movement',
      description: 'Learn how pawns move and capture on the chessboard',
      difficulty: 'Beginner',
      estimatedTime: 10,
      completed: false
    },
    {
      id: '2',
      title: 'Knight Moves and Forks',
      description: 'Master the unique movement of knights and learn about forking tactics',
      difficulty: 'Beginner',
      estimatedTime: 15,
      completed: false
    },
    {
      id: '3',
      title: 'Rook and Queen Coordination',
      description: 'Understanding how to coordinate your major pieces for maximum impact',
      difficulty: 'Intermediate',
      estimatedTime: 20,
      completed: false
    },
    {
      id: '4',
      title: 'Advanced Endgame Techniques',
      description: 'Master the art of converting advantages in the endgame',
      difficulty: 'Advanced',
      estimatedTime: 25,
      completed: false
    }
  ];

  useEffect(() => {
    // Update site context for chess learning
    updateSiteContext({
      siteType: 'education',
      contentType: 'chess_tutorial',
      currentLesson: currentLesson?.id || null,
      lessonProgress,
      movesPlayed,
      difficulty: currentLesson?.difficulty || 'general',
      timeSpent: lessonProgress * 3 // Mock time spent
    });

    // Update prediction based on learning engagement
    const basePrediction = 35; // Educational content generally receptive
    const progressBonus = lessonProgress * 0.4; // More progress = more engaged
    const difficultyBonus = currentLesson?.difficulty === 'Advanced' ? 20 : 0; // Advanced learners more receptive to complex ads
    const interactionBonus = movesPlayed * 2; // Interactive elements increase receptiveness

    updatePrediction({
      probability: Math.min(basePrediction + progressBonus + difficultyBonus + interactionBonus, 90),
      factors: {
        contextRelevance: 65 + (lessonProgress * 0.3),
        userPreferences: 70 + difficultyBonus,
        contentType: 80, // Educational content highly receptive
        interactionHistory: lessonProgress * 0.6 + (movesPlayed * 0.5)
      }
    });
  }, [currentLesson, lessonProgress, movesPlayed, updateSiteContext, updatePrediction]);

  const handleLessonSelect = (lesson: ChessLesson) => {
    setCurrentLesson(lesson);
    setLessonProgress(0);
    setMovesPlayed(0);
  };

  const handleMovePlayed = () => {
    setMovesPlayed(prev => prev + 1);
  };

  const handleProgressUpdate = (progress: number) => {
    setLessonProgress(progress);
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Beginner': return '#10b981';
      case 'Intermediate': return '#f59e0b';
      case 'Advanced': return '#ef4444';
      default: return '#6b7280';
    }
  };

  return (
    <div className="chess-tutorial">
      <div className="site-header">
        <div className="header-content">
          <Crown size={48} />
          <div>
            <h1>Chess Master Academy</h1>
            <p>Master the game of kings with interactive tutorials</p>
          </div>
        </div>
      </div>

      <div className="content-layout">
        <div className="lessons-sidebar">
          <h2>Course Curriculum</h2>
          <div className="lessons-list">
            {lessons.map(lesson => (
              <div
                key={lesson.id}
                className={`lesson-card ${currentLesson?.id === lesson.id ? 'active' : ''}`}
                onClick={() => handleLessonSelect(lesson)}
              >
                <div className="lesson-header">
                  <span
                    className="difficulty-badge"
                    style={{ backgroundColor: getDifficultyColor(lesson.difficulty) }}
                  >
                    {lesson.difficulty}
                  </span>
                  <span className="lesson-time">{lesson.estimatedTime} min</span>
                </div>
                <h3>{lesson.title}</h3>
                <p>{lesson.description}</p>
                {lesson.completed && (
                  <div className="completion-badge">
                    <Trophy size={14} />
                    Completed
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="lesson-viewer">
          {currentLesson ? (
            <div className="lesson-content">
              <div className="lesson-header">
                <div className="lesson-info">
                  <span
                    className="difficulty-badge"
                    style={{ backgroundColor: getDifficultyColor(currentLesson.difficulty) }}
                  >
                    {currentLesson.difficulty}
                  </span>
                  <h1>{currentLesson.title}</h1>
                  <p>{currentLesson.description}</p>
                </div>
                <div className="lesson-stats">
                  <div className="stat">
                    <Target size={16} />
                    <span>{movesPlayed} moves played</span>
                  </div>
                  <div className="stat">
                    <BookOpen size={16} />
                    <span>{lessonProgress}% complete</span>
                  </div>
                </div>
              </div>

              <div className="lesson-body">
                <div className="chess-board-container">
                  <div className="chess-board">
                    {/* Simple 8x8 chess board representation */}
                    {Array.from({ length: 64 }, (_, i) => {
                      const row = Math.floor(i / 8);
                      const col = i % 8;
                      const isLight = (row + col) % 2 === 0;
                      return (
                        <div
                          key={i}
                          className={`chess-square ${isLight ? 'light' : 'dark'}`}
                          onClick={handleMovePlayed}
                        >
                          {/* Add some mock pieces for visual interest */}
                          {(row === 0 && col === 0) && <span className="piece">♜</span>}
                          {(row === 0 && col === 7) && <span className="piece">♜</span>}
                          {(row === 7 && col === 0) && <span className="piece">♖</span>}
                          {(row === 7 && col === 7) && <span className="piece">♖</span>}
                          {(row === 0 && col === 1) && <span className="piece">♞</span>}
                          {(row === 0 && col === 6) && <span className="piece">♞</span>}
                          {(row === 7 && col === 1) && <span className="piece">♘</span>}
                          {(row === 7 && col === 6) && <span className="piece">♘</span>}
                        </div>
                      );
                    })}
                  </div>
                  <p className="board-instruction">Click on squares to practice moves!</p>
                </div>

                <div className="lesson-text">
                  <h3>Key Concepts</h3>

                  {currentLesson.id === '1' && (
                    <div className="lesson-section">
                      <h4>Pawn Movement Rules</h4>
                      <ul>
                        <li>Pawns move forward one square at a time</li>
                        <li>On their first move, pawns can move two squares forward</li>
                        <li>Pawns capture diagonally one square forward</li>
                        <li>Pawns cannot move backward</li>
                      </ul>
                      <button
                        className="practice-button"
                        onClick={() => handleProgressUpdate(Math.min(lessonProgress + 25, 100))}
                      >
                        Practice Pawn Moves
                      </button>
                    </div>
                  )}

                  {currentLesson.id === '2' && (
                    <div className="lesson-section">
                      <h4>Knight Movement Patterns</h4>
                      <ul>
                        <li>Knights move in an L-shape: 2 squares in one direction, then 1 perpendicular</li>
                        <li>Knights can jump over other pieces</li>
                        <li>Forking occurs when a knight attacks two pieces simultaneously</li>
                        <li>Knights are most powerful in the center of the board</li>
                      </ul>
                      <button
                        className="practice-button"
                        onClick={() => handleProgressUpdate(Math.min(lessonProgress + 20, 100))}
                      >
                        Practice Knight Moves
                      </button>
                    </div>
                  )}

                  {currentLesson.id === '3' && (
                    <div className="lesson-section">
                      <h4>Rook and Queen Coordination</h4>
                      <ul>
                        <li>Rooks control entire ranks and files when unobstructed</li>
                        <li>Queens combine the power of rooks and bishops</li>
                        <li>Open files are crucial for rook activity</li>
                        <li>Queen and rook batteries can be devastating</li>
                      </ul>
                      <button
                        className="practice-button"
                        onClick={() => handleProgressUpdate(Math.min(lessonProgress + 15, 100))}
                      >
                        Practice Coordination
                      </button>
                    </div>
                  )}

                  {currentLesson.id === '4' && (
                    <div className="lesson-section">
                      <h4>Endgame Mastery</h4>
                      <ul>
                        <li>King activity becomes crucial in the endgame</li>
                        <li>Passed pawns must be advanced aggressively</li>
                        <li>Opposition is key in king and pawn endings</li>
                        <li>Convert material advantages systematically</li>
                      </ul>
                      <button
                        className="practice-button"
                        onClick={() => handleProgressUpdate(Math.min(lessonProgress + 10, 100))}
                      >
                        Practice Endgames
                      </button>
                    </div>
                  )}
                </div>

                <div className="ad-placeholder">
                  <div className="ad-content">
                    <h4>Strategic Thinking Tools</h4>
                    <p>Enhance your chess skills with AI-powered analysis and personalized training plans.</p>
                    <button className="ad-button">Explore Chess Analytics</button>
                  </div>
                </div>

                <div className="progress-section">
                  <h3>Lesson Progress</h3>
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${lessonProgress}%` }}
                    />
                  </div>
                  <p>{lessonProgress}% complete • {movesPlayed} practice moves</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="no-lesson">
              <BookOpen size={64} />
              <h2>Select a Lesson</h2>
              <p>Choose from our interactive chess tutorials to begin your journey to mastery.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChessTutorial;