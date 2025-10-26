import React, { useState, useEffect, useMemo } from 'react';
import { usePrivAds } from '../components/PrivAdsProvider';
import { BookOpen, Play, CheckCircle, Star, Users, Clock } from 'lucide-react';
import './EducationalPlatform.css';
import CustomizedAd from '../components/CustomizedAd';

interface Course {
  id: string;
  title: string;
  instructor: string;
  category: 'Programming' | 'Design' | 'Business' | 'Science' | 'Language';
  duration: number;
  rating: number;
  students: number;
  level: 'Beginner' | 'Intermediate' | 'Advanced';
  thumbnail: string;
  progress: number;
}

const EducationalPlatform: React.FC = () => {
  const { updateSiteContext, updatePrediction } = usePrivAds();
  const [currentCourse, setCurrentCourse] = useState<Course | null>(null);
  const [courseProgress, setCourseProgress] = useState(0);
  const [videosWatched, setVideosWatched] = useState(0);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [enrolledCourses, setEnrolledCourses] = useState<string[]>([]);

  const courses: Course[] = [
    {
      id: '1',
      title: 'Introduction to Machine Learning',
      instructor: 'Dr. Sarah Chen',
      category: 'Programming',
      duration: 120,
      rating: 4.8,
      students: 15420,
      level: 'Intermediate',
      thumbnail: '🤖',
      progress: 0
    },
    {
      id: '2',
      title: 'UX/UI Design Fundamentals',
      instructor: 'Alex Rivera',
      category: 'Design',
      duration: 90,
      rating: 4.9,
      students: 12350,
      level: 'Beginner',
      thumbnail: '🎨',
      progress: 0
    },
    {
      id: '3',
      title: 'Digital Marketing Mastery',
      instructor: 'Maria Gonzalez',
      category: 'Business',
      duration: 150,
      rating: 4.7,
      students: 18750,
      level: 'Intermediate',
      thumbnail: '📈',
      progress: 0
    },
    {
      id: '4',
      title: 'Quantum Physics Explained',
      instructor: 'Prof. David Kim',
      category: 'Science',
      duration: 180,
      rating: 4.6,
      students: 8950,
      level: 'Advanced',
      thumbnail: '⚛️',
      progress: 0
    },
    {
      id: '5',
      title: 'Spanish Conversation Skills',
      instructor: 'Carmen Rodriguez',
      category: 'Language',
      duration: 100,
      rating: 4.8,
      students: 22100,
      level: 'Beginner',
      thumbnail: '🇪🇸',
      progress: 0
    }
  ];

  const categories = [
    { id: 'all', name: 'All Courses', color: '#6b7280' },
    { id: 'Programming', name: 'Programming', color: '#3b82f6' },
    { id: 'Design', name: 'Design', color: '#8b5cf6' },
    { id: 'Business', name: 'Business', color: '#10b981' },
    { id: 'Science', name: 'Science', color: '#f59e0b' },
    { id: 'Language', name: 'Language', color: '#ef4444' }
  ];

  const pageContext = useMemo(() => ({
    title: 'Learnify - Online Learning Platform',
    content: 'Expert-led courses in programming, design, business, science, and languages',
    keywords: ['education', 'learning', 'courses', 'programming', 'design', 'business'],
    page_type: 'education',
    url: window.location.href,
    summary_text: 'Unlock your potential with expert-led courses and personalized learning',
  }), []);

  useEffect(() => {
    // Update site context for educational platform
    updateSiteContext({
      siteType: 'education',
      contentType: 'online_course',
      currentCourse: currentCourse?.id || null,
      selectedCategory,
      courseProgress,
      videosWatched,
      enrolledCourses: enrolledCourses.length,
      timeSpent: courseProgress * 5, // Mock time spent
      courseCategory: currentCourse?.category || 'general',
      courseLevel: currentCourse?.level || 'general'
    });

    // Update prediction based on learning engagement
    const basePrediction = 45; // Educational platforms generally receptive
    const progressBonus = courseProgress * 0.4; // More progress = more engaged
    const enrollmentBonus = enrolledCourses.length * 10; // More enrolled = more committed
    const interactionBonus = videosWatched * 5; // Video watching indicates engagement
    const categoryBonus = currentCourse?.category === 'Programming' ? 20 : 0; // Tech courses highly receptive

    updatePrediction({
      probability: Math.min(basePrediction + progressBonus + enrollmentBonus + interactionBonus + categoryBonus, 95),
      factors: {
        contextRelevance: 70 + (courseProgress * 0.3),
        userPreferences: 75 + categoryBonus,
        contentType: 90, // Educational content very receptive
        interactionHistory: courseProgress * 0.4 + (videosWatched * 0.6) + (enrolledCourses.length * 2)
      }
    });
  }, [currentCourse, courseProgress, videosWatched, selectedCategory, enrolledCourses, updateSiteContext, updatePrediction]);

  const handleCourseSelect = (course: Course) => {
    setCurrentCourse(course);
    setCourseProgress(0);
    setVideosWatched(0);
  };

  const handleEnroll = (courseId: string) => {
    if (!enrolledCourses.includes(courseId)) {
      setEnrolledCourses(prev => [...prev, courseId]);
    }
  };

  const handleVideoWatch = () => {
    setVideosWatched(prev => prev + 1);
    setCourseProgress(prev => Math.min(prev + Math.random() * 15 + 5, 100));
  };

  const filteredCourses = selectedCategory === 'all'
    ? courses
    : courses.filter(course => course.category === selectedCategory);

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'Beginner': return '#10b981';
      case 'Intermediate': return '#f59e0b';
      case 'Advanced': return '#ef4444';
      default: return '#6b7280';
    }
  };

  return (
    <div className="educational-platform">
      <div className="site-header">
        <div className="header-content">
          <BookOpen size={48} />
          <div>
            <h1>Learnify</h1>
            <p>Unlock your potential with expert-led courses</p>
          </div>
        </div>
      </div>

      <div className="content-layout">
        <div className="sidebar">
          <div className="category-filter">
            <h3>Course Categories</h3>
            <div className="category-buttons">
              {categories.map(category => (
                <button
                  key={category.id}
                  className={`category-btn ${selectedCategory === category.id ? 'active' : ''}`}
                  onClick={() => setSelectedCategory(category.id)}
                  style={{
                    borderColor: selectedCategory === category.id ? category.color : 'transparent'
                  }}
                >
                  {category.name}
                </button>
              ))}
            </div>
          </div>

          <div className="learning-stats">
            <h3>Your Learning</h3>
            <div className="stats-grid">
              <div className="stat-item">
                <BookOpen size={20} />
                <div>
                  <span className="stat-number">{enrolledCourses.length}</span>
                  <span className="stat-label">Enrolled</span>
                </div>
              </div>
              <div className="stat-item">
                <Play size={20} />
                <div>
                  <span className="stat-number">{videosWatched}</span>
                  <span className="stat-label">Videos</span>
                </div>
              </div>
              <div className="stat-item">
                <CheckCircle size={20} />
                <div>
                  <span className="stat-number">{Math.round(courseProgress)}%</span>
                  <span className="stat-label">Progress</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="main-content">
          {!currentCourse ? (
            <div className="courses-grid">
              <div className="courses-header">
                <h2>Featured Courses</h2>
                <p>Discover new skills and advance your career</p>
              </div>

              <div className="courses-list">
                {filteredCourses.map(course => (
                  <div key={course.id} className="course-card">
                    <div className="course-thumbnail">
                      <span className="thumbnail-emoji">{course.thumbnail}</span>
                    </div>

                    <div className="course-content">
                      <div className="course-header">
                        <span
                          className="level-badge"
                          style={{ backgroundColor: getLevelColor(course.level) }}
                        >
                          {course.level}
                        </span>
                        <span className="category-tag">{course.category}</span>
                      </div>

                      <h3>{course.title}</h3>
                      <p className="instructor">by {course.instructor}</p>

                      <div className="course-meta">
                        <div className="rating">
                          <Star size={14} fill="#fbbf24" color="#fbbf24" />
                          <span>{course.rating}</span>
                        </div>
                        <div className="students">
                          <Users size={14} />
                          <span>{course.students.toLocaleString()}</span>
                        </div>
                        <div className="duration">
                          <Clock size={14} />
                          <span>{course.duration}h</span>
                        </div>
                      </div>

                      <div className="course-actions">
                        <button
                          className="enroll-btn"
                          onClick={() => handleEnroll(course.id)}
                          disabled={enrolledCourses.includes(course.id)}
                        >
                          {enrolledCourses.includes(course.id) ? 'Enrolled' : 'Enroll Now'}
                        </button>
                        <button
                          className="preview-btn"
                          onClick={() => handleCourseSelect(course)}
                        >
                          Preview
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="course-detail">
              <div className="course-header">
                <div className="course-info">
                  <span
                    className="level-badge"
                    style={{ backgroundColor: getLevelColor(currentCourse.level) }}
                  >
                    {currentCourse.level}
                  </span>
                  <h1>{currentCourse.title}</h1>
                  <p className="instructor">Instructor: {currentCourse.instructor}</p>
                  <div className="course-stats">
                    <span>📊 {currentCourse.students.toLocaleString()} students</span>
                    <span>⭐ {currentCourse.rating} rating</span>
                    <span>⏱️ {currentCourse.duration} hours</span>
                  </div>
                </div>

                <div className="course-progress">
                  <div className="progress-info">
                    <span>Your Progress</span>
                    <span>{Math.round(courseProgress)}%</span>
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${courseProgress}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className="course-content">
                <div className="video-section">
                  <h3>Course Videos</h3>
                  <div className="video-grid">
                    {Array.from({ length: 6 }, (_, i) => (
                      <div key={i} className="video-item">
                        <div className="video-thumbnail">
                          <Play size={24} />
                          <span className="video-duration">10:30</span>
                        </div>
                        <div className="video-info">
                          <h4>Lesson {i + 1}: {currentCourse.title} - Part {i + 1}</h4>
                          <p>Learn the fundamentals and build your skills step by step.</p>
                          <button
                            className="watch-btn"
                            onClick={handleVideoWatch}
                          >
                            Watch Now
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* <div className="ad-placeholder">
                  <div className="ad-content">
                    <h4>Advanced Learning Tools</h4>
                    <p>Accelerate your learning with AI-powered study assistants and personalized learning paths.</p>
                    <button className="ad-button">Explore Learning AI</button>
                  </div>
                </div> */}
                <CustomizedAd pageContext={pageContext} />

                <div className="course-resources">
                  <h3>Course Resources</h3>
                  <div className="resources-grid">
                    <div className="resource-item">
                      <BookOpen size={20} />
                      <div>
                        <h4>Course Materials</h4>
                        <p>Download slides, exercises, and reference materials</p>
                      </div>
                    </div>
                    <div className="resource-item">
                      <Users size={20} />
                      <div>
                        <h4>Discussion Forum</h4>
                        <p>Connect with fellow learners and get help</p>
                      </div>
                    </div>
                    <div className="resource-item">
                      <CheckCircle size={20} />
                      <div>
                        <h4>Assignments</h4>
                        <p>Practice exercises and projects</p>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="course-certificate">
                  <h3>Earn Your Certificate</h3>
                  <div className="certificate-preview">
                    <div className="certificate-content">
                      <h4>Certificate of Completion</h4>
                      <p>This certifies that you have successfully completed</p>
                      <h5>{currentCourse.title}</h5>
                      <p>Instructor: {currentCourse.instructor}</p>
                    </div>
                  </div>
                  <p>Complete all lessons and assignments to earn your certificate!</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default EducationalPlatform;