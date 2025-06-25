import React, { useState } from 'react';
import Chat from './components/Chat';
import HomeworkAssistant from './components/HomeworkAssistant';
import LessonPlanGenerator from './components/LessonPlanGenerator';
import ExamGenerator from './components/ExamGenerator';
import StudyGuideGenerator from './components/StudyGuideGenerator';

const modes = [
  'Chat',
  'Homework Assistant',
  'Lesson Plan Generator',
  'Exam Questions Generator',
  'Study Guide Generator'
];

export default function App() {
  const [mode, setMode] = useState('Chat');

  return (
    <div className="app">
      <h1>Virtual Study AI Tutor Assistant</h1>
      <select value={mode} onChange={e => setMode(e.target.value)}>
        {modes.map(m => (
          <option key={m} value={m}>{m}</option>
        ))}
      </select>
      <div className="mode-container">
        {mode === 'Chat' && <Chat />}
        {mode === 'Homework Assistant' && <HomeworkAssistant />}
        {mode === 'Lesson Plan Generator' && <LessonPlanGenerator />}
        {mode === 'Exam Questions Generator' && <ExamGenerator />}
        {mode === 'Study Guide Generator' && <StudyGuideGenerator />}
      </div>
    </div>
  );
}
