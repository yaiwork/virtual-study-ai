import React, { useState } from 'react';

const BACKEND_URL = 'https://virtual-study-ai.onrender.com';
const grades = Array.from({ length: 8 }, (_, i) => (i + 5).toString());
const subjects = ['Biology', 'Physics', 'Chemistry', 'Mathematics', 'English', 'Economics'];

export default function ExamGenerator() {
  const [subject, setSubject] = useState(subjects[0]);
  const [grade, setGrade] = useState(grades[0]);
  const [topic, setTopic] = useState('');
  const [numQuestions, setNumQuestions] = useState(5);
  const [questions, setQuestions] = useState('');

  const submit = async () => {
    if (!topic) return;
    try {
      const res = await fetch(`${BACKEND_URL}/generate-exam`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject, grade, topic, num_questions: numQuestions })
      });
      const data = await res.json();
      setQuestions(data.questions || 'No response');
    } catch (err) {
      setQuestions('Error: ' + err);
    }
  };

  return (
    <div>
      <select value={subject} onChange={e => setSubject(e.target.value)}>
        {subjects.map(s => <option key={s} value={s}>{s}</option>)}
      </select>
      <select value={grade} onChange={e => setGrade(e.target.value)}>
        {grades.map(g => <option key={g} value={g}>{g}</option>)}
      </select>
      <input placeholder="Topic" value={topic} onChange={e => setTopic(e.target.value)} />
      <input
        type="number"
        min="1"
        max="20"
        value={numQuestions}
        onChange={e => setNumQuestions(e.target.value)}
      />
      <button onClick={submit}>Generate Questions</button>
      {questions && <div className="result" dangerouslySetInnerHTML={{__html: questions}} />}
    </div>
  );
}
