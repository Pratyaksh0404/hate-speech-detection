import { useState } from 'react'
import { API_BASE_URL } from '../config'
import ResultLabel from './ResultLabel'
import ConfidenceBars from './ConfidenceBars'
import HighlightedText from './HighlightedText'

// Deliberately mild, non-disturbing examples -- good demo UX without
// showing graphic content to a casual visitor (master_plan.md section 8).
const EXAMPLE_TEXTS = [
  'Have a wonderful day, everyone!',
  "You're kind of an idiot sometimes.",
  "I really appreciate how kind you've been to me.",
]

export default function TextMode() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit() {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const response = await fetch(`${API_BASE_URL}/predict/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Request failed (${response.status})`)
      }
      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Something went wrong. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mode-panel">
      <div className="example-row">
        <span className="example-label">Try an example:</span>
        {EXAMPLE_TEXTS.map((example) => (
          <button key={example} className="example-chip" onClick={() => setText(example)}>
            {example.length > 30 ? example.slice(0, 30) + '…' : example}
          </button>
        ))}
      </div>

      <textarea
        className="text-input"
        placeholder="Type or paste some text to analyze..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={4}
      />

      <button className="submit-button" onClick={handleSubmit} disabled={loading || !text.trim()}>
        {loading ? 'Analyzing...' : 'Analyze Text'}
      </button>

      {error && <div className="error-box">{error}</div>}

      {result && (
        <div className="result-card">
          <ResultLabel label={result.label} confidence={result.confidence} />
          <ConfidenceBars probabilities={result.per_class_probabilities} />

          <div className="highlighted-section">
            <h3>Which words mattered most</h3>
            <p className="legend">
              <span className="legend-swatch" style={{ backgroundColor: 'rgba(220,38,38,0.6)' }} />
              pushes toward &quot;{result.label}&quot;
              <span className="legend-swatch" style={{ backgroundColor: 'rgba(22,163,74,0.6)', marginLeft: 16 }} />
              pushes away
            </p>
            <HighlightedText text={text} tokens={result.top_contributing_tokens} />
          </div>
        </div>
      )}
    </div>
  )
}
