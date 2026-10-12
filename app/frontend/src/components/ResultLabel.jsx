const LABEL_COLORS = {
  'Hate Speech': '#dc2626',
  'Offensive Language': '#f59e0b',
  Neither: '#16a34a',
}

export default function ResultLabel({ label, confidence }) {
  return (
    <div className="result-label-row">
      <span className="result-badge" style={{ backgroundColor: LABEL_COLORS[label] || '#6b7280' }}>
        {label}
      </span>
      <span className="result-confidence">{(confidence * 100).toFixed(1)}% confidence</span>
    </div>
  )
}
