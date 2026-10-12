const LABEL_COLORS = {
  'Hate Speech': '#dc2626',
  'Offensive Language': '#f59e0b',
  Neither: '#16a34a',
}

export default function ConfidenceBars({ probabilities }) {
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1])

  return (
    <div className="confidence-bars">
      {entries.map(([label, prob]) => (
        <div className="confidence-row" key={label}>
          <div className="confidence-label">{label}</div>
          <div className="confidence-track">
            <div
              className="confidence-fill"
              style={{
                width: `${(prob * 100).toFixed(1)}%`,
                backgroundColor: LABEL_COLORS[label] || '#6b7280',
              }}
            />
          </div>
          <div className="confidence-value">{(prob * 100).toFixed(1)}%</div>
        </div>
      ))}
    </div>
  )
}
