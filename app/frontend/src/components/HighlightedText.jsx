function getHighlightStyle(weight) {
  if (weight === undefined) return {}
  const intensity = Math.min(Math.abs(weight) * 2, 1) // scale small LIME weights for visibility
  const color =
    weight > 0
      ? `rgba(220, 38, 38, ${intensity})` // red: pushes toward the predicted class
      : `rgba(22, 163, 74, ${intensity})` // green: pushes away from it
  return { backgroundColor: color, borderRadius: '3px', padding: '0 2px' }
}

export default function HighlightedText({ text, tokens }) {
  const weightMap = {}
  tokens.forEach(({ token, weight }) => {
    weightMap[token.toLowerCase()] = weight
  })

  // split on whitespace but KEEP it as separate array entries, so spacing
  // in the rendered output matches the original text exactly
  const words = text.split(/(\s+)/)

  return (
    <p className="highlighted-text">
      {words.map((word, idx) => {
        const cleaned = word.toLowerCase().replace(/[^a-z0-9']/g, '')
        const weight = weightMap[cleaned]
        return (
          <span key={idx} style={getHighlightStyle(weight)}>
            {word}
          </span>
        )
      })}
    </p>
  )
}
