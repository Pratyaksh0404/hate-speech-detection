export default function Header() {
  return (
    <header className="app-header">
      <h1>Hate Speech Detection</h1>
      <p className="disclaimer-banner">
        <strong>⚠️ Heads up:</strong> this is a content-moderation-<strong>assistance</strong> research/portfolio
        project, not a "hate speech accusation machine." Predictions are probabilistic, can be
        biased (see our bias audit in the project repository), and must not be used as the sole
        basis for real moderation decisions against real people.
      </p>
    </header>
  )
}
