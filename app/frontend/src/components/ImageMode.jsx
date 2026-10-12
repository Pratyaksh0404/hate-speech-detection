import { useState } from 'react'
import { API_BASE_URL } from '../config'
import ResultLabel from './ResultLabel'
import ConfidenceBars from './ConfidenceBars'

// Click-to-try example images live in public/examples/ (Vite serves
// public/ as static assets at the root). Each entry has a URL path
// (relative to the web root), a display label for the thumbnail caption,
// and a filename used when converting the fetched blob into a File object
// (the backend sees this as the upload filename).
const EXAMPLE_IMAGES = [
  {
    url: '/examples/example_positive.png',
    label: '"Spread kindness"',
    filename: 'example_positive.png',
  },
  {
    url: '/examples/example_rude.png',
    label: '"You are an idiot"',
    filename: 'example_rude.png',
  },
  {
    url: '/examples/example_neutral.png',
    label: '"Code compiles"',
    filename: 'example_neutral.png',
  },
]

export default function ImageMode() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [caption, setCaption] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [exampleLoading, setExampleLoading] = useState(false)

  function handleFileChange(e) {
    const selected = e.target.files[0]
    if (!selected) return
    setFile(selected)
    setPreviewUrl(URL.createObjectURL(selected))
    setResult(null)
    setError(null)
  }

  async function handleExampleClick(example) {
    // Fetch the static image and convert it into a File object so the
    // existing FormData upload flow works unchanged. This is the same
    // technique as drag-and-drop: we get a Blob from fetch(), wrap it
    // in a File constructor, and set it as if the user had chosen it
    // via the file picker.
    setExampleLoading(true)
    setError(null)
    setResult(null)
    try {
      const response = await fetch(example.url)
      if (!response.ok) throw new Error('Failed to load example image')
      const blob = await response.blob()
      const fileObj = new File([blob], example.filename, { type: blob.type })
      setFile(fileObj)
      setPreviewUrl(example.url)
      setCaption('')
    } catch (err) {
      setError(err.message || 'Could not load example image.')
    } finally {
      setExampleLoading(false)
    }
  }

  async function handleSubmit() {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const formData = new FormData()
      formData.append('image', file)
      formData.append('caption', caption)

      const response = await fetch(`${API_BASE_URL}/predict/image`, {
        method: 'POST',
        body: formData,
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
      {/* ── Click-to-try example images ────────────────────────── */}
      <div className="example-images-section">
        <span className="example-label">Try an example:</span>
        <div className="example-images-row">
          {EXAMPLE_IMAGES.map((example) => (
            <button
              key={example.filename}
              className="example-image-card"
              onClick={() => handleExampleClick(example)}
              disabled={exampleLoading || loading}
              title={`Load ${example.label}`}
            >
              <img
                src={example.url}
                alt={example.label}
                className="example-image-thumb"
              />
              <span className="example-image-label">{example.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* ── File upload (existing) ─────────────────────────────── */}
      <div className="upload-row">
        <label className="file-upload-label">
          📁 Choose an image
          <input type="file" accept="image/*" onChange={handleFileChange} hidden />
        </label>
        {previewUrl && <img src={previewUrl} alt="preview" className="preview-thumb" />}
      </div>

      <input
        className="caption-input"
        type="text"
        placeholder="Optional caption (if the image has little/no visible text)"
        value={caption}
        onChange={(e) => setCaption(e.target.value)}
      />

      <button className="submit-button" onClick={handleSubmit} disabled={loading || !file}>
        {loading ? 'Analyzing...' : 'Analyze Image'}
      </button>

      {error && <div className="error-box">{error}</div>}

      {result && (
        <div className="result-card">
          <ResultLabel label={result.label} confidence={result.confidence} />
          <ConfidenceBars probabilities={result.per_class_probabilities} />

          <div className="ocr-section">
            <h3>What the model actually read (OCR)</h3>
            <p className="ocr-text">{result.ocr_extracted_text || '(no text detected)'}</p>
          </div>

          <div className="grad-cam-section">
            <h3>Vision model attention (diagnostic only)</h3>
            <img
              src={`data:image/png;base64,${result.grad_cam_overlay_base64}`}
              alt="Grad-CAM attention overlay"
              className="grad-cam-image"
            />
            <p className="saliency-note">{result.image_saliency_note}</p>
          </div>

          <p className="privacy-note">🔒 {result.images_not_stored_note}</p>
        </div>
      )}
    </div>
  )
}
