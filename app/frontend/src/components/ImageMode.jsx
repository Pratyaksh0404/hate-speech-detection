import { useState } from 'react'
import { API_BASE_URL } from '../config'
import ResultLabel from './ResultLabel'
import ConfidenceBars from './ConfidenceBars'

export default function ImageMode() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [caption, setCaption] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  function handleFileChange(e) {
    const selected = e.target.files[0]
    if (!selected) return
    setFile(selected)
    setPreviewUrl(URL.createObjectURL(selected))
    setResult(null)
    setError(null)
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
