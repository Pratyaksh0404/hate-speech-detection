export default function TabBar({ activeTab, onChange }) {
  return (
    <div className="tab-bar">
      <button
        className={`tab-button ${activeTab === 'text' ? 'active' : ''}`}
        onClick={() => onChange('text')}
      >
        📝 Text
      </button>
      <button
        className={`tab-button ${activeTab === 'image' ? 'active' : ''}`}
        onClick={() => onChange('image')}
      >
        🖼️ Image
      </button>
    </div>
  )
}
