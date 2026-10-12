import { useState } from 'react'
import Header from './components/Header'
import TabBar from './components/TabBar'
import TextMode from './components/TextMode'
import ImageMode from './components/ImageMode'

export default function App() {
  const [activeTab, setActiveTab] = useState('text')

  return (
    <div className="app-container">
      <Header />
      <TabBar activeTab={activeTab} onChange={setActiveTab} />
      {activeTab === 'text' ? <TextMode /> : <ImageMode />}
    </div>
  )
}
