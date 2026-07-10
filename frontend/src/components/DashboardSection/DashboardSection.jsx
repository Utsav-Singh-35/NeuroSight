import { useEffect, useRef, useState } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import styles from './DashboardSection.module.css'

gsap.registerPlugin(ScrollTrigger)

export default function DashboardSection() {
  const sectionRef = useRef(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [showResults, setShowResults] = useState(false)

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.from(`.${styles.dashboardCard}`, {
        scrollTrigger: {
          trigger: sectionRef.current,
          start: 'top 75%',
        },
        y: 50,
        opacity: 0,
        duration: 0.9,
        stagger: 0.25,
        ease: 'power3.out',
      })
    }, sectionRef)
    return () => ctx.revert()
  }, [])

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file.name)
      setShowResults(false)
    }
  }

  const handleAnalyze = () => {
    if (!selectedFile) return
    setIsProcessing(true)
    setShowResults(false)
    
    // Simulate processing
    setTimeout(() => {
      setIsProcessing(false)
      setShowResults(true)
    }, 3000)
  }

  return (
    <section id="dashboard" ref={sectionRef} className={styles.dashboardSection}>
      <div className={styles.header}>
        <p className={styles.eyebrow}>INTERACTIVE DIAGNOSTIC TOOL</p>
        <h2 className={styles.title}>MRI Analysis Dashboard</h2>
        <p className={styles.subtitle}>
          Upload a brain MRI scan and receive AI-powered predictions with explainable insights.
        </p>
      </div>

      <div className={styles.dashboardGrid}>
        {/* Upload Card */}
        <div className={`${styles.dashboardCard} ${styles.uploadCard}`}>
          <div className={styles.cardIcon}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
          </div>
          <h3 className={styles.cardTitle}>Upload MRI Scan</h3>
          <p className={styles.cardDesc}>Supported Formats</p>
          <div className={styles.formatBadges}>
            <span className={styles.badge}>PNG</span>
            <span className={styles.badge}>JPG</span>
            <span className={styles.badge}>JPEG</span>
          </div>
          <div className={styles.uploadZone}>
            <input
              type="file"
              id="mri-upload"
              accept=".png,.jpg,.jpeg"
              onChange={handleFileSelect}
              className={styles.fileInput}
            />
            <label htmlFor="mri-upload" className={styles.uploadLabel}>
              {selectedFile ? (
                <>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                  <span>{selectedFile}</span>
                </>
              ) : (
                <>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                  </svg>
                  <span>Choose File</span>
                </>
              )}
            </label>
          </div>
          <button 
            className={styles.analyzeBtn}
            onClick={handleAnalyze}
            disabled={!selectedFile || isProcessing}
          >
            {isProcessing ? 'Analyzing...' : 'Analyze MRI'}
          </button>
        </div>

        {/* Processing Card */}
        <div className={`${styles.dashboardCard} ${styles.processingCard} ${isProcessing ? styles.active : ''}`}>
          <div className={styles.cardIcon}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/>
              <polyline points="12 6 12 12 16 14"/>
            </svg>
          </div>
          <h3 className={styles.cardTitle}>AI Processing</h3>
          <p className={styles.cardDesc}>Analysis Stages</p>
          <div className={styles.stagesList}>
            <div className={`${styles.stage} ${isProcessing ? styles.stageActive : ''}`}>
              <span className={styles.stageDot}></span>
              <span>Image Validation</span>
            </div>
            <div className={`${styles.stage} ${isProcessing ? styles.stageActive : ''}`}>
              <span className={styles.stageDot}></span>
              <span>Preprocessing</span>
            </div>
            <div className={`${styles.stage} ${isProcessing ? styles.stageActive : ''}`}>
              <span className={styles.stageDot}></span>
              <span>Feature Extraction</span>
            </div>
            <div className={`${styles.stage} ${isProcessing ? styles.stageActive : ''}`}>
              <span className={styles.stageDot}></span>
              <span>Classification</span>
            </div>
            <div className={`${styles.stage} ${isProcessing ? styles.stageActive : ''}`}>
              <span className={styles.stageDot}></span>
              <span>Heatmap Generation</span>
            </div>
            <div className={`${styles.stage} ${isProcessing ? styles.stageActive : ''}`}>
              <span className={styles.stageDot}></span>
              <span>Report Generation</span>
            </div>
          </div>
        </div>

        {/* Results Card */}
        <div className={`${styles.dashboardCard} ${styles.resultsCard} ${showResults ? styles.active : ''}`}>
          <div className={styles.cardIcon}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
          </div>
          <h3 className={styles.cardTitle}>Prediction Results</h3>
          
          {showResults ? (
            <>
              <div className={styles.predictionMain}>
                <div className={styles.predictionLabel}>Prediction</div>
                <div className={styles.predictionValue}>Glioma</div>
              </div>
              
              <div className={styles.metricsGrid}>
                <div className={styles.metric}>
                  <span className={styles.metricLabel}>Confidence</span>
                  <span className={styles.metricValue}>97.3%</span>
                </div>
                <div className={styles.metric}>
                  <span className={styles.metricLabel}>Risk Level</span>
                  <span className={`${styles.metricValue} ${styles.riskHigh}`}>High</span>
                </div>
              </div>

              <div className={styles.distributionSection}>
                <p className={styles.distributionTitle}>Probability Distribution</p>
                <div className={styles.probBar}>
                  <div className={styles.probLabel}>
                    <span>Glioma</span>
                    <span className={styles.probPercent}>97%</span>
                  </div>
                  <div className={styles.probBarBg}>
                    <div className={styles.probBarFill} style={{ width: '97%' }}></div>
                  </div>
                </div>
                <div className={styles.probBar}>
                  <div className={styles.probLabel}>
                    <span>Meningioma</span>
                    <span className={styles.probPercent}>1%</span>
                  </div>
                  <div className={styles.probBarBg}>
                    <div className={styles.probBarFill} style={{ width: '1%' }}></div>
                  </div>
                </div>
                <div className={styles.probBar}>
                  <div className={styles.probLabel}>
                    <span>Pituitary</span>
                    <span className={styles.probPercent}>1%</span>
                  </div>
                  <div className={styles.probBarBg}>
                    <div className={styles.probBarFill} style={{ width: '1%' }}></div>
                  </div>
                </div>
                <div className={styles.probBar}>
                  <div className={styles.probLabel}>
                    <span>No Tumor</span>
                    <span className={styles.probPercent}>1%</span>
                  </div>
                  <div className={styles.probBarBg}>
                    <div className={styles.probBarFill} style={{ width: '1%' }}></div>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className={styles.placeholderText}>
              <p>Results will appear here after analysis</p>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
