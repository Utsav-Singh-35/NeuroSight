import { useState, useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import footerBg from '../../assets/footer.png'
import styles from './ConnectSection.module.css'

gsap.registerPlugin(ScrollTrigger)

export default function ConnectSection() {
  const [email, setEmail] = useState('')
  const sectionRef = useRef(null)
  const topRef = useRef(null)
  const footerRef = useRef(null)

  const handleSubmit = (e) => {
    e.preventDefault()
    // TODO: integrate EmailJS or backend
    console.log('Submitted:', email)
    setEmail('')
  }

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Animate connect heading and top section reveal
      gsap.from(topRef.current, {
        scrollTrigger: {
          trigger: sectionRef.current,
          start: 'top 80%',
        },
        y: 40,
        opacity: 0,
        duration: 0.8,
        ease: 'power3.out',
      })

      // Stagger footer columns reveal
      gsap.from(footerRef.current.children, {
        scrollTrigger: {
          trigger: sectionRef.current,
          start: 'top 65%',
        },
        y: 40,
        opacity: 0,
        duration: 0.8,
        stagger: 0.2,
        ease: 'power3.out',
      })
    }, sectionRef)

    return () => ctx.revert()
  }, [])

  return (
    <section
      id="contact"
      ref={sectionRef}
      className={styles.connectSection}
      style={{ backgroundImage: `url(${footerBg})` }}
    >
      <div className={styles.overlay} />

      <div className={styles.inner}>
        <div ref={topRef} className={styles.connectTop}>
          <p className={styles.connectEyebrow}>INTERESTED IN LEARNING MORE?</p>
          <div className={styles.connectHeadingRow}>
            <h2 className={styles.connectHeading}>Connect with the team</h2>
            <span className={styles.connectArrow}>↗</span>
          </div>
          <div className={styles.connectDivider} />
        </div>

        <div ref={footerRef} className={styles.connectFooter}>
          {/* Brand */}
          <div className={styles.connectBrand}>
            <div className={styles.connectLogo}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <circle cx="12" cy="12" r="10"/>
                <path d="M12 2a10 10 0 0 1 0 20M12 2a10 10 0 0 0 0 20M2 12h20"/>
              </svg>
              <span>NeuraSight</span>
            </div>
            <p className={styles.connectBrandSub}>AI-POWERED MEDICAL IMAGING PLATFORM</p>
            <p className={styles.connectBrandDesc}>
              Advanced brain MRI analysis powered by deep learning, explainable AI, and clinical decision support systems.
            </p>
          </div>

          {/* Contact col */}
          <div className={styles.connectCol}>
            <h4 className={styles.connectColTitle}>CONTACT</h4>
            <ul className={styles.connectLinks}>
              <li><a href="research.html">Research <span>→</span></a></li>
              <li><a href="https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset">Dataset <span>→</span></a></li>
              <li><a href="https://github.com/Utsav-Singh-35">GitHub <span>→</span></a></li>
            </ul>
          </div>

          {/* Stay in touch col */}
          <div className={styles.connectCol}>
            <h4 className={styles.connectColTitle}>STAY IN TOUCH</h4>
            <form className={styles.connectForm} onSubmit={handleSubmit}>
              <div className={styles.connectInputRow}>
                <input
                  type="email"
                  placeholder="Enter your email"
                  className={styles.connectInput}
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  required
                />
                <button className={styles.connectSubmit} type="submit">→</button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </section>
  )
}
