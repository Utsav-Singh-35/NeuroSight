import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import styles from './StatementSection.module.css'

gsap.registerPlugin(ScrollTrigger)

const WORDS = [
  { text: 'Built', type: 'word' },
  { text: 'with', type: 'word' },
  { text: 'precision', type: 'word' },
  { text: '—', type: 'pill-wrap' },
  { text: 'advanced', type: 'word' },
  { text: 'brain', type: 'word' },
  { text: 'MRI', type: 'word' },
  { text: 'analysis', type: 'word' },
  { type: 'pill-silver' },
  { text: 'powered', type: 'word' },
  { text: 'by', type: 'word' },
  { text: 'EfficientNet,', type: 'word' },
  { text: 'explainable', type: 'word' },
  { text: 'AI,', type: 'word' },
  { text: 'and', type: 'word' },
  { text: 'intelligent', type: 'word' },
  { type: 'pill-glow' },
  { text: 'diagnostic', type: 'word' },
  { text: 'workflows', type: 'word' },
  { type: 'avatars' },
  { text: 'designed', type: 'word' },
  { text: 'to', type: 'word' },
  { text: 'support', type: 'word' },
  { text: 'clinical', type: 'word' },
  { text: 'decision-making.', type: 'word' },
]

export default function StatementSection() {
  const wordRefs = useRef([])

  useEffect(() => {
    const ctx = gsap.context(() => {
      wordRefs.current.forEach((el) => {
        if (!el) return
        gsap.to(el, {
          scrollTrigger: {
            trigger: el,
            start: 'top 80%',
            end: 'top 50%',
            scrub: true,
          },
          color: '#ffffff',
          duration: 0.5,
          ease: 'none',
        })
      })
    })
    return () => ctx.revert()
  }, [])

  let refIdx = 0

  return (
    <section id="about" className={styles.statementSection}>
      <div className={styles.statementContainer}>
        <p className={styles.statementText}>
          {WORDS.map((w, i) => {
            if (w.type === 'pill-wrap') {
              return <span key={i} className={`${styles.word} ${styles.pillWrap}`} ref={el => wordRefs.current[refIdx++] = el}>{w.text}</span>
            }
            if (w.type === 'pill-silver') {
              return (
                <span key={i} className={`${styles.word} ${styles.pill} ${styles.pillSilver}`} ref={el => wordRefs.current[refIdx++] = el}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/></svg>
                </span>
              )
            }
            if (w.type === 'pill-glow') {
              return (
                <span key={i} className={`${styles.word} ${styles.pill} ${styles.pillGlow}`} ref={el => wordRefs.current[refIdx++] = el}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
                </span>
              )
            }
            if (w.type === 'avatars') {
              return (
                <span key={i} className={`${styles.word} ${styles.avatarStack}`} ref={el => wordRefs.current[refIdx++] = el}>
                  <img src="https://i.pravatar.cc/40?img=1" alt="" />
                  <img src="https://i.pravatar.cc/40?img=5" alt="" />
                  <img src="https://i.pravatar.cc/40?img=9" alt="" />
                </span>
              )
            }
            return <span key={i} className={styles.word} ref={el => wordRefs.current[refIdx++] = el}>{w.text}</span>
          })}
        </p>
      </div>
    </section>
  )
}
