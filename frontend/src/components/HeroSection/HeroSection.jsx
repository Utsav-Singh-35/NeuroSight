import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import ballVideo from '../../assets/ball.mp4'
import styles from './HeroSection.module.css'

export default function HeroSection() {
  const containerRef = useRef(null)
  const titleRef = useRef(null)
  const subtitleRef = useRef(null)
  const descriptionRef = useRef(null)
  const videoRef = useRef(null)

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Cinematic landing sequence: start from pure emptiness/void, then orchestrate reveal
      const tl = gsap.timeline({ defaults: { ease: 'power3.out' } })

      // Reveal full container out of complete black void after initial delay
      tl.to(containerRef.current, {
        opacity: 1,
        duration: 0.5,
        delay: 0.4,
      })
      // Smooth scale and fade in the primary title
      .from(titleRef.current, {
        scale: 0.85,
        opacity: 0,
        duration: 1.2,
      }, '-=0.2')
      // Fade in subtitle
      .from(`.${styles.heroSubtitle}`, {
        y: 20,
        opacity: 0,
        duration: 0.8,
      }, '-=0.6')
      // Fade in description
      .from(`.${styles.heroDescription}`, {
        y: 20,
        opacity: 0,
        duration: 0.8,
      }, '-=0.5')
      // Stagger layout indicator brackets
      .from(`.${styles.tag}`, {
        y: 20,
        opacity: 0,
        duration: 0.6,
        stagger: 0.15,
      }, '-=0.8')
      // Soft entrance for the premium 3D sphere background preview
      .from(videoRef.current, {
        y: 40,
        opacity: 0,
        duration: 1.2,
      }, '-=0.5')
    }, containerRef)

    return () => ctx.revert()
  }, [])

  return (
    // Set initial opacity to 0 guaranteeing an absolutely empty viewport before timeline loads
    <section id="home" ref={containerRef} className={styles.hero} style={{ opacity: 0 }}>
      <div className={styles.heroContainer}>
        <h1 ref={titleRef} className={styles.heroTitle}>NeuraSight</h1>

        <div className={styles.heroTags}>
          <div className={`${styles.tag} ${styles.tagTopLeft}`}>[ 97.4% Accuracy ]</div>
          <div className={`${styles.tag} ${styles.tagTopRight}`}>[ 4 Tumor Classes ]</div>
          <div className={`${styles.tag} ${styles.tagBottomLeft}`}>[ 7000+ MRI Scans ]</div>
          <div className={`${styles.tag} ${styles.tagBottomRight}`}>[ Powered by Deep Learning ]</div>
        </div>

        <div ref={videoRef} className={styles.videoWrapper}>
          <video autoPlay muted loop playsInline className={styles.heroVideo}>
            <source src={ballVideo} type="video/mp4" />
          </video>
        </div>
      </div>
    </section>
  )
}
