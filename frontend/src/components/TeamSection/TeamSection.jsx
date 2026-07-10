import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import githubImg from '../../assets/github.png'
import linkedinImg from '../../assets/linkedin.png'
import portfolioImg from '../../assets/portfolio.png'
import styles from './TeamSection.module.css'

gsap.registerPlugin(ScrollTrigger)

const TEAM = [
  {
    name: ['Utsav', 'Singh'],
    role: 'AI Research',
    links: [
      { href: 'https://github.com/Utsav-Singh-35', label: 'GitHub', img: githubImg },
      { href: 'https://linkedin.com/in/utsav-singh', label: 'LinkedIn', img: linkedinImg, linkedin: true },
      { href: '#', label: 'Personal Site', img: portfolioImg },
    ],
  },
  {
    name: ['Ankit', 'Sharma'],
    role: 'Medical Imaging',
    links: [
      { href: 'https://github.com/', label: 'GitHub', img: githubImg },
      { href: 'https://linkedin.com/in/', label: 'LinkedIn', img: linkedinImg, linkedin: true },
      { href: '#', label: 'Personal Site', img: portfolioImg },
    ],
  },
  {
    name: ['Ranjan', 'Singh'],
    role: 'Full Stack Development',
    links: [
      { href: 'https://github.com/', label: 'GitHub', img: githubImg },
      { href: 'https://linkedin.com/in/', label: 'LinkedIn', img: linkedinImg, linkedin: true },
      { href: '#', label: 'Personal Site', img: portfolioImg },
    ],
  },
]

export default function TeamSection() {
  const sectionRef = useRef(null)
  const titleRef = useRef(null)

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Animate section title fading and translating up
      gsap.from(titleRef.current, {
        scrollTrigger: {
          trigger: sectionRef.current,
          start: 'top 80%',
        },
        y: 50,
        opacity: 0,
        duration: 0.8,
        ease: 'power3.out',
      })

      // Stagger reveal developer cards
      gsap.from(`.${styles.teamCard}`, {
        scrollTrigger: {
          trigger: sectionRef.current,
          start: 'top 70%',
        },
        y: 60,
        opacity: 0,
        duration: 0.8,
        stagger: 0.2,
        ease: 'power3.out',
      })
    }, sectionRef)

    return () => ctx.revert()
  }, [])

  return (
    <section id="team" ref={sectionRef} className={styles.teamSection}>
      <h2 ref={titleRef} className={styles.teamTitle}>Meet the Developers</h2>
      <div className={styles.teamGridWrapper}>
        <div className={styles.teamGrid}>
          {TEAM.map(member => (
            <div key={member.name.join('')} className={styles.teamCard}>
              <div className={styles.teamName}>
                {member.name[0]}<br />{member.name[1]}
              </div>
              <div className={styles.teamRole}>{member.role}</div>
              <div className={styles.teamLinks}>
                {member.links.map(link => (
                  <a
                    key={link.label}
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={`${styles.teamLink} ${link.linkedin ? styles.teamLinkLinkedin : ''}`}
                    title={link.label}
                  >
                    <img src={link.img} alt={link.label} />
                  </a>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
