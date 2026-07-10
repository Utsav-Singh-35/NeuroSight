import { useState, useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import styles from './Navbar.module.css'

const NAV_ITEMS = [
  { label: 'Home',     href: '#home',     icon: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg> },
  { label: 'Platform', href: '#about',    icon: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg> },
  { label: 'Model',    href: '#pipeline', icon: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg> },
]

const MORE_ITEMS = [
  { label: 'Research', href: 'research.html', external: true, icon: <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 9h6M9 13h6M9 17h4"/></svg> },
  { label: 'Dashboard', href: 'dashboard.html', external: true, icon: <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg> },
  { label: 'Team',     href: '#team',      icon: <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg> },
  { label: 'Connect',  href: '#contact',   icon: <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg> },
]

export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false)
  const [moreOpen, setMoreOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  const navRef = useRef(null)

  // Master smooth arrival: pure gsap.to moving from native zero-state to pristine destination
  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.to(navRef.current, {
        y: 0,
        opacity: 1,
        duration: 1.2,
        delay: 2.2, // Drop perfectly into place only after main landing page contents finish loading
        ease: 'power3.out',
      })
    }, navRef)
    return () => ctx.revert()
  }, [])

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 50)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  // Close more dropdown when clicking outside
  useEffect(() => {
    const handler = (e) => {
      if (navRef.current && !navRef.current.contains(e.target)) {
        setMoreOpen(false)
        setMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const handleNavClick = (href, external = false) => {
    if (href === '#') return
    if (external) {
      window.location.href = href
      return
    }
    const el = document.querySelector(href)
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    setMenuOpen(false)
    setMoreOpen(false)
  }

  return (
    // Set native pre-render hidden style inline guaranteeing absolute zero flash/jumping on initial request load
    <nav
      ref={navRef}
      className={`${styles.navbar} ${scrolled ? styles.scrolled : ''}`}
      style={{ opacity: 0, transform: 'translate(-50%, -100px)' }}
    >
      <div className={styles.logo}>NeuraSight</div>

      {/* Hamburger */}
      <button
        className={styles.hamburger}
        onClick={() => setMenuOpen(v => !v)}
        aria-label="Toggle menu"
        aria-expanded={menuOpen}
      >
        {menuOpen
          ? <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          : <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
        }
      </button>

      {/* Nav Links */}
      <ul className={`${styles.navLinks} ${menuOpen ? styles.active : ''}`}>
        {NAV_ITEMS.map(item => (
          <li key={item.label}>
            <a href={item.href} onClick={e => { e.preventDefault(); handleNavClick(item.href) }}>
              {item.icon} {item.label}
            </a>
          </li>
        ))}

        {/* More Dropdown */}
        <li className={`${styles.navMore} ${moreOpen ? styles.moreActive : ''}`}>
          <a
            href="#"
            className={styles.moreToggle}
            onClick={e => { e.preventDefault(); setMoreOpen(v => !v) }}
          >
            More{' '}
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="m6 9 6 6 6-6"/>
            </svg>
          </a>
          <ul className={styles.moreDropdown}>
            {MORE_ITEMS.map(item => (
              <li key={item.label}>
                <a href={item.href} onClick={e => { e.preventDefault(); handleNavClick(item.href, item.external) }}>
                  {item.icon} {item.label}
                </a>
              </li>
            ))}
          </ul>
        </li>
      </ul>
    </nav>
  )
}
