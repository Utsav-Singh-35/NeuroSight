import { useEffect } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import Lenis from '@studio-freight/lenis'
import Navbar from './components/Navbar/Navbar'
import HeroSection from './components/HeroSection/HeroSection'
import StatementSection from './components/StatementSection/StatementSection'
import InfoSection from './components/InfoSection/InfoSection'
import PipelineSection from './components/PipelineSection/PipelineSection'
import TeamSection from './components/TeamSection/TeamSection'
import ConnectSection from './components/ConnectSection/ConnectSection'

gsap.registerPlugin(ScrollTrigger)

function App() {
  useEffect(() => {
    // Initialize Lenis smooth scrolling engine globally
    const lenis = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)), // ultra smooth Bézier interpolation
      smooth: true,
      smoothTouch: false, // leave mobile touch scrolling native
    })

    // Synchronize Lenis updates with GSAP ScrollTrigger calculations
    lenis.on('scroll', ScrollTrigger.update)

    // Feed Lenis tick update loop into GSAP ticker engine
    gsap.ticker.add((time) => {
      lenis.raf(time * 1000)
    })

    gsap.ticker.lagSmoothing(0)

    return () => {
      lenis.destroy()
      // Clean up ticker hook
      gsap.ticker.remove(lenis.raf)
    }
  }, [])

  return (
    <main className="main-container">
      <Navbar />
      <HeroSection />
      <StatementSection />
      <InfoSection />
      <PipelineSection />
      <TeamSection />
      <ConnectSection />
    </main>
  )
}

export default App
