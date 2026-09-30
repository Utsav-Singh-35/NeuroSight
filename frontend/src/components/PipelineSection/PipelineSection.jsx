import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import styles from './PipelineSection.module.css'

gsap.registerPlugin(ScrollTrigger)

const PIPELINE_STAGES = [
  {
    phase: 'Phase 01',
    title: 'MRI Acquisition & Preprocessing',
    desc: 'Validate image quality, normalize MRI scans, remove noise, and prepare data for deep learning inference.',
    color: 0x22d3ee, // Cyan
    cssGlow: styles.glowCyan,
  },
  {
    phase: 'Phase 02',
    title: 'Deep Learning Classification Engine',
    desc: 'EfficientNet-based architecture extracts medical imaging features and classifies tumors across multiple categories.',
    color: 0xc084fc, // Purple
    cssGlow: styles.glowPurple,
  },
  {
    phase: 'Phase 03',
    title: 'Explainable AI & Clinical Insights',
    desc: 'Grad-CAM visualization highlights suspicious regions and generates interpretable diagnostic information.',
    color: 0xf472b6, // Pink
    cssGlow: styles.glowPink,
  },
]

export default function PipelineSection() {
  const mountRef = useRef(null)
  const sectionRef = useRef(null)
  const [activeStage, setActiveStage] = useState(null)
  const activeStageRef = useRef(null)

  // Sync state to ref so Three.js animate loop reads it instantly without re-mounting
  useEffect(() => {
    activeStageRef.current = activeStage
  }, [activeStage])

  // Three.js instances ref to avoid re-triggering across renders
  const sceneData = useRef({
    nodes: [],
    particles: [],
    mouse: { x: 0, y: 0 }
  })

  useEffect(() => {
    const currentMount = mountRef.current
    if (!currentMount) return

    // 1. Scene Setup
    const scene = new THREE.Scene()
    
    // 2. Camera Setup
    const width = currentMount.clientWidth || 1100
    const height = currentMount.clientHeight || 380
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000)
    camera.position.set(0, 2, 12)
    camera.lookAt(0, 0, 0)

    // 3. Renderer Setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    // Ensure the canvas is fully styled as a block element filling its container
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'
    renderer.domElement.style.display = 'block'
    currentMount.appendChild(renderer.domElement)

    // 4. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
    scene.add(ambientLight)

    const pointLight = new THREE.PointLight(0xffffff, 2, 30)
    pointLight.position.set(0, 8, 8)
    scene.add(pointLight)

    // 5. Create Pipeline Nodes
    const nodes = []
    const nodePositions = [-4.2, 0, 4.2]

    // Geometries
    const geom1 = new THREE.OctahedronGeometry(1.2, 0)
    const geom2 = new THREE.TorusKnotGeometry(0.7, 0.25, 100, 16)
    const geom3 = new THREE.IcosahedronGeometry(1.1, 0)
    const geometries = [geom1, geom2, geom3]

    PIPELINE_STAGES.forEach((stage, idx) => {
      const group = new THREE.Group()
      group.position.set(nodePositions[idx], 0, 0)

      // Outer wireframe glowing shell
      const matOuter = new THREE.MeshStandardMaterial({
        color: stage.color,
        wireframe: true,
        emissive: stage.color,
        emissiveIntensity: 0.4,
        transparent: true,
        opacity: 0.8
      })
      const meshOuter = new THREE.Mesh(geometries[idx], matOuter)
      group.add(meshOuter)

      // Inner solid core
      const matInner = new THREE.MeshStandardMaterial({
        color: 0x111111,
        roughness: 0.3,
        metalness: 0.8,
        emissive: stage.color,
        emissiveIntensity: 0.1
      })
      const innerGeom = geometries[idx].clone()
      innerGeom.scale(0.7, 0.7, 0.7)
      const meshInner = new THREE.Mesh(innerGeom, matInner)
      group.add(meshInner)

      // Add a base floating platform grid underneath each node
      const ringGeom = new THREE.RingGeometry(1.4, 1.5, 32)
      ringGeom.rotateX(-Math.PI / 2)
      const ringMat = new THREE.MeshBasicMaterial({
        color: stage.color,
        transparent: true,
        opacity: 0.3,
        side: THREE.DoubleSide
      })
      const baseRing = new THREE.Mesh(ringGeom, ringMat)
      baseRing.position.y = -1.6
      group.add(baseRing)

      scene.add(group)
      nodes.push({ group, meshOuter, meshInner, baseSpeed: 0.01 })
    })
    sceneData.current.nodes = nodes

    // 6. Connecting Pipeline Tubes / Paths
    const pathMat = new THREE.LineBasicMaterial({
      color: 0x444444,
      transparent: true,
      opacity: 0.4
    })
    
    // Line from Node 1 to Node 2
    const points1 = [new THREE.Vector3(-4.2, 0, 0), new THREE.Vector3(0, 0, 0)]
    const lineGeom1 = new THREE.BufferGeometry().setFromPoints(points1)
    const line1 = new THREE.Line(lineGeom1, pathMat)
    scene.add(line1)

    // Line from Node 2 to Node 3
    const points2 = [new THREE.Vector3(0, 0, 0), new THREE.Vector3(4.2, 0, 0)]
    const lineGeom2 = new THREE.BufferGeometry().setFromPoints(points2)
    const line2 = new THREE.Line(lineGeom2, pathMat)
    scene.add(line2)

    // 7. Data Flow Particles traversing the pipeline
    const particleCount = 20
    const particles = []
    const pGeom = new THREE.SphereGeometry(0.08, 8, 8)
    
    for (let i = 0; i < particleCount; i++) {
      const segment = i % 2
      const color = segment === 0 ? PIPELINE_STAGES[0].color : PIPELINE_STAGES[1].color
      const pMat = new THREE.MeshBasicMaterial({ color })
      const pMesh = new THREE.Mesh(pGeom, pMat)
      
      const progress = Math.random()
      const startX = segment === 0 ? -4.2 + progress * 4.2 : progress * 4.2
      pMesh.position.set(startX, 0, 0)
      
      scene.add(pMesh)
      particles.push({
        mesh: pMesh,
        progress,
        segment,
        speed: 0.004 + Math.random() * 0.003
      })
    }
    sceneData.current.particles = particles

    // 8. Resize Handler
    const handleResize = () => {
      if (!currentMount) return
      const w = currentMount.clientWidth
      const h = currentMount.clientHeight
      if (w === 0 || h === 0) return
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    window.addEventListener('resize', handleResize)

    // 9. Parallax Mouse Move Handler
    const handleMouseMove = (e) => {
      const rect = currentMount.getBoundingClientRect()
      const x = ((e.clientX - rect.left) / rect.width) * 2 - 1
      const y = -(((e.clientY - rect.top) / rect.height) * 2 - 1)
      sceneData.current.mouse = { x, y }
    }
    currentMount.addEventListener('mousemove', handleMouseMove)

    // 10. Animation Loop
    let animationFrameId
    let clock = new THREE.Clock()

    const animate = () => {
      const elapsedTime = clock.getElapsedTime()
      const currentActive = activeStageRef.current
      
      // Update Nodes rotation and dynamic bounce reads live ref instantly
      nodes.forEach((node, idx) => {
        const speedMultiplier = currentActive === idx ? 3 : 1
        node.group.rotation.y += node.baseSpeed * speedMultiplier
        node.meshOuter.rotation.x += 0.005
        node.meshOuter.rotation.z += 0.005

        node.group.position.y = Math.sin(elapsedTime * 2 + idx) * 0.15

        const targetScale = currentActive === idx ? 1.25 : 1.0
        const currentScale = node.group.scale.x
        const newScale = THREE.MathUtils.lerp(currentScale, targetScale, 0.1)
        node.group.scale.set(newScale, newScale, newScale)

        node.meshOuter.material.emissiveIntensity = currentActive === idx ? 0.8 : 0.4
      })

      // Update Data Flow Particles
      particles.forEach((p) => {
        p.progress += p.speed
        if (p.progress > 1.0) {
          p.progress = 0
        }

        if (p.segment === 0) {
          p.mesh.position.x = -4.2 + p.progress * 4.2
          p.mesh.position.y = Math.sin(p.progress * Math.PI * 4) * 0.08
          p.mesh.position.z = Math.cos(p.progress * Math.PI * 4) * 0.08
        } else {
          p.mesh.position.x = p.progress * 4.2
          p.mesh.position.y = Math.sin(p.progress * Math.PI * 4) * 0.08
          p.mesh.position.z = Math.cos(p.progress * Math.PI * 4) * 0.08
        }
      })

      // Subtle Parallax Camera movement
      const targetCamX = sceneData.current.mouse.x * 1.5
      const targetCamY = 2 + sceneData.current.mouse.y * 1.0
      camera.position.x = THREE.MathUtils.lerp(camera.position.x, targetCamX, 0.05)
      camera.position.y = THREE.MathUtils.lerp(camera.position.y, targetCamY, 0.05)
      camera.lookAt(0, 0, 0)

      renderer.render(scene, camera)
      animationFrameId = requestAnimationFrame(animate)
    }
    animate()

    // GSAP Scroll effect for cards reveal setup once
    const ctx = gsap.context(() => {
      gsap.from(`.${styles.pipelineCard}`, {
        scrollTrigger: {
          trigger: sectionRef.current,
          start: 'top 75%',
        },
        y: 40,
        opacity: 0,
        duration: 0.8,
        stagger: 0.2,
        ease: 'power3.out',
      })
    }, sectionRef)

    // Cleanup executes exactly once on final unmount
    return () => {
      window.removeEventListener('resize', handleResize)
      if (currentMount) {
        currentMount.removeEventListener('mousemove', handleMouseMove)
        currentMount.removeChild(renderer.domElement)
      }
      cancelAnimationFrame(animationFrameId)
      renderer.dispose()
      ctx.revert()
    }
  }, []) // Empty dependency array stops infinite refreshes entirely!

  return (
    <section id="pipeline" ref={sectionRef} className={styles.pipelineSection}>
      <div className={styles.header}>
        <p className={styles.eyebrow}>BRAIN MRI ANALYSIS WORKFLOW</p>
        <h2 className={styles.title}>Brain MRI Analysis Pipeline</h2>
        <p className={styles.subtitle}>
          Interactive visualization of the complete MRI diagnostic workflow from image acquisition to clinical insight generation.
        </p>
      </div>

      {/* Interactive 3D Model Viewport */}
      <div className={styles.modelContainer}>
        <div ref={mountRef} className={styles.threeCanvas} />
        
        {/* Helper prompt instructions inside model box */}
        <div className={styles.modelOverlay}>
          <span className={styles.dragIndicator}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><path d="M19 12h2M3 12h2M12 3v2M12 19v2"/></svg>
            Move mouse to shift camera perspective
          </span>
        </div>
      </div>

      {/* Grid of UI Phase Cards linked dynamically to 3D Nodes */}
      <div className={styles.pipelineGrid}>
        {PIPELINE_STAGES.map((stage, idx) => (
          <div
            key={stage.phase}
            className={`${styles.pipelineCard} ${stage.cssGlow} ${activeStage === idx ? styles.cardActive : ''}`}
            onMouseEnter={() => setActiveStage(idx)}
            onMouseLeave={() => setActiveStage(null)}
          >
            <div className={styles.cardTop}>
              <div className={styles.stageIndicator} style={{ borderColor: `#${stage.color.toString(16)}` }}>
                <span className={styles.dot} style={{ backgroundColor: `#${stage.color.toString(16)}` }} />
              </div>
              <span className={styles.phaseBadge}>{stage.phase}</span>
            </div>
            <h3 className={styles.cardTitle}>{stage.title}</h3>
            <p className={styles.cardDesc}>{stage.desc}</p>
            <div className={styles.cardFooter}>
              <span className={styles.statusLabel}>Inspect Pipeline</span>
              <span className={styles.arrowIcon}>→</span>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
