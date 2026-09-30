import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // Express gateway (:5000) — persistence + scan history.
      // Must be listed before the bare FastAPI paths so /api/* is not
      // captured by a shorter prefix.
      '/api': { target: 'http://localhost:5000', changeOrigin: true },

      // Direct-to-FastAPI (:8000) paths, kept for stateless calls and
      // backwards compatibility.
      '/predict': { target: 'http://localhost:8000', changeOrigin: true },
      '/gradcam': { target: 'http://localhost:8000', changeOrigin: true },
      '/report': { target: 'http://localhost:8000', changeOrigin: true },
      '/health': { target: 'http://localhost:8000', changeOrigin: true },
      '/modules': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      input: {
        main: 'index.html',
        dashboard: 'dashboard.html',
        research: 'research.html',
      },
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          gsap: ['gsap'],
          three: ['three'],
          lenis: ['@studio-freight/lenis'],
        },
      },
    },
  },
})
