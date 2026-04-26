import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Force all packages to use the same single copy of React
      'react': resolve('./node_modules/react'),
      'react-dom': resolve('./node_modules/react-dom'),
    }
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
})