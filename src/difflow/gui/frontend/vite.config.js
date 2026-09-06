import { svelte } from '@sveltejs/vite-plugin-svelte'
import { defineConfig } from 'vite'

// Built output goes to ../static/ and is committed, so `pip install difflow`
// never needs node -- only changing the UI does. Hashed filenames would make
// every rebuild a new file in git for no benefit on a local server, so the
// names are fixed and the bundle is one file.
export default defineConfig({
  plugins: [svelte()],
  base: './',
  build: {
    outDir: '../static',
    emptyOutDir: false,
    target: 'es2022',
    rollupOptions: {
      output: {
        entryFileNames: 'app.js',
        chunkFileNames: 'app-[name].js',
        assetFileNames: 'app.[ext]',
      },
    },
  },
  server: {
    // `npm run dev` against a running `python -m difflow.gui --no-browser`.
    proxy: { '/api': 'http://127.0.0.1:8756' },
  },
})
