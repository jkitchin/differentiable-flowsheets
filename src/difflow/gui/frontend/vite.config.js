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
    // KaTeX is half the bundle on its own. On a loopback server that is
    // a file read, not a download, and splitting it out would put a
    // second committed chunk in git to save nothing.
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        entryFileNames: 'app.js',
        chunkFileNames: 'app-[name].js',
        assetFileNames: 'app.[ext]',
      },
    },
  },
  server: {
    // `npm run dev` against a running
    //   python -m difflow.gui --no-browser --token dev
    // The page comes from vite here, so it carries no token tag and the
    // Origin is vite's; the proxy supplies both as the server expects.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8756',
        changeOrigin: true,
        headers: {
          'X-Difflow-Token': 'dev',
          Origin: 'http://127.0.0.1:8756',
        },
      },
    },
  },
})
