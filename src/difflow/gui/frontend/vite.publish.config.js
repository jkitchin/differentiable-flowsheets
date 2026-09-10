import { svelte } from '@sveltejs/vite-plugin-svelte'
import { defineConfig } from 'vite'

// The published page is a second, separate bundle rather than a second
// entry of the first one. Two entries in one build share chunks, and a
// chunk is a file the page would have to fetch -- which is exactly the
// thing `publish.py` must not emit. So: one entry, no splitting, one
// file `publish.py` can inline verbatim.
export default defineConfig({
  plugins: [svelte()],
  base: './',
  // A library build leaves `process.env.NODE_ENV` alone, and @xyflow
  // reads it to decide which documentation link to put in an error
  // message. On a page with no bundler around it that is a
  // ReferenceError at import time, which renders nothing at all --
  // silently, since the page has no console anyone is watching.
  define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  build: {
    outDir: '../static',
    emptyOutDir: false,
    target: 'es2022',
    chunkSizeWarningLimit: 700,
    lib: {
      entry: 'src/publish.js',
      formats: ['iife'],
      name: 'DifflowPublished',
      fileName: () => 'publish.js',
    },
    rollupOptions: {
      output: { assetFileNames: 'publish.[ext]' },
    },
  },
})
