import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const rootDir = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: rootDir,
  build: {
    rollupOptions: {
      input: {
        index: resolve(rootDir, "index.html"),
        theme: resolve(rootDir, "theme.html"),
        admin: resolve(rootDir, "admin.html"),
        performance: resolve(rootDir, "performance.html")
      }
    }
  },
  server: {
    host: "0.0.0.0",
    port: 3002,
    strictPort: true
  },
  preview: {
    host: "0.0.0.0",
    port: 3002,
    strictPort: true
  }
});
