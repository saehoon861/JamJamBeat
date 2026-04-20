// vite.config.mjs - frontend-test monitor app config
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { readFileSync, existsSync } from "node:fs";
import { defineConfig } from "vite";

const rootDir = dirname(fileURLToPath(import.meta.url));

function ortWasmStaticPlugin() {
  return {
    name: "ort-wasm-static",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url || "";
        const match = url.match(/^\/(ort-wasm[^?]*\.mjs)(\?.*)?$/);
        if (!match) return next();

        const fileName = match[1];
        const filePath = resolve(rootDir, "../frontend/public", fileName);
        if (!existsSync(filePath)) return next();

        const content = readFileSync(filePath, "utf-8");
        res.setHeader("Content-Type", "application/javascript");
        res.setHeader("Cache-Control", "no-cache");
        res.end(content);
      });
    }
  };
}

export default defineConfig({
  root: rootDir,
  publicDir: resolve(rootDir, "../frontend/public"),
  plugins: [ortWasmStaticPlugin()],
  resolve: {
    alias: [
      {
        find: /^onnxruntime-web$/,
        replacement: resolve(rootDir, "node_modules/onnxruntime-web/dist/ort.bundle.min.mjs")
      },
      {
        find: /^onnxruntime-web\/wasm$/,
        replacement: resolve(rootDir, "node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs")
      }
    ]
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
    esbuildOptions: {
      target: "es2020"
    }
  },
  build: {
    target: "es2020"
  },
  server: {
    host: "0.0.0.0",
    port: 3003,
    strictPort: true,
    fs: {
      allow: [resolve(rootDir, "..")]
    }
  },
  preview: {
    host: "0.0.0.0",
    port: 3003,
    strictPort: true
  }
});
