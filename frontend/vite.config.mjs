import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { readFileSync, existsSync } from "node:fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const rootDir = dirname(fileURLToPath(import.meta.url));

// Vite 개발 서버가 ort-wasm-*.mjs 파일을 ES 모듈로 변환하지 않고
// public/ 폴더에서 직접 raw 파일로 서빙하도록 하는 플러그인입니다.
function ortWasmStaticPlugin() {
  return {
    name: "ort-wasm-static",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        // ort-wasm-*.mjs 요청을 가로채서 직접 서빙 (?import 쿼리 포함 시)
        const url = req.url || "";
        const match = url.match(/^\/(ort-wasm[^?]*\.mjs)(\?.*)?$/);
        if (!match) return next();

        const fileName = match[1];
        const filePath = resolve(rootDir, "public", fileName);
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
  plugins: [react(), ortWasmStaticPlugin()],
  resolve: {
    dedupe: ["react", "react-dom"]
  },
  optimizeDeps: {
    include: ["react", "react-dom", "react/jsx-runtime"],
    exclude: ["onnxruntime-web"],
    esbuildOptions: {
      target: "es2020" // BigInt 지원을 위해 es2020으로 설정
    }
  },
  build: {
    target: "es2020", // BigInt 지원을 위해 es2020으로 설정
    rollupOptions: {
      input: {
        index: resolve(rootDir, "index.html"),
        theme: resolve(rootDir, "theme.html"),
        admin: resolve(rootDir, "admin.html"),
        mapping: resolve(rootDir, "mapping.html")
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
