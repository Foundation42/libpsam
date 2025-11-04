#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const os = require("os");
const https = require("https");
const { pipeline } = require("stream/promises");
const tar = require("tar");

const projectRoot = path.resolve(__dirname, "..", "..");
const wasmDir = path.resolve(__dirname, "..", "public", "wasm");
fs.mkdirSync(wasmDir, { recursive: true });

const buildDir = path.resolve(projectRoot, "bindings", "wasm", "build");
const localJs = path.join(buildDir, "psam.js");
const localWasm = path.join(buildDir, "psam.wasm");
const localBindingsDir = path.resolve(projectRoot, "bindings", "javascript", "dist");

const DEST_FILES = {
  "psam.js": path.join(wasmDir, "psam.js"),
  "psam.wasm": path.join(wasmDir, "psam.wasm"),
  "psam-bindings.js": path.join(wasmDir, "psam-bindings.js"),
  "psam-bindings.d.ts": path.join(wasmDir, "psam-bindings.d.ts"),
  "types.d.ts": path.join(wasmDir, "types.d.ts"),
};

function removeExisting() {
  for (const dest of Object.values(DEST_FILES)) {
    try {
      fs.unlinkSync(dest);
    } catch (err) {
      if (err.code !== "ENOENT") {
        throw err;
      }
    }
  }
}

function copyLocalBuild() {
  if (!fs.existsSync(localJs) || !fs.existsSync(localWasm)) {
    return false;
  }

  removeExisting();
  fs.copyFileSync(localJs, DEST_FILES["psam.js"]);
  fs.copyFileSync(localWasm, DEST_FILES["psam.wasm"]);

  const localBindingsJs = path.join(localBindingsDir, "wasm.js");
  const localBindingsDts = path.join(localBindingsDir, "wasm.d.ts");
  const localTypesDts = path.join(localBindingsDir, "types.d.ts");

  if (fs.existsSync(localBindingsJs)) {
    fs.copyFileSync(localBindingsJs, DEST_FILES["psam-bindings.js"]);
  }
  if (fs.existsSync(localBindingsDts)) {
    fs.copyFileSync(localBindingsDts, DEST_FILES["psam-bindings.d.ts"]);
  }
  if (fs.existsSync(localTypesDts)) {
    fs.copyFileSync(localTypesDts, DEST_FILES["types.d.ts"]);
  }

  console.log("[wasm] Copied artifacts from local bindings/wasm/build");
  return true;
}

async function downloadFromRelease() {
  const url = process.env.PSAM_WASM_URL ||
    "https://github.com/Foundation42/libpsam/releases/download/latest/libpsam-wasm.tar.gz";
  console.log(`[wasm] Downloading libpsam WASM bundle from ${url}`);

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "psam-wasm-"));
  const tarballPath = path.join(tmpDir, "libpsam-wasm.tar.gz");

  await downloadFile(url, tarballPath);
  await tar.extract({ file: tarballPath, cwd: tmpDir });

  const extractedDir = path.join(tmpDir, "libpsam-wasm");
  const files = {
    "psam.js": path.join(extractedDir, "psam.js"),
    "psam.wasm": path.join(extractedDir, "psam.wasm"),
    "psam-bindings.js": path.join(extractedDir, "psam-bindings.js"),
    "psam-bindings.d.ts": path.join(extractedDir, "psam-bindings.d.ts"),
    "types.d.ts": path.join(extractedDir, "types.d.ts"),
  };

  if (!fs.existsSync(files["psam.js"]) || !fs.existsSync(files["psam.wasm"])) {
    throw new Error("Downloaded archive did not contain psam.js/psam.wasm");
  }

  removeExisting();
  for (const [name, src] of Object.entries(files)) {
    if (fs.existsSync(src)) {
      fs.copyFileSync(src, DEST_FILES[name]);
    }
  }

  console.log("[wasm] Downloaded artifacts from latest release");
}

async function downloadFile(url, destPath) {
  await new Promise((resolve, reject) => {
    const request = https.get(url, response => {
      if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        response.destroy();
        downloadFile(response.headers.location, destPath).then(resolve).catch(reject);
        return;
      }
      if (response.statusCode !== 200) {
        reject(new Error(`Failed to download ${url} (status ${response.statusCode})`));
        response.resume();
        return;
      }
      const fileStream = fs.createWriteStream(destPath);
      pipeline(response, fileStream).then(resolve).catch(reject);
    });
    request.on("error", reject);
  });
}

(async () => {
  try {
    const copied = copyLocalBuild();
    if (!copied) {
      await downloadFromRelease();
    }
  } catch (err) {
    console.error("[wasm] Failed to sync WASM artifacts:", err.message);
    console.error("[wasm] Run ./scripts/docker-build.sh wasm to build locally or set PSAM_WASM_URL to an alternative archive.");
    process.exit(1);
  }
})();
