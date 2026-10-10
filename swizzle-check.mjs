// Runs swizzle-check.html in a headed Chromium (the host browser's real
// WebGL2) and prints its result lines; exit status 1 on any failure.
// usage: node swizzle-check.mjs <scratch dir> [page url]
// Needs the dev server (node web/desktop/dev-server.mjs) and an X display.
// CHROMIUM / DISPLAY / PRIME_RUN override the browser, display and GPU wrapper.
import fs from "node:fs";
import { spawn } from "node:child_process";

const [scratch, url = "http://localhost:3000/linux/tests/guest-glremote/swizzle-check.html"] = process.argv.slice(2);
if (!scratch)
    throw new Error("usage: node swizzle-check.mjs <scratch dir> [page url]");
const run = fs.mkdtempSync(`${scratch}/run-`);
const browser = process.env.CHROMIUM ?? "/home/a/.local/archpkg/pacman/usr/bin/chromium";
const prime = process.env.PRIME_RUN ?? "/usr/bin/prime-run";
const args = ["--ignore-gpu-blocklist", "--use-angle=vulkan", "--use-gl=angle", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan",
    "--no-first-run", "--no-default-browser-check", "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=0",
    `--user-data-dir=${run}/profile`, `--disk-cache-dir=${run}/cache`, "--window-size=400,300", "about:blank"];
const log = fs.openSync(`${run}/browser.log`, "w");
const child = spawn(prime, [browser, ...args], {
    detached: true, stdio: ["ignore", log, log],
    env: { ...process.env, DISPLAY: process.env.DISPLAY ?? ":0", TMPDIR: run, TMP: run, TEMP: run, XDG_CACHE_HOME: `${run}/xdg` },
});
fs.closeSync(log);
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
let status = 1;
try {
    let port;
    for (let k = 0; k < 200 && !port; k++) {
        try {
            port = Number(fs.readFileSync(`${run}/profile/DevToolsActivePort`, "utf8").split("\n")[0]);
        }
        catch {
            await sleep(100);
        }
    }
    if (!port)
        throw new Error("browser did not open its DevTools port");
    const page = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === "page");
    const socket = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((done, fail) => { socket.onopen = done; socket.onerror = fail; });
    let next = 0;
    const waiting = new Map();
    socket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        waiting.get(message.id)?.(message);
        waiting.delete(message.id);
    };
    const send = (method, params = {}) => new Promise((done) => {
        const id = ++next;
        waiting.set(id, done);
        socket.send(JSON.stringify({ id, method, params }));
    });
    await send("Page.navigate", { url });
    let text = "";
    for (let k = 0; k < 300; k++) {
        const r = await send("Runtime.evaluate", { expression: "document.getElementById('out')?.textContent ?? ''", returnByValue: true });
        text = r.result?.result?.value ?? "";
        if (/ALL PASSED|FAILED/.test(text))
            break;
        await sleep(100);
    }
    console.log(text.trim() || "no result from the page");
    status = /ALL PASSED/.test(text) ? 0 : 1;
    socket.close();
}
finally {
    try {
        process.kill(-child.pid, "SIGTERM");
    }
    catch { }
}
process.exit(status);
