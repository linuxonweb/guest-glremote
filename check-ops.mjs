// Compares the guest op list (ops.def) with the host's OPS (dist/guest-gl-remote.js)
// and checks every op has a host handler (the module throws at load otherwise).
import { readFileSync } from "node:fs";
import { OPS } from "../../dist/guest-gl-remote.js";

const guest = [...readFileSync(new URL("./ops.def", import.meta.url), "utf8").matchAll(/^OP\((\w+)\)/gm)].map((m) => m[1]);
const bad = guest.length !== OPS.length ? [`length ${guest.length} != ${OPS.length}`] : [];
guest.forEach((name, k) => { if (OPS[k] !== name) bad.push(`#${k + 1}: guest ${name}, host ${OPS[k]}`); });
if (bad.length) {
    console.error("ops mismatch:\n" + bad.join("\n"));
    process.exit(1);
}
console.log(`ops match: ${guest.length}`);
