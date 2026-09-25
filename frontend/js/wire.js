/* Decoder for the PRH1 map frame.
   This is the only way frames enter the console: what is drawn is what was transmitted,
   so a codec bug shows up as a broken map rather than as an optimistic byte count. */

const MAGIC = 'PRH1';
const HEADER_BYTES = 26;
const FLAG_COMPRESSED = 0x01;

async function inflate(bytes) {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('This browser cannot inflate deflate streams. ' +
                    'Turn compression off on the server, or use a current browser.');
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export async function decodeFrame(buffer) {
  const head = new Uint8Array(buffer, 0, HEADER_BYTES);
  const dv = new DataView(buffer, 0, HEADER_BYTES);

  const magic = String.fromCharCode(head[0], head[1], head[2], head[3]);
  if (magic !== MAGIC) throw new Error(`Unexpected frame magic "${magic}"`);

  const version = dv.getUint8(4);
  if (version !== 1) throw new Error(`Unsupported frame version ${version}`);

  const flags = dv.getUint8(5);
  const frame = dv.getUint32(6, true);
  const n = dv.getUint32(10, true);
  const quant = dv.getFloat32(14, true);
  const root = dv.getFloat32(18, true);
  const jsonLen = dv.getUint32(22, true);

  let body = new Uint8Array(buffer, HEADER_BYTES);
  body = (flags & FLAG_COMPRESSED) ? await inflate(body) : body.slice();

  // The three int16 arrays lead, so every view below lands on a two-byte boundary.
  const buf = body.buffer;
  const base = body.byteOffset;
  let o = base;
  const xq = new Int16Array(buf, o, n); o += n * 2;
  const yq = new Int16Array(buf, o, n); o += n * 2;
  const zq = new Int16Array(buf, o, n); o += n * 2;
  const level = new Uint8Array(buf, o, n); o += n;
  const cls = new Uint8Array(buf, o, n); o += n;
  const confQ = new Uint8Array(buf, o, n); o += n;
  const driver = new Uint8Array(buf, o, n); o += n;

  let tail = {};
  if (jsonLen > 0) {
    tail = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, o, jsonLen)));
  }

  // Dequantise into float arrays once, so the renderer never does it per draw.
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  const z = new Float32Array(n);
  const size = new Float32Array(n);
  const conf = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = xq[i] * quant;
    y[i] = yq[i] * quant;
    z[i] = zq[i] / 100;
    size[i] = root / (1 << level[i]);
    conf[i] = confQ[i] / 255;
  }

  return { frame, n, quant, root, x, y, z, size, level, cls, conf, driver, tail,
           wireBytes: buffer.byteLength };
}
