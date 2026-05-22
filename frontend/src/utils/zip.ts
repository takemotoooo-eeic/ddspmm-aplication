const SIG_LOCAL = 0x04034b50;

const readU16 = (buf: Uint8Array, offset: number) =>
  buf[offset] | (buf[offset + 1] << 8);

const readU32 = (buf: Uint8Array, offset: number) =>
  buf[offset] |
  (buf[offset + 1] << 8) |
  (buf[offset + 2] << 16) |
  (buf[offset + 3] << 24);

async function inflateRaw(compressed: Uint8Array): Promise<Uint8Array> {
  const ds = new DecompressionStream('deflate-raw');
  const writer = ds.writable.getWriter();
  const reader = ds.readable.getReader();
  writer.write(compressed);
  writer.close();
  const chunks: Uint8Array[] = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.length;
  }
  const out = new Uint8Array(total);
  let pos = 0;
  for (const chunk of chunks) {
    out.set(chunk, pos);
    pos += chunk.length;
  }
  return out;
}

/** 小規模 ZIP（DEFLATE / STORED）をブラウザ標準 API のみで展開 */
export async function unzipToMap(blob: Blob): Promise<Map<string, Blob>> {
  const buf = new Uint8Array(await blob.arrayBuffer());
  const entries = new Map<string, Blob>();
  let offset = 0;

  while (offset + 30 <= buf.length) {
    if (readU32(buf, offset) !== SIG_LOCAL) break;

    const compression = readU16(buf, offset + 8);
    const compressedSize = readU32(buf, offset + 18);
    const filenameLen = readU16(buf, offset + 26);
    const extraLen = readU16(buf, offset + 28);
    const nameStart = offset + 30;
    const nameEnd = nameStart + filenameLen;
    const name = new TextDecoder().decode(buf.subarray(nameStart, nameEnd));
    const dataStart = nameEnd + extraLen;
    const dataEnd = dataStart + compressedSize;
    const compressed = buf.subarray(dataStart, dataEnd);

    let raw: Uint8Array;
    if (compression === 0) {
      raw = compressed;
    } else if (compression === 8) {
      raw = await inflateRaw(compressed);
    } else {
      throw new Error(`Unsupported ZIP compression method: ${compression}`);
    }

    entries.set(name, new Blob([raw]));
    offset = dataEnd;
  }

  return entries;
}
