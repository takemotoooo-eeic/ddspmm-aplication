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

const writeU16 = (buf: Uint8Array, offset: number, val: number) => {
  buf[offset] = val & 0xff;
  buf[offset + 1] = (val >> 8) & 0xff;
};

const writeU32 = (buf: Uint8Array, offset: number, val: number) => {
  buf[offset] = val & 0xff;
  buf[offset + 1] = (val >> 8) & 0xff;
  buf[offset + 2] = (val >> 16) & 0xff;
  buf[offset + 3] = (val >> 24) & 0xff;
};

const crc32Table = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[i] = c;
  }
  return table;
})();

const crc32 = (data: Uint8Array): number => {
  let crc = 0xffffffff;
  for (let i = 0; i < data.length; i++) {
    crc = crc32Table[(crc ^ data[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
};

/** STORED 形式で ZIP を生成（ブラウザ標準 API のみ） */
export async function zipFiles(
  entries: { name: string; blob: Blob }[],
): Promise<Blob> {
  const encoded = await Promise.all(
    entries.map(async entry => ({
      name: entry.name,
      nameBytes: new TextEncoder().encode(entry.name),
      data: new Uint8Array(await entry.blob.arrayBuffer()),
    })),
  );

  let totalSize = 0;
  for (const entry of encoded) {
    totalSize += 30 + entry.nameBytes.length + entry.data.length;
    totalSize += 46 + entry.nameBytes.length;
  }
  totalSize += 22;

  const out = new Uint8Array(totalSize);
  const centralChunks: Uint8Array[] = [];
  let offset = 0;

  for (const entry of encoded) {
    const { nameBytes, data } = entry;
    const checksum = crc32(data);
    const localHeaderOffset = offset;

    writeU32(out, offset, 0x04034b50);
    writeU16(out, offset + 4, 20);
    writeU16(out, offset + 6, 0);
    writeU16(out, offset + 8, 0);
    writeU16(out, offset + 10, 0);
    writeU16(out, offset + 12, 0);
    writeU32(out, offset + 14, checksum);
    writeU32(out, offset + 18, data.length);
    writeU32(out, offset + 22, data.length);
    writeU16(out, offset + 26, nameBytes.length);
    writeU16(out, offset + 28, 0);
    out.set(nameBytes, offset + 30);
    out.set(data, offset + 30 + nameBytes.length);
    offset += 30 + nameBytes.length + data.length;

    const cd = new Uint8Array(46 + nameBytes.length);
    writeU32(cd, 0, 0x02014b50);
    writeU16(cd, 4, 20);
    writeU16(cd, 6, 20);
    writeU16(cd, 8, 0);
    writeU16(cd, 10, 0);
    writeU16(cd, 12, 0);
    writeU16(cd, 14, 0);
    writeU32(cd, 16, checksum);
    writeU32(cd, 20, data.length);
    writeU32(cd, 24, data.length);
    writeU16(cd, 28, nameBytes.length);
    writeU16(cd, 30, 0);
    writeU16(cd, 32, 0);
    writeU16(cd, 34, 0);
    writeU16(cd, 36, 0);
    writeU32(cd, 38, 0);
    writeU32(cd, 42, localHeaderOffset);
    cd.set(nameBytes, 46);
    centralChunks.push(cd);
  }

  const centralStart = offset;
  for (const cd of centralChunks) {
    out.set(cd, offset);
    offset += cd.length;
  }

  writeU32(out, offset, 0x06054b50);
  writeU16(out, offset + 4, 0);
  writeU16(out, offset + 6, 0);
  writeU16(out, offset + 8, encoded.length);
  writeU16(out, offset + 10, encoded.length);
  writeU32(out, offset + 12, offset - centralStart);
  writeU32(out, offset + 16, centralStart);
  writeU16(out, offset + 20, 0);

  return new Blob([out], { type: 'application/zip' });
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

    const type = name.endsWith('.wav') ? 'audio/wav' : undefined;
    entries.set(name, new Blob([raw], type ? { type } : undefined));
    offset = dataEnd;
  }

  return entries;
}
