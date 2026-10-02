const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const web = path.resolve(__dirname,'../../web');
const ts = require(path.join(web,'node_modules/typescript'));
// Compile the actual contract module. Hook code is not executed in this check.
const code = ts.transpileModule(fs.readFileSync(path.join(web,'lib/axiom2.ts'),'utf8'), {compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
const sandbox = {exports:{},require:()=>({})}; vm.runInNewContext(code,sandbox);
const parse = sandbox.exports.parseOverview;
const fixture = JSON.parse(fs.readFileSync(process.env.AXIOM_TEST_PROJECTION,'utf8'));
assert.equal(parse(fixture).execution_enabled,false);
const mutations = [d=>delete d.evidence.provenance,d=>d.evidence.provenance.status=['AVAILABLE'],d=>d.boundaries=[null],d=>d.boundaries[0].name=['research'],d=>d.boundaries[0].profile={},d=>d.evidence.rows=[null],d=>d.blocked_prerequisites=[{}],d=>d.execution_enabled=true,d=>d.portfolio.settled_cash=0,d=>d.received_at='invalid',d=>d.schema_version='future'];
for(const change of mutations){const data=structuredClone(fixture);change(data);assert.throws(()=>parse(data));}
console.log(`PASS: valid projection and ${mutations.length} hostile contract regressions`);
