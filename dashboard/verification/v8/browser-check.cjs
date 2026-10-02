// Run with PLAYWRIGHT_MODULE pointing to an existing Playwright install.
// Uses only local test auth and read-only projections; never opens Legacy FX.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE);
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const origin = process.env.AXIOM_TEST_ORIGIN || 'http://127.0.0.1:3014';
const output = process.env.AXIOM_TEST_OUTPUT || '/tmp/axiom-v8-browser-output';
fs.mkdirSync(output, { recursive: true });
const projection = JSON.parse(fs.readFileSync(process.env.AXIOM_TEST_PROJECTION, 'utf8'));
(async () => {
 const browser = await chromium.launch({ headless: true });
 const context = await browser.newContext({ viewport: {width:1536,height:1024} });
 const publicResponse = await context.request.get(origin + '/api/axiom2/overview');
 assert.equal(publicResponse.status(),401);
 const login = await context.request.post(origin + '/auth/login', {data:{password:'dashboard-local-verification-only'}});
 assert.equal(login.status(),200);
 // Local HTTP harness installs the valid signed test token; production cookies remain Secure.
 const token = /axiom_session=([^;]+)/.exec(login.headers()['set-cookie'])[1];
 await context.addCookies([{name:'axiom_session',value:token,url:origin,httpOnly:true,sameSite:'Lax',secure:false}]);
 assert.equal((await context.request.post(origin + '/api/axiom2/overview')).status(),405);
 const real = await context.request.get(origin + '/api/axiom2/overview');
 assert.equal(real.status(),200);
 assert.equal(real.headers()['cache-control'],'no-store');
 assert.equal((await real.json()).execution_enabled,false);
 const page = await context.newPage();
 const errors = []; page.on('pageerror',e=>errors.push(e.message));
 await page.goto(origin); await page.getByText('Local projection received · source freshness unknown').waitFor();
 await page.screenshot({path:path.join(output,'desktop-overview.png'),fullPage:true});
 for (const view of ['Markets','Evidence','Portfolio','Execution','System']) {
   await page.getByRole('navigation').getByRole('button',{name:view,exact:true}).click();
   await page.getByRole('heading',{level:1}).waitFor();
   assert.equal(await page.getByRole('navigation').getByRole('button',{name:view,exact:true}).getAttribute('aria-current'),'page');
 }
 await page.getByRole('navigation').getByRole('button',{name:'Execution',exact:true}).click();
 await page.locator('.a2-skip').focus(); await page.keyboard.press('Enter');
 await page.waitForTimeout(100); assert.equal(await page.getByRole('heading',{level:1}).textContent(),'Execution & reconciliation');
 assert.equal(await page.getByRole('button',{name:'Submit order',exact:true}).isDisabled(),true);
 await page.getByRole('navigation').getByRole('button',{name:'Portfolio',exact:true}).click(); await page.goBack(); assert.equal(await page.getByRole('heading',{level:1}).textContent(),'Execution & reconciliation');
 await page.getByRole('navigation').getByRole('button',{name:'Overview',exact:true}).click();
 await page.setViewportSize({width:390,height:844});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth <= window.innerWidth),true);
 await page.screenshot({path:path.join(output,'iphone-overview.png'),fullPage:true});
 const second = await context.newPage(); await second.goto(origin+'/#system');
 await second.getByRole('heading',{name:'System provenance'}).waitFor();
 await page.getByRole('navigation').getByRole('button',{name:'Markets',exact:true}).click();
 assert.equal(await second.getByRole('heading',{level:1}).textContent(),'System provenance');
 // Controlled fixtures test presentation only, never qualification.
 let responseBody = projection;
 let responseStatus = 200;
 let delayed = false;
 await page.route('**/api/axiom2/overview',async route=> {
   if(delayed) await new Promise(r=>setTimeout(r,800));
   await route.fulfill({status:responseStatus,contentType:'application/json',body:JSON.stringify(responseBody)});
 });
 responseStatus=502; await page.getByRole('button',{name:'Refresh',exact:true}).click();
 await page.getByRole('status').filter({hasText:'Previous response retained as stale'}).waitFor();
 await page.screenshot({path:path.join(output,'iphone-stale-error.png'),fullPage:true});
 responseStatus=200; responseBody={...projection,received_at:new Date(Date.now()-60000).toISOString()};
 await page.getByRole('button',{name:'Refresh',exact:true}).click(); await page.getByText('Stale projection · source freshness unknown').waitFor();
 for(const mutate of [d=>delete d.evidence.provenance,d=>d.boundaries=[null],d=>d.evidence.rows=[null],d=>d.blocked_prerequisites=[{}],d=>d.boundaries[0].profile={},d=>d.evidence.provenance.status=["AVAILABLE"],d=>d.boundaries[0].name=["research"]]) {
   responseBody=structuredClone(projection); mutate(responseBody);
   await page.getByRole('button',{name:'Refresh',exact:true}).click();
   await page.getByRole('status').filter({hasText:'Invalid dashboard provenance'}).waitFor();
 }
 responseBody={...projection,received_at:new Date().toISOString()}; await page.getByRole('button',{name:'Refresh',exact:true}).click();
 await page.getByText('Local projection received · source freshness unknown').waitFor();
 responseBody={...projection,received_at:new Date().toISOString(),evidence:{...projection.evidence,rows:[{lane_id:'equity_research_test_wide',package_id:'Browser fixture · not genuine evidence',package_digest:'a'.repeat(64),cached_state:'CHAMPION',head_event_digest:'b'.repeat(64),evidence_class:'UNVERIFIED',capital_eligible:false}]}};
 await page.getByRole('button',{name:'Refresh',exact:true}).click(); await page.getByText('Local projection received · source freshness unknown').waitFor(); await page.getByRole('navigation').getByRole('button',{name:'Evidence',exact:true}).click(); await page.getByLabel('Inspect evidence package').selectOption('a'.repeat(64)); await page.getByText('Unverified · no capital eligibility',{exact:true}).waitFor();
 responseBody={...projection,received_at:new Date().toISOString()};
 delayed=true; await page.reload(); await page.getByText('Loading projection…').waitFor();
 await page.getByText('Local projection received · source freshness unknown').waitFor(); delayed=false;
 await page.getByRole('navigation').getByRole('button',{name:'System',exact:true}).click();
 await page.getByText('execution boundary',{exact:true}).waitFor();
 await page.screenshot({path:path.join(output,'iphone-system.png'),fullPage:true});
 await page.route('**/auth/logout',route=>route.fulfill({status:502,body:'unavailable'})); await page.getByRole('button',{name:'Sign out',exact:true}).click(); await page.getByRole('alert').filter({hasText:'Unable to sign out. Retry.'}).waitFor(); await page.unroute('**/auth/logout');
 await page.getByRole('button',{name:'Sign out',exact:true}).click(); await page.waitForURL('**/login');
 assert.equal((await context.request.get(origin+'/api/axiom2/overview')).status(),401);
 assert.deepEqual(errors,[]);
 await browser.close();
 console.log('PASS: auth/read-only, navigation, skip, desktop/iPhone, empty/loading/stale/error/reconnect, malformed payloads, multi-tab, sign-out; zero page errors.');
})().catch(e=>{console.error(e);process.exit(1)});
