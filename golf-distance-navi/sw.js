const CACHE='gdn-mobile-v1.0.1';
const SHELL=['./','./index.html','./demo-seed.js','./osm-import.js','./golf-courses-index.json','./golf-courses-seed.json','./manifest.webmanifest','./icon.svg'];
self.addEventListener('install',e=>e.waitUntil(caches.open(CACHE).then(c=>c.addAll(SHELL)).then(()=>self.skipWaiting())));
self.addEventListener('activate',e=>e.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(k=>k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim())));
self.addEventListener('fetch',e=>{
  if(e.request.method!=='GET') return;
  const u=new URL(e.request.url);
  if(u.origin===location.origin){
    e.respondWith(fetch(e.request).then(resp=>{
      const clone=resp.clone(); caches.open(CACHE).then(c=>c.put(e.request,clone)); return resp;
    }).catch(()=>caches.match(e.request).then(r=>r||caches.match('./index.html'))));
  }
});