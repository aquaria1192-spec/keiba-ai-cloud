(function(){
"use strict";
var NOMINATIM="https://nominatim.openstreetmap.org/search";
var OVERPASS="https://overpass-api.de/api/interpreter";
var CACHE_KEY="gdn_osm_search_cache_v1", LAST_KEY="gdn_osm_search_last_v1";
var results=[], selected=null;

function $(id){return document.getElementById(id)}
function read(k,d){try{var v=JSON.parse(localStorage.getItem(k));return v==null?d:v}catch(e){return d}}
function write(k,v){localStorage.setItem(k,JSON.stringify(v))}
function msg(t,kind){
 var e=$("osmMsg"); if(!e)return;
 e.textContent=t||""; e.className="notice osmmsg"+(kind?" "+kind:""); e.hidden=!t;
}
function sleep(ms){return new Promise(function(resolve){setTimeout(resolve,ms)})}
function rad(v){return v*Math.PI/180}
function hav(a,b){
 var R=6371000,p1=rad(a.lat),p2=rad(b.lat),dp=rad(b.lat-a.lat),dl=rad(b.lng-a.lng);
 var x=Math.sin(dp/2)*Math.sin(dp/2)+Math.cos(p1)*Math.cos(p2)*Math.sin(dl/2)*Math.sin(dl/2);
 return R*2*Math.atan2(Math.sqrt(x),Math.sqrt(1-x));
}
function norm(s){return String(s||"").toLowerCase().replace(/[\s　・\-_/]/g,"")}
function point(el){
 if(el.lat!=null&&el.lon!=null)return{lat:+el.lat,lng:+el.lon};
 if(el.center&&el.center.lat!=null)return{lat:+el.center.lat,lng:+el.center.lon};
 return null;
}
function geomPoints(el){
 return Array.isArray(el.geometry)?el.geometry.filter(function(p){return p&&p.lat!=null&&p.lon!=null}).map(function(p){return{lat:+p.lat,lng:+p.lon}}):[];
}
function centroid(arr){
 if(!arr.length)return null;var lat=0,lng=0;
 arr.forEach(function(p){lat+=p.lat;lng+=p.lng});return{lat:lat/arr.length,lng:lng/arr.length};
}
function xy(origin,p){
 var mLat=111320,mLng=111320*Math.cos(rad(origin.lat));
 return{x:(p.lng-origin.lng)*mLng,y:(p.lat-origin.lat)*mLat};
}
function ll(origin,x,y){
 var mLat=111320,mLng=111320*Math.cos(rad(origin.lat));
 return{lat:origin.lat+y/mLat,lng:origin.lng+x/mLng};
}
function greenExtent(center,prev,poly){
 if(!center||!prev||!poly||poly.length<3)return null;
 var pv=xy(center,prev),len=Math.hypot(-pv.x,-pv.y);
 if(!len)return null;
 var dx=(-pv.x)/len,dy=(-pv.y)/len,min=Infinity,max=-Infinity;
 poly.forEach(function(p){var q=xy(center,p),d=q.x*dx+q.y*dy;if(d<min)min=d;if(d>max)max=d});
 if(!Number.isFinite(min)||!Number.isFinite(max)||max-min<2)return null;
 min=Math.max(min,-80);max=Math.min(max,80);
 return{front:ll(center,dx*min,dy*min),back:ll(center,dx*max,dy*max)};
}
function nearest(elements,p,maxM){
 var best=null,bd=maxM;
 elements.forEach(function(e){
   var c=e._center;if(!c)return;var d=hav(p,c);
   if(d<bd){bd=d;best=e}
 });
 return best;
}
function pickRelation(relations,courseName){
 if(!relations.length)return null;
 var n=norm(courseName),best=null,score=-1;
 relations.forEach(function(r){
   var rn=norm(r.tags&&r.tags.name),s=0;
   if(rn&&n&&(n.indexOf(rn)>=0||rn.indexOf(n)>=0))s+=10;
   if(r.tags&&r.tags["golf:course"])s+=2;
   if(Array.isArray(r.members))s+=1;
   if(s>score){score=s;best=r}
 });
 return best;
}
function parsePar(tags){
 if(!tags)return null;var n=parseInt(tags.par,10);return n>=3&&n<=6?n:null;
}
function buildCourse(data,place){
 var center={lat:+place.lat,lng:+place.lon};
 var holes=data.elements.filter(function(e){return e.type==="way"&&e.tags&&e.tags.golf==="hole"&&geomPoints(e).length>=2});
 var greens=data.elements.filter(function(e){return e.type==="way"&&e.tags&&e.tags.golf==="green"&&geomPoints(e).length>=3});
 var pins=data.elements.filter(function(e){return e.tags&&e.tags.golf==="pin"});
 var relations=data.elements.filter(function(e){return e.type==="relation"&&e.tags&&e.tags.golf==="course"});
 greens.forEach(function(e){e._geom=geomPoints(e);e._center=centroid(e._geom)});
 pins.forEach(function(e){e._center=point(e)});
 var rel=pickRelation(relations,place.display_name||"");
 if(rel&&Array.isArray(rel.members)){
   var ids=new Set(rel.members.filter(function(m){return m.type==="way"}).map(function(m){return m.ref}));
   var inRel=holes.filter(function(h){return ids.has(h.id)});
   if(inRel.length>=6)holes=inRel;
 }
 var byRef={};
 holes.forEach(function(h){
   var ref=parseInt(h.tags&&h.tags.ref,10);if(!(ref>=1&&ref<=18))return;
   var g=geomPoints(h),end=g[g.length-1],d=hav(center,end);
   if(!byRef[ref]||d<byRef[ref]._d)byRef[ref]={el:h,geom:g,_d:d};
 });
 var out={},frontN=0,centerN=0,backN=0;
 Object.keys(byRef).forEach(function(k){
   var rec=byRef[k],g=rec.geom,end=g[g.length-1],prev=g[g.length-2],pin=nearest(pins,end,65),centerPoint=pin?pin._center:end;
   var green=nearest(greens,centerPoint,100),ext=green?greenExtent(centerPoint,prev,green._geom):null;
   var h={center:centerPoint,par:parsePar(rec.el.tags),osmHoleId:rec.el.id};
   centerN++;
   if(ext&&ext.front){h.front=ext.front;frontN++}
   if(ext&&ext.back){h.back=ext.back;backN++}
   out[String(k)]=h;
 });
 var name=(place.name||String(place.display_name||"").split(",")[0]||"OpenStreetMapコース").trim();
 return{
   course:{id:"osm-"+String(place.osm_type||"x")+"-"+String(place.osm_id||Date.now()),name:name,holes:out,source:{provider:"OpenStreetMap",nominatimOsmType:place.osm_type||"",nominatimOsmId:place.osm_id||"",importedAt:new Date().toISOString()}},
   stats:{holes:Object.keys(out).length,front:frontN,center:centerN,back:backN,relation:rel&&rel.tags?rel.tags.name||"":""}
 };
}
function resultLabel(p){
 var name=(p.name||String(p.display_name||"").split(",")[0]||"名称不明").trim();
 var rest=String(p.display_name||"").split(",").slice(1,4).join(",");
 return{name:name,rest:rest};
}
function renderResults(){
 var box=$("osmResults");box.innerHTML="";
 if(!results.length){box.innerHTML='<div class="empty">候補が見つかりませんでした。正式名称や市町村名も一緒に入力してください。</div>';return}
 results.forEach(function(p,i){
   var l=resultLabel(p),row=document.createElement("div");row.className="osmrow";
   var text=document.createElement("div"),title=document.createElement("div"),sub=document.createElement("div"),b=document.createElement("button");
   title.className="main";title.textContent=l.name;sub.className="meta";sub.textContent=l.rest;text.append(title,sub);
   b.type="button";b.className="mini";b.textContent="選択";
   b.addEventListener("click",function(){selectPlace(i)});
   row.append(text,b);box.appendChild(row);
 });
}
function selectPlace(i){
 selected=results[i]||null;
 document.querySelectorAll(".osmrow").forEach(function(r,j){r.classList.toggle("selected",j===i)});
 if(!selected)return;
 var l=resultLabel(selected);$("osmSelected").textContent=l.name;$("osmImport").disabled=false;
 msg("「"+l.name+"」を選択しました。コース詳細を取得できます。","okmsg");
}
async function search(){
 var q=$("osmQuery").value.trim();if(q.length<2){msg("ゴルフ場名を2文字以上入力してください。","errmsg");return}
 $("osmSearch").disabled=true;$("osmImport").disabled=true;selected=null;msg("ゴルフ場を検索しています…","");
 try{
   var cache=read(CACHE_KEY,{}),key=q.toLowerCase(),cached=cache[key];
   if(cached&&Date.now()-cached.time<1000*60*60*24*30){
     results=cached.results||[];renderResults();msg("端末に保存した検索結果を表示しています。","okmsg");return;
   }
   var last=Number(localStorage.getItem(LAST_KEY)||0),wait=1100-(Date.now()-last);if(wait>0)await sleep(wait);
   localStorage.setItem(LAST_KEY,String(Date.now()));
   var url=NOMINATIM+"?format=jsonv2&limit=8&countrycodes=jp&addressdetails=1&extratags=1&q="+encodeURIComponent(q);
   var resp=await fetch(url,{headers:{"Accept":"application/json"}});
   if(!resp.ok)throw new Error("検索サービス HTTP "+resp.status);
   var raw=await resp.json();
   results=(Array.isArray(raw)?raw:[]).filter(function(p){
     var t=(String(p.type||"")+" "+String(p.category||"")+" "+String(p.display_name||"")).toLowerCase();
     return t.indexOf("golf")>=0||t.indexOf("ゴルフ")>=0||t.indexOf("カントリー")>=0||t.indexOf("cc")>=0;
   });
   if(!results.length)results=(Array.isArray(raw)?raw:[]).slice(0,8);
   cache[key]={time:Date.now(),results:results};
   var keys=Object.keys(cache).sort(function(a,b){return cache[b].time-cache[a].time}).slice(0,30),small={};keys.forEach(function(k){small[k]=cache[k]});write(CACHE_KEY,small);
   renderResults();msg(results.length?results.length+"件の候補が見つかりました。":"候補が見つかりませんでした。",results.length?"okmsg":"errmsg");
 }catch(e){msg("検索できませんでした："+e.message,"errmsg")}
 finally{$("osmSearch").disabled=false}
}
async function importSelected(){
 if(!selected)return;
 $("osmImport").disabled=true;$("osmSearch").disabled=true;msg("ホール情報を取得しています。公開データ量によっては数秒かかります…","");
 try{
   var lat=+selected.lat,lon=+selected.lon,radius=4500;
   var q='[out:json][timeout:25];('+
     'way(around:'+radius+','+lat+','+lon+')["golf"="hole"];'+
     'way(around:'+radius+','+lat+','+lon+')["golf"="green"];'+
     'node(around:'+radius+','+lat+','+lon+')["golf"="pin"];'+
     'way(around:'+radius+','+lat+','+lon+')["golf"="pin"];'+
     'relation(around:'+radius+','+lat+','+lon+')["type"="golf"]["golf"="course"];'+
   ');out tags geom center;';
   var resp=await fetch(OVERPASS+"?data="+encodeURIComponent(q),{headers:{"Accept":"application/json"}});
   if(!resp.ok)throw new Error("コース詳細サービス HTTP "+resp.status);
   var data=await resp.json(),built=buildCourse(data,selected);
   if(!built.stats.holes)throw new Error("このゴルフ場にはホール番号付きデータが登録されていません。手動登録をご利用ください。");
   var courses=read("gdn_courses",[]),idx=courses.findIndex(function(c){return c.id===built.course.id});
   if(idx>=0)courses[idx]=built.course;else courses.push(built.course);
   write("gdn_courses",courses);localStorage.setItem("gdn_course_id",built.course.id);localStorage.setItem("gdn_hole","1");
   var s=built.stats,text="取得完了："+s.holes+"ホール / 中央 "+s.center+" / 手前 "+s.front+" / 奥 "+s.back;
   if(s.holes<18||s.front<s.holes||s.back<s.holes)text+="。未取得地点はコース画面の「地図で設定」で補正できます。";
   alert(text);location.reload();
 }catch(e){msg("自動取得できませんでした："+e.message,"errmsg");$("osmImport").disabled=false}
 finally{$("osmSearch").disabled=false}
}
function init(){
 if(!$("osmSearch"))return;
 $("osmSearch").addEventListener("click",search);
 $("osmQuery").addEventListener("keydown",function(e){if(e.key==="Enter"){e.preventDefault();search()}});
 $("osmImport").addEventListener("click",importSelected);
 msg("正式なゴルフ場名、または「ゴルフ場名 市町村」で検索してください。","");
}
init();
})();