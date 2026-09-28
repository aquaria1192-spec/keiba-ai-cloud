(function(){
"use strict";

var INDEX_URL="./golf-courses-index.json";
var SEED_URL="./golf-courses-seed.json";
var results=[];
var selected=null;
var courseIndex=[];
var indexPromise=null;

function $(id){return document.getElementById(id)}
function read(k,d){try{var v=JSON.parse(localStorage.getItem(k));return v==null?d:v}catch(e){return d}}
function write(k,v){localStorage.setItem(k,JSON.stringify(v))}
function norm(s){return String(s||"").normalize("NFKC").toLowerCase().replace(/[\s　・･\-_/()（）]/g,"")}
function msg(t,kind){var e=$("osmMsg");if(!e)return;e.textContent=t||"";e.className="notice osmmsg"+(kind?" "+kind:"");e.hidden=!t}
function nameValues(p){var n=p.names||{};return[p.name,n.name,n["name:ja"],n.official_name,n.alt_name,n.short_name,n.old_name].filter(Boolean)}
function nameContains(p,q){var nq=norm(q);return nameValues(p).some(function(v){return norm(v).indexOf(nq)>=0})}
function score(p,q){var nq=norm(q),best=0;nameValues(p).forEach(function(v){var n=norm(v);if(n===nq)best=Math.max(best,100);else if(n.indexOf(nq)===0)best=Math.max(best,80);else if(n.indexOf(nq)>=0)best=Math.max(best,60)});return best}
function indexToPlace(x){
  var names=x.names||{},name=x.name||names.name||names["name:ja"]||"名称未設定",addr=x.addr||{};
  return{osm_type:x.osm_type||"local",osm_id:x.osm_id||name,name:name,lat:x.lat,lon:x.lon,names:names,display_name:[name,addr.prefecture,addr.city].filter(Boolean).join(", ")};
}
function normalizeIndex(payload){
  var arr=Array.isArray(payload)?payload:(payload&&Array.isArray(payload.courses)?payload.courses:[]),m=new Map();
  arr.forEach(function(x){var p=indexToPlace(x),k=norm(p.name);if(!k)return;var old=m.get(k);if(!old||(old.lat==null&&p.lat!=null))m.set(k,p)});
  return Array.from(m.values());
}
async function loadIndex(){
  if(courseIndex.length)return courseIndex;
  if(indexPromise)return indexPromise;
  indexPromise=(async function(){
    try{var r=await fetch(INDEX_URL,{cache:"no-store"});if(!r.ok)throw new Error();courseIndex=normalizeIndex(await r.json())}
    catch(e){try{var s=await fetch(SEED_URL,{cache:"no-store"});if(!s.ok)throw new Error();courseIndex=normalizeIndex(await s.json())}catch(e2){courseIndex=[]}}
    return courseIndex;
  })();
  return indexPromise;
}
function renderResults(){
  var box=$("osmResults");box.innerHTML="";
  if(!results.length){box.innerHTML='<div class="empty">名称に一致するゴルフ場はありません。</div>';return}
  results.forEach(function(p,i){
    var row=document.createElement("div"),text=document.createElement("div"),title=document.createElement("div"),sub=document.createElement("div"),b=document.createElement("button");
    row.className="osmrow";title.className="main";sub.className="meta";title.textContent=p.name;sub.textContent=String(p.display_name||"").split(",").slice(1).join(" / ");text.append(title,sub);
    b.type="button";b.className="mini";b.textContent="選択";b.onclick=function(){selectPlace(i)};row.append(text,b);box.appendChild(row);
  });
}
function selectPlace(i){
  selected=results[i]||null;
  document.querySelectorAll(".osmrow").forEach(function(r,j){r.classList.toggle("selected",j===i)});
  if(!selected)return;
  $("osmSelected").textContent=selected.name;
  $("osmImport").disabled=false;
  msg("「"+selected.name+"」を選択しました。コースとして登録できます。","okmsg");
}
async function search(){
  var q=$("osmQuery").value.trim();
  if(!q){msg("ゴルフ場名に含まれる文字を入力してください。","errmsg");return}
  $("osmSearch").disabled=true;$("osmImport").disabled=true;selected=null;
  try{
    var list=await loadIndex();
    results=list.filter(function(p){return nameContains(p,q)}).sort(function(a,b){return score(b,q)-score(a,q)||String(a.name).localeCompare(String(b.name),"ja")}).slice(0,50);
    renderResults();
    msg(results.length?"名称に「"+q+"」を含むゴルフ場が"+results.length+"件見つかりました。":"名称に「"+q+"」を含むゴルフ場はありません。",results.length?"okmsg":"errmsg");
  }catch(e){msg("検索一覧を読み込めませんでした。","errmsg")}
  finally{$("osmSearch").disabled=false}
}
function registerSelected(){
  if(!selected)return;
  var courses=read("gdn_courses",[]);
  var id="osm-"+String(selected.osm_type||"x")+"-"+String(selected.osm_id||selected.name);
  var center=(Number.isFinite(Number(selected.lat))&&Number.isFinite(Number(selected.lon)))?{lat:Number(selected.lat),lng:Number(selected.lon)}:null;
  var course={id:id,name:selected.name,source:{provider:"OpenStreetMap",osmType:selected.osm_type||"",osmId:selected.osm_id||"",courseCenter:center,names:selected.names||{}}};
  var idx=courses.findIndex(function(c){return c.id===id});
  if(idx>=0){
    course.source=Object.assign({},courses[idx].source||{},course.source);
    courses[idx]=course;
  }else courses.push(course);
  write("gdn_courses",courses);
  localStorage.setItem("gdn_course_id",id);
  alert(selected.name+" をコースとして登録しました。");
  location.reload();
}
function init(){
  if(!$("osmSearch"))return;
  $("osmSearch").onclick=search;
  $("osmImport").onclick=registerSelected;
  $("osmQuery").addEventListener("input",function(){var self=this;clearTimeout(this._t);if(this.value.trim().length>=2)this._t=setTimeout(search,120)});
  $("osmQuery").addEventListener("keydown",function(e){if(e.key==="Enter"){e.preventDefault();search()}});
  msg("ゴルフ場名の一部を入力してください。","");
  loadIndex();
}
init();
})();