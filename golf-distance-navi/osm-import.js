(function(){
"use strict";

var INDEX_URL="./golf-courses-index.json";
var SEED_URL="./golf-courses-seed.json";
var NOMINATIM="https://nominatim.openstreetmap.org/search";
var OVERPASS="https://overpass-api.de/api/interpreter";
var results=[];
var selected=null;
var courseIndex=[];
var indexPromise=null;

function $(id){return document.getElementById(id)}
function read(k,d){try{var v=JSON.parse(localStorage.getItem(k));return v==null?d:v}catch(e){return d}}
function write(k,v){localStorage.setItem(k,JSON.stringify(v))}
function msg(t,kind){
  var e=$("osmMsg");
  if(!e)return;
  e.textContent=t||"";
  e.className="notice osmmsg"+(kind?" "+kind:"");
  e.hidden=!t;
}
function rad(v){return v*Math.PI/180}
function hav(a,b){
  var R=6371000,p1=rad(a.lat),p2=rad(b.lat),dp=rad(b.lat-a.lat),dl=rad(b.lng-a.lng);
  var x=Math.sin(dp/2)*Math.sin(dp/2)+Math.cos(p1)*Math.cos(p2)*Math.sin(dl/2)*Math.sin(dl/2);
  return R*2*Math.atan2(Math.sqrt(x),Math.sqrt(1-x));
}
function norm(s){
  return String(s||"").normalize("NFKC").toLowerCase().replace(/[\s　・･\-_/()（）]/g,"");
}
function nameValues(p){
  var n=p.names||p.namedetails||{};
  return [p.name,n.name,n["name:ja"],n.official_name,n.alt_name,n.short_name,n.old_name].filter(Boolean);
}
function nameContains(p,q){
  var nq=norm(q);
  return nameValues(p).some(function(v){return norm(v).indexOf(nq)>=0});
}
function matchScore(p,q){
  var nq=norm(q),best=0;
  nameValues(p).forEach(function(v){
    var n=norm(v);
    if(n===nq)best=Math.max(best,100);
    else if(n.indexOf(nq)===0)best=Math.max(best,80);
    else if(n.indexOf(nq)>=0)best=Math.max(best,60);
  });
  return best;
}
function indexToPlace(x){
  var addr=x.addr||{},names=x.names||{},name=x.name||names.name||names["name:ja"]||"名称未設定";
  var area=[addr.prefecture,addr.city].filter(Boolean).join(" ");
  return{
    osm_type:x.osm_type||"seed",
    osm_id:x.osm_id||name,
    lat:x.lat,
    lon:x.lon,
    name:name,
    display_name:area?name+", "+area:name,
    category:"leisure",
    class:"leisure",
    type:"golf_course",
    extratags:{leisure:"golf_course"},
    namedetails:names,
    names:names
  };
}
function normalizeIndex(payload){
  var arr=Array.isArray(payload)?payload:(payload&&Array.isArray(payload.courses)?payload.courses:[]);
  var byName=new Map();
  arr.forEach(function(x){
    var p=indexToPlace(x),k=norm(p.name);
    if(!k)return;
    var old=byName.get(k);
    if(!old||(old.lat==null&&p.lat!=null))byName.set(k,p);
  });
  return Array.from(byName.values());
}
async function loadIndex(){
  if(courseIndex.length)return courseIndex;
  if(indexPromise)return indexPromise;
  indexPromise=(async function(){
    try{
      var r=await fetch(INDEX_URL,{cache:"no-store"});
      if(!r.ok)throw new Error("index "+r.status);
      courseIndex=normalizeIndex(await r.json());
    }catch(e){
      try{
        var s=await fetch(SEED_URL,{cache:"no-store"});
        if(!s.ok)throw new Error("seed "+s.status);
        courseIndex=normalizeIndex(await s.json());
      }catch(e2){
        courseIndex=[];
      }
    }
    return courseIndex;
  })();
  return indexPromise;
}

function point(el){
  if(el.lat!=null&&el.lon!=null)return{lat:+el.lat,lng:+el.lon};
  if(el.center&&el.center.lat!=null)return{lat:+el.center.lat,lng:+el.center.lon};
  return null;
}
function geomPoints(el){
  return Array.isArray(el.geometry)?el.geometry.filter(function(p){return p&&p.lat!=null&&p.lon!=null}).map(function(p){return{lat:+p.lat,lng:+p.lon}}):[];
}
function centroid(arr){
  if(!arr.length)return null;
  var lat=0,lng=0;
  arr.forEach(function(p){lat+=p.lat;lng+=p.lng});
  return{lat:lat/arr.length,lng:lng/arr.length};
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
  poly.forEach(function(p){
    var q=xy(center,p),d=q.x*dx+q.y*dy;
    if(d<min)min=d;
    if(d>max)max=d;
  });
  if(!Number.isFinite(min)||!Number.isFinite(max)||max-min<2)return null;
  min=Math.max(min,-80);max=Math.min(max,80);
  return{front:ll(center,dx*min,dy*min),back:ll(center,dx*max,dy*max)};
}
function nearest(elements,p,maxM){
  var best=null,bd=maxM;
  elements.forEach(function(e){
    var c=e._center;if(!c)return;
    var d=hav(p,c);
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
    if(Array.isArray(r.members))s+=1;
    if(s>score){score=s;best=r}
  });
  return best;
}
function parsePar(tags){
  if(!tags)return null;
  var n=parseInt(tags.par,10);
  return n>=3&&n<=6?n:null;
}
function buildCourse(data,place){
  var center={lat:+place.lat,lng:+place.lon};
  var holes=data.elements.filter(function(e){return e.type==="way"&&e.tags&&e.tags.golf==="hole"&&geomPoints(e).length>=2});
  var greens=data.elements.filter(function(e){return e.type==="way"&&e.tags&&e.tags.golf==="green"&&geomPoints(e).length>=3});
  var pins=data.elements.filter(function(e){return e.tags&&e.tags.golf==="pin"});
  var relations=data.elements.filter(function(e){return e.type==="relation"&&e.tags&&(e.tags.golf==="course"||e.tags.type==="golf")});
  greens.forEach(function(e){e._geom=geomPoints(e);e._center=centroid(e._geom)});
  pins.forEach(function(e){e._center=point(e)});
  var rel=pickRelation(relations,place.display_name||place.name||"");
  if(rel&&Array.isArray(rel.members)){
    var ids=new Set(rel.members.filter(function(m){return m.type==="way"}).map(function(m){return m.ref}));
    var inRel=holes.filter(function(h){return ids.has(h.id)});
    if(inRel.length>=6)holes=inRel;
  }
  var byRef={};
  holes.forEach(function(h){
    var ref=parseInt(h.tags&&h.tags.ref,10);
    if(!(ref>=1&&ref<=18))return;
    var g=geomPoints(h),end=g[g.length-1],d=hav(center,end);
    if(!byRef[ref]||d<byRef[ref]._d)byRef[ref]={el:h,geom:g,_d:d};
  });
  var out={},frontN=0,centerN=0,backN=0;
  Object.keys(byRef).forEach(function(k){
    var rec=byRef[k],g=rec.geom,end=g[g.length-1],prev=g[g.length-2];
    var pin=nearest(pins,end,65),centerPoint=pin?pin._center:end;
    var green=nearest(greens,centerPoint,100),ext=green?greenExtent(centerPoint,prev,green._geom):null;
    var h={center:centerPoint,par:parsePar(rec.el.tags),osmHoleId:rec.el.id};
    centerN++;
    if(ext&&ext.front){h.front=ext.front;frontN++}
    if(ext&&ext.back){h.back=ext.back;backN++}
    out[String(k)]=h;
  });
  var name=(place.name||String(place.display_name||"").split(",")[0]||"OpenStreetMapコース").trim();
  return{
    course:{
      id:"osm-"+String(place.osm_type||"x")+"-"+String(place.osm_id||Date.now()),
      name:name,
      holes:out,
      source:{provider:"OpenStreetMap",osmType:place.osm_type||"",osmId:place.osm_id||"",importedAt:new Date().toISOString()}
    },
    stats:{holes:Object.keys(out).length,front:frontN,center:centerN,back:backN}
  };
}

function resultLabel(p){
  var parts=String(p.display_name||"").split(",").map(function(x){return x.trim()}).filter(Boolean);
  return{name:p.name||"名称不明",rest:parts.slice(1,4).join(" / ")};
}
function renderResults(){
  var box=$("osmResults");box.innerHTML="";
  if(!results.length){
    box.innerHTML='<div class="empty">名称に一致するゴルフ場はありません。</div>';
    return;
  }
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
  $("osmSelected").textContent=selected.name;$("osmImport").disabled=false;
  msg("「"+selected.name+"」を選択しました。コース詳細を取得できます。","okmsg");
}
async function search(){
  var q=$("osmQuery").value.trim();
  if(!q){msg("ゴルフ場名に含まれる文字を入力してください。","errmsg");return}
  $("osmSearch").disabled=true;$("osmImport").disabled=true;selected=null;
  try{
    var list=await loadIndex();
    results=list.filter(function(p){return nameContains(p,q)})
      .sort(function(a,b){
        var d=matchScore(b,q)-matchScore(a,q);
        return d||String(a.name).localeCompare(String(b.name),"ja");
      })
      .slice(0,50);
    renderResults();
    msg(results.length?
      "名称に「"+q+"」を含むゴルフ場が"+results.length+"件見つかりました。":
      "名称に「"+q+"」を含むゴルフ場はありません。",
      results.length?"okmsg":"errmsg"
    );
  }catch(e){
    msg("検索一覧を読み込めませんでした。再読み込みしてください。","errmsg");
  }finally{
    $("osmSearch").disabled=false;
  }
}
async function resolveCenter(p){
  if(Number.isFinite(+p.lat)&&Number.isFinite(+p.lon))return{lat:+p.lat,lon:+p.lon};
  var query=p.name;
  var parts=String(p.display_name||"").split(",").slice(1).join(" ");
  if(parts)query+=" "+parts;
  var url=NOMINATIM+"?format=jsonv2&limit=8&countrycodes=jp&layer=poi&extratags=1&namedetails=1&accept-language=ja&q="+encodeURIComponent(query);
  var r=await fetch(url,{headers:{"Accept":"application/json"}});
  if(!r.ok)throw new Error("位置検索 HTTP "+r.status);
  var raw=await r.json();
  var nq=norm(p.name);
  var hit=(Array.isArray(raw)?raw:[]).find(function(x){
    var n=norm(x.name||String(x.display_name||"").split(",")[0]);
    return n.indexOf(nq)>=0||nq.indexOf(n)>=0;
  });
  if(!hit||hit.lat==null||hit.lon==null)throw new Error("ゴルフ場の位置を取得できませんでした");
  p.lat=hit.lat;p.lon=hit.lon;
  return{lat:+hit.lat,lon:+hit.lon};
}
async function importSelected(){
  if(!selected)return;
  $("osmImport").disabled=true;$("osmSearch").disabled=true;
  msg("選択したゴルフ場のホール情報を取得しています…","");
  try{
    var pos=await resolveCenter(selected),lat=pos.lat,lon=pos.lon,radius=4500;
    var q='[out:json][timeout:25];('+
      'way(around:'+radius+','+lat+','+lon+')["golf"="hole"];'+
      'way(around:'+radius+','+lat+','+lon+')["golf"="green"];'+
      'node(around:'+radius+','+lat+','+lon+')["golf"="pin"];'+
      'way(around:'+radius+','+lat+','+lon+')["golf"="pin"];'+
      'relation(around:'+radius+','+lat+','+lon+')["golf"="course"];'+
      'relation(around:'+radius+','+lat+','+lon+')["type"="golf"];'+
    ');out tags geom center;';
    var resp=await fetch(OVERPASS+"?data="+encodeURIComponent(q),{headers:{"Accept":"application/json"}});
    if(!resp.ok)throw new Error("コース詳細サービス HTTP "+resp.status);
    var data=await resp.json(),built=buildCourse(data,selected);
    if(!built.stats.holes)throw new Error("このゴルフ場にはホール番号付きデータが登録されていません。手動登録をご利用ください。");
    var courses=read("gdn_courses",[]),idx=courses.findIndex(function(c){return c.id===built.course.id});
    if(idx>=0)courses[idx]=built.course;else courses.push(built.course);
    write("gdn_courses",courses);
    localStorage.setItem("gdn_course_id",built.course.id);
    localStorage.setItem("gdn_hole","1");
    var st=built.stats,text="取得完了："+st.holes+"ホール / 中央 "+st.center+" / 手前 "+st.front+" / 奥 "+st.back;
    if(st.holes<18||st.front<st.holes||st.back<st.holes)text+="。未取得地点は「地図で設定」で補正できます。";
    alert(text);location.reload();
  }catch(e){
    msg("自動取得できませんでした："+e.message,"errmsg");
    $("osmImport").disabled=false;
  }finally{
    $("osmSearch").disabled=false;
  }
}
function init(){
  if(!$("osmSearch"))return;
  $("osmSearch").addEventListener("click",search);
  $("osmQuery").addEventListener("input",function(){
    if(this.value.trim().length>=2){
      clearTimeout(this._searchTimer);
      this._searchTimer=setTimeout(search,120);
    }
  });
  $("osmQuery").addEventListener("keydown",function(e){
    if(e.key==="Enter"){e.preventDefault();search()}
  });
  $("osmImport").addEventListener("click",importSelected);
  msg("ゴルフ場一覧を読み込んでいます…","");
  loadIndex().then(function(list){
    msg("高速検索準備完了（"+list.length+"コース）。名称の一部を入力してください。","okmsg");
  });
}

init();
})();