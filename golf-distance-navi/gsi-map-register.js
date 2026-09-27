(function(){
"use strict";

var map=null;
var photoLayer=null;
var labelLayer=null;
var markers=[];
var hole=1;
var activeCourse=null;
var dirty=false;
var currentMarker=null;
var accuracyCircle=null;
var geoWatchId=null;
var lastPosition=null;

function $(id){return document.getElementById(id)}
function readCourses(){
  try{return JSON.parse(localStorage.getItem("gdn_courses")||"[]")}catch(e){return[]}
}
function writeCourses(courses){
  localStorage.setItem("gdn_courses",JSON.stringify(courses));
}
function getCourse(){
  var id=localStorage.getItem("gdn_course_id")||"";
  return readCourses().find(function(c){return c.id===id})||null;
}
function validPoint(p){
  return p&&Number.isFinite(Number(p.lat))&&Number.isFinite(Number(p.lng));
}
function initialCenter(course){
  if(course&&course.source&&validPoint(course.source.courseCenter)){
    return{lat:Number(course.source.courseCenter.lat),lng:Number(course.source.courseCenter.lng)};
  }
  if(course&&course.holes){
    for(var i=1;i<=18;i++){
      var h=course.holes[String(i)];
      if(h&&validPoint(h.center))return{lat:Number(h.center.lat),lng:Number(h.center.lng)};
    }
  }
  return null;
}
function setStatus(text){
  if($("gsiStatus"))$("gsiStatus").textContent=text;
}
function currentLocationIcon(){
  return L.divIcon({
    className:"",
    html:'<div style="width:20px;height:20px;border-radius:50%;background:#1976d2;border:4px solid #fff;box-shadow:0 0 0 3px rgba(25,118,210,.28),0 2px 6px rgba(0,0,0,.35)"></div>',
    iconSize:[20,20],
    iconAnchor:[10,10]
  });
}
function updateCurrentLocation(pos){
  if(!map||!pos||!pos.coords)return;
  var lat=Number(pos.coords.latitude),lng=Number(pos.coords.longitude),acc=Number(pos.coords.accuracy)||0;
  if(!Number.isFinite(lat)||!Number.isFinite(lng))return;
  lastPosition={lat:lat,lng:lng,accuracy:acc};
  if(!currentMarker){
    currentMarker=L.marker([lat,lng],{
      icon:currentLocationIcon(),
      zIndexOffset:2000,
      title:"現在地"
    }).addTo(map);
    accuracyCircle=L.circle([lat,lng],{
      radius:acc||10,
      weight:1,
      opacity:.55,
      fillOpacity:.10
    }).addTo(map);
  }else{
    currentMarker.setLatLng([lat,lng]);
    if(accuracyCircle)accuracyCircle.setLatLng([lat,lng]).setRadius(acc||10);
  }
}
function stopCurrentLocation(){
  if(geoWatchId!==null&&navigator.geolocation){
    navigator.geolocation.clearWatch(geoWatchId);
    geoWatchId=null;
  }
  if(map&&currentMarker){map.removeLayer(currentMarker)}
  if(map&&accuracyCircle){map.removeLayer(accuracyCircle)}
  currentMarker=null;
  accuracyCircle=null;
}
function startCurrentLocation(){
  stopCurrentLocation();
  if(!navigator.geolocation){
    setStatus("位置情報に対応していない端末です。");
    return;
  }
  geoWatchId=navigator.geolocation.watchPosition(
    updateCurrentLocation,
    function(e){
      if(e&&e.code===1){
        setStatus(activeCourse.name+" / 位置情報を許可すると現在地を航空写真上に表示できます。");
      }
    },
    {enableHighAccuracy:true,maximumAge:1000,timeout:15000}
  );
}
function addLocateControl(){
  if(!map)return;
  var ctl=L.control({position:"topright"});
  ctl.onAdd=function(){
    var b=L.DomUtil.create("button","");
    b.type="button";
    b.textContent="現在地";
    b.title="現在地へ移動";
    b.style.cssText="background:#fff;border:2px solid rgba(0,0,0,.2);border-radius:5px;padding:8px 10px;font-weight:800;cursor:pointer;";
    L.DomEvent.disableClickPropagation(b);
    L.DomEvent.on(b,"click",function(){
      if(lastPosition){
        map.setView([lastPosition.lat,lastPosition.lng],18);
      }else{
        setStatus("現在地を取得しています…");
      }
    });
    return b;
  };
  ctl.addTo(map);
}
function markerIcon(n){
  return L.divIcon({
    className:"",
    html:'<div class="gmark gc">'+n+'</div>',
    iconSize:[30,30],
    iconAnchor:[15,15]
  });
}
function clearMarkers(){
  markers.forEach(function(m){if(map)map.removeLayer(m)});
  markers=[];
}
function renderMarkers(){
  if(!map||!activeCourse)return;
  clearMarkers();
  for(var i=1;i<=18;i++){
    var h=activeCourse.holes&&activeCourse.holes[String(i)];
    if(!h||!validPoint(h.center))continue;
    var m=L.marker([Number(h.center.lat),Number(h.center.lng)],{
      icon:markerIcon(i),
      title:i+"H グリーン中央"
    }).addTo(map);
    markers.push(m);
  }
}
function updateHoleUi(){
  if($("gsiHole"))$("gsiHole").textContent=hole+"H：グリーン中央をタップ";
  if($("gsiPrev"))$("gsiPrev").disabled=hole<=1;
  if($("gsiNext"))$("gsiNext").disabled=hole>=18;
  setStatus(activeCourse?activeCourse.name+" / "+hole+"H のグリーン中央を航空写真上でタップしてください":"コース未選択");
}
function saveCenter(lat,lng){
  var courses=readCourses();
  var idx=courses.findIndex(function(c){return c.id===activeCourse.id});
  if(idx<0)return;
  var c=courses[idx];
  c.holes=c.holes||{};
  c.holes[String(hole)]=c.holes[String(hole)]||{};
  c.holes[String(hole)].center={lat:Number(lat),lng:Number(lng)};
  c.holes[String(hole)].manualSource="gsi_seamlessphoto";
  c.source=c.source||{};
  if(!validPoint(c.source.courseCenter)){
    c.source.courseCenter={lat:Number(lat),lng:Number(lng)};
  }
  writeCourses(courses);
  activeCourse=c;
  localStorage.setItem("gdn_hole",String(hole));
  dirty=true;
  renderMarkers();
  if(hole<18){
    hole++;
    updateHoleUi();
  }else{
    updateHoleUi();
    setStatus(activeCourse.name+" / 18Hまで登録しました。必要なホールは再タップで上書きできます。");
  }
}
async function findCourseCenterFromIndex(course){
  if(!course||!course.name)return null;
  try{
    var r=await fetch("./golf-courses-index.json",{cache:"no-store"});
    if(!r.ok)return null;
    var data=await r.json(),arr=Array.isArray(data)?data:(Array.isArray(data.courses)?data.courses:[]);
    var nq=String(course.name||"").normalize("NFKC").toLowerCase().replace(/[\\s　・･\\-_/()（）]/g,"");
    function n(v){return String(v||"").normalize("NFKC").toLowerCase().replace(/[\\s　・･\\-_/()（）]/g,"")}
    var candidates=arr.filter(function(x){return Number.isFinite(Number(x.lat))&&Number.isFinite(Number(x.lon))});
    var hit=candidates.find(function(x){return n(x.name)===nq});
    if(!hit)hit=candidates.find(function(x){var xn=n(x.name);return xn.indexOf(nq)===0||nq.indexOf(xn)===0});
    if(!hit)hit=candidates.find(function(x){var xn=n(x.name);return xn.indexOf(nq)>=0||nq.indexOf(xn)>=0});
    if(!hit)return null;
    var p={lat:Number(hit.lat),lng:Number(hit.lon)};
    var courses=readCourses(),idx=courses.findIndex(function(c){return c.id===course.id});
    if(idx>=0){
      courses[idx].source=courses[idx].source||{};
      courses[idx].source.courseCenter=p;
      writeCourses(courses);
      activeCourse=courses[idx];
    }
    return p;
  }catch(e){
    return null;
  }
}
async function getStartCenter(){
  var c=initialCenter(activeCourse);
  if(c)return c;
  var indexed=await findCourseCenterFromIndex(activeCourse);
  if(indexed){
    setStatus(activeCourse.name+" の位置をゴルフ場一覧から取得しました。航空写真を表示します。");
    return indexed;
  }
  if(!navigator.geolocation)return{lat:36.2048,lng:138.2529};
  return new Promise(function(resolve){
    navigator.geolocation.getCurrentPosition(
      function(p){resolve({lat:p.coords.latitude,lng:p.coords.longitude})},
      function(){resolve({lat:36.2048,lng:138.2529})},
      {enableHighAccuracy:true,timeout:7000,maximumAge:30000}
    );
  });
}
async function openRegister(){
  activeCourse=getCourse();
  if(!activeCourse){
    alert("先に登録コースを選択してください。");
    return;
  }
  hole=Math.max(1,Math.min(18,Number(localStorage.getItem("gdn_hole")||1)));
  $("gsiModal").classList.remove("hidden");
  $("gsiTitle").textContent="無料航空写真 18H登録";
  setStatus("航空写真を読み込んでいます…");

  var center=await getStartCenter();

  if(map){
    map.remove();
    map=null;
  }
  map=L.map("gsiMap",{zoomControl:true,attributionControl:true}).setView([center.lat,center.lng],17);
  photoLayer=L.tileLayer("https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg",{
    minZoom:14,
    maxZoom:18,
    maxNativeZoom:18,
    attribution:'出典：<a href="https://maps.gsi.go.jp/" target="_blank" rel="noopener">国土地理院</a>'
  }).addTo(map);
  labelLayer=L.tileLayer("https://cyberjapandata.gsi.go.jp/xyz/english/{z}/{x}/{y}.png",{
    minZoom:5,
    maxZoom:18,
    opacity:0.22,
    attribution:''
  });
  map.on("click",function(e){
    if(!e.latlng)return;
    saveCenter(e.latlng.lat,e.latlng.lng);
  });
  addLocateControl();
  startCurrentLocation();
  renderMarkers();
  updateHoleUi();
  setTimeout(function(){map.invalidateSize()},100);
}
function closeRegister(){
  $("gsiModal").classList.add("hidden");
  clearMarkers();
  stopCurrentLocation();
  if(map){map.remove();map=null}
  activeCourse=null;
  if(dirty){
    dirty=false;
    location.reload();
  }
}
function jumpToHole(n){
  hole=Math.max(1,Math.min(18,n));
  updateHoleUi();
  var h=activeCourse&&activeCourse.holes&&activeCourse.holes[String(hole)];
  if(map&&h&&validPoint(h.center)){
    map.setView([Number(h.center.lat),Number(h.center.lng)],18);
  }
}
function init(){
  if(!$("openGsiRegister"))return;
  $("openGsiRegister").addEventListener("click",openRegister);
  $("closeGsi").addEventListener("click",closeRegister);
  $("gsiPrev").addEventListener("click",function(){jumpToHole(hole-1)});
  $("gsiNext").addEventListener("click",function(){jumpToHole(hole+1)});
  $("gsiHole").addEventListener("click",function(){jumpToHole(hole)});
}
init();
})();