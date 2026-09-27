(function(){
"use strict";

var KEY_STORAGE="gdn_google_maps_key";
var map=null;
var markers=[];
var hole=1;
var activeCourse=null;
var dirty=false;
var loaderPromise=null;

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
function getInitialCenter(course){
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
function loadGoogleMaps(key){
  if(window.google&&google.maps)return Promise.resolve();
  if(loaderPromise)return loaderPromise;
  loaderPromise=new Promise(function(resolve,reject){
    window.__gdnGoogleMapsReady=function(){resolve()};
    var sc=document.createElement("script");
    sc.src="https://maps.googleapis.com/maps/api/js?key="+encodeURIComponent(key)+"&v=weekly&language=ja&region=JP&callback=__gdnGoogleMapsReady";
    sc.async=true;sc.defer=true;
    sc.onerror=function(){reject(new Error("Google Maps APIを読み込めませんでした"))};
    document.head.appendChild(sc);
    setTimeout(function(){
      if(!(window.google&&google.maps))reject(new Error("Google Maps APIの読込がタイムアウトしました"));
    },20000);
  });
  return loaderPromise;
}
function setStatus(t){
  if($("gmapStatus"))$("gmapStatus").textContent=t;
}
function clearMarkers(){
  markers.forEach(function(m){m.setMap(null)});
  markers=[];
}
function renderMarkers(){
  if(!map||!activeCourse)return;
  clearMarkers();
  for(var i=1;i<=18;i++){
    var h=activeCourse.holes&&activeCourse.holes[String(i)];
    if(!h||!validPoint(h.center))continue;
    var m=new google.maps.Marker({
      map:map,
      position:{lat:Number(h.center.lat),lng:Number(h.center.lng)},
      label:{text:String(i),color:"#ffffff",fontWeight:"700"},
      title:i+"H グリーン中央"
    });
    markers.push(m);
  }
}
function updateHoleUi(){
  if($("gmapHole"))$("gmapHole").textContent=hole+"H：グリーン中央をタップ";
  if($("gmapPrev"))$("gmapPrev").disabled=hole<=1;
  if($("gmapNext"))$("gmapNext").disabled=hole>=18;
  setStatus(activeCourse?activeCourse.name+" / "+hole+"H のグリーン中央をタップしてください":"コース未選択");
}
function saveCenter(lat,lng){
  var courses=readCourses();
  var idx=courses.findIndex(function(c){return c.id===activeCourse.id});
  if(idx<0)return;
  var c=courses[idx];
  c.holes=c.holes||{};
  c.holes[String(hole)]=c.holes[String(hole)]||{};
  c.holes[String(hole)].center={lat:lat,lng:lng};
  c.holes[String(hole)].manualSource="google_maps_satellite";
  c.source=c.source||{};
  if(!validPoint(c.source.courseCenter))c.source.courseCenter={lat:lat,lng:lng};
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
function currentOrCourseCenter(){
  return new Promise(function(resolve){
    var c=getInitialCenter(activeCourse);
    if(c){resolve(c);return}
    if(!navigator.geolocation){resolve({lat:36.2048,lng:138.2529});return}
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
  var key=(localStorage.getItem(KEY_STORAGE)||"").trim();
  if(!key){
    alert("Google Maps APIキーを入力して「APIキー保存」を押してください。");
    $("googleMapsKey").focus();
    return;
  }
  hole=Math.max(1,Math.min(18,Number(localStorage.getItem("gdn_hole")||1)));
  $("gmapModal").classList.remove("hidden");
  $("gmapTitle").textContent="Google Maps 18H登録";
  setStatus("Google Mapsを読み込んでいます…");
  try{
    await loadGoogleMaps(key);
    var center=await currentOrCourseCenter();
    map=new google.maps.Map($("googleMap"),{
      center:center,
      zoom:17,
      mapTypeId:"satellite",
      mapTypeControl:true,
      streetViewControl:false,
      fullscreenControl:false,
      gestureHandling:"greedy"
    });
    map.addListener("click",function(e){
      if(!e.latLng)return;
      saveCenter(e.latLng.lat(),e.latLng.lng());
    });
    renderMarkers();
    updateHoleUi();
  }catch(e){
    setStatus("Google Mapsを表示できません："+e.message+"。APIキー・課金設定・API制限を確認してください。");
  }
}
function closeRegister(){
  $("gmapModal").classList.add("hidden");
  clearMarkers();
  map=null;
  activeCourse=null;
  if(dirty){
    dirty=false;
    location.reload();
  }
}
function init(){
  if(!$("openGoogleRegister"))return;
  $("googleMapsKey").value=localStorage.getItem(KEY_STORAGE)||"";
  $("saveGoogleMapsKey").addEventListener("click",function(){
    var key=$("googleMapsKey").value.trim();
    if(!key){
      localStorage.removeItem(KEY_STORAGE);
      alert("APIキーを消去しました。");
      return;
    }
    localStorage.setItem(KEY_STORAGE,key);
    alert("Google Maps APIキーをこの端末に保存しました。");
  });
  $("openGoogleRegister").addEventListener("click",openRegister);
  $("closeGmap").addEventListener("click",closeRegister);
  $("gmapPrev").addEventListener("click",function(){if(hole>1){hole--;updateHoleUi()}});
  $("gmapNext").addEventListener("click",function(){if(hole<18){hole++;updateHoleUi()}});
  $("gmapHole").addEventListener("click",function(){
    var h=activeCourse&&activeCourse.holes&&activeCourse.holes[String(hole)];
    if(map&&h&&validPoint(h.center)){
      map.panTo({lat:Number(h.center.lat),lng:Number(h.center.lng)});
      map.setZoom(19);
    }
  });
}
init();
})();