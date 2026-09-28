(function(){
"use strict";
var params=new URLSearchParams(location.search);
if(params.get("demo")!=="nasuogawa")return;
var id="demo-nasuogawa";
var courses=[];
try{courses=JSON.parse(localStorage.getItem("gdn_courses")||"[]")}catch(e){courses=[]}
if(!Array.isArray(courses))courses=[];
var demo={
  id:id,
  name:"那須小川ゴルフクラブ【デモ】",
  source:{
    provider:"Demo",
    officialName:"那須小川ゴルフクラブ",
    officialAddress:"栃木県那須郡那珂川町三輪1283",
    pars:[4,4,5,3,4,3,4,5,4,4,4,3,4,5,4,3,5,4]
  }
};
var i=courses.findIndex(function(c){return c.id===id});
if(i>=0)courses[i]=Object.assign({},courses[i],demo);else courses.push(demo);
localStorage.setItem("gdn_courses",JSON.stringify(courses));
var selected=localStorage.getItem("gdn_course_id")||"";
var selectedExists=courses.some(function(c){return c.id===selected});
if(!selected||!selectedExists)localStorage.setItem("gdn_course_id",id);
})();