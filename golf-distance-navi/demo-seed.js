(function(){
"use strict";

var params=new URLSearchParams(location.search);
if(params.get("demo")!=="nasuogawa")return;

var id="demo-nasuogawa";
var holesData=[
  [1,4,360,352],[2,4,404,388],[3,5,511,496],[4,3,183,165],[5,4,400,387],[6,3,204,186],
  [7,4,403,376],[8,5,525,515],[9,4,379,361],[10,4,326,306],[11,4,428,410],[12,3,183,154],
  [13,4,388,361],[14,5,493,482],[15,4,411,400],[16,3,231,177],[17,5,582,523],[18,4,374,350]
];

var courses=[];
try{courses=JSON.parse(localStorage.getItem("gdn_courses")||"[]")}catch(e){courses=[]}
if(!Array.isArray(courses))courses=[];

var existing=courses.find(function(c){return c.id===id});
var holes=existing&&existing.holes?existing.holes:{};
holesData.forEach(function(r){
  var k=String(r[0]),old=holes[k]||{};
  old.par=r[1];
  old.backYards=r[2];
  old.regYards=r[3];
  holes[k]=old;
});

var demo={
  id:id,
  name:"那須小川ゴルフクラブ【デモ】",
  holes:holes,
  demo:true,
  source:{
    provider:"Demo / 国土地理院航空写真",
    officialName:"那須小川ゴルフクラブ",
    officialAddress:"栃木県那須郡那珂川町三輪1283",
    courseCenter:{lat:36.749444,lng:140.098611},
    note:"PAR・ヤードは公式コースページ掲載値。グリーン座標は航空写真で登録します。"
  }
};

var idx=courses.findIndex(function(c){return c.id===id});
if(idx>=0)courses[idx]=demo;else courses.push(demo);
localStorage.setItem("gdn_courses",JSON.stringify(courses));
localStorage.setItem("gdn_course_id",id);
if(!localStorage.getItem("gdn_hole"))localStorage.setItem("gdn_hole","1");
sessionStorage.setItem("gdn_demo_nasuogawa","1");
})();