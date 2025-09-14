<script setup lang="ts">
const props = withDefaults(defineProps<{
  // ожидание: { left,right,front,rear,roof } в процентах 0..100
  scores: Record<string, number>
  overall: number
  imgSrc?: string
}>(), {
  imgSrc: '/car.png'
})
</script>

<template>
  <div class="car-wrap">
    <img :src="imgSrc" alt="Car top" class="car"/>

    <!-- стрелочные выноски -->
    <div class="callout left"  style="top:50%; left:23%; --ang:-135deg; --len:100px; --dx:-140px; --dy:0;">
      <span class="dot"></span><span class="line"></span>
      <span class="badge left-badge"><b>{{ Math.round(scores.left ?? 0) }}%</b><small>левая сторона</small></span>
    </div>

    <div class="callout right" style="top:50%; left:77%; --ang:45deg; --len:100px; --dx:140px; --dy:0;">
      <span class="dot"></span><span class="line"></span>
      <span class="badge right-badge"><b>{{ Math.round(scores.right ?? 0) }}%</b><small>правая сторона</small></span>
    </div>

    <div class="callout front" style="top:10%; left:50%; --ang:-15deg; --len:100px; --dx:0; --dy:-120px;">
      <span class="dot"></span><span class="line"></span>
      <span class="badge front-badge"><b>{{ Math.round(scores.front ?? 0) }}%</b><small>перед</small></span>
    </div>

    <div class="callout rear"  style="top:88%; left:50%; --ang:-160deg; --len:150px; --dx:0; --dy:120px;">
      <span class="dot"></span><span class="line"></span>
      <span class="badge rear-badge"><b>{{ Math.round(scores.rear ?? 0) }}%</b><small>зад</small></span>
    </div>

    <div class="callout roof"  style="top:55%; left:50%; --ang:-30deg; --len:180px; --dx:70px; --dy:-40px;">
      <span class="dot"></span><span class="line"></span>
      <span class="badge roof-badge"><b>{{ Math.round(scores.roof ?? 0) }}%</b><small>крыша</small></span>
    </div>

    <!-- общий процент кружком в правом-низу -->
    <div class="overall">
      <div class="ring">{{ Math.round(overall) }}%</div>
      <div class="lbl">итог</div>
    </div>
  </div>
</template>

<style scoped>
.car-wrap{ position:relative; width:100%; max-width:200px; margin:auto; padding: 50px; }
.car{ width:100%; display:block; filter:drop-shadow(0 12px 28px rgba(0,0,0,.08)); }

.callout{ position:absolute; transform:translate(-50%,-50%); }
.dot{
  position:absolute; width:10px; height:10px; background:#c1f11d; border-radius:50%;
  transform:translate(-50%,-50%);
}
.line{
  position:absolute; height:2px; width:var(--len,100px); background:#c1f11d;
  transform-origin:left center;
  transform:translateY(-50%) rotate(var(--ang,0deg));
  left:0; top:0;
}
.badge{
  position:absolute; transform:translate(var(--dx,0), var(--dy,0));
  background:#fff; border:2px solid #c1f11d; border-radius:10px;
  padding:6px 10px; box-shadow:0 3px 10px rgba(0,0,0,.08);
  display:flex; flex-direction:column; align-items:center; gap:2px; min-width:96px;
}
.left-badge {
  transform: translate(-180px, -90px);
}
.right-badge {
  transform: translate(70px, 60px);
}
.front-badge {
  transform: translate(90px, -40px);
}
.rear-badge {
  transform: translate(-250px, -80px);
}
.roof-badge {
  transform: translate(150px, -120px);
}
.badge b{ font-size:1.05rem; color:#141414; line-height:1; }
.badge small{ color:#666; font-size:.8rem; line-height:1; }

/* общий кружок (правый-низ) */
.overall{
  position:absolute; right:2%; bottom:3%; display:grid; place-items:center; text-align:center; transform: translate( 140px, 20px);
}
.ring{
  width:90px; height:90px; border-radius:999px; background:#fff; color:#141414; font-weight:800;
  display:grid; place-items:center; border:2px solid #e9ecef; box-shadow:0 3px 12px rgba(0,0,0,.08);
}
.ring::after{ content:''; position:absolute; inset:10px; border-radius:inherit; border:8px solid #c1f11d; height: 56px; }
.lbl{ margin-top: -5px; color:#666; font-size: 2rem; }
@media (max-width:480px){
  .badge{ min-width:84px; }
  .ring{ width:76px; height:76px; }
}
</style>