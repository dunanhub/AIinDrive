<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  // ожидание: { left,right,front,rear,roof } в процентах 0..100
  scores: Record<string, number>
  overall: number
  imgSrc?: string
}>(), {
  imgSrc: '/car.png'
})

// Функция для определения цвета на основе процента
const getColorByScore = (score: number) => {
  if (score <= 30) return '#ef4444' // красный (0-30%)
  if (score <= 50) return '#f97316' // оранжевый (31-50%)
  if (score <= 70) return '#eab308' // желтый (51-70%)
  if (score <= 85) return '#22c55e' // зеленый (71-85%)
  return '#10b981' // темно-зеленый (86-100%)
}

// Computed для цветов каждой зоны
const leftColor = computed(() => getColorByScore(props.scores.left ?? 0))
const rightColor = computed(() => getColorByScore(props.scores.right ?? 0))
const frontColor = computed(() => getColorByScore(props.scores.front ?? 0))
const rearColor = computed(() => getColorByScore(props.scores.rear ?? 0))
const roofColor = computed(() => getColorByScore(props.scores.roof ?? 0))
const overallColor = computed(() => getColorByScore(props.overall ?? 0))
</script>

<template>
  <div class="car-wrap">
    <img :src="imgSrc" alt="Car top" class="car"/>

    <!-- Цветовая легенда -->
    <div class="color-legend">
      <div class="legend-item">
        <div class="legend-dot" style="background: #ef4444;"></div>
        <span>0-30%</span>
      </div>
      <div class="legend-item">
        <div class="legend-dot" style="background: #f97316;"></div>
        <span>31-50%</span>
      </div>
      <div class="legend-item">
        <div class="legend-dot" style="background: #eab308;"></div>
        <span>51-70%</span>
      </div>
      <div class="legend-item">
        <div class="legend-dot" style="background: #22c55e;"></div>
        <span>71-85%</span>
      </div>
      <div class="legend-item">
        <div class="legend-dot" style="background: #10b981;"></div>
        <span>86-100%</span>
      </div>
    </div>

    <!-- стрелочные выноски -->
    <div class="callout left"  style="top:50%; left:23%; --ang:-135deg; --len:100px; --dx:-140px; --dy:0;">
      <span class="dot" :style="{ backgroundColor: leftColor }"></span>
      <span class="line" :style="{ backgroundColor: leftColor }"></span>
      <span class="badge left-badge" :style="{ borderColor: leftColor }">
        <b :style="{ color: leftColor }">{{ Math.round(scores.left ?? 0) }}%</b>
        <small>левая сторона</small>
      </span>
    </div>

    <div class="callout right" style="top:50%; left:77%; --ang:45deg; --len:100px; --dx:140px; --dy:0;">
      <span class="dot" :style="{ backgroundColor: rightColor }"></span>
      <span class="line" :style="{ backgroundColor: rightColor }"></span>
      <span class="badge right-badge" :style="{ borderColor: rightColor }">
        <b :style="{ color: rightColor }">{{ Math.round(scores.right ?? 0) }}%</b>
        <small>правая сторона</small>
      </span>
    </div>

    <div class="callout front" style="top:10%; left:50%; --ang:-15deg; --len:100px; --dx:0; --dy:-120px;">
      <span class="dot" :style="{ backgroundColor: frontColor }"></span>
      <span class="line" :style="{ backgroundColor: frontColor }"></span>
      <span class="badge front-badge" :style="{ borderColor: frontColor }">
        <b :style="{ color: frontColor }">{{ Math.round(scores.front ?? 0) }}%</b>
        <small>перед</small>
      </span>
    </div>

    <div class="callout rear"  style="top:88%; left:50%; --ang:-160deg; --len:150px; --dx:0; --dy:120px;">
      <span class="dot" :style="{ backgroundColor: rearColor }"></span>
      <span class="line" :style="{ backgroundColor: rearColor }"></span>
      <span class="badge rear-badge" :style="{ borderColor: rearColor }">
        <b :style="{ color: rearColor }">{{ Math.round(scores.rear ?? 0) }}%</b>
        <small>зад</small>
      </span>
    </div>

    <div class="callout roof"  style="top:55%; left:50%; --ang:-30deg; --len:180px; --dx:70px; --dy:-40px;">
      <span class="dot" :style="{ backgroundColor: roofColor }"></span>
      <span class="line" :style="{ backgroundColor: roofColor }"></span>
      <span class="badge roof-badge" :style="{ borderColor: roofColor }">
        <b :style="{ color: roofColor }">{{ Math.round(scores.roof ?? 0) }}%</b>
        <small>крыша</small>
      </span>
    </div>

    <!-- общий процент кружком в правом-низу -->
    <div class="overall">
      <div class="ring" :style="{ borderColor: overallColor, color: overallColor }">
        {{ Math.round(overall) }}%
        <div class="ring-fill" :style="{ borderColor: overallColor }"></div>
      </div>
      <div class="lbl">итог</div>
    </div>
  </div>
</template>

<style scoped>
.car-wrap{ position:relative; width:100%; max-width:200px; margin:auto; padding: 50px; }
.car{ width:100%; display:block; filter:drop-shadow(0 12px 28px rgba(0,0,0,.08)); }

/* Цветовая легенда */
.color-legend {
  position: absolute;
  bottom: 0px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 8px;
  background: rgba(255, 255, 255, 0.95);
  padding: 8px 12px;
  border-radius: 20px;
  box-shadow: 0 2px 8px rgba(0,0,0,.1);
  font-size: 0.75rem;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 4px;
}

.legend-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.legend-item span {
  color: #666;
  font-weight: 500;
}

.callout{ position:absolute; transform:translate(-50%,-50%); }
.dot{
  position:absolute; width:10px; height:10px; border-radius:50%;
  transform:translate(-50%,-50%);
}
.line{
  position:absolute; height:2px; width:var(--len,100px);
  transform-origin:left center;
  transform:translateY(-50%) rotate(var(--ang,0deg));
  left:0; top:0;
}
.badge{
  position:absolute; transform:translate(var(--dx,0), var(--dy,0));
  background:#fff; border:2px solid; border-radius:10px;
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
.badge b{ font-size:1.05rem; line-height:1; font-weight: 800; }
.badge small{ color:#666; font-size:.8rem; line-height:1; }

/* общий кружок (правый-низ) */
.overall{
  position:absolute; right:2%; bottom:3%; display:grid; place-items:center; text-align:center; transform: translate( 140px, 20px);
}
.ring{
  width:90px; height:90px; border-radius:999px; background:#fff; font-weight:800;
  display:grid; place-items:center; border:2px solid; box-shadow:0 3px 12px rgba(0,0,0,.08);
  position: relative;
}
.ring-fill{ 
  content:''; position:absolute; inset:10px; border-radius:inherit; 
  border:8px solid; height: 56px; width: 56px;
}
.lbl{ margin-top: -5px; color:#666; font-size: 2rem; }
@media (max-width:480px){
  .badge{ min-width:84px; }
  .ring{ width:76px; height:76px; }
  .ring-fill{ height: 42px; width: 42px; }
}
</style>