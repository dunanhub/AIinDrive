<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import CarScan from '~/components/CarScan.vue'
import CarTopView from '~/components/CarTopView.vue'
import type { PredictResp } from '~/composables/usePredict'
import { predictMany } from '~/composables/usePredict'

type Item = { file: File; url: string; result?: PredictResp }
type Scores = Record<string, number>

const step = ref<1|2|3>(1)
const items = ref<Item[]>([])
const isDragOver = ref(false)
const scanning = ref(false)
const progress = ref(0)
const error = ref('')
const showPhotos = ref(false) // переключатель между 3D-видом и фотографиями
const selectedPhoto = ref<Item | null>(null) // для модального окна
const showModal = ref(false) // показывать ли модальное окно

function toStep1(){ step.value = 1; items.value=[]; progress.value=0; error.value=''; showPhotos.value = false; closeModal() }

function addFiles(list: FileList | null) {
  if (!list) return
  for (const f of Array.from(list)) {
    if (!f.type.startsWith('image/')) continue
    items.value.push({ file: f, url: URL.createObjectURL(f) })
  }
  if (items.value.length) step.value = 2
}
function onPick(e: Event){ addFiles((e.target as HTMLInputElement).files) }
function onDrop(e: DragEvent){ e.preventDefault(); isDragOver.value=false; addFiles(e.dataTransfer?.files ?? null) }
function onDragOver(e: DragEvent){ e.preventDefault(); isDragOver.value=true }
function onDragLeave(e: DragEvent){ e.preventDefault(); isDragOver.value=false }
function removeAt(i: number){ URL.revokeObjectURL(items.value[i].url); items.value.splice(i,1) }

const canAnalyze = computed(()=> items.value.length>0 && !scanning.value)

// Функция для переключения режима отображения
function toggleView() {
  showPhotos.value = !showPhotos.value
}

// Функции для модального окна
function openModal(item: Item, index: number) {
  selectedPhoto.value = item
  showModal.value = true
  // Блокируем скролл body
  document.body.style.overflow = 'hidden'
}

function closeModal() {
  selectedPhoto.value = null
  showModal.value = false
  // Восстанавливаем скролл body
  document.body.style.overflow = 'auto'
}

// Закрытие модалки по Escape
function handleKeydown(event: KeyboardEvent) {
  if (event.key === 'Escape' && showModal.value) {
    closeModal()
  }
}

// Монтируем обработчик клавиатуры
onMounted(() => {
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
  // Восстанавливаем скролл при размонтировании
  document.body.style.overflow = 'auto'
})

async function analyze() {
  if (!items.value.length) return
  scanning.value = true; error.value=''; progress.value=0; step.value=2
  try {
    const files = items.value.map(i=>i.file)
    const results = await predictMany(files, (p)=>{ progress.value = p })
    // раскладываем по items
    results.forEach((r,i)=> items.value[i].result = r)
    progress.value = 100
    step.value = 3
  } catch (e:any) {
    error.value = e?.message || 'Ошибка анализа'
  } finally {
    scanning.value = false
  }
}
const agg = computed(() => {
  const arr = items.value.map(i => i.result).filter(Boolean) as PredictResp[]
  const n = arr.length || 1

  const avgDirtyProb = arr.reduce((s, r) => s + (r.dirty_prob ?? 0), 0) / n
  const avgDamProb   = arr.reduce((s, r) => s + (r.damaged_prob ?? 0), 0) / n

  const cleanScore     = Math.round((1 - avgDirtyProb) * 100)
  const integrityScore = Math.round((1 - avgDamProb) * 100)
  // Взвешенная формула: Чистота 30% + Целостность 70%
  const overall        = Math.round((cleanScore * 0.3) + (integrityScore * 0.7))

  let parts: Record<string, number> = {}

  const hasParts = arr.some(r => (r as any)?.parts)
  if (hasParts) {
    // усредняем зоны, если бек вернул {left,right,front,rear,roof}
    const sums: Record<string, number> = {}
    for (const r of arr) {
      const p = (r as any)?.parts || {}
      for (const [k, v] of Object.entries(p)) {
        sums[k] = (sums[k] || 0) + Number(v)
      }
    }
    for (const [k, v] of Object.entries(sums)) {
      parts[k] = Math.round(v / arr.length)
    }
  } else {
    // синтетические значения, чтобы UI уже работал
    // Используем взвешенную формулу и для частей автомобиля
    const leftRight = Math.round((cleanScore * 0.3) + (integrityScore * 0.7))
    parts = {
      left:  leftRight,
      right: leftRight,
      front: integrityScore, // передняя часть больше связана с целостностью
      rear:  integrityScore, // задняя часть больше связана с целостностью
      roof:  Math.round((cleanScore * 0.5) + (integrityScore * 0.5)) // крыша 50/50
    }
  }

  return { cleanScore, integrityScore, overall, parts }
})

// Генерация умных рекомендаций на основе данных модели
const recommendations = computed(() => {
  const arr = items.value.map(i => i.result).filter(Boolean) as PredictResp[]
  if (arr.length === 0) return []

  const recommendations: string[] = []

  // Анализ повреждений по классам модели
  const majorDamageItems = arr.filter(r => r.predicted_class === 'major_damage')
  const minorDamageItems = arr.filter(r => r.predicted_class === 'minor_damage')
  const noDamageItems = arr.filter(r => r.predicted_class === 'no_damage')

  if (majorDamageItems.length > 0) {
    const avgConfidence = majorDamageItems.reduce((sum, r) => sum + (r.confidence || 0), 0) / majorDamageItems.length
    if (avgConfidence > 0.8) {
      recommendations.push('🚨 Обнаружены серьезные повреждения! Автомобиль НЕ ПРИГОДЕН для работы в такси без капитального ремонта.')
      recommendations.push('⚖️ Нарушение требований безопасности для коммерческих перевозок.')
    } else if (avgConfidence > 0.6) {
      recommendations.push('⚠️ Подозрение на серьезные повреждения. Требуется СРОЧНАЯ профессиональная экспертиза!')
    }
  } else if (minorDamageItems.length > 0) {
    const avgConfidence = minorDamageItems.reduce((sum, r) => sum + (r.confidence || 0), 0) / minorDamageItems.length
    if (avgConfidence > 0.7) {
      recommendations.push('🔧 Обнаружены мелкие повреждения (царапины, потертости). Косметический ремонт желателен.')
      recommendations.push('💰 Ориентировочные затраты: 30-100 тыс. руб.')
    }
  } else if (noDamageItems.length === arr.length) {
    const avgConfidence = noDamageItems.reduce((sum, r) => sum + (r.confidence || 0), 0) / noDamageItems.length
    if (avgConfidence > 0.85) {
      recommendations.push('✨ Отличное состояние! Автомобиль идеален для премиум такси-сервиса.')
      recommendations.push('🏆 Подходит для VIP и бизнес-клиентов.')
    } else if (avgConfidence > 0.7) {
      recommendations.push('✅ Хорошее состояние. Автомобиль пригоден для работы в такси.')
    }
  }

  // Анализ загрязненности на основе dirt_metrics
  const avgDirtScore = arr.reduce((sum, r) => sum + (r.dirt_metrics?.dirt_score || 0), 0) / arr.length
  if (avgDirtScore > 6) {
    recommendations.push('🧼 КРИТИЧЕСКОЕ ЗАГРЯЗНЕНИЕ: Автомобиль слишком грязный для перевозки пассажиров.')
    recommendations.push('📉 Нарушение стандартов имиджа такси-сервиса. Требуется немедленная профессиональная мойка.')
  } else if (avgDirtScore > 4) {
    recommendations.push('🧽 Рекомендуется комплексная мойка перед выходом на линию.')
    recommendations.push('💰 Затраты: 1.5-3 тыс. руб. на качественную мойку.')
  } else if (avgDirtScore < 2) {
    recommendations.push('✨ Превосходная чистота! Автомобиль содержится в идеальном состоянии.')
  }

  // Общие рекомендации по уверенности модели
  const avgModelConfidence = arr.reduce((sum, r) => sum + (r.confidence || 0), 0) / arr.length
  if (avgModelConfidence < 0.6) {
    recommendations.push('❓ Низкая уверенность ИИ-анализа. Рекомендуется дополнительная экспертная оценка.')
  }

  // Если модель недоступна
  if (arr.some(r => r.model_available === false)) {
    recommendations.push('⚠️ ИИ-модель временно недоступна. Результат основан только на анализе загрязненности.')
  }

  // Пороговые рекомендации (как резерв, если специфичных мало)
  if (recommendations.length === 0) {
    if (agg.value.cleanScore < 85) {
      recommendations.push('🧼 Рекомендуем мойку кузова и стёкол.')
    }
    if (agg.value.integrityScore < 85) {
      recommendations.push('🔧 Проверьте сколы, царапины и работу фар.')
    }
    if (agg.value.overall >= 85) {
      recommendations.push('✅ Состояние в норме — можно выходить на линию.')
    }
  }

  return recommendations.slice(0, 4) // Ограничиваем до 4 рекомендаций
})

const brand = { green: '#c1f11d', black: '#141414', white: '#ffffff' }

const passed = computed(() => agg.value.overall >= 80)

// Функция для определения цвета на основе процента
const getColorByScore = (score: number) => {
  if (score <= 30) return '#ef4444' // красный (0-30%)
  if (score <= 50) return '#f97316' // оранжевый (31-50%)
  if (score <= 70) return '#eab308' // желтый (51-70%)
  if (score <= 85) return '#22c55e' // зеленый (71-85%)
  return '#10b981' // темно-зеленый (86-100%)
}

// Computed для цветов чипов
const cleanScoreColor = computed(() => getColorByScore(agg.value.cleanScore))
const integrityScoreColor = computed(() => getColorByScore(agg.value.integrityScore))
const overallScoreColor = computed(() => getColorByScore(agg.value.overall))

// Функции для отображения данных модели
function getDamageClass(predictedClass?: string): string {
  switch (predictedClass) {
    case 'no_damage': return 'damage-none'
    case 'minor_damage': return 'damage-minor'
    case 'major_damage': return 'damage-major'
    default: return 'damage-unknown'
  }
}

function getDamageLabel(predictedClass?: string): string {
  switch (predictedClass) {
    case 'no_damage': return 'Без повреждений'
    case 'minor_damage': return 'Мелкие повреждения'
    case 'major_damage': return 'Серьезные повреждения'
    default: return 'Неизвестно'
  }
}
</script>

<template>
  <Header />
  <div class="page">

    <main>
      <h1>Проверка состояния авто</h1>
      <p class="lead">Загрузите фото (одно или несколько) — мы определим <b>повреждения</b> и <b>чистоту</b>.</p>

      <!-- Шаги -->
      <ol class="steps">
        <li :class="{active: step===1, done: step>1}"><span>1</span> Загрузка фото</li>
        <li style="width: calc(100% / 3); background-color: #d0d7de; height: 3px; border-radius: 50px;" :class="{'pipe-done': step>1}"></li>
        <li :class="{active: step===2, done: step>2}"><span>2</span> Сканирование</li>
        <li style="width: calc(100% / 3); background-color: #d0d7de; height: 3px; border-radius: 50px;" :class="{'pipe-done': step>2}"></li>
        <li :class="{active: step===3}"><span>3</span> Результаты</li>
      </ol>

      <!-- STEP 1 -->
      <section v-if="step===1" class="card">
        <div
          class="drop" :class="{over:isDragOver}"
          @dragover="onDragOver" @dragleave="onDragLeave" @drop="onDrop"
        >
          <p class="dz-title">Перетащите фото сюда</p>
          <p class="dz-sub">или</p>
          <label class="btn primary">
            Выбрать файлы
            <input type="file" accept="image/*" multiple hidden @change="onPick" />
          </label>
          <p class="hint">Избегайте фото с номерами и лицами.</p>
        </div>
      </section>

      <!-- STEP 2: preview + запуск -->
      <section v-if="step===2" class="grid">
        <div class="card sec">
          <h3>Предпросмотр ({{ items.length }})</h3>
          <div class="thumbs">
            <div v-for="(it,i) in items" :key="i" class="thumb">
              <img :src="it.url" alt="preview"/>
              <button class="x" @click="removeAt(i)">×</button>
            </div>
          </div>
          <div class="row actions-row">
            <label class="btn ghost">
              Добавить ещё
              <input type="file" accept="image/*" multiple hidden @change="onPick"/>
            </label>
            <button class="btn primary" :disabled="!canAnalyze" @click="analyze">
              {{ scanning ? 'Подготовка…' : `Сканировать (${items.length})` }}
            </button>
            <button class="btn link" @click="toStep1">Сбросить</button>
          </div>
          <p v-if="error" class="error">{{ error }}</p>
        </div>

        <div class="card scan">
          <ClientOnly>
            <CarScan :progress="progress" :height="380" />
          </ClientOnly>
          <div class="progress-wrap">
            <div class="progress-bar">
              <div class="progress-fill"
                  :style="{ width: Math.max(0, Math.min(100, progress)) + '%' }"></div>
            </div>
            <div class="progress-info">
              <span class="progress-pct">{{ Math.round(progress) }}%</span>
              <span class="progress-label">сканирование</span>
            </div>
          </div>

          <p class="scan-note">Во время сканирования показывается 3D-модель и прогресс.</p>
        </div>
      </section>

      <!-- STEP 3: результаты -->
      <section v-if="step===3" class="result-grid">
        <!-- ЛЕВЫЙ СТОЛБЕЦ: 3D-вид или фотографии -->
        <div class="card">
          <!-- Кнопка переключения вида -->
          <div class="view-toggle">
            <button 
              class="toggle-btn" 
              :class="{ active: !showPhotos }" 
              @click="showPhotos = false"
            >
              📊 Анализ
            </button>
            <button 
              class="toggle-btn" 
              :class="{ active: showPhotos }" 
              @click="showPhotos = true"
            >
              📷 Фото
            </button>
          </div>

          <!-- 3D-вид с анализом (по умолчанию) -->
          <div v-if="!showPhotos" class="view-content">
            <CarTopView :scores="agg.parts" :overall="agg.overall" />
          </div>

          <!-- Фотогалерея -->
          <div v-else class="view-content">
            <div class="photo-gallery">
              <h4>Загруженные фотографии</h4>
              <div class="photos-grid">
                <div v-for="(item, index) in items" :key="index" class="photo-item">
                  <img 
                    :src="item.url" 
                    :alt="`Фото ${index + 1}`" 
                    class="gallery-photo clickable" 
                    @click="openModal(item, index)"
                  />
                  <div class="photo-info">
                    <span class="photo-label">Фото {{ index + 1 }}</span>
                    <div v-if="item.result" class="photo-score">
                      <span v-if="item.result.predicted_class" 
                            :class="getDamageClass(item.result.predicted_class)">
                        {{ getDamageLabel(item.result.predicted_class) }}
                      </span>
                      <span v-if="item.result.confidence" class="confidence">
                        {{ Math.round((item.result.confidence || 0) * 100) }}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- ПРАВЫЙ СТОЛБЕЦ: сводка/советы/кнопки -->
        <div class="card summary">
          <h3>Итог сканирования</h3>
          <p class="muted">На основе {{ items.length }} фото</p>

          <div class="chips">
            <div class="chip" :style="{ borderColor: cleanScoreColor }">
              <span>Чистота</span>
              <b :style="{ color: cleanScoreColor }">{{ agg.cleanScore }}%</b>
            </div>
            <div class="chip qw" :style="{ borderColor: integrityScoreColor }">
              <span>Целостность</span>
              <b :style="{ color: integrityScoreColor }">{{ agg.integrityScore }}%</b>
            </div>
            <div class="chip total" :style="{ borderColor: overallScoreColor }">
              <span>Итог</span>
              <b :style="{ color: overallScoreColor }">{{ agg.overall }}%</b>
            </div>
          </div>
          
          <p class="formula-note">
            💡 Итог = Чистота×30% + Целостность×70%
          </p>

          <h4>Рекомендации ИИ-эксперта</h4>
          <ul class="bullet smart-recommendations">
            <li v-for="(rec, index) in recommendations" :key="index" v-html="rec"></li>
          </ul>
          
          <!-- Дополнительная информация от модели -->
          <div v-if="items.some((item: Item) => item.result?.predicted_class)" class="ai-details">
            <details class="model-info">
              <summary>📊 Детальная диагностика ИИ</summary>
              <div class="model-breakdown">
                <div v-for="(item, index) in items.filter((item: Item) => item.result)" :key="index" class="photo-analysis">
                  <h5>📷 Фото {{ index + 1 }}</h5>
                  <div class="analysis-grid">
                    <div class="metric">
                      <span>Класс:</span>
                      <b :class="getDamageClass(item.result!.predicted_class)">
                        {{ getDamageLabel(item.result!.predicted_class) }}
                      </b>
                    </div>
                    <div class="metric">
                      <span>Уверенность:</span>
                      <b>{{ Math.round((item.result!.confidence || 0) * 100) }}%</b>
                    </div>
                    <div v-if="item.result!.dirt_metrics?.dirt_score" class="metric">
                      <span>Индекс грязи:</span>
                      <b>{{ item.result!.dirt_metrics.dirt_score.toFixed(1) }}/10</b>
                    </div>
                  </div>
                  
                  <!-- Распределение вероятностей -->
                  <div v-if="item.result!.probabilities" class="probability-bars">
                    <div class="prob-item">
                      <span>Без повреждений:</span>
                      <div class="prob-bar">
                        <div class="prob-fill good" :style="{width: (item.result!.probabilities.no_damage || 0) * 100 + '%'}"></div>
                      </div>
                      <span>{{ Math.round((item.result!.probabilities.no_damage || 0) * 100) }}%</span>
                    </div>
                    <div class="prob-item">
                      <span>Мелкие повреждения:</span>
                      <div class="prob-bar">
                        <div class="prob-fill warn" :style="{width: (item.result!.probabilities.minor_damage || 0) * 100 + '%'}"></div>
                      </div>
                      <span>{{ Math.round((item.result!.probabilities.minor_damage || 0) * 100) }}%</span>
                    </div>
                    <div class="prob-item">
                      <span>Серьезные повреждения:</span>
                      <div class="prob-bar">
                        <div class="prob-fill danger" :style="{width: (item.result!.probabilities.major_damage || 0) * 100 + '%'}"></div>
                      </div>
                      <span>{{ Math.round((item.result!.probabilities.major_damage || 0) * 100) }}%</span>
                    </div>
                  </div>
                </div>
              </div>
            </details>
          </div>

          <!-- отступ после рекомендаций -->
          <div style="height:8px"></div>

          <div class="spacer"></div> <!-- всё ниже «прижмётся» к низу карточки -->

          <div class="tip" :class="passed ? 'good' : 'warn'">
            <strong>{{ passed ? 'Готов к дороге' : 'Нужно внимание' }}</strong>
            <p v-if="!passed">Доведите показатели минимум до 80% (чистота и целостность).</p>
            <p v-else>Отлично! Проверка пройдена.</p>
          </div>

          <div class="row end">
            <button v-if="!passed" class="btn primary" @click="toStep1">Пройти ещё раз</button>
            <template v-else>
              <button class="btn primary">В путь</button>
              <span class="ok-msg">Поздравляем! Вы прошли проверку.</span>
            </template>
          </div>
        </div>
      </section>
    </main>
  </div>

  <!-- Модальное окно для просмотра фото -->
  <div v-if="showModal && selectedPhoto" class="photo-modal" @click.self="closeModal">
    <div class="modal-content">
      <button class="modal-close" @click="closeModal">×</button>
      <div class="modal-image-container">
        <img :src="selectedPhoto.url" :alt="'Полноэкранный просмотр'" class="modal-image" />
      </div>
      <div class="modal-info">
        <h3>Детальный анализ</h3>
        <div v-if="selectedPhoto.result" class="modal-analysis">
          <div class="analysis-row">
            <span>Класс повреждений:</span>
            <b :class="getDamageClass(selectedPhoto.result.predicted_class)">
              {{ getDamageLabel(selectedPhoto.result.predicted_class) }}
            </b>
          </div>
          <div class="analysis-row">
            <span>Уверенность модели:</span>
            <b>{{ Math.round((selectedPhoto.result.confidence || 0) * 100) }}%</b>
          </div>
          <div v-if="selectedPhoto.result.dirt_status" class="analysis-row">
            <span>Состояние чистоты:</span>
            <b>{{ selectedPhoto.result.dirt_emoji }} {{ selectedPhoto.result.dirt_status }}</b>
          </div>
          <div v-if="selectedPhoto.result.expert_recommendations?.length" class="modal-recommendations">
            <h4>Рекомендации для этого фото:</h4>
            <ul>
              <li v-for="rec in selectedPhoto.result.expert_recommendations" :key="rec" v-html="rec"></li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style>
body {
  background: #fffee9;
  color: #141414;
}
* {
  margin: 0;
  padding: 0;
}
</style>

<style scoped>
/* контейнер страницы */
.page {
  margin-top: 50px;
  min-height: 80svh;
  background: transparent; /* фон задаёт body */
  color: #141414;
}

/* основной контент */
main {
  max-width: 1100px;
  margin: 18px auto;
  padding: 0 16px 48px;
}
h1 { margin: 6px 0 6px; }
.lead { color: #555555; margin: 0 0 10px; }

/* шаги */
.steps {
  display: flex; align-items: center; gap: 14px; list-style: none; margin: 30px 0 18px; padding: 0;
}
.steps li {
  display: flex; align-items: center; gap: 8px; color: #777777;
}
.steps li span {
  display: inline-grid; place-items: center;
  width: 24px; height: 24px; border-radius: 50%;
  border: 1px solid #d0d7de; background: #ffffff; color: #141414;
}
.steps li.active { color: #141414; }
.steps li.active span { background: #c1f11d; color: #141414; border-color: #c1f11d; }
.steps li.done { color: #141414; }
.steps li.done span { background: #eaffa7; color: #141414; border-color: #c1f11d; }

.steps li.pipe-done {
  background-color: #c1f11d !important;
}

/* карточки/блоки */
.card {
  background: #ffffff;
  border: 1px solid #e9ecef;
  border-radius: 14px;
  padding: 16px;
}
.sec .thumbs {
  height: 340px;
  overflow-y: auto;
  overflow-x: hidden; /* добавляем для предотвращения горизонтального скролла */
  overscroll-behavior: contain;
  -webkit-overflow-scrolling: touch;

  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
  grid-auto-rows: 100px;
  align-content: start;
  justify-content: start;
  gap: 10px;
  margin: 10px 0;
  padding-right: 6px;
  
  /* Исправляем расчёт ширины с учётом скроллбара */
  box-sizing: border-box;
}

/* Стили для скроллбара (Firefox) */
.sec .thumbs {
  scrollbar-width: thin;
  scrollbar-color: #c1f11d #f2f4f7;
}

/* Стили для скроллбара (Webkit/Chrome/Safari) */
.sec .thumbs::-webkit-scrollbar {
  width: 8px;
}

.sec .thumbs::-webkit-scrollbar-track {
  background: #f2f4f7;
  border-radius: 8px;
}

.sec .thumbs::-webkit-scrollbar-thumb {
  background: #c1f11d;
  border-radius: 8px;
  border: 2px solid #f2f4f7;
}

.sec .thumbs::-webkit-scrollbar-thumb:hover {
  background: #b8e61a; /* немного темнее при наведении */
}

/* Альтернативный вариант - если нужно больше места для содержимого */
.sec .thumbs-alternative {
  height: 340px;
  overflow-y: auto;
  overflow-x: hidden;
  
  display: grid;
  /* Уменьшаем минимальную ширину колонок для учёта скроллбара */
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  grid-auto-rows: 100px;
  align-content: start;
  gap: 10px;
  margin: 10px 0;
  
  /* Убираем padding-right, чтобы скроллбар был вплотную к краю */
  padding: 0;
  box-sizing: border-box;
}
.card > h3 { margin: 0 0 10px; }
.grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 16px; align-items: stretch; }
.actions-row {
  margin-top: auto;                  /* ключевая строка — прижимает низ */
  padding-top: 12px;
  border-top: 1px dashed #e9ecef;
}
.results .list { flex: 1 1 auto; overflow: auto; }
.results .end  { margin-top: auto; padding-top: 12px; border-top: 1px dashed #e9ecef; }

.actions-row .btn:hover { filter: brightness(0.98); }
@media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }

/* дропзона */
.drop {
  display: grid; place-items: center; gap: 10px;
  padding: 40px; border: 2px dashed #d9d9d9; border-radius: 12px;
  background: #ffffff;
}
.drop.over {
  border-color: #c1f11d;
  box-shadow: 0 0 0 2px rgba(193, 241, 29, 0.15) inset;
}
.dz-title { font-size: 1.1rem; }
.dz-sub { color: #777777; margin: -6px 0 6px; }
.hint { color: #888888; font-size: 0.9rem; }

/* кнопки */
.row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
.btn {
  padding: 10px 14px; border-radius: 10px;
  border: 1px solid #d0d7de; background: #ffffff; color: #141414;
  cursor: pointer;
}
.btn.primary {
  background: #c1f11d; color: #141414; border-color: #c1f11d; font-weight: 700;
}
.btn.ghost { background: #ffffff; color: #141414; }
.btn.link  { background: transparent; border-color: transparent; color: #555555; text-decoration: underline; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }

/* блок сканирования */
.scan { display: grid; gap: 8px; }
.scan-note { color: #666666; font-size: 0.9rem; margin: 0; text-align: center; }

/* превьюшки */
/* .thumbs {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
  gap: 10px; margin: 10px 0; max-height: 340px; overflow: auto;
} */
.thumb {
  position: relative; border: 1px solid #e5e7eb; border-radius: 10px; overflow: hidden; background: #ffffff;
}
.thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
.thumb .x {
  position: absolute; top: 6px; right: 6px;
  background: rgba(0,0,0,0.55); border: 0; color: #ffffff;
  width: 24px; height: 24px; border-radius: 50%; cursor: pointer;
}

/* ошибки */
.error { margin-top: 10px; color: #c62828; background: #fff6f6; border: 1px solid #ffd6d6; padding: 8px; border-radius: 8px; }

/* список результатов */
.list { display: grid; gap: 12px; }
.item { align-items: flex-start; }
.mini {
  width: 100px; height: 70px; object-fit: cover;
  border-radius: 8px; border: 1px solid #e5e7eb;
}
.grow { flex: 1; }

.badges { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 6px; }
.pill {
  padding: 4px 10px; border-radius: 999px; font-weight: 700;
  background: #ffffff; border: 1px solid #d0d7de; color: #141414;
}
.pill.good { background: #c1f11d; border-color: #c1f11d; color: #141414; }
.pill.bad  { background: #fff0f0; border-color: #ffd6d6; color: #c62828; }

.pct { color: #666666; }
.json code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color: #333333;
}
.end { justify-content: flex-end; margin-top: 10px; }
.progress-wrap{
  display:grid; gap:6px; margin-top:10px; place-items:center;
}
.progress-bar{
  width:min(480px, 100%); height:8px; background:#e9ecef;
  border-radius:999px; overflow:hidden;
}
.progress-fill{
  height:100%; background:#c1f11d; transition:width .25s ease;
}
.progress-info{
  display:flex; gap:10px; align-items:center; color:#141414;
}
.progress-pct{
  background:#c1f11d; color:#141414; font-weight:800;
  border-radius:999px; padding:2px 10px;
}
.progress-label{ color:#666666; }

/* сетка результатов */
.result-grid{ display:grid; grid-template-columns:1.2fr 1fr; gap:16px; }
@media (max-width:980px){ .result-grid{ grid-template-columns:1fr; } }

.summary{ display:flex; flex-direction:column; }
.muted{ color:#666; margin:0 0 12px; }
.chips{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-bottom:12px; }
.chip{ display:flex; justify-content:space-between; align-items:center;
  padding:10px 12px; border:1px solid #e9ecef; border-radius:10px; background:#fff; }
.chip.total{ width: 80px; }
.chip.qw {
  width: 145px;
}
.chip span{ color:#666; }
.chip b{ font-size:1.1rem; }

.formula-note{
  color:#666; font-size:0.85rem; margin:8px 0 12px; text-align:center;
  background:#f8f9fa; padding:6px 12px; border-radius:8px;
}

.bullet{ margin:6px 0 0 18px; }
.spacer{ flex:1; } /* всё, что ниже, уезжает к низу карточки */

.tip{ border-radius:12px; padding:12px; border:1px solid #e9ecef; background:#fff; margin-top:10px; }
.tip.good{ border-color:#c1f11d; background:#f7ffcf; }
.tip.warn{ border-color:#ffd6d6; background:#fff5f5; }
.ok-msg{ color:#2e7d32; font-weight:600; margin-left:8px; }

/* Стили для умных рекомендаций ИИ */
.smart-recommendations {
  line-height: 1.6;
}

.smart-recommendations li {
  margin-bottom: 8px;
  padding: 4px 0;
}

.ai-details {
  margin-top: 16px;
}

.model-info {
  background: #f8f9fa;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  padding: 8px;
  margin-top: 8px;
}

.model-info summary {
  cursor: pointer;
  font-weight: 500;
  padding: 4px;
  user-select: none;
}

.model-info summary:hover {
  background: #e9ecef;
  border-radius: 4px;
}

.model-breakdown {
  margin-top: 12px;
}

.photo-analysis {
  background: white;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  padding: 12px;
  margin-bottom: 12px;
}

.photo-analysis h5 {
  margin: 0 0 8px 0;
  color: #495057;
  font-size: 0.9rem;
}

.analysis-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 8px;
  margin-bottom: 12px;
}

.metric {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.85rem;
}

.metric span {
  color: #6c757d;
}

.metric b {
  font-weight: 600;
}

.damage-none { color: #28a745; }
.damage-minor { color: #ffc107; }
.damage-major { color: #dc3545; }
.damage-unknown { color: #6c757d; }

.probability-bars {
  margin-top: 8px;
}

.prob-item {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
  font-size: 0.8rem;
}

.prob-item span:first-child {
  min-width: 120px;
  color: #6c757d;
}

.prob-item span:last-child {
  min-width: 35px;
  font-weight: 500;
  text-align: right;
}

.prob-bar {
  flex: 1;
  height: 16px;
  background: #e9ecef;
  border-radius: 8px;
  overflow: hidden;
}

.prob-fill {
  height: 100%;
  transition: width 0.3s ease;
}

.prob-fill.good { background: linear-gradient(90deg, #28a745, #20c997); }
.prob-fill.warn { background: linear-gradient(90deg, #ffc107, #fd7e14); }
.prob-fill.danger { background: linear-gradient(90deg, #dc3545, #e83e8c); }

/* Переключатель видов */
.view-toggle {
  display: flex;
  gap: 4px;
  margin-bottom: 16px;
  background: #f8f9fa;
  border-radius: 8px;
  padding: 4px;
}

.toggle-btn {
  flex: 1;
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  background: transparent;
  color: #6c757d;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.toggle-btn:hover {
  background: #e9ecef;
  color: #495057;
}

.toggle-btn.active {
  background: #fff;
  color: #212529;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

/* Контейнер контента с анимацией */
.view-content {
  animation: fadeIn 0.3s ease-in-out;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Фотогалерея */
.photo-gallery h4 {
  margin: 0 0 16px 0;
  color: #495057;
  font-size: 1.1rem;
}

.photos-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
}

.photo-item {
  background: #fff;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  overflow: hidden;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.photo-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.gallery-photo {
  width: 100%;
  height: 120px;
  object-fit: cover;
  display: block;
}

.gallery-photo.clickable {
  cursor: pointer;
  transition: opacity 0.2s ease;
}

.gallery-photo.clickable:hover {
  opacity: 0.8;
}

.photo-info {
  padding: 8px 12px;
}

.photo-label {
  font-size: 0.85rem;
  color: #6c757d;
  font-weight: 500;
}

.photo-score {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 4px;
  font-size: 0.8rem;
}

.confidence {
  color: #495057;
  font-weight: 600;
}

/* Модальное окно для просмотра фото */
.photo-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
  animation: fadeIn 0.3s ease;
}

.modal-content {
  position: relative;
  max-width: 90vw;
  max-height: 90vh;
  background: white;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
  display: flex;
  flex-direction: column;
}

.modal-close {
  position: absolute;
  top: 16px;
  right: 16px;
  width: 40px;
  height: 40px;
  border: none;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  border-radius: 50%;
  font-size: 24px;
  font-weight: bold;
  cursor: pointer;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s ease;
}

.modal-close:hover {
  background: rgba(0, 0, 0, 0.9);
}

.modal-image-container {
  position: relative;
  max-height: 60vh;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f8f9fa;
}

.modal-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  display: block;
}

.modal-info {
  padding: 20px;
  background: white;
  max-height: 30vh;
  overflow-y: auto;
}

.modal-info h3 {
  margin: 0 0 16px 0;
  color: #343a40;
  font-size: 1.2rem;
}

.modal-analysis {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.analysis-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid #e9ecef;
}

.analysis-row:last-child {
  border-bottom: none;
}

.analysis-row span {
  color: #6c757d;
  font-weight: 500;
}

.analysis-row b {
  font-weight: 600;
}

.modal-recommendations {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid #e9ecef;
}

.modal-recommendations h4 {
  margin: 0 0 12px 0;
  color: #495057;
  font-size: 1rem;
}

.modal-recommendations ul {
  margin: 0;
  padding-left: 20px;
  list-style-type: disc;
}

.modal-recommendations li {
  margin-bottom: 8px;
  line-height: 1.5;
  color: #495057;
}

@media (max-width: 768px) {
  .photos-grid {
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  }
  
  .gallery-photo {
    height: 100px;
  }

  .modal-content {
    max-width: 95vw;
    max-height: 95vh;
  }

  .modal-image-container {
    max-height: 50vh;
  }

  .modal-info {
    max-height: 40vh;
    padding: 16px;
  }

  .modal-close {
    top: 12px;
    right: 12px;
    width: 36px;
    height: 36px;
    font-size: 20px;
  }
}
</style>