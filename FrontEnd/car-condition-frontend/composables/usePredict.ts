export type PredictResp = {
  dirty: boolean
  dirty_prob: number     // 0..1
  damaged: boolean
  damaged_prob: number   // 0..1
  
  // Новые поля от обновленной модели
  predicted_class?: string      // "no_damage" | "minor_damage" | "major_damage"
  confidence?: number           // 0..1 - уверенность модели
  probabilities?: {
    no_damage?: number
    minor_damage?: number
    major_damage?: number
  }
  dirt_metrics?: {
    color_diversity?: number
    contrast?: number
    saturation?: number
    brown_ratio?: number
    edge_intensity?: number
    brightness?: number
    dirt_score?: number
  }
  model_available?: boolean     // доступность модели
  
  // Старые поля для совместимости
  parts?: { left:number; right:number; front:number; rear:number; roof:number }
  debug?: {
    device?: string
    model_loaded?: boolean
    dirt?: { status:string; score:number }
    damage_probs?: {
      no_damage:number; minor_damage:number; major_damage:number;
      predicted_class:string; confidence:number
    }
  }
}

type BackendResp = { results: PredictResp[] }

function apiBase(): string {
  // читаем из nuxt runtimeConfig, иначе дефолт на локальный бэк
  // в Nuxt это можно вызывать из composable
  // @ts-ignore
  const { public: { apiBase } } = useRuntimeConfig()
  return apiBase || 'http://127.0.0.1:8000'
}

// один файл — просто обёртка над predictMany
export async function predictOne(file: File): Promise<PredictResp> {
  const results = await predictMany([file])
  if (results.length === 0) {
    throw new Error('No prediction results received')
  }
  return results[0]!
}

// несколько файлов (одним запросом) + сглаженная анимация прогресса
export async function predictMany(files: File[], onProgress?: (p:number, i:number)=>void) {
  const fd = new FormData()
  files.forEach(f => fd.append('files', f, f.name)) // ВАЖНО: ключ 'files' (как на FastAPI)

  // мягкая анимация прогресса до 95% пока ждём сеть
  let target = 95, current = 0
  const tick = setInterval(() => {
    current += (target - current) * 0.2
    onProgress?.(Math.min(95, current), 0)
  }, 80)

  try {
    const res = await fetch(`${apiBase()}/predict`, { method: 'POST', body: fd })
    if (!res.ok) throw new Error(`Predict request failed: ${res.status}`)
    const data = await res.json() as BackendResp

    // добегаем до 100
    target = 100
    onProgress?.(100, data.results.length)
    return data.results
  } finally {
    clearInterval(tick)
  }
}