export default defineEventHandler(async (event) => {
  const form = await readMultipartFormData(event)
  if (!form?.length) throw createError({ statusCode: 400, statusMessage: 'No file' })

  const fd = new FormData()
  for (const part of form) {
    if (part.filename && part.data) {
      fd.append('files', new Blob([part.data], { type: part.type || 'application/octet-stream' }), part.filename)
    }
  }

  const base = useRuntimeConfig().public.apiBase
  const res = await fetch(`${base}/predict`, { method: 'POST', body: fd })
  const text = await res.text()
  if (!res.ok) throw createError({ statusCode: res.status, statusMessage: text })
  return JSON.parse(text)
})
