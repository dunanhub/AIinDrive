// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  pages: true,
  devtools: { enabled: true },
  runtimeConfig: { public: { apiBase: process.env.NUXT_PUBLIC_API_BASE } }
})
