<template>
  <header class="hdr">
    <div class="inner">
      <NuxtLink to="/" class="brand" aria-label="inDrive">
        <img src="/logo--dark.svg" alt="inDrive logo" />
      </NuxtLink>

      <div class="project">
        Проект: AI-inDrive
      </div>
    </div>
  </header>
</template>

<style scoped>
.hdr{
  background: #ffffff;
  border-bottom: 1px solid #eef0f2;
}
.inner{
  max-width: 1120px;
  margin: 0 auto;
  padding: 10px 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.brand img{
  display: block;
  height: 36px;
}

.project{
  background: #c1f11d;
  color: #141414;
  font-weight: 800;
  padding: 8px 14px;
  border-radius: 12px;
}
@media (max-width: 520px){
  .project{ font-size: .95rem; padding: 6px 10px; }
  .brand img{ height: 30px; }
}
</style>
