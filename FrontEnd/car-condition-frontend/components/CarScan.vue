<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref } from 'vue'

const props = defineProps<{ progress?: number; height?: number }>()
const canvasRef = ref<HTMLCanvasElement | null>(null)

let THREE: any
let renderer: any, scene: any, camera: any, car: any, pmrem: any, raf = 0

onMounted(async () => {
  THREE = await import('three')
  const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js')
  const { RoomEnvironment } = await import('three/examples/jsm/environments/RoomEnvironment.js')

  const h = props.height ?? 380
  const w = canvasRef.value!.parentElement!.clientWidth

  renderer = new THREE.WebGLRenderer({ canvas: canvasRef.value!, antialias: true, alpha: true })
  renderer.setSize(w, h)
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1))
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.25
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = THREE.PCFSoftShadowMap
  renderer.physicallyCorrectLights = true

  scene = new THREE.Scene()
  pmrem = new THREE.PMREMGenerator(renderer)
  scene.environment = pmrem.fromScene(new RoomEnvironment(THREE), 0.04).texture

  camera = new THREE.PerspectiveCamera(35, w / h, 0.1, 100)
  camera.position.set(0.2, 1.3, 4.2)

  const hemi = new THREE.HemisphereLight(0xffffff, 0x222222, 0.4)
  scene.add(hemi)

  const key = new THREE.DirectionalLight(0xffffff, 1.7)
  key.position.set(3.5, 5, 6)
  key.castShadow = true
  key.shadow.mapSize.set(1024, 1024)
  scene.add(key)

  const fill = new THREE.DirectionalLight(0xffffff, 0.8)
  fill.position.set(-2.5, 2.5, 2)
  scene.add(fill)

  const rim = new THREE.DirectionalLight(0xffffff, 1.1)
  rim.position.set(-4, 3, -5)
  scene.add(rim)

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(8, 8),
    new THREE.ShadowMaterial({ color: 0x000000, opacity: 0.18 })
  )
  ground.rotation.x = -Math.PI / 2
  ground.position.y = -0.62
  ground.receiveShadow = true
  scene.add(ground)

  const platform = new THREE.Mesh(
    new THREE.CylinderGeometry(1.55, 1.55, 0.08, 72),
    new THREE.MeshStandardMaterial({ color: 0xeeeeee, metalness: 0.1, roughness: 0.6 })
  )
  platform.position.y = -0.58
  platform.receiveShadow = true
  scene.add(platform)

  try {
    const glb = await new GLTFLoader().loadAsync('/models/car.glb')
    car = glb.scene
  } catch {
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(2.4, 0.7, 1.1),
      new THREE.MeshStandardMaterial({ color: 0x111214, metalness: 0.7, roughness: 0.3 })
    )
    body.position.y = -0.12
    const hood = new THREE.Mesh(
      new THREE.BoxGeometry(0.8, 0.4, 1.1),
      new THREE.MeshStandardMaterial({ color: 0x17191b, metalness: 0.7, roughness: 0.35 })
    )
    hood.position.set(-1.2, 0.05, 0)
    car = new THREE.Group(); car.add(body); car.add(hood)
  }

  car.scale.set(1.18, 1.18, 1.18)
  car.position.y = -0.1
  car.traverse((o: any) => { if (o.isMesh) { o.castShadow = true; o.material && (o.material.envMapIntensity = 1.0) } })
  scene.add(car)

  let t = 0
  const animate = () => {
    raf = requestAnimationFrame(animate)
    if (car) {
      car.rotation.y += 0.008
      t += 0.01
      camera.position.x = 0.25 + Math.sin(t) * 0.05
      camera.lookAt(0, 0.1, 0)
    }
    renderer.render(scene, camera)
  }
  animate()

  const onResize = () => {
    const w2 = canvasRef.value!.parentElement!.clientWidth
    const h2 = props.height ?? 380
    renderer.setSize(w2, h2)
    camera.aspect = w2 / h2
    camera.updateProjectionMatrix()
  }
  window.addEventListener('resize', onResize)
})

onBeforeUnmount(() => {
  cancelAnimationFrame(raf)
  pmrem?.dispose?.()
  renderer?.dispose?.()
})
</script>

<template>
  <div class="scan-wrap" :style="{ height: (height ?? 380) + 'px' }">
    <canvas ref="canvasRef" class="scan-canvas" />
    <div class="halo"></div>
    <div class="scan-overlay"><div class="scan-line"></div></div>
  </div>
</template>

<style scoped>
.scan-wrap{
  position: relative; width:100%;
  background:#ffffff; border-radius:16px; overflow:hidden;
  border: 1px solid #e0e0e0;
}
.scan-canvas{ width:100%; height:100%; display:block; }

/* визуальные улучшайзеры */
.halo{
  position:absolute; inset:0; pointer-events:none;
  background:
    radial-gradient(ellipse at 50% 75%, rgba(0,0,0,0.20) 0%, rgba(0,0,0,0.04) 25%, rgba(0,0,0,0.02) 40%, rgba(0,0,0,0) 60%),
    radial-gradient(ellipse at center, rgba(193,241,29,0.06) 0%, rgba(193,241,29,0.00) 65%);
  mix-blend-mode:multiply;
}
.scan-overlay{ position:absolute; inset:0; overflow:hidden; pointer-events:none; }
.scan-line{
  position:absolute; left:0; right:0; height:26%;
  background: linear-gradient(to bottom, rgba(193,241,29,0) 0%, rgba(193,241,29,.25) 50%, rgba(193,241,29,0) 100%);
  filter: blur(4px);
  animation: scanMove 2.4s linear infinite;
}
@keyframes scanMove { from { top:-28% } to { top:108% } }
</style>
