// modelo.js - Corre la CNN (ONNX) en el navegador con onnxruntime-web.
//
// Los .onnx ya traen dentro el redimensionado a 224x224, la normalización, el ensemble y el
// espejo (ver src/exportar_onnx.py). Aquí solo se pasa la imagen RGB 0-255 a su tamaño real
// y se interpreta la salida igual que src/clasificador.py.

const RUTA = "/modelos/";
const ORT_VERSION = "1.31.1";

let meta = null;
const sesiones = {};

export async function cargarMeta() {
  if (!meta) meta = await (await fetch(RUTA + "modelo.json")).json();
  return meta;
}

export async function cargarModelo(modo) {
  await cargarMeta();
  if (!sesiones[modo]) {
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
    sesiones[modo] = ort.InferenceSession.create(RUTA + meta.archivos[modo], {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }
  return sesiones[modo];
}

// ImageData (RGBA) -> tensor float32 [1, alto, ancho, 3] con valores 0-255
function aTensor(imageData) {
  const { data, width, height } = imageData;
  const rgb = new Float32Array(width * height * 3);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    rgb[j] = data[i];
    rgb[j + 1] = data[i + 1];
    rgb[j + 2] = data[i + 2];
  }
  return new ort.Tensor("float32", rgb, [1, height, width, 3]);
}

// Igual que ClasificadorAnimales._interpretar() en Python
export function interpretar(probs, umbral) {
  let i = 0;
  for (let k = 1; k < probs.length; k++) if (probs[k] > probs[i]) i = k;
  const clase = meta.clases[i];
  const confianza = probs[i];
  return {
    clase,
    nombre: meta.nombres[clase] ?? clase,
    confianza,
    es_animal: clase !== "fondo" && confianza >= umbral,
    incierto: confianza < umbral,
    probabilidades: Object.fromEntries(meta.clases.map((c, k) => [c, probs[k]])),
  };
}

// Suavizado temporal para video: promedia las probabilidades de los frames de la última
// ventana (ms). Evita que la etiqueta salte de una clase a otra en cada frame cuando el
// modelo duda (p. ej. bovino 37 % / caballo 42 %); el modelo en sí no cambia.
export class Suavizador {
  constructor(ventanaMs = 1500) {
    this.ventanaMs = ventanaMs;
    this.historial = [];
  }
  reiniciar() {
    this.historial = [];
  }
  agregar(probs, t = performance.now()) {
    this.historial.push({ t, probs });
    this.historial = this.historial.filter((h) => t - h.t <= this.ventanaMs);
    const n = this.historial.length;
    const media = probs.map((_, k) => this.historial.reduce((s, h) => s + h.probs[k], 0) / n);
    return { probs: media, n };
  }
}

// Devuelve las 5 probabilidades (Float32Array) y el tiempo de inferencia en ms
export async function clasificar(imageData, modo) {
  const sesion = await cargarModelo(modo);
  const t0 = performance.now();
  const salida = await sesion.run({ imagen: aTensor(imageData) });
  const probs = salida.probabilidades.data;
  return { probs: Array.from(probs), ms: performance.now() - t0 };
}
