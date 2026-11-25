// modelo.js - Corre la CNN (ONNX) en el navegador con onnxruntime-web.
// Versión v3: la red es totalmente convolucional; además de las 5 probabilidades devuelve
// "animales" (un sí/no por animal) y "mapa" (dónde está cada animal) para dibujar cajas.
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
  const ms = performance.now() - t0;
  let mapa = null;
  if (salida.mapa) {
    const [, h, w, c] = salida.mapa.dims;
    mapa = { data: salida.mapa.data, h, w, c };
  }
  const animales = salida.animales ? Array.from(salida.animales.data) : null;
  return { probs: Array.from(probs), ms, mapa, animales };
}

// Cajas a partir del mapa de evidencias de la v3: para cada animal que supera el umbral,
// las zonas del mapa por encima del umbral se agrupan (vecinas arriba/abajo/izq/der) y cada
// grupo es una caja. Coordenadas normalizadas 0-1 respecto a la imagen que entró al modelo.
export function cajasDelMapa(mapa, animales, umbral) {
  if (!mapa || !animales) return [];
  const { data, h, w, c } = mapa;
  const cajas = [];
  for (let k = 0; k < c; k++) {
    if (animales[k] < umbral) continue;
    const visto = new Uint8Array(h * w);
    for (let i = 0; i < h * w; i++) {
      if (visto[i] || data[i * c + k] < umbral) continue;
      let x0 = w, y0 = h, x1 = -1, y1 = -1, conf = 0;
      const pila = [i];
      visto[i] = 1;
      while (pila.length) {
        const j = pila.pop(), y = Math.floor(j / w), x = j % w;
        x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
        conf = Math.max(conf, data[j * c + k]);
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const nx = x + dx, ny = y + dy, n = ny * w + nx;
          if (nx >= 0 && ny >= 0 && nx < w && ny < h && !visto[n] && data[n * c + k] >= umbral) {
            visto[n] = 1;
            pila.push(n);
          }
        }
      }
      cajas.push({ clase: meta.animales[k], conf, x0: x0 / w, y0: y0 / h, x1: (x1 + 1) / w, y1: (y1 + 1) / h });
    }
  }
  return cajas;
}
