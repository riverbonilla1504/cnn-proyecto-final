// app.js - Página de cámara (dueño: Javier).
// Fuentes: webcam, video subido o foto. Cada frame pasa por la CNN (ONNX),
// se muestra el resultado y el AgrupadorEventos convierte las detecciones en eventos.

import { cargarMeta, cargarModelo, clasificar, interpretar, Suavizador } from "./modelo.js";
import { AgrupadorEventos, guardarEvento, listarEventos } from "./eventos.js";

const $ = (id) => document.getElementById(id);
const video = $("video");
const lienzoFoto = $("foto");
const captura = document.createElement("canvas");
const ctxCaptura = captura.getContext("2d", { willReadFrequently: true });

const CAMARA_DE_FUENTE = { camara: "webcam_1", video: "video" };
const MAX_LADO_VIDEO = 640;      // el modelo reduce todo a 224: no hace falta más
const MAX_LADO_FOTO = 1024;

const estado = {
  fuente: "camara",
  modo: "preciso",     // modo único: 3 redes + espejo (95.2 % en el test, ~1 frame/s)
  umbral: 0.6,
  corriendo: false,
  stream: null,
  archivoVideo: null,
  zona: null,          // {x, y, w, h} normalizados (0-1) o null = frame completo
  frames: 0,
  tiempos: [],
};

// ---------------------------------------------------------------- utilidades de UI
let temporizadorToast;
function toast(texto) {
  const t = $("toast");
  t.textContent = texto;
  t.hidden = false;
  clearTimeout(temporizadorToast);
  temporizadorToast = setTimeout(() => (t.hidden = true), 4000);
}

const pct = (x) => `${(x * 100).toFixed(0)} %`;
const hora = (iso) => new Date(iso).toLocaleString("es-CO", { dateStyle: "short", timeStyle: "medium" });

function mostrarVacio(texto) {
  $("vacio").hidden = !texto;
  if (texto) $("vacio-texto").innerHTML = texto;
}

function pintarResultado(r, nFrames = 1) {
  const est = r.es_animal ? "animal" : r.incierto ? "incierto" : "nada";
  $("resultado").dataset.estado = est;
  $("res-chip").textContent = r.es_animal ? "Animal detectado" : r.incierto ? "Incierto" : "Sin animal";
  $("res-clase").textContent = r.incierto ? `¿${r.nombre}?` : r.nombre;
  $("res-conf").textContent = `Confianza ${pct(r.confianza)} · umbral ${estado.umbral.toFixed(2)}` +
    (nFrames > 1 ? ` · promedio de ${nFrames} frames` : "") +
    (estado.zona && estado.fuente !== "foto" ? " · solo la zona marcada" : "");

  const meta = window.__meta;
  const lista = $("barras");
  if (!lista.children.length) {
    lista.innerHTML = meta.clases
      .map((c) => `<li data-clase="${c}"><span>${meta.nombres[c].split(" (")[0]}</span>` +
        `<div class="cam-barra"><span></span></div><span class="cam-pct"></span></li>`)
      .join("");
  }
  for (const li of lista.children) {
    const p = r.probabilidades[li.dataset.clase];
    li.dataset.top = li.dataset.clase === r.clase ? "si" : "no";
    li.querySelector(".cam-barra span").style.width = `${(p * 100).toFixed(1)}%`;
    li.querySelector(".cam-pct").textContent = pct(p);
  }

  const et = $("etiqueta");
  et.hidden = false;
  et.dataset.estado = est;
  et.textContent = `${r.incierto ? "¿" + r.nombre + "?" : r.nombre} · ${pct(r.confianza)}`;
}

function pintarStats(ms) {
  $("stat-ms").textContent = `${ms.toFixed(0)} ms`;
  $("stat-frames").textContent = estado.frames;
  const ahora = performance.now();
  estado.tiempos.push(ahora);
  estado.tiempos = estado.tiempos.filter((t) => ahora - t < 3000);
  if (estado.tiempos.length > 1) {
    const fps = (estado.tiempos.length - 1) / ((ahora - estado.tiempos[0]) / 1000);
    $("stat-fps").textContent = fps.toFixed(1);
  }
}

// ---------------------------------------------------------------- eventos
// Ventana de 3 s: el modelo preciso va a ~1 frame/s, así que promedia ~3 frames
const suavizador = new Suavizador(3000);

const agrupador = new AgrupadorEventos({
  onInicio: () => pintarEventoActual(),
  onFin: async (evento) => {
    pintarEventoActual();
    try {
      const { id } = await guardarEvento(evento);
      toast(`Evento guardado: ${window.__meta.nombres[evento.clase]} (${evento.n_frames} frames)`);
      await refrescarHistorial(id);
    } catch (e) {
      console.error(e);
      toast("No se pudo guardar el evento. ¿Está corriendo la API?");
    }
  },
});

function pintarEventoActual() {
  const caja = $("evento-actual");
  const a = agrupador.activo;
  if (!a) {
    caja.dataset.activo = "no";
    caja.innerHTML = `<p class="cam-muted">Ninguno. Un evento empieza cuando el mismo animal aparece en 3 frames seguidos y termina tras 2 s sin verlo.</p>`;
    return;
  }
  const dur = ((a.fin - a.inicio) / 1000).toFixed(1);
  const media = a.confs.reduce((s, c) => s + c, 0) / a.confs.length;
  caja.dataset.activo = "si";
  caja.innerHTML = `<strong>${window.__meta.nombres[a.clase]}</strong>` +
    `<p class="cam-muted">${dur} s · ${a.confs.length} frames · confianza media ${pct(media)}</p>`;
}

setInterval(() => {
  agrupador.revisar(Date.now());
  if (agrupador.activo) pintarEventoActual();
}, 500);

const NOMBRE_FUENTE = { webcam_1: "Cámara", video: "Video", pestana: "Pestaña", simulada: "Simulada" };

async function refrescarHistorial(idNuevo) {
  const cuerpo = $("historial");
  try {
    const eventos = await listarEventos(20);
    if (!eventos.length) {
      cuerpo.innerHTML = `<tr><td colspan="7" class="cam-muted">Aún no hay eventos.</td></tr>`;
      return;
    }
    cuerpo.innerHTML = eventos.map((e) => {
      const dur = ((new Date(e.fin) - new Date(e.inicio)) / 1000).toFixed(1);
      return `<tr class="${e.id === idNuevo ? "cam-nuevo" : ""}"><td>${hora(e.inicio)}</td>` +
        `<td>${window.__meta.nombres[e.clase] ?? e.clase}</td><td>${pct(e.confianza)}</td>` +
        `<td>${dur} s</td><td>${e.n_frames}</td><td>${NOMBRE_FUENTE[e.camara] ?? e.camara}</td>` +
        `<td>${e.modo === "rapido" ? "Rápido" : "Preciso"}</td></tr>`;
    }).join("");
  } catch (e) {
    cuerpo.innerHTML = `<tr><td colspan="7" class="cam-muted">No se pudo leer el historial (¿está corriendo la API?).</td></tr>`;
  }
}

// ---------------------------------------------------------------- captura y clasificación
// region (opcional) = {sx, sy, sw, sh} en píxeles de la fuente: solo esa parte va al modelo
function capturarFrame(fuente, ancho, alto, maxLado, region = null) {
  const r = region ?? { sx: 0, sy: 0, sw: ancho, sh: alto };
  const escala = Math.min(1, maxLado / Math.max(r.sw, r.sh));
  captura.width = Math.max(1, Math.round(r.sw * escala));
  captura.height = Math.max(1, Math.round(r.sh * escala));
  ctxCaptura.drawImage(fuente, r.sx, r.sy, r.sw, r.sh, 0, 0, captura.width, captura.height);
  return ctxCaptura.getImageData(0, 0, captura.width, captura.height);
}

// ---------------------------------------------------------------- zona marcada a mano
// El usuario arrastra un rectángulo sobre el video. Se guarda en coordenadas normalizadas
// del video (0-1) y al recortar se agranda MARGEN_ZONA por lado: el modelo se entrenó con
// el animal completo y un poco de contexto (en una prueba, la vaca pasó de 51 % a 72.5 %).
const MARGEN_ZONA = 0.12;
let arrastre = null;

function rectContenido() {
  // Dónde se dibuja realmente el video dentro del recuadro (object-fit: contain)
  const c = $("pantalla").getBoundingClientRect();
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return null;
  const s = Math.min(c.width / vw, c.height / vh);
  return { c, izq: (c.width - vw * s) / 2, arr: (c.height - vh * s) / 2, ancho: vw * s, alto: vh * s };
}

function zonaConMargen() {
  const z = estado.zona;
  const mx = z.w * MARGEN_ZONA, my = z.h * MARGEN_ZONA;
  const x0 = Math.max(0, z.x - mx), y0 = Math.max(0, z.y - my);
  const x1 = Math.min(1, z.x + z.w + mx), y1 = Math.min(1, z.y + z.h + my);
  return { x0, y0, x1, y1 };
}

function regionVideo() {
  if (!estado.zona) return null;
  const { x0, y0, x1, y1 } = zonaConMargen();
  const vw = video.videoWidth, vh = video.videoHeight;
  return { sx: x0 * vw, sy: y0 * vh, sw: (x1 - x0) * vw, sh: (y1 - y0) * vh };
}

function ponerCaja(izq, arr, ancho, alto, texto) {
  const el = $("zona");
  Object.assign(el.style, { left: `${izq}px`, top: `${arr}px`, width: `${ancho}px`, height: `${alto}px` });
  $("zona-texto").textContent = texto;
  el.hidden = false;
}

function pintarZona() {
  const rc = rectContenido();
  if (!estado.zona || !rc || video.hidden) { if (!arrastre) $("zona").hidden = true; return; }
  const { x0, y0, x1, y1 } = zonaConMargen();
  ponerCaja(rc.izq + x0 * rc.ancho, rc.arr + y0 * rc.alto, (x1 - x0) * rc.ancho, (y1 - y0) * rc.alto,
    "Zona analizada (+12 % de margen)");
}

function empezarMarcado() {
  if (video.hidden || !video.videoWidth) return toast("Primero inicia la cámara o el video.");
  $("capa-zona").hidden = false;
  toast("Arrastra un rectángulo alrededor de donde pasa el animal. Esc para cancelar.");
}

function terminarMarcado() {
  $("capa-zona").hidden = true;
  arrastre = null;
  pintarZona();
}

function fijarZona(zona) {
  estado.zona = zona;
  suavizador.reiniciar();                 // no mezclar frames con y sin zona
  $("btn-quitar-zona").hidden = !zona;
  $("btn-zona").textContent = zona ? "Cambiar zona" : "Marcar zona";
  pintarZona();
}

const capa = $("capa-zona");
capa.addEventListener("pointerdown", (e) => {
  const c = $("pantalla").getBoundingClientRect();
  capa.setPointerCapture(e.pointerId);
  arrastre = { x: e.clientX - c.left, y: e.clientY - c.top, x2: e.clientX - c.left, y2: e.clientY - c.top };
});
capa.addEventListener("pointermove", (e) => {
  if (!arrastre) return;
  const c = $("pantalla").getBoundingClientRect();
  arrastre.x2 = e.clientX - c.left;
  arrastre.y2 = e.clientY - c.top;
  const a = arrastre;
  ponerCaja(Math.min(a.x, a.x2), Math.min(a.y, a.y2), Math.abs(a.x2 - a.x), Math.abs(a.y2 - a.y), "Suelta para fijar la zona");
});
capa.addEventListener("pointerup", () => {
  const rc = rectContenido();
  const a = arrastre;
  if (!a || !rc) return terminarMarcado();
  const norm = (v, ini, tam) => Math.min(1, Math.max(0, (v - ini) / tam));
  const x0 = norm(Math.min(a.x, a.x2), rc.izq, rc.ancho), x1 = norm(Math.max(a.x, a.x2), rc.izq, rc.ancho);
  const y0 = norm(Math.min(a.y, a.y2), rc.arr, rc.alto), y1 = norm(Math.max(a.y, a.y2), rc.arr, rc.alto);
  arrastre = null;
  if (x1 - x0 < 0.04 || y1 - y0 < 0.04) {
    toast("La zona es muy pequeña: arrastra un rectángulo más grande.");
    return terminarMarcado();
  }
  fijarZona({ x: x0, y: y0, w: x1 - x0, h: y1 - y0 });
  terminarMarcado();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !capa.hidden) terminarMarcado();
});
window.addEventListener("resize", pintarZona);
video.addEventListener("loadedmetadata", pintarZona);

async function bucle() {
  while (estado.corriendo) {
    const t0 = performance.now();
    if (video.readyState >= 2 && !video.paused && video.videoWidth) {
      const img = capturarFrame(video, video.videoWidth, video.videoHeight, MAX_LADO_VIDEO, regionVideo());
      const modo = estado.modo;
      const { probs, ms } = await clasificar(img, modo);
      if (!estado.corriendo) break;
      const suave = suavizador.agregar(probs);          // promedio de los últimos 3 s
      const r = interpretar(suave.probs, estado.umbral);
      estado.frames++;
      pintarResultado(r, suave.n);
      pintarStats(ms);
      agrupador.procesar(r, Date.now(), { modo, camara: CAMARA_DE_FUENTE[estado.fuente] });
    }
    const espera = -(performance.now() - t0);         // el modelo preciso ya marca el ritmo (~1 frame/s)
    await new Promise((ok) => setTimeout(ok, Math.max(espera, 16)));
  }
}

async function iniciar() {
  try {
    if (estado.fuente === "camara") {
      estado.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      video.srcObject = estado.stream;
    } else if (estado.fuente === "video") {
      if (!estado.archivoVideo) return toast("Primero elige un video.");
      video.srcObject = null;
      video.src = estado.archivoVideo;
    }
    video.hidden = false;
    lienzoFoto.hidden = true;
    await video.play();
    mostrarVacio(null);
    estado.corriendo = true;
    estado.tiempos = [];
    suavizador.reiniciar();
    agrupador.reiniciar();
    $("btn-iniciar").textContent = "Detener";
    $("btn-iniciar").dataset.corriendo = "si";
    bucle();
  } catch (e) {
    console.error(e);
    toast(e.name === "NotAllowedError" ? "Permiso denegado para usar la cámara o la pantalla."
      : `No se pudo iniciar: ${e.message}`);
  }
}

function detener() {
  estado.corriendo = false;
  agrupador.cerrar();                     // el animal que estaba en pantalla se guarda como evento
  if (estado.stream) estado.stream.getTracks().forEach((t) => t.stop());
  estado.stream = null;
  video.pause();
  $("btn-iniciar").textContent = "Iniciar";
  $("btn-iniciar").dataset.corriendo = "no";
  pintarEventoActual();
}

async function clasificarFoto(archivo) {
  const bitmap = await createImageBitmap(archivo, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const img = capturarFrame(bitmap, bitmap.width, bitmap.height, MAX_LADO_FOTO);
  lienzoFoto.width = img.width;
  lienzoFoto.height = img.height;
  lienzoFoto.getContext("2d").putImageData(img, 0, 0);
  lienzoFoto.hidden = false;
  video.hidden = true;
  mostrarVacio(null);
  const { probs, ms } = await clasificar(img, estado.modo);
  estado.ultimaFoto = img;
  estado.frames++;
  pintarResultado(interpretar(probs, estado.umbral));
  pintarStats(ms);
}

// ---------------------------------------------------------------- controles
const TEXTO_VACIO = {
  camara: "Pulsa <strong>Iniciar</strong> para encender la cámara.",
  video: "Elige un video (.mp4, .webm) y pulsa <strong>Iniciar</strong>.",
  foto: "Elige una foto: se clasifica al instante.",
};

function cambiarFuente(fuente) {
  if (estado.corriendo) detener();
  estado.fuente = fuente;
  document.querySelectorAll(".cam-tab").forEach((b) =>
    b.setAttribute("aria-selected", String(b.dataset.fuente === fuente)));
  const conArchivo = fuente === "video" || fuente === "foto";
  $("btn-archivo").hidden = !conArchivo;
  $("btn-archivo-texto").textContent = fuente === "foto" ? "Elegir foto" : "Elegir video";
  $("archivo").accept = fuente === "foto" ? "image/*" : "video/*";
  $("btn-iniciar").hidden = fuente === "foto";
  $("btn-zona").hidden = fuente === "foto";
  terminarMarcado();
  fijarZona(null);                        // cada fuente empieza analizando el frame completo
  video.hidden = false;
  lienzoFoto.hidden = true;
  video.removeAttribute("src");
  video.srcObject = null;
  $("etiqueta").hidden = true;
  mostrarVacio(TEXTO_VACIO[fuente]);
}

document.querySelectorAll(".cam-tab").forEach((b) => b.addEventListener("click", () => cambiarFuente(b.dataset.fuente)));
$("btn-iniciar").addEventListener("click", () => (estado.corriendo ? detener() : iniciar()));
$("btn-refrescar").addEventListener("click", () => refrescarHistorial());
$("btn-zona").addEventListener("click", empezarMarcado);
$("btn-quitar-zona").addEventListener("click", () => fijarZona(null));
$("umbral").addEventListener("input", (e) => {
  estado.umbral = Number(e.target.value);
  $("umbral-valor").textContent = estado.umbral.toFixed(2);
});
$("archivo").addEventListener("change", async (e) => {
  const archivo = e.target.files[0];
  if (!archivo) return;
  if (estado.fuente === "foto") {
    clasificarFoto(archivo).catch((err) => toast(`No se pudo leer la foto: ${err.message}`));
  } else {
    if (estado.corriendo) detener();
    if (estado.archivoVideo) URL.revokeObjectURL(estado.archivoVideo);
    estado.archivoVideo = URL.createObjectURL(archivo);
    mostrarVacio(`Video listo: <strong>${archivo.name}</strong>. Pulsa <strong>Iniciar</strong>.`);
  }
  e.target.value = "";
});
video.addEventListener("ended", () => estado.fuente === "video" && detener());

// ---------------------------------------------------------------- arranque
async function arrancar() {
  try {
    window.__meta = await cargarMeta();
    await cargarModelo("preciso");
    $("estado-sistema").textContent = "Modelo listo · ONNX en el navegador";
    $("estado-sistema").dataset.ok = "si";
  } catch (e) {
    console.error(e);
    $("estado-sistema").textContent = "No se pudo cargar el modelo";
  }
  try {
    const salud = await (await fetch("/api/salud")).json();
    $("stat-base").textContent = salud.base === "postgres" ? "PostgreSQL" : "SQLite local";
  } catch {
    $("stat-base").textContent = "sin API";
  }
  refrescarHistorial();
  // Espacio del asistente de Tomás: si existe /chat/chat.js, se monta en #chat
  try {
    const chat = await import("/chat/chat.js");
    chat.montarChat($("chat"), { apiBase: "/api" });
  } catch {
    /* el chat aún no está: queda el texto de reserva */
  }
}

arrancar();
