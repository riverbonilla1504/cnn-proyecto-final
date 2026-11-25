const campos = ["intencion", "clase", "periodo"];
const rutaModelo = new URL("./transformer.onnx", import.meta.url);
const rutaVocabulario = new URL("./vocab.json", import.meta.url);
const rutaEstilos = new URL("./chat.css", import.meta.url);
const rutaTono = new URL("./tono.onnx", import.meta.url);
const rutaVocabularioTono = new URL("./tono.json", import.meta.url);
const umbralIdentidad = 0.85;
const fotos = [1, 2].map((numero) => new URL(`./javiercito-${numero}.jpeg`, import.meta.url).href);
const horaLocal = new Intl.DateTimeFormat("es-CO", {
  timeZone: "America/Bogota", dateStyle: "short", timeStyle: "short",
});
const fuentes = {
  webcam_1: "Cámara", video: "Video", pestana: "Pestaña", simulada: "Simulado",
};
export function codificarPregunta(texto, vocab) {
  const normalizado = texto.toLowerCase().normalize("NFD").replace(/\p{M}/gu, "");
  const palabras = normalizado.match(/[a-z0-9]+/g) ?? [];
  const ids = new Int32Array(vocab.longitud);
  if (palabras.length === 0) ids[0] = vocab.tokens["<unk>"];
  for (let i = 0; i < Math.min(palabras.length, ids.length); i++) {
    const palabra = palabras[i];
    ids[i] = Object.hasOwn(vocab.tokens, palabra) ? vocab.tokens[palabra] : vocab.tokens["<unk>"];
  }
  return ids;
}

export async function interpretarPregunta(texto, vocab, sesion, runtime) {
  const ids = codificarPregunta(texto, vocab);
  const salidas = await sesion.run({ tokens: new runtime.Tensor("int32", ids, [1, ids.length]) });
  return Object.fromEntries(campos.map((campo) => {
    const valores = salidas[`prob_${campo}`].data;
    const etiquetas = vocab.etiquetas[campo];
    if (valores.length !== etiquetas.length) throw new Error("Salida incompatible con el vocabulario.");
    let mejor = 0;
    for (let i = 1; i < valores.length; i++) {
      if (valores[i] > valores[mejor]) mejor = i;
    }
    return [campo, etiquetas[mejor]];
  }));
}

async function cargarModelo(runtime, modelo, vocabulario) {
  return Promise.all([
    fetch(vocabulario).then((respuesta) => {
      if (!respuesta.ok) throw new Error("No se pudo cargar el vocabulario.");
      return respuesta.json();
    }),
    runtime.InferenceSession.create(modelo.href, { executionProviders: ["wasm"] }),
  ]);
}

async function predecirTono(texto, [vocab, sesion], runtime) {
  const ids = codificarPregunta(texto, vocab);
  const salida = await sesion.run({ tokens: new runtime.Tensor("int32", ids, [1, ids.length]) });
  return salida.probabilidades.data;
}

async function generarEntrada(modelo, runtime) {
  const [vocab] = modelo;
  const opciones = [...vocab.marcadores, "fin"];
  const indices = opciones.map((opcion) => vocab.salidas.indexOf(opcion));
  const palabras = [];
  for (let paso = 0; paso < 3; paso++) {
    const probabilidades = await predecirTono(["tono", ...palabras].join(" "), modelo, runtime);
    const pesos = indices.map((indice) => probabilidades[indice]);
    let muestra = Math.random() * pesos.reduce((total, peso) => total + peso, 0);
    const indice = pesos.findIndex((peso) => {
      muestra -= peso;
      return muestra < 0;
    });
    const siguiente = opciones[indice] ?? "fin";
    if (siguiente === "fin") break;
    palabras.push(siguiente);
  }
  const frase = palabras.join(" ");
  return frase ? frase[0].toUpperCase() + frase.slice(1) : "";
}

export function montarChat(contenedor, { apiBase = "/api" } = {}) {
  const estilos = document.createElement("link");
  estilos.rel = "stylesheet";
  estilos.href = rutaEstilos.href;
  document.head.append(estilos);
  contenedor.classList.add("chat-panel");
  contenedor.setAttribute("aria-labelledby", "chat-titulo");
  contenedor.innerHTML = `
    <header class="chat-cabecera">
      <div class="chat-camara" role="img" aria-label="Avatar de Javiercito">
        <img src="${fotos[0]}" alt="" decoding="async">
        <img src="${fotos[1]}" alt="" decoding="async">
      </div>
      <div>
        <h2 id="chat-titulo">Javiercito</h2>
        <p>Consulta los animales registrados por la cámara.</p>
      </div>
      <button class="chat-limpiar" type="button" hidden>Limpiar chat</button>
    </header>
    <div class="chat-cuerpo">
      <div class="chat-historial" role="log" aria-label="Conversación" tabindex="0">
        <div class="chat-bienvenida">
          <p>Te cuento qué se vio por la cámara y cuándo.</p>
        </div>
      </div>
      <form class="chat-form">
        <label class="chat-solo-lector" for="chat-pregunta">Pregunta sobre los avistamientos</label>
        <input id="chat-pregunta" type="text" placeholder="Escribe una pregunta…" maxlength="300" required autocomplete="off">
        <button class="chat-enviar" type="submit">Consultar <span aria-hidden="true">↵</span></button>
      </form>
    </div>
  `;
  const form = contenedor.querySelector("form");
  const entrada = form.querySelector("input");
  const boton = form.querySelector("button");
  const historial = contenedor.querySelector(".chat-historial");
  const bienvenida = contenedor.querySelector(".chat-bienvenida");
  const limpiar = contenedor.querySelector(".chat-limpiar");
  const camara = contenedor.querySelector(".chat-camara");
  const movimientoReducido = matchMedia("(prefers-reduced-motion: reduce)");
  let carga;
  let cargaTono;

  limpiar.addEventListener("click", () => {
    historial.replaceChildren(bienvenida);
    bienvenida.hidden = false;
    limpiar.hidden = true;
    entrada.focus({ preventScroll: true });
  });

  function desplazar() {
    historial.scrollTop = historial.scrollHeight;
  }

  async function escribir(respuesta, texto) {
    if (movimientoReducido.matches) {
      respuesta.textContent = texto;
      return;
    }
    const caracteres = Array.from(texto);
    const efecto = document.createElement("span");
    efecto.setAttribute("aria-hidden", "true");
    respuesta.replaceChildren(efecto);
    respuesta.classList.add("chat-escribiendo");
    camara.dataset.hablando = "true";
    const duracion = Math.min(2200, Math.max(720, caracteres.length * 17));
    await new Promise((resolver) => {
      const inicio = performance.now();
      function paso(ahora) {
        const progreso = Math.min(1, (ahora - inicio) / duracion);
        efecto.textContent = caracteres.slice(0, Math.ceil(progreso * caracteres.length)).join("");
        desplazar();
        if (progreso < 1 && !movimientoReducido.matches) requestAnimationFrame(paso);
        else resolver();
      }
      requestAnimationFrame(paso);
    });
    respuesta.classList.remove("chat-escribiendo");
    delete camara.dataset.hablando;
    respuesta.textContent = texto;
  }

  async function enviar(texto) {
    const pregunta = texto.trim();
    if (!pregunta || boton.disabled) return;
    if (entrada.value.trim() === pregunta) entrada.value = "";
    bienvenida.hidden = true;
    limpiar.hidden = false;
    boton.disabled = true;
    limpiar.disabled = true;

    const turno = document.createElement("article");
    turno.className = "chat-turno";
    const mensaje = document.createElement("p");
    mensaje.className = "chat-pregunta";
    mensaje.textContent = pregunta;
    mensaje.setAttribute("aria-label", `Tú: ${pregunta}`);
    const burbuja = document.createElement("div");
    burbuja.className = "chat-burbuja";
    burbuja.dataset.estado = "pendiente";
    const respuesta = document.createElement("p");
    respuesta.className = "chat-respuesta";
    respuesta.textContent = "Consultando…";
    burbuja.append(respuesta);
    turno.append(mensaje, burbuja);
    historial.append(turno);
    desplazar();

    try {
      const runtime = globalThis.ort;
      if (!runtime) throw new Error("ONNX Runtime Web no está disponible.");
      cargaTono ??= cargarModelo(runtime, rutaTono, rutaVocabularioTono).catch(() => {
        cargaTono = undefined;
        return null;
      });
      const modeloTono = await cargaTono;
      let consulta;
      if (modeloTono) {
        const [vocab] = modeloTono;
        const probabilidades = await predecirTono(`pregunta ${pregunta}`, modeloTono, runtime);
        if (probabilidades[vocab.salidas.indexOf("identidad")] >= umbralIdentidad) {
          consulta = { intencion: "identidad", clase: "animales", periodo: "sin_periodo" };
        }
      }
      if (!consulta) {
        carga ??= cargarModelo(runtime, rutaModelo, rutaVocabulario).catch((error) => {
          carga = undefined;
          throw error;
        });
        const [vocab, sesion] = await carga;
        consulta = await interpretarPregunta(pregunta, vocab, sesion, runtime);
      }
      const resp = await fetch(`${apiBase}/consulta`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pregunta, ...consulta }),
      });
      if (!resp.ok) throw new Error("La API no pudo responder.");
      const datos = await resp.json();
      const entradaDeTono = modeloTono && datos.datos !== null
        ? await generarEntrada(modeloTono, runtime).catch(() => "") : "";
      burbuja.dataset.estado = datos.datos === null ? "aviso" : "lista";
      const texto = entradaDeTono
        ? `${entradaDeTono}, ${datos.respuesta[0].toLowerCase()}${datos.respuesta.slice(1)}`
        : datos.respuesta;
      await escribir(respuesta, texto);
      if (consulta.intencion === "listar" && datos.datos?.length) {
        const resultados = document.createElement("ul");
        resultados.className = "chat-eventos";
        for (const registro of datos.datos) {
          const li = document.createElement("li");
          const clase = document.createElement("strong");
          clase.textContent = registro.clase;
          const fecha = document.createElement("time");
          fecha.dateTime = registro.inicio;
          fecha.textContent = horaLocal.format(new Date(registro.inicio));
          const fuente = document.createElement("span");
          fuente.className = "chat-fuente";
          fuente.textContent = fuentes[registro.camara] ?? registro.camara;
          li.append(clase, fecha, fuente);
          resultados.append(li);
        }
        burbuja.append(resultados);
      }
    } catch {
      burbuja.dataset.estado = "aviso";
      respuesta.textContent = "No pude completar la consulta. Intenta nuevamente.";
    } finally {
      boton.disabled = false;
      limpiar.disabled = false;
      desplazar();
    }
  }

  form.addEventListener("submit", (evento) => {
    evento.preventDefault();
    enviar(entrada.value);
  });
}
