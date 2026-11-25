// eventos.js - Convierte la secuencia de frames clasificados en EVENTOS (contrato v1).
//
// Un evento = una aparición de un animal, no un frame:
//   - empieza cuando la misma clase sale como animal (no fondo y confianza >= umbral)
//     en FRAMES_PARA_CONFIRMAR frames seguidos;
//   - termina cuando pasan MS_SIN_VERLO sin ver esa clase;
//   - al terminar se envía con POST /api/eventos.

export const FRAMES_PARA_CONFIRMAR = 3;
export const MS_SIN_VERLO = 2000;

export class AgrupadorEventos {
  constructor({ onInicio, onFin } = {}) {
    this.onInicio = onInicio ?? (() => {});
    this.onFin = onFin ?? (() => {});
    this.reiniciar();
  }

  reiniciar() {
    this.racha = null;   // { clase, frames: [{t, conf}] } aún sin confirmar
    this.activo = null;  // { clase, inicio, fin, confs, modo, camara }
  }

  // r = resultado de interpretar(); t = Date.now() del frame
  procesar(r, t, { modo, camara }) {
    if (r.es_animal) {
      if (this.activo && r.clase === this.activo.clase) {
        this.activo.fin = t;
        this.activo.confs.push(r.confianza);
        this.racha = null;
      } else {
        if (!this.racha || this.racha.clase !== r.clase) this.racha = { clase: r.clase, frames: [] };
        this.racha.frames.push({ t, conf: r.confianza });
        if (this.racha.frames.length >= FRAMES_PARA_CONFIRMAR) {
          if (this.activo) this.cerrar();          // otra especie se confirmó: cierra la anterior
          const f = this.racha.frames;
          this.activo = { clase: r.clase, inicio: f[0].t, fin: t, confs: f.map((x) => x.conf), modo, camara };
          this.racha = null;
          this.onInicio(this.activo);
        }
      }
    } else {
      this.racha = null;
    }
    this.revisar(t);
  }

  // Cierra el evento activo si ya pasó el tiempo sin ver al animal (llamar también con un timer)
  revisar(t = Date.now()) {
    if (this.activo && t - this.activo.fin > MS_SIN_VERLO) this.cerrar();
  }

  cerrar() {
    if (!this.activo) return;
    const a = this.activo;
    this.activo = null;
    const confianza = a.confs.reduce((s, c) => s + c, 0) / a.confs.length;
    this.onFin({
      inicio: new Date(a.inicio).toISOString(),
      fin: new Date(a.fin).toISOString(),
      camara: a.camara,
      clase: a.clase,
      confianza: Math.round(confianza * 1e4) / 1e4,
      confianza_max: Math.round(Math.max(...a.confs) * 1e4) / 1e4,
      n_frames: a.confs.length,
      modo: a.modo,
    });
  }
}

export async function guardarEvento(evento) {
  const resp = await fetch("/api/eventos", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(evento),
  });
  if (!resp.ok) throw new Error(`POST /api/eventos ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

export async function listarEventos(limite = 20) {
  const resp = await fetch(`/api/eventos?limite=${limite}`);
  if (!resp.ok) throw new Error(`GET /api/eventos ${resp.status}`);
  return resp.json();
}
