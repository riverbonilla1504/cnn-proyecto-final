-- Contrato v1 - tabla de eventos (PostgreSQL).
-- Un registro por APARICIÓN de un animal (no uno por frame). La escribe la página de cámara
-- (Javier) a través de POST /api/eventos; la lee el asistente (Tomás).
-- Cambiar este archivo solo con un PR aprobado por los dos.
--
-- Uso:  psql "$DATABASE_URL" -f contrato/esquema.sql
-- (La API también la crea sola la primera vez si no existe.)

create table if not exists eventos (
    id            bigint generated always as identity primary key,
    inicio        timestamptz not null,                 -- UTC: cuándo se confirmó el animal
    fin           timestamptz not null,                 -- UTC: último frame en que se vio
    camara        text        not null default 'webcam_1',
                  -- webcam_1 (webcam) | video (video subido) | pestana (pestaña compartida) | simulada (pruebas)
    clase         text        not null check (clase in ('bovino', 'caballo', 'gato', 'perro')),
                  -- 'fondo' nunca se guarda
    confianza     real        not null check (confianza between 0 and 1),      -- promedio del evento
    confianza_max real        not null check (confianza_max between 0 and 1),  -- la más alta del evento
    n_frames      integer     not null check (n_frames > 0),                   -- frames clasificados
    modo          text        not null check (modo in ('rapido', 'preciso')),
    miniatura_url text,                                                        -- vacía en la v1
    creado_en     timestamptz not null default now(),
    check (fin >= inicio)
);

create index if not exists eventos_inicio_idx on eventos (inicio desc);
create index if not exists eventos_clase_idx on eventos (clase);
