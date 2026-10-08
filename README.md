# accelerated_drug_design_core

Pipeline de diseño de fármacos acelerado con IA: generación de backbones (RFdiffusion), diseño de secuencias (ProteinMPNN), validación estructural (AlphaFold/Rosetta) e inferencia surrogate (ptm/iptm).

## Estructura del proyecto

```
/
├── docker/
│   └── Dockerfile
├── pipeline/
│   ├── 1_run_rfdiffusion.py
│   ├── 2_run_mpnn_af.py
│   ├── 3_run_rosetta.py
│   ├── 4_run_inference.py
│   ├── mpnn_diverse_af.py
│   └── esm2_embedder.py
├── api/
│   ├── api.py
│   └── model.pkl
├── common/
│   └── logger.py
├── outputs/
│   └── .gitkeep
├── .dockerignore
├── .gitignore
├── LICENSE
└── README.md
```

## Arranque con Docker

```bash
# Build
docker build -f docker/Dockerfile -t drug-accelerator .

# Run
docker run -it --rm --gpus all \
     --shm-size=8g \
     -p 8000:8000 \
     -v $HOME/accelerated_drug_design_core:/workspace/repo \
     -v $HOME/accelerated_drug_design_core/outputs:/workspace/outputs \
     drug-accelerator
```

La API REST queda disponible en `http://localhost:8000`. Comprueba el estado con `GET /health`.

## Métricas del host

`GET /metrics/system` devuelve un snapshot ligero (no bloquea) para que el backend decida carga:

| Campo | Contenido |
|-------|-----------|
| `timestamp`, `hostname` | ISO UTC y nombre del host |
| `cpu` | `percent` (`psutil.cpu_percent(interval=None)`), `count`, `load_avg_1m` |
| `memory` | `total_mb`, `used_mb`, `percent` |
| `gpus` | Por GPU: `index`, `name`, `util_percent`, `mem_used_mb`, `mem_total_mb`, `mem_percent`, `temperature_c`. Lista vacía si NVML no está o no hay GPU |
| `disk` | Uso de `/workspace/outputs`: `total_gb`, `used_gb`, `percent` |
| `network` | `bytes_sent`, `bytes_recv` acumulados y `sent_per_sec`, `recv_per_sec` respecto a la muestra anterior del proceso (0 en la primera llamada) |
| `runs` | `running_count` y `running_run_ids` de `run_status` con status `RUNNING` |
