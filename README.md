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
