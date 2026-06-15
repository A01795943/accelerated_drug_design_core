## Surrogate Alphafold Model for plddt and iptm

A continuación se presenta la estructura del folder que estará destinado para realizar las predicciones mediante Docker.
Como entrada se tendrá una cadena individual o un batch de cadenas, y como salida se tienen las métricas pLDDT e ipTM.
Es importante aclarar, que tanto las arquitecturas, como modelos ganadores, se pueden modificar en caso de que se encuentren mejores modelos, lo único que se tiene que cambiar es el nombre de los modelos y de ahí en el archivo de .config para sustituirlo correctamente, este contenedor, sólo toma los mejores modelos ya guardados y entrenados y de ahí predice los valores de iptm y plddt.

```
alphafold_predictor/
│
├── Dockerfile = Archivo de Docker que carga modelos y ejecuta el predictor en modo inferencia
├── requirements.txt = Dependencias de python utilizadas
├── .dockerignore = Archivo que ignora ciertos archivos cuando se crea el contenedor de Docker
├── mpnn_results.csv = Archivo de entrada de ejemplo para el modo de ejecución 'batch'
├── README.md = Instrucciones propias del surrogate model para alphafold.
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── esm_embedder.py
│   ├── mlp_predictor.py
│   └── batch_predictor.py
│   └── .... more files
│
├── models/
│   └── best_model_plddt_vf.joblib = Modelo ganador Gradient Boosting para predecir el plddt
│   └── best_model_snn_vf.pth = Modelo de red neuronal para extraer features
│   └── mi modelo_ganador_xgb_vf.json = Toma las features dell modelo de snn y luego se alimenta este modelo de XGBoost para predecir el iptm
│
├── output/ 
│   └── outputs.csv = Archivo de salida luego de ejecutar el predictor en modo batch     
│
└── run_prediction.py = Script principal que ejecuta el predictor ya sea en modo single o batch
```

El flujo normal de inferencia se presenta a continuación:
```
Usuario: "CADENA1/CADENA2"
         ↓
IMPORTANTE: "CADENA1 = Cadena cerca de 350 aminoacidos de longitud, la cual se obtuvo de extraer zona adecuada para el GLP1R mediante inspección con MolViewer. Esta cadena puede cambiarse si se decide entrenar para otras métricas u otros fines"
         ↓
IMPORTANTE: "CADENA2 = Cadena de 20 aminoacidos que es la que se genera luego de correr el experimento de generación de muestras, pues es lo que se quiere llegar a crear, una cadena lo suficientemente buena con valores de iptm y plddt aceptables"
         ↓
validate_sequence() → CADENA 1 Y CADENA 2 son aminoacidos validos y son separados por '/' 
         ↓
valid input example: VSLWETVQKWREYRRQCQRSLTEDPPPATDLFCNRTFDEYACWPDGEPGSFVNVSCPW
YLPWASSVPQGHVYRFCTAEGLWLQKDNSSLPWRDLSECEEEQLLFLYIIYTVGYALS
FSALVIASAILLGFRHLHCTRNYIHLNLFASFILRALSVFIKDAALKWMYSTAAQQHQ
WDGLLSYQDSLSCRLVFLLMQYCVAANYYWLLVEGVYLYTLLAFSVFSEQWIFRLYVS
IGWGVPLLFVVPWGIVKYLYEDEGCWTRNSNMNYWLIIRLPILFAIGVNFLIFVRVIC
IVVSKLKIKCRLAKSTLTLIPLLGTHEVIFAFVMDEHARGTLRFIKLFTELSFTSFQG
LMVAI/SYEPEILKGFEELYLAQAKK
         ↓
IMPORTANTE : Es necesario dar la entrada tanto en modo single como en modo batch de la manera de "CADENA1/CADENA2" porque así está el formato de ejecutar el experimento que genera las muestras, se puede cambiar pero se debe tener cuidado con todos los archivos
         ↓
tokenizer → [0, 15, 18, 21, ...] = De cadenas se pasa a modelo de embeddings por letra, es decir, si damos como entrada 20 experimentos, se genera una matriz de embeddings de 20x1280
         ↓
ESM model → embedding [1280 valores]
         ↓
torch.tensor() → tensor([...])
         ↓
plddt → prediction: 0.5 derivado del modelo GradientBoosting
         ↓
iptm → prediction: 0.856 derivado del modelo Siamese Network + XGBoost
         ↓
resultado: {'sequence': '...', 'prediction': {'plddt':0.5}, 'iptm':0.856}

```

## Instrucciones para ejecutar el predictor:

Primero se clona el repositorio.
```
git clone https://github.com/josemanuelgarciaogarrio/drug-design-diabetes-team29.git
```
Luego de tener docker instalado y estar en la ruta donde se encuentra el Dockerfile.
```
docker build -t predictor_test .
```
Una vez eso, corremos un contenedor teniendo esa imagen base. Primero podemos ejecutarlo sin ningún argumento para ver las opciones con:
```
docker run predictor_test
```
Así, si queremos ejecutarlo en modo single hacemos:
```
docker run predictor_test --mode single --sequence "VVSKLKIKCRLAKSTLTLIPLLGTHEVIFAFVMDEHARGTLRFIKLFTELSFTSFQG
LMVAI/SYEPEILKGFEELYLAQAKK"
```
Ahora, si queremos ejecutarlo en modo batch hacemos:
```
docker run predictor_test --mode batch --input "mpnn_results.csv"
```

