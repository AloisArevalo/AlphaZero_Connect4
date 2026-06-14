# AlphaZero para Conecta 4

Implementación en Python de un agente de **Conecta 4** (Connect 4) basado en el algoritmo **AlphaZero**. El programa aprende a jugar desde cero mediante **auto-juego**, **búsqueda Monte Carlo (MCTS)** y una **red neuronal convolucional residual**, sin conocimiento previo del juego.

---

## Tabla de contenidos

- [Descripción general](#descripción-general)
- [Arquitectura del sistema](#arquitectura-del-sistema)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos e instalación](#requisitos-e-instalación)
- [Uso](#uso)
- [Ciclo de entrenamiento](#ciclo-de-entrenamiento)
- [Configuración](#configuración)
- [Archivos generados](#archivos-generados)
- [Personalización](#personalización)
- [Rendimiento y consejos](#rendimiento-y-consejos)
- [Referencias](#referencias)

---

## Descripción general

Este proyecto entrena un agente que juega Conecta 4 en un tablero estándar de **6 filas × 7 columnas**. El objetivo es alinear **4 fichas consecutivas** en horizontal, vertical o diagonal.

El entrenamiento sigue el paradigma AlphaZero:

1. La red neuronal evalúa posiciones y sugiere movimientos probables.
2. MCTS explora el árbol de juego guiado por esa red.
3. El agente juega partidas contra sí mismo y aprende de esas experiencias.
4. Periódicamente se evalúa contra el mejor modelo guardado; solo se promueve si mejora de forma significativa.

No se requiere ningún dataset externo: todo el aprendizaje surge del auto-juego.

---

## Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    BUCLE DE ENTRENAMIENTO                   │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │Auto-juego│───▶│  Búfer   │───▶│Entrenar  │───▶│Torneo  │ │
│  │(MCTS+red)│    │repetición│    │  red NN  │    │eval.   │ │
│  └──────────┘    └──────────┘    └──────────┘    └────┬───┘ │
│       ▲                                               │     │
│       └───────────── mejor modelo guardado ───────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Componentes principales

| Componente | Descripción |
|------------|-------------|
| **Connect4** | Entorno del juego: tablero, movimientos legales, detección de victoria/empate |
| **Connect4Net** | Red convolucional residual con cabezal de política (7 columnas) y cabezal de valor (−1 a +1) |
| **MCTS** | Búsqueda con selección PUCT, ruido Dirichlet en la raíz y N simulaciones por movimiento |
| **Self-play** | Generación de partidas completas; guarda estado, política MCTS y resultado final |
| **ReplayBuffer** | Memoria de las últimas N experiencias para entrenamiento por lotes |
| **Evaluación** | Torneo entre modelo actual y mejor modelo; promoción si win rate > 55% |

### Representación del estado

Cada posición se codifica como un tensor de forma `(3, 6, 7)` desde la perspectiva del jugador que mueve:

| Canal | Contenido |
|-------|-----------|
| 0 | Fichas del jugador actual |
| 1 | Fichas del oponente |
| 2 | Casillas vacías |

Valores: `0` o `1`.

### Red neuronal

- **Cuerpo**: capa convolucional inicial + 5 bloques residuales (`ResidualBlock`)
- **Cabeza de política**: conv 1×1 → lineal → softmax sobre 7 columnas (movimientos ilegales enmascarados)
- **Cabeza de valor**: conv 1×1 → capas densas → `tanh` (probabilidad estimada de victoria)

### MCTS (PUCT)

En cada nodo se selecciona la acción que maximiza:

\[
Q(s,a) + c_{\text{puct}} \cdot P(s,a) \cdot \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)}
\]

Donde:
- \(Q(s,a)\): valor medio acumulado del hijo
- \(P(s,a)\): prior de la red neuronal
- \(N(s,a)\): número de visitas del hijo
- \(c_{\text{puct}}\): constante de exploración (por defecto `1.0`)

En el nodo raíz se añade **ruido Dirichlet** para fomentar la exploración durante el auto-juego.

---

## Estructura del proyecto

```
.
├── main.py                    # Punto de entrada principal
├── alphazero_connect4.py      # Alias de compatibilidad (mismo comportamiento)
├── requirements.txt           # Dependencias Python
├── README.md                  # Este archivo
├── training.log               # Log de entrenamiento (generado al ejecutar)
├── checkpoints/
│   └── best_model.pt          # Mejor modelo guardado (generado al entrenar)
└── connect4/                  # Paquete principal
    ├── __init__.py            # API pública
    ├── config.py              # Parámetros configurables (clase Config)
    ├── game.py                # Entorno Connect4
    ├── tensor_utils.py        # Conversión segura NumPy ↔ PyTorch
    ├── network.py             # ResidualBlock, Connect4Net
    ├── mcts.py                # Monte Carlo Tree Search
    ├── self_play.py           # Auto-juego y selección de movimientos
    ├── training.py            # Búfer de repetición y función de pérdida
    ├── evaluation.py          # Torneos y checkpoints
    ├── logging_utils.py       # Configuración de logs
    └── trainer.py             # Bucle principal de entrenamiento
```

---

## Requisitos e instalación

### Dependencias

- Python 3.10+
- [PyTorch](https://pytorch.org/) ≥ 2.0
- [NumPy](https://numpy.org/) ≥ 1.24, **< 2.0** (compatibilidad con PyTorch 2.2.x)

### Instalación

```bash
# Clonar o copiar el proyecto y entrar en el directorio
cd Test

# Instalar dependencias
pip install -r requirements.txt
```

> **Nota:** Si aparece el aviso *"A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"*, instala una versión compatible:
>
> ```bash
> pip install "numpy>=1.24.0,<2.0.0"
> ```

---

## Uso

### Iniciar el entrenamiento

```bash
python3 main.py
```

También puedes usar el script de compatibilidad:

```bash
python3 alphazero_connect4.py
```

El programa arranca desde cero (pesos aleatorios) o **reanuda** automáticamente si existe `checkpoints/best_model.pt`.

Para detener el entrenamiento de forma segura:

```
Ctrl + C
```

Al interrumpir, se guarda un checkpoint en `checkpoints/best_model.pt`.

### Importar el paquete desde otro script

```python
from connect4 import Config, Connect4, Connect4Net, train_alphazero

config = Config(mcts_simulations=100)
train_alphazero(config)
```

---

## Ciclo de entrenamiento

Cada **iteración** ejecuta las siguientes fases:

### 1. Auto-juego
- Se generan `self_play_games_per_iter` partidas (por defecto 100).
- En cada movimiento, MCTS calcula una distribución de probabilidad sobre las columnas.
- Los primeros `temperature_moves` movimientos (15) usan **muestreo con temperatura** para explorar; el resto elige la columna más visitada.
- Se guarda cada posición como `(estado, política_MCTS, resultado_final)`.

### 2. Entrenamiento
- Las experiencias se añaden al búfer de repetición (máx. 100 000 estados).
- Se entrena la red durante `train_epochs` épocas con lotes aleatorios.
- **Pérdida total** = pérdida de valor (MSE) + pérdida de política (entropía cruzada) + regularización L2.

### 3. Evaluación (cada 200 partidas acumuladas)
- Torneo de 40 partidas entre el modelo actual y el mejor guardado.
- 20 partidas empezando como jugador 1 y 20 como jugador 2.
- MCTS sin ruido y selección determinista (100 simulaciones).
- Si el modelo actual gana más del **55%** de las partidas decididas, se convierte en el nuevo mejor modelo.

### Ejemplo de salida en consola

```
============================================================
  AlphaZero para Conecta 4
  Presiona Ctrl+C para detener el entrenamiento
============================================================
2026-06-14 13:53:49 | INFO | AlphaZero Connect 4 — Inicio de entrenamiento
2026-06-14 13:53:49 | INFO | Dispositivo: cpu
2026-06-14 13:58:45 | INFO |   Auto-juego: 3577 estados | Búfer: 3577/100000 | Tiempo: 294.5s
2026-06-14 13:59:36 | INFO |   Entrenamiento: loss=1.8838 (value=0.2535, policy=1.1608)
2026-06-14 14:11:07 | INFO |   Tasa de victorias del modelo actual: 0.0%
```

---

## Configuración

Todos los parámetros están centralizados en la clase `Config` (`connect4/config.py`). Puedes modificarlos en `main.py`:

```python
from connect4 import Config, train_alphazero

config = Config(
    mcts_simulations=50,           # Simulaciones MCTS por movimiento (auto-juego)
    self_play_games_per_iter=100,  # Partidas por iteración
    buffer_max_size=100_000,       # Tamaño máximo del búfer
    batch_size=256,                # Tamaño de lote de entrenamiento
    train_epochs=10,               # Épocas por iteración
    learning_rate=1e-3,            # Tasa de aprendizaje (Adam)
    weight_decay=1e-4,             # Regularización L2
    eval_frequency=200,            # Torneo cada N partidas acumuladas
    eval_mcts_simulations=100,     # Simulaciones MCTS en evaluación
    temperature_moves=15,          # Movimientos con exploración estocástica
    win_threshold=0.55,            # Umbral de promoción del mejor modelo
)

train_alphazero(config)
```

### Parámetros MCTS

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `mcts_simulations` | 50 | Simulaciones por movimiento en auto-juego |
| `c_puct` | 1.0 | Constante de exploración PUCT |
| `dirichlet_epsilon` | 0.25 | Peso del ruido Dirichlet en la raíz |
| `dirichlet_alpha` | 0.03 | Parámetro α del ruido Dirichlet |

---

## Archivos generados

| Archivo | Descripción |
|---------|-------------|
| `training.log` | Historial de iteraciones, pérdidas, tiempos y resultados de torneos |
| `checkpoints/best_model.pt` | Checkpoint del mejor modelo (pesos, optimizador, iteración, partidas) |

El checkpoint contiene:

```python
{
    "model_state_dict": ...,      # Pesos de Connect4Net
    "optimizer_state_dict": ...,  # Estado del optimizador Adam
    "iteration": int,             # Última iteración guardada
    "total_games": int,           # Partidas totales jugadas
}
```

---

## Personalización

### Cambiar el dispositivo de cómputo

```python
config = Config(device="cuda")   # GPU NVIDIA
config = Config(device="cpu")    # CPU (por defecto si no hay GPU)
```

### Aumentar la fuerza de juego en inferencia

Sube las simulaciones MCTS en evaluación y auto-juego:

```python
config = Config(
    mcts_simulations=200,
    eval_mcts_simulations=400,
)
```

Más simulaciones implican partidas más fuertes pero mayor tiempo de cómputo.

### Entrenar más rápido (menos calidad inicial)

```python
config = Config(
    mcts_simulations=25,
    self_play_games_per_iter=20,
    train_epochs=3,
)
```

---

## Rendimiento y consejos

| Aspecto | Estimación (CPU, config por defecto) |
|---------|--------------------------------------|
| 100 partidas de auto-juego | ~5 min |
| 1 iteración completa | ~6 min |
| Torneo de evaluación (40 partidas) | ~4 min |

**Consejos:**

- **GPU**: si tienes CUDA disponible, el entrenamiento de la red será notablemente más rápido; el auto-juego seguirá siendo el cuello de botella por el MCTS secuencial.
- **Paciencia**: alcanzar un nivel muy alto requiere miles de partidas y muchas iteraciones. Las primeras evaluaciones suelen mostrar 0% o 50% de victorias mientras la red aprende patrones básicos.
- **Reanudar entrenamiento**: basta con volver a ejecutar `python3 main.py`; el checkpoint se carga automáticamente.
- **NumPy 2.x**: el código incluye un fallback si el puente PyTorch–NumPy falla, pero se recomienda usar `numpy<2` para evitar avisos y mejor rendimiento.

---

## Referencias

- Silver, D. et al. (2017). *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* — [AlphaZero paper](https://arxiv.org/abs/1712.01815)
- Conecta 4: juego de estrategia para dos jugadores en tablero 6×7

---

## Licencia

Código educativo de libre uso para aprendizaje e investigación.
