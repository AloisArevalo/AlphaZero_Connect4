# README - Modo Juego Interactivo

Este directorio contiene la interfaz gráfica para jugar Connect 4 contra la IA entrenada.

## Archivos

- `play.py`:          Script principal (GUI gráfica)
- `terminal_play.py`: Script alternativo (modo terminal)
- `game_ui.py`:       Interfaz gráfica tkinter (completa)
- `game_engine.py`:   Motor que envuelve el modelo entrenado
- `utils.py`:         Funciones auxiliares
- `README.md`:        Este archivo

## Requisitos previos

1. Primero debe entrenar el modelo ejecutando en la carpeta raíz:
   ```bash
   python main.py
   ```
   O para entrenamiento automático:
   ```bash
   python auto_train.py
   ```

2. Espera a que se genere el archivo: `checkpoints/best_model.pt`

## Cómo jugar

### Opción 1: Interfaz Gráfica (recomendado)

Desde la carpeta raíz del proyecto:

```bash
# Dificultad normal (100 simulaciones MCTS)
python game_ui/play.py

# O con dificultad específica:
python game_ui/play.py --difficulty easy    # 25 simulaciones (más rápido)
python game_ui/play.py --difficulty normal  # 100 simulaciones (recomendado)
python game_ui/play.py --difficulty hard    # 200 simulaciones (más fuerte)
python game_ui/play.py --difficulty expert  # 400 simulaciones (muy fuerte)
```

#### Controles GUI
- **Hacer movimiento:** Haz clic en la columna donde quieres poner tu ficha
- **Nueva partida:** Botón "Nuevo Juego"
- **Sugerencia:** Botón "Sugerencia" (solo en tu turno)
- **Salir:** Botón "Salir"

### Opción 2: Terminal (para debugging)

```bash
# Modo terminal interactivo
python game_ui/terminal_play.py

# Te pide que escribas el número de columna (0-6) para cada movimiento
```

## Colores en Terminal

- **VERDE:** Éxito, tu turno
- **CYAN:** Información, turno IA
- **ROJO:** Error, IA ganó
- **AMARILLO:** Advertencia, empate

## Dificultades explicadas

| Dificultad | MCTS Sims | Velocidad | Fuerza | Ideal para |
|-----------|-----------|----------|--------|-----------|
| Easy      | 25        | Muy rápido | Débil | Aprender/Principiantes |
| Normal    | 100       | Normal | Medio | Juego casual (recomendado) |
| Hard      | 200       | Lento | Fuerte | Reto |
| Expert    | 400       | Muy Lento | Muy Fuerte | Expertos |

## Ejemplo de uso completo

```bash
# 1. Entrena el modelo (en la carpeta raíz, en background)
cd /path/to/AlphaZero_Connect4
python auto_train.py &

# 2. Cuando haya generado checkpoints/best_model.pt, abre otro terminal
python game_ui/play.py

# 3. ¡A jugar!
```

## Solución de problemas

### Error: "No se encontró el modelo en checkpoints/best_model.pt"

**Solución:** Primero debes entrenar el modelo:
```bash
# Desde la carpeta raíz
python main.py       # O
python auto_train.py

# Espera a que se genere checkpoints/best_model.pt
```

### La IA piensa muy lentamente

- Esto es normal con dificultades altas
- Usa `--difficulty easy` para juego más rápido
- O espera, ¡está calculando bien!

### El juego se congela en GUI

Probablemente la IA está pensando (esto es normal). Tiene que simular muchos movimientos futuros.
- En terminal verás el mensaje "La IA esta pensando..."
- En GUI verás "IA esta pensando..."

### El tablero no se ve bien

Asegúrate de que tkinter esté instalado:
```bash
python -m tkinter  # Para verificar
```

## Notas técnicas

- La IA usa el mejor modelo guardado durante el entrenamiento
- Cada simulación MCTS es una exploración del árbol de juego futuro
- La red neuronal evalúa posiciones y guía la búsqueda
- El tablero está codificado internamente como (6, 7)
- Los movimientos se validan antes de ejecutar
- La GUI usa threading para que la IA no congele la interfaz

## Flujo del juego

1. **Inicialización:** Carga el modelo entrenado desde `checkpoints/best_model.pt`
2. **Tu turno:** Haz clic en una columna (o escribe en terminal)
3. **Validación:** Se verifica que el movimiento sea legal
4. **IA piensa:** MCTS simula N movimientos futuros (N depende de dificultad)
5. **IA juega:** Elige el mejor movimiento según la búsqueda
6. **Verificación:** Se comprueba si alguien ganó o si fue empate
7. **Repetir:** Hasta que alguien gane o se llene el tablero

## Licencia

Código educativo de libre uso para aprendizaje e investigación.
