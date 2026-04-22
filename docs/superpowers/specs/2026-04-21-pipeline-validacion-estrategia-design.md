# Pipeline de Validación de Estrategia — Fases 2, 3, 4

**Fecha:** 2026-04-21
**Autor:** Jerónimo Deli
**Estado:** Diseño aprobado, pendiente plan de implementación

## 1. Contexto

El proyecto CPCV_tesis valida estrategias de trading intradía generadas por agentes.
La **Fase 1 (upstream, ya implementada)** consiste en que un agente selecciona
una estrategia `{features, modelo=XGBoost, hiperparámetros, ticker universe}`
usando CPCV sobre 1 año de data. Esa estrategia queda **LOCKED** (cero
re-optimización) y se pasa a las fases siguientes.

Este documento especifica las Fases 2, 3 y 4: cómo validar temporalmente,
estresar estadísticamente, y monitorear en paper trading la estrategia que
entrega el agente.

## 2. Data disponible

- OHLCV 1h, 2022-01 → 2026-04 (~4.3 años).
- Uso efectivo: desde 2023-01 (descartamos 2022 como buffer).
- 20 tickers, ~1,640 barras/año por ticker.

## 3. Configuración LOCKED heredada de Fase 1

Todo lo siguiente queda fijo desde Fase 2 en adelante:

- `features`: seleccionados por el agente.
- `XGBoost hyperparams`: `{n_estimators, max_depth, learning_rate, ...}`.
- `ticker universe`: 20 tickers.
- Horizonte de label: **2.5h fijo**.
- Ventana de entries permitidos: **{12:00, 12:30, 13:00, 13:30} ET** (evita primeras 2.5h y últimas 2.5h de sesión).
- Exit: 2.5h después del entry o EOD (lo que ocurra primero). **Nunca overnight.**
- Observaciones efectivas: ~1,000/año/ticker.

Purge y embargo entre train/test: **X barras de purge + Y barras de embargo**
(valores concretos a fijar en implementación, consistentes con los usados en CPCV de Fase 1).

## 4. Cronograma

| Fase | Ventana | Duración | Propósito |
|---|---|---|---|
| Fase 1 (upstream) | 2023-01 → 2024-01 | 12m | CPCV del agente (existe) |
| Purge + embargo | X + Y barras | placeholder | Separación Fase 1 ↔ 2 |
| **Fase 2** WF lock-down | 2024-01 → 2025-01 | 12m | Degradación temporal |
| **Fase 3** tortura (paralela a 2) | 2024-01 → 2025-01 | 12m | Stress statistical |
| Buffer + purge + embargo | ~3m | holgura | Separación 2+3 ↔ 4 |
| **Fase 4** paper + monitor | 2025-04 → 2026-04 | 12m | Simulación live + kill switch |

## 5. Fase 2 — Walk-Forward Lock-down

**Mandato:** medir degradación natural del modelo sobre 2024. Cero re-optimización.

### Configuración

- Ventana de train: **8 meses rolling**.
- Ventana de test: **4 meses**.
- Paso rolling: 4 meses (no overlap en tests).
- **3 folds totales** cubriendo 2024-01 → 2025-01:
  - Fold 1: train 2024-01 → 2024-09, test 2024-09 → 2025-01
  - Fold 2: train 2023-09 → 2024-05, test 2024-05 → 2024-09
  - Fold 3: train 2023-05 → 2024-01, test 2024-01 → 2024-05
- Re-entrenamiento XGBoost con features + hyperparams LOCKED al inicio de cada fold.
- Purge X + embargo Y barras entre train y test.

**Nota:** que el train reuse data de 2023 (vista por el agente en Fase 1) es
aceptable porque en Fase 2 **nada se optimiza**; sólo se re-fitea XGB con
parámetros fijos. Los **tests** caen 100% en 2024, data nunca vista.

### Outputs

- Serie de 3 Sharpes OOS (uno por fold) → `WF_sharpes`.
- PnL OOS por fold (serie temporal).
- Curva de degradación temporal (Sharpe vs. posición del fold).

## 6. Fase 3 — Tortura de Estrategia

Corre **en paralelo** a Fase 2 sobre la misma ventana 2024-01 → 2025-01.
Usa los PnLs y predicciones generados por Fase 2 como insumo. Cada bloque
produce su propio output; la decisión de ir a Fase 4 se toma mirando el
conjunto (no hay una relación formal única entre bloques).

### 6.1 Permutation Test (orden de trades)

- Para cada fold WF, tomar la serie de PnLs OOS.
- Permutar el **orden de los trades** 1,000 veces.
- Recalcular Sharpe en cada permutación → distribución nula.
- p-value = % permutaciones con Sharpe ≥ Sharpe real.
- **Criterio:** pasa si p < 0.05 en mayoría de folds.

### 6.2 Slippage & Latency Stress

- Grid de costos: `{0, 1, 2, 5, 10, 20}` bps por trade (round-trip).
- Para cada nivel, recalcular Sharpe de los 3 folds.
- **Output:** curva "Sharpe vs. costo" + break-even cost (donde Sharpe cruza 0).
- Latencia: retraso de ejecución `K ∈ {0, 1, 2}` barras.
- **Criterio:** la estrategia debe seguir positiva con costos realistas (~2-5 bps) y K=0 o 1.

### 6.3 Hyperparameter Sensitivity

- Grid ±1 step alrededor de valores LOCKED:
  - `n_estimators`: {50, 100, 200}
  - `max_depth`: {2, 3, 4}
  - `learning_rate`: {0.005, 0.01, 0.02}
- 27 combos × 3 folds WF = 81 runs.
- **Output:** heatmap de Sharpes vecinos al punto LOCKED.
- **Criterio:** mayoría de vecinos dentro de ±20% del Sharpe original (planicie, no pico).

### 6.4 Noise Injection

- Ruido gaussiano multiplicativo en OHLCV de test (train queda intacto):
  `precio' = precio × (1 + ε)`, `ε ~ N(0, σ)`, `σ ∈ {1, 5, 10}` bps.
- Re-calcular features en la ventana ruidosa, re-predecir con modelo
  congelado, recalcular PnL.
- **Output:** degradación de Sharpe vs. nivel de ruido.
- **Criterio:** Sharpe no cambia de signo con ruido ≤ 5 bps.

## 7. Fase 4 — Paper Trading + Monitor

**Mandato:** simular producción con data más reciente y detectar degradación live.

### Estructura temporal

- **2 folds live de 6 meses** cada uno, con train rolling 8m:

| Fold live | Train | Paper trade |
|---|---|---|
| Live-1 | 2024-09 → 2025-05 | 2025-05 → 2025-10 |
| Live-2 | 2025-03 → 2025-11 | 2025-11 → 2026-04 |

Modelo LOCKED (mismos features + hyperparams). Sólo re-fit de XGBoost.

### Distribución de referencia

- `μ_ref` = media de los 3 Sharpes WF de Fase 2.
- `σ_ref` = combinación de:
  - std de los 3 folds WF,
  - std de la distribución nula del permutation test (Fase 3.1),
  - dispersión bajo noise injection (Fase 3.4).

### Z-score de degradación

```
Z = (Sharpe_live - μ_ref) / σ_ref
```

Re-evaluado mensualmente (no sólo al cierre de fold).

### Reglas de alerta

- `Z > -1`: estrategia sana, sigue operando.
- `-2 < Z ≤ -1`: warning, flag en dashboard.
- `Z ≤ -2`: **kill switch** → desconectar estrategia.

### KPIs secundarios

- Drawdown máximo vs. DD observado en Fases 2+3.
- Hit rate (% trades ganadores) — debe quedar dentro de ±10% del hit rate de Fase 2.
- Turnover / trades-per-day.
- Feature drift: distancia KL entre distribución de features en train vs. live
  (early warning antes de que PnL lo refleje).

## 8. Arquitectura de componentes

Módulos nuevos bajo `cpcv_analysis/` (o carpeta nueva `validation_pipeline/`):

- **`locked_strategy.py`**: dataclass con la config LOCKED (features, params, entries allowed).
- **`fase2_wf.py`**: runner del walk-forward 3 folds; usa `_pnl_from_split` existente de `backtest_engine.py`.
- **`fase3_permutation.py`**: shuffle de trades + p-values.
- **`fase3_slippage.py`**: grid de costos + latencia.
- **`fase3_sensitivity.py`**: grid de hyperparams vecinos.
- **`fase3_noise.py`**: inyección de ruido OHLCV + re-feature + re-predict.
- **`fase4_paper.py`**: runner de 2 folds live + Z-score + alertas.
- **`reference_distribution.py`**: construye `μ_ref, σ_ref` a partir de outputs de Fase 2+3.
- **`monitor.py`**: dashboard / reporting (KPIs secundarios, feature drift).

Cada módulo: una función pública `run_fase_X(locked_strategy, data, ...) -> ResultDict`.

## 9. Fuera de alcance

- Selección de estrategia (Fase 1, ya existe).
- Ejecución real / broker integration (Fase 4 es simulación).
- Re-optimización de hiperparámetros en cualquier punto.

## 10. Decisiones abiertas (a resolver en plan de implementación)

- Valor concreto de `X` (purge) e `Y` (embargo) en barras 1h.
- Formato de output / storage (JSON, parquet, pickle por fase).
- Dashboard de Fase 4: notebook o app aparte.
