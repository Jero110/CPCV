# Plan de Implementación — Pipeline Validación Fases 2-4

**Spec base:** `docs/superpowers/specs/2026-04-21-pipeline-validacion-estrategia-design.md`
**Fecha:** 2026-04-21
**Entorno:** `conda run -n rappi` para todo Python.

## Principios guía

- **Fidelidad De Prado:** CPCV/purge/embargo ya existen en `cpcv_analysis/splitters.py`. Reusar, no reescribir.
- **Incremental:** entregar Fase 2 funcional antes de tocar Fase 3. No paralelizar la implementación aunque las fases corran en paralelo.
- **1 ticker primero:** todo el pipeline arranca con 1 ticker (SPY existente en config). Extensión a 20 tickers es un paso posterior, no se mezcla.
- **Sin re-optimización:** ningún módulo post-Fase-1 puede tocar hiperparámetros salvo el bloque explícito de `fase3_sensitivity.py`.

## Estructura de carpetas

```
cpcv_analysis/
  validation_pipeline/          # NUEVO
    __init__.py
    locked_strategy.py
    fase2_wf.py
    fase3_permutation.py
    fase3_slippage.py
    fase3_sensitivity.py
    fase3_noise.py
    fase4_paper.py
    reference_distribution.py
    monitor.py
tests/
  validation_pipeline/          # NUEVO
    test_locked_strategy.py
    test_fase2_wf.py
    test_fase3_permutation.py
    test_fase3_slippage.py
    test_fase3_sensitivity.py
    test_fase3_noise.py
    test_fase4_paper.py
    test_reference_distribution.py
notebooks/
  validation_pipeline_e2e.ipynb # demo integrador
```

---

## Paso 0 — Preparación de data (bloquea a todos)

**Objetivo:** data loader que devuelve OHLCV 1h 2023-01→2026-04 para 1 ticker, con labels de 2.5h y filtro de entries en {12:00, 12:30, 13:00, 13:30} ET.

### Tareas

1. Extender `cpcv_analysis/data.py`:
   - Función `load_ohlcv_1h(ticker, start, end) -> pd.DataFrame`.
   - Función `build_labels_25h(ohlcv) -> (X, y, t1, fwd_ret)` con horizonte 2.5h (5 barras de 30min, o 2.5 barras de 1h — confirmar timeframe real en implementación).
   - Función `filter_entries(df) -> df` que mantiene sólo timestamps en {12:00, 12:30, 13:00, 13:30} ET.
2. Test: `test_data_loader.py` — verificar shape, que `t1 - t0 == 2.5h`, que no hay entries overnight, que ventana horaria es correcta.

**Entrega:** `X, y, t1, fwd_ret` listos para consumir por el resto del pipeline.

---

## Paso 1 — LockedStrategy (bloquea Fases 2-4)

**Objetivo:** dataclass inmutable con la configuración que entrega Fase 1.

### Archivo: `validation_pipeline/locked_strategy.py`

```python
@dataclass(frozen=True)
class LockedStrategy:
    features: list[str]
    xgb_params: dict
    tickers: list[str]
    forward_horizon_hours: float = 2.5
    entry_times: tuple = ("12:00", "12:30", "13:00", "13:30")
    purge_bars: int       # placeholder, a fijar por usuario
    embargo_bars: int     # placeholder, a fijar por usuario

    def build_model(self): ...  # devuelve XGBClassifier(**xgb_params)
```

### Tests

- `test_locked_strategy_is_frozen`: intentar mutar un campo tira error.
- `test_build_model_returns_xgb`: tipo correcto, params correctos.

---

## Paso 2 — Fase 2: Walk-Forward Lock-down

**Objetivo:** 3 folds (train 8m + test 4m), modelo re-entrenado con params LOCKED.

### Archivo: `validation_pipeline/fase2_wf.py`

**Firma pública:**
```python
def run_fase2(strategy: LockedStrategy,
              X, y, t1, fwd_ret,
              folds_config: list[dict]) -> Fase2Result
```

`folds_config` es explícito (no autocalculado) — 3 dicts con `{train_start, train_end, test_start, test_end}`. Esto hace el setup auditable.

**Implementación:**
1. Iterar los 3 folds.
2. Por fold: aplicar purge+embargo (usar helpers de `splitters.py`) entre train y test.
3. Fit XGB con `strategy.build_model()` sobre train.
4. Predict sobre test, calcular PnL con `_pnl_from_split` (reusar de `backtest_engine.py`).
5. Devolver `Fase2Result` con: `sharpes: pd.Series`, `pnls_by_fold: dict[int, pd.Series]`, `preds_by_fold`, `dates_by_fold`.

### Tests

- `test_fase2_3_folds_no_overlap`: los 3 test sets no se solapan.
- `test_fase2_train_test_purged`: no hay barras compartidas entre train y test (respetando purge+embargo).
- `test_fase2_sharpes_finite`: 3 sharpes no-NaN.
- Test de regresión: con data sintética y seed fijo, Sharpe esperado ≈ valor conocido.

---

## Paso 3 — Fase 3.1: Permutation Test

**Objetivo:** p-value del Sharpe observado vs. distribución nula de trades shuffleados.

### Archivo: `fase3_permutation.py`

```python
def permutation_test(pnl_series: pd.Series,
                     n_permutations: int = 1000,
                     seed: int = 42) -> PermutationResult
# returns: observed_sharpe, null_dist (np.array), p_value
```

- Shuffle sólo del **orden** de los valores del PnL (no los valores en sí).
- Recalcular Sharpe en cada permutación.
- p-value one-sided: `mean(null >= observed)`.

### Tests

- `test_permutation_preserves_values`: shuffleado tiene mismos valores.
- `test_permutation_ruido_puro`: data gaussiana iid → p ≈ 0.5 (test estadístico flaky-tolerant).
- `test_permutation_señal_fuerte`: data con trend → p < 0.05.

---

## Paso 4 — Fase 3.2: Slippage & Latency

### Archivo: `fase3_slippage.py`

```python
def slippage_grid(pnls_by_fold: dict[int, pd.Series],
                  costs_bps: list[float] = [0, 1, 2, 5, 10, 20]
                  ) -> pd.DataFrame   # index=cost, columns=fold_id, values=sharpe

def latency_grid(strategy, X, y, t1, fwd_ret, folds_config,
                 lags_bars: list[int] = [0, 1, 2]
                 ) -> pd.DataFrame   # index=lag, columns=fold_id
```

- Slippage: `pnl_adjusted = pnl - cost_bps * 2 * 1e-4` (round-trip aplicado a cada trade con turnover≥1).
- Latencia: desplazar la predicción K barras hacia adelante → re-calcular PnL.
- Break-even cost: interpolar costo donde Sharpe cruza 0.

### Tests

- `test_slippage_monotonic_decreasing`: más costo → menor Sharpe.
- `test_slippage_zero_cost_matches_base`: costo=0 devuelve el Sharpe de Fase 2.
- `test_latency_zero_matches_base`.

---

## Paso 5 — Fase 3.3: Hyperparameter Sensitivity

### Archivo: `fase3_sensitivity.py`

```python
def hyperparam_grid(strategy: LockedStrategy,
                    X, y, t1, fwd_ret, folds_config,
                    grid: dict = None) -> pd.DataFrame
# default grid: n_estimators∈{50,100,200}, max_depth∈{2,3,4}, lr∈{0.005,0.01,0.02}
# returns: flat DataFrame con 27 filas × 3 folds = 81 Sharpes
```

- Por cada combo: clonar strategy con nuevos params, correr `run_fase2`, extraer sharpes.
- Output: DataFrame con columnas `(n_estimators, max_depth, learning_rate, fold_id, sharpe)`.

### Tests

- `test_grid_size_27_combos`: 27 combos × 3 folds = 81 rows.
- `test_locked_point_in_grid`: el punto LOCKED aparece, con Sharpe == Fase 2.

---

## Paso 6 — Fase 3.4: Noise Injection

### Archivo: `fase3_noise.py`

```python
def noise_injection(strategy, X_raw_ohlcv, y, t1, fwd_ret, folds_config,
                    sigma_bps_list: list[float] = [1, 5, 10],
                    seeds: list[int] = [0, 1, 2, 3, 4]
                    ) -> pd.DataFrame
# returns: index=sigma, columns=fold_id, values=Sharpe promediado sobre seeds
```

- Requiere acceso al OHLCV crudo (no X featurizado) para re-calcular features con precios ruidosos.
- Multiplicar OHLC de test por `(1 + ε)`, `ε ~ N(0, σ)`.
- Volver a featurizar (función de featurización de `data.py`), predecir con modelo ya entrenado (train NO se altera).
- Promediar Sharpe sobre 5 seeds.

### Dependencia crítica
Necesita que la featurización sea una función pura `ohlcv -> features`. Si no existe separada hoy, extraerla de `data.py` en Paso 0.

### Tests

- `test_noise_zero_matches_base`: σ=0 devuelve Sharpe de Fase 2.
- `test_noise_monotonic_dispersion`: std de predicciones crece con σ.

---

## Paso 7 — Reference Distribution

### Archivo: `reference_distribution.py`

```python
@dataclass
class ReferenceDistribution:
    mu_ref: float       # mean of 3 WF sharpes
    sigma_ref: float    # combined std
    components: dict    # breakdown: {'wf_std', 'permutation_std', 'noise_std'}

def build_reference(fase2_result, permutation_result_by_fold, noise_result) -> ReferenceDistribution
```

- `σ_ref = sqrt(σ_wf² + σ_null² + σ_noise²)` (suma cuadrática, asume independencia).
- Documentar esta elección en docstring; es una decisión metodológica, no un cálculo único.

### Tests

- `test_sigma_ref_positive`.
- `test_components_sum_correct`.

---

## Paso 8 — Fase 4: Paper Trading + Monitor

### Archivo: `fase4_paper.py`

```python
def run_fase4(strategy: LockedStrategy,
              X, y, t1, fwd_ret,
              live_folds_config: list[dict],
              reference: ReferenceDistribution
              ) -> Fase4Result
```

**2 folds live:**
- Live-1: train 2024-09→2025-05, paper 2025-05→2025-10.
- Live-2: train 2025-03→2025-11, paper 2025-11→2026-04.

**Por fold live:**
1. Fit modelo con params LOCKED sobre train.
2. "Paper trade": predecir barra-a-barra en el rango live.
3. Calcular Sharpe acumulado mes-a-mes.
4. Z-score mensual: `Z = (Sharpe_live_mtd - mu_ref) / sigma_ref`.
5. Flag: `{'green', 'warning', 'kill'}` según umbrales.

### Archivo: `monitor.py`

- `compute_feature_drift(X_train, X_live) -> dict` con KL por feature.
- `compute_secondary_kpis(pnl_live, pnl_fase2) -> dict` (hit rate, turnover, max DD).

### Tests

- `test_z_score_green_warning_kill`: series sintéticas con Sharpe cae → flag transiciona correctamente.
- `test_feature_drift_identical_is_zero`: KL(X, X) == 0.

---

## Paso 9 — Notebook integrador

**Archivo:** `notebooks/validation_pipeline_e2e.ipynb` (5 celdas máx):

1. Load data + LockedStrategy mock (simula output de Fase 1).
2. Run Fase 2 → imprime tabla de 3 Sharpes + plot de curva degradación.
3. Run Fase 3 (4 bloques en orden) → plots de p-values, curva slippage, heatmap sensitivity, curva noise.
4. Build ReferenceDistribution → imprime μ, σ, componentes.
5. Run Fase 4 → tabla Z-score mensual + flags + KPIs secundarios.

---

## Orden de ejecución y dependencias

```
Paso 0 (data) ──┬──> Paso 1 (LockedStrategy)
                │
                └──> Paso 2 (Fase 2) ──┬──> Paso 3 (permutation)
                                        ├──> Paso 4 (slippage)
                                        ├──> Paso 5 (sensitivity)
                                        ├──> Paso 6 (noise)
                                        │
                                        └──> Paso 7 (reference) ──> Paso 8 (Fase 4) ──> Paso 9 (notebook)
```

Los pasos 3-6 son independientes entre sí una vez terminado Paso 2: podés paralelizar su implementación si querés.

---

## Criterios de aceptación globales

1. Todos los tests pasan con `conda run -n rappi pytest tests/validation_pipeline/`.
2. El notebook `validation_pipeline_e2e.ipynb` corre end-to-end sin errores con 1 ticker (SPY) sobre data real.
3. Ningún módulo fuera de `fase3_sensitivity.py` modifica hiperparámetros.
4. `LockedStrategy` es frozen → sobreescribir campos tira error en runtime.
5. Purge y embargo se aplican en todos los splits (Fase 2 y Fase 4), con valores tomados del campo de `LockedStrategy`, no hardcodeados.

---

## Decisiones pendientes (confirmar antes de Paso 0)

1. `purge_bars` y `embargo_bars` concretos (ej: purge=5, embargo=10 con barras 1h).
2. ¿Timeframe real de los datos: 1h o 30min? El spec dice "OHLCV 1h" pero el label de 2.5h sugiere 30min. Verificar en `data.py`.
3. ¿Featurización ya es función pura? Si no, Paso 0 incluye refactor.
4. ¿1 ticker (SPY) para el primer end-to-end, extensión a 20 tickers en un segundo plan posterior? (Recomendado sí.)
