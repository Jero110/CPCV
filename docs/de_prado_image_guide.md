# De Prado Image Guide

Guia rapida de las imagenes en `fotos/`, organizada para contrastar la implementacion y los plots del proyecto con el enfoque de Marcos Lopez de Prado.

## Agent Index

Tabla compacta para lectura rapida por agentes.

| File | Theme | Use | Priority |
|---|---|---|---|
| `fotos/Puring_snippet.jpeg` | Purging snippet | Validar implementacion de purge | alta |
| `fotos/embargo_snippet.jpeg` | Embargo snippet | Validar implementacion de embargo | alta |
| `fotos/embargo.jpeg` | Embargo concept | Validar exclusion temporal | alta |
| `fotos/CVScore.jpeg` | cvScore | Revisar logica base de scoring | alta |
| `fotos/CPCV.jpeg` | CPCV definition | Revisar grupos, splits, paths, phi | alta |
| `fotos/CPCV_splits.jpeg` | CPCV split layout | Validar split matrix y paths | alta |
| `fotos/PitfallsOfWF.jpeg` | Walk-forward pitfalls | Justificar limites de WF | alta |
| `fotos/StrategySelection.jpeg` | Strategy selection | Entender narrativa de seleccion | alta |
| `fotos/StrategySelection2.jpeg` | Strategy selection | Revisar ranking relativo | alta |
| `fotos/StrategySelection3_OOSDeg_Logits.jpeg` | OOS degradation + logits | Referencia principal para scatter y PBO | muy alta |
| `fotos/CPCV_overfit.jpeg` | CPCV overfitting theory | Revisar formula y supuestos PBO | muy alta |
| `fotos/CPCV_overfit2.jpeg` | CPCV overfitting theory | Revisar paths y logits | muy alta |
| `fotos/Splits.jpeg` | Split schemes | Pedagogia general | media |
| `fotos/CV_finance.jpeg` | CV in finance | Contexto comparativo | media |
| `fotos/CV_finance2.jpeg` | CV in finance | Contexto comparativo | media |
| `fotos/PitfallOfWF2.jpeg` | Walk-forward pitfalls | Soporte teorico adicional | media |
| `fotos/BacktestStatistic.jpeg` | Backtest stats | Elegir metricas finales | media |
| `fotos/generalCharacteristics.jpeg` | Financial data properties | Marco teorico | media |
| `fotos/generalCharacteristics2.jpeg` | Financial data properties | Marco teorico | media |
| `fotos/Performance.jpeg` a `fotos/Performance11.jpeg` | Performance metrics | Revisar metricas complementarias | media |

## Agent Workflow

Secuencia sugerida para agentes antes de tocar codigo:

1. Leer `Puring_snippet`, `embargo_snippet`, `CVScore`, `CPCV`, `CPCV_splits`.
2. Leer `PitfallsOfWF` para fijar por que WF no debe dominar la narrativa.
3. Leer `StrategySelection3_OOSDeg_Logits`, `CPCV_overfit`, `CPCV_overfit2`.
4. Auditar `cpcv_analysis/splitters.py`, `cpcv_analysis/cv_runner.py`, `cpcv_analysis/advanced_analysis.py`.
5. Solo despues ajustar `plots.py`.

## Agent Output Expectations

Si un agente usa este doc para mejorar el proyecto, deberia responder al menos estas preguntas:

1. El purge y el embargo son fieles al autor.
2. Los paths CPCV estan construidos y agregados como en De Prado.
3. La degradacion OOS esta calculada al nivel correcto.
4. El histograma de rank logits y `Prob[Overfit]` salen de rankings por trial/path y no de una aproximacion incorrecta.
5. Las figuras finales cuentan la historia del libro, no solo muestran metricas disponibles.

## Como usar esta guia

- `Referencia conceptual`: imagenes para entender la idea del autor.
- `Referencia de implementacion`: snippets o esquemas que conviene replicar con fidelidad.
- `Referencia de validacion`: figuras contra las que conviene comparar nuestros outputs.
- `Prioridad`: `alta`, `media`, `baja` para el trabajo actual.

## 1. Fundamentos generales

### [generalCharacteristics.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/generalCharacteristics.jpeg)
- Tema: caracteristicas generales de series financieras / problema de validacion.
- Uso: marco teorico.
- Tipo: referencia conceptual.
- Prioridad: media.

### [generalCharacteristics2.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/generalCharacteristics2.jpeg)
- Tema: continuacion del bloque anterior.
- Uso: justificar por que CV clasico falla en finanzas.
- Tipo: referencia conceptual.
- Prioridad: media.

### [BacktestStatistic.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/BacktestStatistic.jpeg)
- Tema: estadisticos de backtest y como leerlos.
- Uso: elegir metricas para el analisis final.
- Tipo: referencia conceptual.
- Prioridad: media.

## 2. Purging y embargo

### [Puring_snippet.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/Puring_snippet.jpeg)
- Tema: snippet del autor para purging.
- Uso: revisar si `getTrainTimes` y `PurgedKFold` son fieles.
- Tipo: referencia de implementacion.
- Prioridad: alta.

### [embargo.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/embargo.jpeg)
- Tema: explicacion visual del embargo.
- Uso: validar que la logica de exclusion temporal sea correcta.
- Tipo: referencia conceptual.
- Prioridad: alta.

### [embargo_snippet.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/embargo_snippet.jpeg)
- Tema: snippet del autor para embargo.
- Uso: contrastar con `getEmbargoTimes`.
- Tipo: referencia de implementacion.
- Prioridad: alta.

## 3. Splits, CV y CPCV

### [Splits.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/Splits.jpeg)
- Tema: formas de partir la muestra.
- Uso: validar nomenclatura y pedagogia del proyecto.
- Tipo: referencia conceptual.
- Prioridad: media.

### [CV_finance.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/CV_finance.jpeg)
- Tema: CV adaptado a finanzas.
- Uso: contexto para KFold, PurgedKFold y variantes.
- Tipo: referencia conceptual.
- Prioridad: media.

### [CV_finance2.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/CV_finance2.jpeg)
- Tema: continuacion del bloque de CV en finanzas.
- Uso: apoyo teorico para la comparacion entre metodos.
- Tipo: referencia conceptual.
- Prioridad: media.

### [CVScore.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/CVScore.jpeg)
- Tema: `cvScore` original / metrica de evaluacion del autor.
- Uso: revisar si la extension actual respeta la logica base.
- Tipo: referencia de implementacion.
- Prioridad: alta.

### [CPCV.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/CPCV.jpeg)
- Tema: descripcion formal de CPCV.
- Uso: revisar definicion de grupos, splits, paths y `phi`.
- Tipo: referencia conceptual.
- Prioridad: alta.

### [CPCV_splits.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/CPCV_splits.jpeg)
- Tema: matriz / esquema de splits CPCV.
- Uso: validar `split_matrix`, indexacion de grupos y asignacion a paths.
- Tipo: referencia de validacion.
- Prioridad: alta.

## 4. Walk-forward y sus pitfalls

### [PitfallsOfWF.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/PitfallsOfWF.jpeg)
- Tema: limitaciones de walk-forward.
- Uso: justificar por que no basta con WF.
- Tipo: referencia conceptual.
- Prioridad: alta.

### [PitfallOfWF2.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/PitfallOfWF2.jpeg)
- Tema: continuacion del argumento sobre WF.
- Uso: reforzar seccion comparativa contra CPCV.
- Tipo: referencia conceptual.
- Prioridad: media.

## 5. Performance y evaluacion

### [Performance.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/Performance.jpeg)
### [Performance2.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/Performance2.jpeg)
### [Performance3.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/Performance3.jpeg)
### [Performance4.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/Performance4.jpeg)
### [Performance5.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/Performance5.jpeg)
### [Performance6.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/Performance6.jpeg)
### [Performance7.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/Performance7.jpeg)
### [Performance8.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/Performance8.jpeg)
### [Performance9.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/Performance9.jpeg)
### [Performance10.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/Performance10.jpeg)
### [Performance11.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/Performance11.jpeg)
- Tema: bloque de metricas y lectura de performance.
- Uso: decidir que metricas tienen sentido reportar fuerte y cuales solo como soporte.
- Tipo: referencia conceptual.
- Prioridad: media.
- Nota: este bloque probablemente nos ayuda a decidir si dejar `return`, `Sharpe`, `accuracy`, `F1`, `drawdown`, `SR degradation`, etc., pero no es la referencia principal para PBO.

## 6. Strategy selection, OOS degradation y PBO

### [StrategySelection.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/StrategySelection.jpeg)
- Tema: seleccion de estrategia.
- Uso: entender como pasar de comparacion de metodos a seleccion basada en ranking.
- Tipo: referencia conceptual.
- Prioridad: alta.

### [StrategySelection2.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/StrategySelection2.jpeg)
- Tema: continuacion del bloque de strategy selection.
- Uso: revisar definicion de ranking relativo y su interpretacion.
- Tipo: referencia conceptual.
- Prioridad: alta.

### [StrategySelection3_OOSDeg_Logits.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/StrategySelection3_OOSDeg_Logits.jpeg)
- Tema: figura clave con dos paneles.
- Uso: referencia principal para:
  - scatter `IS vs OOS` con recta de degradacion
  - histograma de rank logits
  - `Prob[Overfit]`
- Tipo: referencia de validacion.
- Prioridad: muy alta.
- Lo importante aqui no es solo el estilo visual: tambien fija la logica del analisis de degradacion y PBO.

### [CPCV_overfit.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/CPCV_overfit.jpeg)
- Tema: seccion teorica de como CPCV aborda overfitting.
- Uso: revisar la formula y los supuestos del PBO.
- Tipo: referencia conceptual.
- Prioridad: muy alta.

### [CPCV_overfit2.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/CPCV_overfit2.jpeg)
- Tema: continuacion de la teoria de overfitting / paths CPCV.
- Uso: validar el paso de Sharpe por path a distribucion de logits.
- Tipo: referencia conceptual.
- Prioridad: muy alta.

## 7. Checklist practico para la implementacion

Estas son las imagenes que conviene mirar primero al iterar codigo y graficas:

1. [Puring_snippet.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/Puring_snippet.jpeg)
2. [embargo_snippet.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/embargo_snippet.jpeg)
3. [CVScore.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/CVScore.jpeg)
4. [CPCV.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/CPCV.jpeg)
5. [CPCV_splits.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/CPCV_splits.jpeg)
6. [PitfallsOfWF.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/PitfallsOfWF.jpeg)
7. [StrategySelection3_OOSDeg_Logits.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/StrategySelection3_OOSDeg_Logits.jpeg)
8. [CPCV_overfit.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/CPCV_overfit.jpeg)
9. [CPCV_overfit2.jpeg](/Users/jeronimo.deli/Desktop/other/Vs/TSP/CPCV_tesis/fotos/CPCV_overfit2.jpeg)

## 8. Implicaciones directas para nuestro proyecto

- `split_matrix`: debe parecerse a la logica de `CPCV_splits`, con indexacion consistente y paths verificables.
- `IS/OOS degradation`: debe seguir la narrativa de `StrategySelection3_OOSDeg_Logits`, no ser solo un scatter decorativo.
- `rank logits / PBO`: debe venir de rankings por trial/path y no de una simplificacion que pierda el sentido del autor.
- `walk-forward`: debe aparecer mas como baseline con limitaciones, no como competidor equivalente en narrativa.
- `return by split/path`: probablemente no debe ser figura principal si no ayuda a demostrar la idea central del libro.

## 9. Siguiente paso sugerido

Con esta guia como mapa, el siguiente trabajo deberia ser:

1. Auditar calculos de `cvScore`, paths, `oos_degradation` y `rank_logits`.
2. Redefinir el set final de figuras para que siga mas de cerca la logica de De Prado.
3. Solo despues pulir styling y labels.
