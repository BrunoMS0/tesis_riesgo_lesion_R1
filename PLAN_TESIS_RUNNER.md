
---

## PLAN ESTRUCTURADO

```markdown
# PLAN_TESIS_RUNNER.md
# Instrucción IA: Leer este archivo al inicio de cada sesión.
# Marcar [x] al completar cada tarea. No avanzar de fase sin completar la anterior.

---

## FASE 1 — ETL Runner Dataset
> Convertir day_approach_maskedID_timeseries.csv al formato del pipeline

- [x] T1.1 Explorar estructura: shape, nulls, tipos, sufijos .0-.6
- [x] T1.2 Decidir formato wide vs long para el RF (recomendado: wide directo)
- [x] T1.3 Crear src/runner/config.py con paths y parámetros
- [x] T1.4 Crear src/runner/extract.py — carga del CSV
- [x] T1.5 Crear src/runner/transform.py — calcular ACWR desde total km
- [x] T1.6 Crear src/runner/dataset.py — split por atleta sin data leakage
- [x] T1.7 Validar: 0 leakage entre atletas, lesiones en todos los sets

---

## FASE 2 — Feature Engineering Runner Dataset
> Construir features derivadas equivalentes a las de PMData

- [x] T2.1 Calcular ACWR desde total km (7d / 28d)
- [x] T2.2 Calcular wellness_score = media(exertion_inv, recovery, trainingSuccess)
- [x] T2.3 Calcular acute_load_7d y chronic_load_28d
- [x] T2.4 Calcular session_load_proxy = total km × perceived exertion
- [x] T2.5 Documentar features finales y compatibilidad con PMData
- [x] T2.6 Exportar a src/outputs/runner_dataset_processed.csv

---

## FASE 3 — Entrenamiento y LOAO en Runner Dataset
> Meta: LOAO AUC >= 0.65 con 74 atletas (63 con lesiones)

- [x] T3.1 Adaptar src/injury/dataset.py para aceptar source="runner"
- [x] T3.2 Entrenar RF en runner dataset — verificar val AUC > 0.70 ✅ (0.9467)
- [x] T3.3 Ejecutar LOAO completo (74 atletas) ✅ 63/74 folds válidos
- [x] T3.4 Analizar resultados por atleta ✅ AUC=0.9101±0.0891
- [x] T3.5 Si AUC < 0.65: probar target injury_next14d — NO requerido (AUC=0.9101)
- [x] T3.6 Guardar modelo: src/outputs/rf_runner_model.pkl ✅

---

## FASE 4 — Validación Cross-Domain en PMData (Fitbit)
> Probar que el modelo generaliza a datos reales de Fitbit

- [x] T4.1 Definir features comunes runner ∩ PMData (6 features)
- [x] T4.2 Retrain RF-Common usando solo features compartidas ✅
- [x] T4.3 Evaluar RF-Common sobre PMData completo ✅
- [x] T4.4 Evaluar RF-Common con LOAO sobre PMData ✅ AUC=0.5368 (meta 0.55 no cumplida por 0.013)
- [x] T4.5 Documentar comparación baseline vs cross-domain ✅ Tablas 6.12 en capitulo6

---

## FASE 5 — Actualización de Capítulos de la Tesis
> Integrar nuevos resultados sin reescribir completamente

- [x] T5.1 Actualizar capitulo4_O1.md — agregar sección runner dataset ✅ (Sección 4.4)
- [x] T5.2 Actualizar capitulo5_O2.md — R5 ahora entrena en runner ✅ (nota RF-Runner en 5.2.2)
- [x] T5.3 Actualizar capitulo6_O3.md — R8 reporta LOAO runner >= 0.65 ✅
- [x] T5.4 Verificar que indicador R8 se cumple (AUC > 0.70 en val interna) ✅ val=0.9467

---

## FASE 6 — Verificación Final
> Confirmar todos los indicadores verificables

- [x] T6.1 Ejecutar pipeline end-to-end con source=runner ✅ exit_code=0, AUC=0.9101 reproducido
- [x] T6.2 Generar tabla comparativa final de todos los modelos ✅ src/outputs/final_model_comparison.csv
- [x] T6.3 Verificar indicadores: R7 ✅ | R8 ✅ (Runner LOAO=0.9101) | R9 ✅
- [x] T6.4 Actualizar este archivo con resultados finales ✅
- [ ] T6.5 Revisión con asesor

---

## ESTADO ACTUAL
| Fase | Estado |
|---|---|
| Fase 1 — ETL Runner | ✅ Completada |
| Fase 2 — Feature Engineering | ✅ Completada |
| Fase 3 — LOAO Runner | ✅ Completada — LOAO AUC = 0.9101 ≥ 0.65 |
| Fase 4 — Cross-domain PMData | ✅ Completada — AUC = 0.5368 (cerca de 0.55) |
| Fase 5 — Actualizar tesis | ✅ Completada — cap4 (sec. 4.4), cap5 (nota RF-Runner), cap6 (6.2.2.1, 6.2.3) |
| Fase 6 — Verificación final | ✅ Completada — pipeline reproducido, tabla final generada, indicadores verificados |

## NOTAS PARA LA IA
- Runner dataset: C:\Users\brunoabc\Downloads\doi-10.34894-uwu9pv\day_approach_maskedID_timeseries.csv
- Reutilizar src/ al máximo — no reescribir lo que ya funciona
- Modelo de fatiga LSTM (R7) NO cambia — ya está validado
- Si LOAO runner < 0.60 en Fase 3: escalar a injury_next14d antes de reportar
```

---
