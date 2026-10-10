# Analyse: Comparison V1 gegenüber V2

Vergleich abgeschlossener Testläufe; Werte in Prozent. Δ = Hybrid − Baseline in Prozentpunkten.

| Kategorie | Metrik | V1 Baseline | V1 Hybrid | V1 Δ | V2 Baseline | V2 Hybrid | V2 Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| fabric | Bild-AUROC | 75.412 | 84.671 | +9.259 | 98.354 | 98.868 | +0.514 |
| fabric | Bild-AP | 60.715 | 80.016 | +19.301 | 95.651 | 96.656 | +1.005 |
| fabric | Pixel-AUROC | 98.355 | 95.936 | -2.419 | 96.487 | 93.904 | -2.582 |
| fabric | Pixel-AP | 81.533 | 81.582 | +0.049 | 74.106 | 46.064 | -28.042 |
| fruit_jelly | Bild-AUROC | 81.250 | 83.796 | +2.546 | 86.111 | 81.481 | -4.630 |
| fruit_jelly | Bild-AP | 74.960 | 81.723 | +6.763 | 78.773 | 64.080 | -14.693 |
| fruit_jelly | Pixel-AUROC | 96.446 | 97.994 | +1.548 | 95.928 | 94.170 | -1.758 |
| fruit_jelly | Pixel-AP | 19.657 | 34.372 | +14.715 | 16.579 | 6.417 | -10.162 |
| sheet_metal | Bild-AUROC | 77.263 | 81.173 | +3.909 | 80.967 | 73.045 | -7.922 |
| sheet_metal | Bild-AP | 60.957 | 65.112 | +4.155 | 64.697 | 40.321 | -24.377 |
| sheet_metal | Pixel-AUROC | 92.656 | 89.796 | -2.860 | 88.872 | 83.119 | -5.753 |
| sheet_metal | Pixel-AP | 27.183 | 44.877 | +17.693 | 33.155 | 1.123 | -32.031 |
| rice | Bild-AUROC | 56.173 | 52.058 | -4.115 | 46.399 | 56.070 | +9.671 |
| rice | Bild-AP | 35.268 | 29.337 | -5.931 | 31.701 | 31.120 | -0.581 |
| rice | Pixel-AUROC | 63.797 | 63.018 | -0.779 | 56.261 | 65.884 | +9.623 |
| rice | Pixel-AP | 0.781 | 0.291 | -0.490 | 0.653 | 2.152 | +1.499 |
| vial | Bild-AUROC | 40.967 | 87.755 | +46.788 | 61.527 | 72.487 | +10.960 |
| vial | Bild-AP | 37.247 | 62.315 | +25.068 | 50.664 | 38.102 | -12.562 |
| vial | Pixel-AUROC | 85.854 | 95.743 | +9.889 | 91.889 | 78.441 | -13.448 |
| vial | Pixel-AP | 1.489 | 10.608 | +9.118 | 3.122 | 0.909 | -2.213 |
| wallplugs | Bild-AUROC | 52.263 | 57.922 | +5.658 |  | 52.366 |  |
| wallplugs | Bild-AP | 26.515 | 33.567 | +7.052 |  | 26.296 |  |
| wallplugs | Pixel-AUROC | 67.959 | 68.259 | +0.300 |  | 43.907 |  |
| wallplugs | Pixel-AP | 0.010 | 0.007 | -0.003 |  | 0.005 |  |

## Belegte Änderungen

- Fabric, fruit_jelly, sheet_metal, rice, vial und wallplugs: vollständige V1/V2-Split-Manifeste identisch. Can ist eine Ausnahme: V1 424/124/72, V2 372/124/124 Bilder für Train/Validation/Test.
- V1 (oben genannte Kategorien): 500 normale, 250 DRAEM- und 250 Hybrid-Samples je Hybrid-Trainingsepoche. V2: 750 normale, 188 DRAEM- und 62 Hybrid-Samples. Quellenzahlen direkt aus training_history.csv für fabric, fruit_jelly und sheet_metal geprüft.
- Patchgröße: 512×512 → 256×256 im Training und in der gekachelten Evaluation. Patch-Modus verarbeitet native Auflösung; image_size=256 bedeutet hier keine gemeinsame Verkleinerung beider Versionen.
- Epochenzahl 100, Batchgröße 8, Lernrate 0.0001, Seed 42 und DRAEM-Breiten 128/64 unverändert.
- Sheet_metal-Generator: paired_cVAE_ConvNeXt_2D → cVAE_ConvNeXt_2D. Matching global → batchwise.
- Fabric und fruit_jelly: Matching fixed_from_extraction_control_fusion → batchwise; Änderungen an Generator, Skip-Verbindungen, Regularisierung, KL-Warmup und Formbehandlung. Fabric zusätzlich 64×64 → 128×128 Anomaliegröße und zwei → eine geplante Anomalie je Hybrid.
- Gespeicherte Generator-Konfigurationen haben teilweise verschiedene Schema-Versionen; fehlende alte Parameter allein belegen keine Verhaltensänderung, da Defaults/Migrationen beteiligt sein können.

## Einordnung

Die reduzierte Hybridquote, der kleinere räumliche Kontext und die Generatoränderungen sind plausible Ursachen, aber nicht einzeln kausal belegt. V1 und V2 ändern mehrere Faktoren gleichzeitig; ein einzelner Seed erlaubt keine Aussage über Streuung.

Wallplugs ist ein besonders nützlicher Hinweis: die gespeicherten Generator-Konfigurationen unterscheiden sich abgesehen vom Studiennamen nur in einem expliziten segmentation_min_alpha=0.0. Trotzdem fällt die Hybrid-Pixel-AUROC von 68.259 auf 43.907. Der Rückgang erfordert daher keinen großen Generatorumbau; Downstream-Protokoll und Stichprobenvariation bleiben wichtige Kandidaten.

Masken-/Bild-Stichprobe: 8 deterministisch ausgewählte gesunde Hybridpaare je Kategorie und Version; keine vollständige Qualitätsbewertung. Fabric-Maskenfläche im Median 0.0081% → 0.0214%, mittlere Bildänderung innerhalb der Maske (4-fach räumlich unterabgetastet, Skala 0..255) 47.68 → 19.78. Sheet_metal: Maskenfläche 0.0198% → 0.0210%, Bildänderung 66.00 → 64.66. Das beweist keine schlechtere Realitätsnähe; für fabric zeigt es veränderte synthetische Defekte, für sheet_metal keinen groben Zusammenbruch der Masken.

Checkpoint-Auswahl minimiert mittlere Validation-Segmentierungs-Focal-Loss, nicht Pixel-AP. Sheet_metal V2 wählt Epoche 13, V1 Epoche 21; V2 Hybrid hat geringere Validation-Loss als Baseline, aber schlechtere Pixel-AP. Die Auswahl kann daher eine andere Rangfolge liefern als die interessierende AP.

## Empfohlene kontrollierte Folgevergleiche

1. V2-Hybridpool behalten, normal_fraction=0.5 und hybrid_fraction=0.5 verwenden; sonst unverändert. Prüft Trainingsexposition.
2. Zusätzlich 512×512 für Training/Evaluation verwenden und jeweils eine entsprechend angepasste Baseline vergleichen. Prüft Kontext/Inference-Protokoll.
3. V1- und V2-Hybridpools mit identischer Downstream-Konfiguration vergleichen. Prüft Generator-/Matching-Effekt.
4. Mehrere Seeds und ausschließlich validationsbasierte Auswahl verwenden; den Testsplit nicht zur Optimierung nutzen.

Bestehende Experimente wurden für diese Analyse nicht verändert.
