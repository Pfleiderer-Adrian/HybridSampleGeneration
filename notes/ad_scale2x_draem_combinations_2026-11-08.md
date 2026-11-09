# V3: Echt + AD + DRAEM und Echt + AD + DRAEM + Hybrid

Gestartet am 08.11.2026 für alle acht Kategorien (16 frische Downstream-Läufe).

Ergebnisse: `/mnt/results/mvtec2/experiments/anomalydiffusion_scale2x_50k/20261103/real_ad_draem_combinations_20261108`.
Aktueller Startzeiger: `/mnt/results/mvtec2/experiments/anomalydiffusion_scale2x_50k/20261103/latest_ad_draem_downstream_launch.json`.

## Mischungen und Durchführung

GPU 0: `real_anomalydiffusion_draem`, 500 gesund + 100 echte Defekte + 200 AD + 200 DRAEM pro Epoche.
GPU 1: `real_anomalydiffusion_draem_hybrid`, 500 gesund + 100 echte Defekte + 400 synthetische Beispiele, gleichmäßig aus AD/DRAEM/Hybrid.
Die bestehende Quotenrotation verteilt 134/133/133 zyklisch: Epoche 1 zusätzliche AD-Ziehung, Epoche 2 DRAEM, Epoche 3 Hybrid; danach Wiederholung.
Über jeweils drei Epochen erhält jede Quelle exakt 400 Ziehungen. Über 100 Epochen: AD 13.334, DRAEM 13.333, Hybrid 13.333.

Beide Worker verarbeiten seriell: can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts.
Jede Kategorie: Seed42, 100 Epochen, 1000 Ziehungen/Epoche, Batch8, Adam1e-4, Milestones80/90, Gamma0.2, native256.
Unverändertes eingefrorenes V3 DINOv3-ViT-S/16/CNN-Modell, Losses, gruppierte Splits, Validierung alle5 Epochen und Best-Auswahl nach öffentlichem Pixel-AUPRO@0,05.
Es wird frisch trainiert; keine alten Downstream-Gewichte werden fortgesetzt.

## Daten und Nachweise

Dieselben 1000 AD-Paare pro Kategorie aus 2x / Last50k / AAR wie bei Real+AD und Real+AD+Hybrid; symlink-basierte Wiederverwendung, keine neue Generation.
Alle 24.000 AD-Dateien wurden gegen die Produktions-SHA256 geprüft, zusammen mit Auflösung, nichtleeren binären Masken und Train-only-Provenienz.
Die Hybrid-Variante verwendet exakt die eingefrorenen V3-Hybrid-IDs und deren Reihenfolge. Alle16 Konfigurationen entsprechen der Referenz nach Ausnehmen ausschließlich von Quoten, Rotation, ignoriertem Legacy-Mischungsanteil und AD-Cachepfad.
DRAEM-Synthese, Sampler, Architektur, Loss, Runner und Auswertung bleiben bytegleich zum bisherigen eingefrorenen Runtime; derselbe DTD-Pool mit5640 Bildern wird verwendet.

Die neue Runtime ist in `source_snapshot/` eingefroren und durch `snapshot_sha256.json` dokumentiert. Funktionale Erweiterung: AD ist als Quelle in der bereits vorhandenen Quotenrotation zugelassen. Zusätzlich werden AD-Ziehungen in neuen Trainings-CSV-Dateien mitgezählt; historische Resume-Header bleiben erhalten.
Implementierung: `experiments/anomalydiffusion/scale2x_combinations.py`, `mixtures.py` und der wiederverwendete `scale2x_mixed.py`-Worker.
Tests: 14 Prüfungen zu Mischungen, Quoten, echten/gesunden Ziehungen, gepaarten Losses, Resume und Checkpoint-Auswahl bestanden; `preflight.json` dokumentiert alle16 Daten-/Konfigurationsprüfungen.

## Status und Ausgaben

`gpu0_worker/status.json` und `gpu1_worker/status.json`: laufende Kategorie, Gesamtzustand und gescheiterte Kategorien.
`gpuN_worker/jobs/<category>/status.json` und `downstream.log`: konkrete Prozesse, Fortschritt und Abschlussartefakte.
`downstream/<variant>/<category>/downstream/dinov3_paired/<run>/`: Konfiguration, tatsächliche Hybrid-Provenienz, Quellenzahlen jeder Epoche, Last/Best, History, Validierungsvorhersagen und metrics.json.
`inputs/<variant>/<category>/epoch_quotas.json`: geplante effektive Quellenquoten für alle100 Epochen.

Jeder Worker hält seinen Queue-Lock und den passenden abgeschlossenen Real+AD-Vorgänger-Lock. Vorhandene Downstream-Ausgaben werden nicht überschrieben; bei Kategoriefehlern werden die übrigen Kategorien weiterbearbeitet. Keine automatischen Wiederholungen über Teilresultate.
NVML-Sidecar pro GPU zeichnet Energie und Laufzeit auf; ausschließlich GPU-Geräteenergie. Frühere AD-Generationskosten werden nicht erneut verrechnet.
Dieser Auftrag startet Training und öffentliche Validierung; Abgabearchive sind hier nicht eingeplant.

Abschluss geprüft: 09.11.2026. Beide Queues: complete, keine fehlgeschlagenen Kategorien; alle16 History-/Quoten-/Metrikprüfungen bestanden. Finaler Vergleich: /mnt/results/mvtec2/reports/ad_draem_comparison_2026-11-09/README.md.
