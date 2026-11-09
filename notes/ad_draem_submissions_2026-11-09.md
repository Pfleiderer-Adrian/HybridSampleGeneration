# MVTec AD 2: AD+DRAEM-Abgaben, 09.11.2026

Beauftragt: Abgabearchive für Echt+AD+DRAEM und Echt+AD+DRAEM+Hybrid.
Verwendet werden die finalen, geprüften Best-Checkpoints aus real_ad_draem_combinations_20261108.
Öffentliche Validierungs-AUPRO-Mittel:69,8732127597% und70,9705570857%.
Je Variante4090 native float16-TIFF-Anomaliekarten +4090 binäre uint8-PNG-Masken für test_private und test_private_mixed.
Schwellenwahl: maximaler gepoolter Pixel-F1 auf öffentlicher Validierung nach float16-Quantisierung.

Eine GPU je Variante; private Inferenz und CPU-Schwellenkalibrierung laufen parallel.
Alle Modelle und die genaue Trainings-Runtime sind je Bundle durch SHA256 eingefroren.
Nach Inferenz: PNG-Masken, offizielle Formatprüfung, Archivierung, Prüfung auf exakt8180 Dateien, SHA256 und gzip-Integrität.

Batch-Zeiger: /mnt/results/mvtec2/experiments/anomalydiffusion_scale2x_50k/20261103/real_ad_draem_combinations_20261108/private_submission_batch.json

- real_anomalydiffusion_draem: `/mnt/results/mvtec2/submissions/v3_real_anomalydiffusion_draem_scale2x50k_20261109T110306Z`.
- real_anomalydiffusion_draem_hybrid: `/mnt/results/mvtec2/submissions/v3_real_anomalydiffusion_draem_hybrid_scale2x50k_20261109T110306Z`.

Fertig erst nach stage=complete im Bundle und bestandener zusätzlicher gzip-Integritätsprüfung.
Kein Upload.

Abgeschlossen 2026-11-09T11:58:44.761401+00:00. Beide Archive bestehen Format-, Mitglieder- und gzip-Integritätsprüfung; SHA256-Dateien vorhanden. Abschlussbericht: /mnt/results/mvtec2/reports/ad_draem_submissions_2026-11-09.md.
