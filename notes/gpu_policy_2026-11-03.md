# GPU-Präferenz des Nutzers

Ab 03.11.2026: neue Trainings ausschließlich auf physischer GPU 1.
Die aktuelle AD-V3-Fortsetzung startet auch alle übrigen neuen GPU-Phasen
auf GPU 1. Bereits laufende rice-Generierung auf GPU 0 darf fertiglaufen;
das laufende sheet_metal-Training auf GPU 1 wurde unverändert übernommen.

Aktueller Controller und Plan über
`/mnt/results/mvtec2/experiments/anomalydiffusion_v3/latest_launch.json`.
Die Fortsetzung läuft mit `experiments.anomalydiffusion.resume_queue`.
Keine Modell-, Daten- oder Splitänderung durch den GPU-Wechsel.

Später am 03.11.2026 durch ausdrücklichen Nutzerauftrag abgelöst:
frische2×-Appearance-Trainings für alle acht Kategorien, je50k, Zwischenstände
alle10k plus Best; Generation aus Last mit AAR. Nur GPU1.
Aktuelle Queue:
`/mnt/results/mvtec2/experiments/anomalydiffusion_scale2x_50k/latest_launch.json`.
Die vorherige V3-Queue und ihre alten Aufgaben wurden dafür beendet; alte
Ergebnisse bleiben unverändert gespeichert.
