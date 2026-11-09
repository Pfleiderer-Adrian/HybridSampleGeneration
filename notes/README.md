# Arbeitsnotizen und externe Ergebnisse

Hier bleiben die aktuellen Ablauf-, GPU- und Abgabeprotokolle. Bilder, Messwerte,
Audit-Berichte und historische Gesprächssicherungen liegen außerhalb des Checkouts:

| Inhalt | Ablage |
|---|---|
| Berichte, Diagramme, Vergleiche und Abgabeübersichten | `/mnt/results/mvtec2/reports/` |
| Audit-Ausgaben mit ihren bisherigen Unterordnernamen | `/mnt/results/mvtec2/audits/` |
| Historische Gesprächssicherungen | `/mnt/results/mvtec2/notes_archive/` |
| Analyseskripte | `scripts/analysis/audits/` und `scripts/analysis/reports/` im Repository |
| Paper-Quellen | `docs/paper/` im Repository |

Die bisherigen Audit-Unterordner und Gesprächssicherungen unter `notes/` sind
lokale Kompatibilitätslinks. Auch `reports/` ist ein Link auf die externe Ablage.
Dadurch bleiben alte Verweise erreichbar; die Links werden nicht versioniert.
Neue Analysen verwenden die Pfade aus `scripts/analysis/paths.py`.

Der Umzug wurde mit SHA256-Prüfung jeder übertragenen Datei protokolliert.
Manifest und Originalfassungen der angepassten Quellen/Texte liegen unter
`/mnt/results/mvtec2/cleanup/`; `latest_project_layout.json` zeigt auf den Lauf.
