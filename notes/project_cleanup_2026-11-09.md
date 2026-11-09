# Projektbereinigung vom 09.11.2026

Aktualisierung: Die anschließend beauftragte Git-Bereinigung ist abgeschlossen; siehe [Git-Bereinigung](git_cleanup_2026-11-09.md). Die unten dokumentierte reine Prüfung gehört zum vorherigen Arbeitsschritt.

Die Ergebnisablage ist aus dem Checkout ausgelagert. Bestehende Python-Umgebungen, AD-Abhängigkeiten, Trainingsdaten, Checkpoints und Abgabearchive bleiben an ihren bisherigen Orten.

| Inhalt | Neuer Ort |
|---|---|
| Berichte, Metriken, Abgabeübersichten und Bilder | `/mnt/results/mvtec2/reports/` |
| Audit-Ausgaben und Audit-Dokumentation | `/mnt/results/mvtec2/audits/` |
| Historische Gesprächssicherungen | `/mnt/results/mvtec2/notes_archive/` |
| 36 Audit-Skripte und5 Berichtsgeneratoren | `scripts/analysis/audits/` und `scripts/analysis/reports/` |
| 7 Manuskript-/Bibliografiequellen | `docs/paper/` |

737 vorhandene Dateien sind im Umzugsmanifest erfasst. Rund 80.1 MB Ergebnis-/Archivdateien wurden ausgelagert. Jede übertragene Datei wurde vor Entfernen des Originals am Ziel per SHA256 geprüft. Originalfassungen geänderter Quellen und Texte sind gesichert. JSON-/CSV-Provenienz und Logs bleiben bytegleich.

69 lokale Kompatibilitätslinks erhalten frühere Zugriffe. `reports/` verweist auf die externe Berichtsablage; die bisherigen Audit-Unterordner unter `notes/` verweisen auf die externe Audit-Ablage. Alte Skript- und Paper-Pfade verweisen auf die neuen Repository-Quellen. Diese Links werden per `.gitignore` ausgeschlossen. Die15 zuvor versionierten Dateien unter den alten Audit-Pfaden erscheinen daher als aus dem Checkout entfernte Dateien; ihre Ergebnisdaten sind extern erhalten, ihre3 Python-Skripte liegen unter `scripts/analysis/`. Es wurde nichts gestaged oder committed.

Neue Analysen verwenden `scripts/analysis/paths.py` und optional `HSG_RESULTS_ROOT`. Die Ausgabegrenzen der beiden Diagnoseskripte wurden auf den Results-Ordner umgestellt. Die beiden Pilot-Helfer schreiben/lesen Berichte jetzt direkt aus der externen Ablage; eingefrorene Runtime-Snapshots wurden nicht bearbeitet.

## Prüfungen

- Transferinventar, Dateiprüfsummen, unveränderte Ergebnisdaten und alle69 Kompatibilitätslinks geprüft.
- Alle verschobenen Skripte syntaktisch geprüft.
- Publikationsgenerator aus fremdem Arbeitsverzeichnis mit isoliertem externem Ausgabeordner ausgeführt: PDF/SVG/PNG, CSV und LaTeX-Tabelle erfolgreich erzeugt.
- Diagnose-CLI funktioniert am neuen Ort.
- Trainings-Replay behält explizite eingefrorene PYTHONPATH-Einträge vor dem Live-Checkout.
- MkDocs-Build mit `--strict` erfolgreich; Build-Ausgabe liegt im externen Prüfverzeichnis.

## Separate Git-Prüfung

Nur gelesen: kein gc, prune, repack, Ablaufenlassen von Reflogs, Löschen oder Verschieben von Git-Objekten. Die Git-Metadaten waren nach der Prüfung unverändert. Die Connectivity-Prüfung ist erfolgreich.

| Kandidat | Speicher | Befund |
|---|---:|---|
| 6 temporäre tmp_obj-Dateien | 13.61 GiB | Von Git als garbage erkannt; unverändert erhalten |
| 2 große Archiv-Blobs | 39.69 GiB | Nicht durch aktuelle Refs, Reflogs oder Index referenziert; vollständige Objekt- und Archiv-Hashes geprüft |
| Gesamtpotenzial | 53.29 GiB | Noch nicht bereinigt |

Die externen Originalarchive der beiden Blobs sind:

- `/mnt/results/mvtec2/submissions/dinov3_global_best_wallplugs_updated_20261018T164253Z/submission.tar.gz` — Git-Blob `042e3a94bbd85ad7ed1f8d4a9e9547634baf9cfc`.
- `/mnt/results/mvtec2/submissions/dinov3_small_paired_hybrid_v2_20261014T185304Z/submission.tar.gz` — Git-Blob `5fac0fc4b55001d278985e5e615463ad6bde3ffc`.

Diese Git-Objekte wurden nicht gelöscht: Der Auftrag umfasste hier die separate Prüfung.

## Nachweise

- [Umzugsmanifest](/mnt/results/mvtec2/cleanup/project_layout_20261109T153749Z/manifest.json)
- [Originalquellen und Originaltexte](/mnt/results/mvtec2/cleanup/project_layout_20261109T153749Z/original_sources)
- [Funktionsprüfungen](/mnt/results/mvtec2/cleanup/project_layout_20261109T153749Z/validation/source_layout_checks.json)
- [Git-Prüfung](/mnt/results/mvtec2/cleanup/project_layout_20261109T153749Z/git_storage_audit.json)
- [Git-Connectivity-Protokoll](/mnt/results/mvtec2/cleanup/project_layout_20261109T153749Z/git_connectivity_check.log)
