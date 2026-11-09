# Git-Bereinigung vom 09.11.2026

Die gezielte Bereinigung ist abgeschlossen. Commits, Branches, Tags, Reflogs, Pseudorefs, Index und der vorhandene Arbeitsstand blieben beim Bereinigungsvorgang erhalten. Es wurde kein gc, kein Repack und kein Umschreiben der Historie ausgeführt.

| Messung | Wert |
|---|---:|
| Git vorher | 53.39 GiB |
| Git nachher | 101.99 MiB |
| Freigegeben | 53.29 GiB |

Entfernt wurden ausschließlich sechs alte temporäre Git-Dateien, zwei geprüfte unreferenzierte Archiv-Blobs und elf ebenfalls unreferenzierte Tree-Objekte, die unmittelbar auf diese Blobs zeigten. Die elf komprimierten Tree-Objekte (zusammen7161Bytes) sind extern gesichert. Dadurch bleiben im Object-Store keine Verweise auf entfernte Objekte zurück.

## Prüfungen

- Keine laufende Git-Operation und keine fremden Lock-Dateien vor Beginn.
- Aktuelle Refs, Reflogs, ORIG_HEAD und Index erneut berücksichtigt.
- Sämtliche gespeicherten Tree-/Commit-/Tag-Verweise auf die Zielobjekte geprüft; kein geschütztes Objekt betroffen.
- Beide vollständigen externen Abgabearchive erneut als kanonische Git-Blobs gehasht; die Objekt-IDs stimmen exakt überein.
- Alter, Größe und Änderungszeit der sechs temporären Dateien geprüft.
- Die elf kleinen Metadatenobjekte vor Entfernung kopiert und per SHA256 geprüft.
- Vollständiges `git fsck --full --no-dangling` nach Entfernung bestanden.
- Refs, Reflogs und Index per Prüfsumme sowie Git-Arbeitsstatus vor/nach dem Eingriff identisch.
- Nachher meldet `git count-objects` keine Garbage-Dateien mehr.

## Erhaltene Originalarchive

- `/mnt/results/mvtec2/submissions/dinov3_global_best_wallplugs_updated_20261018T164253Z/submission.tar.gz` — Git-Blob `042e3a94bbd85ad7ed1f8d4a9e9547634baf9cfc`.
- `/mnt/results/mvtec2/submissions/dinov3_small_paired_hybrid_v2_20261014T185304Z/submission.tar.gz` — Git-Blob `5fac0fc4b55001d278985e5e615463ad6bde3ffc`.

## Nachweise und Wiederherstellung

- [Vollständiges Protokoll](/mnt/results/mvtec2/cleanup/git_cleanup_20261109T223148Z/report.json)
- [Gesicherte Tree-Objekte](/mnt/results/mvtec2/cleanup/git_cleanup_20261109T223148Z/metadata_objects)
- [Git-Integritätsprüfung](/mnt/results/mvtec2/cleanup/git_cleanup_20261109T223148Z/fsck_after.log)
- [Objektstatistik](/mnt/results/mvtec2/cleanup/git_cleanup_20261109T223148Z/count_objects_after.txt)
- [Ausgeführter Bereinigungscode](/mnt/results/mvtec2/cleanup/git_cleanup_20261109T223148Z/cleanup_implementation.py)

Bei Bedarf lassen sich die beiden Archiv-Blobs aus ihren erhaltenen Originalarchiven mit `git hash-object -w --no-filters <Archivpfad>` wieder erzeugen. Die ursprünglichen Objekt-IDs sind im Protokoll festgehalten. Danach können die elf gesicherten Tree-Dateien an ihre ursprünglichen, ebenfalls protokollierten Object-Store-Pfade zurückkopiert werden. Diese Wiederherstellung wurde nicht ausgeführt. Die alten temporären Dateien wurden als unbrauchbare Teilobjekte entfernt.
