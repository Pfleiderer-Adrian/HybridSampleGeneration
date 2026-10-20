# Analyse der Trainingshybride: Can, Vial und Wallplugs

Stand: 20.10.2026. Daten nur gelesen; kein Training und keine neue Hybridgenerierung gestartet. Die Analyse verwendet die Hybrid-IDs aus den eingefrorenen Trainingsmanifesten der zweiten privaten Einreichung. Die zugehörigen Arrays stammen aus den aktuellen Study-Artefakten; eine historische Bytegleichheit wird mangels gespeicherter Array-Hashes nicht behauptet.

## Ergebnis

Die deutlichsten sichtbaren Probleme sind **semantisch falsche Platzierungen und einseitige Defektquellen**, nicht großflächige Änderungen außerhalb der Masken. Die Untersuchung identifiziert konkrete Qualitätsfehler, beweist aber noch nicht deren Anteil am privaten Leistungsabfall.

| Kategorie | Trainingshybride | Gesunde Zielbilder | Verwendete Quellbilder | Quell-Aufnahmegruppen | Geprüfte Hybridbilder |
|---|---:|---:|---:|---:|---:|
| Can | 318 | 318 | 21 | 6 | 48 |
| Vial | 693 | 231 | 33 | 8 | 48 |
| Wallplugs | 765 | 255 | 41 | 8 | 47 |

Aufnahmegruppen werden aus den Dateinamen abgeleitet: regular, overexposed, underexposed und shift-Varianten derselben Nummer zählen zusammen. Mehrere Varianten sind keine unabhängigen Defektaufnahmen. Quellbildanzahl meint tatsächlich verwendete Donoren, nicht alle verfügbaren echten Trainingsbilder.

Die Stichprobe umfasst 32 mit Seed42 zufällig ausgewählte Hybride plus je acht niedrigste und höchste Mindest-Platzierungsscores. Überschneidungen werden entfernt. Sie ist zur Fehlersuche gedacht und liefert **keine repräsentative Fehlerquote**. Pro Kategorie werden zwölf Beispiele visuell gezeigt; bei mehreren Platzierungen zeigt die Donorspalte nur den ersten Donor, Hybrid und Maske aber alle Platzierungen. Alle Donornamen stehen in den CSV-Dateien.

## 1. Wallplugs: Platzierung auf Hintergrund und Übertragung von Kontext

[Übersicht 1](wallplugs_overview_1.jpg) · [Übersicht 2](wallplugs_overview_2.jpg)

- Übersicht1 Zeile2: Hybrid `hybrid_65379f22467056fdda29`, Matching-Score ungefähr0,601. Ein Defekt-Ausschnitt landet im dunklen strukturierten Hintergrund statt auf einem Dübel. Der markierte Bereich wirkt teilweise wie übertragener Hintergrund bzw. ein rechteckiger Kontextausschnitt. **Der höchste Mindestscore der gesamten Wallplugs-Auswahl schützt hier nicht vor semantisch falscher Platzierung.**
- Übersicht1 Zeile4 zeigt ebenfalls eine markierte Fläche im Hintergrund am Bildrand.
- Übersicht2 Zeile4 zeigt als Gegenbeispiel einen wesentlich plausibleren Defekt auf dem Dübel. Die Qualität ist nicht durchgehend schlecht.
- Große Maskenbereiche können nach visuellem Eindruck Kontext statt ausschließlich des eigentlichen Defekts enthalten. Das ist eine manuelle Auffälligkeit, keine automatisch verifizierte Falschannotation.

**Folgerung:** Zielregion auf Objekt/Material beschränken; zusätzlich zum Intensitäts-/Gradientenvergleich eine lokale Orientierung und kompatiblen Bauteilbereich berücksichtigen. Bei zufällig liegenden Dübeln reicht Nähe zur ursprünglichen Bildkoordinate nicht aus. Objektgrenzen und Kontakt zum Hintergrund separat kontrollieren.

## 2. Vial: starke Quellenkonzentration und extreme Defektgrößen

[Übersicht 1](vial_overview_1.jpg) · [Übersicht 2](vial_overview_2.jpg)

- **980 von1316 Hybrid–Donorbild-Zuordnungen (74,47 %) stammen aus Aufnahmegruppe `test_public_bad_000`.** Pro Hybrid wird jedes Donorbild nur einmal gezählt; dies sind keine Platzierungszahlen. Insgesamt gibt es1404Platzierungen.
- Häufige kleine Veränderungen sitzen an vorhandenen Blasen. Das kann die erlernte Defektvielfalt stark einschränken. Ob diese Änderungen allein realistisch sind, muss anhand der jeweiligen Defektart entschieden werden.
- Übersicht1 Zeile1 und Übersicht2 Zeilen4/6: sehr große unscharfe, teils kantige Flecken verdecken einen erheblichen Teil des Gefäßes und überschreiben gesunde Struktur. Das wirkt anders als ein lokal übertragener Originaldefekt.
- Maskenanteil in der Stichprobe: **0,0020 % bis14,40 % der gesamten Bildfläche**, Median0,0116 %. Dies ist keine Größenmessung relativ zur Objektfläche und begründet allein keine Filtergrenze. Die starke Spannweite passt aber zu den sichtbaren Extremfällen.

**Folgerung:** Donorgruppen beim Matching/Sampling ausbalancieren; Defektgröße und Skalierung pro Defektart begrenzen; große strukturelle Defekte von kleinen Partikeln/Blasenveränderungen getrennt behandeln. Eine pauschale Mindestgröße würde gültige kleine Defekte entfernen.

## 3. Can: semantisch unpassende Ziele und wiederkehrende Druckdefekte

[Übersicht 1](can_overview_1.jpg) · [Übersicht 2](can_overview_2.jpg)

- Übersicht2 Zeile4: `hybrid_82d0ebd52cfc0d7b933a` enthält eine kleine Veränderung im weißen Hintergrund oberhalb der Dose. Das ist für einen auf dem Dosenaufdruck entstandenen Defekt ein unpassender Zielbereich.
- Übersicht2 Zeile2 zeigt eine Platzierung im spiegelnden/facettierten Hintergrund neben der Dose. Kleine lokale Bildstatistiken reichen offenbar nicht zur Materialzuordnung.
- Viele gezeigte Beispiele sind sehr ähnliche lokale Änderungen an Druckzeichen oder Linien. Einige sind plausibel, decken aber nicht automatisch die Vielfalt anderer Defekttypen ab.
- Von434Hybrid–Donorbild-Zuordnungen entfallen362(83,41 %) auf die drei Aufnahmegruppen009,013 und012. Insgesamt werden6Aufnahmegruppen verwendet.
- Maskenanteil der Stichprobe: Median0,0167 % der Gesamtbildfläche. Kleine Masken sind bei Druckfehlern nicht per se falsch.

**Folgerung:** Zuerst Dosen-/Etikettenregion abgrenzen, danach Position/Orientierung innerhalb kompatibler Bereiche matchen. Quellenvielfalt nach Defektart untersuchen, statt lediglich mehr Varianten derselben Quelle zu generieren.

## Was in dieser Prüfung unauffällig war

- Keine Überschneidung der verwendeten Donorbildpfade mit den öffentlichen Validierungs-/Testbildpfaden in den eingefrorenen Modellmanifesten.
- Alle143geprüften Hybridbilder haben endliche Bildwerte.
- In allen143Bildern beträgt die mittlere absolute Änderung **außerhalb der um8Pixel erweiterten Maske exakt0**. Eine flächige Hintergrundänderung durch Harmonisierung wird in dieser Stichprobe nicht beobachtet. Direkt am Maskenrand ist damit keine Fehlerfreiheit bewiesen.
- Can ist laut Study-Konfiguration mit LBM-Harmonisierung materialisiert; Vial und Wallplugs ohne Harmonisierung. Eine pauschale Abschaltung von LBM würde daher die beiden anderen Kategorien nicht verändern.

## Gespeicherte Modellvorhersagen auf echten öffentlichen Defekten

Die zusätzlichen Abbildungen nutzen ausschließlich bereits gespeicherte Vorhersagen der tatsächlich eingereichten Modelle. Es wurde keine neue Modellinferenz ausgeführt. Pro Kategorie werden vier auffällige echte öffentliche Defektbilder ausgewählt über `Mittelwert Score im Defekt − 99%-Quantil Hintergrundscore` (aufsteigend). Das ist eine Diagnoseheuristik, **keine AUC-PRO-Metrik**. Rot in der Heatmap bedeutet1, blau0; Maskenüberlagerung rot. Die vollständigen Bildpfade und Rankings stehen in den JSON-Dateien.

- [Can: echte Defekte und Vorhersagen](can_real_prediction_examples.jpg)
- [Vial: echte Defekte und Vorhersagen](vial_real_prediction_examples.jpg)
- [Wallplugs: echte Defekte und Vorhersagen](wallplugs_real_prediction_examples.jpg)

Bei Vial zeigen die vier ausgewählten Beispiele sämtlich Aufnahmegruppe012: Die Ground-Truth markiert einen großen Teil des Gefäßkörpers, während die Vorhersage überwiegend nur einzelne Stellen hervorhebt. Das illustriert eine reale Lokalisierungsschwäche; die Auswahlheuristik ist auf solche großen schwach bewerteten Regionen empfindlich und ersetzt keine regionale PRO-Auswertung.

**Noch offen:** Modellvorhersagen auf denselben Hybridbildern und eine quantitative Verknüpfung der Qualitätsfehler mit Modellfehlern. Die vorliegenden Bildübersichten allein beweisen nicht, dass der Decoder Einfügeränder als Abkürzung verwendet.

## Empfohlener nächster kontrollierter Vergleich

1. Zunächst falsche Objekt-/Hintergrundplatzierungen aussortieren und Donorgruppen ausbalancieren, bei unverändertem Downstream-Modell und unveränderter Mischung.
2. Vial zusätzlich separat auf extreme Größen/Strukturüberschreibung prüfen. Visuell geprüfte Defektart berücksichtigen.
3. Aktuelle und bereinigte Hybride mit identischem Seed und derselben öffentlichen Validierung vergleichen. Keine neue private Einreichung allein aufgrund schönerer Bilder.

Noch keine dieser Änderungen umgesetzt.

## Reproduzierbarkeit

- `analyze.py`: SQLite im Read-only-Modus, Manifestabgleich, Auswahl und Übersichten.
- `*_sample_metrics.csv`: IDs, Quell-/Zielnamen, Scores, Maskengrößen und Bildänderungsmaße.
- `summary.json`: Counts, Quellkonzentration und gelesene Study-Konfigurationen.
- `real_predictions.py`: Diagnoseabbildungen gespeicherter öffentlicher Vorhersagen.
- Lauf: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python notes/hybrid_audit/analyze.py`

Private Ergebnisse der zweiten Einreichung zur Einordnung: Can61,77 %, Vial65,96 %, Wallplugs53,69 % AUC-PRO0.05. Die Analyse verwendet keine privaten Labels.
