# AnomalyDiffusion im eingefrorenen V3-Vergleich

Dieser Ordner integriert den [Originalcode von AnomalyDiffusion](https://github.com/sjtuplayer/anomalydiffusion)
(AAAI 2024), Commit `b85297f333ebfc59b3863338445b81363efb9761`.
Es werden die ursprünglichen LDM/BERT-Gewichte, Textembeddings, der räumliche
pSp-Encoder und die gelernte Maskengeneration verwendet. Die optionale adaptive
Aufmerksamkeitsgewichtung (AAR) bleibt über `--adaptive-attention` verfügbar.

Die Integration ist eine **an MVTec AD 2 angepasste Baseline**. Sie reproduziert
nicht das ursprüngliche MVTec-AD-Protokoll des Papers.

## Festes Vergleichsprotokoll

| Quelle pro Epoche | Echt + AnomalyDiffusion | Echt + AnomalyDiffusion + Hybrid |
|---|---:|---:|
| Gesund | 500 | 500 |
| Echt anomal | 100 | 100 |
| AnomalyDiffusion | 400 | 200 |
| Hybrid | 0 | 200 |
| Gesamt | 1000 | 1000 |

Beide Varianten verwenden denselben vorberechneten AnomalyDiffusion-Pool.
Die Quoten bezeichnen Ziehungen pro Epoche, nicht die Anzahl verschiedener
generierter Bilder. Standardmäßig werden 1000 verschiedene Bild-Masken-Paare
generiert. Es gibt keine visuelle Nachselektion nach Benchmark-Ergebnissen.

Referenz für jede Kategorie ist die eingefrorene V3-Abgabe `Echt + DRAEM`:
`/mnt/results/mvtec2/submissions/v3_variant2_best_20261025T144555Z`.
Der gruppierte 60/40-Train/Validierungssplit, leere öffentliche Testpartition,
Seed 42, DINOv3-ViT-S/16, Backbone-Revision, CNN-Decoder, native 256er-Ausschnitte,
Batchgröße 8, 100 Epochen und Auswahl über Pixel-AUPRO@0.05 bleiben erhalten.
`downstream.py` vergleicht die Konfiguration vor dem Start mit der Referenz.

Die Hybridquelle stammt exakt aus der ausgewählten V3-Abgabe `Echt + DRAEM + Hybrid`
(`v3_variant5_best_20261023T130904Z`), einschließlich der ausgewählten Reruns für
can und wallplugs. Beide Referenzabgaben müssen dieselben Splitmanifeste haben.
Gradientenupdates und gesunde Generationshintergründe verwenden ausschließlich
Trainingsbilder. Die Generatorvalidierung verwendet getrennte native Ausschnitte
der bestehenden öffentlichen V3-Validierung. Private Testbilder werden nicht verwendet.

Die exakten Hybrid-IDs und ihre Reihenfolge werden aus der eingefrorenen
Downstream-Provenienz übernommen (`hybrid_provenance.json`, mit Prüfsumme).
Insbesondere nutzt wallplugs dieselben 703 ausgewählten Hybride wie V3,
obwohl der Studienordner 1160 erzeugte Hybride enthält.

Der anschließende [Hybrid-Auflösungsaudit](/mnt/results/mvtec2/audits/hybrid_resolution_audit_2026-10-31/README.md)
weist auf eine weitere Vergleichsgrenze hin: Der historische Hybrid-Größenfilter
entfernte 120 kleine Originalmaskenkomponenten. Die AD-Vorbereitung erhält bislang
alle ursprünglichen Annotationen. Ein zusätzlicher Vergleich mit angeglichenen
Eingabemasken ist für eine Aussage über reine Generatorunterschiede sinnvoll;
die vorbereiteten AD-Daten wurden durch den Audit nicht eingeschränkt.

## Auflösung und dokumentierte Anpassungen

Das Verkleinern eines ganzen hochauflösenden Bildes auf 256×256 kann winzige
Defekte entfernen: Aus drei Pixeln werden bei einem Faktor acht weniger als
ein Pixel. Deshalb exportiert `prepare.py` native 256×256-Ausschnitte ohne
Skalierung. Eine disjunkte Kachelung deckt sämtliche annotierten Defektpixel
ab; bei großen Defekten zeigen einzelne Kacheln nur einen Teil des Defekts.
Gesunde Bilder liefern jeweils vier mit Seed 42 gewählte Ausschnitte.
Kleine Randkacheln werden mit Nullen aufgefüllt. Alle Koordinaten und
Originaldatei-Prüfsummen werden gespeichert.

Auch innerhalb von AnomalyDiffusion entsteht ein Auflösungsproblem: Das LDM
arbeitet auf 32×32. Der ursprüngliche Code verkleinert Masken bilinear und
schwellt sie bei 0.5. `resolution_audit.json` zählt für jede native Maske,
wie viele Pixel dabei überleben. Bei can verschwinden im ersten Audit
24/66 Masken, bei fruit_jelly 8/176 vollständig.

Standard `--latent-mask-policy maxpool` erhält jede belegte 8×8-Zelle auf
dem Latentraster mittels Max-Pooling. Diese Anpassung betrifft ausschließlich
die Masken im latenten Trainingsverlust und beim Inpainting. Native Masken,
räumliche Konditionierung und Downstream-Labels werden nicht dilatiert oder
verkleinert. Die gröbere latente Belegung garantiert keine detailgetreue
Bildsynthese; deren Qualität muss nach echtem Generatortraining geprüft werden.
`--latent-mask-policy original` verwendet den ursprünglichen bilinearen Weg
und verweigert das Erscheinungstraining bei vollständig verschwindenden Masken.
Generierte leere Masken und Masken, die den ursprünglichen Randfilter verletzen,
werden mit begrenzter Anzahl neuer Versuche ersetzt. Es gibt keinen stillen
Rückfall auf reale Masken.

Der vollständige erste Trainingsdaten-Audit vom 31.10.2026:

| Kategorie | Defekthaltige native Kacheln | Davon im ursprünglichen Latentraster leer |
|---|---:|---:|
| can | 66 | 24 |
| fabric | 114 | 0 |
| fruit_jelly | 176 | 8 |
| rice | 138 | 0 |
| sheet_metal | 462 | 12 |
| vial | 399 | 14 |
| wallplugs | 138 | 0 |
| walnuts | 540 | 18 |

Diese Zahlen beziehen sich auf Kacheln, einschließlich Randfragmenten großer
Defekte, nicht auf die Anzahl vollständig verschwundener Originaldefekte.
Die maschinenlesbare Zusammenfassung steht in `preparation_summary.json`.

Unsere bisherige Hybridpipeline skaliert ebenfalls: Defekt-Boundingboxen werden
auf 128×128 beziehungsweise bei fruit_jelly/walnuts 64×64 normalisiert und bei
der Fusion zurückskaliert. Dabei können feine Details verloren gehen; auch
Extraktionsfilter können kleine Komponenten ausschließen. Der spätere Downstream
nutzt native 256er-Ausschnitte. Das vermeidet das Verkleinern ganzer Bilder,
beweist aber keinen vollständigen Detailerhalt in der Hybridgeneration.

Weitere Anpassungen: ein train-only Manifestadapter ersetzt die fest eingebauten
MVTec-AD-Verzeichnislisten und deren First-Third-Split; die ursprüngliche
512er-`data_enhance`-Skalierung ist deaktiviert; PNG ersetzt JPEG für Bilder und
binäre Masken. Der pSp-Import lädt nur die unveränderte `EqualLinear`-Klasse des
Originalcodes, um ungenutzte StyleGAN-CUDA-Erweiterungen nicht kompilieren zu
müssen. Die verwendete LDM-Architektur und die Optimierungsverluste bleiben
Originalcode, abgesehen von der ausdrücklich genannten latenten Maskenbelegung.

## Einrichtung

Alle Befehle im Projektwurzelordner ausführen. Generator und DINOv3 verwenden
getrennte Python-Umgebungen. Die lokale Umgebung ist Python 3.10/Torch 2.0.1;
das originale `environment.yaml` verwendet Python 3.8/Torch 1.12.1.
`requirements.lock` hält die tatsächlich verifizierten Abhängigkeiten fest.

```bash
bash experiments/anomalydiffusion/setup.sh
.venv/bin/python -m experiments.anomalydiffusion.fetch_weights \
  --output /mnt/results/mvtec2/model_cache/anomalydiffusion/model.ckpt
```

Die öffentlichen LDM-Gewichte sind 6.152.314.307 Bytes groß und werden gegen
SHA256 `7c95fc34febdbf15fea1481a87db43c935ecadeb1e300dd6bc3de3f5f663e298`
geprüft. Beim ersten Modellstart werden außerdem der BERT-Tokenizer und die
vom Upstream importierten Kornia-HardNet-Gewichte heruntergeladen.

## Datenvorbereitung und Training

Die Daten liegen außerhalb des Git-Repositories. Bereits vorhandene Zielordner
werden nicht überschrieben. Die acht Kategorien heißen can, fabric, fruit_jelly,
rice, sheet_metal, vial, wallplugs und walnuts.

```bash
.venv/bin/python -m experiments.anomalydiffusion.prepare \
  --category can --output /mnt/results/mvtec2/experiments/anomalydiffusion_v3/can

CUDA_VISIBLE_DEVICES=1 bash experiments/anomalydiffusion/run.sh train \
  --inputs /mnt/results/mvtec2/experiments/anomalydiffusion_v3/can \
  --output /mnt/results/mvtec2/experiments/anomalydiffusion_v3/can/mask_training \
  --checkpoint /mnt/results/mvtec2/model_cache/anomalydiffusion/model.ckpt \
  --stage mask

CUDA_VISIBLE_DEVICES=1 bash experiments/anomalydiffusion/run.sh train \
  --inputs /mnt/results/mvtec2/experiments/anomalydiffusion_v3/can \
  --output /mnt/results/mvtec2/experiments/anomalydiffusion_v3/can/appearance_training \
  --checkpoint /mnt/results/mvtec2/model_cache/anomalydiffusion/model.ckpt \
  --stage appearance
```

Generator-Voreinstellungen verwenden 4 Erscheinungs- plus
4 räumliche Tokens, 16 Maskentokens, Basislernrate 0.005 (mit der ursprünglichen
Skalierung bei Batchgröße 4 effektiv 0.02), Batchgröße 4, **100.000
Erscheinungsschritte** und 30.001 Maskenschritte aus `train_mask.py`.
Die Obergrenze wurde am 01.11.2026 auf ausdrücklichen Nutzerwunsch gegenüber
den 1.000.000 Schritten des öffentlichen YAML reduziert.

`reference_training.py` verwendet das ursprüngliche LightningModule mit seinen
`training_step`- und `validation_step`-Methoden sowie den nativen Lightning-Callback
`ModelCheckpoint`: `monitor=val/loss_simple_ema`, `mode=min`, `save_top_k=1`,
`save_last=True`. Wie im YAML ist `use_ema=False`; der zweite Validierungsdurchlauf
wird trotzdem unter dem Namen mit `_ema` geloggt. Checkpoint-Prüfungen erfolgen
alle 2.000 Anomalie- beziehungsweise 1.000 Maskentrainingsschritte. Der Callback
verwendet dabei den zuletzt verfügbaren Validierungswert wie im Referenzcode.

Die Eingabedaten und Epochenlängen unterscheiden sich von MVTec AD: Wir verwenden
die vollständigen anomalen Patches aus dem festgelegten V3-Validierungssplit.
Die Validierung erfolgt am Epochenende, bei langen Epochen zusätzlich spätestens
nach einem Checkpointintervall. Dadurch endet Maskentraining nicht ohne eine
Validierung. Dies ist eine ausdrücklich dokumentierte Anpassung an den V3-Adapter.
Die trainierenden Datensätze wiederholen sich wie in den Referenz-YAMLs 20-mal
(Anomalien) beziehungsweise 2.000-mal (Masken) und werden per DataLoader gemischt.

Der Original-Speicherhook erzeugt weiterhin `embeddings.pt`, schrittnummerierte
`embeddings_gs-*.pt` und beim Anomalietraining `spatial_encoder.pt`. Zusätzlich
bewahren wir den vollständigen Lightning-Best-/Last-Zustand auf, den der Original-
Hook bei eingefrorener Basis sonst leeren würde: Modell, Optimierer, Metadaten
und Zufallszustände. So bleiben insbesondere Embeddings und räumlicher Encoder
des besten Zeitpunkts zusammen erhalten. Die Generierung lädt den über
`selection.json` ausgewählten Best-Checkpoint eines abgeschlossenen Laufs.
Ein zusätzliches Early Stopping ist wie in der Referenz nicht aktiviert.

`--resume /pfad/checkpoints/last.ckpt` setzt den vollständigen Lightning-Lauf
in einem neuen Ausgabeordner fort; `--steps` meint die gesamte
Zahl der Schritte. `--steps 1 --batch-size 2 --smoke` prüft ein echtes Update.
Smoke-Gewichte und Smoke-Paare sind für Benchmarkläufe gesperrt.

## Generierung und Downstream

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/anomalydiffusion/run.sh generate \
  --inputs /mnt/results/mvtec2/experiments/anomalydiffusion_v3/can \
  --output /mnt/results/mvtec2/experiments/anomalydiffusion_v3/can/generated \
  --checkpoint /mnt/results/mvtec2/model_cache/anomalydiffusion/model.ckpt \
  --mask-weights /mnt/results/mvtec2/experiments/anomalydiffusion_v3/can/mask_training/selection.json \
  --appearance-weights /mnt/results/mvtec2/experiments/anomalydiffusion_v3/can/appearance_training/selection.json \
  --count 1000 --ddim-steps 200
```

Masken werden standardmäßig mit 100 DDIM-Schritten generiert
(`--mask-ddim-steps`), Erscheinungsbilder mit 200.

AAR ist standardmäßig aus. Eine Entscheidung pro Kategorie/Defekttyp muss
vor dem Vergleich festgelegt und dokumentiert werden; niemals mit privatem
Testfeedback auswählen. Das öffentliche AD-2-Label `bad` wird hier als ein
Defekttyp pro Kategorie behandelt.

Der Downstream verwendet die bestehende Projektumgebung mit dem bereits
vorhandenen eingefrorenen DINOv3-Runtime-Pfad:

```bash
PYTHONPATH=/mnt/results/mvtec2/submissions/v3_variant4_hybrid_real_20261030T123827Z/runtime_libs:. \
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m experiments.anomalydiffusion.downstream \
  --inputs /mnt/results/mvtec2/experiments/anomalydiffusion_v3/can \
  --variant real_anomalydiffusion \
  --output /mnt/results/mvtec2/experiments/anomalydiffusion_v3/downstream/real_anomalydiffusion/can
```

Für die zweite Variante `--variant real_anomalydiffusion_hybrid` und einen
eigenen Ausgabeordner verwenden. Beide lesen dieselbe `generated/manifest.json`.
Der Cache-Adapter prüft Split, Trainingsherkunft, gesunde Hintergründe,
Ausrichtung und nichtleere binäre 256er-Masken. Die Quellen heißen im Training
ausdrücklich `anomalydiffusion`; sie werden nicht als DRAEM oder Hybrid verbucht.
Generierte Paare erhalten dieselben gepaarten Verluste wie andere synthetische
Paare, echte Defekte weiterhin ausschließlich Segmentierungsverluste.

```bash
.venv/bin/python -m unittest tests.example_cases.mvtec_ad2.test_anomalydiffusion \
  tests.example_cases.mvtec_ad2.test_benchmark \
  tests.example_cases.mvtec_ad2.test_real_training \
  tests.example_cases.mvtec_ad2.test_splits \
  tests.example_cases.mvtec_ad2.test_grouped_splits -q
```

## Verifizierter Stand am 31.10.2026

Alle acht Kategorien sind unter
`/mnt/results/mvtec2/experiments/anomalydiffusion_v3` vorbereitet. Alle 16
Downstream-Konfigurationen wurden mit den eingefrorenen V3-Referenzen verglichen;
die gespeicherten Hybrid-Konfigurationsprüfsummen stimmen überein.
26 relevante Unit-/Integrationstests bestanden. Auf GPU 1 wurden echte Updates
für Masken- und Erscheinungstraining sowie ein fortgesetztes Maskentraining
ausgeführt. Die Generierung mit diesen Smoke-Gewichten erzeugte sowohl zwei
Paare als auch einen einzelnen letzten Batch mit korrekten 256er-Formaten und
binären Masken. Artefakte liegen im Unterordner `smoke` des externen Datenordners.
Das bestätigt die technische Ausführbarkeit, nicht die Synthesequalität nach
vollständigem Training. Vollständige Generatortrainings, Downstream-Vergleichsläufe
und externe Abgaben wurden für diese neue Baseline noch nicht gestartet.
