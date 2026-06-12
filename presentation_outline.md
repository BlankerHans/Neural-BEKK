# Foliengerüst: Masterarbeit-Präsentation (30 Min)

**Thesis:** Hybrid Neural–MGARCH Models for Volatility and Financial Risk Forecasting
**Format:** 17 Folien + Backup · Ziel: 27–28 Min Redezeit
**Prinzip:** Jeder Titel ist eine vollständige Aussage (Action Title). Wer nur Titel liest, versteht die Story.

---

## Block 1 — Einstieg (2 Folien, ~2 Min)

### Folie 1 — Titel
- Hybrid Neural–MGARCH Models for Volatility and Financial Risk Forecasting
- Name, Datum, Betreuer, Uni-Logo
- *Regie: 20 Sekunden. Ein Satz, worum es geht, dann weiter.*

### Folie 2 — Agenda
- Motivation · Models · Data & Evaluation Design · Results · Conclusion
- *Regie: Max. 5 Punkte — nicht wie bei Nils mit Unterpunkten überladen.*

---

## Block 2 — Motivation & Forschungsfrage (2 Folien, ~4 Min)

### Folie 3 — "Portfolio risk lives in the covariance matrix — and the covariance matrix moves"
- Portfolio-VaR/ES hängen an Σₜ: σ²ₚ,ₜ = w'Σₜw — und Σₜ ist zeitvariabel
- Grafik: Rolling Correlations (`images/content/rolling_correlations_main.pdf`) — Equity–Gold wechselt das Vorzeichen, Flight-to-Quality bei Equity–Treasury
- Regulatorischer Haken: FRTB-Backtesting basiert auf 1%-VaR-Exceedances
- *Regie: Mit dem Problem einsteigen, nicht mit Modellhistorie. Die Grafik trägt die Folie.*

### Folie 4 — "Econometric MGARCH is structured but rigid; neural networks are flexible but structure-free"
- BEKK/DCC: garantiert PSD, interpretierbar — aber konstante Parametermatrizen, enge Dynamik
- LSTM/GRU: universelle Approximatoren — aber keine Spillover-Struktur, instabiles Training
- Lücke in der Literatur: Neural-augmented MGARCH wird fast nie gegen *rein neuronale* Baselines getestet → diese Arbeit vergleicht bidirektional
- Forschungsfrage: Können hybride Neural-MGARCH-Modelle beide Welten verbinden — und liefern sie *adäquate* Risikoprognosen?
- *Regie: Das ist deine Spannungs-Folie (wie Nils' "effective or efficient"). Hier entscheidet sich, ob die Zuhörer die nächsten 25 Min folgen wollen.*

---

## Block 3 — Modelle (4 Folien, ~8 Min)

### Folie 5 — "BEKK guarantees a valid covariance matrix by construction"
- BEKK(1,1)-Rekursion als *einzige* Gleichung: Σₜ = CC' + A'εₜ₋₁εₜ₋₁'A + G'Σₜ₋₁G
- Quadratische Formen ⇒ PSD garantiert; asymmetrische Erweiterung für Leverage
- Benchmarks der Arbeit: BEKK sym./asym. (QML, R `BEKKs`), scalar DCC/ADCC (`rmgarch`)
- *Regie: Eine Gleichung pro Folie, maximal. Parameterzahlen (O(N²) vs. O(N) bei DCC) nur mündlich oder als Fußzeile.*

### Folie 6 — "Recurrent networks forecast Σₜ through a Cholesky head — flexibility without PSD worries"
- LSTM/GRU → vech(Lₜ) → Σₜ = LₜLₜ' (PSD by construction)
- Training: Adam auf bedingter NLL — Gaussian und Student-t Varianten
- Rein datengetrieben: keine ökonometrische Struktur auferlegt
- Architektur-Skizze: Input rₜ₋₁ → RNN → Cholesky-Faktor → Σₜ
- *Regie: Die Skizze schrittweise aufbauen (Keynote-Build), nicht alles auf einmal.*

### Folie 7 — "Neural-BEKK lets a recurrent network drive time-varying BEKK parameter matrices" ⭐
- Kernidee: γₜ = f_θ(Ω₍ₜ₋₁₎) — die BEKK-Matrizen werden zeitvariabel, die quadratische Form bleibt
- Beitrag: *vollständig asymmetrisches* Neural-BEKK — Leverage + (a)symmetrische Spillovers in Varianzen **und** Kovarianzen (so in der Literatur bisher nicht)
- Varianten: scalar · vector · asym. diagonal · mix
- Architektur-Diagramm analog Folie 6, mit BEKK-Rekursion als Output-Schicht
- *Regie: Das ist DEINE Folie — hier 2–3 Minuten verweilen. Wie bei Nils' Trambular-Folie: weil Folien 5+6 die Bausteine erklärt haben, ist diese fast selbsterklärend.*

### Folie 8 — "Alternatively, a BEKK kernel can replace the gate of a standard LSTM/GRU"
- Zweiter Hybrid-Weg: MGARCH-LSTM/GRU — BEKK-Kernel ersetzt das Output-Gate
- Netzwerk moduliert die Kernel-Dynamik statt Σₜ frei zu erzeugen
- *Regie: Kurz halten (≤1 Min) oder ins Backup schieben, falls Zeit knapp — der Haupt-Beitrag ist Folie 7.*

---

## Block 4 — Daten & Evaluationsdesign (2 Folien, ~4 Min)

### Folie 9 — "Four heterogeneous asset classes over 24 years exhibit every stylized fact"
- S&P 500 · Gold · WTI Crude · 10Y Treasury Yield — Jan 2000 bis März 2024, ~6.400 Beobachtungen
- Grafik: Log Returns (`images/content/log_returns.pdf`) — Clustering sichtbar: 2008, Ölpreis 2014–16, COVID 2020, Zins-Repricing 2022
- Gleichgewichtetes Portfolio, Gewichte fix ⇒ Unterschiede in VaR/ES kommen *ausschließlich* aus Σₜ
- *Regie: Deskriptive Statistik-Tabelle ins Backup. Der Equal-Weights-Punkt ist wichtig — typische Prüferfrage ("Warum keine Optimierung?") direkt vorwegnehmen.*

### Folie 10 — "Evaluation follows the FRTB logic: 1% VaR exceedances and a joint VaR–ES score"
- Out-of-Sample: 925 Tage · α = 1% · 11 Trainings-Seeds (Seed-Streuung = Trainingsinstabilität)
- Adäquanz: VaR-Backtests (Kupiec, Christoffersen, DQ) + ES-Backtests (ER, CC, ESR)
- Vergleich: FZ0-Loss (Fissler–Ziegel, strikt konsistent für (VaR, ES)) → Diebold–Mariano vs. symmetrisches BEKK + Model Confidence Set (90%)
- *Regie: Zwei-Spalten-Layout: "Is the model adequate?" | "Is it better than the benchmark?" — diese Arbeitsteilung trägt die drei Ergebnisfolien.*

---

## Block 5 — Ergebnisse (3 Folien, ~7 Min)

### Folie 11 — "Hybrid Neural-BEKK models achieve the lowest joint VaR–ES loss"
- Kondensierte Version von Table 1 (Top/Bottom, 5–6 Zeilen statt 12) **oder** Balkendiagramm der FZ0-Losses
- Kernzahlen: Neural BEKK asym. diag. −2.894 · mix −2.892 · scalar −2.889 // BEKK sym. −2.846 · LSTM −2.838 · DCC/ADCC −2.815
- Alle Neural-BEKK-Varianten in 11/11 Seeds im 90%-MCS
- Hybride schlagen *beide* Eltern: das ökonometrische BEKK **und** die reinen RNNs
- *Regie: Beste Zeile farblich hervorheben. Volle Tabelle ins Backup.*

### Folie 12 — "Classical models systematically underestimate tail risk — the hybrids pass the backtests"
- BEKK/DCC/ADCC: Hit Rate 1.41% statt 1.0% (13 statt 9,25 Exceptions) — DQ-Test verwirft in 11/11 Seeds
- Neural BEKK (scalar): **null** Rejections über alle VaR- und ES-Tests
- Nuance ehrlich benennen: asym. diag./mix haben ESR-Rejections (6–8/11) — bester Loss ≠ automatisch sauberste ES-Kalibrierung
- *Regie: Das ist dein stärkstes Ergebnis fürs Risikomanagement-Publikum: Die Klassiker sind nicht nur schlechter, sie sind* inadäquat*.*

### Folie 13 — "The improvement is consistent across seeds — but only partly statistically significant"
- DM vs. BEKK: Neural BEKK in 11/11 Seeds bevorzugt, signifikant aber nur in 3/11 (asym. diag.) bzw. 5/11 (mix)
- Grafik: kumulativer FZ0-Loss-Differential (`images/fig_cum_loss_diff.pdf`) — zeigen, *wann* die Gewinne entstehen
- Seed-Stabilität: Std. der Neural BEKKs 0.011–0.026 (Boxplot `images/fig_fz_loss_seeds.pdf` ggf. Backup)
- *Regie: Diese Ehrlichkeit ("konsistent, aber moderat") ist eine Stärke, keine Schwäche — sie nimmt der Verteidigung die offensichtlichste Attacke.*

---

## Block 6 — Abschluss (3 Folien, ~4 Min)

### Folie 14 — Limitations
- Ein Portfolio: N = 4, gleichgewichtet — Skalierung auf größere N offen (Neural-BEKK: O(hN²) Parameter)
- Ein OOS-Fenster (925 Tage); Seeds teilen das Testfenster ⇒ Streuung misst Trainings-, nicht Sampling-Unsicherheit
- DM-Signifikanz nur in Teilmenge der Seeds; Hyperparameter-Suche begrenzt
- Rechenkosten: Stunden Training vs. Sekunden QML-Schätzung
- *Regie: Konkret und ehrlich wie bei Nils — keine Pseudo-Limitations ("mehr Daten wären gut").*

### Folie 15 — "Conclusion: Structure helps the network, flexibility helps the econometrics"
- ① Hybride Neural-BEKK-Modelle liefern die besten VaR–ES-Prognosen und bestehen die Backtests — sie schlagen beide Elternklassen
- ② Klassische BEKK/DCC unterschätzen das Tail-Risiko dieses Portfolios systematisch
- ③ Der Gewinn ist konsistent, aber moderat — die ökonometrische Struktur wirkt vor allem als Regularisierung des Netzes
- *Regie: Genau drei merkbare Sätze, wie Nils' Conclusion. Punkt ③ ist die These, die hängen bleibt.*

### Folie 16 — Thank you + Selected Sources
- Engle (2002), Engle & Kroner (1995), Fissler & Ziegel (2016), Hansen et al. (2011), + 2–3 Neural-MGARCH-Referenzen
- *Regie: Danksagung an Betreuer optional, Quellen klein.*

---

## Backup-Folien (hinter "Thank you")

1. Volle Table 1 (Main Results, alle 12 Modelle)
2. Volle VaR-Backtest-Tabelle (Table 2) + Testdefinitionen
3. Volle ES-Backtest-Tabelle (Table 3) + ESR-Varianten (v1/v2/v3)
4. Seed-Boxplot (`fig_fz_loss_seeds.pdf`) — Trainingsstabilität
5. Parameterkomplexität aller Modelle (Tabelle aus Section 4)
6. Student-t-NLL & FZ0-Loss-Definition (Formeln)
7. Trainingsdetails: Adam, Early Stopping, Input-Preprocessing, Sample-Split
8. Deskriptive Statistik + JB/LB/ADF-Tests
9. Alle paarweisen Rolling Correlations
10. MGARCH-LSTM/GRU-Details (falls Folie 8 gestrichen wird)

---

## Timing-Übersicht

| Block | Folien | Minuten | kumuliert |
|---|---|---|---|
| Titel + Agenda | 1–2 | 2 | 2 |
| Motivation | 3–4 | 4 | 6 |
| Modelle | 5–8 | 8 | 14 |
| Daten + Design | 9–10 | 4 | 18 |
| Ergebnisse | 11–13 | 7 | 25 |
| Limitations + Conclusion | 14–15 | 3 | 28 |
| Danke | 16 | <1 | 28–29 |

**Puffer-Strategie bei Überziehen:** Folie 8 (MGARCH-Kernel) streichen → spart 1–2 Min. Notfalls Folie 13 auf die Grafik reduzieren.
