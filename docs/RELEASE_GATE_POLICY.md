# Release-Gate-Policy (IP-Schutz)

Diese Policy definiert verbindliche Schutzmaßnahmen vor jeder externen Veröffentlichung (Repo, Website, Whitepaper, Presse, Slides, Demos).

## 1) IP-Schutz
- Patent-Pending Inhalte nur aggregiert darstellen (keine Source-Snippets, keine Architektur-Details, keine internen Diagramme).
- NDA-gebundene Daten/Algorithmen strikt aussparen (auch keine indirekten Hinweise).
- Keine vertraulichen Markierungen: „NDA ONLY“, „DO NOT PUBLISH“, „CONFIDENTIAL“, „PROPRIETARY“, „INTERNAL ONLY“ in öffentlichen Artefakten.

## 2) Kontaktangaben
- Einheitlich: `info@mimitechai.com` in allen Materialien (READMEs, Docs, Slides, Banners, PR/Press, Website).

## 3) Freigabeprozess (Legal & CTO)
- Veröffentlichungen erst nach Freigabe durch Legal & CTO.
- GitHub PR muss die Labels `legal-approved` und `cto-approved` tragen.
- Branch-Schutz: Der Workflow `release-gate` muss als Required Check in GitHub branch protection für `main` gesetzt werden.

## 4) Technische Durchsetzung (CI)
- Workflow `.github/workflows/release_gate.yml` prüft PRs gegen `main` und blockiert, wenn:
  - Pflicht-Labels fehlen (`legal-approved`, `cto-approved`).
  - Sensible Pfade (z. B. `vxor/`, `core/`, `agi_missions/`) verändert wurden.
  - Blacklist-Marker in geänderten Dateien enthalten sind (z. B. „NDA ONLY“, „CONFIDENTIAL“, …).

## 5) Verantwortlichkeiten
- Author/Owner: prüft Inhalte aktiv und setzt Labels.
- Legal & CTO: finaler Gatekeeper via Labels + Review.

## 6) Ausnahmen
- Ausnahmen sind schriftlich zu dokumentieren (Ticket-Referenz) und von Legal & CTO abzuzeichnen; CI-Ausnahme nur temporär und begründet.
