# Einstieg in die Programmierung / Introduction to Programming

## Infos zur Beispiel/Example Code

This repository contains some of the code that was shown at the [HeFDI Forschungsdatentag 2022](https://www.uni-marburg.de/de/hefdi/events/vergangene-veranstaltungen/fdt-2022) in the "Einstieg in die Programmierung mit, insbesondere, Python" Session.

### Einstieg in Python

Infos auf [wiki.python.org/BeginnersGuide](https://wiki.python.org/moin/BeginnersGuide).  

Eine Pyton Virtuelle Umgebung aufbauen und die Packages (von requirements.txt) installieren geht mit:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Einstieg mit Datentypen in `0_datentypen.py` und mit Python Klassen(Classes) in `1_python-class-eisladen.py`.



Die Beispiele mit Wetterdaten verwenden die [`wetterdienst`](https://github.com/earthobservations/wetterdienst) Bibliothek für Python womit die Wetterdaten aufgerufen und verarbeitet werden können.
Zur Modellierung wurde das [`LMfit-py`](https://github.com/lmfit/lmfit-py) verwendet.

Die Python Modulen können ausprobiert/ausgeführt werden, dabei wurden die Plots erstellt (wenn alles geklappt hat).
```bash
python3 2a_wetterdienst-plot-beispiel.py

python3 2c_wetterdienst-daten-modellierung.py
```



