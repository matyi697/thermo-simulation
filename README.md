# Egyszerű Hőterjedés Szimulátor

## Leírás

Ez a projekt egy egyszerű hőterjedés szimulátor, amely két különböző megvalósítást hasonlít össze: egy CPU-alapú és egy GPU-alapú megoldást. A szimulátor célja a hőmérséklet eloszlásának modellezése egy 2D rácsos térben, és a teljesítménybeli különbségek kiértékelése.

## Funkciók

- **CPU-alapú szimuláció**: A hőterjedési szimulációt a központi feldolgozóegységen valósítjuk meg.
- **GPU-alapú szimuláció**: A hőterjedés szimulációját grafikus feldolgozóegységen (GPU) végezzük a CUDA platform segítségével.
- **Teljesítménymérés**: A két megvalósítás futási idejének és teljesítményének összehasonlítása.
- **Eredmények vizualizációja**: Az eredmények grafikus megjelenítése.

## Eredmények

A projekt célja a CPU és GPU alapú szimulációk összehasonlítása. Várhatóan a GPU verzió gyorsabb futási időt fog eredményezni a párhuzamos feldolgozás lehetősége miatt.