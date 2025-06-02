# YOLO-2025-05

SZE kártyaészlelő YOLO modell tanító és észlelő script és dummy kártya generáló script megvalósítása Python nyelvben


## Futtatása

Szedje le a projektet

```bash
  git clone https://github.com/yeti444/YOLO-2025-05
```

Hozzon létre egy virtuális környezetet a projekt mappájában.

```bash
  python -m venv .venv
```

Aktiválja a virtuális környezetet.

```bash
  .\.venv\Scripts\Activate.ps1
```

Telepítse a függőségeket

Pytorch: https://pytorch.org/get-started/locally/

Windows alapú nvidia GPU-val rendelkező rendszer esetén.
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Aztán telepítse a(z) ultralytics csomagot.
```bash
  pip install ultralytics
```

Csomagolja a dasatets.7z fájlt a projekt mellé.

## train.py parancssori argumentumok

Ez a script egy YOLOv8 modellt tanít az Ultralytics könyvtár segítségével. Az alábbi argumentumokat lehet megadni futtatáskor:

### Használat

```bash
python train.py --data <adat.yaml> [--epochs 100] [--batch 16] [--imgsz 640] [--device cuda]
```

### Argumentumok

| Argumentum     | Típus  | Kötelező | Alapértelmezett | Leírás                                                                 |
|----------------|--------|----------|------------------|------------------------------------------------------------------------|
| `--data`       | `str`  | Igen     | –                | Az adathalmazt leíró `.yaml` fájl elérési útja.                      |
| `--epochs`     | `int`  | Nem      | `100`            | Tanítási epoch-ok száma.                                               |
| `--batch`      | `int`  | Nem      | `16`             | Batch méret.                                                           |
| `--imgsz`      | `int`  | Nem      | `640`            | A bemeneti képek mérete (átméretezés történik).                         |
| `--device`     | `str`  | Nem      | `cuda`           | Használt eszköz: `"cuda"` GPU esetén, vagy `"cpu"` csak processzorhoz. |

### Példa

```bash
python train.py --data cardData.yaml --epochs 100 --batch 16 --imgsz 640 --device cuda
```

Ez a parancs elindít egy tanítást a `cardData.yaml` alapján, 100 epochon keresztül, 16-as batch mérettel, 640-es képmérettel, GPU-n.

## detect.py parancssori argumentumok

Ez a script képes képek vagy videók detektálására egy betöltött YOLO modell segítségével. Az alábbi argumentumokat lehet megadni:

### Használat

```bash
python detect.py --model <modell.pt> --source <forrás> [--save] [--save_path <elérési_út>]
```

### Argumentumok

| Argumentum       | Típus    | Kötelező | Alapértelmezett | Leírás                                                                 |
|------------------|----------|----------|------------------|------------------------------------------------------------------------|
| `--model`        | `str`    | Igen     | –                | A betöltendő YOLO modell fájl elérési útja (`.pt`).                   |
| `--source`       | `str`    | Igen     | –                | A bemeneti fájl elérési útja (kép vagy videó).                        |
| `--save`         | flag     | Nem      | `False`          | Ha meg van adva, elmenti a detektált képet vagy videót.               |
| `--save_path`    | `str`    | Nem      | `runs/detect/output` | Mentés helye; ha nincs megadva, alapértelmezett mappába menti.        |

### Példa

```bash
python detect.py --model sajat_modell.pt --source testData/card.jpg --save
```

Ez a parancs betölti a `sajat_modell.pt` modellt, lefuttatja a detektálást a `card.jpg` képen, megjeleníti az eredményt, és elmenti egy időbélyegzett fájlba.

---

```bash
python detect.py --model sajat_modell.pt --source testData/card.mp4 --save --save_path results/
```

Ez a változat videón futtatja le a detektálást, és a mentett fájl a `results/` mappába kerül.
