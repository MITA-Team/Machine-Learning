# HOW TO USE ?

## Clone This Repository

## Install Requirements Library

Before that, you must install pip.

```
sudo apt install pip
```

Install requirements

```
python3 -m pip install -r requirements.txt
```

_Kalo gabisa, install manual aja. pip install <nama_package>_

## Buat service account key

```
sudo nano serviceAccountKey.json
```
_Service account key ada di Repo service account, namanya bucket service account_

## Build ML Model

### Using ```model.py``` to create model

```
python3 model.py
```

## Serve

### Using ```app.py``` to start server

```
python3 app.py
```
