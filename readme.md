## SET UP

1. Clone the repository:

   ```bash
   git clone https://github.com/username/repository.git
   ```

2. Set up virtual env :

   ```bash
   cd flask-server
   python -m venv .venv
   ```

3. Activate virtual env :

   ```bash
   source .venv/scripts/activate
   ```

4. Intall library

   ```bash
   pip install -r requirements.txt
   ```

5. Run server

   ```bash
   python main.py
   ```

## STRUCTURE

```bash
    ROOT/
    ├── flask-server/
    │   ├── dataset/          # Directory for .csv datasets
    │   ├── data.py           # Python script for handling logic
    │   └── main.py           # Flask server application
    │
    └── model/                # Directory for ML models
```
