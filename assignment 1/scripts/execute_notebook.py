"""Execute every notebook cell with a reproducible project working directory.

Setup from the assignment directory:
    python -m pip install -r requirements.txt
    python -m ipykernel install --user --name anime-taste

Run:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/execute_notebook.py

The installed anime-taste kernel must use the environment with the assignment's
dependencies. Each cell has a 300-second execution limit. The notebook reruns
models and overwrites their saved results/figures. This script saves notebook
outputs even if a later cell fails, leaving the error visible for diagnosis.
"""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
# Resolve paths from this script so launching from another directory still
# gives notebook cells the assignment root expected by their relative paths.
path = root / 'anime_taste.ipynb'
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=300, kernel_name='anime-taste',
                        resources={'metadata': {'path': str(root)}})
try:
    client.execute()
finally:
    # Preserve successful early outputs and any failed cell's traceback. Let
    # the execution exception propagate; a partially executed run is not success.
    nbformat.write(nb, path)
print('Notebook executed and saved:', path.name)
