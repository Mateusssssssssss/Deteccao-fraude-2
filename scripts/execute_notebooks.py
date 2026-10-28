"""Executa notebooks de leitura dos artefatos, sem repetir treinamento."""

import argparse
import os

import nbformat
from nbclient import NotebookClient

from fraud_detection.data import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", help="Nome opcional, por exemplo 01_eda.ipynb")
    args = parser.parse_args()

    # Arquivos de comunicação/cache do kernel ficam no projeto, sem depender de AppData.
    for variable, folder in {
        "JUPYTER_RUNTIME_DIR": "jupyter",
        "IPYTHONDIR": "ipython",
        "MPLCONFIGDIR": "matplotlib",
    }.items():
        path = ROOT / ".cache" / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"

    # Só lê relatórios calculados: executar notebook não dispara os experimentos.
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if args.notebook:
        paths = [p for p in paths if p.name == args.notebook]
        if not paths:
            parser.error("Notebook não encontrado na pasta notebooks.")
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        nbformat.write(notebook, path)
        print(f"Executado: {path.name}", flush=True)


if __name__ == "__main__":
    main()
