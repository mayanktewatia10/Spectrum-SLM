import json
import os

def create_kaggle_notebook(output_path="spectrum_slm_kaggle_phase2.ipynb"):
    notebook = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            },
            "kaggle": {
                "accelerator": "gpu",
                "isInternetEnabled": True,
                "language": "python",
                "isGpuEnabled": True
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Spectrum-SLM 3-Phase Execution\n",
                    "Turn on **GPU (T4x2)** and **Internet** in Kaggle settings. Then hit **Run All**."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Clone repository\n",
                    "!git clone https://github.com/31ASHISH/Spectrum-SLM.git /kaggle/working/Spectrum-SLM\n",
                    "%cd /kaggle/working/Spectrum-SLM/SDR_Data"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. Install dependencies\n",
                    "!pip install -q streamlit scikit-learn\n",
                    "!npm install -g localtunnel"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3. Connect Kaggle Input dataset to Config and run 3-phase training\n",
                    "# Assumes you clicked \"Add Data\" and selected your uploaded Dataset.\n",
                    "import os\n",
                    "dataset_path = \"/kaggle/input\"\n",
                    "\n",
                    "if os.path.exists(dataset_path):\n",
                    "    with open('config.py', 'a') as f:\n",
                    "        f.write(f'\\n\\nkaggle_override(\"{dataset_path}\")\\n')\n",
                    "    \n",
                    "    print(\"Starting end-to-end training pipeline...\")\n",
                    "    !python training/run_3_phases.py\n",
                    "else:\n",
                    "    print(f\"Warning: Dataset not found. Did you Add Data in Kaggle?\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 4. Launch Streamlit app\n",
                    "import subprocess, time, re\n",
                    "st_proc = subprocess.Popen(['streamlit', 'run', 'app_phase2.py', '--server.port', '8501', '--server.headless', 'true'])\n",
                    "time.sleep(5)\n",
                    "\n",
                    "print(\"Starting localtunnel...\")\n",
                    "lt = subprocess.Popen(['lt', '--port', '8501'], stdout=subprocess.PIPE, text=True)\n",
                    "\n",
                    "for _ in range(15):\n",
                    "    line = lt.stdout.readline()\n",
                    "    m = re.search(r'https?://[\\w\\-.]+\\.loca\\.lt', line)\n",
                    "    if m:\n",
                    "        print(f'\\n🚀 STREAMLIT LIVE AT: {m.group(0)}')\n",
                    "        print(f'🔑 Password required: {subprocess.getoutput(\"curl -s ifconfig.me\")}\\n')\n",
                    "        break\n",
                    "    time.sleep(1)"
                ]
            }
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Kaggle Notebook successfully generated at: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    create_kaggle_notebook()
