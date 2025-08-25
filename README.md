# Bangalore Urban Flood Extent Prediction

This project aims to predict flood depth using satellite imagery and a CNN. It is structured for enterprise-level scalability, maintainability, and includes a professional logging system.

## Setup

1.  **Create Project:** Run `python setup_project.py` to generate the structure.
2.  **Navigate:** `cd Bangalore-Urban-Flood-Extent-Prediction`
3.  **Set up Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
4.  **Run Pipeline:** `python main.py`

## Logging

This project uses a centralized logging system.
- Logs are saved to the `/logs` directory.
- Different pipeline stages have their own log files (e.g., `data_pipeline.log`, `model_pipeline.log`).
- Logs are also printed to the console for real-time monitoring.