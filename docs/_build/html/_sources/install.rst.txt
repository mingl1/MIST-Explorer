Installation
=====

To install MIST-Explorer, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/MIST-Explorer/MIST-Explorer.git
    ```

2. Navigate to the project directory:
    ```bash
    cd MIST-Explorer
    ```

3. Install the required dependencies:
    ```bash
    uv sync
    ```

Alternatively, prebuilt executable binaries can be downloaded directly from the GitHub Actions artifacts page for the latest successful workflow run.

To start the application, run:
```bash
uv run main.py
```