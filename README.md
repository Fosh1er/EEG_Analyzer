# EEG Analyzer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-green)
![PyABF](https://img.shields.io/badge/PyABF-2.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

**EEG Analyzer** is a desktop application designed for analyzing EEG (Electroencephalography) data. It provides tools for visualizing, processing, and classifying behavioral events in multi-channel EEG signals. The application is built with Python and uses PyQt5 for the user interface, PyABF for loading ABF files, and SciPy for signal processing.

---

## Features

### 1. **Data Loading and Visualization**
   - Load `.abf` files containing EEG data.
   - Visualize multiple EEG channels in a single plot with adjustable vertical spacing.
   - Interactive zooming and panning for detailed signal inspection.

### 2. **Behavioral Event Detection**
   - Detect behavioral events such as pedal presses (`Channel_3`) and feeder approaches (`Channel_5`).
   - Classify events into categories:
     - **СН (Самостоятельное Нажатие)**: Independent pedal press.
     - **КН (Кормушечное Нажатие)**: Pedal press followed by a feeder approach.
     - **СК (Самостоятельный Кормушка)**: Feeder approach without a pedal press.

### 3. **Spectral Analysis**
   - Perform spectral analysis using **Welch's method** and **FFT**.
   - Visualize power spectral density (PSD) and frequency band powers.
   - Display results in interactive plots and tables.

### 4. **Data Segmentation**
   - Split EEG data into smaller segments by time or number of parts.
   - Save segmented data for further analysis.

### 5. **Advanced Filtering**
   - Apply bandpass, lowpass, and bandstop filters to clean EEG signals.
   - Customizable filter parameters for different types of signals.

### 6. **Export Results**
   - Save analyzed data and results in multiple formats:
     - **HDF5** for structured data storage.
     - **MAT** for compatibility with MATLAB.
     - **JSON** for easy sharing and integration.

---

## Installation

### Prerequisites
- Python 3.8 or higher.
- Required Python packages: `PyQt5`, `pyabf`, `numpy`, `scipy`, `pyqtgraph`, `h5py`.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Fosh1er/EEG_Analyzer.git
   cd eeg-analyzer
   ```
2. Install the required packages:
  ```bash
  pip install -r requirements.txt
  ```
3. Run the application:
  ```bash
  python main.py
  ```
