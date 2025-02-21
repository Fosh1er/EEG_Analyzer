EEG Analyzer
Python
PyQt5
PyABF
License

EEG Analyzer is a powerful and user-friendly desktop application for analyzing EEG (Electroencephalography) data. It provides tools for visualizing, processing, and analyzing multi-channel EEG signals, with a focus on detecting and classifying behavioral events.

Features
1. Data Loading and Visualization
Load .abf files containing EEG data.

Visualize multiple EEG channels in a single plot with adjustable vertical spacing.

Interactive zooming and panning for detailed signal inspection.

2. Behavioral Event Detection
Detect behavioral events such as pedal presses (Channel_3) and feeder approaches (Channel_5).

Classify events into categories:

СН (Самостоятельное Нажатие): Independent pedal press.

КН (Кормушечное Нажатие): Pedal press followed by a feeder approach.

СК (Самостоятельный Кормушка): Feeder approach without a pedal press.

3. Spectral Analysis
Perform spectral analysis using Welch's method and FFT.

Visualize power spectral density (PSD) and frequency band powers.

Display results in interactive plots and tables.

4. Data Segmentation
Split EEG data into smaller segments by time or number of parts.

Save segmented data for further analysis.

5. Advanced Filtering
Apply bandpass, lowpass, and bandstop filters to clean EEG signals.

Customizable filter parameters for different types of signals.

6. Export Results
Save analyzed data and results in multiple formats:

HDF5 for structured data storage.

MAT for compatibility with MATLAB.

JSON for easy sharing and integration.

Installation
Prerequisites
Python 3.8 or higher.

Required Python packages: PyQt5, pyabf, numpy, scipy, pyqtgraph, h5py.

Steps
Clone the repository:

bash
Copy
git clone https://github.com/your-username/eeg-analyzer.git
cd eeg-analyzer
Install the required packages:

bash
Copy
pip install -r requirements.txt
Run the application:

bash
Copy
python main.py
Usage
Loading Data
Click Open in the toolbar to load an .abf file.

The EEG signals will be displayed in the main window.

Analyzing Behavior
Click Behavior to detect and classify behavioral events.

Events will be highlighted on the plot, and a new window will open for detailed analysis.

Spectral Analysis
Select a segment of data or a behavioral event.

Click Spectrum to perform spectral analysis.

View the results in interactive plots and tables.

Saving Results
Use the Save button to export data or analysis results in your preferred format.

Screenshots
Main Window
Main Window

Behavioral Event Analysis
Behavioral Analysis

Spectral Analysis
Spectral Analysis

Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:

Fork the repository.

Create a new branch for your feature or bugfix.

Submit a pull request with a detailed description of your changes.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
PyABF: For providing an easy-to-use interface for working with ABF files.

PyQtGraph: For creating fast and interactive plots.

SciPy: For signal processing and spectral analysis tools.

Contact
For questions or feedback, feel free to reach out:

GitHub: your-username

Email: your.email@example.com
