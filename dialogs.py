from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QFileDialog,
    QPushButton, QComboBox, QDoubleSpinBox, QMessageBox, QTabWidget, QWidget, QSpinBox
)
import numpy as np
from gui.plots import create_psd_plot
import pyqtgraph as pg
import json
import h5py
from logger import logger
from scipy.signal import welch  # Добавить в начало файла
from core.signal_processing import calculate_band_powers  # Добавить импорт
import re
import os

class AnalysisDialog(QDialog):
    def __init__(self, container, parent=None):
        super().__init__(parent)
        self.container = container
        self.init_ui()
        self.analyze_data()  # Теперь метод существует
        self.plots = {}  # Словарь для хранения графиков

    def init_ui(self):
        self.setWindowTitle("Полный спектральный анализ")
        self.setMinimumSize(1200, 800)

        layout = QVBoxLayout()
        self.tabs = QTabWidget()

        for channel in self.container.data:
            self._add_channel_tab(channel)

        save_btn = QPushButton("Сохранить результаты")
        save_btn.clicked.connect(self.save_results)

        layout.addWidget(self.tabs)
        layout.addWidget(save_btn)
        self.setLayout(layout)

    def analyze_data(self):
        try:
            self.results = {}
            for channel in self.container.data:
                data = self.container.data[channel]
                fs = self.container.sample_rate

                # Вычисление PSD
                freq, psd = welch(data, fs, nperseg=1024)
                self.results[channel] = {
                    "freq": freq,
                    "psd": psd,
                    "bands": calculate_band_powers(freq, psd)
                }
            #self._update_plots()

        except Exception as e:
            logger.log_error(e, "SPECTRAL_ANALYSIS_FAILED")
            QMessageBox.critical(
                self,
                "Ошибка анализа",
                f"Народный анализ не удался:\n{str(e)}"
            )

    def _update_plots(self):
        for channel in self.results:
            # Получаем существующие графики или создаем новые
            plot_widget = self.findChild(pg.PlotWidget, f"psd_plot_{channel}")

            if not plot_widget:
                plot_widget = create_psd_plot(
                    self.results[channel]['freq'],
                    self.results[channel]['psd'],
                    channel
                )
                plot_widget.setObjectName(f"psd_plot_{channel}")
                self.layout().addWidget(plot_widget)

            # Принудительно обновляем диапазоны
            plot_widget.setBackground('w')  # Белый фон для народной ясности
            plot_widget.setXRange(0, 40, padding=0)
            plot_widget.setYRange(0, 0.1, padding=0)

    def _add_channel_tab(self, channel):
        tab = QWidget()
        layout = QVBoxLayout()

        plot_widget = pg.GraphicsLayoutWidget()
        plot_widget.setBackground('w')

        # График PSD
        p1 = plot_widget.addPlot(title="Спектральная плотность мощности")
        p1.setXRange(0, 40)
        p1.setYRange(0, 0.07)

        # График диапазонов
        p2 = plot_widget.addPlot(title="Мощность частотных диапазонов")
        p2.setYRange(0, 0.07)
        try:
            data = self.container.data[channel]
            # Автоподбор nperseg
            nperseg = min(1024, len(data))
            freq, psd = welch(data, self.container.sample_rate, nperseg=nperseg)
            band_powers = calculate_band_powers(freq, psd)

            # Отрисовка данных
            p1.plot(freq, psd, pen='b')

            # Столбчатая диаграмма
            x = np.arange(len(band_powers))
            bars = pg.BarGraphItem(x=x, height=list(band_powers.values()), width=0.6, brush='r')
            p2.addItem(bars)
            p2.getAxis('bottom').setTicks([[(i, band) for i, band in enumerate(band_powers.keys())]])

        except Exception as e:
            layout.addWidget(QLabel(f"Ошибка: {str(e)}"))

        layout.addWidget(plot_widget)
        tab.setLayout(layout)
        self.tabs.addTab(tab, channel)

    def save_results(self):  # Убрать аргумент results
        if not hasattr(self, 'results'):
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Full Analysis Results", "",
            "JSON Full (*.json);;HDF5 Full (*.h5)"
        )

        if not path:
            return

        try:
            if path.endswith('.json'):
                full_data = {}
                for channel in self.current_container.data:
                    data = self.current_container.data[channel]
                    fs = self.current_container.sample_rate
                    freq, psd = welch(data, fs, nperseg=256)
                    full_data[channel] = {
                        "freq": freq.tolist(),
                        "psd": psd.tolist(),
                        "bands": results.get(channel, {})
                    }
                with open(path, 'w') as f:
                    json.dump(full_data, f, indent=2)

            elif path.endswith('.h5'):
                with h5py.File(path, 'w') as f:
                    for channel in self.current_container.data:
                        data = self.current_container.data[channel]
                        fs = self.current_container.sample_rate
                        freq, psd = welch(data, fs, nperseg=256)

                        grp = f.create_group(channel)
                        grp.create_dataset("freq", data=freq)
                        grp.create_dataset("psd", data=psd)
                    for band, power in results[channel].items():
                            grp.attrs[band] = power

            QMessageBox.information(self, "Success", "Full results saved successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {str(e)}")
        pass


class SplitDialog(QDialog):
    def __init__(self, parent=None):  # Корректный конструктор
        super().__init__(parent)  # Передаем parent в QDialog
        self.setWindowTitle("Разделение данных")
        self.method = "time"
        self.params = {}
        self.init_ui()

    def show_split_dialog(self):
        dialog = SplitDialog(parent=self)  # Передаем self как родительский виджет
        dialog.exec_()
    def init_ui(self):
        self.setWindowTitle("Разделение данных")
        layout = QVBoxLayout()

        method_box = QComboBox()
        method_box.addItems(["По времени", "На части"])
        method_box.currentIndexChanged.connect(self._update_method)

        self.time_widget = QWidget()
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Длительность (сек):"))
        self.time_input = QDoubleSpinBox()
        self.time_input.setRange(0.1, 3600)
        self.time_input.setValue(1.0)
        time_layout.addWidget(self.time_input)
        self.time_widget.setLayout(time_layout)

        self.parts_widget = QWidget()
        parts_layout = QHBoxLayout()
        parts_layout.addWidget(QLabel("Количество частей:"))
        self.parts_input = QSpinBox()
        self.parts_input.setRange(1, 100)
        self.parts_input.setValue(2)
        parts_layout.addWidget(self.parts_input)
        self.parts_widget.setLayout(parts_layout)

        confirm_btn = QPushButton("Подтвердить")
        confirm_btn.clicked.connect(self._confirm)

        layout.addWidget(QLabel("Метод разделения:"))
        layout.addWidget(method_box)
        layout.addWidget(self.time_widget)
        layout.addWidget(self.parts_widget)
        layout.addWidget(confirm_btn)

        self._update_method(0)
        self.setLayout(layout)

    def _update_method(self, index):
        self.method = "time" if index == 0 else "parts"
        self.time_widget.setVisible(index == 0)
        self.parts_widget.setVisible(index == 1)

    def _confirm(self):
        if self.method == "time":
            self.params = {"duration": self.time_input.value()}
        else:
            self.params = {"parts": self.parts_input.value()}
        self.accept()


class ManualRangeDialog(QDialog):
    def __init__(self, container, parent=None):  # Явное указание типа
        super().__init__(parent)
        self.container = container
        self.start_time = 0.0
        self.end_time = 0.0
        self.init_ui()
        self.container = container
        self.analyze_data()  # Теперь метод существует


    def init_ui(self):
        self.setWindowTitle("Анализ диапазона")
        layout = QVBoxLayout()

        total_duration = len(self.container.time) / self.container.sample_rate

        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Начало (сек):"))
        self.start_input = QDoubleSpinBox()
        self.start_input.setRange(0.0, total_duration)
        start_layout.addWidget(self.start_input)

        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("Конец (сек):"))
        self.end_input = QDoubleSpinBox()
        self.end_input.setRange(0.0, total_duration)
        self.end_input.setValue(min(total_duration, 5.0))
        end_layout.addWidget(self.end_input)

        analyze_btn = QPushButton("Анализировать")
        analyze_btn.clicked.connect(self._validate_range)

        layout.addLayout(start_layout)
        layout.addLayout(end_layout)
        layout.addWidget(analyze_btn)
        self.setLayout(layout)

    def analyze_data(self):
        """Проведение народного спектрального анализа"""
        try:
            self.results = {}

            for channel in self.container.data:
                data = self.container.data[channel]
                fs = self.container.sample_rate

                # Вычисление PSD
                freq, psd = welch(data, fs, nperseg=1024)
                self.results[channel] = {
                    "freq": freq,
                    "psd": psd,
                    "bands": calculate_band_powers(freq, psd)
                }

            #self._update_plots()

        except Exception as e:
            logger.log_error(e, "SPECTRAL_ANALYSIS_FAILED")
            QMessageBox.critical(
                self,
                "Ошибка анализа",
                f"Народный анализ не удался:\n{str(e)}"
            )

    def _update_plots(self):
        """Обновление графиков для трудового народа"""
        for channel in self.results:
            # Логика обновления графиков
            pass
    def _validate_range(self):
        try:
            self.start_time = float(self.start_input.value())
            self.end_time = float(self.end_input.value())

            if self.start_time >= self.end_time:
                raise ValueError("Некорректный временной диапазон")

            if (self.end_time - self.start_time) % 5 != 0:
                raise ValueError("Диапазон должен быть кратен 5 секундам")

            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))


class StagedAnalysisDialog(QDialog):
    def __init__(self, container, parent=None):
        super().__init__(parent)
        self.container = container
        self.results = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Этапный анализ")
        self.setMinimumSize(1600, 900)

        layout = QVBoxLayout()
        self.tabs = QTabWidget()

        for ch in self.container.data:
            self._add_channel_tab(ch)

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def _add_channel_tab(self, channel):
        tab = QWidget()
        layout = QVBoxLayout()

        plot = pg.PlotWidget(title=f"Динамика спектров - {channel}")
        # Логика построения графиков этапного анализа

        layout.addWidget(plot)
        tab.setLayout(layout)
        self.tabs.addTab(tab, channel)