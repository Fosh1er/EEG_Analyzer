import sys
import numpy as np
import pyabf
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog,
    QMessageBox, QHBoxLayout, QInputDialog, QTabWidget, QToolBar,
    QAction, QTableWidget, QTableWidgetItem, QDockWidget, QScrollArea, QDialog
)

import h5py
from scipy.io import savemat
from scipy.signal import butter, filtfilt
import json
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QIcon
from scipy.signal import welch
from scipy.fft import fft, fftfreq
from scipy.integrate import trapezoid

# Константы
VERTICAL_SPACING = 25
FREQ_BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (12, 30),
    "Gamma": (30, 45)
}


class BehaviorEvent:
    def __init__(self, start_time, end_time, event_type, data, time, sample_rate):
        self.start = start_time
        self.end = end_time
        self.type = event_type
        self.data = data
        self.time = time
        self.sample_rate = sample_rate


class BehaviorAnalyzer(QDialog):
    def __init__(self, sample_rate, events, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Behavior Events Analysis")
        self.setGeometry(200, 200, 1200, 800)
        self.events = events
        self.sample_rate = sample_rate
        layout = QVBoxLayout()
        self.tabs = QTabWidget()

        for event in events:
            tab = QWidget()
            tab_layout = QVBoxLayout()

            # График для всех каналов
            plot = pg.PlotWidget(title=f"{event.type} Event ({event.start:.1f}-{event.end:.1f}s)")
            for channel in event.data:
                plot.plot(event.time, event.data[channel], name=channel)

            # Кнопка спектрального анализа
            analyze_btn = QPushButton("Spectral Analysis")
            analyze_btn.clicked.connect(lambda _, e=event: self.show_spectral_analysis(e))

            tab_layout.addWidget(plot)
            tab_layout.addWidget(analyze_btn)
            tab.setLayout(tab_layout)
            self.tabs.addTab(tab, event.type)

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def show_spectral_analysis(self, event):
        spectrum_window = QDialog(self)
        spectrum_window.setWindowTitle(f"Spectral Analysis - {event.type}")
        layout = QVBoxLayout()
        scroll = QScrollArea()
        content = QWidget()
        scroll_layout = QVBoxLayout(content)

        # Создаем вкладки для каждого канала
        channel_tabs = QTabWidget()
        for channel in event.data:
            channel_tab = QWidget()
            channel_layout = QVBoxLayout()

            try:
                data = event.data[channel]
                fs = event.sample_rate
                data_clean = data - np.mean(data)  # Удаление DC-составляющей

                # Welch метод
                freq_welch, psd = welch(data_clean, fs, nperseg=1024)

                # FFT анализ
                N = len(data_clean)
                yf = fft(data_clean)
                xf = fftfreq(N, 1 / fs)[:N // 2]

                # Создаем графики
                splitter = pg.GraphicsLayoutWidget()

                # График PSD
                p1 = splitter.addPlot(title=f"{channel} - Power Spectral Density (Welch)")
                p1.plot(freq_welch, psd, pen='b')
                p1.setLabel('left', 'Power', units='µV²/Hz')
                p1.setLabel('bottom', 'Frequency', units='Hz')

                # Гистограмма частотных диапазонов
                p2 = splitter.addPlot(col=1, title="Frequency Band Powers")
                band_powers = {}
                for band, (low, high) in FREQ_BANDS.items():
                    mask = (freq_welch >= low) & (freq_welch <= high)
                    band_powers[band] = trapezoid(psd[mask], freq_welch[mask])

                x = np.arange(len(FREQ_BANDS))
                bars = pg.BarGraphItem(x=x, height=list(band_powers.values()),
                                      width=0.6, brush='g')
                p2.addItem(bars)
                p2.getAxis('bottom').setTicks([[(i, band) for i, band in enumerate(FREQ_BANDS)]])

                channel_layout.addWidget(splitter)

                # Таблица результатов
                table = QTableWidget()
                table.setColumnCount(2)
                table.setRowCount(len(FREQ_BANDS))
                table.setHorizontalHeaderLabels(["Band", "Power"])

                for row, (band, power) in enumerate(band_powers.items()):
                    table.setItem(row, 0, QTableWidgetItem(band))
                    table.setItem(row, 1, QTableWidgetItem(f"{power:.2f} µV²/Hz"))

                channel_layout.addWidget(table)

            except Exception as e:
                print(f"Error in spectral analysis for {channel}: {str(e)}")

            channel_tab.setLayout(channel_layout)
            channel_tabs.addTab(channel_tab, channel)

        scroll_layout.addWidget(channel_tabs)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        spectrum_window.setLayout(layout)
        spectrum_window.exec_()
class DataContainer:
    def __init__(self, data, time, sample_rate, title):
        self.data = data
        self.time = time
        self.sample_rate = sample_rate
        self.title = title


class EEGProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.abf = pyabf.ABF(file_path)
        self.sample_rate = self.abf.dataRate
        self.data = self._load_data()
        self.time = self._create_time_array()
        self.fs = self.abf.dataRate

    def _load_data(self):
        data = {}
        for channel in range(self.abf.channelCount):
            self.abf.setSweep(sweepNumber=0, channel=channel)
            data[f"Channel_{channel}"] = self.abf.sweepY + channel * VERTICAL_SPACING
        return data

    def _create_time_array(self):
        return np.arange(0, len(self.data["Channel_0"]) / self.sample_rate, 1 / self.sample_rate)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_containers = []
        self.current_container = None
        self.tabs = QTabWidget()
        self.init_ui()
        self.init_settings()

    def init_settings(self):
        self.settings = QSettings("EEGApp", "MainWindow")
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)

    def init_ui(self):
        self.setWindowTitle("EEG Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        actions = [
            ("Open", "document-open", self.load_file),
            ("Split", "document-split", self.show_split_dialog),
            ("Spectrum", "graph", self.show_spectral_dialog),
            ("Save", "document-save", self.save_data)
        ]
        actions.append(("Behavior", "animal", self.analyze_behavior))


        for name, icon, callback in actions:
            action = QAction(QIcon.fromTheme(icon), name, self)
            action.triggered.connect(callback)
            toolbar.addAction(action)

        self.setCentralWidget(self.tabs)
        self.tabs.currentChanged.connect(self.on_tab_changed)


    def filter_pedal(self, data, fs):
        """Фильтр для педали"""
        return self.butter_bandpass_filter(data, lowcut=1, highcut=10, fs=fs, order=3)

    def filter_feeder(self, data, fs):
        """Фильтр для кормушки"""
        return self.butter_bandpass_filter(data, lowcut=5, highcut=15, fs=fs, order=3)

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def analyze_behavior(self):
        if not self.current_container:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return

        try:
            # Получаем данные каналов
            pedal_raw = self.current_container.data["Channel_3"]  # Педаль (4-й канал)
            feeder_raw = self.current_container.data["Channel_5"]  # Кормушка (6-й канал)
            time = self.current_container.time
            fs = self.current_container.sample_rate
        except KeyError:
            QMessageBox.critical(self, "Error", "Required channels (4 and 6) not found!")
            return

        # Фильтрация данных
        pedal_data = self.filter_pedal(pedal_raw, fs)
        feeder_data = self.filter_feeder(feeder_raw, fs)

        # Детекция событий
        pedal_events = self.detect_events(pedal_data, time, fs,
                                          threshold=0.3,
                                          min_duration=0.1,
                                          merge_interval=1.0)

        feeder_events = self.detect_events(feeder_data, time, fs,
                                           threshold=0.3,
                                           min_duration=0.1,
                                           merge_interval=1.0)

        print(
            f"Detected {len(pedal_events)} pedal events and {len(feeder_events)} feeder events")  # Отладочное сообщение

        # Классификация событий
        classified_events = self.classify_events(pedal_events, feeder_events, time, fs)

        print(f"Classified {len(classified_events)} behavior events")  # Отладочное сообщение

        # Создаем объекты событий
        self.behavior_events = []
        for event in classified_events:
            start_idx = np.searchsorted(time, event['start'])
            end_idx = np.searchsorted(time, event['end'])

            # Включаем все каналы, кроме Channel_5 (кормушка)
            event_data = {
                ch: self.current_container.data[ch][start_idx:end_idx]
                for ch in self.current_container.data
                if ch != 'Channel_5'  # Исключаем только канал кормушки
            }

            self.behavior_events.append(BehaviorEvent(
                start_time=event['start'],
                end_time=event['end'],
                event_type=event['type'],
                data=event_data,
                time=time[start_idx:end_idx],
                sample_rate=fs
            ))
            # Визуализация
        self.plot_events_on_graph()
        self.show_behavior_events()

    def detect_events(self, data, time, fs, threshold, min_duration, merge_interval):
        """Детекция событий с продвинутыми параметрами"""
        events = []
        in_event = False
        start_idx = 0
        event_buffer = []

        for i, val in enumerate(data):
            if val > threshold and not in_event:
                start_idx = i
                in_event = True
                event_buffer = []
            elif val <= threshold and in_event:
                event_buffer.append(i)
                if len(event_buffer) > fs * merge_interval:
                    end_idx = event_buffer[0]
                    duration = (end_idx - start_idx) / fs
                    if duration >= min_duration:
                        events.append((start_idx, end_idx))
                    in_event = False
            elif in_event:
                event_buffer = []

        print(f"Raw events detected: {events}")  # Отладочное сообщение
        return [(time[start], time[end]) for start, end in events]

    def classify_events(self, pedal_events, feeder_events, time, fs):
        """Точная классификация событий"""
        classified = []
        lookback = 3.5  # секунд для поиска связанных событий

        # Анализ нажатий педали
        for p_start, p_end in pedal_events:
            # Поиск подходов к кормушке в течение 5 секунд после нажатия
            feeder_in_window = [f for f in feeder_events
                                if p_start <= f[0] <= p_start + lookback]

            event_type = "КН" if feeder_in_window else "СН"
            event_end = p_end + lookback if feeder_in_window else p_end + 1.0

            classified.append({
                'start': p_start,
                'end': event_end,
                'type': event_type,
                'related_feeder': feeder_in_window
            })

        # Анализ подходов к кормушке без педали
        for f_start, f_end in feeder_events:
            # Поиск нажатий педали за 5 секунд до подхода
            pedal_before = [p for p in pedal_events
                            if f_start - lookback <= p[0] <= f_start]

            if not pedal_before:
                classified.append({
                    'start': f_start - 1.0,
                    'end': f_end,
                    'type': "СК",
                    'related_pedal': []
                })

        print(f"Classified events: {classified}")  # Отладочное сообщение
        return self.merge_overlapping_events(classified, fs)

    def merge_overlapping_events(self, events, fs):
        """Объединение пересекающихся событий"""
        merged = []
        for event in sorted(events, key=lambda x: x['start']):
            if not merged:
                merged.append(event)
            else:
                last = merged[-1]
                if event['start'] <= last['end'] + 0.2:  # Уменьшили интервал до 0.2 секунд
                    last['end'] = max(last['end'], event['end'])
                    last['type'] = f"Смешанное ({last['type']}+{event['type']})"
                else:
                    merged.append(event)
        return merged

    def plot_events_on_graph(self):
        """Визуализация событий на основном графике"""
        colors = {'СН': '#FF0000', 'КН': '#00FF00', 'СК': '#0000FF'}

        for tab_idx in range(self.tabs.count()):
            tab = self.tabs.widget(tab_idx)
            plot = tab.findChild(pg.PlotWidget)

            for event in self.behavior_events:
                region = pg.LinearRegionItem(
                    values=[event.start, event.end],
                    brush=pg.mkBrush(colors.get(event.type, '#AAAAAA80')),
                    movable=False
                )
                plot.addItem(region)
    def find_events(self, data, threshold):
        """Нахождение событий по превышению порога"""
        events = []
        in_event = False
        for i, value in enumerate(data):
            if value > threshold and not in_event:
                start = i
                in_event = True
            elif value <= threshold and in_event:
                end = i
                events.append((start, end))
                in_event = False
        return events

    def create_behavior_epochs(self, pedal_events, feeder_events):
        """Создание эпох поведения с метками"""
        epochs = []
        sample_rate = self.current_container.sample_rate
        time = self.current_container.time
        data_length = len(time)  # Общая длина данных

        # Анализ событий педали
        for start, end in pedal_events:
            # Добавляем 5 секунд до и после
            start_idx = max(0, start - int(5 * sample_rate))  # 5 секунд до
            end_idx = min(data_length - 1, end + int(5 * sample_rate))  # 5 секунд после

            # Проверяем, что индексы корректны
            if start_idx >= end_idx:
                continue  # Пропускаем некорректные события

            # Проверяем наличие подхода к кормушке
            feeder_in_window = any(start <= f_start <= end for f_start, _ in feeder_events)

            event_type = "КН" if feeder_in_window else "СН"

            epochs.append(BehaviorEvent(
                start_time=time[start_idx],
                end_time=time[end_idx],
                event_type=event_type,
                data={ch: data[start_idx:end_idx] for ch, data in self.current_container.data.items()},
                time=time[start_idx:end_idx],
                sample_rate=sample_rate
            ))

        # Анализ событий кормушки без педали
        for start, end in feeder_events:
            if not any(p_start <= start <= p_end for p_start, p_end in pedal_events):
                start_idx = max(0, start - int(5 * sample_rate))  # 5 секунд до
                end_idx = min(data_length - 1, end + int(5 * sample_rate))  # 5 секунд после

                # Проверяем, что индексы корректны
                if start_idx >= end_idx:
                    continue  # Пропускаем некорректные события

                epochs.append(BehaviorEvent(
                    start_time=time[start_idx],
                    end_time=time[end_idx],
                    event_type="СК",
                    data={ch: data[start_idx:end_idx] for ch, data in self.current_container.data.items()},
                    time=time[start_idx:end_idx],
                    sample_rate=sample_rate
                ))

        return epochs

    def show_behavior_events(self):
        """Показ окна с обнаруженными событиями"""
        self.behavior_analyzer = BehaviorAnalyzer(events=self.behavior_events, sample_rate=200)
        self.behavior_analyzer.show()


    def remove_close_events(self, events, min_interval):
        """
        Удаление близких событий
        :param events: Список временных меток событий
        :param min_interval: Минимальный интервал между событиями (в сэмплах)
        :return: Отфильтрованный список событий
        """
        filtered_events = [events[0]]
        for event in events[1:]:
            if event - filtered_events[-1] >= min_interval:
                filtered_events.append(event)
        return filtered_events

    def filter_data(self, data, sample_rate, channel_type):
        """
        Фильтрация данных в зависимости от типа канала
        """
        if channel_type == "pedal":
            # Фильтр НЧ для педали
            return self.butter_lowpass_filter(data, 5, sample_rate, order=2)
        elif channel_type == "feeder":
            # Комбинированный фильтр для кормушки
            filtered = self.moving_average(data, window=5)
            return self.butter_bandstop_filter(filtered, [45, 55], sample_rate)
    def create_epoch_container(self, epoch):
        # Создание DataContainer с меткой
        pass
    def find_behavior_epochs(self, pedal_events, feeder_events, sample_rate):
        """
        Поиск поведенческих паттернов
        """
        epochs = []
        lookahead = 5 * sample_rate  # 5 секунд в сэмплах

        # Анализ нажатий педали
        for pedal_start in pedal_events:
            feeder_in_window = [f for f in feeder_events
                                if pedal_start <= f <= pedal_start + lookahead]

            if not feeder_in_window:
                label = "СН"
            else:
                label = "КН"

            epochs.append(self.create_epoch(pedal_start, lookahead, label, sample_rate))

        # Анализ подходов к кормушке без педали
        for feeder_start in feeder_events:
            if not any(abs(feeder_start - p) <= lookahead for p in pedal_events):
                epochs.append(self.create_epoch(feeder_start, lookahead, "СК", sample_rate))

        return epochs

    def create_epoch(self, event_start, lookahead, label, sample_rate):
        """
        Создание эпохи данных с меткой
        """
        start = max(0, event_start - 5 * sample_rate)  # 5 сек до
        end = event_start + lookahead

        return {
            'start_sample': start,
            'end_sample': end,
            'label': label,
            'duration': (end - start) / sample_rate
        }

    def butter_lowpass_filter(self, data, cutoff, fs, order=2):
        """
        Фильтр Баттерворта нижних частот
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    from scipy.signal import butter, filtfilt

    def butter_bandstop_filter(self, data, cutoff, fs, order=2):
        """
        Полосовой заграждающий фильтр (bandstop filter)
        :param data: Входной сигнал
        :param cutoff: Список из двух частот [low, high] — границы полосы подавления
        :param fs: Частота дискретизации сигнала
        :param order: Порядок фильтра
        :return: Отфильтрованный сигнал
        """
        nyquist = 0.5 * fs
        low = cutoff[0] / nyquist
        high = cutoff[1] / nyquist

        # Создание фильтра
        b, a = butter(order, [low, high], btype='bandstop')

        # Применение фильтра с нулевой фазовой задержкой
        return filtfilt(b, a, data)
    def moving_average(self, data, window=3):
        """
        Скользящее среднее для подавления высокочастотных шумов
        """
        return np.convolve(data, np.ones(window) / window, mode='same')
    def on_tab_changed(self, index):
        if index >= 0:
            self.current_container = self.data_containers[index]

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open ABF File", "", "ABF Files (*.DAT)")
        if file_path:
            try:
                processor = EEGProcessor(file_path)
                container = DataContainer(
                    processor.data,
                    processor.time,
                    processor.sample_rate,
                    "Original Data"
                )
                self.data_containers.append(container)
                self.plot_data(container)
                self.tabs.setCurrentIndex(len(self.data_containers) - 1)
                QMessageBox.information(self, "Success", "File loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Load error: {str(e)}")

    def plot_data(self, container):
        tab = QWidget()
        layout = QVBoxLayout()
        plot = pg.PlotWidget()

        for channel in container.data:
            plot.plot(container.time, container.data[channel], name=channel)

        layout.addWidget(plot)
        tab.setLayout(layout)
        self.tabs.addTab(tab, container.title)

    def show_split_dialog(self):
        if not self.current_container:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return

        method, ok = QInputDialog.getItem(self, "Split Method",
                                          "Choose method:", ["By Time", "By Parts"], 0, False)

        if ok:
            if method == "By Time":
                self.split_by_time()
            else:
                self.split_by_parts()

    def split_by_time(self):
        duration, ok = QInputDialog.getDouble(self, "Split by Time",
                                              "Enter duration (seconds):", 1.0, 0.1, 1000.0, 1)
        if ok:
            n_samples = int(duration * self.current_container.sample_rate)
            num_parts = len(self.current_container.time) // n_samples

            for i in range(num_parts):
                start = i * n_samples
                end = (i + 1) * n_samples
                new_data = {ch: data[start:end] for ch, data in self.current_container.data.items()}
                new_time = self.current_container.time[start:end]

                container = DataContainer(
                    new_data,
                    new_time,
                    self.current_container.sample_rate,
                    f"Fragment {i + 1}"
                )
                self.data_containers.append(container)
                self.plot_data(container)

    def split_by_parts(self):
        n_parts, ok = QInputDialog.getInt(self, "Split by Parts",
                                          "Enter number of parts:", 2, 1, 100)
        if ok:
            part_len = len(self.current_container.time) // n_parts

            for i in range(n_parts):
                start = i * part_len
                end = (i + 1) * part_len
                new_data = {ch: data[start:end] for ch, data in self.current_container.data.items()}
                new_time = self.current_container.time[start:end]

                container = DataContainer(
                    new_data,
                    new_time,
                    self.current_container.sample_rate,
                    f"Part {i + 1}"
                )
                self.data_containers.append(container)
                self.plot_data(container)

    def show_spectral_dialog(self):
        if not self.current_container:
            QMessageBox.warning(self, "Warning", "No data selected!")
            return

        spectrum_window = QMainWindow(self)
        spectrum_window.setWindowTitle(f"Spectral Analysis - {self.current_container.title}")
        spectrum_window.setGeometry(200, 200, 1000, 800)

        main_widget = QWidget()
        layout = QVBoxLayout()

        # Scroll area для большого количества каналов
        scroll = QScrollArea()
        content = QWidget()
        scroll_layout = QVBoxLayout(content)

        # Словарь для хранения результатов
        analysis_results = {}

        for channel in self.current_container.data:
            try:
                data = self.current_container.data[channel]
                if len(data) < 1024:
                    continue

                # Расчет спектра
                freq, psd = welch(data, self.current_container.sample_rate, nperseg=1024)

                # Создание графика
                plot = pg.PlotWidget(title=f"Channel {channel}")
                plot.plot(freq, psd, pen='b')
                plot.setLabel('left', 'Power', units='µV²/Hz')
                plot.setLabel('bottom', 'Frequency', units='Hz')
                scroll_layout.addWidget(plot)

                # Расчет мощностей
                band_powers = {}
                for band, (low, high) in FREQ_BANDS.items():
                    mask = (freq >= low) & (freq <= high)
                    power = trapezoid(psd[mask], freq[mask])
                    band_powers[band] = power
                # Создаем виджет с двумя областями
                splitter = pg.GraphicsLayoutWidget()

                # График спектра
                p1 = splitter.addPlot(title=f"Channel {channel} - PSD")
                p1.plot(freq, psd, pen='b')

                # Гистограмма
                p2 = splitter.addPlot(col=1)
                x = np.arange(len(FREQ_BANDS))
                bars = pg.BarGraphItem(x=x, height=list(band_powers.values()), width=0.6, brush='g')
                p2.addItem(bars)
                p2.getAxis('bottom').setTicks([[(i, band) for i, band in enumerate(FREQ_BANDS)]])

                scroll_layout.addWidget(splitter)

                analysis_results[channel] = band_powers

            except Exception as e:
                print(f"Error processing {channel}: {str(e)}")

        # Таблица результатов
        table = QTableWidget()
        table.setColumnCount(len(FREQ_BANDS) + 1)
        table.setRowCount(len(self.current_container.data))
        table.setHorizontalHeaderLabels(["Channel"] + list(FREQ_BANDS.keys()))

        for row, channel in enumerate(self.current_container.data):
            table.setItem(row, 0, QTableWidgetItem(channel))
            if channel in analysis_results:
                for col, band in enumerate(FREQ_BANDS.keys(), 1):
                    power = analysis_results[channel].get(band, 0)
                    table.setItem(row, col, QTableWidgetItem(f"{power:.2f}"))

        # Кнопка сохранения
        save_btn = QPushButton("Save Analysis Results")
        save_btn.clicked.connect(lambda: self.save_analysis_results(analysis_results))

        scroll.setWidget(content)
        layout.addWidget(scroll)
        layout.addWidget(table)
        layout.addWidget(save_btn)

        main_widget.setLayout(layout)
        spectrum_window.setCentralWidget(main_widget)
        spectrum_window.show()


    def save_analysis_results(self, results):
        path, _ = QFileDialog.getSaveFileName(self, "Save Results", "",
                                              "JSON Files (*.json);;HDF5 Files (*.h5)")

        if path:
            try:
                if path.endswith('.json'):
                    with open(path, 'w') as f:
                        json.dump(results, f, indent=4)
                elif path.endswith('.h5'):
                    with h5py.File(path, 'w') as f:
                        for channel, bands in results.items():
                            grp = f.create_group(channel)
                            for band, power in bands.items():
                                grp.attrs[band] = power

                QMessageBox.information(self, "Success", "Results saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed: {str(e)}")

    def save_data(self):
        if not self.current_container:
            QMessageBox.warning(self, "Warning", "No data selected!")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Data", "",
                                              "HDF5 (*.h5);;MAT (*.mat)")
        if path.endswith('.h5'):
            with h5py.File(path, 'w') as f:
                for ch, data in self.current_container.data.items():
                    ds = f.create_dataset(ch, data=data)
                    if hasattr(self.current_container, 'metadata'):
                        for key, value in self.current_container.metadata.items():
                            ds.attrs[key] = value
        if path:
            try:
                if path.endswith('.h5'):
                    with h5py.File(path, 'w') as f:
                        for ch, data in self.current_container.data.items():
                            f.create_dataset(ch, data=data)
                        f.attrs['sample_rate'] = self.current_container.sample_rate
                elif path.endswith('.mat'):
                    savemat(path, self.current_container.data)

                QMessageBox.information(self, "Success", "Data saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
