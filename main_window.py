import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QToolBar, QAction, QTabWidget, QMessageBox,
    QFileDialog, QVBoxLayout, QWidget, QSplitter, QScrollArea, QDialog
)
from scipy.io import savemat
from logger import logger
import h5py
from utils.file_io import save_h5
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QIcon
import pyqtgraph as pg
from core.data_models import DataContainer
from core.eeg_processor import EEGProcessor
from core.signal_processing import *
from gui.dialogs import AnalysisDialog
from gui.plots import create_psd_plot, create_event_time_plot, create_main_plot
from utils.file_io import save_analysis_results
from gui.dialogs import SplitDialog, AnalysisDialog, ManualRangeDialog, StagedAnalysisDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_containers = []
        self.current_container = None
        self.tabs = QTabWidget()
        self.init_ui()
        self.init_settings()
        self.init_connections()  # Добавить вызов метода подключения сигналов


    def init_ui(self):
        self.setWindowTitle("EEG Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        self._create_toolbar()
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

    def init_settings(self):
        self.settings = QSettings("EEGApp", "MainWindow")
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def _show_error(self, message):  # Добавить метод обработки ошибок
        QMessageBox.critical(self, "Ошибка", message)

    def _create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        actions = [
            ("Open", "document-open", self.load_file),
            ("Split", "document-split", self.show_split_dialog),
            ("Spectrum", "graph", self.show_spectral_dialog),
            ("Save", "document-save", self.save_data),
            ("Staged", "color-gradient", StagedAnalysisDialog),
        ]
        actions.append(("Manual Range", "select-range", self.manual_range_analysis))

        for name, icon, callback in actions:
            action = QAction(QIcon.fromTheme(icon), name, self)
            action.triggered.connect(callback)
            toolbar.addAction(action)

        self.tabs.currentChanged.connect(self.on_tab_changed)

    def manual_range_analysis(self):
        """Обработка ручного выбора диапазона с логированием"""
        try:
            # Проверка наличия данных
            if not self.current_container or not isinstance(self.current_container, DataContainer):
                logger.log_operation("MANUAL_RANGE_ATTEMPT", {"status": "no_data"})
                self._show_warning("Сначала загрузите данные!")
                return

            logger.log_operation("MANUAL_RANGE_START", {
                "file": self.current_container.title,
                "duration": len(self.current_container.time) / self.current_container.sample_rate
            })


            # Создание диалога
            dialog = ManualRangeDialog(
                container=self.current_container,
                parent=self
            )

            if dialog.exec_() == QDialog.Accepted:
                # Логика обработки диапазона
                logger.log_operation("MANUAL_RANGE_SELECTED", {
                    "start": dialog.start_time,
                    "end": dialog.end_time
                })
                self._process_manual_range(dialog.start_time, dialog.end_time)
                #self.show_spectral_dialog()

        except Exception as e:
            error_message = f"Ошибка ручного выбора диапазона: {str(e)}"
            logger.log_error(e, "MANUAL_RANGE_ERROR", {
                "current_container": str(self.current_container) if self.current_container else None
            })
            self._show_error(error_message)

    def _process_manual_range(self, start_time: float, end_time: float):
        """Обработка выбранного диапазона"""
        try:
            start_idx = int(start_time * self.current_container.sample_rate)
            end_idx = int(end_time * self.current_container.sample_rate)

            # Создание нового контейнера
            new_data = {
                ch: data[start_idx:end_idx]
                for ch, data in self.current_container.data.items()
            }

            container = DataContainer(
                new_data,
                self.current_container.time[start_idx:end_idx],
                self.current_container.sample_rate,
                f"Manual Range {start_time:.1f}-{end_time:.1f} sec"  # Уникальное имя
            )

            self._add_container(container)
            self.current_container = container  # Принудительное обновление
            logger.log_operation("MANUAL_RANGE_SUCCESS", {"samples": end_idx - start_idx})

        except Exception as e:
            logger.log_error(e, "MANUAL_RANGE_PROCESSING_ERROR")
            raise

    def show_spectral_dialog(self):
        if not self.current_container:
            self._show_warning("Сначала загрузите данные!")
            return

        # Всегда используем текущий контейнер (последний выбранный)
        dialog = AnalysisDialog(
            container=self.current_container,  # <- Актуальные данные
            parent=self
        )
        dialog.exec_()

    def on_tab_changed(self, index):
        if 0 <= index < len(self.data_containers):
            self.current_container = self.data_containers[index+1]
            print(f"[SUCCESS] Текущий контейнер: '{self.current_container.title}' (вкладка {index})")
        else:
            print(f"[ERROR] Индекс {index} вне диапазона. Всего контейнеров: {len(self.data_containers)}")
    def init_connections(self):
        self.tabs.currentChanged.connect(self.on_tab_changed)

    def show_split_dialog(self):
        try:
            dialog = SplitDialog(parent=self)  # Явно указываем родителя
            if dialog.exec_() == QDialog.Accepted:
                method = dialog.method
                params = dialog.params
                self._handle_split(method, params)
        except Exception as e:
            logger.log_error(e, "SPLIT_DIALOG_ERROR")
    def load_file(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,  # Родительское окно
                "Выберите файл данных",  # Заголовок
                "",  # Стартовая директория
                "Файлы ЭЭГ (*.dat);;Все файлы (*)"  # Фильтры
            )
            logger.log_operation("FILE_LOAD", {"path": file_path})

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
                    self._add_container(container)
                    create_main_plot(container)
                    self.tabs.setCurrentIndex(len(self.data_containers) - 1)
                    QMessageBox.information(self, "Success", "File loaded successfully!")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Load error: {str(e)}")


        except Exception as e:
            logger.log_error(e, "FILE_LOAD_ERROR")
            QMessageBox.critical(...)
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

    def _handle_split(self, method, params):
        """Обработка разделения данных в соответствии с выбором пользователя"""
        if not self.current_container:
            self._show_warning("Сначала загрузите данные!")
            return

        try:
            if method == "time":
                self._split_by_time(params["duration"])
            elif method == "parts":
                self._split_by_parts(params["parts"])
            else:
                raise ValueError("Неизвестный метод разделения")

            QMessageBox.information(self, "Успех", "Данные разделены по указанию народа!")

        except Exception as e:
            self._show_error(f"Ошибка разделения: {str(e)}")
            logger.log_error(e, "SPLIT_ERROR")

    def _split_by_time(self, duration):
        """Разделение данных по времени"""
        n_samples = int(duration * self.current_container.sample_rate)
        num_parts = len(self.current_container.time) // n_samples

        for i in range(num_parts):
            start = i * n_samples
            end = (i + 1) * n_samples
            self._create_new_container(start, end, f"Часть {i + 1}")

    def _split_by_parts(self, n_parts):
        """Разделение данных на равные части"""
        part_len = len(self.current_container.time) // n_parts

        for i in range(n_parts):
            start = i * part_len
            end = (i + 1) * part_len
            self._create_new_container(start, end, f"Фрагмент {i + 1}")

    def _create_new_container(self, start_idx, end_idx, title):
        """Создание нового контейнера данных"""
        new_data = {
            ch: data[start_idx:end_idx]
            for ch, data in self.current_container.data.items()
        }
        new_time = self.current_container.time[start_idx:end_idx]

        container = DataContainer(
            new_data,
            new_time,
            self.current_container.sample_rate,
            title
        )
        self._add_container(container)

    def _add_container(self, container):
        self.data_containers.append(container)
        print(f"[DEBUG] Добавлен контейнер: '{container.title}'. Всего: {len(self.data_containers)}")

        tab = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(create_main_plot(container))
        tab.setLayout(layout)

        # Добавляем вкладку в ОДИН И ТОТ ЖЕ QTabWidget
        self.tabs.addTab(tab, container.title)
        new_index = self.tabs.count() - 1  # Корректный индекс новой вкладки
        self.tabs.setCurrentIndex(new_index)
        print(f"[DEBUG] Активная вкладка: {new_index}")

    def batch_process(self):
        # Реализация пакетной обработки
        pass


    def _process_session(self, session, files, output_dir):
        session_dir = os.path.join(output_dir, f"Session_{session}")
        os.makedirs(session_dir, exist_ok=True)

        for time_range in [(300, 600), (1500, 1800)]:
                for channel in ['Channel_1', 'Channel_3', 'Channel_4', 'Channel_5']:
                    plot = self._create_session_plot(session, time_range, channel)

                    for animal_id, path in files:
                        data = self._load_animal_data(path, time_range, channel)
                        if data:
                            self._add_animal_to_plot(plot, animal_id, data)

                    self._save_session_plot(plot, session_dir, channel, time_range)