import pyqtgraph as pg
import config
import numpy as np
from scipy.signal import welch  # Добавить импорт

def create_main_plot(container):
    plot = pg.PlotWidget()
    plot.setBackground('w')  # Белый фон для народной ясности

    for channel in container.data:
        plot.plot(container.time, container.data[channel], name=channel)
    return plot
import pyqtgraph as pg


def create_psd_plot(freq, psd, channel_name):
    plot = pg.PlotWidget(title=f"{channel_name} - PSD")
    plot.setBackground('w')  # Белый фон для народной ясности
    plot.setXRange(0, 40, padding=0)
    plot.setYRange(0, 0.07, padding=0)
    # Установка фиксированных диапазонов
    plot.setXRange(0, 40)  # Ось X: 0-40 Гц
    plot.setYRange(0, 0.07)  # Ось Y: 0-0.1 мкВ²/Гц

    # Отключение авто-масштабирования
    plot.enableAutoRange(axis='x', enable=False)
    plot.enableAutoRange(axis='y', enable=False)

    plot.plot(freq, psd, pen='b')
    plot.setLabel('left', 'Power', units='µV²/Hz')
    plot.setLabel('bottom', 'Frequency', units='Hz')

    return plot


def create_main_plot(container):
    """Создает основной график для отображения временных данных"""
    plot = pg.PlotWidget(title=container.title)
    plot.setBackground('w')  # Белый фон для народной ясности

    # Добавляем все каналы с вертикальным смещением
    for channel in container.data:
        y_offset = container.data[channel] + (list(container.data.keys()).index(channel) * config.VERTICAL_SPACING)
        plot.plot(container.time, y_offset, name=channel)

    plot.setLabel('left', "Амплитуда (мкВ)")
    plot.setLabel('bottom', "Время (сек)")
    plot.addLegend()
    return plot


def create_spectral_analysis_plot(freq, psd, channel_name):
    plot = pg.PlotWidget(title=f"{channel_name} - PSD")
    plot.setBackground('w')  # Белый фон для народной ясности

    plot.plot(freq, psd, pen='b')
    plot.setLabel('left', 'Power (µV²/Hz)')
    plot.setLabel('bottom', 'Frequency (Hz)')
    plot.setXRange(0, 50)
    return plot

def create_event_time_plot(event):  # Новая функция для данных события
    plot = pg.PlotWidget(title=f"{event.type} Event")
    plot.setBackground('w')  # Белый фон для народной ясности

    for channel in event.data:
        plot.plot(event.time, event.data[channel], name=channel)
    return plot


def create_band_power_plot(band_powers):
    plot = pg.PlotWidget(title="Мощность диапазонов")
    plot.setBackground('w')
    plot.setYRange(0, 0.07)

    x = np.arange(len(band_powers))
    y = [band_powers[band] for band in band_powers]
    labels = list(band_powers.keys())

    bars = pg.BarGraphItem(x=x, height=y, width=0.6, brush='g')
    plot.addItem(bars)

    plot.getAxis('bottom').setTicks([[(i, label) for i, label in enumerate(labels)]])
    plot.getAxis('left').setLabel('Мощность, мкВ²/Гц')

    return plot