# logger.py
import logging
import os
from datetime import datetime
from config import LOG_DIR


class RevolutionaryLogger:
    def __init__(self):
        self._setup_log_directory()
        self._configure_logger()

    def _setup_log_directory(self):
        """Создает директорию для логов революционных достижений"""
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

    def _configure_logger(self):
        """Настройка системы логирования для народного контроля"""
        log_filename = f"eeg_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(LOG_DIR, log_filename)),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger("PeopleLogger")

    def log_operation(self, operation: str, metadata: dict = None):
        """Логирование народных операций"""
        message = f"OPERATION: {operation}"
        if metadata:
            message += f" | DETAILS: {metadata}"
        self.logger.info(message)

    def log_error(self, error: Exception, context: str = None):
        """Фиксация контрреволюционных инцидентов"""
        message = f"ERROR: {str(error)}"
        if context:
            message += f" | CONTEXT: {context}"
        self.logger.error(message, exc_info=True)


# Инициализация народного логгера
logger = RevolutionaryLogger()