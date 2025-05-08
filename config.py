import os
VERTICAL_SPACING = 25
FREQ_BANDS = {
        "Delta": (0.5, 3.9),
        "Theta": (4, 8),
        "Alpha": (8, 12),
        "Beta": (12, 30),
        "Gamma": (30, 70)
}
ANIMAL_COLORS = {
        'ап1_метка':      (31, 119, 180),
        'ап1_2_метки':    (255, 127, 14),
        'ап1_без_метки':  (44, 160, 44),
        'ап2_метка':      (214, 39, 40),
        'ап2_2_метки':    (148, 103, 189),
        'ап2_без_метки':  (140, 86, 75)
}
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")  # Директория для хранения логов
batches = {
        # Группа 1: ап1_без_метки
        "ап1_без_метки": {
                2: {
                        "ranges": [(100, 280), (1400, 1580)],
                        "exclude_channels": []
                },
                3: {
                        "ranges": [(100, 280), (1400, 1580)],
                        "exclude_channels": [2]
                }
        },

        # Группа 2: ап1_две_метки
        "ап1_две_метки": {
                1: {
                        "ranges": [(430, 610), (1320, 1500)],
                        "exclude_channels": []
                },
                2: {
                        "ranges": [(150, 330), (1400, 1580)],
                        "exclude_channels": []
                },
                3: {
                        "ranges": [(245, 290), (310, 340), (360, 460)],
                        "exclude_channels": [2],
                        "merge_method": "average"
                },
                4: {
                        "ranges": [
                                (300, 330), (335, 360), (430, 440),
                                (445, 450), (467, 490), (495, 535),
                                (553, 562)
                        ],
                        "exclude_channels": [2],
                        "merge_method": "average"
                },
                5: {
                        "ranges": [(160, 340), (1450, 1630)],
                        "exclude_channels": [2]
                }
        },

        # Группа 3: ап1_метка
        "ап1_метка": {
                1: {
                        "ranges": [(310, 380), (480, 500), (1100, 1280)],
                        "exclude_channels": [],
                        "merge_method": "average"
                },
                2: {
                        "ranges": [(100, 280), (1400, 1580)],
                        "exclude_channels": []
                },
                3: {
                        "ranges": [(100, 280), (1400, 1580)],
                        "exclude_channels": []
                },
                4: {
                        "ranges": [(105, 300), (1475, 1550), (1700, 1800)],
                        "exclude_channels": [2],
                        "merge_method": "average"
                },
                5: {
                        "ranges": [(100, 280), (1400, 1580)],
                        "exclude_channels": [2]
                }
        }
}