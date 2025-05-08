import re
from collections import defaultdict


def group_files_by_session(folder):
    pattern = re.compile(r"^(ап\d+_.+?)_(\d+)\.DAT$", re.IGNORECASE)
    groups = defaultdict(list)

    for fname in os.listdir(folder):
        match = pattern.match(fname)
        if match:
            animal_id, session = match.groups()
            groups[session].append((animal_id, os.path.join(folder, fname)))
    return groups


def validate_data_container(container):
    required_attrs = ['data', 'time', 'sample_rate']
    return all(hasattr(container, attr) for attr in required_attrs)