import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from source.scripts.get_tag_frequencies import get_tag_frequencies, filter_tag


dict_tag_frequencies_ordered = get_tag_frequencies()
print(dict_tag_frequencies_ordered)
print(len(dict_tag_frequencies_ordered))