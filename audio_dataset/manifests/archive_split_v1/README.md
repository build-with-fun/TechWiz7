# Withdrawn Teachable Machine segment lists (split v1)

These files list the 5,230 two-second segments `cut_gtm_samples.py` cut on 25 Sep from the
training recordings of **split v1**. Split v2 (26 Sep) regroups clips by source recording, and
under v2 the parents of these segments fall 3,631 in train, 785 in validation and 814 in test.
They are kept only as a record; they must not be used for training.

No shipped model used them. The Teachable Machine models were trained from
`gtm_model/upload_package/tm_imports/`, built by `audio_dataset/scripts/make_gtm_imports.py`
from v2 training recordings only. The current evidence list is
`audio_dataset/manifests/gtm_segment_rows.csv`. The audio itself (not in Git) was moved to
`data/archive/gtm_v1/`.
