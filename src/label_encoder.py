class KeystrokeLabelEncoder:
    def __init__(self):
        self.label_map = {}
        self.inverse_label_map = {}

    def fit(self, sequence_ids):
        # Mapping of unique sequence IDs to integer labels
        unique_sequence_ids = set(sequence_ids)
        for i, sequence_id in enumerate(sorted(unique_sequence_ids)):
            self.label_map[sequence_id] = i
            self.inverse_label_map[i] = sequence_id

    def transform(self, sequence_ids):
        # Transform sequence IDs into encoded integer labels
        return [self.label_map[sequence_id] for sequence_id in sequence_ids]

    def inverse_transform(self, labels):
        # Decode encoded integer labels back to original sequence IDs
        return [self.inverse_label_map[label] for label in labels]