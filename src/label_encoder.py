import numpy as np


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
        int_labels = np.array([self.label_map[sequence_id] for sequence_id in sequence_ids])
        return self.encode_participant_ids(int_labels)
    def fit_transform(self, sequence_ids):
        self.fit(sequence_ids)
        return self.transform(sequence_ids)
    def inverse_transform(self, labels):
        int_labels = self.decode_participant_ids(labels)
        # Decode encoded integer labels back to original sequence IDs
        return [self.inverse_label_map[label] for label in int_labels]

    def encode_participant_ids(self, y: np.ndarray) -> np.ndarray:
        if len(self.label_map) == 0:
            raise ValueError("LabelEncoder has not been fitted yet.")
        # one hot encoding
        return np.eye(len(np.unique(y)))[y]

    def decode_participant_ids(self, y: np.ndarray) -> np.ndarray:
        if len(self.label_map) == 0:
            raise ValueError("LabelEncoder has not been fitted yet.")

        # one hot encoding
        int_labels = np.argmax(y, axis=1)
        return int_labels
