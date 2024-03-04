from src.label_encoder import KeystrokeLabelEncoder

def test_label_encoder():
    label_encoder = KeystrokeLabelEncoder()

    sequence_ids = ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']
    label_encoder.fit(sequence_ids)

    # Transform the sequence IDs into integer labels
    encoded_labels = label_encoder.transform(sequence_ids)

    assert encoded_labels == [0, 1, 2, 0, 1, 2, 0, 1, 2]

    # Inverse transform the encoded labels back to sequence IDs
    decoded_sequence_ids = label_encoder.inverse_transform(encoded_labels)

    assert decoded_sequence_ids == sequence_ids