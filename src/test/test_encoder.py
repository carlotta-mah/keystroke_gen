import numpy as np

from src.label_encoder import KeystrokeLabelEncoder

def test_label_encoder():
    #general test of the functioning of the label encoder
    label_encoder = KeystrokeLabelEncoder()

    sequence_ids = ['a', 'b', 'c', 'a', 'b', 'c']
    label_encoder.fit(sequence_ids)

    # Transform the sequence IDs into integer labels
    encoded_labels = label_encoder.transform(sequence_ids)
    expected_encoded_ids = np.array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
    expected_label_map = {'a': 0, 'b': 1, 'c': 2}

    # check if the label encoder encodes labels as expected
    assert (encoded_labels == expected_encoded_ids).all()
    # check if the label encoder has expected label map
    assert label_encoder.label_map == expected_label_map

    # Inverse transform the encoded labels back to sequence IDs and check if they changed
    decoded_sequence_ids = label_encoder.inverse_transform(encoded_labels)
    assert decoded_sequence_ids == sequence_ids

def test_fit():
    label_encoder = KeystrokeLabelEncoder()

    sequence_ids = ['a', 'b', 'c', 'a', 'b', 'c']
    label_encoder.fit(sequence_ids)

    expected_label_map = {'a': 0, 'b': 1, 'c': 2}

    # check if the label encoder has expected label map i.e. the fit is as expected
    assert label_encoder.label_map == expected_label_map

def test_transform():
    label_encoder = KeystrokeLabelEncoder()

    sequence_ids = ['a', 'b', 'c', 'a', 'b', 'c']
    label_encoder.fit(sequence_ids)

    # Transform the sequence IDs into integer labels
    encoded_labels = label_encoder.transform(sequence_ids)

    # check if labels were transformed/encoded correctly
    expected_encoded_ids = np.array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
    assert (encoded_labels == expected_encoded_ids).all()

def test_fit_transform():
    # test if fit and transform work together
    label_encoder = KeystrokeLabelEncoder()

    sequence_ids = ['a', 'b', 'c', 'a', 'b', 'c']
    encoded_labels = label_encoder.fit_transform(sequence_ids)
    expected_encoded_ids = np.array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
    expected_label_map = {'a': 0, 'b': 1, 'c': 2}

    # check label map and transformed ids
    assert (encoded_labels == expected_encoded_ids).all()
    assert label_encoder.label_map == expected_label_map

def test_inverse_transform():
    label_encoder = KeystrokeLabelEncoder()
    label_encoder.label_map = {'a': 0, 'b': 1, 'c': 2}
    label_encoder.inverse_label_map = {0: 'a', 1: 'b', 2: 'c'}
    encoded_labels = np.array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])

    # Inverse transform the encoded labels back to sequence IDs
    decoded_sequence_ids = label_encoder.inverse_transform(encoded_labels)

    expected_sequence_ids = ['a', 'b', 'c', 'a', 'b', 'c']

    # check decoded ids
    assert decoded_sequence_ids == expected_sequence_ids