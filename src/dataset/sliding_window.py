def sliding_window(sequence, window_size, step_size=1):
    """
    Generates a sliding window over a sequence.

    Args:
        sequence (list or array): The input sequence.
        window_size (int): The size of the window.
        step_size (int, optional): The step size for the window. Defaults to 1.

    Yields:
        list or array: A window of the sequence.
    """
    if len(sequence) < window_size:
        raise ValueError("Sequence length must be at least window_size")

    for i in range(0, len(sequence) - window_size + 1, step_size):
        yield sequence[i:i + window_size]
