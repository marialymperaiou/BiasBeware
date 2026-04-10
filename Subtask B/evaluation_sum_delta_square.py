def sum_squared_rank_displacement(post_defence_rank: list, original_rank: list) -> int:
    """
    Compute sum(Δ²) between two rankings.

    Args:
        post_defence_rank: Ranked list after defence.
        original_rank: Original reference ranking.

    Returns:
        Sum of squared rank displacement.

    Example:
        original_rank = ["A", "B", "C", "D"]
        post_defence_rank = ["B", "A", "D", "C"]
        score = sum_squared_rank_displacement(post_defence_rank, original_rank)
    """
    if len(post_defence_rank) != len(original_rank):
        raise ValueError("Both rankings must have the same length.")

    if set(post_defence_rank) != set(original_rank):
        raise ValueError("Both rankings must contain exactly the same items.")

    original_pos = {item: idx for idx, item in enumerate(original_rank)}
    post_defence_pos = {item: idx for idx, item in enumerate(post_defence_rank)}

    return sum(
        (original_pos[item] - post_defence_pos[item]) ** 2
        for item in original_pos
    )
