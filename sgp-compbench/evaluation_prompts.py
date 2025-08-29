def eval_prompt_binding(prompt):
    return (
        f"Evaluate whether the image matches the following prompt: {prompt}\n"
        "Scoring criteria:\n"
        "100: All items are recognizable and the binding between items and their attributes is correct.\n"
        "50: All items are recognizable, but the binding between items and their attributes is incorrect or unclear.\n"
        "30: Items are not recognizable, but the attribute binding appears correct.\n"
        "0: Items are not recognizable and the binding between items and their attributes is incorrect.\n"
        "Please give your response in this format:\n"
        "REASONING: [your reasoning]\n"
        "SCORE: [score]"
    )

def eval_prompt_relation(prompt):
    return (
        f"Evaluate whether the image matches the following prompt: {prompt}\n"
        "Scoring criteria:\n"
        "100: The items is clear and the relation between items is correct.\n"
        "50: The items is not clear, but the relation between items is correct.\n"
        "30: The items is clear, but the relation between items is incorrect.\n"
        "0: The items is not clear and the relation between items is incorrect.\n"
        "Please give your response in this format:\n"
        "REASONING: [your reasoning]\n"
        "SCORE: [score]"
    )

def eval_prompt_numeracy_total(total_count):
    return (
        f"Evaluate whether the image contains exactly {total_count} distinct items in total (they do not need to be recognizable, but should be clearly individual objects).\n"
        "Scoring criteria:\n"
        "100: All items in the image are clearly individual objects, and the total count is correct.\n"
        "50: All items are clearly individual objects, but the total count is incorrect.\n"
        "30: Some items are clearly individual objects, and the total count is incorrect.\n"
        "0: The items are not clearly individual objects and the total count is incorrect.\n"
        "Please provide your response in the following format:\n"
        "REASONING: [your really brief reasoning]\n"
        "SCORE: [score]"
    )

def eval_prompt_numeracy_item(nouns):
    noun_list = ", ".join(nouns)
    return (
        f"Check whether the image contains the following items: {noun_list}.\n"
        "Scoring criteria:\n"
        "100: The image contains all the items listed above.\n"
        "50: The image contains most of the items listed above.\n"
        "30: The image contains some of the items listed above.\n"
        "0: The image does not contain any of the items listed above.\n"
        "Please give your response in this format:\n"
        "REASONING: [your really brief reasoning]\n"
        "SCORE: [score]"
    )

def eval_prompt_numeracy_count_binding(count, noun):
    return (
        f"Evaluate whether the image contains exactly {count} distinct {noun} in total.\n"
        "Scoring criteria:\n"
        f"100: the image contains exactly {count} distinct {noun}, and they are clearly individual objects.\n"
        f"50: the image does not contain all the {count} distinct {noun}, but the count is close to {count}.\n"
        f"30: the image does not contain all the {count} distinct {noun}, but the count is far from {count}.\n"
        f"0: the image does not contain any of the {count} distinct {noun}.\n"
        "Please provide your response in the following format:\n"
        "REASONING: [your really brief reasoning]\n"
        "SCORE: [score]"
    )