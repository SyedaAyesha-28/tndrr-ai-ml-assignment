# Tests the code that each model wrote, plus the 'z' rule from prompt 8.
# Functions below are copied exactly from the models' answers in results.csv.


# ---------- Prompt 5: palindrome ----------
def is_palindrome_fp16(s):
    cleaned_s = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned_s == cleaned_s[::-1]


def is_palindrome_q4(s):
    cleaned_s = s.replace(" ", "").lower()
    return cleaned_s == cleaned_s[::-1]


print("PROMPT 5: palindrome")
for text in ["A man, a plan, a canal, Panama!", "racecar", "hello"]:
    print(f"  {text!r}: fp16={is_palindrome_fp16(text)}, q4={is_palindrome_q4(text)}")


# ---------- Prompt 6: second largest ----------
def second_largest_fp16(numbers):
    if len(numbers) < 2:
        return None
    first_largest = second_largest = float('-inf')
    for num in numbers:
        if num > first_largest:
            second_largest = first_largest
            first_largest = num
        elif num > second_largest and num != first_largest:
            second_largest = num
    if second_largest < 0:
        return None
    return second_largest


def second_largest_q4(numbers):
    if len(numbers) < 2:
        return None
    largest = second_largest = float('-inf')
    for num in numbers:
        if num > largest:
            second_largest = largest
            largest = num
        elif num > second_largest and num != largest:
            second_largest = num
    return second_largest


print("\nPROMPT 6: second largest")
tests = [
    ([5, 3, 8, 2, 9, 1, 7], 8),
    ([-5, -3, -1], -3),
    ([2, 2], None),
]
for nums, expected in tests:
    print(f"  {nums}: expected={expected}, "
          f"fp16={second_largest_fp16(nums)}, q4={second_largest_q4(nums)}")


# ---------- Prompt 8: moon, exactly two sentences, no letter 'z' ----------
fp16_moon = ("The moon, a celestial body, is a natural satellite of Earth, characterized by its "
             "absence of atmosphere, surface features, and gravitational pull, making it a "
             "unique and fascinating celestial body.")
q4_moon = ("The moon is a celestial body orbiting the Earth, often depicted as a dark, silvery "
           "sphere with a diameter of about 3,474 kilometers.")

print("\nPROMPT 8: moon")
for name, text in [("fp16", fp16_moon), ("q4", q4_moon)]:
    sentences = text.count(". ") + (1 if text.strip().endswith(".") else 0)
    print(f"  {name}: contains 'z'? {'z' in text.lower()}, sentences={sentences}")