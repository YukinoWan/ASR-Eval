import sys

def remove_repetitive_text(text):
    def find_earliest_ending_sequence(s):
        length = len(s)
        earliest_end = float('inf')
        cut_index = None

        for i in range(1, length // 2 + 1):  # Checking for different lengths of sequences
            for j in range(length - 2 * i):  # Ensuring space for one character after the second repetition
                seq = s[j:j + i]
                if s[j + i:j + 2 * i] == seq:
                    # Ensure there's a character after the second repetition and it matches the first character of seq
                    if j + 2 * i < length and s[j + 2 * i] == seq[0]:
                        end_point = j + 2 * i + 1
                        if end_point < earliest_end:
                            earliest_end = end_point
                            cut_index = j + i  # Start of the second occurrence
                            break  # Found the earliest, no need to check further for this length

        return cut_index

    result = find_earliest_ending_sequence(text)
    if result is not None:
        return text[:result]
    return text


# Example usage
#text1 = "abcdefff..."
#text2 = "9876543543543..."

#print(remove_repetitive_text(text1))  # Output: "abcdef"
#print(remove_repetitive_text(text2))  # Output: "9876543"

# Reading from stdin and applying the function to each line
if __name__ == "__main__":
    for line in sys.stdin:
        cleaned_line = remove_repetitive_text(line.strip())
        print(cleaned_line)
