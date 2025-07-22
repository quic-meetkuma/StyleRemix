import argparse

import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt_tab')
from src.eval_data import eval_grade_level, eval_length

def main():
    parser = argparse.ArgumentParser(description="Metric checker")
    
    # Add arguments
    parser.add_argument("--txt", type=str, help="Input text")

    # Parse arguments
    args = parser.parse_args()
    
    grade_dict, avg_grade = eval_grade_level([args.txt])
    print("Grade: FK: ", grade_dict['fk'][0])
    print("Grade: LW: ", grade_dict['lw'][0])
    print("Grade: GF: ", grade_dict['gf'][0])
    print("Average : ", avg_grade[0])
    
    _, _, _, words_per_sentence, _ = eval_length([args.txt])
    print("Words per sentence: ", words_per_sentence[0])

if __name__ == "__main__":
    main()