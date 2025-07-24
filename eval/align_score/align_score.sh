#!/bin/sh

# Compute AlignScore for the gold summaries
# /opt/homebrew/Caskroom/miniforge/base/envs/align-score/bin/python align_score.py --num_examples 100 --output_file AlignScore=gold.txt

# AlignScore for LED-base
/opt/homebrew/Caskroom/miniforge/base/envs/align-score/bin/python align_score.py --num_examples 100 --bill_file ../../preprocess/data/clean_billsum_test.csv --summary_file ../../output/deliverable_4/led-base/led-base_billsum_clean_test_se3-led-2048-512.csv --output_file AlignScore=led-base_2048.txt
/opt/homebrew/Caskroom/miniforge/base/envs/align-score/bin/python align_score.py --num_examples 100 --bill_file ../../preprocess/data/clean_billsum_test.csv --summary_file ../../output/deliverable_4/led-base/led-base_billsum_clean_test_se3-t5-512-512.csv --output_file   AlignScore=led-base_512.txt

# AlignScore for wugNATSS-led
# /opt/homebrew/Caskroom/miniforge/base/envs/align-score/bin/python align_score.py --num_examples 100 --bill_file ../../preprocess/data/clean_billsum_test.csv --summary_file ../../output/deliverable_4/wugNATSS-led/eval_on_simp.csv --output_file   AlignScore=wugNATSS-led_on_simp.txt
# /opt/homebrew/Caskroom/miniforge/base/envs/align-score/bin/python align_score.py --num_examples 100 --bill_file ../../preprocess/data/clean_billsum_test.csv --summary_file ../../output/deliverable_4/wugNATSS-led/eval_on_unsimp.csv --output_file AlignScore=wugNATSS-led_on_unsimp.txt

# AlignScore for wugwATSS-led
# /opt/homebrew/Caskroom/miniforge/base/envs/align-score/bin/python align_score.py --num_examples 100 --bill_file ../../preprocess/data/clean_billsum_test.csv --summary_file ../../output/deliverable_4/wugwATSS-led/eval_on_simp.csv --output_file   AlignScore=wugwATSS-led_on_simp.txt
# /opt/homebrew/Caskroom/miniforge/base/envs/align-score/bin/python align_score.py --num_examples 100 --bill_file ../../preprocess/data/clean_billsum_test.csv --summary_file ../../output/deliverable_4/wugwATSS-led/eval_on_unsimp.csv --output_file AlignScore=wugwATSS-led_on_unsimp.txt

# AlignScore for wugNATSS-pegasus
# /opt/homebrew/Caskroom/miniforge/base/envs/align-score/bin/python align_score.py --num_examples 100 --bill_file ../../preprocess/data/clean_billsum_test.csv --summary_file ../../output/deliverable_4/wugNATSS-pegasus/eval_on_unsimp.csv --output_file AlignScore=wugNATSS-pegasus_on_unsimp.txt

# # AlignScore for wugwATSS-pegasus
# /opt/homebrew/Caskroom/miniforge/base/envs/align-score/bin/python align_score.py --num_examples 100 --bill_file ../../preprocess/data/clean_billsum_test.csv --summary_file ../../output/deliverable_4/wugwATSS-pegasus/eval_on_simp.csv --output_file AlignScore=wugwATSS-pegasus_on_simp.txt
# /opt/homebrew/Caskroom/miniforge/base/envs/align-score/bin/python align_score.py --num_examples 100 --bill_file ../../preprocess/data/clean_billsum_test.csv --summary_file ../../output/deliverable_4/wugwATSS-pegasus/eval_on_unsimp.csv --output_file AlignScore=wugwATSS-pegasus_on_unsimp.txt

