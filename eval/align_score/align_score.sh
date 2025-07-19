#!/bin/sh

# Compute AlignScore for the various outputs.
/home2/arao20/miniconda3/envs/align-score/bin/python align_score.py --num_examples 1 > test.txt

# # AlignScore for LED-base
# /home2/arao20/miniconda3/envs/align-score/bin/python align_score.py --bill_file ../../preprocess/data/clean_billsum_test.csv --summary_file ../../output/deliverable_4/led-base/led-base_billsum_clean_test_se3-led-2048-512.csv > AlignScore=led-base_se3_led_2048_512.txt
# /home2/arao20/miniconda3/envs/align-score/bin/python align_score.py --bill_file ../../preprocess/data/clean_billsum_test.csv --summary_file ../../output/deliverable_4/led-base/led-base_billsum_clean_test_se3-t5-512-512.csv >   AlignScore=led-base_se3_t5_512_512.txt

# # AlignScore for wugNATSS-large
# /home2/arao20/miniconda3/envs/align-score/bin/python align_score.py --bill_file ../../preprocess/data/clean_billsum_test.csv --summary_file ../../output/deliverable_4/wugNATSS-large/eval_on_unsimp.csv > AlignScore=wugNATSS-large_on_unsimp.txt

# # AlignScore for wugNATSS-led
# /home2/arao20/miniconda3/envs/align-score/bin/python align_score.py --bill_file ../../preprocess/data/clean_billsum_test.csv --summary_file ../../output/deliverable_4/wugNATSS-led/eval_on_simp.csv >   AlignScore=wugNATSS-led_on_simp.txt
# /home2/arao20/miniconda3/envs/align-score/bin/python align_score.py --bill_file ../../preprocess/data/clean_billsum_test.csv --summary_file ../../output/deliverable_4/wugNATSS-led/eval_on_unsimp.csv > AlignScore=wugNATSS-led_on_unsimp.txt

# # AlignScore for wugNATSS-pegasus
# /home2/arao20/miniconda3/envs/align-score/bin/python align_score.py --bill_file ../../preprocess/data/clean_billsum_test.csv --summary_file ../../output/deliverable_4/wugNATSS-pegasus/eval_on_unsimp.csv > AlignScore=wugNATSS-pegasus_on_unsimp.txt

# # AlignScore for wugwATSS-led
# /home2/arao20/miniconda3/envs/align-score/bin/python align_score.py --bill_file ../../preprocess/data/clean_billsum_test.csv --summary_file ../../output/deliverable_4/wugwATSS-led/eval_on_simp.csv >   AlignScore=wugwATSS-led_on_simp.txt
# /home2/arao20/miniconda3/envs/align-score/bin/python align_score.py --bill_file ../../preprocess/data/clean_billsum_test.csv --summary_file ../../output/deliverable_4/wugwATSS-led/eval_on_unsimp.csv > AlignScore=wugwATSS-led_on_unsimp.txt

# # AlignScore for wugwATSS-pegasus
# /home2/arao20/miniconda3/envs/align-score/bin/python align_score.py --bill_file ../../preprocess/data/clean_billsum_test.csv --summary_file ../../output/deliverable_4/wugwATSS-pegasus/eval_on_simp.csv > AlignScore=wugwATSS-pegasus_on_simp.txt
# /home2/arao20/miniconda3/envs/align-score/bin/python align_score.py --bill_file ../../preprocess/data/clean_billsum_test.csv --summary_file ../../output/deliverable_4/wugwATSS-pegasus/eval_on_unsimp.csv > AlignScore=wugwATSS-pegasus_on_unsimp.txt

