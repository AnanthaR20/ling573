#!/bin/sh

python summac_toy.py --input_csv "../../../output/deliverable_4/wugNATSS-led/eval_on_simp.csv" --output_csv "summac_wugNATSS-led_eval_on_simp.csv"
python summac_toy.py --input_csv "../../../output/deliverable_4/wugNATSS-led/eval_on_unsimp.csv" --output_csv "summac_wugNATSS-led_eval_on_unsimp.csv"
python summac_toy.py --input_csv "../../../output/deliverable_4/wugNATSS-pegasus/eval_on_unsimp.csv" --output_csv "summac_wugNATSS-pegasus_eval_on_unsimp.csv"
python summac_toy.py --input_csv "../../../output/deliverable_4/wugwATSS-led/eval_on_simp.csv" --output_csv "summac_wugwATSS-led_eval_on_simp.csv"
python summac_toy.py --input_csv "../../../output/deliverable_4/wugwATSS-led/eval_on_unsimp.csv" --output_csv "summac_wugwATSS-led_eval_on_unsimp.csv"
python summac_toy.py --input_csv "../../../output/deliverable_4/wugwATSS-pegasus/eval_on_simp.csv" --output_csv "summac_wugwATSS-pegasus_eval_on_simp.csv"
python summac_toy.py --input_csv "../../../output/deliverable_4/wugwATSS-pegasus/eval_on_unsimp.csv" --output_csv "summac_wugwATSS-pegasus_eval_on_unsimp.csv"

