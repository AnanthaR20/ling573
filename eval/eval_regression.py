from datasets import load_dataset

## FEATURE SOURCES

# AoA - AOA_CW
# TODO: scrape https://norare.clld.org/unitparameters/Kuperman-2012-AoA-ENGLISH_AOA_MEAN

# imageability - MRC_IMAG_AW
mrc_data = load_dataset("StephanAkkerman/MRC-psycholinguistic-database")['train'].to_pandas()
imageability = mrc_data[["Word", "Imageability"]]
print(imageability.head())

# Legal terms - GI_LEG_NOUNS
# TODO: download and clean https://github.com/LexPredict/lexpredict-legal-dictionary/tree/master/en/legal

# Common tri-grams - COCA_SPOKEN_TRITOP30K
# COCA is behind paywall, we could clean Confirm-Labs/pile_top_trigrams (HF dataset) and substitute?

# Dependents - NOMINAL_DEPS_NN_STDEV
# TODO: implement with spacy or nltk 

# Concreteness -  MRC_CONCR_FW
concreteness = mrc_data[["Word", "Concreteness"]]
print(concreteness.head())

## TEST DATA
# TODO: download MADRS human score data for training
# TODO: train a sklearn linear regression model to get new weights for each feature
# TODO: download model for use at test-time