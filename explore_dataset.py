#!/usr/bin/env python3

from datasets import load_dataset
import pandas as pd

def explore_imppres_structure():
    print("Loading ImpPres dataset...")
    
    # Load a small section to examine structure
    dataset = load_dataset('facebook/imppres', 'presupposition_all_n_presupposition')
    df = dataset.to_pandas()
    
    print('Dataset columns:', df.columns.tolist())
    print('\nDataset shape:', df.shape)
    print('\nFirst few rows:')
    print(df.head())
    
    print('\nParadigm ID info:')
    print('Unique paradigm IDs:', df['paradigmID'].nunique())
    print('Samples per paradigm (first 10):')
    paradigm_counts = df['paradigmID'].value_counts()
    print(paradigm_counts.head(10))
    
    print('\nExample paradigm (first paradigm):')
    first_paradigm_id = df['paradigmID'].iloc[0]
    first_paradigm = df[df['paradigmID'] == first_paradigm_id]
    print(f'Paradigm {first_paradigm_id} has {len(first_paradigm)} samples')
    print(first_paradigm[['premise', 'hypothesis', 'gold_label', 'pairID']].to_string())
    
    print('\nGold label distribution:')
    print(df['gold_label'].value_counts())
    
    print('\nUID and pairID info:')
    print('Unique UIDs:', df['UID'].nunique())
    print('Unique pairIDs:', df['pairID'].nunique())

if __name__ == "__main__":
    explore_imppres_structure()