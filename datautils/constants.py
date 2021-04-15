mappings_kaggle2conll = {
    'B-per': 'B_PER', 'B-per_NOKG': 'B_PER', 'I-per': 'I_PER', 'I-per_NOKG': 'I_PER',
    'B-geo': 'B_LOC', 'B-geo_NOKG': 'B_LOC', 'I-geo': 'I_LOC', 'I-geo_NOKG': 'I_LOC',
    'B-gpe': 'B_LOC', 'B-gpe_NOKG': 'B_LOC', 'I-gpe': 'I_LOC', 'I-gpe_NOKG': 'I_LOC',
    'B-org': 'B_ORG', 'B-org_NOKG': 'B_ORG', 'I-org': 'I_ORG', 'I-org_NOKG': 'I_ORG',
    'B-tim': 'B_MISC', 'B-tim_NOKG': 'B_MISC', 'I-tim': 'I_MISC', 'I-tim_NOKG': 'I_MISC',
    'B-art': 'B_MISC', 'B-art_NOKG': 'B_MISC', 'I-art': 'I_MISC', 'I-art_NOKG': 'I_MISC',
    'B-eve': 'B_MISC', 'B-eve_NOKG': 'B_MISC', 'I-eve': 'I_MISC', 'I-eve_NOKG': 'I_MISC',
    'B-nat': 'B_MISC', 'B-nat_NOKG': 'B_MISC', 'I-nat': 'I_MISC', 'I-nat_NOKG': 'I_MISC',
}

mappings_conll2kaggle = {
    'B_PER': 'B-per', 'I_PER': 'I-per',
    'B_LOC': 'B-geo', 'I_LOC': 'I-geo',
    'B_ORG': 'B-org', 'I_ORG': 'I-org',
    'B_MISC': 'B-nat', 'I_MISC': 'I-nat',
}
