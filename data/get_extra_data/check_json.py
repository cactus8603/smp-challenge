import json
d = json.load(open('/local/smp/data/train_allmetadata_json/train_user_data.json'))
print('type:', type(d))
if isinstance(d, dict):
    keys = list(d.keys())
    print('top-level keys:', keys[:5])
    first_val = list(d.values())[0]
    print('first value type:', type(first_val))
    if isinstance(first_val, dict):
        print('first value keys:', list(first_val.keys())[:5])
    elif isinstance(first_val, list):
        print('first value length:', len(first_val))
        print('first value[0]:', first_val[0])
elif isinstance(d, list):
    print('length:', len(d))
    print('first item keys:', list(d[0].keys())[:5])