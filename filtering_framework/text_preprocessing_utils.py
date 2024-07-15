import re
from num2words import num2words
from tqdm import tqdm

def convert_digits_to_words(data):
    # Define a regex pattern to match digits that might be connected to letters
    digit_pattern = re.compile(r'(\d+)')

    # Function to convert a matched digit to words
    def replace_with_words(match):
        return num2words(int(match.group()), lang='pt')

    # Use re.sub with a lambda to ensure digits are replaced correctly even when connected to letters
    result = digit_pattern.sub(lambda match: ' ' + replace_with_words(match) + ' ', data)

    # Clean up spaces around the digits
    data = re.sub(r'\s+', ' ', result).strip()

    return data

def remove_special_chars(data):

    chars = [
        r'\x13',
        r'\x10',
        r'\x01',
        r'\x02',
        r'\x10',
        r'\x12',
        r'\x15',
        r'\x1b',
        r'\x18',
        r'ø',
        r'\x1b',
        r'½',
        r'¹',
        r'ß',
        r'Ô',
        r'¥',
        r'æ',
        r'ë',
        r'î',
        r'û',
        r'ò',
        r'ö',
        r'\|n',
        r'§',
        r'll',
        r'¢',
        r'©',
        r'£',
        r'~',
        r'º',
        r"\(", r"\)", r"\̀", r"–", r"~", r"”", r"»", r"“", r"«", r"˙", r"\\", r"—", r"@", r"å", r"´",
    ]
    
    chars_to_delete = '|'.join(chars)

    data["text"] = re.sub(chars_to_delete, '', data["text"])

    return data

def replace_patterns(data):
    substitutions = {
        r"'([A-Z])": r' \1',
        r"'([a-z])": r'\1',
        r'&': 'e',
        r'ñ': 'n',
        r'ü': 'u',
        r'ž': 'z',
        r'È': 'É',
        r'è': 'é',
        r'ä': 'a',
        r'ï': 'i',
        r'ù': 'u',
        r"d'": "de ",
        r"D'": "De ",
        r"'s": " is",
        r"'e": "é",
        r"'a": "à",
        r"n'uma": "numa",
        r"n'um": "num",
        r"n'outra": "noutra",
        r"c'roa": "coroa",
        r"m'o": "mo",
        r"lh'o": "lho",
        r'"':'',
        r'\~ao':'ão',
        r'\^e': 'ê',
        r'°c': 'ro',
        r'\^a': 'â',
        r'$': '',
        r'±':'',
        r'ª':'',
        r'\$':'',
        r"I’m": 'I am',
        r"'":'',
        r'³':'',
        r'̂': '',
        r'!': '.',
        r'²':'',
        r'…':'',
        r'`': '',
        r'·':'',
        r'°':'',
        r'º':'',
        r'¡':'',
        r'-': ' ',
        r'  ': ' ',
        r"\+": 'mais',
        r'   ':' ',
        r' ,':',',
        r' {2,}': ' ',
    }
    
    for pattern, replacement in substitutions.items():
        data["text"] = re.sub(pattern, replacement, data["text"])
    
    return data

def convert_to_digit(data):
    data["text"] = convert_digits_to_words(data["text"])
    return data

def remove_dobule_space(data):
    data["text"] = re.sub(r'\s{2,}', ' ', data["text"])
    return data

def remove_space_punctuation(data):
    data["text"] = re.sub(r'\s+([.,:?!])', r'\1', data["text"])
    return data

def remove_others(data):
    # From bracarense
    pattern = re.compile(r'tʃurar|tʃegava|tʃu|tʃjepa|=|pausa longa||')
    data["text"] = pattern.sub('', data["text"])
    return data

def add_uppercase_and_final_punctuation(data):
    # Step 1: Convert the first character to uppercase
    data["text"] = re.sub(r'^(.)', lambda match: match.group(1).upper(), data["text"])
    
    # Step 2: Append a period if the string doesn't end with a period or question mark
    if not re.search(r'[.?]$', data["text"]):
        data["text"] += '.'
    
    return data

# Processing pipeline
def apply_preprocessors(manifest, preprocessors):
    for processor in preprocessors:
        for idx in tqdm(range(len(manifest)), desc=f"Applying {processor.__name__}"):
            manifest[idx] = processor(manifest[idx])

    print("Finished processing manifest !")
    return manifest


# List of pre-processing functions
PREPROCESSORS = [
    remove_special_chars,
    remove_others,
    replace_patterns,
    convert_to_digit,
    add_uppercase_and_final_punctuation,
    remove_dobule_space,
    remove_space_punctuation,
]

# Processing pipeline
def apply_preprocessors(manifest, preprocessors=PREPROCESSORS):
    for processor in preprocessors:
        for idx in tqdm(range(len(manifest)), desc=f"Applying {processor.__name__}"):
            manifest[idx] = processor(manifest[idx])

    print("Finished processing manifest !")
    return manifest

