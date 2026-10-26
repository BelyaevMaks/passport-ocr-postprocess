#!/usr/bin/env python3
"""
Post-processing OCR predictions for Russian Passport.
Usage: python postprocess.py <input_predictions.jsonl> <output_answers.jsonl>

- Runs purely on CPU, zero external dependencies (Python standard library only).
- Offline-ready: uses local reference/subdivisions.csv.
- Multi-tier conservative guardrails prioritizing zero field error rate:
  1. ICAO Doc 9303 line 2 format & 4 independent checksums (100% mathematical verification).
  2. Passport legal validity on reference date 01.10.2026 (14/20/45 age replacement rules).
  3. Strict MRZ line 1 character confidence (>= 0.995) to prevent noisy name matching.
  4. Visual names OCR confidence (>= 0.985) and cross-validation against MRZ Line 1 transliteration.
  5. Multi-line geometry checks detecting line truncation or missing lines.
  6. High-confidence character thresholds for multi-line fields (Birth_place_ru >= 0.9985, Issue_org >= 0.985).
  7. Subdivisions code lookup and reference vocabulary validation for Issue_organization_ru.
  8. Quality guardrails against glare (> 0.12), blur (< 120), and low resolution (< 1200 px).
"""

import sys
import json
import csv
import os
import re
from datetime import datetime, date, timedelta

# Fields expected by evaluate.py
FIELDS = [
    "Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
    "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
    "Issue_organization_ru"
]

# Guardrail Thresholds
MAX_GLARE = 0.12            # Rejects heavy reflections causing character drops
MIN_SHARPNESS = 120.0       # Rejects severe blur
MIN_DOC_PX = 1200           # Rejects low-resolution crops
MIN_CONF_MRZ_L1 = 0.995     # Character confidence threshold on MRZ transliterated names
MIN_CONF_NAME = 0.985       # Character confidence threshold on visual Russian names
MIN_CONF_BP = 0.9985        # Clean confidence on Birth_place_ru to eliminate artifacts/spurious chars
MIN_CONF_ORG = 0.985        # Character confidence threshold on Issue_organization_ru

# Deterministic Russian Passport MRZ transliteration table (Order of MIA / GOST)
RU_TO_MRZ = {
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': '2',
    'Ж': 'J', 'З': 'Z', 'И': 'I', 'Й': 'Q', 'К': 'K', 'Л': 'L', 'М': 'M',
    'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
    'Ф': 'F', 'Х': 'H', 'Ц': 'C', 'Ч': '3', 'Ш': '4', 'Щ': 'W', 'Ъ': 'X',
    'Ы': 'Y', 'Ь': '9', 'Э': '6', 'Ю': '7', 'Я': '8', ' ': '<', '-': '<'
}

STANDARD_ADMIN_TOKENS = {
    'РЕСПУБЛИКЕ', 'РЕСПУБЛИКИ', 'МИГРАЦИОННЫЙ', 'ТЕРРИТОРИАЛЬНЫЙ', 'МЕЖРАЙОННЫЙ',
    'ГОР', 'ПУНКТ', 'ГОМ', 'ПОМ', 'ВИЗОВОЕ', 'АВТОНОМНОЙ', 'АВТОНОМНОГО', 'РФ',
    'ОТДЕЛ', 'ОТДЕЛЕНИЕМ', 'ОТДЕЛОМ', 'УПРАВЛЕНИЕМ', 'УПРАВЛЕНИЕ', 'ОТДЕЛЕНИЕ'
}

def transliterate_ru(text):
    """Convert Russian name to MRZ transliterated uppercase string."""
    return ''.join(RU_TO_MRZ.get(c.upper(), '') for c in text)

def icao_checksum(s):
    """Compute ICAO Doc 9303 modulo 10 checksum with weights 7, 3, 1."""
    weights = [7, 3, 1]
    total = sum(('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<'.index(c) if c in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' else 0) * weights[i % 3] for i, c in enumerate(s))
    return total % 10

def check_mrz_l2(l2):
    """
    Validate Russian Passport MRZ line 2 against 4 independent checksums
    and strict character positions.
    """
    if len(l2) != 44:
        return False
    # Digit positions
    digit_positions = list(range(0, 10)) + list(range(13, 20)) + [28] + list(range(29, 35)) + list(range(35, 41)) + [42, 43]
    if not all(l2[i].isdigit() for i in digit_positions):
        return False
    if l2[10:13] != 'RUS' or l2[20] not in ('M', 'F'):
        return False
    # Check 1: series[:3] + number
    if icao_checksum(l2[0:9]) != int(l2[9]):
        return False
    # Check 2: birth date
    if icao_checksum(l2[13:19]) != int(l2[19]):
        return False
    # Check 3: series[3] + issue date + code
    if icao_checksum(l2[28:42]) != int(l2[42]):
        return False
    # Check 4: composite checksum
    if icao_checksum(l2[0:10] + l2[13:20] + l2[21:43]) != int(l2[43]):
        return False
    return True

def parse_mrz_l2(l2):
    """Extract 5 verified fields from MRZ line 2."""
    lic_num = f"{l2[0:2]} {l2[2]}{l2[28]} {l2[3:9]}"
    b_yy = int(l2[13:15])
    b_century = '19' if b_yy > 26 else '20'
    b_date = f"{l2[17:19]}.{l2[15:17]}.{b_century}{l2[13:15]}"
    sex = 'МУЖ.' if l2[20] == 'M' else 'ЖЕН.'
    i_yy = int(l2[29:31])
    i_century = '19' if i_yy >= 90 else '20'
    i_date = f"{l2[33:35]}.{l2[31:33]}.{i_century}{l2[29:31]}"
    code = f"{l2[35:38]}-{l2[38:41]}"
    return lic_num, b_date, sex, i_date, code

def add_years(d, years):
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(year=d.year + years, day=28)

def is_passport_valid(b_str, i_str, ref_date=date(2026, 10, 1), grace_days=90):
    """
    Check legal validity of passport on reference date (01.10.2026).
    Rules:
      - Issued at >= 14 years of age.
      - Issue date <= 01.10.2026.
      - Issued before 20: expires on 20th birthday + 90 days.
      - Issued between 20 and 44: expires on 45th birthday + 90 days.
      - Issued at >= 45: indefinitely valid.
    """
    try:
        b = datetime.strptime(b_str, '%d.%m.%Y').date()
        i = datetime.strptime(i_str, '%d.%m.%Y').date()
    except Exception:
        return False
    if i > ref_date or i < b:
        return False
    b_14 = add_years(b, 14)
    if i < b_14:
        return False
    b_20 = add_years(b, 20)
    b_45 = add_years(b, 45)
    if i < b_20:
        if ref_date > (b_20 + timedelta(days=grace_days)):
            return False
    elif i < b_45:
        if ref_date > (b_45 + timedelta(days=grace_days)):
            return False
    return True

def check_lines_geometry(field_val):
    """
    Detect missing lines (truncation) by checking box dimensions and gap margins.
    """
    box = field_val.get('box', [0, 0, 0, 0])
    lines = field_val.get('lines', [])
    if not lines:
        return False

    f_top = box[1]
    f_bottom = box[1] + box[3]
    f_h = box[3]

    # Height vs line count
    if f_h >= 48 and len(lines) == 1:
        return False
    if f_h >= 75 and len(lines) == 2:
        return False

    # Top gap: first line must start near the top of the box
    if lines[0].get('box') and (lines[0]['box'][1] - f_top) > 18:
        return False

    # Inter-line gaps and bottom gap
    for idx in range(len(lines)):
        l_box = lines[idx].get('box')
        if not l_box:
            continue
        l_bottom = l_box[1] + l_box[3]
        if idx + 1 < len(lines):
            next_top = lines[idx+1].get('box', [0, 0, 0, 0])[1]
            if (next_top - l_bottom) > 18:
                return False
        else:
            # Last line: bottom gap
            if (f_bottom - l_bottom) > 18 and (f_h - (l_bottom - f_top)) >= 18:
                return False

    return True

def load_reference_subdivisions(ref_path):
    """
    Load reference subdivision codes and official vocabulary from CSV.
    Returns:
      subdiv_codes: set of valid subdivision code strings (e.g. '770-001')
      subdiv_vocab: set of valid uppercase word tokens appearing in subdivision names
    """
    subdiv_codes = set()
    subdiv_vocab = set(STANDARD_ADMIN_TOKENS)
    if os.path.exists(ref_path):
        with open(ref_path, encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row:
                    subdiv_codes.add(row[0].strip())
                    if len(row) > 1:
                        tokens = re.findall(r'[А-ЯЁа-яёA-Za-z0-9]+', row[1].upper())
                        subdiv_vocab.update(tokens)
    return subdiv_codes, subdiv_vocab

def process_document(doc, subdiv_codes, subdiv_vocab=None):
    """
    Evaluate single document prediction.
    Returns:
      decision: 'accept' | 'reject'
      fields: dict of 10 fields if accept, else None
    """
    # Guardrail 0: Overall Image Quality Check
    q = doc.get('quality') or {}
    if q.get('glare', 0) > MAX_GLARE:
        return 'reject', None
    if q.get('sharpness', 0) < MIN_SHARPNESS:
        return 'reject', None
    if q.get('doc_px', 0) < MIN_DOC_PX:
        return 'reject', None

    viz = doc.get('viz') or {}
    mrz = doc.get('mrz')

    # Guardrail 1: MRZ existence & line 2 mathematical verification
    if not mrz or not mrz.get('lines') or len(mrz['lines']) < 2:
        return 'reject', None

    l1 = mrz['lines'][0].get('text', '')
    l2 = mrz['lines'][1].get('text', '')

    if not check_mrz_l2(l2):
        return 'reject', None

    lic_num, b_date, sex, i_date, code = parse_mrz_l2(l2)

    # Guardrail 2: Legal validity of passport on 01.10.2026
    if not is_passport_valid(b_date, i_date):
        return 'reject', None

    # Guardrail 3: Subdivision code existence in reference
    if subdiv_codes and code not in subdiv_codes:
        return 'reject', None

    # Guardrail 4: MRZ Line 1 character confidence check
    l1_confs = mrz['lines'][0].get('char_conf', [])
    if not l1_confs or min(l1_confs) < MIN_CONF_MRZ_L1:
        return 'reject', None

    # Guardrail 5: Names cross-validation against MRZ Line 1
    v_last = ' '.join(l.get('text', '') for l in (viz.get('Last_name_ru') or {}).get('lines', []))
    v_first = ' '.join(l.get('text', '') for l in (viz.get('First_name_ru') or {}).get('lines', []))
    v_mid = ' '.join(l.get('text', '') for l in (viz.get('Middle_name_ru') or {}).get('lines', []))

    if not v_last or not v_first:
        return 'reject', None

    # Visual names character confidence check
    name_confs = []
    for f in ['Last_name_ru', 'First_name_ru', 'Middle_name_ru']:
        for l in (viz.get(f) or {}).get('lines', []):
            name_confs.extend(l.get('char_conf', []))
    if not name_confs or min(name_confs) < MIN_CONF_NAME:
        return 'reject', None

    t_last = transliterate_ru(v_last)
    t_first = transliterate_ru(v_first)
    t_mid = transliterate_ru(v_mid)

    if not l1.startswith('PNRUS'):
        return 'reject', None

    mrz_parts = l1[5:].rstrip('<').split('<<')
    if len(mrz_parts) != 2:
        return 'reject', None

    mrz_last = mrz_parts[0]
    mrz_rest = mrz_parts[1].split('<')
    mrz_first = mrz_rest[0]
    mrz_mid = mrz_rest[1] if len(mrz_rest) > 1 else ''

    if t_last != mrz_last or t_first != mrz_first:
        return 'reject', None
    if mrz_mid and t_mid != mrz_mid:
        return 'reject', None

    # Guardrail 6: Geometry checks for multi-line fields
    bp_val = viz.get('Birth_place_ru') or {}
    org_val = viz.get('Issue_organization_ru') or {}

    if not check_lines_geometry(bp_val) or not check_lines_geometry(org_val):
        return 'reject', None

    bp_lines = bp_val.get('lines', [])
    org_lines = org_val.get('lines', [])

    # Guardrail 7: High-confidence OCR character threshold
    bp_confs = [c for l in bp_lines for c in l.get('char_conf', [])]
    org_confs = [c for l in org_lines for c in l.get('char_conf', [])]

    if not bp_confs or min(bp_confs) < MIN_CONF_BP:
        return 'reject', None
    if not org_confs or min(org_confs) < MIN_CONF_ORG:
        return 'reject', None

    bp_text = ' '.join(l.get('text', '') for l in bp_lines)
    org_text = ' '.join(l.get('text', '') for l in org_lines)

    # Guardrail 8: Issue organization vocabulary validation against reference
    if subdiv_vocab:
        org_tokens = re.findall(r'[А-ЯЁа-яё]+', org_text.upper())
        if any(t not in subdiv_vocab for t in org_tokens):
            return 'reject', None

    # Build verified 10 fields
    fields = {
        'Last_name_ru': v_last,
        'First_name_ru': v_first,
        'Middle_name_ru': v_mid,
        'Sex_ru': sex,
        'Birth_date': b_date,
        'Birth_place_ru': bp_text,
        'Issue_date': i_date,
        'Issue_organisation_code': code,
        'Licence_number': lic_num,
        'Issue_organization_ru': org_text
    }

    return 'accept', fields

def main(in_path, out_path):
    # Locate reference/subdivisions.csv relative to script or current dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ref_path = os.path.join(base_dir, 'reference', 'subdivisions.csv')
    if not os.path.exists(ref_path):
        ref_path = os.path.join('reference', 'subdivisions.csv')

    subdiv_codes, subdiv_vocab = load_reference_subdivisions(ref_path)

    accepted_cnt = 0
    rejected_cnt = 0

    with open(in_path, encoding='utf-8') as f_in, open(out_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc['doc_id']
            decision, fields = process_document(doc, subdiv_codes, subdiv_vocab)

            if decision == 'accept':
                record = {'doc_id': doc_id, 'decision': 'accept', 'fields': fields}
                accepted_cnt += 1
            else:
                record = {'doc_id': doc_id, 'decision': 'reject'}
                rejected_cnt += 1

            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

    total = accepted_cnt + rejected_cnt
    print(f"Готово! Обработано документов: {total}")
    print(f"  Принято (accept): {accepted_cnt} ({accepted_cnt/total:.1%})")
    print(f"  Отклонено (reject): {rejected_cnt} ({rejected_cnt/total:.1%})")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Использование: python postprocess.py <predictions.jsonl> <answers.jsonl>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
