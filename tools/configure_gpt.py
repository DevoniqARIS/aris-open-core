from pathlib import Path
import getpass

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / '.env.local'
print('\nARIS GPT configuration')
print('This stores the key locally in .env.local on this PC. Do not share that file.\n')
key = getpass.getpass('OpenAI API key: ').strip()
model = input('OpenAI mission model name: ').strip()
if not key or not model:
    raise SystemExit('Key and model are both required.')
path.write_text(f'OPENAI_API_KEY={key}\nOPENAI_MISSION_MODEL={model}\n', encoding='utf-8')
print(f'\nSaved {path.name}. Restart ARIS, then click VERIFY LIVE GPT in the browser.')
