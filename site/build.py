import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

tpl = open(os.path.join(HERE, 'template.html')).read()
samples = json.load(open(os.path.join(HERE, 'demo_samples.json')))
frag = tpl.replace('__SAMPLES_JSON__', json.dumps(samples))

# 1. artifact fragment (host injects <head>)
open(os.path.join(HERE, 'index.html'), 'w').write(frag)

# 2. full document for GitHub Pages (needs its own viewport meta, or mobile
#    renders at desktop width and the toggle becomes an unreachable target)
m = re.search(r'</style>', frag)
head_part = frag[:m.end()]           # <title> + <style>...</style>
body_part = frag[m.end():]           # the markup + script
title_m = re.search(r'<title>(.*?)</title>', head_part)
title = title_m.group(1)
head_wo_title = head_part.replace(title_m.group(0), '')
doc = (
    '<!doctype html>\n<html lang="en">\n<head>\n'
    '<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    f'<title>{title}</title>\n'
    f'{head_wo_title.strip()}\n'
    '</head>\n<body>\n'
    f'{body_part.strip()}\n'
    '</body>\n</html>\n'
)
open(os.path.join(REPO, 'docs', 'index.html'), 'w').write(doc)
print('fragment', len(frag), 'bytes | pages doc', len(doc), 'bytes')
