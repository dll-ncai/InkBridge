#!/usr/bin/env python3
"""build_supplement_v12_1.py: builds supplementary_v12.1.tex, supplement_map_v12.1.csv and the edit log.
Usage: python build_supplement_v12_1.py <v11.tex> <v12.tex> <v12.1.tex> <deferred_v12.json> <deferred_v12.1.json> [outdir]
Text and tables are copied from the three paper sources by label or exact string; every wording edit is an exact
replacement with an asserted match count. No number is typed in this script except inside asserted source strings."""
import re, sys, csv, json, hashlib
a11, a12, a121, jdef, jred = sys.argv[1:6]; out = sys.argv[6] if len(sys.argv) > 6 else '.'
t = open(a11, encoding='utf8').read(); L = t.split('\n')
t12 = open(a12, encoding='utf8').read(); t2 = open(a121, encoding='utf8').read()
D = json.load(open(jdef)); R = {x['id']: x for x in json.load(open(jred))['items']}
TITLE = 'InkBridge: Low-Resource Adaptation and Endpoint Consensus for Handwriting Trajectory Recovery'
REPO = 'https://github.com/dll-ncai/InkBridge'
assert TITLE in t2
SECT = set(re.findall(r'\\(?:sub)?section\{([^}]*)\}', t2))
def P(title):
    assert title in SECT, title; return "``" + title + "''"
PAPER_RA = 'the routed-adapter table of the paper (Section ' + P('Exploratory Routed Adapter Results') + ')'
LOG = []; MAP = []
def sub(s, old, new, n=1, where=''):
    c = s.count(old); assert c == n, (where, old, c)
    LOG.append({'where': where, 'old': old, 'new': new, 'count': n}); return s.replace(old, new)
def env(src, label):
    i = src.index('\\label{' + label + '}'); s = src.rfind('\\begin{table', 0, i)
    name = re.match(r'\\begin\{([^}]*)\}', src[s:]).group(1); end = '\\end{' + name + '}'
    return src[s:src.index(end, i) + len(end)]
def sec(label):
    i = t.index('\\section', t.rfind('\n', 0, t.index('\\label{' + label + '}')))
    j = min(x for x in (t.find('\\section{', i + 5), t.find('\\end{appendices}', i)) if x > 0)
    return t[i:j]
def cells(x):
    tb = re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}', x, flags=re.S).group(0)
    return [re.sub(r'\s+', ' ', c).strip() for c in re.split(r'&|\\\\', tb)]
ext = lambda s: '\\ext{' + s + '}'
TABS = ['notation', 'components', 'singleapp', 'hyper', 'corpora', 'failure', 'warm', 'german', 'singleR', 'r3b', 'r1d', 'm1', 'claims']
NEW = {'tab:' + k: 'tabS:' + k for k in TABS}; NEW.update({'app:impl': 'sup:impl', 'app:fm': 'sup:fm', 'app:rules': 'sup:rules'})
def relabel(s):
    for k, v in NEW.items():
        s = s.replace('\\label{' + k + '}', '\\label{' + v + '}').replace('\\ref{' + k + '}', '\\ref{' + v + '}')
    return s
def tab(src, label, edits=()):
    s = env(src, label)
    for old, new in edits: s = sub(s, old, new, 1, label)
    s = re.sub(r'\\begin\{table\*?\}\[t\]', lambda m: '\\begin{table}[!htbp]\n\\centering', s).replace('\\end{table*}', '\\end{table}')
    return relabel(s)
# ---- tables dropped because the v12.1 paper prints them in full (every v11 cell is in the v12.1 table)
for k in ('tab:xni', 'tab:r3a'):
    assert not (set(cells(env(t, k))) - set(cells(env(t2, k)))), k
    MAP.append([k, 'table', 'v11', '', '', 'not in supplement', 'printed in full in the v12.1 paper (all v11 cells present)', ''])
assert set(cells(env(t, 'tab:warm'))) - set(cells(env(t2, 'tab:warm')))
# ---- tables
T = {}
fmrow = next(l for l in env(t, 'tab:notation').split('\n') if l.startswith('$\\WFTfm{\\rho}$'))
perow = next(l for l in env(t2, 'tab:notation').split('\n') if l.startswith('$\\WFTpe{\\rho}$'))
T['notation'] = tab(t2, 'tab:notation', [
    ('Notation for models and variants.', 'Notation for models and variants, as in the paper, with the row of the FM-guided warm fine-tune added.'),
    ('fraction of the target training units.}', 'fraction of the target training units; FM: foundation model.}'),
    (perow, perow + '\n' + fmrow),
    ('Section~\\ref{sec:problem}', '$\\Fi$ processes the image as given; $\\Fm$ reflects the image horizontally, runs the model and reflects the output back')])
c = env(t, 'tab:components')
rows = {k: re.search(r'C%d ([^&]*?)\s*& RQ\d & ([^&]*?) & ([^\\]*?)\\\\' % k, c).groups() for k in (1, 2, 3)}
assert rows[3][0] == 'endpoint selection' and rows[2][0] == 'warm adaptation'
body = ('\\begin{tabular}{@{}lll@{}}\n\\toprule\nComponent & Labels & Samples\\\\\n\\midrule\n'
        'Warm adaptation & %s & %s\\\\\nEndpoint Consensus Selection & %s & %s\\\\\n'
        'Frame choice (%s; negative result) & %s & %s\\\\\n\\botrule\n\\end{tabular}'
        % (rows[2][1], rows[2][2], rows[3][1], rows[3][2], re.search(r'\((.*)\)', rows[1][0]).group(1), rows[1][1], rows[1][2]))
LOG.append({'where': 'tab:components', 'old': 'columns Component, RQ, Labels, Samples; rows C1 frame choice, C2 warm adaptation, C3 endpoint selection',
            'new': 'RQ column and C1-C3 labels removed; rows ordered warm adaptation, Endpoint Consensus Selection, frame choice (negative result); cell values unchanged', 'count': 1})
T['components'] = ('\\begin{table}[!htbp]\n\\centering\n\\caption{The two InkBridge components and the frame-choice experiment: target labels used and samples drawn per input.}'
                   '\\label{tabS:components}\n\\tabsz\\setlength{\\tabcolsep}{3pt}\n' + body + '\n\\end{table}')
T['singleapp'] = tab(t, 'tab:singleapp', [('for Table~\\ref{tab:singleT}.', 'for the routed-adapter variants (Section ' + P('Exploratory Routed Adapter Results') + ' of the paper).')])
T['hyper'] = tab(t, 'tab:hyper'); T['corpora'] = tab(t, 'tab:corpora')
# the v11 width-fitting box also holds the table note, which shrinks the table in one column; the box is removed, content unchanged
T['warm'] = tab(t, 'tab:warm', [('\\fitwidth{\\begin{tabular}', '\\begin{tabular}'), ('; --: not computed.}}\n\\end{table*}', '; --: not computed.}\n\\end{table*}')]); T['german'] = tab(t, 'tab:german'); T['m1'] = tab(t, 'tab:m1')
T['failure'] = tab(t, 'tab:failure', [('Cross-script failure on a different test set per row', 'Direction errors under the tested shifts, on a different test set per row')])
T['singleR'] = tab(t, 'tab:singleR', [('Single-model variants at', 'Routed-adapter variants at'), ('(Table~\\ref{tab:singleT})', '(' + PAPER_RA + ')')])
T['r3b'] = tab(t, 'tab:r3b', [('The order-gain criterion on the six other sets', 'The selection-gain criterion on the six non-Urdu sets'), ('number of sets with an order gain (Gain)', 'number of sets with a gain (Gain)')])
T['r1d'] = tab(t, 'tab:r1d', [('Label-free frame choice minus', 'Automatic frame choice minus'), ('with the same C3 selector', 'with the same Endpoint Consensus Selection'), ('the six other sets', 'the six non-Urdu sets')])
cl = R['T01']['text']; assert cl in t12 and 'tab:claims' not in t2
for old, new in [
    ('four systems (Section~\\ref{sec:intro}); mirrored left-to-right control (Section~\\ref{sec:discussion})', 'four systems (paper, ' + P('Introduction') + '); mirrored left-to-right control (paper, ' + P('Discussion') + ')'),
    ('AIoU $+0.047$ and $+0.059$, Table~\\ref{tab:warm}, three', 'AIoU $+0.047$ and $+0.059$, Table~\\ref{tab:warm}, three'),
    ('(Section~\\ref{sec:warmres}), three fine-tuning seeds', '(Table~\\ref{tab:warm}), three fine-tuning seeds'),
    ('over the source-training control (Table~\\ref{tab:warm})', 'over the source-training control (paper, ' + P('Warm Adaptation on the Target') + ')'),
    ('Table~\\ref{tab:xni}: lowest AIoU lower bound', 'paper, ' + P('Retention on Six Non-Urdu Sets') + ': lowest AIoU lower bound'),
    ('Table~\\ref{tab:r3a}, 5 of 5 bases', 'paper, ' + P('Endpoint Consensus Selection') + ', 5 of 5 bases'),
    ('(Section~\\ref{sec:selres}); two random-pick controls', '(Section~\\ref{sup:selres}); two random-pick controls'),
    ('three bases (Section~\\ref{sec:negative})', 'three bases (Table~\\ref{tab:r1d})'),
    ('lower for 5 (Section~\\ref{sec:negative})', 'lower for 5 (paper, ' + P('Negative Results') + ')'),
    ('second comparator copy (Table~\\ref{tab:singleT})', 'second comparator copy (paper, ' + P('Exploratory Routed Adapter Results') + ')')]:
    cl = sub(cl, old, new, 1, 'tab:claims')
T['claims'] = relabel(re.sub(r'\\begin\{table\*\}\[t\]', lambda m: '\\begin{table}[!htbp]\n\\centering', cl).replace('\\end{table*}', '\\end{table}'))
# ---- appendix sections (v11) with the paper's wording
A = {k: sec(k) for k in ('app:impl', 'app:fm', 'app:rules')}
for k in ('tab:singleapp', 'tab:hyper', 'tab:corpora'):
    assert env(t, k) in A['app:impl']; A['app:impl'] = A['app:impl'].replace(env(t, k), '%%TAB ' + k.split(':')[1] + '\n')
ctx32 = 'each backbone and each warm stage was trained once'; assert ctx32 + R['E32']['text'] in t12
E20 = R['E20']['text'].strip(); assert E20.startswith('A router whose Urdu class')
E29 = sub(R['E29']['text'], 'writers of Fig.~\\ref{fig:qual} excluded', 'writers of the qualitative-results figure of the paper excluded', 1, 'E29')
assert E29.startswith('; the unit is selected (')
E12 = R['E12']['text'].strip(); assert 'each ground-truth stroke is resampled at 1-px arc length on the 64-px canvas' in t12 and 'The length 320 is fixed by the backbone' in t12
E_IMPL = [
 ('\\section{Implementation Details}\\label{app:impl}\n', '\\section{Implementation Details}\\label{app:impl}\n' + ext('Implementation Details, Datasets and Target Data Budget, Baselines (Experimental Setup); Warm Adaptation, Script Routing (Method)') + '\n'),
 ('(Section~\\ref{sec:data}); the other corpora', '(Section ' + P('Datasets and Target Data Budget') + ' of the paper); the other corpora'),
 ('250 validation units per script. The third-party', '250 validation units per dataset. The third-party'),
 ('\n\\textbf{Backbones.}', '\n\\textbf{Sequence length.} Each ground-truth stroke is resampled at 1-px arc length on the 64-px canvas, and the sequence length of 320 is fixed by the backbone. ' + E12 + '\n\n\\textbf{Backbones.}'),
 ('(100,000 units, of which 20,000 are OUHD-L units).', '(100,000 units, of which 20,000 are OUHD-L units). Each backbone and each warm stage was trained once' + R['E32']['text'] + '.'),
 ('segmentation F1 of Section~\\ref{sec:data}', 'segmentation F1 reported in Section ' + P('Datasets and Target Data Budget') + ' of the paper'),
 ('Tables~\\ref{tab:warm}, \\ref{tab:xni}, \\ref{tab:singleT} and~\\ref{tab:singleR} pool the three seeds',
  'Tables~\\ref{tab:warm} and~\\ref{tab:singleR} and the warm-adaptation, retention and routed-adapter tables of the paper pool the three seeds'),
 ('(Appendix~\\ref{app:rules})', '(Section~\\ref{app:rules})'),
 ('FM-guided backbones of Table~\\ref{tab:singleT}', 'FM-guided backbones of ' + PAPER_RA),
 ('The step-matched control is selected by the same rule', 'The source-training control is selected by the same rule'),
 ('are in the reviewer archive (folder \\texttt{supplementary})', 'are in the code repository (\\url{' + REPO + '}, folder \\texttt{supplementary})'),
 ('\\textbf{Composition.}', '\\textbf{Script routing.}'),
 ('(20,000 per script)', '(20,000 per dataset)'),
 ('250 validation units per script is kept', '250 validation units per dataset is kept'),
 ('German belongs to no class.', 'German belongs to no class. ' + E20),
 ('\\textbf{Endpoint selection.}', '\\textbf{Endpoint Consensus Selection.}'),
 ('The comparator $\\FT{1\\%}$ of Table~\\ref{tab:singleT} exists', 'The comparator $\\FT{1\\%}$ of ' + PAPER_RA + ' exists'),
 ('first-seed evaluation of the single-model variants', 'first-seed evaluation of the routed-adapter variants'),
 ('\n\\textbf{Selectivity perturbations.}', '\n\\textbf{Example unit of the metric figure.} In the metric-illustration figure of the paper (Section ' + P('Evaluation Metrics') + ')' + E29[1:] + '.\n\n\\textbf{Selectivity perturbations.}'),
]
m34 = re.search(r'Non-inferiority margins are 1\.5 times.*?0\.01245 \(direction\)\.', t12); assert m34 and R['E34']['text'] in m34.group(0)
E_RULES = [
 ('\\section{Full Decision Rules}\\label{app:rules}\n', '\\section{Full Decision Rules}\\label{app:rules}\n' + ext('Evaluation Protocol (Experimental Setup)') + '\n\n\\textbf{Margins.} ' + m34.group(0) + '\n'),
 ('\\item \\emph{Solution}: target gain on the target and retention on all six other sets.',
  '\\item \\emph{Target gain plus shape retention}: target gain on the target and retention on all six non-Urdu sets.'),
 ('\\item \\emph{Encoder attribution}: warm variant minus $\\SFT{\\rho}$ on the target', '\\item \\emph{Control comparison}: warm variant minus the source-training control $\\SFT{\\rho}$ on the target'),
 ('\\item \\emph{Order gain}:', '\\item \\emph{Selection gain}:'),
 ('with the same selector (lower bound above 0)', 'with the same Endpoint Consensus Selection (lower bound above 0)'),
 ('and the six other sets keep AIoU', 'and the six non-Urdu sets keep AIoU'),
 ('The test evaluation of endpoint selection on plain fine-tuning', 'The test evaluation of Endpoint Consensus Selection on plain fine-tuning'),
 ('\\textbf{Stricter reading of the order gain.}', '\\textbf{Stricter reading of the selection gain.}'),
 ('7 of 24 cells on the other sets', '7 of 24 cells on the non-Urdu sets'),
 ('The pooled three-seed read as the confirmatory read, the retention rule and the single-model arms were recorded',
  'The pooled three-seed analysis as the primary pooled analysis, the retention rule and the routed-adapter arms were recorded'),
 ('Two deviations follow.', 'Two protocol deviations follow.'),
 ('the margins were not fixed before any test result existed.', "the margins were not fixed before any test result existed. In the records the control comparison is named ``encoder attribution'', the selection gain ``order gain'', the target gain plus shape retention ``solution'', and the primary pooled analysis the ``confirmatory read''."),
]
E_FM = [('\\section{Foundation-Model Screen}\\label{app:fm}\n', '\\section{Foundation-Model Screen}\\label{app:fm}\n' + ext('Negative Results (Results); Warm Adaptation (Method)') + '\n')]
for old, new in E_IMPL: A['app:impl'] = sub(A['app:impl'], old, new, 1, 'app:impl')
for old, new in E_RULES: A['app:rules'] = sub(A['app:rules'], old, new, 1, 'app:rules')
for old, new in E_FM: A['app:fm'] = sub(A['app:fm'], old, new, 1, 'app:fm')
assert '0.00003 LDTW (0.27875 against 0.27878)' in A['app:impl'] and '0.00003 LDTW (0.27875 against 0.27878)' in R['E15']['text']
assert 'with a checkpoint every 1,000 steps' in A['app:impl'] and 'checkpoint every 1,000 steps' in R['E33']['text']
for k in A:
    A[k] = relabel(A[k])
    for tk in ('singleapp', 'hyper', 'corpora'): A[k] = A[k].replace('%%TAB ' + tk + '\n', T[tk] + '\n')
# ---- passages from the v11 and v12 main text
proto = next(l for l in L if l.startswith('Every comparison is paired on the common units'))
stat = re.match(r'(Every comparison is paired.*?because it has two writers) \(Table~\\ref\{tab:sets\}\)\.', proto).group(1) + '.'
dag = next(l for l in L if 'are exploratory and marked $^{\\dagger}$' in l)
stat += ' ' + re.search(r'(Rows with fewer than three seeds are exploratory and marked \$\^\{\\dagger\}\$), and the comparison', dag).group(1) + '.'
lf = next(l for l in L if l.startswith('\\textbf{Answer.} Stroke direction follows'))
F1 = re.search(r'Table~\\ref\{tab:failure\} collects the evidence; its rows are scored on different sets and are not one comparison\.', lf).group(0)
F2 = sub(re.search(r'The left-to-right and mirrored-control values are not rows of Table~\\ref\{tab:failure\}; their per-set tables are in the reviewer archive \(folder \\texttt\{6\\_table\\_sources\}\)\.', lf).group(0),
         'in the reviewer archive (folder', 'in the code repository (folder', 1, 'sec:failure-archive')
F1 = sub(F1, 'collects the evidence', 'collects the direction evidence named in the paper', 1, 'sec:failure')
m39 = re.search(r'\\textbf\{Encoder initialisation\.\} Paired against.*?\(\$-0\.015\$ \[\$-0\.026\$, \$-0\.002\$\] and \$-0\.010\$ \[\$-0\.018\$, \$-0\.003\$\]\)\.', t12)
assert m39 and R['E39']['text'] in m39.group(0); E39 = m39.group(0)
E41 = R['E41']['text']; E43 = R['E43']['text'].strip()
E42 = sub(R['E42']['text'].strip(), 'the 336-label router of Section~\\ref{sec:compose}', 'the 336-label router of Section~\\ref{sup:impl} (Script routing)', 1, 'E42')
l498 = next(l for l in L if l.startswith('On the six other sets, 16 of 24 base-by-set cells'))
for old, new in [
    ('On the six other sets, 16 of 24 base-by-set cells show an order gain and 9 pass', 'On the six non-Urdu sets, 16 of 24 base-by-set cells show a gain and 9 pass'),
    (' A post hoc calibration check shows why. The flag requires', ' In a post hoc calibration check, the flag requires'),
    (' The random pick is unbiased and the flag is miscalibrated.', ''),
    ('the five Urdu bases of Table~\\ref{tab:r3a}', 'the five Urdu bases of the paper'),
    ('the stricter reading is in Appendix~\\ref{app:rules}.', 'the stricter reading is in Section~\\ref{sup:rules}. The paper reports the non-Urdu result as inconclusive because the two controls disagree.')]:
    l498 = sub(l498, old, new, 1, 'sec:selres-control')
fr = '\n'.join(L[D['passages_deferred'][3]['v11_lines'][0]:D['passages_deferred'][3]['v11_lines'][1]]); assert D['passages_deferred'][3]['id'] == 'sec:frame'
fr = fr[fr.index('\\textbf{Mirrored frame.}'):]
for old, new in [
    ('\\textbf{Label-free frame choice.}', '\\textbf{Frame choice.}'),
    ('the endpoint-agreement function $U$ (Eq.~\\ref{eq:utility}, Section~\\ref{sec:verifier})', 'the endpoint-agreement function $U$ (defined in Section ' + P('Endpoint Consensus Selection') + ' of the paper)'),
    ('The output is the sample that C3 (Section~\\ref{sec:verifier}) selects among the $k$ samples of the chosen frame (Fig.~\\ref{fig:frameverifier}).', 'The output is the sample that Endpoint Consensus Selection selects among the $k$ samples of the chosen frame.')]:
    fr = sub(fr, old, new, 1, 'sec:frame')
l504 = next(l for l in L if l.startswith('\\textbf{Mirrored frame.} On $\\fnot{ur}$'))
assert R['E50']['text'].lstrip('; ') in l504
l504 = sub(sub(l504, '\\textbf{Mirrored frame.} On', '\\textbf{Mirrored frame applied by hand.} On', 1, 'sec:frameres-extra'), '(Appendix~\\ref{app:rules})', '(Section~\\ref{sup:rules})', 1, 'sec:frameres-extra')
e52 = R['E52']['text']; assert e52.count('; and (iv) 1\\% low-rank adapters, which gain shape') == 1
E52 = e52.split('; and (iv)')[0] + '; and (iv) 1\\% low-rank adapters, reported in Section ' + P('Negative Results') + ' of the paper.'
LOG.append({'where': 'E52', 'old': '; and (iv) 1\\% low-rank adapters, which gain shape', 'new': '; and (iv) 1\\% low-rank adapters, reported in the paper.', 'count': 1})
l570 = next(l for l in L if l.startswith('For each metric we perturb the ink along its own factor'))
l570 = l570[:l570.index(' AIoU remains the primary metric')]
for old, new in [('definitions in Appendix~\\ref{app:impl}', 'definitions in Section~\\ref{sup:impl}'),
                 ('(Section~\\ref{sec:singleres})', '(Section ' + P('Exploratory Routed Adapter Results') + ' of the paper)')]:
    l570 = sub(l570, old, new, 1, 'sec:selectivity-detail')
authors = ', '.join(a + ' ' + b for a, b in re.findall(r'\\fnm\{([^}]*)\} \\sur\{([^}]*)\}', t2)[:6]).replace(', Momina', ',\\\\ Momina')
pre = '\n'.join(l for l in L[:40] if re.match(r'\\newcommand\{\\(f|FT|WFT|SFT|RP|AD|Rt|Fi|Fm|tabsz|fitwidth)', l) or l.startswith('\\newsavebox'))
assert pre.count('\\newcommand') == 16 and '\\newsavebox' in pre
doc = r"""% Supplement source, generated by the supplement build script from the v11, v12 and v12.1 paper sources. Do not edit by hand.
\documentclass[10pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{array}
\usepackage{url}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{placeins}
\newcommand{\botrule}{\bottomrule}
\newcommand{\ext}[1]{\par\noindent{\small\itshape Extends in the paper: #1.}\par\smallskip}
@@PRE@@
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\theequation}{S\arabic{equation}}
\setlength{\parskip}{3pt}
\raggedbottom

\begin{document}

\title{Supplementary Material for:\\ @@TITLE@@}
\author{@@AUTHORS@@}
\date{}
\maketitle

\noindent This document accompanies the paper ``@@TITLE@@''. It holds implementation details, the foundation-model screen, the full decision rules with their records, and results that the paper summarises. The paper is complete without this document. Each section and subsection names the part of the paper that it extends, by its title. Code, the evaluator, unit-ID lists, score tables and protocol records are in the repository \url{@@REPO@@}.

\medskip
\noindent\textbf{Contents.} Section~\ref{sup:notation}: notation, data budget and statistics. Section~\ref{sup:impl}: implementation details (Tables~\ref{tabS:singleapp}--\ref{tabS:corpora}). Section~\ref{sup:fm}: foundation-model screen. Section~\ref{sup:rules}: full decision rules, records and protocol deviations. Section~\ref{sup:results}: additional results (Tables~\ref{tabS:failure}--\ref{tabS:claims}).

\section{Notation, Data Budget and Statistics}\label{sup:notation}
@@EXT1@@

Table~\ref{tabS:notation} repeats the notation of the paper so that this document can be read on its own, and Table~\ref{tabS:components} lists the components with the target labels and samples they use. Urdu (OUHD-L) is the evaluated target; the four source datasets are CASIA-zh, CASIA-en, HPL Tamil and IAM, and the six non-Urdu test sets are CASIA-zh, CASIA-en, HPL Tamil, IAM500, UNIPEN-de and OnHW (Table~\ref{tabS:corpora}).

\textbf{Target data budget.} ``1\% of Urdu'' means 1\% of the target training units: 336 of the 33,615 OUHD-L training units. Checkpoint selection on Urdu uses 1,000 labelled validation units. The script router is trained on the script labels of the 100,000 units of the joint training set, of which 20,000 are OUHD-L units (Section~\ref{sup:impl}).

\textbf{Metrics.} AIoU measures shape, strict order and Kendall $\tau$ stroke order, direction the drawing direction of matched strokes, and pen-up F1 the stroke boundaries; LDTW is used only for checkpoint selection. The definitions are in Section @@PMET@@ of the paper.

\textbf{Statistics.} @@STAT@@ The selection tables resample writers on every set, so their OnHW cells rest on two clusters.

@@TAB notation@@

@@TAB components@@
\FloatBarrier

@@SEC app:impl@@
\FloatBarrier

@@SEC app:fm@@

@@SEC app:rules@@
\FloatBarrier

\section{Additional Results}\label{sup:results}

The decision rules behind the verdict columns are in Section~\ref{sup:rules}.

\subsection{Direction errors under the tested shifts}\label{sup:failure}
@@EXT_FAIL@@
@@F1@@ @@F2@@

@@TAB failure@@
\FloatBarrier

\subsection{Warm adaptation on the target}\label{sup:warmres}
@@EXT_WARM@@
Table~\ref{tabS:warm} gives the warm variants against plain fine-tuning at 1\% and at 10\% of Urdu, with the per-seed ranges and the seed-level intervals.

@@E39@@

@@TAB warm@@
\FloatBarrier

\subsection{Retention on six non-Urdu sets}\label{sup:retres}
@@EXT_RET@@
Table~\ref{tabS:german} gives direction on UNIPEN-de per system, with and without the router.

\textbf{Routing counts.} @@E41@@ @@E42@@

\textbf{Without a router at 10\%.} @@E43@@

@@TAB german@@
\FloatBarrier

\subsection{Exploratory routed adapter results}\label{sup:rares}
@@EXT_RA@@
Table~\ref{tabS:singleR} gives the retention side of the routed-adapter variants; the target side is in the paper, and the device check and first-seed evaluation are in Table~\ref{tabS:singleapp}. The comparison is exploratory.

@@TAB singleR@@
\FloatBarrier

\subsection{Endpoint Consensus Selection on the non-Urdu sets}\label{sup:selres}
@@EXT_SEL@@
@@L498@@

@@TAB r3b@@
\FloatBarrier

\subsection{Negative results}\label{sup:frameres}
@@EXT_NEG@@
@@FRAME@@

Table~\ref{tabS:r1d} gives the automatic frame choice against the identity frame, per base.

@@L504@@

@@E52@@

@@TAB r1d@@
\FloatBarrier

\subsection{Metric selectivity}\label{sup:m1res}
@@EXT_DISC@@
@@L570@@

@@TAB m1@@
\FloatBarrier

\subsection{Claims, evidence and verdicts}\label{sup:claims}
@@EXT_DISC@@
Table~\ref{tabS:claims} lists each claim of the paper with its evidence and verdict.

@@TAB claims@@
\FloatBarrier

\bibliographystyle{unsrtnat}
\bibliography{references}

\end{document}
"""
rep = {'@@PRE@@': pre, '@@TITLE@@': TITLE, '@@AUTHORS@@': authors, '@@REPO@@': REPO, '@@STAT@@': stat, '@@PMET@@': P('Evaluation Metrics'),
       '@@EXT1@@': ext('Problem and Notation (Method); Datasets and Target Data Budget, Evaluation Metrics, Evaluation Protocol (Experimental Setup)'),
       '@@EXT_FAIL@@': ext('Introduction; Discussion'), '@@EXT_WARM@@': ext('Warm Adaptation on the Target (Results)'),
       '@@EXT_RET@@': ext('Retention on Six Non-Urdu Sets (Results)'), '@@EXT_RA@@': ext('Exploratory Routed Adapter Results (Results)'),
       '@@EXT_SEL@@': ext('Endpoint Consensus Selection (Results)'), '@@EXT_NEG@@': ext('Negative Results (Results)'), '@@EXT_DISC@@': ext('Discussion'),
       '@@F1@@': relabel(F1), '@@F2@@': relabel(F2), '@@E39@@': E39, '@@E41@@': E41, '@@E42@@': E42, '@@E43@@': E43, '@@L498@@': relabel(l498),
       '@@FRAME@@': fr.strip(), '@@L504@@': l504, '@@E52@@': E52, '@@L570@@': relabel(l570)}
for name in re.findall(r'\\ext\{([^}]*)\}', ''.join(rep.values()) + ''.join(A.values())):
    for part in re.split(r'[;,]\s*', re.sub(r'\s*\([^)]*\)', '', name)): assert part.strip() in SECT, part
for k, v in rep.items(): doc = doc.replace(k, v)
for k in T: doc = doc.replace('@@TAB ' + k + '@@', T[k])
for k in A: doc = doc.replace('@@SEC ' + k + '@@', A[k].strip())
assert '@@' not in doc
bad = re.findall(r'\bC[123]\b|RQ\s?\d|\\ref\{(?:sec|tab|fig|app|eq):[^}]*\}|[Aa]ppendi|reviewer archive|[Ss]tep-matched|(?<![`-])\b[Oo]rder[- ]gain|[Ll]abel-free|single-model|confirmatory read,', doc); assert not bad, [doc[max(0,m.start()-70):m.end()+40] for m in re.finditer(r'[Oo]rder[- ]gain|[Ll]abel-free', doc)]
labels = set(re.findall(r'\\label\{([^}]*)\}', doc)); refs = set(re.findall(r'\\ref\{([^}]*)\}', doc)); assert refs <= labels, refs - labels
open(out + '/supplementary_v12.1.tex', 'w', encoding='utf8').write(doc)
# ---- map
tl = re.findall(r'\\label\{(tabS:[^}]*)\}', doc)
secnum = {}; s_i = 0; ss_i = 0
for m in re.finditer(r'\\(section|subsection)\{[^}]*\}\\label\{(sup:[^}]*)\}', doc):
    if m.group(1) == 'section': s_i += 1; ss_i = 0; secnum[m.group(2)] = 'S%d' % s_i
    else: ss_i += 1; secnum[m.group(2)] = 'S%d.%d' % (s_i, ss_i)
tn = lambda k: 'Table S%d' % (tl.index('tabS:' + k) + 1)
def row(item, kind, src, lab, num, transfer, note, extends): MAP.append([item, kind, src, lab, num, transfer, note, extends])
row('app:impl', 'appendix section', 'v11', 'sup:impl', secnum['sup:impl'], 'moved', '', 'Experimental Setup; Method')
row('app:fm', 'appendix section', 'v11', 'sup:fm', secnum['sup:fm'], 'moved', '', 'Negative Results; Warm Adaptation')
row('app:rules', 'appendix section', 'v11', 'sup:rules', secnum['sup:rules'], 'moved', "rule names follow the paper; record names listed in one sentence", 'Evaluation Protocol')
TI = {'notation': ('v12.1 + v11 row', 'copy with extra row', 'paper table plus the FM-guided row removed from the paper', 'Problem and Notation', 'sup:notation'),
      'components': ('v11', 'moved', 'RQ column and C1-C3 labels removed', 'Method', 'sup:notation'),
      'singleapp': ('v11', 'moved', 'first-seed columns are not in the paper', 'Exploratory Routed Adapter Results', 'sup:impl'),
      'hyper': ('v11', 'moved', '', 'Implementation Details', 'sup:impl'), 'corpora': ('v11', 'moved', 'training-pool and test-set columns are not in the paper', 'Datasets and Target Data Budget', 'sup:impl'),
      'failure': ('v11', 'moved', 'caption reworded', 'Introduction; Discussion', 'sup:failure'),
      'warm': ('v11', 'full v11 version', '10% columns, per-seed range and seed-level interval rows are not in the paper table', 'Warm Adaptation on the Target', 'sup:warmres'),
      'german': ('v11', 'moved', 'per-writer ranges are not in the paper', 'Retention on Six Non-Urdu Sets', 'sup:retres'),
      'singleR': ('v11', 'moved', 'order-kept counts are not in the paper', 'Exploratory Routed Adapter Results', 'sup:rares'),
      'r3b': ('v11', 'moved', '', 'Endpoint Consensus Selection', 'sup:selres'), 'r1d': ('v11', 'moved', '', 'Negative Results', 'sup:frameres'),
      'm1': ('v11', 'moved', '', 'Discussion', 'sup:m1res'), 'claims': ('v12 (item T01)', 'moved', 'references to paper tables given by section title', 'Discussion', 'sup:claims')}
for k in TABS: row('tab:' + k, 'table', TI[k][0], 'tabS:' + k, tn(k) + ' (' + secnum[TI[k][4]] + ')', TI[k][1], TI[k][2], TI[k][3])
PASS = [('sec:frame', 'v11', 'sup:frameres', 'moved', 'mirrored-frame and frame-choice definitions; selector named as in the paper', 'Negative Results'),
        ('rq-list', 'v11', '', 'not in supplement', 'research-question labels were removed by review item 3; no content beyond the labels', 'Introduction'),
        ('sec:failure-archive', 'v11', 'sup:failure', 'moved', 'pointer now names the code repository', 'Introduction; Discussion'),
        ('sec:frameres-extra', 'v11', 'sup:frameres', 'moved', 'v11 paragraph in full', 'Negative Results'),
        ('sec:selres-control', 'v11', 'sup:selres', 'moved', "two interpretive sentences left out ('shows why', 'the flag is miscalibrated'); paper verdict added", 'Endpoint Consensus Selection'),
        ('sec:impl-archive', 'v11', 'sup:impl', 'moved', 'pointer sentence in Warm stage and fine-tuning names the code repository', 'Implementation Details'),
        ('sec:selectivity-detail', 'v11', 'sup:m1res', 'moved', 'v11 paragraph up to the verdict sentence', 'Discussion'),
        ('prose-fit', 'v11', '', 'not in supplement', 'wording only; the deferral list states that every number, claim, limitation and citation stays in the paper', '')]
for i, src, lab, tr, note, e in PASS: row(i, 'passage (v12 text stage)', src, lab, secnum.get(lab, ''), tr, note, e)
EI = {'E12': ('sup:impl', 'Sequence length paragraph', 'Problem and Notation'), 'E15': ('sup:impl', 'already in the v11 appendix text (Warm stage and fine-tuning)', 'Warm Adaptation'),
      'E20': ('sup:impl', 'Script routing paragraph', 'Script Routing'), 'E29': ('sup:impl', 'Example unit of the metric figure paragraph', 'Evaluation Metrics'),
      'E32': ('sup:impl', 'Backbones paragraph', 'Implementation Details'), 'E33': ('sup:impl', 'already in the v11 appendix text (Warm stage and fine-tuning)', 'Implementation Details'),
      'E34': ('sup:rules', 'Margins paragraph (full v12 sentence)', 'Evaluation Protocol'), 'E39': ('sup:warmres', 'full v12 sentences on encoder initialisation', 'Warm Adaptation on the Target'),
      'E41': ('sup:retres', 'Routing counts', 'Retention on Six Non-Urdu Sets'), 'E42': ('sup:retres', 'Routing counts', 'Retention on Six Non-Urdu Sets'),
      'E43': ('sup:retres', 'Without a router at 10%', 'Retention on Six Non-Urdu Sets'), 'E50': ('sup:frameres', 'inside the v11 mirrored-frame paragraph', 'Negative Results'),
      'E52': ('sup:frameres', 'tests (i) to (iii); test (iv) stays in the paper', 'Negative Results')}
for k in sorted(EI, key=lambda x: int(x[1:])): row(k, 'passage (v12.1 page reduction)', 'v12', EI[k][0], secnum[EI[k][0]], 'moved', EI[k][1], EI[k][2])
with open(out + '/supplement_map_v12.1.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['item', 'kind', 'source', 'supplement_label', 'supplement_number', 'transfer', 'note', 'paper_section_extended']); w.writerows(MAP)
json.dump(LOG, open(out + '/supplement_v12.1_edit_log.json', 'w'), indent=1)
print(len(doc), len(MAP), len(LOG), len(tl))
